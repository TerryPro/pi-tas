/**
 * /tsa — Time Series Analysis command.
 *
 * Entry point is the `input` event, not `pi.registerCommand()`.
 *
 * Why: a registered command can only trigger an agent turn via `pi.sendUserMessage()`,
 * and that injection is dropped in non-interactive modes (`pi -p`, JSON, RPC) — the
 * process exits before the queued turn runs. Transforming the input in place works in
 * every mode, and the bundled `prompts/tsa.md` template still gives `/tsa` autocomplete.
 * (Extension commands take precedence over the `input` event, so the two must not share
 * a name — the template is a shell that the transform bypasses.)
 *
 * Deterministic work done here (no LLM involved):
 *   1. validate the input file
 *   2. create the task directory in the current working directory
 *   3. write `context.json` (run metadata + options) and `pyproject.toml`
 *   4. copy the reference pipeline scripts from the bundled skill
 *   5. create/sync the uv virtual environment
 *   6. rewrite the prompt so the agent runs the `time-series-analysis` skill
 *
 * All analysis logic lives in the skill (skills/time-series-analysis/), not here.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { stat, mkdir, writeFile, cp, readdir } from "node:fs/promises";
import { dirname, basename, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SKILL_NAME = "time-series-analysis";
const RUN_PREFIX = "tsa-run-";
const COMMAND_PATTERN = /^\/tsa(?=\s|$)([\s\S]*)$/;

const DATA_EXTENSIONS = new Set([
  ".csv", ".tsv", ".txt", ".dat", ".json", ".jsonl", ".ndjson",
  ".parquet", ".pq", ".feather", ".xlsx", ".xls", ".orc", ".dta",
]);

const PYPROJECT = `[project]
name = "tsa-run"
version = "0.0.0"
description = "Ephemeral environment for a /tsa analysis run"
requires-python = ">=3.10"
dependencies = [
  "pandas>=2.1",
  "numpy>=1.26",
  "scipy>=1.11",
  "statsmodels>=0.14",
  "scikit-learn>=1.4",
  "matplotlib>=3.8",
  "pyarrow>=15",
  "openpyxl>=3.1",
]
`;

const USAGE = [
  "用法: /tsa <数据文件> [选项]",
  "",
  "选项:",
  "  --time-col <列名>    时间列（缺省自动探测）",
  "  --value-col <列名>   目标列（缺省自动探测）",
  "  --freq <频率>        重采样频率，如 D/W/M/h",
  "  --agg <聚合方式>     重采样聚合，默认 mean",
  "  --horizon <步数>     预测步数（缺省 = 2 个周期）",
  "  --name <标识>        任务目录名标识",
  "  --no-setup           跳过 uv 虚拟环境创建",
  "",
  "说明: 只有第一个非 -- 参数会被当作数据文件；其余位置参数作为额外要求传给分析流程。",
].join("\n");

type Options = {
  input?: string;
  timeCol?: string;
  valueCol?: string;
  freq?: string;
  agg?: string;
  horizon?: number;
  name?: string;
  noSetup: boolean;
  extra: string[];
};

// ---------------------------------------------------------------------------
// 参数解析
// ---------------------------------------------------------------------------

function tokenize(raw: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let quote: string | null = null;
  for (const char of raw) {
    if (quote) {
      if (char === quote) quote = null;
      else current += char;
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current) tokens.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  if (current) tokens.push(current);
  return tokens;
}

function parseOptions(raw: string): Options {
  const options: Options = { noSetup: false, extra: [] };
  const tokens = tokenize(raw);
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (!token.startsWith("--")) {
      if (!options.input) options.input = token;
      else options.extra.push(token);
      continue;
    }
    const eq = token.indexOf("=");
    const key = (eq === -1 ? token : token.slice(0, eq)).toLowerCase();
    let value: string | undefined = eq === -1 ? undefined : token.slice(eq + 1);
    const takeValue = () => {
      if (value !== undefined) return value;
      const next = tokens[++i];
      return next === undefined ? "" : next;
    };

    if (key === "--no-setup" || key === "--no-env") {
      options.noSetup = true;
      continue;
    }
    switch (key) {
      case "--time":
      case "--time-col":
      case "--time-column":
      case "--index-col":
        options.timeCol = takeValue();
        break;
      case "--value":
      case "--value-col":
      case "--value-column":
      case "--target":
      case "--target-col":
        options.valueCol = takeValue();
        break;
      case "--freq":
      case "--frequency":
        options.freq = takeValue();
        break;
      case "--agg":
      case "--aggregate":
        options.agg = takeValue();
        break;
      case "--horizon":
      case "--h":
        options.horizon = Number(takeValue());
        break;
      case "--name":
      case "--slug":
        options.name = takeValue();
        break;
      default:
        options.extra.push(token);
    }
  }
  return options;
}

// ---------------------------------------------------------------------------
// 路径工具
// ---------------------------------------------------------------------------

function hereDir(): string {
  try {
    return dirname(fileURLToPath(import.meta.url));
  } catch {
    return process.cwd();
  }
}

async function findSkillDir(): Promise<string | null> {
  let dir = hereDir();
  for (let i = 0; i < 8; i++) {
    const candidate = join(dir, "skills", SKILL_NAME);
    try {
      if ((await stat(candidate)).isDirectory()) return candidate;
    } catch {
      // keep walking up
    }
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

function timestampSlug(date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}` +
    `-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
  );
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return slug || "data";
}

function isDataFile(name: string): boolean {
  return DATA_EXTENSIONS.has(extname(name).toLowerCase());
}

const SKIP_DIRS = new Set([
  "node_modules", ".git", ".venv", "venv", "dist", "build", "__pycache__", ".cache",
]);

/** 扫当前目录及一层子目录（跳过噪音目录），返回相对路径候选。 */
async function listDataFiles(cwd: string, limit = 25): Promise<string[]> {
  const found: string[] = [];
  const scan = async (dir: string, prefix: string) => {
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (found.length >= limit) return;
      if (entry.isFile() && isDataFile(entry.name)) found.push(prefix + entry.name);
    }
  };

  await scan(cwd, "");
  let subdirs = 0;
  try {
    const entries = await readdir(cwd, { withFileTypes: true });
    for (const entry of entries) {
      if (found.length >= limit || subdirs >= 8) break;
      if (!entry.isDirectory() || entry.name.startsWith(".") || SKIP_DIRS.has(entry.name)) continue;
      subdirs++;
      await scan(join(cwd, entry.name), `${entry.name}/`);
    }
  } catch {
    // 无权限等情况下只保留顶层结果
  }
  return found;
}

// ---------------------------------------------------------------------------
// 提示词
// ---------------------------------------------------------------------------

function buildPrompt(args: {
  runDir: string;
  inputPath: string;
  skillDir: string | null;
  options: Options;
  envReady: boolean;
  envNote: string;
}): string {
  const { runDir, inputPath, skillDir, options, envReady, envNote } = args;
  const skillLine = skillDir
    ? [
        "先读取技能文件，它是本次分析的唯一流程规范：",
        `- ${join(skillDir, "SKILL.md")}`,
        `- 逐步细则: ${join(skillDir, "references", "step-details.md")}`,
        `- 数据契约: ${join(skillDir, "references", "data-contract.md")}`,
      ].join("\n")
    : "本次未定位到技能目录，请用 read 工具自行找到 skills/time-series-analysis/SKILL.md 并读取。";

  const optionLines = [
    `time_col=${options.timeCol ?? "(自动探测)"}`,
    `value_col=${options.valueCol ?? "(自动探测)"}`,
    `freq=${options.freq ?? "(自动探测)"}`,
    `agg=${options.agg ?? "mean"}`,
    `horizon=${Number.isFinite(options.horizon as number) ? options.horizon : "(默认 2 个周期)"}`,
  ];
  if (options.extra.length) optionLines.push(`其他要求=${options.extra.join(" ")}`);

  return [
    "执行一次完整的时序数据分析（TSA 流水线）。",
    "",
    `- 输入数据: ${inputPath}`,
    `- 任务目录: ${runDir}`,
    `- 运行环境: ${envReady ? "uv 虚拟环境已就绪" : "uv 虚拟环境未就绪"}`,
    `- 环境说明: ${envNote}`,
    `- 选项: ${optionLines.join("; ")}`,
    "",
    skillLine,
    "",
    "要求：",
    "1. 所有 Python 代码通过 `uv run` 在任务目录内执行，不要污染全局环境。",
    "2. 按技能规定的 5 个步骤执行：数据概览与质量检查 → 平稳性与分解 → 建模与回测 → 异常与变点检测 → 汇总报告。",
    "3. 每一步都要留下可复查的产物：out/stepN_*.json、out/stepN_*.md、figures/*.png。",
    `4. 参考脚本已复制到 ${join(runDir, "scripts")}，可直接复用；发现缺陷就修正脚本，不要绕过它们。`,
    "5. 遇到错误就修脚本重跑，不要伪造结果；确实无法完成的步骤，在产物和报告里明确标注「未完成 + 原因」。",
    `6. 最后生成中文报告 ${join(runDir, "report.md")}，填写其中的 TSA:SUMMARY / TSA:CONCLUSIONS 占位符，并简要汇报关键发现。`,
    "",
    "现在开始执行，不要只给计划。",
  ].join("\n");
}

function buildFailurePrompt(message: string): string {
  return [
    "用户尝试用 /tsa 启动时序分析，但任务无法创建：",
    message,
    "",
    "不要执行任何分析或创建文件。直接用简洁的中文向用户说明这个问题，并给出可操作的修复建议。",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------

type ScaffoldResult = { kind: "prompt"; text: string } | { kind: "failed"; message: string };

async function scaffold(
  rawArgs: string,
  ctx: ExtensionContext,
  pi: ExtensionAPI,
): Promise<ScaffoldResult> {
  const options = parseOptions(rawArgs);
  const cwd = ctx.cwd ?? process.cwd();

  if (!options.input) {
    const candidates = await listDataFiles(cwd);
    if (candidates.length === 1) {
      options.input = candidates[0];
      ctx.ui?.notify(`未指定数据文件，已自动选用 ${options.input}`, "info");
    } else if (candidates.length === 0) {
      return {
        kind: "failed",
        message: `缺少数据文件参数，且当前目录（含一层子目录）没有找到常见格式的数据文件。\n\n${USAGE}`,
      };
    } else {
      return {
        kind: "failed",
        message: [
          "缺少数据文件参数。当前目录下的候选文件：",
          ...candidates.map((file) => `  - ${file}`),
          "",
          "请让用户从中选一个，然后重新执行 /tsa <文件>。",
          "",
          USAGE,
        ].join("\n"),
      };
    }
  }

  const inputPath = resolve(cwd, options.input.replace(/^@/, ""));
  let fileSize = 0;
  try {
    const info = await stat(inputPath);
    if (!info.isFile()) throw new Error("not a file");
    fileSize = info.size;
  } catch {
    return { kind: "failed", message: `找不到输入文件: ${inputPath}` };
  }
  if (fileSize === 0) {
    return { kind: "failed", message: `输入文件为空: ${inputPath}` };
  }

  const ext = extname(inputPath).toLowerCase();
  if (!DATA_EXTENSIONS.has(ext)) {
    ctx.ui?.notify(`扩展名 ${ext || "(无)"} 不是常见数据格式，将按纯文本表格尝试解析。`, "warning");
  }
  const strayFiles = options.extra.filter((token) => isDataFile(token));
  if (strayFiles.length) {
    ctx.ui?.notify(`只有第一个位置参数会被当作数据文件，已忽略: ${strayFiles.join(", ")}`, "warning");
  }

  const slug = slugify(options.name ?? basename(inputPath, ext));
  const runDir = join(cwd, `${RUN_PREFIX}${slug}-${timestampSlug()}`);
  await mkdir(join(runDir, "scripts"), { recursive: true });
  await mkdir(join(runDir, "out"), { recursive: true });
  await mkdir(join(runDir, "figures"), { recursive: true });

  const skillDir = await findSkillDir();
  let scriptsCopied = 0;
  if (skillDir) {
    try {
      const files = await readdir(join(skillDir, "scripts"));
      for (const file of files) {
        if (!file.endsWith(".py")) continue;
        await cp(join(skillDir, "scripts", file), join(runDir, "scripts", file));
        scriptsCopied++;
      }
    } catch {
      // best effort
    }
  }

  const context = {
    schema: "tsa/context@1",
    created_at: new Date().toISOString(),
    input_path: inputPath,
    input_file: basename(inputPath),
    input_bytes: fileSize,
    run_dir: runDir,
    cwd,
    slug,
    scripts_copied: scriptsCopied,
    options: {
      time_col: options.timeCol ?? null,
      value_col: options.valueCol ?? null,
      freq: options.freq ?? null,
      agg: options.agg ?? "mean",
      horizon: Number.isFinite(options.horizon as number) ? options.horizon : null,
      extra: options.extra,
    },
    resolved: {},
    steps: [
      { id: 1, name: "profile", script: "step1_profile.py", status: "pending" },
      { id: 2, name: "stationarity", script: "step2_stationarity.py", status: "pending" },
      { id: 3, name: "modeling", script: "step3_modeling.py", status: "pending" },
      { id: 4, name: "anomaly", script: "step4_anomaly.py", status: "pending" },
      { id: 5, name: "report", script: "assemble_report.py", status: "pending" },
    ],
  };
  await writeFile(join(runDir, "context.json"), `${JSON.stringify(context, null, 2)}\n`, "utf8");
  await writeFile(join(runDir, "pyproject.toml"), PYPROJECT, "utf8");

  let envReady = false;
  let envNote = "使用 --no-setup 跳过环境创建，请在任务目录内执行 `uv sync`";
  if (!options.noSetup) {
    const version = await pi.exec("uv", ["--version"], { timeout: 20_000 }).catch(() => null);
    if (!version || version.code !== 0) {
      envNote = "未找到 uv。请安装 https://docs.astral.sh/uv/ ，然后执行 `uv sync`；或改用系统 python（需用户明确同意）";
      ctx.ui?.notify("未找到 uv 可执行文件，环境创建已跳过。", "error");
    } else {
      ctx.ui?.notify("正在创建 uv 虚拟环境（首次运行需要联网下载依赖）...", "info");
      const sync = await pi
        .exec("uv", ["sync"], { cwd: runDir, timeout: 900_000 })
        .catch((error: unknown) => ({ code: -1, stdout: "", stderr: String(error) }));
      if (sync.code === 0) {
        envReady = true;
        envNote = `uv sync 成功（${version.stdout.trim()}）`;
      } else {
        envNote = `uv sync 失败（退出码 ${sync.code}）：${(sync.stderr || sync.stdout || "").trim().slice(-400)}。请在任务目录内手动排查后重试`;
        ctx.ui?.notify("uv sync 失败，请在任务目录内手动排查后重试。", "error");
      }
    }
  }

  ctx.ui?.notify(`TSA 任务已创建: ${runDir}`, "info");
  return {
    kind: "prompt",
    text: buildPrompt({ runDir, inputPath, skillDir, options, envReady, envNote }),
  };
}

export default function (pi: ExtensionAPI) {
  pi.on("input", async (event, ctx) => {
    const match = COMMAND_PATTERN.exec((event.text ?? "").trim());
    if (!match) return { action: "continue" as const };

    let result: ScaffoldResult;
    try {
      result = await scaffold(match[1] ?? "", ctx, pi);
    } catch (error) {
      result = { kind: "failed", message: `创建任务失败: ${error instanceof Error ? error.message : String(error)}` };
    }

    return {
      action: "transform" as const,
      text: result.kind === "prompt" ? result.text : buildFailurePrompt(result.message),
    };
  });
}
