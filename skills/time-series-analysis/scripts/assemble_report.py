#!/usr/bin/env python
"""Step 5 — 汇总报告.

用法:
    uv run python scripts/assemble_report.py

把 context.json 元数据与 out/stepN_*.md 片段合并成 report.md，
并写出 out/run_manifest.json。

report.md 中会保留两个占位符，等待 agent 填成正文:
    <!-- TSA:SUMMARY -->        -> 摘要（3-6 条要点）
    <!-- TSA:CONCLUSIONS -->    -> 结论、不确定性、建议、后续工作
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsa_lib as lib  # noqa: E402

STEP, NAME = 5, "report"

SECTIONS = [
    (1, "profile", "step1_profile.py", "step1_profile.md"),
    (2, "stationarity", "step2_stationarity.py", "step2_stationarity.md"),
    (3, "modeling", "step3_modeling.py", "step3_modeling.md"),
    (4, "anomaly", "step4_anomaly.py", "step4_anomaly.md"),
]

SUMMARY_SLOT = "<!-- TSA:SUMMARY -->"
CONCLUSIONS_SLOT = "<!-- TSA:CONCLUSIONS -->"

PLACEHOLDER_HINT = {
    SUMMARY_SLOT: (
        "<!-- TSA:SUMMARY -->\n"
        "_(待填写：3–6 条要点。第一条给最重要的结论，每条都带具体数字；"
        "包含数据质量、是否平稳/有季节性、最优模型及其相对基线的改善、异常与变点数量。)_\n"
    ),
    CONCLUSIONS_SLOT: (
        "<!-- TSA:CONCLUSIONS -->\n"
        "_(待填写：结论 / 不确定性 / 建议（带优先级）/ 后续工作。)_\n"
    ),
}


def read_section(run_dir: Path, filename: str) -> str:
    path = run_dir / "out" / filename
    if not path.exists():
        return f"> ⚠️ 缺少产物 `out/{filename}`，该步骤未执行或已失败。\n"
    return path.read_text(encoding="utf-8").strip() + "\n"


def collect_status(run_dir: Path, step_id: int, name: str, script: str) -> dict:
    path = run_dir / "out" / f"step{step_id}_{name}.json"
    if not path.exists():
        return {"id": step_id, "name": name, "script": script, "status": "missing",
                "artifact_json": None, "artifact_md": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "id": step_id,
        "name": name,
        "script": script,
        "status": data.get("status", "unknown"),
        "artifact_json": f"out/step{step_id}_{name}.json",
        "artifact_md": f"out/step{step_id}_{name}.md",
        "elapsed_sec": data.get("elapsed_sec"),
        "warnings": data.get("warnings", []),
        "errors": data.get("errors", []),
        "notes": "",
    }


def step(run_dir: Path, warnings: list[str]):
    ctx = lib.load_context(run_dir)
    resolved = ctx.get("resolved") or {}
    options = ctx.get("options") or {}
    input_file = ctx.get("input_file") or Path(ctx.get("input_path", "?")).name

    header_rows = [
        ["生成时间", lib.now_iso()],
        ["数据文件", f"`{input_file}`"],
        ["观测区间", f"{resolved.get('start')} → {resolved.get('end')}"],
        ["观测点数", f"{lib.fmt(resolved.get('n_obs'))}（时间槽 {lib.fmt(resolved.get('n_slots'))}）"],
        ["频率 / 周期", f"`{resolved.get('freq') or '未知'}` / {lib.fmt(resolved.get('period'))}"],
        ["时间列 / 目标列", f"`{resolved.get('time_col')}` / `{resolved.get('value_col')}`"],
        ["数据质量", f"**{resolved.get('quality_grade', '未评级')}**"],
        ["预测步长", lib.fmt(resolved.get("horizon"))],
        ["任务目录", f"`{ctx.get('run_dir') or run_dir}`"],
        ["请求参数", "、".join(f"{k}={v}" for k, v in options.items() if v not in (None, "", [])) or "（全部自动探测）"],
    ]

    manifest_steps = [collect_status(run_dir, step_id, name, script) for step_id, name, script, _ in SECTIONS]
    # step5 自身的 JSON 在本步结束后才落盘，不能因此报「缺失」
    report_status = {"id": STEP, "name": NAME, "script": "assemble_report.py", "status": "ok",
                     "artifact_json": None, "artifact_md": None,
                     "elapsed_sec": None, "warnings": [], "errors": [], "notes": "由本步骤生成"}
    manifest_steps.append(report_status)

    failed = [s for s in manifest_steps if s["status"] in ("failed", "missing", "skipped")]
    alert = ""
    if failed:
        lines = ["", "> ⚠️ **本次分析存在未完成或失败的步骤，结论强度因此受限：**", ">"]
        for entry in failed:
            reason = "; ".join(entry.get("errors") or entry.get("warnings") or [])[:200] or "无详细说明"
            lines.append(f"> - 步骤 {entry['id']} `{entry['name']}`：**{entry['status']}** — {reason}")
        lines.append("")
        alert = "\n".join(lines)

    parts = [
        f"# 时序数据分析报告 — {input_file}",
        "",
        lib.md_table(["项目", "值"], header_rows),
        "",
    ]
    if alert:
        parts.extend([alert, ""])

    parts.extend(["## 摘要", "", PLACEHOLDER_HINT[SUMMARY_SLOT], ""])
    for step_id, _name, _script, filename in SECTIONS:
        parts.append(read_section(run_dir, filename))
        parts.append("")
    parts.extend(["## 5. 结论与建议", "", PLACEHOLDER_HINT[CONCLUSIONS_SLOT], ""])

    # ---- 附录 A: 产物清单 --------------------------------------------------
    parts.append("## 附录 A. 产物清单\n")
    parts.append(lib.md_table(
        ["文件", "说明"],
        [
            ["`context.json`", "运行元数据与已解析的数据契约"],
            ["`out/stepN_*.json`", "每步的机器可读结果（报告中的数字都来源于此）"],
            ["`out/stepN_*.md`", "每步的报告片段"],
            ["`figures/*.png`", "图表"],
            ["`out/run_manifest.json`", "本次运行的产物清单与状态"],
            ["`report.md`", "本报告"],
        ],
    ))

    figures = sorted((run_dir / "figures").glob("*.png"))
    if figures:
        parts.append("\n### 图表\n")
        for figure in figures:
            parts.append(f"- `figures/{figure.name}`")

    # ---- 附录 B: 复现方式 --------------------------------------------------
    parts.append("\n## 附录 B. 复现方式\n")
    parts.append("```bash")
    parts.append(f"cd {ctx.get('run_dir') or run_dir}")
    parts.append("uv sync")
    parts.append("uv run python scripts/step1_profile.py")
    parts.append("uv run python scripts/step2_stationarity.py")
    parts.append("uv run python scripts/step3_modeling.py")
    parts.append("uv run python scripts/step4_anomaly.py")
    parts.append("uv run python scripts/assemble_report.py")
    parts.append("```")
    parts.append("")
    parts.append("报告中的每个数字都可以在 `out/*.json` 中找到来源；若重新执行，随机过程已固定种子，结果应可复现。\n")

    report = "\n".join(parts).replace("\n\n\n", "\n\n").rstrip() + "\n"
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "run_dir": str(ctx.get("run_dir") or run_dir),
        "input_file": input_file,
        "input_path": ctx.get("input_path"),
        "steps": manifest_steps,
        "report": "report.md",
        "figures": [f"figures/{f.name}" for f in figures],
        "generated_at": lib.now_iso(),
    }
    lib.save_json(run_dir / "out" / "run_manifest.json", manifest)

    placeholders_left = report.count(SUMMARY_SLOT) + report.count(CONCLUSIONS_SLOT)
    if placeholders_left:
        warnings.append(f"report.md 仍有 {placeholders_left} 个占位符待填写（TSA:SUMMARY / TSA:CONCLUSIONS）")

    md = (
        "## 5. 汇总报告\n\n"
        f"- 已生成 `report.md`（{len(report.splitlines())} 行）。\n"
        f"- 步骤状态：" + "、".join(f"{s['id']}:{s['name']}={s['status']}" for s in manifest_steps) + "。\n"
        f"- 图表 {len(figures)} 张。\n"
        + (f"- ⚠️ 仍有占位符待填写：{placeholders_left} 个。\n" if placeholders_left else "- ✅ 占位符已全部填写。\n")
    )

    data = {
        "report_path": str(run_dir / "report.md"),
        "report_lines": len(report.splitlines()),
        "figures": [f"figures/{f.name}" for f in figures],
        "steps": manifest_steps,
        "placeholders_remaining": placeholders_left,
    }
    return data, md


if __name__ == "__main__":
    lib.run_step(STEP, NAME, step)
