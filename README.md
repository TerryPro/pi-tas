# pi-tsa

给 [Pi](https://pi.dev) 用的时序数据分析包：一个 `/tsa` 命令 + 一套技能驱动、`uv` 隔离的 5 步分析流水线，
最终产出**中文 markdown 报告**。

```
/tsa data/sales.csv --value-col amount --freq D --horizon 30
```

## 安装

```bash
pi install /absolute/path/to/pi-extension      # 本地目录
pi install git:github.com/TerryPro/pi-tas@v0.1.1  # 或 npm
```

依赖：`uv >= 0.4`（<https://docs.astral.sh/uv/>）。Python 依赖由包自己声明，首次运行会联网安装。

## 命令

```
/tsa <数据文件> [选项]

  --time-col <列名>    时间列（缺省自动探测）
  --value-col <列名>   目标列（缺省自动探测）
  --freq <频率>        重采样频率，如 D/W/M/h
  --agg <聚合方式>     重采样聚合，默认 mean
  --horizon <步数>     预测步数（缺省 = 2 个周期，上限 n/4）
  --name <标识>        任务目录名标识
  --no-setup           跳过 uv 虚拟环境创建
```

支持格式：csv / tsv / txt / json / jsonl / parquet / feather / xlsx / orc / dta。

## 它做了什么

`/tsa` 分成两半：**确定性的准备工作**由扩展完成，**分析本身**由 skill 驱动 agent 完成。

扩展（不调用模型，毫秒级）：

1. 校验输入文件（存在、非空、扩展名）
2. 在当前目录创建任务目录 `tsa-run-<slug>-<YYYYMMDD-HHmmss>/`
3. 写入 `context.json`（运行元数据 + 选项）与 `pyproject.toml`
4. 把技能里的参考脚本复制到 `<run>/scripts/`
5. 执行 `uv sync` 建立独立虚拟环境
6. 把 prompt 重写成「按技能跑 5 步流水线」的具体指令

Agent 然后按技能执行 5 个步骤（每步都有机器可读产物）：

| # | 步骤 | 主要方法 | 产物 |
|---|------|----------|------|
| 1 | 数据概览与质量检查 | 时间列/目标列探测、频率与周期推断、缺口与缺失分析、异常粗筛 | `out/step1_profile.{json,md}`、`figures/01_*.png` |
| 2 | 平稳性与分解 | ADF / KPSS（原序列、一阶差分、季节差分）、STL、趋势/季节强度、ACF/PACF | `out/step2_stationarity.{json,md}`、`figures/02_*.png` |
| 3 | 建模与回测 | 4 个基线 + ARIMA 网格 + SARIMA + ETS + Ridge/RandomForest，滚动原点回测（MAE/RMSE/MAPE/sMAPE），冠军模型重训预测 + 80% 区间 | `out/step3_modeling.{json,md}`、`figures/03_*.png` |
| 4 | 异常与变点检测 | STL 残差稳健 z-score、滚动中位数/MAD 漂移段、二分分割 + CUSUM 变点（可选 ruptures PELT 交叉验证） | `out/step4_anomaly.{json,md}`、`figures/04_*.png` |
| 5 | 汇总报告 | 合并各步片段 + 元数据，留下两个占位符由 agent 填写 | `report.md`、`out/run_manifest.json` |

完成后目录长这样：

```
tsa-run-sales-20260919-150832/
├── context.json        # 单一事实来源：resolved 字段被 step1 回写
├── pyproject.toml
├── .venv/
├── scripts/            # 参考实现（可直接改）
├── out/                # 每步 JSON + markdown 片段
├── figures/            # 8 张图
└── report.md           # 最终交付物
```

复现：

```bash
cd tsa-run-sales-20260919-150832
uv sync
uv run python scripts/step1_profile.py
uv run python scripts/step2_stationarity.py
uv run python scripts/step3_modeling.py
uv run python scripts/step4_anomaly.py
uv run python scripts/assemble_report.py
```

## 仓库结构

```
extensions/tsa.ts                     # /tsa 入口：校验 + 脚手架 + prompt 重写
prompts/tsa.md                        # 提供 /tsa 的自动补全与描述（独立兜底用法）
sample-data/                          # 可直接拿来试跑的示例数据
├── retail_daily_sales.csv            # 1096 行日频零售数据（含缺失/缺口/重复/异常/漂移）
├── make_sample_data.py               # 固定种子的生成器
└── README.md                         # 植入了什么、应当测出什么
skills/time-series-analysis/
├── SKILL.md                          # 流水线规范、硬性规则、失败处理
├── references/data-contract.md       # context.json / out/*.json / 报告结构契约
├── references/step-details.md        # 每步的判定标准、方法菜单、常见坑
└── scripts/                          # 参考实现（会被复制进任务目录）
    ├── tsa_lib.py                    # 读数据、探测列、重采样、JSON 落盘、matplotlib 配置
    ├── step1_profile.py … step4_anomaly.py
    └── assemble_report.py
```

第一次试用直接用自带示例数据：

```bash
/tsa sample-data/retail_daily_sales.csv
```

## 几个设计决定

**为什么用 `input` 事件而不是 `pi.registerCommand()`？**
注册命令只能用 `pi.sendUserMessage()` 触发 agent 回合，而这个注入在非交互模式（`pi -p`、JSON、RPC）下会被丢弃——
进程在排队回合执行前就退出了。就地改写输入在所有模式下都生效。`prompts/tsa.md` 仍然让 `/tsa` 出现在自动补全里，
且 `input` 事件先于模板展开执行，所以模板体只在扩展未加载时才生效。

**为什么参考脚本要预置？**
`tsa_lib.py` 把「读数据、探测列、重采样、JSON 落盘、无头绘图」这些易错环节固定下来，
agent 只需针对数据调整分析逻辑。技能明确要求「发现脚本缺陷就修脚本再重跑，不要绕过」。

**为什么报告里的数字必须来自 JSON？**
`SKILL.md` 的硬性规则禁止 LLM 口算统计量。每个数字都能在 `out/*.json` 里找到来源，报告因此可核对。

**图表标签用英文？**
默认字体常常没有中文字形，会出现方块乱码。`tsa_lib.configure_matplotlib()` 会检测到 CJK 字体时返回 `cjk=True`，
脚本再用中英标签切换；否则一律英文标签。正文报告始终是中文。

## 已知限制

- 变点检测是启发式的：年度谐波与线性趋势的拟合可能吸收持续时间接近一年的水平偏移，
  报告里会给出首轮切分候选与明确提示；装 `ruptures` 可获得 PELT 交叉验证。
- `--freq` 推断失败时（不规则采样）不做重采样，季节建模会退化为趋势分析。
- 预测区间：统计模型优先用自带区间，否则统一用回测残差的稳健尺度按 `√h` 扩张的经验区间。
- ML 模型需要 ≥ 200 个观测点才会启用；`ruptures`、`lightgbm`、`prophet` 都不在默认依赖里。

## 开发

```bash
pi --no-extensions -e ./extensions/tsa.ts -p "/tsa sample.csv"   # 单次验证
pi -e ./extensions/tsa.ts                                        # 交互式调试
```

改了 `extensions/` 下的文件后，在 Pi 里执行 `/reload` 即可热重载（本地路径安装不会被复制，改完即读）。

### 加载须知（踩过的坑）

**1. 本地安装只是一个指针，不是副本。**
`pi install <dir>` 对目录型来源只做 `existsSync` 校验，然后把**原样的路径字符串**写进 settings，不复制任何文件。
后果：目录被改名/移动/删除后，它的资源会**静默消失**，而 `pi list` 仍然显示该条目（陈旧条目）。
共享/发布时请用 git 或 npm 来源（那两种会克隆/安装到 `~/.pi/agent/{git,npm}/`）。

**2. `package.json` 的 `pi` 清单优先于约定目录。**
只要 `pi` 对象里存在合法的 `extensions` / `skills` / `prompts` / `themes` 数组，pi 就**只**用清单，
不再扫描同名约定目录；没有清单时才回退到 `extensions/`、`skills/`、`prompts/`、`themes/`。
`pi.video` / `pi.image` 只用于包画廊，不参与加载。

**3. SKILL.md 的 frontmatter 必须是合法 YAML，否则技能静默不加载。**
pi 用标准 `yaml` 包解析 frontmatter；解析抛错时它只推一条 warning diagnostic 并把技能丢弃。
最隐蔽的坑：**未加引号的纯量里不能出现 `": "`**。例如

```yaml
# ❌ 解析失败："Nested mappings are not allowed in compact mappings"
description: Pipeline for a time series file: profile and quality audit.

# ✅ 安全写法一：块标量（推荐，能写多行）
description: >-
  Pipeline for a time series file: profile and quality audit.

# ✅ 安全写法二：加引号
description: "Pipeline for a time series file: profile and quality audit."
```

注意中文全角冒号 `：` 不受影响。改完 frontmatter 后务必实测一次：

```bash
# 应能在输出里看到 time-series-analysis
pi -p "只回答你当前可用的 skills 名称，逗号分隔，不要调用工具。"
```

**4. 包资源的优先级最低。**
pi 的资源优先级为：项目设置项 > 项目自动发现 > 用户设置项 > 用户自动发现 > **包（rank 4）**。
所以项目里若有 `.pi/skills/time-series-analysis/`，它会盖掉本包的技能。

**5. `keywords: ["pi-package"]` 与 `files` 不影响加载。**
前者只用于 <https://pi.dev/packages> 画廊收录，后者是 npm 发布的文件白名单。

**6. 用户作用域安装不依赖项目信任。**
`scope === "project"` 才需要项目被信任；写进 `~/.pi/agent/settings.json` 的用户级包在任何目录都加载。

**7. 资源开关用 `pi config`**（Tab 切全局/项目），或用 settings 的对象形式做过滤：

```json
{"packages": [{"source": "F:/TRAEWS/pi-extension", "extensions": [], "skills": ["skills"]}]}
```

## License

MIT
