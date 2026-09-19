---
name: time-series-analysis
description: >-
  End-to-end time-series analysis pipeline for a tabular time-series file: profile and
  data-quality audit, stationarity and decomposition, rolling-origin backtested forecasting,
  anomaly and change-point detection, ending in a Chinese markdown report. Use when the user
  runs /tsa, asks to analyze or forecast a time series, or asks for seasonality, anomaly, or
  change-point analysis of a time-stamped dataset.
metadata:
  version: 0.1.0
  entrypoint: /tsa
---

# 时序数据分析流水线 (TSA)

对一个**带时间戳的表格文件**执行完整分析，最终产出中文 markdown 报告。
整个分析在**任务目录**中运行：独立的 `uv` 虚拟环境，全部代码与产物可复查、可复现。

## 触发方式

```bash
/tsa data/sales.csv                          # 自动探测时间列 / 目标列 / 频率
/tsa data/sales.csv --value-col amount --freq D --horizon 30
/tsa metrics.parquet --time-col ts --no-setup
```

也可以手工启动（没有 `/tsa` 时）：自行创建任务目录，写好 `context.json` 与 `pyproject.toml`，再按下面的流程走。

## 任务目录布局

```
tsa-run-<slug>-<YYYYMMDD-HHmmss>/
├── context.json          # 运行元数据（由 /tsa 写入，step1 会补全 resolved 字段）
├── pyproject.toml        # 依赖声明
├── .venv/                # uv 创建的虚拟环境
├── scripts/              # 分析脚本（参考脚本已由 /tsa 复制进来）
├── out/                  # 每步的机器可读产物 + markdown 片段
│   ├── step1_profile.json / step1_profile.md
│   ├── step2_stationarity.json / step2_stationarity.md
│   ├── step3_modeling.json / step3_modeling.md
│   ├── step4_anomaly.json / step4_anomaly.md
│   └── run_manifest.json
├── figures/              # PNG 图表
├── report.md             # 最终中文报告
└── RUN_LOG.md            # 执行日志（可选但推荐）
```

## 硬性规则

1. **一律通过 `uv` 执行，绝不污染全局环境。**
   ```bash
   cd <run_dir>
   uv sync                       # 环境不存在或依赖变动时
   uv run python scripts/step1_profile.py
   ```
   不要用 `pip install`，不要用系统 `python`，不要 `uv pip install --system`。
2. **所有命令的 cwd 必须是任务目录**，脚本默认把「脚本的上级目录」当作任务目录。
   需要指向别处时显式加 `--run <run_dir>`。
3. **每步都要落盘产物**：`out/stepN_*.json`（机器可读）+ `out/stepN_*.md`（报告片段）+ `figures/*.png`（图）。
   报告片段由脚本生成，不要手写——手写的数字无法核对。
4. **`context.json` 是单一事实来源。** step1 探测到的 `time_col` / `value_col` / `freq` / `period` 会写回
   `context.json.resolved`，后续步骤全部读它，不得各自重新猜测。
5. **不允许伪造结果。** 某个模型跑不动、缺依赖、数据不适合，就在产物和报告里写「未完成 + 具体原因 + 已尝试的做法」。
6. **可复现**：随机过程必须固定随机种子；脚本不能依赖交互输入；所有图表在 headless 模式生成。
7. **报告是中文，图表标签用英文**（默认字体不一定有中文字形，用英文标签避免方块乱码；
   若 `tsa_lib.configure_matplotlib()` 返回 `cjk=True` 才可以用中文标签）。
8. **不要让 LLM 口算统计量。** 任何写进报告的数字都必须来自脚本产出的 JSON/markdown。

## 工作流

按顺序执行。每步结束后检查产物是否合理，再进入下一步。

### Step 1 — 数据概览与质量检查

```bash
uv run python scripts/step1_profile.py
```

- 读取数据，识别时间列与目标列（命令行 `--time-col` / `--value-col` 优先，其次 `context.json.options`，最后自动探测）。
- 推断采样频率，与理论周期比对，找出**缺口（gap）、重复时间戳、乱序、时区问题**。
- 缺失值统计（按列 + 连续缺失区间）、异常值粗筛（IQR / 稳健 z-score）、描述统计。
- 产出：`out/step1_profile.json`、`out/step1_profile.md`、`figures/01_series.png`、`figures/01_rolling.png`。
- **并且**把 `resolved.time_col`、`resolved.value_col`、`resolved.freq`、`resolved.period`、`resolved.n_obs` 写回 `context.json`。

判定：若有效观测点 < 30 或时间列无法解析，停止后续建模步骤，只输出数据质量报告并说明原因。

详见 [references/step-details.md](references/step-details.md#step-1)。

### Step 2 — 平稳性与分解

```bash
uv run python scripts/step2_stationarity.py
```

- ADF / KPSS 检验：原序列、一阶差分、季节差分（按 `period`）。
- STL 分解（季节稳健），计算**趋势强度 / 季节强度**，给 `period < 2` 或点数不足时优雅降级为仅趋势。
- ACF / PACF 关键滞后值，给出 `p`、`d`、`q`、`P`、`D`、`Q` 的**建议区间**。
- 产出：`out/step2_stationarity.json`、`out/step2_stationarity.md`、`figures/02_stl.png`、`figures/02_acf_pacf.png`。

详见 [references/step-details.md](references/step-details.md#step-2)。

### Step 3 — 建模与回测

```bash
uv run python scripts/step3_modeling.py
```

- 基线：`naive`、`seasonal_naive`、`mean`、`drift`。**基线是及格线，模型打不过基线就必须说出来。**
- 统计模型：`ARIMA`（在小网格上按 AIC 选阶，受 Step 2 建议约束）、`ETS`（Holt-Winters）。
- 机器学习（可选，数据量足够时）：滞后 + 日历特征 + `Ridge`/`RandomForest`。
- **滚动原点回测（rolling-origin）**：至少 3 折（数据不足则 2 折，或退化为单次 hold-out），
  每折用相同预测步长 `horizon`（默认 = 2 × period，最小 1，最大 ≤ n/4），报告 MAE / RMSE / MAPE / sMAPE。
- 用回测冠军在全量数据上重训，输出未来 `horizon` 步预测 + 预测区间。
- **必须做配对显著性检验**（同一回测折内做差），并列出显著劣于基线的模型；
  只看平均 RMSE 差距会把“折间波动”当成“模型优势”。
- 产出：`out/step3_modeling.json`、`out/step3_modeling.md`、`figures/03_backtest.png`、`figures/03_forecast.png`。

详见 [references/step-details.md](references/step-details.md#step-3)。

### Step 4 — 异常与变点检测

```bash
uv run python scripts/step4_anomaly.py
```

- 三条互补证据：
  1. STL 残差的稳健 z-score（MAD）→ 点异常；
  2. 同一残差 z 上连续 ≥ 2 天的聚集 → 多日异常窗口（不要用局部 MAD 比值法，它会给出一堆假窗口）；
  3. 均值/方差变点（去周季节 + 去年度谐波后做分段线性二分分割 + 去趋势 CUSUM；
     若环境里装了 `ruptures` 则用 PELT 做交叉验证）→ 永久性结构断裂。
- 输出异常时间点/时间段清单（含方向、幅度、偏离倍数），按严重程度排序，限制在合理数量（默认 top 20 个点 + 全部时间段）。
- 产出：`out/step4_anomaly.json`、`out/step4_anomaly.md`、`figures/04_anomalies.png`、`figures/04_changepoints.png`。

详见 [references/step-details.md](references/step-details.md#step-4)。

### Step 5 — 汇总报告

```bash
uv run python scripts/assemble_report.py
```

- 合并四个步骤的 markdown 片段 + `context.json` 元数据 → `report.md`。
- 报告里保留两个占位符，由**你**（agent）填写成正文：
  - `<!-- TSA:SUMMARY -->`：3–6 条要点，先给结论再给依据；
  - `<!-- TSA:CONCLUSIONS -->`：结论、风险与不确定性、可执行建议、后续工作。
- 填好占位符后，`report.md` 就是唯一交付物。同时把每步的状态更新进 `out/run_manifest.json`。

报告结构见 [references/data-contract.md](references/data-contract.md#报告结构)。

## 失败处理

| 情况 | 处理 |
|------|------|
| `uv` 不存在 | 告知用户安装方式，改用系统 python 仅在用户明确同意后；否则只做数据质量部分 |
| 依赖装不上 | 缩小依赖集（例如去掉 scikit-learn），把降级情况写进报告 |
| 时间列解析失败 | 让用户用 `--time-col` 指定；或把首列当作序号索引并明确标注这是假设 |
| 数据点太少 / 频率不规则 | 跳过建模与异常步骤，只出质量报告 + 趋势描述 |
| 某模型不收敛 | 记录收敛警告，跳过该模型，保留其余结果 |
| 脚本有 bug | **修脚本再重跑**，不要把结果手工写进产物 |

## 参考文件

- [references/step-details.md](references/step-details.md) — 每一步的判定标准、插值与建模方法菜单、常见坑
- [references/data-contract.md](references/data-contract.md) — `context.json` 与 `out/*.json` 的字段契约、报告结构
- `scripts/tsa_lib.py` — 共享工具（读数据、探测列、重采样、JSON 落盘、matplotlib 配置）
- `scripts/step1_profile.py` … `scripts/step4_anomaly.py`、`scripts/assemble_report.py` — 参考实现
