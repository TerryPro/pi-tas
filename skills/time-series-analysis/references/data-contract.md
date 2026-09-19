# 数据契约

所有跨步骤通信都走文件，不走内存。字段名固定，后续步骤只读这些契约。

## `context.json`

```jsonc
{
  "schema": "tsa/context@1",
  "created_at": "2026-02-14T10:00:00.000Z",
  "input_path": "/abs/path/data.csv",
  "input_file": "data.csv",
  "run_dir": "/abs/path/tsa-run-data-20260214-100000",
  "cwd": "/abs/path",
  "slug": "data",
  "options": {                 // 命令行/用户传入的原始意图，可能为 null
    "time_col": null,
    "value_col": null,
    "freq": null,
    "agg": "mean",
    "horizon": null,
    "extra": []
  },
  "resolved": {                // step1 写入，之后的步骤只信这里
    "time_col": "date",
    "value_col": "sales",
    "freq": "D",
    "period": 7,
    "n_obs": 730,
    "start": "2024-01-01T00:00:00",
    "end": "2025-12-30T00:00:00",
    "agg": "mean",
    "horizon": 14,
    "has_seasonality": true,
    "quality_grade": "ok"      // ok | warn | poor
  },
  "steps": [
    { "id": 1, "name": "profile", "script": "step1_profile.py", "status": "pending" }
  ]
}
```

写回 `resolved` 时**必须保留 `options` 与 `steps`**（局部更新，不要整体覆盖）。

`horizon` 解析优先级：`options.horizon` → `2 * period` → `1`；再夹到 `[1, floor(n_obs / 4)]`。

## `out/stepN_<name>.json`

统一外壳：

```jsonc
{
  "step": 1,
  "name": "profile",
  "status": "ok",          // ok | warn | skipped | failed
  "started_at": "...",
  "finished_at": "...",
  "elapsed_sec": 1.23,
  "warnings": ["..."],
  "errors": [],
  "data": { }              // 各步自定义，见下
}
```

各步 `data` 的关键字段：

| 步骤 | 关键字段 |
|------|----------|
| 1 profile | `shape`, `columns`, `time_col`, `value_col`, `dtype`, `freq`, `period`, `n_missing`, `missing_runs`, `duplicate_timestamps`, `gaps`, `describe`, `outliers_iqr`, `quality_grade`, `notes` |
| 2 stationarity | `tests` (adf/kpss × level/diff/seasonal_diff，含 statistic/pvalue/stationary), `decomposition` (`period`, `trend_strength`, `seasonal_strength`, `resid_std`), `acf` (`lags`, `values`, `significant_lags`), `pacf`, `suggested_orders` (`p`,`d`,`q`,`P`,`D`,`Q`,`m`) |
| 3 modeling | `backtest` (`folds`, `horizon`, `metrics`: {model: {MAE,RMSE,MAPE,sMAPE}}), `ranking`, `ranking_by_rmse`, `best_model`, `baseline_best`, `improvement_vs_baseline`, `champion_significance`, `significance_vs_baseline` ({model,baseline,n_folds,mean_delta_rmse,paired_se,t_stat,df,critical_t_5pct,significant_5pct,better}), `forecast` (`model`, `index`, `mean`, `lower`, `upper`, `level`), `config`, `fit_notes` |
| 4 anomaly | `anomalies` (list of {timestamp, value, score, direction, severity, method, iqr_hit}), `anomaly_candidates`, `anomaly_rate`, `episodes` / 多日异常窗口 ({start,end,periods,level,reference,level_shift,level_shift_pct,max_abs_z,direction,missing_days_inside,method,kind,note}), `changepoints` ({index,timestamp,type,delta,level_before,level_after,mean_before,mean_after,variance_ratio,gain,penalty,method,corroborated_by,confidence}), `changepoint_candidates`, `cusum_points`, `ruptures_points`, `config`, `summary` |

`Timestamp`、`numpy` 标量、`NaN` 都要能在 JSON 里表达：用 `tsa_lib.save_json()`，它会把非有限浮点转成 `null`。

## `out/stepN_<name>.md`

- 纯 markdown 片段，**包含自己的 `## N. 标题`**，由脚本生成。
- 表格用 markdown 表格；图用相对路径 `![](../figures/xx.png)`（`report.md` 与 `figures/` 同级，所以是 `figures/xx.png`）。
  片段文件在 `out/` 下，脚本生成时按「相对于 `report.md`」写路径，即 `figures/...`，不要写 `../figures/...`。
- 不要在片段里写「摘要」「结论」——那两段由 agent 在 `report.md` 里填。

## 图表约定

| 文件 | 内容 |
|------|------|
| `figures/01_series.png` | 原始序列（含缺口标记） |
| `figures/01_rolling.png` | 滚动均值/标准差 |
| `figures/02_stl.png` | STL 分解四联图 |
| `figures/02_acf_pacf.png` | ACF / PACF |
| `figures/03_backtest.png` | 各模型回测指标对比 |
| `figures/03_forecast.png` | 历史尾部 + 预测 + 区间 |
| `figures/04_anomalies.png` | 异常点标注 |
| `figures/04_changepoints.png` | 变点标注 |

标签用英文（见 SKILL.md 规则 7）。

## `out/run_manifest.json`

```jsonc
{
  "run_dir": "...",
  "input_file": "data.csv",
  "steps": [
    { "id": 1, "name": "profile", "script": "step1_profile.py", "status": "ok",
      "artifact_json": "out/step1_profile.json", "artifact_md": "out/step1_profile.md",
      "elapsed_sec": 1.23, "notes": "" }
  ],
  "report": "report.md",
  "generated_at": "..."
}
```

## 报告结构

`report.md` 由 `assemble_report.py` 生成骨架：

```markdown
# 时序数据分析报告 — <input_file>

> 生成时间 / 数据文件 / 观测区间 / 频率 / 观测点 / 环境

## 摘要
<!-- TSA:SUMMARY -->

## 1. 数据概览与质量检查
...（step1 片段）

## 2. 平稳性与分解
...（step2 片段）

## 3. 建模与回测
...（step3 片段）

## 4. 异常与变点检测
...（step4 片段）

## 5. 结论与建议
<!-- TSA:CONCLUSIONS -->

## 附录 A. 产物清单
## 附录 B. 复现方式
```

`## 摘要` 要求 3–6 条要点：最重要的结论放第一条，每条带具体数字。
`## 5. 结论与建议` 要求包含：结论、不确定性/风险、可执行建议（带优先级）、后续工作。

若某步 `status` 不是 `ok`，骨架会在对应位置插入告警块，你必须在摘要与结论中如实反映。
