#!/usr/bin/env python
"""Step 1 — 数据概览与质量检查.

用法:
    uv run python scripts/step1_profile.py [--input data.csv] [--time-col ts] [--value-col y] [--freq D]

产物:
    out/step1_profile.json, out/step1_profile.md
    figures/01_series.png, figures/01_rolling.png
    并回写 context.json.resolved
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsa_lib as lib  # noqa: E402

STEP, NAME = 1, "profile"


def missing_runs(series: pd.Series, min_len: int = 2, limit: int = 10) -> list[dict]:
    """连续缺失区间."""
    is_na = series.isna()
    if not is_na.any():
        return []
    groups = is_na.ne(is_na.shift()).cumsum()
    runs = []
    for _, chunk in series.groupby(groups):
        if chunk.isna().all() and len(chunk) >= min_len:
            runs.append({"start": str(chunk.index[0]), "periods": int(len(chunk))})
    runs.sort(key=lambda item: item["periods"], reverse=True)
    return runs[:limit]


def gap_report(observed: pd.DatetimeIndex, freq: str | None, limit: int = 10) -> dict:
    """基于重采样前的时间戳计算真实缺口."""
    observed = pd.DatetimeIndex(observed).unique().sort_values()
    if not freq or len(observed) < 3:
        return {"checked": False, "reason": "未能推断频率或时间点过少"}
    expected = pd.date_range(observed.min(), observed.max(), freq=freq)
    missing = expected.difference(observed)
    gaps = []
    if len(missing):
        # 在规则网格上做游程编码，与 DatetimeIndex 的底层单位无关
        missing_mask = pd.Series(~expected.isin(observed)).to_numpy()
        groups = pd.Series(missing_mask).ne(pd.Series(missing_mask).shift()).cumsum()
        for _, chunk in pd.Series(expected).groupby(groups.to_numpy()):
            if len(chunk) == 0 or not bool(missing_mask[chunk.index[0]]):
                continue
            gaps.append({"start": str(chunk.iloc[0]), "end": str(chunk.iloc[-1]), "periods": int(len(chunk))})
        gaps.sort(key=lambda item: item["periods"], reverse=True)
    return {
        "checked": True,
        "expected_points": int(len(expected)),
        "actual_points": int(len(observed)),
        "missing_slots": int(len(missing)),
        "slot_rate": float(len(missing) / max(len(expected), 1)),
        "largest_gaps": gaps[:limit],
    }


def build_frame(run_dir: Path, settings: dict, warnings: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    input_path = settings["input_path"]
    if not input_path:
        raise ValueError("context.json 缺少 input_path，请用 --input 指定数据文件")
    input_path = Path(input_path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {input_path}")

    raw = lib.read_table(input_path)
    series, info = lib.prepare_frame(
        raw,
        time_col=settings.get("time_col"),
        value_col=settings.get("value_col"),
        freq=settings.get("freq"),
        agg=settings.get("agg") or "mean",
    )
    warnings.extend(info.get("warnings", []))
    return raw, series, info


def step(run_dir: Path, warnings: list[str]):
    settings = lib.load_settings(run_dir, lib.parse_argv())
    raw, frame, prep = build_frame(run_dir, settings, warnings)
    series = frame.iloc[:, 0]
    target = series.name

    freq = prep.get("freq")
    value_dtype = str(raw[prep["value_col"]].dtype)
    period = lib.period_for_freq(freq)
    has_seasonality = period > 1 and len(series) >= 2 * period
    if period > 1 and not has_seasonality:
        warnings.append(f"观测点不足（{len(series)}），无法估计周期 {period} 的季节项，已按无季节性处理")
        period = 1
        has_seasonality = False

    observed = series.dropna()
    n_obs = int(observed.size)
    missing = int(series.isna().sum())
    missing_pct = float(missing / max(len(series), 1))

    # ---- 无信息列 ---------------------------------------------------------
    constant_columns = [str(c) for c in raw.columns if raw[c].nunique(dropna=True) <= 1]
    all_null_columns = [str(c) for c in raw.columns if raw[c].isna().all()]
    if all_null_columns:
        warnings.append(f"存在全空列: {', '.join(all_null_columns)}")

    observed_index = prep.get("observed_index", series.index)
    gaps = gap_report(observed_index, freq)
    duplicates = int(prep.get("duplicate_timestamps", 0))

    # ---- 描述统计 ---------------------------------------------------------
    describe = {}
    if n_obs:
        stats = observed.describe()
        describe = {
            "count": int(stats["count"]),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]) if n_obs > 1 else 0.0,
            "min": float(stats["min"]),
            "p25": float(stats["25%"]),
            "median": float(stats["50%"]),
            "p75": float(stats["75%"]),
            "max": float(stats["max"]),
            "skew": float(observed.skew()) if n_obs > 2 else None,
            "kurtosis": float(observed.kurtosis()) if n_obs > 3 else None,
        }

    # ---- 异常粗筛 ---------------------------------------------------------
    outliers = {"iqr": 0, "robust_z": 0, "both": 0, "rate": 0.0, "top": []}
    if n_obs >= 10:
        q1, q3 = float(observed.quantile(0.25)), float(observed.quantile(0.75))
        iqr = q3 - q1
        iqr_mask = (observed < q1 - 1.5 * iqr) | (observed > q3 + 1.5 * iqr)
        median = float(observed.median())
        mad = float((observed - median).abs().median())
        scale = 1.4826 * mad if mad > 0 else (float(observed.std()) or 1.0)
        z = (observed - median) / scale
        z_mask = z.abs() > 3.5
        both = iqr_mask & z_mask
        combined_mask = iqr_mask | z_mask
        selected = pd.DataFrame({"value": observed[combined_mask], "z": z[combined_mask]})
        selected["abs_z"] = selected["z"].abs()
        top = selected.sort_values("abs_z", ascending=False).head(10)
        outliers = {
            "iqr": int(iqr_mask.sum()),
            "robust_z": int(z_mask.sum()),
            "both": int(both.sum()),
            "rate": float((iqr_mask | z_mask).mean()),
            "median": median,
            "mad": mad,
            "top": [
                {"timestamp": str(idx), "value": float(row["value"]), "robust_z": float(row["z"])}
                for idx, row in top.iterrows()
            ],
        }

    # ---- 数据质量评级 -----------------------------------------------------
    gap_rate = gaps.get("slot_rate", 0.0) if gaps.get("checked") else 0.0
    if n_obs < 30 or not freq:
        grade = "poor"
    elif missing_pct > 0.25 or gap_rate > 0.2:
        grade = "poor"
    elif missing_pct > 0.05 or gap_rate > 0.05 or duplicates > 0 or prep.get("unsorted"):
        grade = "warn"
    else:
        grade = "ok"
    if grade != "ok":
        warnings.append(f"WARN_ONLY: 数据质量评级为 {grade}（缺失 {missing_pct:.1%}，缺口 {gap_rate:.1%}，重复时间戳 {duplicates}）")

    # ---- 周期选择 ---------------------------------------------------------
    n_for_horizon = max(n_obs, 1)
    default_horizon = max(period if period > 1 else 1, 1) * 2
    horizon_raw = settings.get("horizon")
    horizon = int(horizon_raw) if str(horizon_raw or "").strip().isdigit() and int(horizon_raw) > 0 else default_horizon
    horizon = int(min(max(horizon, 1), max(n_for_horizon // 4, 1)))

    resolved = {
        "time_col": prep["time_col"],
        "value_col": prep["value_col"],
        "freq": freq,
        "period": period,
        "has_seasonality": has_seasonality,
        "n_obs": n_obs,
        "n_slots": int(len(series)),
        "start": str(series.index.min()) if len(series) else None,
        "end": str(series.index.max()) if len(series) else None,
        "agg": settings.get("agg") or "mean",
        "horizon": horizon,
        "quality_grade": grade,
        "input_rows": int(len(raw)),        "input_columns": [str(c) for c in raw.columns][:200],
    }
    lib.update_resolved(run_dir, resolved)

    # ---- 图表 -------------------------------------------------------------
    plt, cjk = lib.configure_matplotlib()
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    fig, ax = plt.subplots()
    ax.plot(series.index, series.to_numpy(), linewidth=0.9, color="#1f77b4")
    if missing:
        for span in lib_missing_index_runs(series):
            ax.axvspan(span[0], span[1], color="#d62728", alpha=0.15, linewidth=0)
    ax.set_title(L("原始序列（阴影=缺失区间）", f"{target} — raw series (shaded = missing)"))
    ax.set_xlabel(L("时间", "time"))
    ax.set_ylabel(target)
    lib.savefig(fig, run_dir, "01_series.png")

    window = period * 2 if period > 1 else 7
    if n_obs >= window + 1:
        rolling_mean = series.rolling(window, min_periods=1).mean()
        rolling_std = series.rolling(window, min_periods=2).std()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax1.plot(series.index, series.to_numpy(), linewidth=0.8, alpha=0.6, label=L("原始", "raw"))
        ax1.plot(rolling_mean.index, rolling_mean.to_numpy(), linewidth=1.6, label=L(f"滚动均值({window})", f"rolling mean({window})"))
        ax1.legend(loc="best")
        ax1.set_title(L("滚动均值", "Rolling mean"))
        ax2.plot(rolling_std.index, rolling_std.to_numpy(), linewidth=1.4, color="#ff7f0e", label=L(f"滚动标准差({window})", f"rolling std({window})"))
        ax2.legend(loc="best")
        ax2.set_title(L("滚动标准差", "Rolling std"))
        lib.savefig(fig, run_dir, "01_rolling.png")
    else:
        warnings.append("观测点过少，跳过滚动统计图")

    # ---- markdown ---------------------------------------------------------
    overview_rows = [
        ["数据文件", f"`{Path(settings['input_path']).name}`"],
        ["行 × 列（原始）", f"{len(raw):,} × {raw.shape[1]:,}"],
        ["时间列 / 目标列", f"`{prep['time_col']}` / `{prep['value_col']}`"],
        ["取值类型", f"`{value_dtype}`"],
        ["观测区间", f"{resolved['start']} → {resolved['end']}"],
        ["时间点 / 有效值", f"{len(series):,} / {n_obs:,}"],
        ["推断频率", f"`{freq or '未知'}`（周期 {period}）"],
        ["重采样补齐", "是" if prep.get("resampled") else "否"],
        ["质量评级", f"**{grade}**"],
    ]

    quality_rows = [
        ["解析失败的时间戳", lib.fmt(prep.get("unparsed_timestamps", 0))],
        ["无法解析为数字的取值", lib.fmt(prep.get("non_numeric_values", 0))],
        ["缺失值", f"{missing:,} ({missing_pct:.2%})"],
        ["重复时间戳", lib.fmt(duplicates)],
        ["已删除的无时间戳行", lib.fmt(prep.get("dropped_rows", 0))],
        ["时间戳乱序", "是" if prep.get("unsorted") else "否"],
        ["无常量列 / 无全空列", f"常量列 {len(constant_columns)} 个，全空列 {len(all_null_columns)} 个"],
    ]
    if gaps.get("checked"):
        quality_rows.append(["期望 / 实际时间点", f"{gaps['expected_points']:,} / {gaps['actual_points']:,}"])
        quality_rows.append(["缺口槽位", f"{gaps['missing_slots']:,} ({gaps['slot_rate']:.2%})"])
        if prep.get("filled_slots"):
            quality_rows.append(["重采样补齐的空槽", lib.fmt(prep["filled_slots"])])

    parts = [f"## {STEP}. 数据概览与质量检查\n"]
    parts.append("### 基本信息\n")
    parts.append(lib.md_table(["项目", "值"], overview_rows))
    parts.append("\n### 质量检查\n")
    parts.append(lib.md_table(["项目", "值"], quality_rows))

    runs = missing_runs(series)
    if runs:
        parts.append("\n### 最长连续缺失区间\n")
        parts.append(lib.md_table(["起点", "缺失长度(周期)"], [[r["start"], r["periods"]] for r in runs]))

    if gaps.get("checked") and gaps["largest_gaps"]:
        parts.append("\n### 最大缺口\n")
        parts.append(lib.md_table(
            ["起点", "终点", "缺失槽位"],
            [[g["start"], g["end"], g["periods"]] for g in gaps["largest_gaps"]],
        ))

    if describe:
        parts.append("\n### 描述统计\n")
        parts.append(lib.md_table(
            ["指标", f"`{target}`"],
            [
                ["count", lib.fmt(describe["count"])],
                ["mean", lib.fmt(describe["mean"])],
                ["std", lib.fmt(describe["std"])],
                ["min", lib.fmt(describe["min"])],
                ["p25", lib.fmt(describe["p25"])],
                ["median", lib.fmt(describe["median"])],
                ["p75", lib.fmt(describe["p75"])],
                ["max", lib.fmt(describe["max"])],
                ["skew", lib.fmt(describe["skew"])],
                ["kurtosis", lib.fmt(describe["kurtosis"])],
            ],
        ))

    if outliers.get("iqr") or outliers.get("robust_z"):
        parts.append("\n### 异常值粗筛（仅用于初步判断，正式检测见第 4 节）\n")
        parts.append(
            f"- IQR(1.5) 命中 {outliers['iqr']} 个，稳健 z(3.5) 命中 {outliers['robust_z']} 个，"
            f"两者同时命中 {outliers['both']} 个；合并占比 {outliers['rate']:.2%}。\n"
        )
        if outliers.get("top"):
            parts.append("\n")
            parts.append(lib.md_table(
                ["时间", "取值", "稳健 z"],
                [[row["timestamp"], lib.fmt(row["value"]), lib.fmt(row["robust_z"], 2)] for row in outliers["top"]],
            ))

    parts.append("\n### 图表\n")
    parts.append(lib.fig_md("01_series.png", "原始序列"))
    if n_obs >= window + 1:
        parts.append(lib.fig_md("01_rolling.png", "滚动均值与滚动标准差"))

    parts.append("\n### 结论要点\n")
    parts.append(
        f"- 本序列为 **{freq or '未知频率'}** 采样，共 {n_obs:,} 个有效观测，"
        f"覆盖 {resolved['start']} 至 {resolved['end']}。\n"
        f"- 质量评级 **{grade}**：缺失 {missing_pct:.2%}，重复时间戳 {duplicates} 个"
        + (f"，缺口比例 {gaps['slot_rate']:.2%}" if gaps.get("checked") else "")
        + "。\n"
        f"- 季节周期取 **{period}**（{'可用于分解与季节模型' if period > 1 else '不做季节性建模'}），"
        f"回测/预测步长 `horizon = {horizon}`。\n"
    )
    if warnings:
        parts.append("\n" + "\n".join(f"- ⚠️ {w.replace('WARN_ONLY: ', '')}" for w in warnings) + "\n")

    data = {
        "shape": [int(len(raw)), int(raw.shape[1])],
        "columns": [str(c) for c in raw.columns][:200],
        "time_col": prep["time_col"],
        "value_col": prep["value_col"],
        "dtype": value_dtype,
        "freq": freq,
        "period": period,
        "has_seasonality": has_seasonality,
        "n_slots": int(len(series)),
        "n_obs": n_obs,
        "n_missing": missing,
        "missing_pct": missing_pct,
        "duplicate_timestamps": duplicates,
        "unparsed_timestamps": prep.get("unparsed_timestamps", 0),
        "dropped_rows": prep.get("dropped_rows", 0),
        "unsorted": bool(prep.get("unsorted")),
        "resampled": bool(prep.get("resampled")),
        "gaps": gaps,
        "missing_runs": runs,
        "filled_slots": prep.get("filled_slots", 0),
        "constant_columns": constant_columns,
        "all_null_columns": all_null_columns,
        "describe": describe,
        "outliers_iqr": outliers.get("iqr", 0),
        "outliers_robust_z": outliers.get("robust_z", 0),
        "outlier_rate": outliers.get("rate", 0.0),
        "outlier_examples": outliers.get("top", []),
        "quality_grade": grade,
        "horizon": horizon,
        "time_range": [resolved["start"], resolved["end"]],
        "notes": [w.replace("WARN_ONLY: ", "") for w in warnings],
    }
    return data, "\n".join(parts)


def lib_missing_index_runs(series: pd.Series, limit: int = 200) -> list[tuple]:
    """连续 NaN 的 (起, 止) 区间，用于图上阴影."""
    is_na = series.isna()
    if not is_na.any():
        return []
    groups = is_na.ne(is_na.shift()).cumsum()
    spans = []
    for _, chunk in series.groupby(groups):
        if chunk.isna().all():
            spans.append((chunk.index[0], chunk.index[-1]))
    return spans[:limit]


if __name__ == "__main__":
    lib.run_step(STEP, NAME, step)
