#!/usr/bin/env python
"""Step 4 — 异常与变点检测.

用法:
    uv run python scripts/step4_anomaly.py [--max-anomalies 20]

产物:
    out/step4_anomaly.json, out/step4_anomaly.md
    figures/04_anomalies.png, figures/04_changepoints.png

三类互补证据:
    A. STL 残差的稳健 z-score         -> 单点异常
    B. 滚动中位数/MAD 偏离            -> 水平漂移时间段
    C. 二分分割 + CUSUM（可选 ruptures）-> 结构性变点
"""

from __future__ import annotations

import math
import sys
import warnings as pywarnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsa_lib as lib  # noqa: E402

STEP, NAME = 4, "anomaly"


def robust_scale(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 1.0
    mad = float(np.median(np.abs(clean - np.median(clean))))
    if mad > 0:
        return 1.4826 * mad
    std = float(np.std(clean))
    return std if std > 0 else 1.0


def stl_residuals(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, str]:
    """返回 (残差, 去季节化序列 = 趋势+残差, 方法名)."""
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    if period > 1 and np.isfinite(filled).sum() >= 2 * period:
        from statsmodels.tsa.seasonal import STL

        seasonal_window = period + 1 if period % 2 == 0 else period
        try:
            with pywarnings.catch_warnings():
                pywarnings.simplefilter("ignore")
                result = STL(filled, period=period, seasonal=seasonal_window, robust=True).fit()
            residual = np.asarray(result.resid, dtype=float)
            deseasonalized = filled - np.asarray(result.seasonal, dtype=float)
            return residual, deseasonalized, "STL_resid"
        except Exception:  # noqa: BLE001
            pass
    window = period if period > 1 else 7
    trend = pd.Series(filled).rolling(window, center=True, min_periods=1).median().to_numpy()
    return filled - trend, trend, "detrended"


def cluster_anomalies(
    series: pd.Series,
    z: pd.Series,
    scale: float,
    threshold: float = 3.5,
    min_days: int = 2,
    filled: pd.Series | None = None,
) -> list[dict]:
    """连续多日异常窗口（异常聚集）。

    为什么不用“滚动中位数偏离 > k×滚动 MAD”那种做法：局部 MAD 会在平滑区间变得极小，
    于是普通噪声抱团就能触发（实测在干净数据上凭空报出 7 个 2 天假窗口），
    而对真正注入的 3 天 +200 偏离反而漏报——它不是良定义的。
    永久水平偏移也不属于这里（那是变点检测的职责）。

    这里直接用与点异常同一个残差稳健 z（全局尺度），把连续 ≥ min_days 天的
    |z| > threshold 合并成一个窗口，语义明确、可验证。

    series 用于统计窗口内有多少天原本是缺失值（levels 取自 filled）：
    缺失段被线性插值后往往会在残差上聚成一个“异常”，必须告诉读者这是插值产物而不是真实事件。
    """
    levels = filled if filled is not None else series
    flag = (z.abs() > threshold).fillna(False)
    if not bool(flag.any()):
        return []

    episodes: list[dict] = []
    groups = flag.ne(flag.shift()).cumsum()
    for _, chunk in series.groupby(groups):
        if chunk.empty or not bool(flag.loc[chunk.index[0]]):
            continue
        if len(chunk) < min_days:
            continue
        chunk_z = z.loc[chunk.index]
        chunk_levels = levels.loc[chunk.index]
        mean_z = float(chunk_z.mean())
        level = float(chunk_levels.mean())
        reference = float(level - mean_z * scale)
        missing_days = int(chunk.isna().sum())
        note = (
            f"窗口内有 {missing_days}/{len(chunk)} 天原本缺失（已线性插值），该聚集更可能是插值产物而非真实事件"
            if missing_days
            else ""
        )
        episodes.append(
            {
                "start": str(chunk.index[0]),
                "end": str(chunk.index[-1]),
                "periods": int(len(chunk)),
                "level": level,
                "reference": reference,
                "level_shift": level - reference,
                "level_shift_pct": float((level - reference) / reference) if reference not in (0, None) else None,
                "max_abs_z": float(chunk_z.abs().max()),
                "direction": "high" if mean_z >= 0 else "low",
                "missing_days_inside": missing_days,
                "method": f"consecutive_residual_z(threshold={threshold}, min_days={min_days})",
                "kind": "anomaly_cluster",
                "note": note,
            }
        )
    episodes.sort(key=lambda item: item["max_abs_z"], reverse=True)
    return episodes


def remove_slow_cycle(values: np.ndarray, index: pd.DatetimeIndex) -> tuple[np.ndarray, str]:
    """用「线性趋势 + 年度傅里叶项」拟合并减去慢周期分量。

    周季节性已由 STL 去掉，但日均序列往往还有年度周期（振幅常大于真实的水平偏移），
    不去掉它，二分分割会把年周期当成变点。跨度不足 1.5 年时不做处理。
    """
    values = np.asarray(values, dtype=float)
    try:
        elapsed_days = (index - index[0]).total_seconds() / 86400.0
    except Exception:  # noqa: BLE001
        return values, "none"
    if len(values) < 60 or float(elapsed_days[-1]) < 540:
        return values, "none"
    finite = np.isfinite(values)
    if finite.sum() < 60:
        return values, "none"

    t = np.asarray(elapsed_days, dtype=float)
    columns = [np.ones_like(t)]
    # 只拟合年度谐波，**不**放线性趋势项：
    # 全局斜率会与真实的水平偏移相互污染（偏移把斜率拉陡，残差里就剩下一个假斜率，
    # 分段分割会把这个假斜率切成“一年一个假变点”）。
    # 残留的趋势交给下游的分段线性分割处理，它不会把平滑趋势当成断裂。
    for harmonic in (1, 2, 3):
        columns.append(np.sin(2 * np.pi * harmonic * t / 365.25))
        columns.append(np.cos(2 * np.pi * harmonic * t / 365.25))
    design = np.column_stack(columns)

    y = np.where(finite, values, float(np.nanmedian(values)))
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return values, "none"
    # 不要用 IRLS / 稳健加权：它会把水平偏移当成离群点降权，
    # 反过来扭曲谐波系数，使残差里的偏移变得难以识别（实测更差）。
    return y - design @ coef, "lstsq(intercept + annual_fourier_1-3)"


def _sse(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sum((values - values.mean()) ** 2))


def _fit_line(values: np.ndarray, start: int, end: int) -> tuple[float, float]:
    """区间内的最小二乘直线（x 以区间起点为 0）: 返回 (slope, intercept)."""
    segment = values[start:end]
    size = segment.size
    if size == 0:
        return 0.0, 0.0
    if size < 3:
        return 0.0, float(segment.mean())
    x = np.arange(size, dtype=float)
    x_centered = x - x.mean()
    y_centered = segment - segment.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 1e-12:
        return 0.0, float(segment.mean())
    slope = float(np.dot(x_centered, y_centered)) / denominator
    intercept = float(segment.mean() - slope * x.mean())
    return slope, intercept


def _sse_between(values: np.ndarray, start: int, end: int, linear: bool = True) -> float:
    """区段的 SSE。

    linear=True 时先用区间内的最小二乘直线拟合再算残差平方和。
    这样平滑趋势的“切分增益”≈ 0，而真正的水平偏移仍然有巨大增益，
    避免了在有趋势的残差上一刀一刀切出假变点。
    """
    segment = values[start:end]
    size = segment.size
    if size == 0:
        return 0.0
    if not linear or size < 3:
        return _sse(segment)
    x = np.arange(size, dtype=float)
    x_centered = x - x.mean()
    y_centered = segment - segment.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 1e-12:
        return _sse(segment)
    slope = float(np.dot(x_centered, y_centered)) / denominator
    residual = y_centered - slope * x_centered
    return float(np.dot(residual, residual))


def binary_segmentation(
    values: np.ndarray,
    max_changepoints: int = 5,
    min_segment: int = 30,
    return_candidates: bool = False,
):
    n = len(values)
    if n < 2 * min_segment:
        return []
    variance = float(np.var(values)) or 1.0
    penalty = 3.0 * math.log(n) * variance
    stride = max(1, n // 1000)
    segments = [(0, n)]
    found: list[dict] = []
    first_round_candidates: list[dict] = []

    for round_id in range(max_changepoints):
        best = None
        candidates: list[dict] = []
        for left, right in segments:
            if right - left < 2 * min_segment:
                continue
            base = _sse_between(values, left, right)
            for split in range(left + min_segment, right - min_segment + 1, stride):
                gain = base - (_sse_between(values, left, split) + _sse_between(values, split, right))
                candidates.append({"index": int(split), "gain": float(gain)})
                if best is None or gain > best[0]:
                    best = (gain, split, (left, right))
        if round_id == 0 and return_candidates:
            candidates.sort(key=lambda item: item["gain"], reverse=True)
            seen: list[int] = []
            for candidate in candidates:
                if len(seen) >= 8:
                    break
                if any(abs(candidate["index"] - other) < min_segment for other in seen):
                    continue
                seen.append(candidate["index"])
                first_round_candidates.append({**candidate, "above_penalty": bool(candidate["gain"] > penalty)})
        if best is None or best[0] <= penalty:
            break
        gain, split, segment = best
        before_std = math.sqrt(max(_sse_between(values, segment[0], split) / max(split - segment[0] - 2, 1), 0.0))
        after_std = math.sqrt(max(_sse_between(values, split, segment[1]) / max(segment[1] - split - 2, 1), 0.0))
        # 用「分断点处两条直线的跳跃量」作为水平偏移，而不是段均值之差
        # （段均值差会把区间内的趋势算进去，在增长型序列上系统性高估偏移）
        slope_left, intercept_left = _fit_line(values, segment[0], split)
        _, intercept_right = _fit_line(values, split, segment[1])
        level_before = slope_left * (split - segment[0]) + intercept_left
        level_after = intercept_right
        found.append(
            {
                "index": int(split),
                "type": "mean",
                "delta": float(level_after - level_before),
                "level_before": float(level_before),
                "level_after": float(level_after),
                "mean_before": float(values[segment[0] : split].mean()),
                "mean_after": float(values[split : segment[1]].mean()),
                "variance_ratio": float((after_std or 1e-12) / (before_std or 1e-12)) if before_std else None,
                "gain": float(gain),
                "penalty": float(penalty),
                "method": "binary_segmentation(piecewise_linear)",
            }
        )
        segments.remove(segment)
        segments.extend([(segment[0], split), (split, segment[1])])
    found.sort(key=lambda item: item["index"])
    return (found, first_round_candidates) if return_candidates else found


def cusum_changepoints(values: np.ndarray, threshold: float = 4.0) -> list[int]:
    n = len(values)
    if n < 20:
        return []
    # 必须先去掉线性趋势：直接对有趋势的序列做 CUSUM，累计和几乎必然在中点越界
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    residual = values - (slope * x + intercept)
    std = float(np.std(residual)) or 1.0
    standardized = residual / std
    cumulative = np.cumsum(standardized)
    drift = np.arange(1, n + 1) / n * cumulative[-1]
    centered = cumulative - drift
    scale = float(np.std(centered)) or 1.0
    magnitude = np.abs(centered) / scale
    points: list[int] = []
    inside = magnitude > threshold
    if not inside.any():
        return []
    groups = pd.Series(inside).ne(pd.Series(inside).shift()).cumsum()
    for _, chunk in pd.Series(magnitude).groupby(groups):
        if bool(inside.iloc[chunk.index[0]]):
            points.append(int(chunk.idxmax()))
    return sorted(set(points))


def ruptures_changepoints(values: np.ndarray) -> tuple[list[int], str]:
    try:
        import ruptures as rpt  # type: ignore
    except ImportError:
        return [], "ruptures 未安装，仅使用内置方法"
    try:
        penalty = 3.0 * math.log(len(values)) * (float(np.var(values)) or 1.0)
        algo = rpt.Pelt(model="rbf", min_size=max(5, len(values) // 50)).fit(values)
        breaks = algo.predict(pen=penalty)
        return [int(b) for b in breaks[:-1]], "ruptures.PELT(rbf)"
    except Exception as exc:  # noqa: BLE001
        return [], f"ruptures 运行失败: {exc}"


def step(run_dir: Path, warnings: list[str]):
    settings = lib.load_settings(run_dir, lib.parse_argv(), prefer_resolved=True)
    frame, prep = lib.load_series(run_dir, settings)
    series = frame.iloc[:, 0]
    target = series.name

    period = int(settings.get("period") or 1)
    max_anomalies = int(settings.get("max_anomalies") or 20)
    values = series.to_numpy(dtype=float)

    if int(np.isfinite(values).sum()) < 20:
        warnings.append("有效观测少于 20，异常检测不可靠")
        raise SystemExit(lib.SKIP_STEP_EXIT_CODE)

    # ---- A. 残差异常 ------------------------------------------------------
    residual, deseasonalized, method = stl_residuals(values, period)
    residual = np.asarray(residual, dtype=float)
    median = float(np.nanmedian(residual))
    scale = robust_scale(residual)
    z = (residual - median) / scale

    clean = series.dropna()
    clean_index = clean.index
    clean_z = pd.Series(np.asarray(z)[: len(values)], index=series.index).reindex(clean_index)

    iqr = 0.0
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1
    iqr_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
    z_mask = clean_z.abs() > 3.5
    combined = z_mask | iqr_mask

    anomalies = []
    for timestamp in clean.index[combined.fillna(False)]:
        score = float(clean_z.get(timestamp, 0.0) or 0.0)
        both = bool(z_mask.get(timestamp, False)) and bool(iqr_mask.get(timestamp, False))
        anomalies.append(
            {
                "timestamp": str(timestamp),
                "value": float(clean.loc[timestamp]),
                "score": score,
                "direction": "high" if score >= 0 else "low",
                "severity": "high" if both else "medium",
                "method": f"{method}+robust_z",
                "iqr_hit": bool(iqr_mask.get(timestamp, False)),
            }
        )
    anomalies.sort(key=lambda item: abs(item["score"]), reverse=True)

    total_candidates = len(anomalies)
    anomaly_rate = float(total_candidates / max(len(clean), 1))
    if anomaly_rate > 0.10:
        warnings.append(
            f"异常率 {anomaly_rate:.1%} 偏高，更可能是整体波动性上升或模型设定不当，而不是离散异常事件"
        )
    anomalies = anomalies[:max_anomalies]

    # ---- B. 多日异常聚集 --------------------------------------------------
    # 用与点异常同一个残差 z（全局稳健尺度），把连续 ≥2 天异常合并成窗口
    residual_z = pd.Series(
        np.asarray(z)[: len(series)], index=series.index, dtype=float
    ).reindex(series.index)
    filled_series = pd.Series(values, index=series.index, name=series.name).interpolate(limit_direction="both")
    episodes = cluster_anomalies(series, residual_z, scale, threshold=3.5, min_days=2, filled=filled_series)

    # ---- C. 变点 ----------------------------------------------------------
    # 在「去周季节 + 去年周期」的序列上检测水平偏移，否则季节峰会被误判为变点
    change_base = np.nan_to_num(deseasonalized, nan=median)
    change_base, cycle_note = remove_slow_cycle(change_base, series.index)
    # 靠近两端的切分点不可靠（STL 边界效应），要求每段至少有 min_segment 个点
    min_segment = int(max(6 * period, 45))
    internal, top_candidates = binary_segmentation(change_base, min_segment=min_segment, return_candidates=True)
    cusum_points = cusum_changepoints(change_base)
    rupture_points, rupture_note = ruptures_changepoints(change_base)
    for candidate in top_candidates:
        position = min(candidate["index"], len(series) - 1)
        candidate["timestamp"] = str(series.index[position])
    if rupture_points:
        warnings.append(f"ruptures 交叉验证命中 {len(rupture_points)} 个变点，见 JSON")

    for point in internal:
        index = min(point["index"], len(series) - 1)
        point["timestamp"] = str(series.index[index])
        supporting = [name for name, pts in (("cusum", cusum_points),) if any(abs(p - point["index"]) <= 2 for p in pts)]
        if any(abs(p - point["index"]) <= 2 for p in rupture_points):
            supporting.append("ruptures")
        point["corroborated_by"] = supporting
        point["confidence"] = "high" if supporting else "medium"

    cusum_extra = []
    for point in cusum_points:
        if not any(abs(point - item["index"]) <= 2 for item in internal):
            index = min(point, len(series) - 1)
            cusum_extra.append(
                {
                    "index": int(point),
                    "timestamp": str(series.index[index]),
                    "type": "mean",
                    "delta": None,
                    "variance_ratio": None,
                    "gain": None,
                    "method": "cusum",
                    "corroborated_by": [],
                    "confidence": "low",
                }
            )
    changepoints = sorted(internal + cusum_extra, key=lambda item: item["index"])

    # ---- 图表 -------------------------------------------------------------
    plt, cjk = lib.configure_matplotlib()
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    fig, ax = plt.subplots()
    ax.plot(series.index, values, linewidth=0.9, color="#1f77b4", label=L("观测值", "observed"))
    high = [item for item in anomalies if item["direction"] == "high"]
    low = [item for item in anomalies if item["direction"] == "low"]
    if high:
        ax.scatter(pd.to_datetime([i["timestamp"] for i in high]), [i["value"] for i in high],
                   color="#d62728", s=28, zorder=3, label=L("偏高异常", "high anomaly"))
    if low:
        ax.scatter(pd.to_datetime([i["timestamp"] for i in low]), [i["value"] for i in low],
                   color="#9467bd", s=28, zorder=3, marker="v", label=L("偏低异常", "low anomaly"))
    for episode in episodes[:5]:
        ax.axvspan(pd.to_datetime(episode["start"]), pd.to_datetime(episode["end"]), color="#ff7f0e", alpha=0.15, linewidth=0)
    ax.set_title(L(f"异常点（残差稳健 z>3.5 或 IQR，检测基准 {method}）", f"Anomalies (residual robust |z|>3.5 or IQR, base={method})"))
    ax.legend(loc="best")
    lib.savefig(fig, run_dir, "04_anomalies.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(series.index, values, linewidth=0.9)
    for point in changepoints:
        ax1.axvline(pd.to_datetime(point["timestamp"]), color="#d62728", linestyle="--", linewidth=1.1)
    ax1.set_title(L(f"变点位置（{len(changepoints)} 个）", f"Change points ({len(changepoints)})"))
    ax2.plot(series.index, residual, linewidth=0.8, color="#7f7f7f")
    ax2.axhline(3.5 * scale, color="#d62728", linestyle=":", linewidth=1)
    ax2.axhline(-3.5 * scale, color="#d62728", linestyle=":", linewidth=1)
    ax2.set_title(L("残差与异常阈值", "Residual and anomaly thresholds"))
    fig.tight_layout()
    lib.savefig(fig, run_dir, "04_changepoints.png")

    # ---- markdown ---------------------------------------------------------
    parts = [f"## {STEP}. 异常与变点检测\n"]
    parts.append("### 检测配置\n")
    parts.append(lib.md_table(
        ["项目", "值"],
        [
            ["残差基准", f"`{method}`（周期 {period}）"],
            ["稳健尺度", lib.fmt(scale)],
            ["点异常判据", "残差稳健 |z| > 3.5 或 IQR(1.5)"],
            ["多日异常聚集", "连续 ≥ 2 天的残差稳健 |z| > 3.5"],
            ["永久水平偏移", "去周季节 + 年度谐波后做分段线性二分分割 + 去趋势 CUSUM"],
            ["变点方法", f"去周季节 + 年度谐波（{cycle_note}）后做分段线性二分分割（BIC 风格惩罚）+ 去趋势 CUSUM；{rupture_note}"],        ],
    ))

    parts.append("\n### 点异常\n")
    parts.append(
        f"共命中 **{total_candidates}** 个候选（占比 {anomaly_rate:.2%}），下表按偏离程度列出前 {len(anomalies)} 个。\n\n"
    )
    if anomalies:
        parts.append(lib.md_table(
            ["时间", "取值", "稳健 z", "方向", "严重度", "方法"],
            [[a["timestamp"], lib.fmt(a["value"]), lib.fmt(a["score"], 2),
              "偏高" if a["direction"] == "high" else "偏低",
              "高" if a["severity"] == "high" else "中", a["method"]] for a in anomalies],
        ))
    else:
        parts.append("未检测到显著点异常。\n")
    parts.append(lib.fig_md("04_anomalies.png", "异常点与漂移区间"))

    parts.append("\n### 多日异常聚集\n")
    parts.append(
        "连续 ≥ 2 天残差超阈值的窗口（多日促销 / 连续几天的系统异常）。"
        "**永久性水平偏移在同一列里不会出现，请看结构性变点**。\n\n"
    )
    if episodes:
        parts.append(lib.md_table(
            ["起点", "终点", "持续天数", "最大 |z|", "区间均值", "参考水平", "偏移", "窗口内缺失天数"],
            [[e["start"], e["end"], e["periods"], lib.fmt(e["max_abs_z"], 2),
              lib.fmt(e["level"]), lib.fmt(e["reference"]), lib.fmt(e["level_shift"]),
              e["missing_days_inside"]] for e in episodes],
        ))
        flagged = [e for e in episodes if e["missing_days_inside"]]
        if flagged:
            parts.append("\n" + "\n".join(f"- ⚠️ {e['note']}" for e in flagged) + "\n")
    else:
        parts.append("未检测到连续多日的异常聚集。\n")

    parts.append("\n### 结构性变点\n")
    if changepoints:
        parts.append(lib.md_table(
            ["位置", "时间", "类型", "均值变化", "方差比", "方法", "交叉验证", "置信度"],
            [[c["index"], c["timestamp"], c["type"], lib.fmt(c["delta"]), lib.fmt(c["variance_ratio"], 2),
              c["method"], ", ".join(c["corroborated_by"]) or "-",
              {"high": "高", "medium": "中", "low": "低"}[c["confidence"]]] for c in changepoints],
        ))
        parts.append(
            "\n> 变点数量受惩罚项影响很大，这里使用 `penalty = 3·ln(n)·Var(y)`；"
            "换惩罚会得到不同数量，因此结论应以「排序+交叉验证」为准，而不是绝对条数。\n"
        )
        if top_candidates:
            parts.append("\n首轮切分候选（含未达惩罚阈值者，便于人工判断）：\n\n")
            parts.append(lib.md_table(
                ["位置", "时间", "SSE 下降", "超过惩罚阈值"],
                [[c["index"], c["timestamp"], lib.fmt(c["gain"]), "是" if c["above_penalty"] else "否"]
                 for c in top_candidates],
            ))
        parts.append(
            "\n> ⚠️ 年度谐波拟合**可能与持续时间接近一整年的水平偏移相互吸收**，"
            "因此变点数量偏少时应回头对照 `04_anomalies.png` 与原始序列图人工确认；"
            "变点列表中的 `置信度=中` 表示没有第二种方法交叉验证。\n"
        )
    else:
        parts.append("未检测到满足惩罚阈值的变点。\n")
    parts.append(lib.fig_md("04_changepoints.png", "变点与残差"))

    if warnings:
        parts.append("\n" + "\n".join(f"- ⚠️ {w.replace('WARN_ONLY: ', '')}" for w in warnings) + "\n")

    data = {
        "config": {
            "base_method": method,
            "period": period,
            "robust_scale": scale,
            "residual_median": median,
            "max_anomalies": max_anomalies,
            "cycle_removal": cycle_note,
            "ruptures": rupture_note,
        },
        "anomalies": anomalies,
        "anomaly_candidates": total_candidates,
        "anomaly_rate": anomaly_rate,
        "episodes": episodes,
        "changepoints": changepoints,
        "changepoint_candidates": top_candidates,
        "min_segment": min_segment,
        "cusum_points": cusum_points,
        "ruptures_points": rupture_points,
        "summary": {
            "n_anomalies": total_candidates,
            "n_anomalies_reported": len(anomalies),
            "anomaly_rate": anomaly_rate,
            "n_episodes": len(episodes),
            "n_changepoints": len(changepoints),
            "dominant_method": method,
            "n_high_severity": sum(1 for a in anomalies if a["severity"] == "high"),
        },
    }
    return data, "\n".join(parts)


if __name__ == "__main__":
    lib.run_step(STEP, NAME, step)
