#!/usr/bin/env python
"""Step 2 — 平稳性检验与分解.

用法:
    uv run python scripts/step2_stationarity.py

产物:
    out/step2_stationarity.json, out/step2_stationarity.md
    figures/02_stl.png, figures/02_acf_pacf.png
"""

from __future__ import annotations

import sys
import warnings as pywarnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsa_lib as lib  # noqa: E402

STEP, NAME = 2, "stationarity"
MIN_OBS = 12


def stationarity_tests(x: np.ndarray, label: str, warnings: list[str], with_trend: bool = False) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss

    out: dict = {"label": label, "n": int(len(x))}
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < MIN_OBS or np.allclose(x, x[0]):
        out["note"] = "样本过短或为常数序列，检验结果不可靠"
        return out

    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        try:
            stat, pvalue, usedlag, nobs, crit, _ = adfuller(x, autolag="AIC", regression="ct" if with_trend else "c")
            out["adf"] = {
                "statistic": float(stat),
                "pvalue": float(pvalue),
                "usedlag": int(usedlag),
                "nobs": int(nobs),
                "critical_values": {k: float(v) for k, v in crit.items()},
                "stationary": bool(pvalue < 0.05),
                "regression": "ct" if with_trend else "c",
            }
        except Exception as exc:  # noqa: BLE001
            out["adf"] = {"error": f"{type(exc).__name__}: {exc}"}
            warnings.append(f"{label} 的 ADF 检验失败: {exc}")
        try:
            stat, pvalue, lags, crit = kpss(x, regression="c", nlags="auto")
            out["kpss"] = {
                "statistic": float(stat),
                "pvalue": float(pvalue),
                "lags": int(lags),
                "critical_values": {k: float(v) for k, v in crit.items()},
                "stationary": bool(pvalue > 0.05),
            }
        except Exception as exc:  # noqa: BLE001
            out["kpss"] = {"error": f"{type(exc).__name__}: {exc}"}
            warnings.append(f"{label} 的 KPSS 检验失败: {exc}")
    return out


def decompose(series: pd.Series, period: int, warnings: list[str]) -> dict:
    """STL 分解 + 趋势/季节强度."""
    values = series.to_numpy(dtype=float)
    out: dict = {"period": period}
    if period > 1 and np.isfinite(values).sum() >= 2 * period:
        from statsmodels.tsa.seasonal import STL

        seasonal_window = period + 1 if period % 2 == 0 else period
        filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
        try:
            stl = STL(filled, period=period, seasonal=seasonal_window, robust=True)
            result = stl.fit()
            trend = np.asarray(result.trend)
            seasonal = np.asarray(result.seasonal)
            resid = np.asarray(result.resid)
            out["method"] = "STL(robust=True)"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"STL 分解失败，降级为滚动均值趋势: {exc}")
            out["method"] = "rolling_mean_fallback"
            trend = pd.Series(filled).rolling(period, center=True, min_periods=1).mean().to_numpy()
            seasonal = np.zeros_like(filled)
            resid = filled - trend
    else:
        if period > 1:
            warnings.append("观测点不足以做季节分解，仅提取趋势")
        out["method"] = "rolling_mean"
        filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
        window = period if period > 1 else 7
        trend = pd.Series(filled).rolling(window, center=True, min_periods=1).mean().to_numpy()
        seasonal = np.zeros_like(filled)
        resid = filled - trend

    def strength(component: np.ndarray, residual: np.ndarray) -> float:
        denominator = float(np.nanvar(component + residual))
        if denominator <= 0:
            return 0.0
        return float(max(0.0, 1.0 - float(np.nanvar(residual)) / denominator))

    out.update(
        {
            "trend_strength": strength(trend, resid),
            "seasonal_strength": strength(seasonal, resid) if period > 1 else 0.0,
            "resid_std": float(np.nanstd(resid)),
            "trend_slope": float(np.polyfit(np.arange(len(trend)), trend, 1)[0]) if len(trend) > 2 else None,
        }
    )
    out["_components"] = {"trend": trend, "seasonal": seasonal, "resid": resid}
    return out


def acf_pacf(values: np.ndarray, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.tsa.stattools import acf, pacf

    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        try:
            acf_values = acf(values, nlags=nlags, fft=False, missing="drop")
        except Exception:  # noqa: BLE001
            clean = values[np.isfinite(values)]
            acf_values = acf(clean, nlags=nlags, fft=False)
        try:
            pacf_values = pacf(values, nlags=nlags, method="ywm", missing="drop")
        except Exception:  # noqa: BLE001
            clean = values[np.isfinite(values)]
            pacf_values = pacf(clean, nlags=nlags, method="ywm")
    return np.asarray(acf_values), np.asarray(pacf_values)


def significant_lags(values: np.ndarray, conf: float, limit: int) -> list[int]:
    return [int(i) for i in range(1, min(len(values), limit + 1)) if abs(values[i]) > conf]


def suggest_orders(acf_values, pacf_values, conf: float, period: int) -> dict:
    p_lags = significant_lags(pacf_values, conf, 3)
    q_lags = significant_lags(acf_values, conf, 3)
    orders = {
        "p": max(p_lags) if p_lags else 0,
        "q": max(q_lags) if q_lags else 0,
        "m": period,
        "P": 0,
        "Q": 0,
        "pacf_significant": significant_lags(pacf_values, conf, 24),
        "acf_significant": significant_lags(acf_values, conf, 24),
    }
    if period > 1:
        if any(lag in orders["acf_significant"] for lag in (period, 2 * period)):
            orders["Q"] = 1
        if any(lag in orders["pacf_significant"] for lag in (period, 2 * period)):
            orders["P"] = 1
    return orders


def step(run_dir: Path, warnings: list[str]):
    settings = lib.load_settings(run_dir, lib.parse_argv(), prefer_resolved=True)
    frame, prep = lib.load_series(run_dir, settings)
    series = frame.iloc[:, 0]
    target = series.name

    period = int(settings.get("period") or lib.period_for_freq(prep.get("freq")) or 1)
    n_obs = int(series.notna().sum())
    if n_obs < MIN_OBS:
        warnings.append(f"有效观测仅 {n_obs} 个，平稳性检验不可靠")
    if n_obs < 8:
        raise SystemExit(lib.SKIP_STEP_EXIT_CODE)

    level = series.to_numpy(dtype=float)
    diff = np.diff(level)
    seasonal_diff = (
        level[period:] - level[:-period]
        if period > 1 and len(level) > 2 * period
        else np.array([])
    )

    tests = {
        "level": stationarity_tests(level, f"原序列({target})", warnings),
        "diff": stationarity_tests(diff, "一阶差分", warnings),
    }
    if seasonal_diff.size >= MIN_OBS:
        tests["seasonal_diff"] = stationarity_tests(seasonal_diff, f"季节差分({period})", warnings)
    if tests["level"].get("adf", {}).get("stationary") is False:
        tests["level_trend"] = stationarity_tests(level, "原序列(含趋势项)", warnings, with_trend=True)

    # ---- 差分次数判定 -----------------------------------------------------
    level_test = tests["level"]
    level_adf_ok = bool(level_test.get("adf", {}).get("stationary"))
    level_kpss_ok = bool(level_test.get("kpss", {}).get("stationary"))
    diff_adf_ok = bool(tests["diff"].get("adf", {}).get("stationary"))

    if level_adf_ok and level_kpss_ok:
        d = 0
        d_reason = "原序列 ADF 与 KPSS 一致判定为平稳"
    elif diff_adf_ok:
        d = 1
        d_reason = "原序列不平稳，一阶差分后 ADF 判定平稳"
    elif level_adf_ok:
        # ADF 与 KPSS 结论冲突：宁可保守取 d=0，避免过度差分
        d = 0
        d_reason = "ADF 判定原序列平稳但 KPSS 拒绝，结论冲突；按 ADF 取 d=0（宁可少差分）"
        warnings.append("ADF 与 KPSS 对原序列的结论冲突，差分次数建议置信度较低，已保守取 d=0")
    else:
        d = 2
        d_reason = "一阶差分后仍不平稳，建议 d=2（估计不稳定，需人工确认）"
        warnings.append("一阶差分后仍不平稳，差分次数建议缺乏统计支撑")

    # ---- 分解 -------------------------------------------------------------
    decomposition = decompose(series, period, warnings)
    components = decomposition.pop("_components")
    seasonal_strength = float(decomposition.get("seasonal_strength") or 0.0)

    seasonal_diff_ok = bool(tests.get("seasonal_diff", {}).get("adf", {}).get("stationary"))
    D = 1 if (period > 1 and seasonal_strength >= 0.3 and seasonal_diff_ok) else 0
    if period > 1 and D == 0 and seasonal_strength >= 0.3:
        warnings.append("季节强度较高但季节差分未改善平稳性，暂用 D=0")

    # ---- ACF / PACF -------------------------------------------------------
    # 与图保持一致：d > 0 时在差分序列上算 ACF/PACF。
    # （在强趋势的原序列上算 ACF 会得到虚假的长记忆显著峰，与图矛盾）
    nlags = int(min(max(2 * period, 20), max(len(level) // 3, 3), 60))
    plot_source = diff if d > 0 else level
    source_label = "一阶差分" if d > 0 else "原序列"
    acf_values, pacf_values = acf_pacf(plot_source, nlags)
    conf = float(1.96 / np.sqrt(max(n_obs, 2)))
    orders = suggest_orders(acf_values, pacf_values, conf, period)
    orders["acf_pacf_series"] = source_label
    orders["d"] = d
    orders["D"] = D
    orders["d_reason"] = d_reason
    lib.update_resolved(run_dir, {"suggested_orders": orders, "trend_strength": decomposition.get("trend_strength"), "seasonal_strength": seasonal_strength})

    # ---- 图表 -------------------------------------------------------------
    plt, cjk = lib.configure_matplotlib()
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    index = series.index
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(index, level, linewidth=0.9)
    axes[0].set_title(L("观测值", "observed"))
    axes[1].plot(index, components["trend"], linewidth=1.4, color="#2ca02c")
    axes[1].set_title(L("趋势", "trend"))
    axes[2].plot(index, components["seasonal"], linewidth=0.9, color="#ff7f0e")
    axes[2].set_title(L(f"季节项(period={period})", f"seasonal (period={period})"))
    axes[3].plot(index, components["resid"], linewidth=0.8, color="#d62728")
    axes[3].set_title(L("残差", "residual"))
    fig.suptitle(L(f"分解方法: {decomposition['method']}", f"decomposition: {decomposition['method']}"))
    fig.tight_layout()
    lib.savefig(fig, run_dir, "02_stl.png")

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plot_source_label = L("一阶差分", "1st diff") if d > 0 else L("原序列", "level")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        plot_acf(pd.Series(plot_source).dropna(), lags=nlags, ax=axes[0], title=f"ACF ({plot_source_label})")
        plot_pacf(pd.Series(plot_source).dropna(), lags=nlags, ax=axes[1], method="ywm", title=f"PACF ({plot_source_label})")
    fig.tight_layout()
    lib.savefig(fig, run_dir, "02_acf_pacf.png")

    # ---- markdown ---------------------------------------------------------
    def test_row(label: str, block: dict) -> list[str]:
        adf = block.get("adf") or {}
        kpss = block.get("kpss") or {}
        return [
            label,
            lib.fmt(block.get("n")),
            f"{lib.fmt(adf.get('statistic'), 3)} (p={lib.fmt(adf.get('pvalue'), 4)})",
            "平稳" if adf.get("stationary") else ("不平稳" if adf else "-"),
            f"{lib.fmt(kpss.get('statistic'), 3)} (p={lib.fmt(kpss.get('pvalue'), 4)})",
            "平稳" if kpss.get("stationary") else ("不平稳" if kpss else "-"),
        ]

    parts = [f"## {STEP}. 平稳性与分解\n"]
    parts.append("### 平稳性检验\n")
    parts.append("ADF：p < 0.05 判为平稳；KPSS：p > 0.05 判为平稳。两者结论冲突时需人工判断。\n")
    parts.append(lib.md_table(
        ["序列", "n", "ADF 统计量(p)", "ADF", "KPSS 统计量(p)", "KPSS"],
        [test_row(key, value) for key, value in tests.items()],
    ))

    parts.append("\n### 差分建议\n")
    parts.append(
        f"- `d = {d}`，`D = {D}`（`m = {period}`）。\n"
        f"- 依据：{d_reason}。\n"
    )
    if D:
        parts.append("- 季节差分有效：季节强度与差分后平稳性均支持。\n")
    elif period > 1:
        if seasonal_strength >= 0.3:
            parts.append(
                f"- 季节强度 {lib.fmt(seasonal_strength, 3)}（≥ 0.3，季节性显著），"
                "但季节差分未使 ADF 判定平稳，故暂不做季节差分（D=0），季节项交由 SARIMA/ETS 建模。\n"
            )
        else:
            parts.append(f"- 季节强度 {lib.fmt(seasonal_strength, 3)}（< 0.3，季节性弱），故不做季节差分。\n")

    parts.append("\n### 分解结果\n")
    parts.append(lib.md_table(
        ["项目", "值"],
        [
            ["方法", decomposition["method"]],
            ["周期", lib.fmt(period)],
            ["趋势强度", lib.fmt(decomposition["trend_strength"], 3)],
            ["季节强度", lib.fmt(decomposition["seasonal_strength"], 3)],
            ["残差标准差", lib.fmt(decomposition["resid_std"], 3)],
            ["趋势斜率(每步)", lib.fmt(decomposition["trend_slope"], 4)],
        ],
    ))
    parts.append(lib.fig_md("02_stl.png", "STL 分解"))

    parts.append("\n### ACF / PACF\n")
    parts.append(
        f"- 计算序列：`{orders['acf_pacf_series']}`（跟随差分建议，与图一致）。\n"
        f"- 95% 置信带宽度 ±{lib.fmt(conf, 4)}。\n"
        f"- ACF 显著滞后: {orders['acf_significant'] or '无'}\n"
        f"- PACF 显著滞后: {orders['pacf_significant'] or '无'}\n"
    )
    parts.append(lib.fig_md("02_acf_pacf.png", "ACF 与 PACF"))

    parts.append("\n### 建模阶数建议（供第 3 节搜索空间使用）\n")
    parts.append(lib.md_table(
        ["参数", "建议值", "含义"],
        [
            ["p", orders["p"], "非季节 AR 阶"],
            ["d", orders["d"], f"非季节差分 ({d_reason})"],
            ["q", orders["q"], "非季节 MA 阶"],
            ["P", orders["P"], "季节 AR 阶"],
            ["D", orders["D"], "季节差分"],
            ["Q", orders["Q"], "季节 MA 阶"],
            ["m", orders["m"], "季节周期"],
        ],
    ))
    parts.append(
        "\n> 这些只是搜索空间的上界，最终阶数由第 3 节的 AIC/回测决定，不是结论本身。\n"
    )
    if warnings:
        parts.append("\n" + "\n".join(f"- ⚠️ {w.replace('WARN_ONLY: ', '')}" for w in warnings) + "\n")

    data = {
        "tests": tests,
        "decomposition": decomposition,
        "seasonal_strength": seasonal_strength,
        "conf": conf,
        "acf": {"lags": int(nlags), "values": acf_values.tolist()[: nlags + 1], "significant": orders["acf_significant"]},
        "pacf": {"lags": int(nlags), "values": pacf_values.tolist()[: nlags + 1], "significant": orders["pacf_significant"]},
        "suggested_orders": orders,
        "differencing": {"d": d, "D": D, "m": period, "reason": d_reason},
    }
    return data, "\n".join(parts)


if __name__ == "__main__":
    lib.run_step(STEP, NAME, step)
