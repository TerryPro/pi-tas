#!/usr/bin/env python
"""Step 3 — 建模与滚动回测.

用法:
    uv run python scripts/step3_modeling.py

产物:
    out/step3_modeling.json, out/step3_modeling.md
    figures/03_backtest.png, figures/03_forecast.png
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

STEP, NAME = 3, "modeling"
Z80 = 1.2815515655446004  # 80% 区间
COMPLEXITY = {"naive": 0, "mean": 0, "drift": 1, "seasonal_naive": 1, "ets": 3, "arima": 3, "sarima": 4, "ridge": 5, "random_forest": 6}


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        return {"n": 0, "MAE": None, "RMSE": None, "MAPE": None, "sMAPE": None}
    error = actual - predicted
    denominator = np.abs(actual)
    mape = float(np.mean(np.abs(error) / denominator) * 100) if np.all(denominator > 1e-9) else None
    scale = (np.abs(actual) + np.abs(predicted)) / 2
    smape = float(np.mean(np.where(scale > 1e-9, np.abs(error) / np.maximum(scale, 1e-9), 0.0)) * 100)
    return {
        "n": int(actual.size),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": mape,
        "sMAPE": smape,
        "bias": float(np.mean(error)),
    }


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


def baseline_forecast(kind: str, train: np.ndarray, horizon: int, period: int) -> np.ndarray:
    train = np.asarray(train, dtype=float)
    if kind == "naive":
        return np.full(horizon, train[-1], dtype=float)
    if kind == "seasonal_naive":
        if period > 1 and train.size >= period:
            return np.array([train[train.size - period + (i % period)] for i in range(horizon)], dtype=float)
        return np.full(horizon, train[-1], dtype=float)
    if kind == "mean":
        return np.full(horizon, float(np.mean(train)), dtype=float)
    if kind == "drift":
        slope = (train[-1] - train[0]) / max(train.size - 1, 1)
        return train[-1] + slope * np.arange(1, horizon + 1)
    raise KeyError(kind)


def arima_trend(integrated: int) -> str | None:
    """statsmodels 不接受 trend 阶数低于 d+D 的设定（会被差分消掉）。

    d+D == 0 -> "c"（常数）
    d+D == 1 -> "t"（线性趋势，等价于在差分序列上放常数 = 漂移项）
    d+D >= 2 -> None
    """
    if integrated <= 0:
        return "c"
    if integrated == 1:
        return "t"
    return None


def select_arima_order(train: np.ndarray, suggested: dict, warnings: list[str], max_models: int = 40) -> tuple[tuple, dict]:
    from statsmodels.tsa.arima.model import ARIMA

    d0 = int(suggested.get("d", 1) or 0)
    p_max = min(3, int(suggested.get("p", 0) or 0) + 1)
    q_max = min(3, int(suggested.get("q", 0) or 0) + 1)
    d_values = sorted({max(0, d0 - 1), d0, d0 + 1})
    grid = [(p, dv, q) for p in range(p_max + 1) for dv in d_values for q in range(q_max + 1)]
    if len(grid) > max_models:
        step = math.ceil(len(grid) / max_models)
        grid = grid[::step]

    best_order, best_aic, results = (d0, d0, 0), np.inf, []
    failures: list[str] = []
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        for order in grid:
            try:
                model = ARIMA(np.asarray(train, dtype=float), order=order, trend=arima_trend(int(order[1])))
                fitted = model.fit()
                aic = float(fitted.aic)
                results.append({"order": list(order), "aic": aic})
                if np.isfinite(aic) and aic < best_aic:
                    best_aic, best_order = aic, order
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{order}: {type(exc).__name__}")
    if failures:
        warnings.append(
            f"ARIMA 网格中有 {len(failures)}/{len(grid)} 个组合拟合失败，已跳过（例如 {', '.join(failures[:3])}）"
        )
    if not np.isfinite(best_aic):
        warnings.append("ARIMA 所有候选阶数都拟合失败，本步骤将不包含 ARIMA 模型")
    return best_order, {"grid_size": len(grid), "failures": len(failures), "best_aic": None if not np.isfinite(best_aic) else best_aic, "candidates": sorted(results, key=lambda item: item["aic"])[:10]}


def fit_arima(train: np.ndarray, horizon: int, order: tuple, seasonal: bool, period: int, seasonal_order: tuple | None):
    from statsmodels.tsa.arima.model import ARIMA

    applied_seasonal = (seasonal_order or (0, 0, 0, 0)) if (seasonal and period > 1) else (0, 0, 0, 0)
    integrated = int(order[1]) + int(applied_seasonal[1])
    kwargs = {"order": order, "trend": arima_trend(integrated)}
    if seasonal and period > 1:
        kwargs["seasonal_order"] = applied_seasonal
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        fitted = ARIMA(np.asarray(train, dtype=float), **kwargs).fit()
        forecast = fitted.get_forecast(steps=horizon)
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        try:
            interval = np.asarray(forecast.conf_int(alpha=0.2), dtype=float)
            lower, upper = interval[:, 0], interval[:, 1]
        except Exception:  # noqa: BLE001
            lower = upper = None
    return mean, lower, upper, {}


def fit_ets(train: np.ndarray, horizon: int, period: int, warnings: list[str]):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    train = np.asarray(train, dtype=float)
    attempts = []
    if period > 1 and train.size >= 2 * period:
        attempts.append({"trend": "add", "seasonal": "add", "seasonal_periods": period})
        attempts.append({"trend": "add", "seasonal": None})
    else:
        attempts.append({"trend": "add", "seasonal": None})
    attempts.append({"trend": None, "seasonal": None})

    last_error = None
    for config in attempts:
        try:
            with pywarnings.catch_warnings():
                pywarnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    train,
                    trend=config["trend"],
                    seasonal=config["seasonal"],
                    seasonal_periods=config.get("seasonal_periods"),
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True)
                mean = np.asarray(fitted.forecast(horizon), dtype=float)
            return mean, None, None, config
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    warnings.append(f"ETS 拟合失败: {last_error}")
    return None, None, None, None


def build_supervised(y: np.ndarray, lags: list[int], window: int, index: pd.DatetimeIndex | None):
    """构造监督学习特征矩阵（严格只用历史信息）."""
    frame = pd.DataFrame({"y": y})
    for lag in lags:
        frame[f"lag{lag}"] = frame["y"].shift(lag)
    shifted = frame["y"].shift(1)
    frame["roll_mean"] = shifted.rolling(window, min_periods=1).mean()
    frame["roll_std"] = shifted.rolling(window, min_periods=2).std()
    if index is not None and isinstance(index, pd.DatetimeIndex):
        frame["dow"] = index.dayofweek
        frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
        frame["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
        frame["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)
        frame["month_sin"] = np.sin(2 * np.pi * index.month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * index.month / 12)
    features = [c for c in frame.columns if c != "y"]
    return frame, features


def ml_forecast(kind: str, train: np.ndarray, horizon: int, period: int, index: pd.DatetimeIndex | None, freq: str | None):
    """递归多步预测."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    max_lag = int(min(max(2 * period, 7), 14))
    lags = list(range(1, max_lag + 1))
    window = max(period, 3)
    if index is not None and len(index) != len(train):
        index = index[-len(train) :]
    frame, features = build_supervised(np.asarray(train, dtype=float), lags, window, index)
    usable = frame.dropna()
    if len(usable) < 30:
        return None, None, None, None
    X = usable[features].to_numpy(dtype=float)
    y = usable["y"].to_numpy(dtype=float)

    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    model.fit(X, y)

    history = list(np.asarray(train, dtype=float))
    future_index = None
    if freq:
        try:
            future_index = pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]
        except Exception:  # noqa: BLE001
            future_index = None
    predictions = []
    for step in range(horizon):
        # 追加一个占位值，让 build_supervised 为目标时点生成最后一行特征
        extended_y = np.asarray([*history, history[-1]], dtype=float)
        extended_index = None
        if index is not None and future_index is not None:
            extended_index = pd.DatetimeIndex([*index, *future_index[: step + 1]])
        frame_step, _ = build_supervised(extended_y, lags, window, extended_index)
        row = frame_step.iloc[[-1]][features].to_numpy(dtype=float)
        if not np.isfinite(row).all():
            row = np.nan_to_num(row, nan=float(np.mean(history[-window:])))
        prediction = float(model.predict(row)[0])
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions), None, None, {"lags": lags, "window": window, "n_features": len(features)}


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------


def plan_folds(n: int, horizon: int, period: int) -> list[tuple[int, int, int, int]]:
    min_train = max(2 * period if period > 1 else 10, 20)
    max_folds = (n - min_train) // max(horizon, 1)
    if max_folds < 1:
        return []
    desired = max(3, n // (5 * max(horizon, 1)))
    folds = int(min(max(desired, 1), 10, max_folds))
    blocks = []
    for k in range(folds, 0, -1):
        train_end = n - k * horizon
        blocks.append((0, train_end, train_end, min(train_end + horizon, n)))
    return blocks


def train_model(kind: str, train: np.ndarray, horizon: int, context: dict):
    period = context["period"]
    if kind in ("naive", "seasonal_naive", "mean", "drift"):
        return baseline_forecast(kind, train, horizon, period), None, None, {}
    if kind == "arima":
        return fit_arima(train, horizon, context["arima_order"], False, period, None)
    if kind == "sarima":
        return fit_arima(train, horizon, context["arima_order"], True, period, context["seasonal_order"])
    if kind == "ets":
        return fit_ets(train, horizon, period, context["warnings"])
    if kind in ("ridge", "random_forest"):
        index = context.get("index")
        sub_index = None
        if index is not None and context.get("train_end") is not None:
            sub_index = index[: context["train_end"]]
        return ml_forecast(kind, train, horizon, period, sub_index, context.get("freq"))
    raise KeyError(kind)


def paired_significance(backtest: dict, candidate: str, baseline: str) -> dict | None:
    """同一回测折内做配对比较，避免折间难度差异淹没模型差距。

    只看平均 RMSE 差距（效果量）会误导：实测在 10 折下，
    平均 RMSE 低 24.5% 的模型用配对 t 检验依然达不到 5% 显著水平。
    """
    from scipy import stats

    def fold_rmse(name: str) -> dict:
        return {
            item["fold"]: item["RMSE"]
            for item in backtest.get(name, {}).get("folds", [])
            if item.get("RMSE") is not None
        }

    left, right = fold_rmse(candidate), fold_rmse(baseline)
    common = sorted(set(left) & set(right))
    if len(common) < 3:
        return None
    diff = np.array([left[f] - right[f] for f in common], dtype=float)
    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    if not math.isfinite(se) or se <= 0:
        return None
    t_stat = float(diff.mean() / se)
    df = len(common) - 1
    critical = float(stats.t.ppf(0.975, df))
    return {
        "model": candidate,
        "baseline": baseline,
        "n_folds": len(common),
        "mean_delta_rmse": float(diff.mean()),
        "paired_se": se,
        "t_stat": t_stat,
        "df": df,
        "critical_t_5pct": critical,
        "significant_5pct": bool(abs(t_stat) > critical),
        "better": bool(diff.mean() < 0),
    }


def step(run_dir: Path, warnings: list[str]):
    settings = lib.load_settings(run_dir, lib.parse_argv(), prefer_resolved=True)
    frame, prep = lib.load_series(run_dir, settings)
    series = frame.iloc[:, 0]
    target = series.name

    period = int(settings.get("period") or 1)
    freq = settings.get("freq") or prep.get("freq")
    horizon = int(settings.get("horizon") or (2 * period if period > 1 else 1))
    horizon = max(1, min(horizon, max(len(series) // 4, 1)))

    n_interpolated = int(series.isna().sum())
    if n_interpolated:
        warnings.append(f"建模前插值了 {n_interpolated} 个缺失点（线性插值，两端用最近值填充）")
    values = series.interpolate(limit_direction="both").to_numpy(dtype=float)
    index = series.index

    if len(values) < 20:
        warnings.append("观测点少于 20，无法可靠建模与回测")
        raise SystemExit(lib.SKIP_STEP_EXIT_CODE)

    suggested = (lib.load_context(run_dir).get("resolved") or {})
    stationarity = lib.load_context(run_dir)
    orders = stationarity.get("resolved", {}).get("suggested_orders") if isinstance(stationarity.get("resolved"), dict) else None
    if not orders:
        try:
            import json as _json

            step2_path = run_dir / "out" / "step2_stationarity.json"
            if step2_path.exists():
                orders = _json.loads(step2_path.read_text(encoding="utf-8")).get("data", {}).get("suggested_orders")
        except Exception:  # noqa: BLE001
            orders = None
    orders = orders or {"p": 1, "d": 1 if period == 1 else 1, "q": 1, "P": 0, "D": 0, "Q": 0, "m": period}
    orders.setdefault("m", period)

    # ---- 候选模型 ---------------------------------------------------------
    candidates = ["naive", "mean", "drift"]
    if period > 1 and len(values) >= 2 * period:
        candidates.append("seasonal_naive")

    arima_order, arima_info = select_arima_order(values, orders, warnings)
    if arima_info["best_aic"] is not None:
        candidates.append("arima")
    seasonal_order = (int(orders.get("P", 0) or 0), int(orders.get("D", 0) or 0), int(orders.get("Q", 0) or 0), period)
    if period > 1 and len(values) >= 4 * period and any(seasonal_order[:3]):
        candidates.append("sarima")
    if len(values) >= 20:
        candidates.append("ets")
    if len(values) >= 200:
        candidates.extend(["ridge", "random_forest"])

    folds = plan_folds(len(values), horizon, period)
    if not folds:
        warnings.append("数据长度不足以做滚动回测，退化为单次留出法（最后一段作为测试集）")

    context = {
        "period": period,
        "freq": freq,
        "index": index,
        "arima_order": arima_order,
        "seasonal_order": seasonal_order,
        "warnings": warnings,
    }

    backtest: dict[str, dict] = {}
    residuals: dict[str, list] = {}

    def run_fold(kind: str, train: np.ndarray, test: np.ndarray, fold_id: int, train_end: int) -> tuple[np.ndarray | None, dict]:
        context["train_end"] = train_end
        try:
            mean, _, _, notes = train_model(kind, train, len(test), context)
        except Exception as exc:  # noqa: BLE001
            return None, {"error": f"{type(exc).__name__}: {exc}"}
        if mean is None:
            return None, {"error": "模型返回空预测"}
        mean = np.asarray(mean, dtype=float)
        if mean.size != len(test):
            return None, {"error": f"预测长度 {mean.size} 与测试长度 {len(test)} 不一致"}
        return mean, {"notes": notes, "fold": fold_id}

    if folds:
        for kind in candidates:
            fold_metrics, fold_detail, all_resid = [], [], []
            for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
                train = values[train_start:train_end]
                test = values[test_start:test_end]
                mean, info = run_fold(kind, train, test, fold_id, train_end)
                if mean is None:
                    fold_detail.append({"fold": fold_id, "error": info.get("error")})
                    continue
                metrics = compute_metrics(test, mean)
                fold_metrics.append(metrics)
                all_resid.extend((test - mean).tolist())
                fold_detail.append({"fold": fold_id, **metrics, "train_end": int(train_end)})
            if fold_metrics:
                summary = {
                    key: float(np.mean([m[key] for m in fold_metrics if m[key] is not None]))
                    if any(m[key] is not None for m in fold_metrics)
                    else None
                    for key in ("MAE", "RMSE", "MAPE", "sMAPE")
                }
                summary["n_folds"] = len(fold_metrics)
                backtest[kind] = {"summary": summary, "folds": fold_detail}
                residuals[kind] = all_resid
            else:
                backtest[kind] = {"summary": None, "folds": fold_detail}
                warnings.append(f"模型 {kind} 在所有回测折上都失败，已排除")
    else:
        split = max(int(len(values) * 0.8), min(2 * period, len(values) - horizon))
        train, test = values[:split], values[split : split + horizon]
        for kind in candidates:
            mean, info = run_fold(kind, train, test, 1, split)
            if mean is None:
                backtest[kind] = {"summary": None, "folds": [{"fold": 1, "error": info.get("error")}]}
                continue
            metrics = compute_metrics(test, mean)
            backtest[kind] = {"summary": {**metrics, "n_folds": 1}, "folds": [{"fold": 1, **metrics}]}
            residuals[kind] = (test - mean).tolist()

    ranked = [k for k, v in backtest.items() if v.get("summary") and v["summary"].get("RMSE") is not None]
    ranked.sort(key=lambda k: (backtest[k]["summary"]["RMSE"], COMPLEXITY.get(k, 99)))
    if not ranked:
        warnings.append("所有候选模型都未通过回测，无法给出预测")
        raise SystemExit(lib.SKIP_STEP_EXIT_CODE)

    best_model = ranked[0]
    baseline_names = [m for m in ranked if m in ("naive", "mean", "drift", "seasonal_naive")]
    best_baseline = baseline_names[0] if baseline_names else None
    improvement = None
    if best_baseline and best_baseline != best_model:
        base_rmse = backtest[best_baseline]["summary"]["RMSE"]
        if base_rmse:
            improvement = float((base_rmse - backtest[best_model]["summary"]["RMSE"]) / base_rmse)

    # ---- 最终预测 ---------------------------------------------------------
    context["train_end"] = len(values)
    forecast_mean, forecast_lower, forecast_upper, forecast_notes = train_model(best_model, values, horizon, context)
    if forecast_mean is None:
        forecast_mean = baseline_forecast("naive", values, horizon, period)
        forecast_lower = forecast_upper = None
        warnings.append(f"冠军模型 {best_model} 在全量数据上重训失败，回退到 naive 基线")

    method_note = "模型自带区间"
    if forecast_lower is None or forecast_upper is None:
        residual = np.asarray(residuals.get(best_model) or [], dtype=float)
        residual = residual[np.isfinite(residual)]
        sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual)))) if residual.size >= 10 else float(np.nanstd(values))
        if not math.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(values)) or 0.0
        horizon_scale = np.sqrt(np.arange(1, horizon + 1))
        forecast_lower = forecast_mean - Z80 * sigma * horizon_scale
        forecast_upper = forecast_mean + Z80 * sigma * horizon_scale
        method_note = f"回测残差稳健尺度 σ={sigma:.4g}，按 √h 扩张的 80% 区间"

    if freq:
        try:
            forecast_index = pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]
        except Exception:  # noqa: BLE001
            forecast_index = pd.RangeIndex(start=len(values), stop=len(values) + horizon)
    else:
        forecast_index = pd.RangeIndex(start=len(values), stop=len(values) + horizon)

    beats_baseline = None
    if best_baseline and best_baseline != best_model:
        beats_baseline = bool(improvement is not None and improvement > 0.02)
        if not beats_baseline:
            warnings.append(
                f"冠军模型 {best_model} 相对最佳基线 {best_baseline} 的 RMSE 改善仅 "
                f"{(improvement or 0):.2%}（< 2%），不建议在生产中使用复杂模型"
            )

    # ---- 配对显著性：只用效果量判定会误导 ----
    significance = {}
    if best_baseline:
        for name in ranked:
            if name == best_baseline:
                continue
            result = paired_significance(backtest, name, best_baseline)
            if result:
                significance[name] = result
    champion_significance = significance.get(best_model) if best_model != best_baseline else None
    if champion_significance and not champion_significance["significant_5pct"]:
        warnings.append(
            f"冠军模型 {best_model} 相对 {best_baseline} 的平均 RMSE 改善虽然可观，"
            f"但配对 t 检验 t={champion_significance['t_stat']:.2f}（df={champion_significance['df']}，"
            f"临界值 {champion_significance['critical_t_5pct']:.2f}）未达 5% 显著水平，"
            "折间波动很大，结论应按「方向性改善」而非「已验证优势」看待"
        )

    # ---- 图表 -------------------------------------------------------------
    plt, cjk = lib.configure_matplotlib()
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    ranked_by_rmse = sorted(backtest.items(), key=lambda item: (item[1].get("summary") or {}).get("RMSE") or 1e18)
    names = [k for k, v in ranked_by_rmse if v.get("summary")]
    rmses = [backtest[k]["summary"]["RMSE"] for k in names]
    maes = [backtest[k]["summary"]["MAE"] for k in names]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, max(3, 0.5 * len(names) + 1.5)))
    ax.barh(positions - 0.2, rmses, height=0.4, label="RMSE", color="#1f77b4")
    ax.barh(positions + 0.2, maes, height=0.4, label="MAE", color="#ff7f0e")
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title(L("滚动回测误差对比（越小越好）", "Rolling-origin backtest error (lower is better)"))
    ax.legend()
    lib.savefig(fig, run_dir, "03_backtest.png")

    fig, ax = plt.subplots()
    tail = max(int(min(len(values), 4 * horizon)), 20)
    ax.plot(index[-tail:], values[-tail:], linewidth=1.2, label=L("历史", "history"))
    ax.plot(forecast_index, forecast_mean, linewidth=1.6, color="#d62728", label=L(f"预测({best_model})", f"forecast ({best_model})"))
    ax.fill_between(forecast_index, forecast_lower, forecast_upper, color="#d62728", alpha=0.18, label=L("80% 区间", "80% interval"))
    ax.set_title(L(f"{target} — {horizon} 步预测", f"{target} — {horizon}-step forecast"))
    ax.legend(loc="best")
    lib.savefig(fig, run_dir, "03_forecast.png")

    # ---- markdown ---------------------------------------------------------
    metric_rows = []
    for name in names:
        summary = backtest[name]["summary"]
        sig = significance.get(name)
        if sig:
            marker = "✅" if sig["significant_5pct"] else "—"
            sig_cell = f"{sig['t_stat']:+.2f} ({marker})"
        elif name == best_baseline:
            sig_cell = "基准"
        else:
            sig_cell = "-"
        metric_rows.append([
            f"**{name}**" if name == best_model else name,
            lib.fmt(summary["MAE"]),
            lib.fmt(summary["RMSE"]),
            lib.fmt(summary["MAPE"], 2) if summary["MAPE"] is not None else "n/a",
            lib.fmt(summary["sMAPE"], 2),
            lib.fmt(summary["n_folds"]),
            sig_cell,
        ])

    parts = [f"## {STEP}. 建模与回测\n"]
    parts.append("### 建模配置\n")
    parts.append(lib.md_table(
        ["项目", "值"],
        [
            ["目标列", f"`{target}`"],
            ["观测点", f"{len(values):,}（插值 {n_interpolated}）"],
            ["频率 / 周期", f"`{freq or '未知'}` / {period}"],
            ["预测步长 horizon", lib.fmt(horizon)],
            ["回测方式", f"滚动原点，{len(folds) or 1} 折" + (f" × {horizon} 步" if folds else "（单次留出）")],
            ["ARIMA 阶数", f"ARIMA{arima_order}" + (f" × SARIMA{seasonal_order}" if "sarima" in candidates else "")],
            ["ARIMA 网格", f"{arima_info['grid_size']} 组，失败 {arima_info['failures']} 组，最优 AIC {lib.fmt(arima_info['best_aic'], 2)}"],
            ["候选模型", ", ".join(candidates)],
            ["区间方法", method_note],
        ],
    ))

    parts.append("\n### 回测结果（按 RMSE 升序）\n")
    parts.append(lib.md_table(["模型", "MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "折数", "配对 t (vs 最佳基线)"], metric_rows))
    parts.append(
        "\n> 基线模型（naive / mean / drift / seasonal_naive）是及格线：复杂模型必须显著优于基线才值得使用。\n"
        "> 「配对 t」是在**同一回测折内**与最佳基线做差的 t 统计量（✅ = 达到 5% 显著）；\n"
        "> 只看平均 RMSE 差距会误导——折间难度差异常常远大于模型之间的差距。\n"
    )
    parts.append(lib.fig_md("03_backtest.png", "回测误差对比"))

    parts.append("\n### 模型选择\n")
    best_summary = backtest[best_model]["summary"]
    parts.append(
        f"- 冠军模型：**{best_model}**（RMSE = {lib.fmt(best_summary['RMSE'])}，MAE = {lib.fmt(best_summary['MAE'])}）。\n"
    )
    if best_baseline:
        parts.append(
            f"- 最佳基线：`{best_baseline}`（RMSE = {lib.fmt(backtest[best_baseline]['summary']['RMSE'])}）。\n"
        )
        if best_baseline == best_model:
            parts.append("- **复杂模型没有跑赢基线**，因此推荐使用基线模型。\n")
        elif improvement is not None:
            parts.append(
                f"- 相对最佳基线的 RMSE 改善：**{improvement:.1%}**"
                + ("（改善不足 2%，实用价值有限）\n" if improvement <= 0.02 else "（效果量可观）\n")
            )
            if champion_significance:
                parts.append(
                    f"- 配对显著性：t = {champion_significance['t_stat']:.2f}"
                    f"（df = {champion_significance['df']}，5% 临界值 "
                    f"{champion_significance['critical_t_5pct']:.2f}，配对 SE = {champion_significance['paired_se']:.2f}）→ "
                    + ("**达到显著**\n" if champion_significance["significant_5pct"] else
                       "**未达显著**：只能当作方向性改善，折间波动覆盖了模型差距\n")
                )
            if best_baseline != best_model:
                worse = [
                    name for name, sig in significance.items()
                    if sig["significant_5pct"] and not sig["better"]
                ]
                if worse:
                    parts.append(
                        f"- ⚠️ 显著**劣于**最佳基线的模型（不应采用）：{', '.join(sorted(worse))}\n"
                    )
    if forecast_notes:
        note = "、".join(f"{k}={v}" for k, v in forecast_notes.items()) if isinstance(forecast_notes, dict) else str(forecast_notes)
        parts.append(f"- 拟合备注：{note}\n")

    parts.append("\n### 未来预测\n")
    display_rows = []
    shown = range(horizon) if horizon <= 24 else list(range(12)) + [horizon - 1]
    for i in shown:
        display_rows.append([
            str(forecast_index[i]),
            lib.fmt(forecast_mean[i]),
            lib.fmt(forecast_lower[i]),
            lib.fmt(forecast_upper[i]),
        ])
    parts.append(lib.md_table(["时间", "预测值", "80% 下界", "80% 上界"], display_rows))
    if horizon > 24:
        parts.append(f"\n（仅展示前 12 步与最后 1 步，完整 {horizon} 步见 `out/step3_modeling.json`）\n")
    parts.append(lib.fig_md("03_forecast.png", "预测与区间"))

    if arima_info.get("candidates"):
        parts.append("\n### ARIMA 网格 Top 候选\n")
        parts.append(lib.md_table(
            ["阶数 (p,d,q)", "AIC"],
            [[f"({c['order'][0]}, {c['order'][1]}, {c['order'][2]})", lib.fmt(c["aic"], 2)] for c in arima_info["candidates"][:5]],
        ))
    if warnings:
        parts.append("\n" + "\n".join(f"- ⚠️ {w.replace('WARN_ONLY: ', '')}" for w in warnings) + "\n")

    data = {
        "config": {
            "target": target,
            "n_obs": int(len(values)),
            "n_interpolated": n_interpolated,
            "freq": freq,
            "period": period,
            "horizon": horizon,
            "folds": len(folds),
            "candidates": candidates,
            "interval_method": method_note,
        },
        "arima_search": arima_info,
        "backtest": {k: v for k, v in backtest.items()},
        "ranking": ranked,
        "ranking_by_rmse": names,
        "best_model": best_model,
        "baseline_best": best_baseline,
        "best_baseline": best_baseline,
        "improvement_vs_baseline": improvement,
        "significance_vs_baseline": significance,
        "champion_significance": champion_significance,
        "beats_baseline": beats_baseline,
        "forecast": {
            "model": best_model,
            "index": [str(i) for i in forecast_index],
            "mean": forecast_mean.tolist(),
            "lower": np.asarray(forecast_lower).tolist(),
            "upper": np.asarray(forecast_upper).tolist(),
            "level": 0.8,
        },
        "residual_scale": {
            k: float(np.nanstd(np.asarray(v, dtype=float))) for k, v in residuals.items() if len(v)
        },
        "fit_notes": forecast_notes,
    }
    return data, "\n".join(parts)


if __name__ == "__main__":
    lib.run_step(STEP, NAME, step)
