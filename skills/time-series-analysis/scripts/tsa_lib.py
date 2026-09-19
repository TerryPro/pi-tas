"""tsa_lib — 时序分析流水线的共享工具库.

设计原则:
  * 任务目录 (run_dir) 是唯一事实来源: context.json + out/ + figures/
  * 所有脚本都用 `uv run python scripts/stepN_xxx.py` 在任务目录内执行
  * 脚本默认把「脚本所在目录的上一级」当作任务目录, 可用 --run 覆盖
  * 任何写进报告的统计量都必须来自这里产出的 JSON

依赖: pandas, numpy, scipy, statsmodels, matplotlib (见 pyproject.toml)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SKIP_STEP_EXIT_CODE",
    "configure_matplotlib",
    "detect_time_column",
    "detect_value_column",
    "emit",
    "fig_path",
    "fmt",
    "infer_freq",
    "load_context",
    "load_series",
    "load_settings",
    "md_table",
    "now_iso",
    "parse_argv",
    "period_for_freq",
    "prepare_frame",
    "read_table",
    "run_dir_from_argv",
    "run_step",
    "save_json",
    "savefig",
    "to_numeric_series",
    "update_resolved",
    "update_step_status",
]

# 用于「数据不足, 跳过本步」的退出码, 与「脚本出错」区分开
SKIP_STEP_EXIT_CODE = 3

# ---------------------------------------------------------------------------
# 任务目录 / 上下文
# ---------------------------------------------------------------------------

_DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1]

# matplotlib 缓存目录必须在使用 matplotlib 之前设置
if "MPLCONFIGDIR" not in os.environ:
    try:
        _cache = _DEFAULT_RUN_DIR / ".cache" / "matplotlib"
        _cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_cache)
    except OSError:  # pragma: no cover - 只读文件系统等极端情况
        pass


_FLAG_ALIASES = {
    "--run": "run",
    "--run-dir": "run",
    "--input": "input",
    "--data": "input",
    "--time-col": "time_col",
    "--time-column": "time_col",
    "--time": "time_col",
    "--value-col": "value_col",
    "--value-column": "value_col",
    "--value": "value_col",
    "--target-col": "value_col",
    "--target": "value_col",
    "--freq": "freq",
    "--frequency": "freq",
    "--agg": "agg",
    "--aggregate": "agg",
    "--horizon": "horizon",
    "--max-anomalies": "max_anomalies",
}


def parse_argv(argv: list[str] | None = None) -> dict:
    """极简的 `--key value` / `--key=value` 解析器."""
    argv = list(sys.argv[1:] if argv is None else argv)
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--") and "=" in token:
            key, value = token.split("=", 1)
        else:
            key, value = token, None
        name = _FLAG_ALIASES.get(key)
        if name is None:
            i += 1
            continue
        if value is None:
            i += 1
            value = argv[i] if i < len(argv) else ""
        out[name] = value
        i += 1
    return out


def run_dir_from_argv(default_parents: int = 1) -> Path:
    """优先取 --run, 否则用脚本目录的上一级."""
    run_dir = _DEFAULT_RUN_DIR
    if default_parents != 1:
        run_dir = Path(__file__).resolve().parents[default_parents]
    override = parse_argv().get("run")
    if override:
        run_dir = Path(override).expanduser()
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"任务目录不存在: {run_dir}（用 --run <dir> 指定）")
    return run_dir


def load_context(run_dir: Path | str) -> dict:
    path = Path(run_dir) / "context.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path | str, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def sanitize(obj):
    """把 numpy / pandas / NaN 转成合法 JSON 值."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitize(v) for v in sorted(obj, key=str)]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    return str(obj)


def update_resolved(run_dir: Path | str, patch: dict) -> dict:
    """局部更新 context.json.resolved，保留 options / steps."""
    run_dir = Path(run_dir)
    ctx = load_context(run_dir)
    resolved = dict(ctx.get("resolved") or {})
    resolved.update({k: sanitize(v) for k, v in patch.items()})
    ctx["resolved"] = resolved
    save_json(run_dir / "context.json", ctx)
    return resolved


def update_step_status(run_dir: Path | str, step_id: int, status: str, notes: str = "") -> None:
    run_dir = Path(run_dir)
    ctx = load_context(run_dir)
    steps = ctx.get("steps")
    if not isinstance(steps, list):
        return
    for entry in steps:
        if entry.get("id") == step_id:
            entry["status"] = status
            if notes:
                entry["notes"] = notes[:500]
            break
    ctx["steps"] = steps
    save_json(run_dir / "context.json", ctx)


def load_settings(run_dir: Path | str, overrides: dict | None = None, prefer_resolved: bool = False) -> dict:
    """把 context.json + 命令行覆盖合并成一份可用设置."""
    overrides = overrides or {}
    ctx = load_context(run_dir)
    resolved = ctx.get("resolved") or {}
    options = ctx.get("options") or {}

    def pick(key: str, default=None):
        value = overrides.get(key)
        if value not in (None, ""):
            return value
        if prefer_resolved and resolved.get(key) not in (None, ""):
            return resolved[key]
        if options.get(key) not in (None, ""):
            return options[key]
        if resolved.get(key) not in (None, ""):
            return resolved[key]
        return default

    return {
        "input_path": overrides.get("input") or ctx.get("input_path"),
        "time_col": pick("time_col"),
        "value_col": pick("value_col"),
        "freq": pick("freq"),
        "agg": pick("agg", "mean"),
        "horizon": pick("horizon"),
        "max_anomalies": pick("max_anomalies", 20),
        "period": resolved.get("period"),
        "n_obs": resolved.get("n_obs"),
        "context": ctx,
        "overrides": overrides,
    }


# ---------------------------------------------------------------------------
# 时间 / 格式化
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def start_timer() -> dict:
    return {"mono": time.monotonic(), "iso": now_iso()}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "-"
        if value == 0:
            return "0"
        magnitude = abs(float(value))
        if magnitude >= 1e6 or magnitude < 1e-3:
            return f"{float(value):.3e}"
        return f"{float(value):,.{digits}f}"
    return str(value)


def md_table(headers: list, rows: list[list]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def fig_path(run_dir: Path | str, name: str) -> Path:
    path = Path(run_dir) / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fig_md(name: str, caption: str = "") -> str:
    """生成 report.md 中可用的图片引用（路径相对 report.md）."""
    rel = f"figures/{name}"
    return f"\n![{caption or name}]({rel})\n" + (f"\n*{caption}*\n" if caption else "")


# ---------------------------------------------------------------------------
# 数据读取与列探测
# ---------------------------------------------------------------------------


def read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".csv", ".txt", ".dat", ""):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            sample = handle.readline()
        counts = {sep: sample.count(sep) for sep in [",", "\t", ";", "|"]}
        sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
        return pd.read_csv(path, sep=sep)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".json",):
        try:
            return pd.read_json(path)
        except ValueError:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        return pd.DataFrame(value)
            return pd.json_normalize(payload)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext == ".orc":
        return pd.read_orc(path)
    if ext == ".dta":
        return pd.read_stata(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


_TIME_HINTS = (
    "date", "time", "timestamp", "datetime", "ds", "period", "month",
    "日期", "时间", "月份", "时间戳",
)
_VALUE_HINTS = (
    "value", "values", "y", "target", "count", "sales", "amount",
    "close", "price", "qty", "quantity", "数值", "数量", "销量",
)


def _parse_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def detect_time_column(df: pd.DataFrame, hint: str | None = None) -> str | None:
    if hint:
        for column in df.columns:
            if str(column) == str(hint):
                return column
    for column in df.columns:
        if str(column).strip().lower() in _TIME_HINTS:
            return column
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
    best, best_rate = None, 0.0
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        sample = series.dropna().astype(str).head(300)
        if sample.empty:
            continue
        rate = float(_parse_datetime(sample).notna().mean())
        if rate > best_rate:
            best, best_rate = column, rate
    return best if best_rate >= 0.8 else None


def from_epoch(series: pd.Series) -> pd.Series:
    """把 10/13 位整数当成 unix 时间戳."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return _parse_datetime(series)
    magnitude = float(finite.abs().median())
    unit = "s" if magnitude < 1e11 else "ms"
    return pd.to_datetime(values, unit=unit, errors="coerce", origin="unix")


def detect_value_column(df: pd.DataFrame, exclude=(), hint: str | None = None) -> str | None:
    exclude = {str(c) for c in exclude}
    if hint:
        for column in df.columns:
            if str(column) == str(hint):
                return column
    numeric = [c for c in df.columns if str(c) not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return None
    for column in numeric:
        if str(column).strip().lower() in _VALUE_HINTS:
            return column
    def score(column: str) -> tuple[float, float]:
        series = df[column]
        variance = float(series.std(skipna=True)) if series.notna().sum() > 2 else 0.0
        return (float(series.notna().mean()), variance if math.isfinite(variance) else 0.0)
    return max(numeric, key=score)


def to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"^[^\d\-.+]+", "", regex=True)
        .str.replace(r"[^\d.\-+eE]+$", "", regex=True)
        .replace({"": None, "-": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def load_series(run_dir: Path | str, settings: dict | None = None, overrides: dict | None = None):
    """按 resolved 设置重新读入并整理数据，返回 (frame, info).

    步骤 2-4 都用它，保证各步看到的是同一份序列。
    """
    if settings is None:
        settings = load_settings(run_dir, overrides, prefer_resolved=True)
    input_path = settings.get("input_path")
    if not input_path:
        raise ValueError("context.json 缺少 input_path，无法定位数据文件")
    input_path = Path(input_path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {input_path}")
    raw = read_table(input_path)
    return prepare_frame(
        raw,
        time_col=settings.get("time_col"),
        value_col=settings.get("value_col"),
        freq=settings.get("freq"),
        agg=settings.get("agg") or "mean",
        target_name=str(settings["value_col"]) if settings.get("value_col") else None,
    )


def infer_freq(index: pd.DatetimeIndex) -> str | None:
    if len(index) < 3:
        return None
    try:
        inferred = pd.infer_freq(index)
    except (ValueError, TypeError):
        inferred = None
    if inferred:
        return inferred
    # 注意: pandas >= 2 的 DatetimeIndex 分辨率可能是 s/ms/us/ns，
    # view("int64") 的单位随分辨率变化，不能假设纳秒。total_seconds() 与分辨率无关。
    deltas = pd.Series(index).diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    median = float(np.median(deltas))
    candidates = [
        (1.0, "s"), (60.0, "min"), (900.0, "15min"), (3600.0, "h"),
        (86400.0, "D"), (7 * 86400.0, "W"), (30.44 * 86400.0, "MS"),
        (91.31 * 86400.0, "QS"), (365.25 * 86400.0, "YS"),
    ]
    best = min(candidates, key=lambda item: abs(math.log(max(median, 1e-9) / item[0])))
    ratio = max(median, 1e-9) / best[0]
    if not (0.75 <= ratio <= 1.33):
        return None
    return best[1]


_PERIOD_TABLE = [
    ("MS", 12), ("MIN", 60), ("H", 24), ("D", 7), ("B", 5), ("W", 52),
    ("Q", 4), ("Y", 1), ("A", 1), ("M", 12), ("S", 60), ("T", 60),
]


def period_for_freq(freq: str | None) -> int:
    if not freq:
        return 1
    base = str(freq).strip().upper()
    for key, period in sorted(_PERIOD_TABLE, key=lambda item: -len(item[0])):
        if base.startswith(key):
            return period
    return 1


def prepare_frame(
    df: pd.DataFrame,
    time_col: str | None = None,
    value_col: str | None = None,
    freq: str | None = None,
    agg: str = "mean",
    target_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """返回 (单列时序 DataFrame, 处理信息).

    target_name 为 None 时，输出列名跟随探测到的原始数值列名（报告里显示的就是它）。
    """
    info: dict = {"warnings": [], "dropped_rows": 0, "duplicate_timestamps": 0, "resampled": False}

    resolved_time = time_col or detect_time_column(df)
    if resolved_time is None:
        raise ValueError("无法识别时间列，请用 --time-col 指定")
    info["time_col"] = str(resolved_time)

    raw_time = df[resolved_time]
    timestamps = _parse_datetime(raw_time)
    if timestamps.notna().mean() < 0.5 and pd.api.types.is_numeric_dtype(raw_time):
        timestamps = from_epoch(raw_time)
        info["time_parsed_as"] = "epoch"
    info["unparsed_timestamps"] = int(timestamps.isna().sum())

    resolved_value = value_col or detect_value_column(df, exclude=[resolved_time])
    if resolved_value is None:
        raise ValueError("无法识别数值列，请用 --value-col 指定")
    info["value_col"] = str(resolved_value)
    column_name = str(target_name) if target_name else str(resolved_value)
    info["target_name"] = column_name
    values = to_numeric_series(df[resolved_value])
    info["non_numeric_values"] = int(values.isna().sum() - df[resolved_value].isna().sum())
    non_numeric_rate = float(values.isna().mean() - df[resolved_value].isna().mean())
    if non_numeric_rate > 0.01:
        info["warnings"].append(f"数值列 {resolved_value} 有 {non_numeric_rate:.1%} 的值无法解析为数字")

    # 内部先用哨兵列名，避免与时间列或目标列重名
    sentinel = "__tsa_value__"
    frame = pd.DataFrame({"timestamp": timestamps, sentinel: values})
    before = len(frame)
    frame = frame.dropna(subset=["timestamp"])
    info["dropped_rows"] = before - len(frame)

    frame = frame.sort_values("timestamp")
    info["unsorted"] = bool((timestamps.dropna().values != frame["timestamp"].values).any())

    duplicated = int(frame["timestamp"].duplicated().sum())
    info["duplicate_timestamps"] = duplicated
    if duplicated:
        frame = frame.groupby("timestamp", as_index=False)[sentinel].agg(agg)

    series = frame.set_index("timestamp")[sentinel].rename(column_name).astype(float)
    # 重采样/补齐之前的时间戳，用于缺口分析（重采样会把缺口变成 NaN，掩盖真实缺口）
    info["observed_index"] = series.index

    resolved_freq = freq or infer_freq(series.index)
    info["freq"] = resolved_freq
    # 重复时间戳已在上面按 agg 聚合，此处 index 已唯一，因此允许重采样补齐日历缺口
    if resolved_freq and len(series) > 2 and series.index.is_unique:
        expected = pd.date_range(series.index.min(), series.index.max(), freq=resolved_freq)
        if len(expected) < len(series):
            info["warnings"].append("推断频率与数据密度不符，保留原始索引（可能是不规则采样）")
        elif len(expected) > len(series):
            series = series.reindex(expected)
            info["resampled"] = True
            info["filled_slots"] = int(len(expected) - len(info["observed_index"]))
    return series.to_frame(column_name), info


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

_CJK_FONTS = (
    "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS",
)


def configure_matplotlib(prefer_cjk: bool = True):
    """配置 headless matplotlib，返回 (plt, cjk_available)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = False
    if prefer_cjk:
        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in _CJK_FONTS:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name, *plt.rcParams["font.sans-serif"]]
                cjk = True
                break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (11, 4)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    return plt, cjk


def savefig(fig, run_dir: Path | str, name: str) -> str:
    path = fig_path(run_dir, name)
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# 步骤执行外壳
# ---------------------------------------------------------------------------


def emit(
    run_dir: Path | str,
    step: int,
    name: str,
    *,
    data: dict | None = None,
    md: str | None = None,
    status: str = "ok",
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    started: dict | None = None,
) -> dict:
    run_dir = Path(run_dir)
    payload = {
        "step": step,
        "name": name,
        "status": status,
        "started_at": started["iso"] if started else None,
        "finished_at": now_iso(),
        "elapsed_sec": round(time.monotonic() - started["mono"], 3) if started else None,
        "warnings": list(warnings or []),
        "errors": list(errors or []),
        "data": data or {},
    }
    save_json(run_dir / "out" / f"step{step}_{name}.json", payload)
    if md:
        (run_dir / "out" / f"step{step}_{name}.md").write_text(md.strip() + "\n", encoding="utf-8")
    note = (errors or warnings or [""])[0]
    update_step_status(run_dir, step, status, str(note))
    print(f"[tsa] step{step} {name}: {status} -> out/step{step}_{name}.json")
    for warning in warnings or []:
        print(f"[tsa]   warning: {warning}")
    return payload


def run_step(step: int, name: str, fn):
    """执行 step 函数 fn(run_dir, warnings) -> (data, md).

    成功: 写出 status=ok/warn 的产物。
    SystemExit(SKIP_STEP_EXIT_CODE): 写出 status=skipped 的产物后正常退出。
    其他异常: 写出 status=failed 的产物, 打印 traceback 并以退出码 1 结束。
    """
    run_dir = run_dir_from_argv()
    started = start_timer()
    warnings: list[str] = []
    try:
        result = fn(run_dir, warnings)
    except SystemExit as exc:
        if exc.code == SKIP_STEP_EXIT_CODE:
            emit(
                run_dir, step, name,
                data={"reason": "insufficient data"},
                md=f"## {step}. 跳过\n\n数据条件不满足，本步骤已跳过。\n",
                status="skipped",
                warnings=warnings,
                started=started,
            )
            return
        raise
    except Exception as exc:  # noqa: BLE001 - 需要把失败也落盘
        traceback.print_exc()
        emit(
            run_dir, step, name,
            data={},
            md=(
                f"## {step}. 未完成\n\n"
                f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                f"详细堆栈见 `out/step{step}_{name}.json` 与终端输出。\n"
            ),
            status="failed",
            warnings=warnings,
            errors=[f"{type(exc).__name__}: {exc}"],
            started=started,
        )
        raise

    data, md = result if isinstance(result, tuple) else (result, None)
    status = "warn" if any(str(w).startswith("WARN_ONLY") for w in warnings) else "ok"
    emit(run_dir, step, name, data=data, md=md, status=status, warnings=warnings, started=started)
