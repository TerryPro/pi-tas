#!/usr/bin/env python
"""生成配合 /tsa 使用的示例时序数据（仅标准库，固定随机种子）。

    python sample-data/make_sample_data.py [--out sample-data/retail_daily_sales.csv]

产物: sample-data/retail_daily_sales.csv

数据设定：单门店日频零售销量，2023-01-01 → 2025-12-31（1096 天）。
刻意植入的"真实世界"结构，用来覆盖 /tsa 的四个分析步骤：

  1. 概览与质量检查
     - 5 天连续缺失（系统故障）、2 天单点缺失
     - 1 个整行缺失的时间戳（日历缺口）
     - 1 个重复时间戳（事后修正，需要按 mean 聚合）
     - 2 行乱序
     - 一个常量列 `region`、一个二元列 `is_promo`
  2. 平稳性与分解
     - 稳定的星期效应（周末高峰）+ 年度周期（夏季峰、年末次峰）
     - 三年累计 +98 的增长趋势
  3. 建模与回测
     - 线性趋势 + 双周期 + 高斯噪声，季节朴素法只能拿到部分信息
  4. 异常与变点检测
     - 4 个点异常（3 个突增、1 个突降）
     - 2 次水平漂移（渠道导流抬升、竞品分流下调）
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, timedelta

SEED = 20260319
START = date(2023, 1, 1)
DAYS = 1096  # 含 2024 闰年，2023-01-01 .. 2025-12-31

BASE = 820.0
TREND_PER_DAY = 0.09

# 周一..周日（weekday() 顺序 0..6）
WEEKLY = {0: -62, 1: -78, 2: -24, 3: -8, 4: 44, 5: 152, 6: 118}

# 水平漂移: (生效日期, 偏移量, 说明)
LEVEL_SHIFTS = [
    (date(2024, 9, 1), 150.0, "渠道导流 / 新增配送覆盖"),
    (date(2025, 10, 1), -90.0, "竞品在同一商圈开业"),
]

# 点异常: (日期, 偏差, 说明)
SPIKES = [
    (date(2024, 6, 18), 265.0, "直播带货爆单"),
    (date(2025, 1, 25), -235.0, "收银系统故障，半天无数据"),
    (date(2025, 8, 8), 305.0, "周年庆大促"),
    (date(2024, 11, 11), 240.0, "双十一"),
]

# 缺失值: 连续段与单点
MISSING_BLOCK = (date(2024, 5, 6), 5)
MISSING_DAYS = {date(2024, 10, 3), date(2025, 6, 11)}

# 整行缺失的时间戳（日历缺口）
DROPPED_DAY = date(2024, 8, 15)

# 重复时间戳（同一天两行，值略有差异）与乱序行下标
DUPLICATE_DAY = date(2025, 4, 2)
UNSORTED_AT = 640


def seasonality(day: date) -> float:
    """年度周期（夏季峰）+ 年末购物季次峰。

    节假日用高斯峰而不是月度平台，避免在月初/年初制造人工阶跃
    （那种阶跃是真实的零售数据不会有的，却会让变点检测器误报）。
    """
    doy = day.timetuple().tm_yday
    summer = 120.0 * math.cos(2 * math.pi * (doy - 172) / 365.25)

    def bump(center_doy: float, width_days: float, amplitude: float) -> float:
        # 环形距离，保证跨年连续
        delta = (doy - center_doy + 182.625) % 365.25 - 182.625
        return amplitude * math.exp(-0.5 * (delta / width_days) ** 2)

    holiday = bump(315, 12, 150.0) + bump(355, 10, 210.0)
    return summer + holiday


def build_rows() -> list[list[str]]:
    rng = random.Random(SEED)
    rows: list[list[str]] = []
    missing_block_days = {
        MISSING_BLOCK[0] + timedelta(days=offset) for offset in range(MISSING_BLOCK[1])
    }

    for i in range(DAYS):
        day = START + timedelta(days=i)
        if day == DROPPED_DAY:
            continue

        value = BASE + TREND_PER_DAY * i + WEEKLY[day.weekday()] + seasonality(day)
        for effective, offset, _ in LEVEL_SHIFTS:
            if day >= effective:
                value += offset
        for spike_day, delta, _ in SPIKES:
            if day == spike_day:
                value += delta
        value += rng.gauss(0.0, 28.0)

        price = 39.9 + 0.0016 * i + rng.gauss(0.0, 0.35)
        promo = 1 if (day.weekday() >= 5 or rng.random() < 0.06) else 0

        blank = day in missing_block_days or day in MISSING_DAYS
        rows.append(
            [
                day.isoformat(),
                "华东",
                "" if blank else str(int(round(value))),
                f"{max(price, 19.9):.2f}",
                str(promo),
            ]
        )

    # 重复时间戳：同日补一行"事后修正"
    for position, row in enumerate(rows):
        if row[0] == DUPLICATE_DAY.isoformat():
            corrected = list(row)
            corrected[2] = str(int(corrected[2]) + 4)
            rows.insert(position + 1, corrected)
            break

    # 乱序：交换相邻两行
    rows[UNSORTED_AT], rows[UNSORTED_AT + 1] = rows[UNSORTED_AT + 1], rows[UNSORTED_AT]

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 /tsa 示例数据")
    parser.add_argument("--out", default="sample-data/retail_daily_sales.csv")
    args = parser.parse_args()

    rows = build_rows()
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "region", "sales", "unit_price", "is_promo"])
        writer.writerows(rows)

    print(f"wrote {args.out}")
    print(f"  data rows   : {len(rows)}")
    print(f"  date range  : {rows[0][0]} .. {max(r[0] for r in rows)}")
    print(f"  level shifts: {len(LEVEL_SHIFTS)}")
    print(f"  point spikes: {len(SPIKES)}")
    print(f"  missing     : {MISSING_BLOCK[1]} 连续 + {len(MISSING_DAYS)} 单点")
    print(f"  gap / dup   : 1 个缺口 ({DROPPED_DAY}) / 1 个重复 ({DUPLICATE_DAY})")


if __name__ == "__main__":
    main()
