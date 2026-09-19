---
name: tsa
version: 0.2.0
description: 时序数据分析：创建 uv 隔离环境并运行 5 步分析流水线，产出中文 markdown 报告
argument-hint: "<data-file> [--value-col X] [--freq D] [--horizon 14]"
---
运行 `time-series-analysis` 技能描述的完整时序分析流水线。

目标文件：$1
附加要求：${@:2}

请先用 read 工具读取技能文件 `SKILL.md`（在 pi 包 `skills/time-series-analysis/` 目录下，
可用 `pi list` 或搜索 `time-series-analysis/SKILL.md` 定位），然后严格按技能中的 5 个步骤执行：
1. 数据概览与质量检查
2. 平稳性与分解
3. 建模与回测（滚动原点）
4. 异常与变点检测
5. 汇总报告

工作方式：

- 在当前目录创建任务目录 `tsa-run-<slug>-<时间戳>/`，内含 `scripts/`、`out/`、`figures/`。
- 用 `uv` 建立独立虚拟环境（`uv init` / `uv sync`），所有代码通过 `uv run` 执行，不要污染全局环境。
- 每一步编写并执行 Python 脚本，产物落盘为 `out/stepN_*.json`（机器可读）与 `out/stepN_*.md`（报告片段）。
- 最终把结果汇总成中文 markdown 报告 `report.md`，并在最后汇报关键发现。

如果缺少目标文件（`$1` 为空）或路径不存在，只做一件事：列出当前目录下的数据文件候选，让用户选一个，不要自行开始分析。

---

> 注意：当 pi-tsa 扩展已加载时，`/tsa` 在到达模板展开之前就被扩展的 `input` 钩子接管，
> 上面这段正文不会执行（本文件的作用是让 `/tsa` 出现在自动补全与命令列表里，
> 并提供未加载扩展时的兑底行为）。
