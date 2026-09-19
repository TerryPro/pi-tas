# 逐步细则

每一节包含：目标、方法菜单、判定标准、常见坑。脚本是参考实现，**数据不配合时就修改脚本**，但不要跳过产物。

---

## Step 1

### 目标
把「一个文件」变成「一条可信的等间隔时间序列」，并留下审计证据。

### 方法菜单

**时间列探测顺序**
1. `--time-col` 命令行
2. `context.json.options.time_col`
3. 列名命中 `date/time/timestamp/datetime/ds/period/日期/时间/月份`（大小写不敏感）
4. 对各 object/string 列做 `pd.to_datetime(errors="coerce")`，取解析成功率 ≥ 80% 且最高者
5. 都不行 → 把首列当序号索引，**必须在报告里明确标注这是假设**

**目标列探测顺序**
1. `--value-col`
2. `context.json.options.value_col`
3. 列名命中 `value/y/target/count/sales/amount/close/price/数值/数量`
4. 数值列中非空率最高、且标准差 > 0 的列

**频率推断**：`pd.infer_freq(index)`；失败则用相邻时间差的中位数映射
（≈1s→`s`，≈60s→`min`，≈3600s→`h`，1d→`D`，7d→`W`，28–31d→`MS`，89–92d→`QS`，365d→`YS`）。
推断出的频率与 `infer_freq` 结果不一致时，两者都记录，并在报告里说明。

**周期 `period`**：`D→7`、`B→5`、`W→52`、`MS/M→12`、`QS/Q→4`、`YS/Y→1`、`h→24`、`min/s→60`。
若 `period >= n_obs / 4`，把 `period` 置为 `1` 并把 `has_seasonality=false`（样本量不足以估计季节项）。

**质量检查清单**
- 解析后 NaT 的行数
- 重复时间戳数量（保留聚合策略要记录：默认取均值）
- 缺口：`期望点数 vs 实际点数`，列出最长的 10 段缺口（起止 + 缺失长度）
- 缺失值：按列统计 `n_missing / missing_pct`，以及最长的连续缺失区间
- 异常粗筛：IQR 法则 `x < Q1-1.5IQR or x > Q3+1.5IQR`，以及稳健 z `|x-median|/1.4826MAD > 3.5`
- 常量列 / 全空列 / 时间列重复率
- 描述统计：count / mean / std / min / 25% / 50% / 75% / max / skew / kurtosis

**质量评级**
| 等级 | 条件 |
|------|------|
| `ok` | 缺失 < 5%，无乱序，缺口 ≤ 5%，点数 ≥ 100 |
| `warn` | 缺失 5–25%，或缺口 5–20%，或存在重复时间戳 |
| `poor` | 缺失 > 25%，或缺口 > 20%，或点数 < 30，或时间列解析失败 |

### 常见坑
- CSV 用 `；`、制表符或 `;` 分隔 → `tsa_lib.read_table` 已做分隔符嗅探，但仍要检查列数是否为 1。
- 时区：`tz_localize` 不要静默丢弃时区，报告里要写出时区。
- **`DatetimeIndex.view("int64")` 的单位不是纳秒**：pandas ≥ 2 可能用 `s/ms/us/ns` 分辨率，
  手算秒数一律改用 `.diff().dt.total_seconds()`，否则频率推断会在某些 pandas 版本上静默失败。
- 数据是**累计值**（cumsum）而不是增量 → 看图就能发现单调不减，必须在报告里说明并考虑差分。
- 一份文件里有多条序列（如多门店）→ 默认只分析探测到的那一列；如果发现明显的分组列且用户没指定，在摘要里提示可以按组分别分析（不要自作主张展开成多序列，除非用户要求）。
- 数值列被读成字符串（千分位逗号、货币符号）→ 清洗后再转换，记录转换比例。

---

## Step 2

### 目标
判断序列需要几次差分才能平稳，季节性有多强，为 Step 3 提供阶数建议。

### 方法菜单

**平稳性检验**（对 3 个版本各做一遍）
| 版本 | 说明 |
|------|------|
| `level` | 原序列 |
| `diff` | 一阶差分 |
| `seasonal_diff` | 季节差分（`period > 1` 时才做） |

- ADF：`statsmodels.tsa.stattools.adfuller`，`autolag="AIC"`，`regression="c"`（有明显趋势时用 `"ct"`）。
  p < 0.05 → 拒绝单位根 → 平稳。
- KPSS：`kpss(regression="c", nlags="auto")`。p > 0.05 → 不能拒绝平稳 → 平稳。
- ADF 与 KPSS 结论矛盾时**不要强行下结论**，报告里并列写出，并用图表（滚动均值/方差）辅助判断。

**差分次数 `d` 的判定**
```
ADF(level) 平稳 且 KPSS(level) 平稳   → d = 0
否则 ADF(diff) 平稳                 → d = 1
否则 ADF(level) 平稳（与 KPSS 冲突） → d = 0（保守，避免过度差分，并记录冲突告警）
否则                                → d = 2（并标注为不稳定估计）
```
`D`：仅当 `period > 1` 且 `seasonal_strength >= 0.3` 且季节差分后 ADF 改善时取 1，否则 0。

**分解**
- `period > 1` 且 `n_obs >= 2 * period`：`STL(s, period=period, robust=True)`，`seasonal` 用 `7` 或 `period+1(奇数)`。
- 否则：只做趋势提取（`rolling(period or 7).mean()`），`seasonal_strength = 0`。

强度公式（沿用 Hyndman）：
```
trend_strength    = max(0, 1 - Var(resid) / Var(trend + resid))
seasonal_strength = max(0, 1 - Var(resid) / Var(seasonal + resid))
```
> 1 → 强趋势；< 0.3 → 弱季节。

**ACF / PACF**
- `nlags = min(40, n_obs // 3)`，多条序列长度 ≥ 2*period 时至少画到 `2 * period`。
- **在识别用序列上计算**：`d > 0` 时用一阶差分序列，否则用原序列；图与「显著滞后」列表必须用同一个序列，
  否则会出现「图上无长记忆、列表却一堆显著滞后」的自相矛盾。
- 显著滞后：`|acf| > 1.96 / sqrt(n)`。
- 建议阶数（作为 Step 3 网格的上界，不是硬性结论）：
  - `d` 来自上面；`D` 同理。
  - PACF 在 lag p 后截尾、ACF 拖尾 → AR(p)
  - ACF 在 lag q 后截尾、PACF 拖尾 → MA(q)
  - 季节滞后 `period`、`2*period` 处的显著峰 → `P`/`Q` 候选 0–1
- 输出 `suggested_orders = {p, d, q, P, D, Q, m}`，其中 `p,q ≤ 3`、`P,Q ≤ 1`。

### 常见坑
- 单位根检验对短序列（n < 50）功效很低，结论要弱化措辞。
- ADF 与 KPSS 冲突时不要直接给出 d=2：过度差分会引入不可逆的 MA 结构。先看是否有趋势/季节/结构断裂，
  再保守取较小的 d，并把冲突写进报告。
- 序列有强烈趋势时 ADF 用 `regression="ct"`，否则几乎必然「不平稳」，但这只是模型设定问题。
- 不要把「不平稳」等同于「要差分」：先看是不是季节性或结构断裂造成的（Step 4 会揭示）。
- `infer_freq` 对不规则日频（缺周末）会返回 `None`，此时用中位差推断并允许频率标记为 `irregular`。

---

## Step 3

### 目标
给出**经过回测验证**的预测，并明确它与基线的差距。

### 方法菜单

**基线（必须全部跑）**
| 模型 | 预测 |
|------|------|
| `naive` | 最后一个观测值 |
| `seasonal_naive` | `period > 1` 时用 `y[t-period]`，否则同上 |
| `mean` | 训练集均值 |
| `drift` | 最后值 + 平均斜率 × h |

**统计模型**
- `ARIMA`：在 Step 2 的 `suggested_orders` 邻域上做小网格
  （`p ∈ 0..min(3,p̂+1)`、`d ∈ {d̂-1,d̂,d̂+1} ∩ ≥0`、`q ∈ 0..min(3,q̂+1)`），按 AIC 选最优；
  网格规模上限 ~40，超了就采样。趋势项按 `d` 选：`d == 0` 用 `trend="c"`，`d == 1` 用 `trend="t"`（漂移），`d >= 2` 用 `None`。
- `SARIMA`：仅当 `period > 1` 且 `n_obs >= 4*period` 时，固定 `P=D=Q` 取 Step 2 建议值（不展开搜索，太慢）。
- `ETS`：`ExponentialSmoothing(trend="add", seasonal="add" if period>1 else None, seasonal_periods=period)`；
  季节性不显著时退化为 `Holt`（`trend="add"`）甚至 `SimpleExpSmoothing`。

**机器学习（可选，`n_obs >= 200` 时）**
- 特征：滞后 `1..min(2*period, 14)`、滚动均值/标准差（窗口 `period`、`2*period`）、
  时间派生（`hour`/`dow`/`month`/`dayofyear`/`is_weekend`），周期项用 `sin/cos` 编码。
- 模型：`Ridge`（必须）与 `RandomForestRegressor(n_estimators=200, random_state=0)`（可选）。
- 递归多步预测；缺失的滞后值用训练集最后已知值回填，并在报告里注明误差会累积。

**滚动原点回测**
- `horizon h` 见 SKILL.md；折数 `folds = clamp(floor(n_obs / (5*h)), 3, 10)`，`n_obs` 太小则降低折数直至 1（单次 hold-out）。
- 每折：训练 `[0, i*h)`，测试 `[i*h, (i+1)*h)`，从后往前构造。
- 指标：MAE、RMSE、MAPE（`y=0` 占比高时改 sMAPE）、sMAPE，附带每折明细与整体均值。
- 排序用 **RMSE 均值**；并列时优先简单模型（基线 > 统计 > ML）。

**最终预测**
- 冠军模型在全量数据上重训，预测 `h` 步。
- 区间：统计模型用 `get_forecast().conf_int(alpha=0.2)`（80%）；ML 用回测残差的分位数 `q10/q90` 构造。
- 若最优模型打不过最好的基线，**必须显式写出**：「所有复杂模型均未显著优于 <baseline>，因此不推荐使用复杂模型」。

**配对显著性（必做）**
- 只比平均 RMSE（效果量）会误导：实测在 10 折回测中，平均 RMSE 低 **24.5%** 的模型，
  配对 t 检验依然 **达不到 5% 显著水平**。
- 在**同一回测折内**做差，对差值做单样本 t 检验（`scipy.stats.t.ppf(0.975, df)`），
  输出 `mean_delta_rmse` / `paired_se` / `t_stat` / `df` / `critical_t_5pct` / `significant_5pct` / `better`。
- 这份统计量必须由脚本落盘（写进 `out/stepN_*.json`），不能在报告里口算——见 SKILL.md 硬性规则 8。
- 同时列出**显著劣于**基线的模型，这些模型不应采用。

### 常见坑
- **statsmodels 不接受低于差分阶数的 trend 项**：`d + D > 0` 时用 `trend="c"` 会直接抛 `ValueError`。
  正确映射：`d+D == 0` → `"c"`；`d+D == 1` → `"t"`（它等价于在差分序列上放常数，即漂移项）；`d+D >= 2` → `None`。
  忘了这条会导致整个 ARIMA 网格大面积拟合失败。
- 回测时**不要**做未来信息的特征工程（滚动统计必须只用历史窗口）。
- `ARIMA` 在近单位根时会收敛警告（`ConvergenceWarning`）→ 记录到 `warnings`，不要静默忽略。
- `MAPE` 在含 0 或负值时报 `inf` → 换 sMAPE 并把 `MAPE` 置 `null`。
- 只做 1 折会严重高估置信度，尽量 ≥ 3 折。
- **不要用平均 RMSE 的差距下“显著”结论**：先看折间标准差。实测折间 RMSE SD ≈ 19，
  而前几名模型之间只差 1–2，这种差距随回测窗口一变就翻盘。必须做配对检验。
- 预测区间不要给成「点预测 ± 固定值」，要用模型自带区间或残差分位数。

---

## Step 4

### 目标
指出「哪里不正常」和「哪里发生了变化」，并区分**单点异常**与**水平漂移**。

### 方法菜单

**A. 残差异常（点异常）**
- 用 Step 2 同样的 STL 设定取残差（无季节性时用 `y - rolling_median(period)`）。
- 稳健 z-score：`z = (resid - median(resid)) / (1.4826 * MAD(resid))`，`|z| > 3.5` 判为异常。
- 同时给 IQR 判定结果，两种方法都命中才标 `severity = "high"`，只命中一种标 `"medium"`。

**B. 多日异常聚集（时间段）**
- **用与点异常同一个残差稳健 z（全局尺度）**，把连续 ≥ 2 天 `|z| > 3.5` 合并成一个窗口，
  记录 `start/end/periods/level/reference/level_shift/max_abs_z/direction`。
- **不要用“滚动中位数偏离 > k×滚动 MAD”那类局部尺度做法**：局部 MAD 会在平滑区间变得极小，
  普通噪声抱团就能触发（实测干净数据上凭空报出 7 个 2 天假窗口）；
  而滚动中位数窗只有 `max(period,5)` 天、会“跟着”长平台走，对持续 3 天的 +200 偏离反而漏报。
  它不是良定义的方法。
- **永久水平偏移不属于这里**，请看 C（变点）。分工要在报告里写清楚。
- 若窗口内有原本缺失、后被插值的天数，必须记 `missing_days_inside` 并明确提示
  「该聚集更可能是插值产物而非真实事件」——缺失段被线性插值后几乎必然在残差上聚集。

**C. 变点（结构断裂）**
- 先把序列处理成“去周季节 + 去年度谐波”的基准，再用**分段线性**二分分割：
  递归寻找使 `SSE(split)` 下降最多的分割点（`penalty = 3·ln(n)·Var(y)`，BIC 风格），深度上限 5。
  分段用**直线**而不是均值去拟合区段，这样平滑趋势的增益 ≈ 0，而真正的水平偏移仍有巨大增益；
  否则在一根趋势线上会一刀一刀切出假变点。
- CUSUM：先做一次线性去趋势再做累积和（不先去除趋势，任何有趋势的序列都会在中点越界），`|CUSUM| > 4σ` 的越界高峰作为交叉验证。
- 若环境里存在 `ruptures`，额外跑 `PELT(model="rbf", pen=...)` 并对比；不一致时以「两法都命中」为高置信度。
- 输出变点列表：`index`、`timestamp`、`type`、`delta`（**断点处两条拟合直线的跳跃量**，不是段均值差）、
  `level_before` / `level_after`（断点两侧的拟合水平）、`variance_ratio`（去趋势后的标准差比）、
  `gain` / `penalty`（是否真的超过阈值）、`method`、`corroborated_by`、`confidence`。

**输出**
- `anomalies`：默认最多 20 条，按 `|z|` 降序；若超过 20 条，额外输出 `anomaly_rate` 并在报告里提示「异常率过高，可能是波动性上升或模型设定不当」。
- `episodes` / `changepoints`：全量输出。
- `summary`：`n_anomalies`、`anomaly_rate`、`n_episodes`、`n_changepoints`、`dominant_method`。

### 常见坑
- 异常检测前必须处理缺失值（插值或跳过），否则 MAD 会被 NaN 污染。
- 若异常率 > 10%，**不要罗列所有点**：说明「序列整体波动性高」并只给 top 异常。
- 变点数量受惩罚项强烈影响：报告里要写出使用的惩罚与「这不是唯一正确答案」。
- **不要用 IRLS / 稳健加权去拟合年度谐波**：它会把真实的水平偏移当成离群点降权，反过来扭曲谐波系数，
  实测反而使变点变得难以识别。用普通最小二乘。
- **不要给年度谐波拟合再加一个全局线性趋势项**：水平偏移会把斜率拉偏（实测偏 2.7 倍），
  残差里剩下一个假斜率，再被分割切成「一年一个假变点」。趋势交给分段线性分割自己处理。
- 当水平偏移发生在季节性转变（如年末购物季）之前不久时，偏移量会被高估：
  「-90 的水平下移」与「季节性回落」在只有一两个月样本时本来就是不可分的，报告要说明这一点。
- 季节性未剔除时的「异常」往往是季节性峰值 → 必须用残差，不要直接在原序列上判异常。

---

## Step 5 检查清单

生成 `report.md` 后逐条自检：

- [ ] `report.md` 存在，且四个步骤片段都被合并进去
- [ ] `<!-- TSA:SUMMARY -->` 和 `<!-- TSA:CONCLUSIONS -->` 已被替换成正文（不再有占位符）
- [ ] 报告里的每个数字都能在 `out/*.json` 里找到来源
- [ ] 图片路径能打开（`figures/*.png` 真实存在）
- [ ] 失败/降级的步骤被明确标注（`status != ok`）
- [ ] `out/run_manifest.json` 的各步 `status` 与实际一致
- [ ] 附录里的复现命令可以直接复制执行
