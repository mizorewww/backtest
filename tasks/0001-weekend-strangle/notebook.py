# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 加密货币期权周末卖方策略回测
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizorewww/backtest/blob/main/tasks/0001-weekend-strangle/notebook.ipynb)
#
# 持有 BTC（或 ETH）现货，每周五 16:00 UTC 卖出当周周日 08:00 UTC 到期的
# short strangle（虚值 call + put 各 1 张），持有到期现金结算，不对冲。
# 数据来自 Deribit 官方公开 API，样本为 2022-09 ~ 2026-08 的每个周五。
#
# 本 notebook 是教学材料：先讲清楚策略为什么可能赚钱，再看数据与假设，
# 然后逐段解释定价与选合约的代码，最后看结果、做敏感性分析、列出局限性。

# %% [markdown]
# ## 1. 策略动机与经济解释
#
# 这个策略不是无风险套利，它的期望收益来自承担两种风险溢价：
#
# - **波动率风险溢价（IV > RV）**：期权的隐含波动率在多数时期系统性地高于
#   随后实现的波动率。买方为"保险"多付了钱，卖方赚取 IV 与 RV 之差。
#   短周期（40 小时）期权把这个溢价压缩到很短的时间里兑现。
# - **周末效应**：持仓窗口是周五 16:00 ~ 周日 08:00 UTC。传统市场闭市、
#   宏观事件少，周末的已实现波动往往偏低；而加密市场 7×24 交易，
#   期权仍按连续时间定价。卖方在这个窗口收取的权利金相对充裕。
# - **代价是尾部风险**：周末流动性最薄，一旦发生地缘/政策冲击，
#   加密是唯一开盘的风险资产，抛压集中在卖方持仓的窗口里。
#   所以预期看到的损益结构是"多次小赢、少数大亏"——高胜率、正期望、
#   左尾很厚。
#
# 持有现货的角色：short call 腿与现货构成 covered call；short put 腿是裸卖
# （实盘中需要额外保证金，本回测未建模保证金占用）。

# %% [markdown]
# ## 2. 数据与假设
#
# **数据**（全部来自 Deribit 官方公开 API，免费、含已到期合约全历史）：
#
# - `get_tradingview_chart_data`：小时 K 线。标的用 BTC/ETH-PERPETUAL，
#   期权用各合约自身 K 线（周五 12:00~18:00 窗口）。
# - `get_delivery_prices`：官方交割价，即周日 08:00 UTC 的结算依据。
# - 历史行权价宇宙：Deribit 的 `get_instruments?expired=true` 只覆盖最近约一天，
#   因此通过**探测合约名**（如 `BTC-27AUG23-26000-C`）重建——不存在的合约返回
#   "instrument not found"。探测范围：现货 ±max(12%, 3.2σ√T) 带宽内，
#   按 strike 步长（BTC 250 / ETH 25）扫描。
#
# **关键假设**：
#
# - 定价模型 Black-Scholes，**r = 0**，T = 40h；IV 从入场成交价二分反推，
#   delta 用同一 IV 计算。
# - 币本位：Deribit 期权面值 1 BTC / 1 ETH，权利金与结算均以币计。
#   1 张腿的币本位 PnL 在数值上等于对 1 单位名义的收益率。
# - 入场价：优先取 16:00 小时 K 线开盘价（要求该小时有成交）；否则依次回退到
#   15:00~12:00 的收盘价、17:00/18:00 的开盘价。
# - 费用（Deribit 费率）：开仓 0.03% 名义、封顶权利金的 12.5%；
#   行权 0.015% 名义、仅 ITM 收取、封顶 payoff 的 12.5%。
#
# 下面先做环境引导（Colab 与本地两分支），再加载数据看规模。

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Colab 分支：克隆仓库、进入任务目录、缺数据则现场下载（约数十分钟）
    os.environ.setdefault("TRADE_DATA_ROOT", "/content/trade-data")
    if not Path("/content/backtest").exists():
        # !git clone --depth 1 https://github.com/mizorewww/backtest.git /content/backtest
    # %cd /content/backtest/tasks/0001-weekend-strangle
    if not (Path(os.environ["TRADE_DATA_ROOT"]) / "deribit" / "weekly").exists():
        # !python download_data.py
else:
    # 本地分支：默认数据根 /Volumes/trade/data，可用 TRADE_DATA_ROOT 覆盖
    os.environ.setdefault("TRADE_DATA_ROOT", "/Volumes/trade/data")

print("工作目录:", Path.cwd())
print("数据根目录:", os.environ["TRADE_DATA_ROOT"])

# %%
import pandas as pd
import matplotlib.pyplot as plt

from backtest import RAW_DIR, load_delivery, load_quotes

# backtest.py import 时切了 Agg 后端，这里切回 inline 让图片进入单元格输出
# %matplotlib inline

# 中文字体：本机（macOS）用 PingFang SC；列表回退兼容其他系统
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

weeks = {cur: load_quotes(cur) for cur in ("BTC", "ETH")}
delivery = {cur: load_delivery(cur) for cur in ("BTC", "ETH")}

for cur in ("BTC", "ETH"):
    wk = weeks[cur]
    n_quotes = sum(len(w["quotes"]) for w in wk)
    src = pd.Series([q["src"] for w in wk for q in w["quotes"]])
    print(f"{cur}: {len(wk)} 个可交易周末（{wk[0]['friday']} ~ {wk[-1]['friday']}），"
          f"候选合约腿共 {n_quotes} 条，"
          f"其中 {src.eq('open@16').mean():.1%} 直接取到 16:00 开盘价，"
          f"其余使用 12:00~18:00 窗口内的最近成交价")
    print(f"    交割价记录 {len(delivery[cur])} 天")

# %% [markdown]
# 两点观察：
#
# - 约 208 个周末、每周几十条候选腿，样本量对"周频策略"而言尚可，
#   但对尾部事件（大涨大跌周末）只有个位数样本——这决定了后面结论的置信度。
# - 不是所有腿都能在 16:00 整点取到价：周末前夕虚值期权成交稀疏，
#   一部分入场价是窗口内的最近成交价。这是一个偏差来源（见第 6 节）。

# %% [markdown]
# ## 3. 方法
#
# 策略逻辑全部在本目录的 `backtest.py` 里，notebook 直接 import，不重复实现。
# 核心分四段讲：**BS 定价**（3.1）、**IV 反推**（3.2）、**按 delta 选合约**、
# **币本位结算与费用**（3.3）。每段先给公式、逐个符号解释，再给对应代码。

# %% [markdown]
# ### 3.1 BS 定价与 delta（r = 0）
#
# 期限极短（40 小时），取无风险利率 $r=0$，Black-Scholes 公式退化为：
#
# $$d_1 = \frac{\ln(S/K) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$
#
# $$C = S\,N(d_1) - K\,N(d_2), \qquad P = K\,N(-d_2) - S\,N(-d_1)$$
#
# 符号逐个解释：
#
# - $S$：入场时标的现货价（USD），即周五 16:00 UTC 的 BTC/ETH 价格；
# - $K$：行权价（USD）；
# - $\sigma$：年化隐含波动率（由 3.2 节从成交价反推）；
# - $T$：到期期限（年），本策略 $T = 40/(365 \times 24) \approx 0.00457$
#   （周五 16:00 → 周日 08:00 UTC）；
# - $N(\cdot)$：标准正态分布的累积分布函数；
# - $C, P$：1 单位名义的 call / put 的 USD 理论价值。
#
# $r=0$ 的合理性：$T$ 只有 40 小时，贴现因子 $e^{-rT}$ 与 1 的差异可忽略，
# 远期价 $Se^{rT}$ 即现货价；这也让权利金的币本位/USD 换算更干净。
#
# delta（对 $S$ 的一阶偏导）在同一假设下为：
#
# $$\Delta_C = N(d_1), \qquad \Delta_P = N(d_1) - 1$$
#
# 选合约规则：在候选池中取 $|\Delta - \Delta^*|$ 最小的行权价，
# call 目标 $\Delta^* = +0.35$，put 目标 $\Delta^* = -0.35$（基准档）。

# %%
import inspect

from backtest import bs_price, bs_delta, invert_iv, run_backtest

print(inspect.getsource(bs_price))
print(inspect.getsource(bs_delta))

# %% [markdown]
# 对应上面的公式：`bs_price` 里 `sq` $= \sigma\sqrt{T}$，`d1`、`d2` 与
# 两式逐一对应；`sigma <= 0` 的分支是 $\sigma \to 0$ 的极限——价格退化为
# 内在价值。`bs_delta` 复用同一个 $d_1$ 算 $N(d_1)$。

# %% [markdown]
# ### 3.2 IV 二分反推
#
# 入场成交价 $P_{mkt}$（USD）是观测值，隐含波动率 $\sigma^*$ 是它的反函数：
#
# $$\text{求 } \sigma^* \text{ 使 } V_{BS}(\sigma^*;\, S, K, T) = P_{mkt}$$
#
# BS 价格对 $\sigma$ 严格单调递增（vega $> 0$），所以方程有唯一解，
# 在 $[\sigma_{lo}, \sigma_{hi}] = [10^{-3},\, 6]$ 上二分 60 次即可收敛到
# 任意精度。两类腿被剔除、不参与选合约：
#
# - $P_{mkt} \le V_{BS}(0)$：价格低于内在价值（数据噪音或陈旧价）；
# - $V_{BS}(\sigma_{hi}) < P_{mkt}$：价格高到 IV = 6（年化 600%）都够不着。
#
# 反推出的 $\sigma^*$ 同时用于 3.1 节算 delta，保证定价口径自洽。

# %%
print(inspect.getsource(invert_iv))

# %% [markdown]
# `bs_price(s, k, 0.0, side)` 即内在价值 $V_{BS}(0)$；循环里的中点更新
# 就是上面单调性二分：理论价低于市价则抬下界，否则压上界。

# %% [markdown]
# 用一个真实的周末走一遍选合约流程：

# %%
wk = weeks["BTC"][0]
s0 = wk["spot"]
cands = [q for q in wk["quotes"] if q["side"] == "C"]
nearest = sorted(cands, key=lambda q: abs(q["delta"] - 0.35))[:5]
print(f"入场周五 {wk['friday']}，BTC 现货 {s0:,.0f} USD，目标 call delta = +0.35")
display(pd.DataFrame(nearest)[["name", "strike", "price", "iv", "delta", "src"]])

chosen = min(cands, key=lambda q: abs(q["delta"] - 0.35))
reprice = bs_price(s0, chosen["strike"], chosen["iv"], "C")
print(f"选中 {chosen['name']}：delta={chosen['delta']:.3f}，IV={chosen['iv']:.2f}（年化）")
print(f"自洽性检验：成交价 {chosen['price'] * s0:.2f} USD vs 用反推 IV 重定价 {reprice:.2f} USD")

# %% [markdown]
# ### 3.3 币本位 payoff 与费用模型
#
# Deribit 期权是币本位的：面值 1 币，权利金 $p$ 与到期赔付都以币计价、
# 以币结算。卖方持有一张腿到期的币本位赔付为：
#
# $$\text{payoff}_C = \frac{\max(S_T - K,\ 0)}{S_T}, \qquad \text{payoff}_P = \frac{\max(K - S_T,\ 0)}{S_T}$$
#
# 其中 $S_T$ 是周日 08:00 UTC 的官方交割价（USD）；USD 赔付除以 $S_T$
# 折成币。单腿卖方 PnL（币）为：
#
# $$\text{PnL}_{leg} = p - \text{payoff} - f_{open} - f_{settle}$$
#
# 费用按 Deribit 费率（名义 = 1 币，封顶 12.5%）：
#
# $$f_{open} = \min\big(0.0003,\ 0.125 \cdot p\big), \qquad f_{settle} = \mathbb{1}\{\text{payoff} > 0\} \cdot \min\big(0.00015,\ 0.125 \cdot \text{payoff}\big)$$
#
# - $f_{open}$：开仓费，名义的 0.03%，封顶权利金的 12.5%——深度虚值合约
#   权利金极低时封顶生效（如 $p = 0.0005$ 时 $0.125p = 0.0000625 < 0.0003$）；
# - $f_{settle}$：行权费，仅 ITM（payoff > 0）收取，名义的 0.015%，
#   封顶 payoff 的 12.5%。
#
# 单周 PnL = call 腿 + put 腿的 $\text{PnL}_{leg}$ 之和。
# 每一周都重复 3.1~3.3：反推 IV → 按 delta 选两腿 → 持有到期结算扣费。
# 完整的逐周循环在 `run_backtest` 里：

# %%
print(inspect.getsource(run_backtest))

# %% [markdown]
# 对照 3.3 的公式看内层循环：`payoff` 即 $\max(\cdot)/S_T$，
# `open_fee` / `settle_fee` 即两个带封顶的 $\min$，`leg_pnl` 即
# $\text{PnL}_{leg}$；外层对每周重复，尾部汇总出网格表的一行。

# %% [markdown]
# ## 4. 结果
#
# 先跑 12 档 delta（0.05~0.60，步长 0.05）× 2 个标的的全网格，
# 结果写入 `results/`（逐笔 CSV、汇总 CSV、累计 PnL 曲线 PNG）。

# %%
from backtest import DELTA_GRID, run_backtest, plot_grid

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BASE_DELTA = 0.35
grids, trades35 = {}, {}
for cur in ("BTC", "ETH"):
    rows, curves = [], {}
    for d in DELTA_GRID:
        t, summ = run_backtest(cur, d, weeks[cur], delivery[cur])
        t.to_csv(RESULTS_DIR / f"trades_deribit_{cur}_d{int(round(d * 100)):02d}.csv",
                 index=False)
        rows.append({"delta": d, **summ})
        curves[d] = t.set_index("friday")["pnl"].cumsum()
        if d == BASE_DELTA:
            trades35[cur] = t
    grids[cur] = pd.DataFrame(rows)
    grids[cur].to_csv(RESULTS_DIR / f"grid_deribit_{cur}.csv", index=False)
    plot_grid(cur, curves, RESULTS_DIR / f"equity_deribit_{cur}.png")
    print(f"{cur}: {len(t)} 周已写入 results/")

# %% [markdown]
# ### 数据卡片（35Δ 基准档）
#
# 口径：**单利**（每周固定卖 1 张 strangle，PnL 直接累加，不再投资）、
# **币本位**（PnL / 权利金 / 手续费的单位分别为 BTC / ETH）、样本为
# 2022-09 ~ 2026-08 的每个周五（窗口与笔数以卡片第一行为准）。
# 最大回撤为币本位累计 PnL 曲线的峰谷差（绝对口径，单位：币）。
# 下面这张卡片由代码从 `results/trades_deribit_*_d35.csv` 计算生成，不手抄。

# %%
import numpy as np


def perf_card(cur: str) -> dict:
    """从 35Δ 逐笔 CSV 计算标准数据卡片的一列（指标集见 data-viz skill）。"""
    t = pd.read_csv(RESULTS_DIR / f"trades_deribit_{cur}_d35.csv")
    pnl = t["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    cum = pnl.cumsum()
    ret = t["pnl_pct"]  # 单利口径下 = 币本位 PnL / 1 单位名义
    return {
        "样本窗口": f"{t['friday'].iloc[0]} ~ {t['friday'].iloc[-1]}",
        "净利润（币）": f"{pnl.sum():+.3f}",
        "交易数 / 胜率": f"{len(t)} / {(pnl > 0).mean():.1%}",
        "盈利因子（总盈利/|总亏损|）": f"{wins.sum() / abs(losses.sum()):.2f}",
        "最大回撤（币）": f"{(cum - cum.cummax()).min():.3f}",
        "平均交易 / 平均盈利笔 / 平均亏损笔":
            f"{pnl.mean():+.4f} / {wins.mean():+.4f} / {losses.mean():+.4f}",
        "最好 / 最差单笔": f"{pnl.max():+.4f} / {pnl.min():+.4f}",
        "Sharpe（周频 ×√52 年化）": f"{ret.mean() / ret.std() * np.sqrt(52):.2f}",
        "总手续费（币）": f"{t['fees'].sum():.4f}",
        "权利金留存率（净利润/权利金合计）": f"{pnl.sum() / t['premium_sum'].sum():.1%}",
    }


card = pd.DataFrame({cur: perf_card(cur) for cur in ("BTC", "ETH")})
card.index.name = "指标"
display(card)

# %% [markdown]
# ### 4.1 基准档位（35Δ）

# %%
rows = []
for cur in ("BTC", "ETH"):
    g = grids[cur].set_index("delta").loc[BASE_DELTA]
    t = trades35[cur]
    rows.append({
        "标的": cur,
        "总 PnL（币）": f"+{g['total_pnl_base']:.3f}",
        "总 PnL（USD，按入场价折算）": f"{g['total_pnl_usd']:,.0f}",
        "周均收益/名义": f"{g['ret_on_notional']:.3%}",
        "胜率": f"{g['win_rate']:.1%}（{int((t['pnl'] > 0).sum())}/{len(t)}）",
        "Sharpe（周频×√52）": f"{g['sharpe_w']:.2f}",
        "最大回撤（币）": f"{g['max_dd_base']:.3f}",
        "权利金合计（币）": f"{g['premium_sum_base']:.3f}",
        "权利金留存率": f"{g['retained']:.1%}",
        "35Δ 腿平均 IV": f"{(t['C_iv'].mean() + t['P_iv'].mean()) / 2:.2f}",
    })
display(pd.DataFrame(rows).set_index("标的").T)

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
for cur in ("BTC", "ETH"):
    t = trades35[cur]
    ax.plot(pd.to_datetime(t["friday"]), t["pnl"].cumsum(), label=f"{cur} 35Δ", lw=1.2)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("基准档位 35Δ：累计 PnL（币本位，单利口径）")
ax.set_xlabel("日期")
ax.set_ylabel("累计 PnL（币）")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# 权益曲线的形态正是第 1 节的预期：长时间缓慢爬升（每周赚权利金），
# 间或被几次急跌打断（尾部周末）。注意两个标的的回撤时点高度重合——
# 打击卖方的是同一批宏观事件，BTC/ETH 双标的不是有效的分散化。

# %% [markdown]
# ### 4.2 逐笔损益分布：多次小赢、少数大亏

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, cur in zip(axes, ("BTC", "ETH")):
    t = trades35[cur]
    ax.hist(t["pnl"], bins=40, color="steelblue")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"{cur} 35Δ 单周 PnL 分布")
    ax.set_xlabel(f"单周 PnL（{cur}）")
    wins, losses = t[t["pnl"] > 0]["pnl"], t[t["pnl"] <= 0]["pnl"]
    print(f"{cur}: 盈利周平均 +{wins.mean():.4f}（上限约 +{wins.max():.4f}），"
          f"亏损周平均 {losses.mean():.4f}，最差单周 {losses.min():.4f}")
fig.tight_layout()
plt.show()

# %%
for cur in ("BTC", "ETH"):
    t = trades35[cur]
    worst = t.nsmallest(5, "pnl")[
        ["friday", "spot_ret", "pnl", "C_strike", "P_strike", "C_payoff", "P_payoff"]]
    print(f"\n{cur} 35Δ 最差 5 周（spot_ret = 周末现货涨跌幅）：")
    display(worst.reset_index(drop=True))

# %% [markdown]
# 分布右偏截断、左尾拉长：盈利周的上限就是权利金（卖方收益有顶），
# 亏损周可以亏掉数十倍单周权利金。注意最差周里 payoff 只在一条腿上出现——
# 暴涨周末亏 call 腿、暴跌周末亏 put 腿，另一条腿的权利金照收。

# %% [markdown]
# ### 4.3 十二档 delta 对比

# %%
for cur in ("BTC", "ETH"):
    g = grids[cur][["delta", "weeks", "total_pnl_base", "ret_on_notional",
                    "win_rate", "sharpe_w", "max_dd_base", "retained"]]
    print(f"\n{cur}（总 PnL 单位：{cur}）：")
    display(g.style.format({
        "total_pnl_base": "{:.3f}", "ret_on_notional": "{:.2%}",
        "win_rate": "{:.1%}", "sharpe_w": "{:.2f}",
        "max_dd_base": "{:.3f}", "retained": "{:.1%}",
    }).hide(axis="index"))

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for cur, color in (("BTC", "darkorange"), ("ETH", "steelblue")):
    g = grids[cur]
    axes[0].plot(g["delta"], g["total_pnl_base"], marker="o", ms=4, label=cur, color=color)
    axes[1].plot(g["delta"], g["sharpe_w"], marker="o", ms=4, label=cur, color=color)
    axes[2].plot(g["delta"], g["retained"], marker="o", ms=4, label=cur, color=color)
for ax, title in zip(axes, ("总 PnL（币）", "Sharpe（年化）", "权利金留存率")):
    ax.set_title(title)
    ax.set_xlabel("目标 delta")
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# 三句话读完这组图（具体数字见上表）：
#
# - **总 PnL 随 delta 单调上升**：行权价越靠近平值，收的权利金越多。
#   但这是"卖更多保险收更多保费"，不是免费午餐——留存率同步大幅下滑。
# - **Sharpe 不是单调的**：BTC 呈双峰（极虚值档与高档都好，中间略低），
#   ETH 总体随 delta 上行。0.35~0.45 中间档在收益、胜率、留存率之间较均衡，
#   这是选 0.35 作基准的理由。
# - **胜率随 delta 下行**：越靠近平值越容易被击穿。

# %% [markdown]
# ## 5. 敏感性分析
#
# 三个方向：复利口径、call/put 非对称 delta、回撤事件归因。
# 逻辑在 `optimize.py`，notebook 直接 import。

# %% [markdown]
# ### 5.1 复利口径
#
# 模型：起始持有 1 单位现货，**每周将全部币本位权益滚入下周名义**——
# 权益 $B_t$ 对应每周卖 $B_t$ 张 strangle，周收益率 $r_t$ 等于单张币本位 PnL，
# 故 `equity = Π(1+r_t)`；USD 口径再乘每周现货涨跌，对比纯囤币。

# %%
from itertools import product

from optimize import simulate, metrics, drawdown_episodes

# optimize.py import 时又切回了 Agg，再切回 inline
# %matplotlib inline

comp35, sym_grids = {}, {}
for cur in ("BTC", "ETH"):
    sym = pd.DataFrame([{"delta": d, **metrics(simulate(weeks[cur], delivery[cur], d, d))}
                        for d in DELTA_GRID])
    sym.to_csv(RESULTS_DIR / f"compound_sym_{cur}.csv", index=False)
    sym_grids[cur] = sym
    t35 = simulate(weeks[cur], delivery[cur], BASE_DELTA, BASE_DELTA)
    comp35[cur] = t35
    m = metrics(t35)
    print(f"{cur} 35Δ 复利口径：期末权益 {m['equity_final']:.2f} 倍（币本位），"
          f"CAGR {m['cagr']:.1%}，Calmar {m['calmar']:.2f}；"
          f"USD 口径 {m['usd_final']:.2f} 倍 vs 囤币 {m['bh_final']:.2f} 倍，"
          f"USD 最大回撤 {m['max_dd_usd']:.1%}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, cur in zip(axes, ("BTC", "ETH")):
    t = comp35[cur]
    dates = pd.to_datetime(t["friday"])
    ax.plot(dates, (1 + t["ret"]).cumprod().values, label="策略（币本位复利）", lw=1.2)
    ax.plot(dates, ((1 + t["ret"]) * t["spot_ratio"]).cumprod().values,
            label="策略（USD 口径）", lw=1.2)
    ax.plot(dates, t["spot_ratio"].cumprod().values, label="纯囤币（USD）", lw=1.2, ls="--")
    ax.set_title(f"{cur} 35Δ 复利净值（期初 = 1）")
    ax.set_xlabel("日期")
    ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# 复利显著放大了与囤币的差距（数字见上）。但要注意这个模型假设
# 每周权利金与赔付可以**无损再投资**为下周名义——实际受合约最小面值
# （1 张 = 1 单位标的）和流动性约束，小额资金无法严格复制这条曲线。

# %% [markdown]
# ### 5.2 非对称 delta：call 与 put 独立选档
#
# 对称 strangle 隐含"两侧风险对称"的假设，但样本里暴涨周末和暴跌周末的
# 频率/幅度并不对称。让 call 腿、put 腿独立选择目标 delta，跑 12×12 = 144
# 个组合（复利口径），看 CAGR 与 Calmar 的热图。

# %%
asym = {}
for cur in ("BTC", "ETH"):
    rows = []
    for dc, dp in product(DELTA_GRID, DELTA_GRID):
        rows.append({"delta_call": dc, "delta_put": dp,
                     **metrics(simulate(weeks[cur], delivery[cur], dc, dp))})
    a = pd.DataFrame(rows)
    a.to_csv(RESULTS_DIR / f"compound_asym_{cur}.csv", index=False)
    asym[cur] = a
    print(f"{cur}: 144 个组合完成")

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for i, cur in enumerate(("BTC", "ETH")):
    for j, (val, title) in enumerate((("cagr", "CAGR"), ("calmar", "Calmar"))):
        piv = asym[cur].pivot(index="delta_put", columns="delta_call", values=val)
        ax = axes[i, j]
        im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
        fig.colorbar(im, ax=ax, label=val)
        ax.set_xticks(range(len(piv.columns)), [f"{c:.2f}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{c:.2f}" for c in piv.index])
        ax.set_xlabel("call delta")
        ax.set_ylabel("put delta")
        ax.set_title(f"{cur}: {title}（复利口径）")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "heatmap_asym.png", dpi=120)
plt.show()

# %%
for cur in ("BTC", "ETH"):
    top5 = asym[cur].nlargest(5, "calmar")[
        ["delta_call", "delta_put", "cagr", "sharpe_w", "max_dd", "calmar", "win_rate"]]
    print(f"\n{cur} 按 Calmar 排序前五：")
    display(top5.style.format({"cagr": "{:.1%}", "sharpe_w": "{:.2f}",
                               "max_dd": "{:.1%}", "calmar": "{:.2f}",
                               "win_rate": "{:.1%}"}).hide(axis="index"))

# %% [markdown]
# 读热图的要点（数字见上表）：
#
# - **最优区是平台而非尖点**：BTC 在 call∈[0.15,0.30] × put∈[0.50,0.60]
#   整个矩形内都成立；ETH 在 call∈[0.35,0.40] × put=0.55 附近同样平坦。
#   选平台中点而非单点，是为了缓解样本内寻优的过拟合。
# - **put 侧吃下行偏度溢价**：put 维持高 delta 收更多权利金，
#   样本内下行周末的赔付频率低于定价隐含。
# - **call 侧降 delta 规避暴涨周末的空头挤压**：样本里几次大回撤中，
#   上涨型的亏损全部来自 call 腿，拉低 call delta 直接削弱这类亏损。
# - 这些都是 **208 周样本内**观察，不构成样本外保证；
#   0.60 是网格边界，"put 越高越好"可能只是边界效应。

# %% [markdown]
# ### 5.3 回撤事件归因
#
# 把 35Δ 对称配置的复利权益切成 peak→trough 区间，看前几大回撤：

# %%
for cur in ("BTC", "ETH"):
    ep = drawdown_episodes(comp35[cur])
    ep.to_csv(RESULTS_DIR / f"drawdowns_{cur}_35-35.csv", index=False)
    print(f"\n{cur} 35/35 复利口径前 {len(ep)} 大回撤区间：")
    display(ep.style.format({"depth": "{:.1%}"}).hide(axis="index"))

# %% [markdown]
# 对这些区间内的周末逐一做事件调研（据公开市场报道与行情复盘，
# 逐笔数据交叉验证），**所有大回撤都是"外生冲击 × 周末流动性真空"的组合**：
#
# | 周末（周五入场） | 事件归因 |
# |---|---|
# | 2023-01-13 / 01-20 | CPI 降温 + FTX 暴雷后修复 + 空头挤压，BTC 两周 17k→23k；亏损全部来自 call 腿 |
# | 2024-04-12 | 伊朗首次直接袭击以色列 + 减半前杠杆清洗 |
# | 2024-08-02 | 日元套息交易平仓；真正的尾部在**周一**，周日 08:00 结算低估了尾部 |
# | 2024-11-08 | 特朗普胜选后的轧空；ETH 亏损全部来自 call 腿 |
# | 2025-01-31 | 特朗普首轮关税行政令 |
# | 2025-06-20 | 美军轰炸伊朗核设施（周末突发，BTC 已基本收复而 ETH 没有） |
# | 2025-10-10 | 特朗普 100% 对华关税威胁，据公开报道全网强平约 $19.3B、为史上最大 |
# | 2026-01-30 ~ 02-13 | 日本国债闪崩 + 格陵兰关税争端 + Fed 鹰派 |
#
# 共性：地缘与政策冲击偏好周末发生（或周末才传入市场），此时传统市场闭市，
# 加密是唯一开盘的风险资产——抛压正好落在本策略的持仓窗口。
# 亏损方向跟随冲击方向；ETH 的尾部约为 BTC 的 1.5~2 倍（非固定比例）。

# %% [markdown]
# ## 6. 局限性
#
# 这份回测在以下方面是**偏乐观**的，解读结果时必须带着这些修正：
#
# - **成交价非买一价**：入场价用小时 K 线开盘价或最近成交价，未考虑买卖价差；
#   周末深度薄的虚值期权实际滑点会系统性侵蚀卖方收益。
# - **部分入场价陈旧**：一部分腿用了 16:00 前后窗口内的最近成交价
#   （最早回退到 12:00），该价格对应的现货与入场时点现货存在偏差。
# - **r=0 近似**：40 小时期限下影响很小，但严格说贴现与远期定价被忽略。
# - **保证金与资金占用未建模**：short put 腿的保证金要求、资金机会成本未计入。
# - **币本位 PnL 的 USD 折算时点**：USD 值按各周入场价折算，仅作量级参考。
# - **样本期环境依赖**：2022-09 ~ 2026-08 的 IV 水平与周末波动特征不保证延续；
#   若未来 IV 中枢下移或周末极端波动增多，结果会显著不同。
# - **行权价宇宙的重建方式**：仅扫描 ±max(12%, 3.2σ√T) 带宽，极端行情周目标
#   delta 档可能落在带宽外而选到次优合约。
# - **样本内寻优的过拟合**：非对称 delta 的最优参数在同一 208 周样本上选出，
#   选"平台"而非单点只能缓解、不能消除过拟合。
# - **周日 08:00 结算低估延续到周一的尾部**：如 2024-08 日元套息平仓的主跌段
#   发生在周一。回测度量的是"周末窗口"的风险，不是"持仓到周一"的风险。
# - **复利模型假设无损再投资**：忽略合约最小面值取整，小额资金无法严格复制。

# %% [markdown]
# ## 7. 复现方法
#
# **本地**（仓库根目录，环境用 uv 管理）：
#
# ```bash
# uv sync
# cd tasks/0001-weekend-strangle
# uv run download_data.py   # 缺数据时从 Deribit 下载；有缓存则跳过
# uv run backtest.py        # 12 档 delta × BTC/ETH，写 results/
# uv run optimize.py        # 复利 + 非对称网格 + 回撤区间
# uv run make_report.py     # 生成 results/report.html
# ```
#
# 数据根目录默认 `/Volumes/trade/data`，可用环境变量 `TRADE_DATA_ROOT` 覆盖。
#
# **Colab**：点顶部 badge 打开本 notebook 直接运行。引导 cell 会检测
# `google.colab` 环境，克隆仓库、进入任务目录，数据不存在时自动运行
# `download_data.py`（首次下载约数十分钟）。
#
# ---
#
# *本 notebook 为研究性回测，所有结果均为历史样本内统计，不构成投资建议。*
