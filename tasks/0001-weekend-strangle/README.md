# 0001 加密货币期权周末卖方策略回测

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizorewww/backtest/blob/main/tasks/0001-weekend-strangle/notebook.ipynb)

持有 BTC（或 ETH）现货，每周五 16:00 UTC 卖出当周周日 08:00 UTC 到期的
short strangle（虚值 call + put 各 1 张），持有到期现金结算，不对冲。
数据来自 Deribit 官方公开 API，2022-09-02 ~ 2026-08-21 各 208 个周末。
任务定义见 [TASK.md](TASK.md)，完整教学叙事见 notebook。

## 结果要点（基准 35Δ，单利、币本位口径）

口径：单利（每周固定卖 1 张 strangle，PnL 直接累加）、币本位（单位分别为
BTC / ETH）、样本 2022-09-02 ~ 2026-08-21 各 208 周。数字由 notebook 数据卡片
cell 从 `results/trades_deribit_*_d35.csv` 计算生成。

| 指标 | BTC | ETH |
|---|---|---|
| 净利润（币） | +1.166 | +1.146 |
| 交易数 / 胜率 | 208 / 81.2% | 208 / 78.8% |
| 盈利因子（总盈利/\|总亏损\|） | 3.43 | 2.30 |
| 最大回撤（币） | −0.105 | −0.111 |
| 平均交易 / 平均盈利笔 / 平均亏损笔 | +0.0056 / +0.0097 / −0.0123 | +0.0055 / +0.0124 / −0.0201 |
| 最好 / 最差单笔 | +0.0269 / −0.0596 | +0.0474 / −0.1018 |
| Sharpe（周频 ×√52 年化） | 3.43 | 2.19 |
| 总手续费（币） | 0.1389 | 0.1406 |
| 权利金留存率（净利润/权利金合计） | 47.1% | 34.3% |

补充：总 PnL 按各周入场价折算 USD 约 64,395（BTC）/ 2,418（ETH），仅作量级参考。

复利口径 35Δ（`results/compound_sym_*.csv`）：BTC 期末权益 3.15 倍、CAGR 33.3%、
Calmar 3.25；ETH 期末权益 3.03 倍、CAGR 31.9%、Calmar 2.92。

非对称 delta 样本内最优（`results/compound_asym_*.csv`，按 Calmar）：
BTC call 0.25 / put 0.60（CAGR 43.3%，Calmar 7.23）；
ETH call 0.40 / put 0.55（CAGR 47.5%，Calmar 4.27）。

损益结构是"多次小赢、少数大亏"：大回撤全部是外生冲击 × 周末流动性真空的组合
（详见 notebook 第 5.3 节事件归因与第 6 节局限性）。**历史样本内统计，不构成投资建议。**

## 复现

```bash
uv sync
cd tasks/0001-weekend-strangle
uv run download_data.py   # 缺数据时从 Deribit 下载；有缓存则跳过
uv run backtest.py        # 12 档 delta × BTC/ETH → results/
uv run optimize.py        # 复利 + 非对称网格 + 回撤区间
uv run make_report.py     # results/report.html
```

数据根目录默认 `/Volumes/trade/data`，可用环境变量 `TRADE_DATA_ROOT` 覆盖。
Notebook 维护流程：改 `notebook.py` 后
`uv run jupytext --sync notebook.py && uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`。

## 输出

`results/` 下：`grid_deribit_{BTC,ETH}.csv`（12 档汇总）、
`trades_deribit_*_d*.csv`（24 个逐笔文件）、`equity_deribit_*.png`（累计 PnL 曲线）、
`compound_sym_*.csv` / `compound_asym_*.csv`（复利与非对称网格）、
`heatmap_*.png`（热图）、`drawdowns_*.csv`（回撤区间）、`report.html`（自包含报告）。
