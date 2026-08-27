# 0001 加密货币期权周末卖方策略回测

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizorewww/backtest/blob/main/tasks/0001-weekend-strangle/notebook.ipynb)

持有 BTC（或 ETH）现货，每周五 16:00 UTC 卖出当周周日 08:00 UTC 到期的
short strangle（虚值 call + put 各 1 张），持有到期现金结算，不对冲。
数据来自 Deribit 官方公开 API，2022-09-02 ~ 2026-08-21 各 208 个周末。
任务定义见 [TASK.md](TASK.md)，完整教学叙事见 notebook。

## 结果要点（基准 35Δ，单利口径，数字来自 `results/grid_deribit_*.csv`）

| 指标 | BTC | ETH |
|---|---|---|
| 总 PnL（币） | +1.166 BTC | +1.146 ETH |
| 总 PnL（USD，按入场价折算） | ≈ 64,395 | ≈ 2,418 |
| 周均收益 / 名义 | 0.561% | 0.551% |
| 胜率 | 81.3%（169/208） | 78.8%（164/208） |
| Sharpe（周频，×√52 年化） | 3.43 | 2.19 |
| 最大回撤（币） | −0.105 | −0.111 |
| 权利金留存率 | 47.1% | 34.3% |

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
