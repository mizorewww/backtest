---
name: backtest
description: |
  执行教学向量化回测任务。当用户说"做个回测"、"跑一下 XX 策略"、"验证 XX 想法"，
  或给出策略规则要求用历史数据验证时使用。覆盖：任务定义（TASK.md）、数据获取
  （/Volumes/trade/data 优先 + 独立下载脚本）、自包含任务目录、jupytext 教学
  notebook、验收门禁、回测方法论常见坑。
---

# 回测任务执行规范

工作仓库：`~/Developer/backtest`（uv 管理，一律 `uv run`）。先读该仓库的 AGENTS.md。

## 流程

1. **确认任务定义**。用户口述的策略先落成 `TASK.md` 再动手，字段：策略假设（一句话
   经济逻辑 + 预期看到什么）、数据（来源/窗口/标的）、规则（入场/出场/费用/仓位，
   精确到可编码）、参数网格、验收标准（具体数字或可检验条件）。规则模糊处向用户
   问清楚，不自己猜。
2. **建任务目录** `tasks/NNNN-<kebab-name>/`，编号取现有最大编号 +1。任务内代码
   自包含，从旧任务复制改是正常做法，不做跨任务复用抽象。
3. **准备数据**（见下节）。
4. **写回测代码**，跑通并核对 TASK.md 验收标准。
5. **写教学 notebook**（见下节），同步执行。
6. **写 README**：标准数据卡片 + Colab badge。
7. **过验收门禁**（见末节），然后汇报：结果要点、与预期是否一致、局限性。

## 数据

- 数据根目录：`Path(os.environ.get("TRADE_DATA_ROOT", "/Volumes/trade/data"))`，
  按数据源建子目录（如 `deribit/`、`binance/`）。
- **先找后下**：先在数据根目录找已有数据，缺失才下载。
- 每个数据源写一个独立下载脚本 `download_data.py`：只依赖 requests + 标准库 +
  pandas（Colab 预装范围），带缓存跳过逻辑，能在 Colab 裸环境直接跑。这个脚本
  是任务可移植性的保证，不是可选项。
- 原始数据只进不出：下载脚本只追加/更新缓存，不回写修改已有原始文件。

## Notebook（教学核心交付物）

- jupytext percent 格式：手写 `notebook.py`（`# %% [markdown]` / `# %%`），
  `uv run jupytext --sync` 生成 ipynb，`uv run jupyter nbconvert --to notebook
  --execute --inplace` 执行。不直接编辑 .ipynb JSON。
- 七段叙事结构：① 策略动机与经济解释 ② 数据与假设 ③ 方法（核心代码逐段讲）
  ④ 结果 ⑤ 敏感性分析 ⑥ 局限性 ⑦ 复现方法。
- 方法节必须含数学推导：定价/信号/费用/指标计算的公式用 LaTeX 写在 markdown cell
  （`$$...$$`），公式后紧跟对应代码，一一对应，读者能对上。
- 结果节开头必须放标准回测数据卡片（指标集与模板见 `data-viz` skill），
  README 同样要有一张。
- 正文数字一律从运行结果计算插入（f-string/Markdown 插值），不硬编码。
- Colab 引导 cell：检测 `google.colab` → clone 仓库 → cd 到任务目录 → 数据缺失
  时跑下载脚本。notebook 必须在 Colab 端到端可跑。
- README 的 badge：
  `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizorewww/backtest/blob/main/tasks/NNNN-<name>/notebook.ipynb)`

## 方法论红线（每个任务都要自查）

- **前视偏差**：每个决策点只用了当时可得的数据？入场价、信号、合约选择逐一确认。
- **成交价假设**：用的是买一/卖一还是标记价/收盘价？滑点怎么处理的？写进局限性。
- **费用**：手续费、资金费率、行权费，漏了会系统性高估收益。
- **本位一致性**：币本位 vs USD 本位的 PnL 不能混着加；折算时点要声明。
- **样本偏差**：样本期市场环境（波动率中枢、牛熊）不代表未来；幸存者偏差。
- **参数网格与多重比较**：扫了 N 组参数挑最好的，要说明这是样本内最优，
  最好用风险调整指标（Sharpe/Calmar）而非总收益选参数。
- **局限性一节不允许为空**——写不出局限性说明还没想清楚。

## 验收门禁（全绿才算完成）

- [ ] `uv run` 下回测脚本和 notebook 端到端无报错
- [ ] TASK.md 验收标准逐项核对通过（不通过要查根因，不许改逻辑凑数）
- [ ] results/ 产物齐全（汇总 CSV、逐笔 CSV、图）
- [ ] notebook 正文无硬编码数字、局限性非空
- [ ] README 有 Colab badge，下载脚本可独立运行
- [ ] 画图遵循 `data-viz` skill
