# AGENTS.md

教学向回测仓库。一个任务一个文件夹，任务内代码自包含，不做跨任务复用抽象。

## 目录约定

- `tasks/NNNN-<kebab-name>/`：每个回测任务一个目录，编号递增。
- 任务目录内含：`TASK.md`（任务定义）、`README.md`（结果说明 + Colab badge）、
  `notebook.py`（jupytext percent 格式源文件）、`notebook.ipynb`（执行产物）、
  任务自己的 `.py` 代码、`data/`（本地缓存）、`results/`（CSV/PNG/HTML）。

## 数据

- 数据根目录是 `/Volumes/trade/data`（可用环境变量 `TRADE_DATA_ROOT` 覆盖）。
- 优先在数据根目录找已有数据；没有再用任务目录里的下载脚本抓新数据。
- 每个数据源必须有独立的下载脚本（如 `download_data.py`），只依赖
  requests/pandas 等 Colab 预装库，保证脚本在 Colab 里也能直接跑。

## Notebook 纪律

- notebook 用 jupytext 配对：改 `notebook.py`，然后
  `uv run jupytext --sync notebook.py` 生成 ipynb，
  `uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb` 执行。
- 叙事结构：动机与经济解释 → 数据与假设 → 方法 → 结果 → 敏感性分析 → 局限性。
- 正文数字从 `results/` 的文件读入，不硬编码。
- 每个任务的 README 必须有 "Open in Colab" badge，指向 GitHub 上的 notebook.ipynb；
  notebook 在 Colab 里必须能端到端跑通（缺数据时自动跑下载脚本）。

## 工作方式

- 接回测任务时先读项目级 skill `.agents/skills/backtest/SKILL.md`。
- 画图遵循项目级 skill `.agents/skills/data-viz/SKILL.md`。
- 环境用 uv：`uv sync` 后一律 `uv run ...`。
