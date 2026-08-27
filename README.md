# backtest

教学向回测仓库。每个回测任务一个独立文件夹（`tasks/NNNN-<name>/`），
包含任务定义、自包含代码、数据下载脚本和可复现的 Jupyter notebook。

## 环境

```bash
uv sync
```

## 任务列表

| # | 任务 | 说明 |
|---|---|---|
| 0001 | [weekend-strangle](tasks/0001-weekend-strangle/) | 加密期权周末卖方策略（Deribit 数据，2022-09 ~ 2026-08） |

## 新任务流程

1. 在 `tasks/` 下建 `NNNN-<name>/`，写 `TASK.md`（假设/数据/规则/参数/验收标准）。
2. 写数据下载脚本（数据存 `/Volumes/trade/data`，优先复用已有数据）。
3. 写回测代码和 `notebook.py`（jupytext），同步执行出 `notebook.ipynb`。
4. 写 README（结果要点 + Colab badge）。

详见 [AGENTS.md](AGENTS.md)。
