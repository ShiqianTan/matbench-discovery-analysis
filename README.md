# Matbench-Discovery downstream analysis

用于复现 Matbench-Discovery 五个下游任务的相关性、模型排名、速度—精度权衡、无速度综合指标（NSC）、GRACE 系列和三轴 SOTA gap 分析。

## Repository layout

```text
.
├── data/raw/          # 本地输入 CSV（不提交到 Git）
├── docs/analysis.md   # 完整分析结论与图表说明
├── scripts/           # 01–09 分析/可视化脚本、共享工具和统一入口
└── outputs/           # 运行后生成的图表与表格（默认不纳入 Git）
```

## Quick start

需要 Python 3.10 或更高版本；先用 `python --version` 确认当前解释器。

运行前，将 `data/README.md` 列出的 5 份 Matbench-Discovery CSV 放入
`data/raw/`。这些数据文件默认由 Git 忽略。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run_all.py
```

输出目录：

- `outputs/figures/`：PNG（300 dpi）和 PDF 图表
- `outputs/tables/`：CSV、Markdown、LaTeX 表格
- `outputs/grace/`：GRACE 模型系列的独立分析

只运行部分步骤：

```bash
python scripts/run_all.py --only 01 03 09
```

也可以将输入和输出放在仓库外：

```bash
python scripts/run_all.py \
  --data-dir /path/to/data \
  --output-dir /path/to/results
```

## Analysis pipeline

| Step | Script | Purpose |
|---:|---|---|
| 01 | `01_within_task_corr.py` | 各任务内部指标相关性 |
| 02 | `02_cross_task_corr.py` | 跨任务与跨指标相关性 |
| 03 | `03_model_landscape.py` | 模型表现热图、排名、嵌入与参数量关系 |
| 04 | `04_tables.py` | 描述统计与汇总表格 |
| 05 | `05_composite_anatomy.py` | CPS / CMDS / CDS 构成分析 |
| 06 | `06_speed_analysis.py` | 跨任务速度相关性与 Pareto 前沿 |
| 07 | `07_new_composite.py` | No-Speed Composite（NSC）及重排名 |
| 08 | `08_grace_family.py` | GRACE 模型系列专项分析 |
| 09 | `09_sota_gap.py` | Coupled / MD / Diatomics 三轴 SOTA gap |

每个编号脚本也可单独运行，例如：

```bash
python scripts/06_speed_analysis.py
```

## Before publishing on GitHub

- 当前未指定开源许可证；公开前请根据代码归属添加合适的 `LICENSE`。
- `data/raw/` 默认不提交；公开再分发前请确认上游数据许可和引用要求。
- 生成结果被 `.gitignore` 排除。如需发布固定版本图表，建议通过 GitHub Release 或单独的归档目录提供。

详细方法、结论和图表索引见 [`docs/analysis.md`](docs/analysis.md)。
