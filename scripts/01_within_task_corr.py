"""Figure 1 - within-task metric correlations (Spearman, clustered)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C


def main():
    C.set_style()
    tasks = C.load_tasks()
    summary = {}
    figs = []
    for task, cols in C.TASK_METRICS.items():
        df = C.numeric(tasks[task][cols])
        corr, pval, n = C.corr_with_p(df, method="spearman")
        order = C.cluster_order(corr)
        fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 2.6, 1.15 * len(cols) + 1.6))
        C.draw_corr_heatmap(ax, corr, pval, order=order,
                            title=f"{task}: within-task metric correlation (Spearman)")
        C.save(fig, f"fig1_{task.lower()}_within_corr")
        # store
        rc = corr.iloc[order, order]
        off = rc.values[np.triu_indices_from(rc.values, k=1)]
        summary[task] = pd.Series(off)
        figs.append((task, corr, pval, order))

    # Combined overview: average |rho| within each task + strongest pairs
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
    for ax, (task, corr, pval, order) in zip(axes, figs):
        C.draw_corr_heatmap(ax, corr, pval, order=order, cbar=False,
                            title=task, label_fs=7)
    fig.suptitle("Within-task metric correlation (Spearman)", y=1.02, fontsize=12)
    fig.tight_layout()
    C.save(fig, "fig1_overview_within_corr")

    # save numeric matrices
    os.makedirs(C.TAB, exist_ok=True)
    for task, corr, pval, order in figs:
        corr.to_csv(os.path.join(C.TAB, f"table_within_corr_{task.lower()}.csv"))
        pval.to_csv(os.path.join(C.TAB, f"table_within_pval_{task.lower()}.csv"))


if __name__ == "__main__":
    main()
