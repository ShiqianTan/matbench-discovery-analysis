"""Figure 2 - cross-task correlations.

Views:
  (a) task-to-task correlation of per-task headline scores (direction-corrected)
  (a2) scatter matrix of those task scores
  (a3) composite scores CPS / CMDS / CDS
  (b) full blocked metric-metric correlation across all tasks (direction-corrected)
  (b2) task-block correlation summary
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C

HEADLINE = {           # official per-task headline metric
    "Discovery": ("F1", +1),
    "Geometry": ("RMSD", -1),
    "Phonons": ("κSRME", -1),
    "MD": ("CMDS", +1),
    "Diatomics": ("CDS", +1),
}


def minmax_dir(s: pd.Series, direction: int) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    x = pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)
    return 1.0 - x if direction < 0 else x


def task_scores():
    tasks = C.load_tasks()
    cols = {}
    for task, (metric, d) in HEADLINE.items():
        src = tasks[task] if metric in tasks[task].columns else tasks["Discovery"]
        cols[task] = minmax_dir(C.numeric(src[[metric]])[metric], d)
    return pd.DataFrame(cols)


def dir_corrected_matrix():
    M = C.combined_metrics()
    out = {}
    for c in M.columns:
        metric = c.split("::")[-1]
        d = C.DIRECTION.get(metric, +1)
        out[c] = C.direction_score(M[c], metric, d)
    return pd.DataFrame(out)


def main():
    C.set_style()
    tasks = C.load_tasks()
    master = tasks["Discovery"]

    # ---- (a) task-to-task correlation -----------------------------------
    T = task_scores()
    T.to_csv(os.path.join(C.TAB, "table_task_scores_normalized.csv"))
    corr2, pval2, _ = C.corr_with_p(T, method="spearman")
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    C.draw_corr_heatmap(ax, corr2, pval2, annot=True, label_fs=10,
                        title="Task-to-task correlation (normalized headline scores)")
    C.save(fig, "fig2a_task_scores_corr")
    corr2.to_csv(os.path.join(C.TAB, "table_task_scores_corr.csv"))
    pval2.to_csv(os.path.join(C.TAB, "table_task_scores_pval.csv"))

    # ---- (a2) scatter matrix --------------------------------------------
    names = list(T.columns)
    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(2.35 * k, 2.35 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(T[names[i]], bins=12, color="#4C72B0", alpha=0.85)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(T[names[j]], T[names[i]], s=22, alpha=0.8,
                           edgecolor="white", linewidth=0.4, color="#C44E52")
                r = corr2.loc[names[i], names[j]]
                ax.text(0.05, 0.95, f"ρ={r:.2f}", transform=ax.transAxes,
                        va="top", ha="left", fontsize=9,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
            else:
                ax.axis("off")
            if i == k - 1:
                ax.set_xlabel(names[j], fontsize=9)
            if j == 0 and i > 0:
                ax.set_ylabel(names[i], fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("Pairwise relationships between per-task normalized scores", y=0.995, fontsize=12)
    fig.tight_layout()
    C.save(fig, "fig2b_task_scores_scatter")

    # ---- (a3) composite scores ------------------------------------------
    comp = C.numeric(master[["CPS", "CMDS", "CDS"]])
    corrC, pvalC, _ = C.corr_with_p(comp, method="spearman")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    C.draw_corr_heatmap(ax, corrC, pvalC, annot=True, label_fs=10,
                        title="Composite-score correlation (CPS / CMDS / CDS)")
    C.save(fig, "fig2c_composite_corr")
    corrC.to_csv(os.path.join(C.TAB, "table_composite_corr.csv"))

    # ---- (b) full blocked, direction-corrected metric correlation -------
    M = dir_corrected_matrix()
    block_cols, bounds = [], []
    for task, cols in C.TASK_METRICS.items():
        for c in cols:
            key = f"{task}::{c}"
            if key in M.columns:
                block_cols.append(key)
        bounds.append(len(block_cols))
    M = M[block_cols]
    corrF, pvalF, _ = C.corr_with_p(M, method="spearman")
    n = len(block_cols)
    fig, ax = plt.subplots(figsize=(0.30 * n + 4, 0.30 * n + 3))
    C.draw_corr_heatmap(ax, corrF, pvalF, annot=False, cbar=True, tri=None)
    ax.set_xticklabels([C.metric_label(x) for x in block_cols], rotation=90, fontsize=6)
    ax.set_yticklabels([C.metric_label(x) for x in block_cols], fontsize=6)
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="black", lw=1.2)
        ax.axvline(b - 0.5, color="black", lw=1.2)
    ax.set_title("Cross-task metric-metric correlation (direction-corrected, Spearman)", pad=10)
    C.save(fig, "fig2d_full_metric_corr")
    corrF.to_csv(os.path.join(C.TAB, "table_cross_task_full_corr.csv"))

    # ---- (b2) block summary ---------------------------------------------
    task_of = np.concatenate([[t] * len([c for c in cols if f"{t}::{c}" in block_cols])
                              for t, cols in C.TASK_METRICS.items()])
    tnames = list(C.TASK_METRICS)
    B = pd.DataFrame(np.nan, index=tnames, columns=tnames, dtype=float)
    for t1 in tnames:
        for t2 in tnames:
            i = np.where(task_of == t1)[0]
            j = np.where(task_of == t2)[0]
            vals = corrF.values[np.ix_(i, j)]
            if t1 == t2:
                vals = vals[np.triu_indices_from(vals, k=1)]
            B.loc[t1, t2] = np.nanmean(vals)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(B.values, cmap=C.corr_colormap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(tnames))); ax.set_yticks(range(len(tnames)))
    ax.set_xticklabels(tnames, rotation=30, ha="right"); ax.set_yticklabels(tnames)
    for i in range(len(tnames)):
        for j in range(len(tnames)):
            v = B.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.6 else "black")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean Spearman ρ (direction-corrected)", fontsize=8)
    ax.set_title("Task-block correlation")
    C.save(fig, "fig2e_task_block_corr")
    B.to_csv(os.path.join(C.TAB, "table_cross_task_block_corr.csv"))

    print("\nTask-score corr:\n", corr2.round(2))
    print("\nComposite corr:\n", corrC.round(2))
    print("\nBlock corr:\n", B.round(2))
    print("\nTask scores:\n", T.round(2))


if __name__ == "__main__":
    main()
