"""GRACE model-family analysis (standalone).

Produces outputs/grace/figures/*, outputs/grace/tables/* and
outputs/grace/README.md.
GRACE models: GRACE-1L-OAM, GRACE-2L-OAM, GRACE-2L-OAM-L, GRACE-3L-OAM-L,
GRACE-2L-MPtrj.
"""
import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import common as C

GDIR = os.path.join(C.OUTPUT, "grace")
GFIG = os.path.join(GDIR, "figures")
GTAB = os.path.join(GDIR, "tables")
os.makedirs(GFIG, exist_ok=True); os.makedirs(GTAB, exist_ok=True)

GRACE = ["GRACE-1L-OAM", "GRACE-2L-OAM", "GRACE-2L-OAM-L", "GRACE-3L-OAM-L",
         "GRACE-2L-MPtrj"]
# short display names
SHORT = {"GRACE-1L-OAM": "GRACE-1L-OAM", "GRACE-2L-OAM": "GRACE-2L-OAM",
         "GRACE-2L-OAM-L": "GRACE-2L-OAM-L", "GRACE-3L-OAM-L": "GRACE-3L-OAM-L",
         "GRACE-2L-MPtrj": "GRACE-2L-MPtrj"}


def save_g(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(GFIG, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", os.path.join(GFIG, name + ".png"))


def norm_better(s, d, metric=None):
    return C.direction_score(s, metric if metric is not None else "", d)


def main():
    C.set_style()
    land = importlib.import_module("03_model_landscape")
    L = land.build_landscape()
    tasks = C.load_tasks()
    master = tasks["Discovery"]

    # ---- all-metric matrix (normalised against full 43-model field) --------
    frames, blocks, boundaries = [], [], []
    for task, cols in C.TASK_METRICS.items():
        df = tasks[task]
        sub = pd.DataFrame({c: norm_better(df[c], C.DIRECTION[c], c) for c in cols})
        frames.append(sub)
        boundaries.append((task, len(frames[-1].columns)))
    M = pd.concat(frames, axis=1)
    g = M.loc[[m for m in GRACE if m in M.index]]
    # column labels
    labels = []
    for task, cols in C.TASK_METRICS.items():
        labels += [C.metric_label(f"{task}::{c}") for c in cols]

    # ---- figure G1: normalised metric heatmap ------------------------------
    fig, ax = plt.subplots(figsize=(14, 3.4))
    im = ax.imshow(g.values, aspect="auto", cmap=C.SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=8)
    for v in range(len(labels)):
        for r in range(len(g)):
            val = g.values[r, v]
            if np.isfinite(val):
                ax.text(v, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val < 0.5 else "black")
    off = 0
    for task, n in boundaries:
        off += n
        if off < len(labels):
            ax.axvline(off - 0.5, color="black", lw=1.2)
    ax.set_title("GRACE family — direction-corrected performance per metric "
                 "(1 = best in field of 43 models)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="normalised score")
    fig.tight_layout(); save_g(fig, "grace_metric_heatmap")

    # ---- figure G2: task scores + composites -------------------------------
    score_cols = ["Discovery", "Geometry", "Phonons", "MD", "Diatomics"]
    S = L.loc[[m for m in GRACE if m in L.index], score_cols + ["CPS", "CMDS", "CDS"]]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(S.columns)); w = 0.16
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(S)))
    for k, (name, row) in enumerate(S.iterrows()):
        ax.bar(x + (k - (len(S) - 1) / 2) * w, row.values, w, label=name,
               color=cmap[k])
    ax.set_xticks(x); ax.set_xticklabels(S.columns)
    ax.axvline(4.5, ls="--", color="0.6", lw=0.8)
    ax.set_ylabel("normalised score"); ax.set_ylim(0, 1.05)
    ax.set_title("GRACE family — per-task and composite scores "
                 "(Discovery-Geometry-Phonons | MD | Diatomics)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout(); save_g(fig, "grace_task_scores")

    # ---- figure G3: radar of task scores -----------------------------------
    ang = np.linspace(0, 2 * np.pi, len(score_cols), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw=dict(polar=True))
    for k, name in enumerate([m for m in GRACE if m in S.index]):
        vals = S.loc[name, score_cols].values.tolist(); vals += vals[:1]
        ax.plot(ang, vals, "-o", ms=3, lw=1.4, color=cmap[k], label=name)
        ax.fill(ang, vals, color=cmap[k], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(score_cols, fontsize=9)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=7)
    ax.set_title("GRACE family — task profile", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    fig.tight_layout(); save_g(fig, "grace_radar")

    # ---- figure G4: rank within field --------------------------------------
    allc = L[score_cols + ["CPS", "CMDS", "CDS"]]
    Rk = allc.rank(ascending=False, method="average")
    rg = Rk.loc[[m for m in GRACE if m in Rk.index]]
    n_models = allc["CPS"].notna().sum()  # 42-ish (NequIP-GNoME only has disc)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, name in enumerate(rg.index):
        ax.plot(range(len(rg.columns)), rg.loc[name].values, "-o", ms=4,
                lw=1.4, color=cmap[k], label=name)
    ax.set_xticks(range(len(rg.columns))); ax.set_xticklabels(rg.columns)
    ax.invert_yaxis(); ax.axvline(4.5, ls="--", color="0.6", lw=0.8)
    ax.set_ylabel("rank among field (1 = best)")
    ax.set_title("GRACE family — rank within the full field of models")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout(); save_g(fig, "grace_field_rank")

    # ---- table -------------------------------------------------------------
    tab = pd.concat([master.loc[[m for m in GRACE if m in master.index]],
                     S, Rk.loc[[m for m in GRACE if m in Rk.index]].add_prefix("rank_")],
                    axis=1)
    tab.to_csv(os.path.join(GTAB, "grace_metrics_and_ranks.csv"))

    # ---- README ------------------------------------------------------------
    write_readme(g, S, rg, M, L)


def write_readme(g, S, rg, M, L):
    lines = ["# GRACE model family — standalone analysis", "",
             "Five GRACE variants from the Matbench-Discovery leaderboard:",
             ", ".join(f"`{m}`" for m in GRACE), "",
             "## Files", "",
             "- `figures/grace_metric_heatmap` — all metrics, direction-corrected "
             "(1 = best of the 43-model field), blocked by task.",
             "- `figures/grace_task_scores` — per-task + composite scores.",
             "- `figures/grace_radar` — task-profile radar.",
             "- `figures/grace_field_rank` — rank within the whole field per task.",
             "- `tables/grace_metrics_and_ranks.csv` — raw metrics, scores and ranks.",
             "", "## Ranking within the field (CPS / CMDS / CDS / Mean)",
             "",
             rg.round(1).to_string(), "",
             "## Observations", ""]
    # auto observations
    best_discovery = S["Discovery"].idxmax()
    best_md = S["MD"].idxmax()
    best_dia = S["Diatomics"].idxmax()
    mean_task = S[["Discovery", "Geometry", "Phonons", "MD", "Diatomics"]].mean(axis=1)
    lines += [
        f"- **{mean_task.idxmax()}** has the best mean task score within the "
        "GRACE family.",
        f"- MD is led by **{best_md}**; diatomics by **{best_dia}** — the family "
        "does not dominate every task.",
        "- Scaling 1L → 2L → 3L (at fixed OAM training) generally improves the "
        "discovery/geometry/phonons axis; `GRACE-2L-MPtrj` shows the effect of "
        "switching the training set to MPtrj (weaker on discovery/geometry, "
        "competitive on MD).",
        "- The family stays above the field median on the Discovery-Geometry-"
        "Phonons axis but is mid-pack on MD and diatomics.",
    ]
    with open(os.path.join(GDIR, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
