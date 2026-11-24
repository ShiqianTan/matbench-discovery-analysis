"""A new composite index with Speed removed.

Components (6 metrics), each direction-corrected to [0,1] and averaged with
equal weights:

    Discovery  F1            (+)
    Geometry   RMSD          (-)
    Phonons    kappa_SRME    (-)
    MD         Delta ADF     (-)
    Diatomics  PBE E MAE     (-)
    Diatomics  PBE F MAE     (-)

These are the metrics that correlate most strongly with the existing CPS
(F1 / RMSD / kappa_SRME), CMDS (Delta ADF) and CDS (PBE E/F MAE), with the
Speed contributions of CMDS/CDS removed.
"""
import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import common as C

COMPONENTS = [
    ("Discovery", "F1", +1),
    ("Geometry", "RMSD", -1),
    ("Phonons", "κSRME", -1),
    ("MD", "ΔADF", -1),
    ("Diatomics", "PBE E MAE", -1),
    ("Diatomics", "PBE F MAE", -1),
]
SHORT = ["F1", "RMSD", r"$\kappa_{\rm SRME}$", r"$\Delta$ADF",
         "PBE E MAE", "PBE F MAE"]


def minmax_better(s: pd.Series, direction: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    return (hi - s) / (hi - lo) if direction < 0 else (s - lo) / (hi - lo)


def build():
    tasks = C.load_tasks()
    parts = {}
    for task, metric, d in COMPONENTS:
        col = f"{task}::{metric}"
        parts[col] = minmax_better(tasks[task][metric], d)
    S = pd.DataFrame(parts)
    n_valid = S.notna().sum(axis=1)
    NSC = S.mean(axis=1)                       # mean over available
    NSC[n_valid < len(COMPONENTS)] = np.nan     # require all six components
    master = tasks["Discovery"]
    out = pd.DataFrame({
        **{c: S[c] for c in S.columns},
        "NSC": NSC, "n_components": n_valid,
        "CPS": pd.to_numeric(master["CPS"], errors="coerce"),
        "CMDS": pd.to_numeric(master["CMDS"], errors="coerce"),
        "CDS": pd.to_numeric(master["CDS"], errors="coerce"),
    })
    return out, master


def ranks(series: pd.Series) -> pd.Series:
    return series.rank(ascending=False, method="average")


def main():
    C.set_style()
    land = importlib.import_module("03_model_landscape")
    L = land.build_landscape()
    out, master = build()
    NSC = out["NSC"]

    # ------- comparison frame ------------------------------------------------
    comp = pd.DataFrame({"NSC": NSC})
    comp["CPS"] = out["CPS"]; comp["CMDS"] = out["CMDS"]; comp["CDS"] = out["CDS"]
    comp["Mean(task)"] = L["Mean(task)"]
    for task in ["Discovery", "Geometry", "Phonons", "MD", "Diatomics"]:
        comp[f"TT_{task}"] = L[task]

    R = comp.apply(ranks)
    R = R.add_prefix("rank_")

    # Spearman of NSC with every other score
    rows = []
    for c in comp.columns:
        if c == "NSC":
            continue
        i = comp[["NSC", c]].dropna().index
        r, p = spearmanr(comp.loc[i, "NSC"], comp.loc[i, c])
        rows.append({"vs": c, "spearman": r, "p": p, "n": len(i)})
    cmp_tab = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    cmp_tab.to_csv(os.path.join(C.TAB, "table8b_NSC_rank_corr.csv"), index=False)

    # ------- ranking table ---------------------------------------------------
    full = pd.concat([out, R], axis=1)
    full = full.sort_values("NSC", ascending=False)
    full.index.name = "Model"
    full.round(4).to_csv(os.path.join(C.TAB, "table8_NSC_ranking.csv"))
    try:
        full.round(3).to_latex(os.path.join(C.TAB, "table8_NSC_ranking.tex"),
                               caption="No-Speed Composite (NSC) ranking and rank "
                                       "comparison against existing scores.",
                               label="tab:nsc")
    except Exception as e:
        print("tex skip", e)

    # ------- figure 10: stacked component + NSC ranking ---------------------
    cols = [f"{t}::{m}" for t, m, _ in COMPONENTS]
    order = NSC.sort_values(ascending=False).index
    ord2 = [m for m in order if not np.isnan(NSC[m])]
    fig, ax = plt.subplots(figsize=(9, 11))
    M = out.loc[ord2, cols]
    bottom = np.zeros(len(ord2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))
    for c, lab, col in zip(cols, SHORT, colors):
        ax.barh(range(len(ord2)), M[c].values / len(cols), left=bottom,
                color=col, label=lab, height=0.82)
        bottom += M[c].values / len(cols)
    ax.set_yticks(range(len(ord2))); ax.set_yticklabels(ord2, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("NSC (mean of six direction-corrected component scores)")
    ax.set_title("No-Speed Composite (NSC): ranking and component contributions")
    ax.set_xlim(0, 1); ax.legend(ncol=3, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.tight_layout()
    C.save(fig, "fig10_NSC_ranking_stacked")

    # ------- figure 11: rank comparison (scatter small-multiples) -----------
    rcols = ["rank_NSC", "rank_CPS", "rank_CMDS", "rank_CDS", "rank_Mean(task)"]
    others = [("rank_CPS", "CPS"), ("rank_CMDS", "CMDS"), ("rank_CDS", "CDS"),
              ("rank_Mean(task)", "Mean (task)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharex=True, sharey=True)
    for ax, (rc, name) in zip(axes, others):
        i = R[["rank_NSC", rc]].dropna().index
        xr, yr = R.loc[i, "rank_NSC"], R.loc[i, rc]
        ax.scatter(xr, yr, s=40, c="#4C72B0", edgecolor="white", lw=0.5,
                   zorder=3)
        lim = [0, R["rank_NSC"].max() + 2]
        ax.plot(lim, lim, "--", color="0.5", lw=0.8, zorder=1)
        r, p = spearmanr(xr, yr)
        d = (yr - xr)                    # >0 => NSC ranks the model better
        for m in d.abs().sort_values(ascending=False).index[:6]:
            ax.annotate(f"{m} ({d[m]:+.0f})", (xr[m], yr[m]), fontsize=6,
                        xytext=(4, 3), textcoords="offset points", zorder=4,
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.4))
        ax.set_title(f"NSC vs {name}\nρ = {r:.2f}{C.star(p)}", fontsize=10)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("NSC rank")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("existing rank")
    fig.suptitle("NSC ranking vs existing rankings (labels = rank change of NSC "
                 "relative to the existing ranking)", y=1.04, fontsize=12)
    fig.tight_layout()
    C.save(fig, "fig11_NSC_rank_comparison")

    # ------- figure 12: NSC vs Mean(task) with rank shifts -------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))
    i = comp[["NSC", "Mean(task)"]].dropna().index
    ax.scatter(comp.loc[i, "Mean(task)"], comp.loc[i, "NSC"], s=45,
               c="#4C72B0", edgecolor="white", lw=0.5, zorder=3)
    d = (R.loc[i, "rank_Mean(task)"] - R.loc[i, "rank_NSC"])  # >0 => NSC better
    for m in d.abs().sort_values(ascending=False).index[:12]:
        ax.annotate(f"{m} ({d[m]:+.0f})", (comp.loc[m, "Mean(task)"],
                    comp.loc[m, "NSC"]), fontsize=6.5, xytext=(4, 3),
                    textcoords="offset points",
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.5))
    r, p = spearmanr(comp.loc[i, "Mean(task)"], comp.loc[i, "NSC"])
    ax.set_xlabel("Mean task score (existing)"); ax.set_ylabel("NSC (no speed)")
    ax.set_title(f"NSC vs existing mean score\nSpearman ρ = {r:.2f}{C.star(p)} "
                 "(labels show rank change vs mean score)")
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    C.save(fig, "fig12_NSC_vs_mean")

    print(cmp_tab.to_string(index=False))
    print("\nTop 10 NSC:")
    print(NSC.sort_values(ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
