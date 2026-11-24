"""Figure 8 - composite-score anatomy.

How strongly does each composite score (CPS, CMDS, CDS) track its own raw
component metrics? Bars show direction-corrected Spearman rho (positive =
component drives the composite toward 'better').
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C
from scipy.stats import spearmanr

COMPONENTS = {
    "CPS": [("Discovery::F1", +1), ("Geometry::RMSD", -1), ("Phonons::κSRME", -1)],
    "CMDS": [(f"MD::{c}", -1) for c in
             ["ΔERMSE", "FRMSE", "ΔADF", "ΔvDOS", "PMAE", "PW1", "ΔP", "Speed"]],
    "CDS": [(f"Diatomics::{c}", -1) for c in
            ["PBE E MAE", "PBE F MAE", "PBE Δr wall", "PBE Δre", "PBE ΔDe",
             "E jump", "F flips", "τ", "Speed"]],
}
COMPOSITE_SOURCE = {"CPS": "Discovery", "CMDS": "Discovery", "CDS": "Discovery"}


def main():
    C.set_style()
    M = C.combined_metrics()
    tasks = C.load_tasks()
    master = tasks["Discovery"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2),
                             gridspec_kw={"width_ratios": [3, 8, 9]})
    for ax, comp in zip(axes, ["CPS", "CMDS", "CDS"]):
        y = C.numeric(master[[comp]])[comp].dropna()
        labels, vals, cols = [], [], []
        for col, d in COMPONENTS[comp]:
            x = M[col]
            idx = x.dropna().index.intersection(y.index)
            if len(idx) < 4:
                continue
            r, _ = spearmanr(x.loc[idx], y.loc[idx])
            labels.append(C.metric_label(col))
            vals.append(d * r)          # >0  => higher component -> higher score
            cols.append("#4C72B0" if d * r >= 0 else "#C44E52")
        order = np.argsort(vals)
        labels = [labels[i] for i in order]
        vals = np.array(vals)[order]
        cols = [cols[i] for i in order]
        ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ with composite (direction-corrected)")
        ax.set_title(f"{comp}", fontsize=11)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.suptitle("Composite-score anatomy: how well each composite tracks its raw components",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    C.save(fig, "fig8_composite_anatomy")

    # save numbers
    recs = []
    for comp in COMPONENTS:
        y = C.numeric(master[[comp]])[comp].dropna()
        for col, d in COMPONENTS[comp]:
            x = M[col]
            idx = x.dropna().index.intersection(y.index)
            r, p = spearmanr(x.loc[idx], y.loc[idx])
            recs.append({"Composite": comp, "Component": col, "rho": r,
                         "direction_corrected_rho": d * r, "p": p, "n": len(idx)})
    pd.DataFrame(recs).to_csv(os.path.join(C.TAB, "table6_composite_anatomy.csv"), index=False)


if __name__ == "__main__":
    main()
