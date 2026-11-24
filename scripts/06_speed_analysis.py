"""Speed analysis.

1. Correlation between the MD-task and Diatomics-task inference speeds.
2. Accuracy-vs-speed Pareto fronts for MD and Diatomics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import common as C

# accuracy components (Speed excluded) --------------------------------------
MD_ACC = ["ΔERMSE", "FRMSE", "ΔADF", "ΔvDOS", "PMAE", "PW1", "ΔP"]
DI_ACC = ["τ", "E flips", "E jump", "PBE Δr wall", "PBE E MAE", "PBE Δre",
          "PBE ΔDe", "PBE F MAE", "PBE Δω", "F flips", "F TV", "F jump"]


def norm_better(series: pd.Series, direction: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if direction < 0:                       # lower is better -> invert
        return (hi - s) / (hi - lo)
    return (s - lo) / (hi - lo)


def accuracy(df, cols):
    parts = [norm_better(df[c], C.DIRECTION[c]) for c in cols]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


def pareto_mask(x: np.ndarray, y: np.ndarray):
    """Non-dominated for (min x, max y)."""
    n = len(x)
    keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i]):
                keep[i] = False
                break
    return keep


def main():
    C.set_style()
    t = C.load_tasks()
    md, di = t["MD"], t["Diatomics"]
    master = t["Discovery"]

    mds = C.numeric(md[["Speed"]])["Speed"]
    dis = C.numeric(di[["Speed"]])["Speed"]
    idx = mds.dropna().index.intersection(dis.dropna().index)
    rho, p = spearmanr(mds.loc[idx], dis.loc[idx])

    md_acc = accuracy(md, MD_ACC)
    di_acc = accuracy(di, DI_ACC)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # ---- panel A: speed-speed correlation ---------------------------------
    ax = axes[0]
    ax.loglog(mds.loc[idx], dis.loc[idx], "o", ms=6, mfc="#4C72B0",
              mec="white", mew=0.6)
    ax.set_xlabel("MD speed (s, log)"); ax.set_ylabel("Diatomics speed (s, log)")
    ax.set_title(f"A  Speed correlation\nSpearman ρ = {rho:.2f}{C.star(p)} "
                 f"(n={len(idx)})", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)

    # ---- panel B/C: Pareto fronts -----------------------------------------
    for ax, sp, acc, name, cmap in [
            (axes[1], mds, md_acc, "MD", "Greens"),
            (axes[2], dis, di_acc, "Diatomics", "Purples")]:
        i = sp.dropna().index.intersection(acc.dropna().index)
        x = sp.loc[i].values; y = acc.loc[i].values
        keep = pareto_mask(x, y)
        ax.scatter(x, y, s=45, c="#B0B0B0", edgecolor="white", lw=0.5,
                   zorder=2, label="dominated")
        ax.scatter(x[keep], y[keep], s=70, c="#C44E52", edgecolor="white",
                   lw=0.7, zorder=3, label="Pareto-optimal")
        # front line
        ordx = np.argsort(x[keep])
        ax.plot(x[keep][ordx], y[keep][ordx], "--", color="#C44E52", lw=1,
                alpha=0.7, zorder=2)
        for k, label in enumerate(i):
            if keep[k]:
                ax.annotate(label, (x[k], y[k]), fontsize=6.5, xytext=(4, 3),
                            textcoords="offset points", zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel(f"{name} speed (s, log)")
        ax.set_ylabel(f"{name} accuracy (speed-free)")
        ax.set_title(f"{'B' if name=='MD' else 'C'}  {name}: accuracy vs speed",
                     fontsize=10)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)

    fig.suptitle("Speed: cross-task consistency and accuracy–speed Pareto fronts",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    C.save(fig, "fig9_speed_pareto")

    # tables
    tab = pd.DataFrame({
        "MD_Speed_s": mds, "Diatomics_Speed_s": dis,
        "MD_accuracy_nospeed": md_acc, "Diatomics_accuracy_nospeed": di_acc,
    })
    tab.to_csv(os.path.join(C.TAB, "table7_speed_analysis.csv"))

    # Pareto membership
    md_keep = pareto_mask(mds.dropna().values, md_acc.loc[mds.dropna().index].values)
    di_keep = pareto_mask(dis.dropna().values, di_acc.loc[dis.dropna().index].values)
    rec = {"MD Pareto-optimal": list(mds.dropna().index[md_keep]),
           "Diatomics Pareto-optimal": list(dis.dropna().index[di_keep])}
    with open(os.path.join(C.TAB, "table7_speed_pareto_members.txt"), "w") as f:
        for k, v in rec.items():
            f.write(f"{k}:\n  " + "\n  ".join(v) + "\n\n")
    print("MD Pareto:", rec["MD Pareto-optimal"])
    print("Diatomics Pareto:", rec["Diatomics Pareto-optimal"])
    print(f"speed rho={rho:.3f} p={p:.2g}")


if __name__ == "__main__":
    main()
