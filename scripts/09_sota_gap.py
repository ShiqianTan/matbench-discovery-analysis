"""Three-axis SOTA gap.

The leaderboard splits into three (partly independent) axes:
  * Coupled axis  = mean(Discovery, Geometry, Phonons)
  * MD            = MD task score
  * Diatomics     = Diatomics task score

These figures show the top-k models on each axis and the overlap between them,
making the empty triple intersection (the SOTA opportunity) explicit.
"""
import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C

AXES = [("Coupled axis\n(Disc+Geo+Phon)", ["Discovery", "Geometry", "Phonons"]),
        ("MD", ["MD"]),
        ("Diatomics", ["Diatomics"])]
SINGLE = "#9AA4AD"          # models unique to one axis
MULTI = ["#4C72B0", "#C44E52", "#55A868", "#D8A21D", "#8172B3", "#6AA6C9"]


def axis_scores(L):
    return {name: L[cols].mean(axis=1) for name, cols in AXES}


def make_figure(L, k, fname):
    C.set_style()
    S = axis_scores(L)
    tops = {name: list(S[name].sort_values(ascending=False).head(k).index)
            for name, _ in AXES}

    # assign colours: multi-axis models get distinct colours
    counts = {}
    for name in tops:
        for m in tops[name]:
            counts[m] = counts.get(m, 0) + 1
    multi = [m for m in counts if counts[m] > 1]
    multi.sort(key=lambda m: -S["Coupled axis\n(Disc+Geo+Phon)"].get(m, 0))
    cmap = {m: MULTI[i % len(MULTI)] for i, m in enumerate(multi)}

    fig, ax = plt.subplots(figsize=(11.5, 1.35 * k + 3.2))
    xpos = {name: i for i, (name, _) in enumerate(AXES)}
    ypos = {name: {m: r for r, m in enumerate(tops[name])} for name, _ in AXES}

    for name, _ in AXES:
        x = xpos[name]
        ymax = k - 1
        ax.add_patch(plt.Rectangle((x - 0.42, -0.6), 0.84, ymax + 1.2,
                     facecolor="0.96", edgecolor="0.8", zorder=0))
        ax.text(x, -0.95, name, ha="center", va="bottom", fontsize=9.5,
                fontweight="bold")
        for m in tops[name]:
            r = ypos[name][m]
            sc = float(S[name][m])
            col = cmap.get(m, SINGLE)
            ax.barh(r, 0.80 * sc, left=x - 0.40, height=0.5, color=col,
                    zorder=3, edgecolor="white", lw=0.6)
            ax.text(x - 0.38, r, f"{m}", ha="left", va="center", fontsize=7.5,
                    zorder=4)
            ax.text(x + 0.40, r, f"{sc:.2f}", ha="right", va="center",
                    fontsize=7, color="0.35", zorder=4)

    # connectors for multi-axis models
    for m in multi:
        xs, ys = [], []
        for name, _ in AXES:
            if m in tops[name]:
                xs.append(xpos[name]); ys.append(ypos[name][m])
        ax.plot(xs, ys, "-", color=cmap[m], lw=1.6, alpha=0.75, zorder=1)

    # overlap statistics
    sets = {name: set(tops[name]) for name, _ in AXES}
    names = [n for n, _ in AXES]
    def ov(a, b): return sets[a] & sets[b]
    triple = sets[names[0]] & sets[names[1]] & sets[names[2]]
    info = (f"models in ≥2 axes: {len(multi)}   |   "
            f"Coupled∩MD: {len(ov(names[0],names[1]))}   "
            f"Coupled∩Diat: {len(ov(names[0],names[2]))}   "
            f"MD∩Diat: {len(ov(names[1],names[2]))}   |   "
            f"triple intersection: {len(triple)}")
    ax.set_xlim(-0.75, len(AXES) - 0.25)
    ax.set_ylim(k - 0.4, -1.5)
    ax.axis("off")
    ax.set_title(f"Three-axis leaderboard — top-{k} per axis\n{info}",
                 fontsize=11.5, pad=8)
    fig.tight_layout()
    C.save(fig, fname)

    # save membership
    recs = []
    for name, _ in AXES:
        for r, m in enumerate(tops[name]):
            recs.append({"axis": name.split("\n")[0], "rank": r + 1,
                         "model": m, "axis_score": float(S[name][m]),
                         "n_axes_top_k": counts[m]})
    pd.DataFrame(recs).to_csv(
        os.path.join(C.TAB, f"table9_three_axis_top{k}.csv"), index=False)
    return info, multi, triple


def main():
    L = importlib.import_module("03_model_landscape").build_landscape()
    for k, f in [(3, "fig13_three_axis_top3"), (5, "fig14_three_axis_top5")]:
        info, multi, triple = make_figure(L, k, f)
        print(f"top-{k}: {info}")


if __name__ == "__main__":
    main()
