"""Figures 3-4 - model landscape: score heatmap, rankings, params, embedding."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common as C
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist

HEADLINE = {
    "Discovery": ("F1", +1), "Geometry": ("RMSD", -1),
    "Phonons": ("κSRME", -1), "MD": ("CMDS", +1), "Diatomics": ("CDS", +1),
}


def minmax_dir(s, direction):
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    x = pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)
    return 1.0 - x if direction < 0 else x


def build_landscape():
    tasks = C.load_tasks()
    master = tasks["Discovery"]
    df = pd.DataFrame(index=master.index)
    for task, (metric, d) in HEADLINE.items():
        src = tasks[task] if metric in tasks[task].columns else master
        df[task] = minmax_dir(C.numeric(src[[metric]])[metric], d)
    for c in ["CPS", "CMDS", "CDS"]:
        df[c] = C.numeric(master[[c]])[c]
    df["Mean(task)"] = df[list(HEADLINE)].mean(axis=1)
    df["Params_M"] = C.params_to_millions(master["Params"])
    return df


def fig_heatmap(L):
    cols = list(HEADLINE) + ["Mean(task)", "CPS", "CMDS", "CDS"]
    order = L.sort_values("Mean(task)", ascending=False).index
    A = L.loc[order, cols]
    fig, ax = plt.subplots(figsize=(6.6, 0.30 * len(A) + 2.4))
    im = ax.imshow(A.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(A))); ax.set_yticklabels(order, fontsize=7)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(A), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for i in range(len(A)):
        for j in range(len(cols)):
            v = A.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("score (higher = better; composite scores raw)", fontsize=8)
    ax.set_title("Model performance landscape across tasks", pad=10)
    C.save(fig, "fig3_model_landscape_heatmap")
    return A


def fig_rankings(L):
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.30 * len(L) + 2))
    for ax, metric in zip(axes, ["CPS", "CMDS", "CDS"]):
        s = L[metric].dropna().sort_values()
        colors = plt.cm.viridis((s.values - s.min()) / (s.max() - s.min() + 1e-9))
        ax.barh(s.index, s.values, color=colors)
        ax.set_title(metric)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.suptitle("Model rankings by composite score", y=1.01, fontsize=12)
    fig.tight_layout()
    C.save(fig, "fig4_composite_rankings")


def fig_params_scatter(L):
    d = L.dropna(subset=["Params_M", "Mean(task)"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    ax.scatter(d["Params_M"], d["Mean(task)"], s=42, color="#4C72B0",
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Mean normalized task score")
    from scipy.stats import spearmanr
    r, p = spearmanr(d["Params_M"], d["Mean(task)"])
    ax.set_title(f"Size vs. mean score (Spearman ρ={r:.2f}{C.star(p)})")
    for name, row in d.iterrows():
        if row["Mean(task)"] > d["Mean(task)"].quantile(0.8) or row["Params_M"] > 300:
            ax.annotate(name, (row["Params_M"], row["Mean(task)"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax = axes[1]
    ax.scatter(d["Params_M"], d["CPS"], s=30, label="CPS", alpha=0.8)
    ax.scatter(d["Params_M"], d["CMDS"], s=30, label="CMDS", alpha=0.8)
    ax.scatter(d["Params_M"], d["CDS"], s=30, label="CDS", alpha=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Composite score"); ax.legend(title="Task")
    ax.set_title("Size vs. per-composite score")
    fig.tight_layout()
    C.save(fig, "fig5_params_vs_performance")


def fig_embedding(L):
    cols = list(HEADLINE)
    X = L[cols].dropna()
    Z = (X - X.mean()) / X.std(ddof=0)
    # PCA
    pca = PCA(n_components=2).fit(Z.values)
    P = pca.transform(Z.values)
    # MDS on Spearman-distance is complex; use Euclidean on z-scores
    mds = MDS(n_components=2, random_state=0, normalized_stress="auto").fit_transform(Z.values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    cmap = plt.cm.viridis((L.loc[Z.index, "Mean(task)"]))
    ms = L.loc[Z.index, "Mean(task)"]
    hi, lo = ms.quantile(0.82), ms.quantile(0.18)
    for ax, emb, title in [(axes[0], P, f"PCA (PC1 {pca.explained_variance_ratio_[0]:.0%}, PC2 {pca.explained_variance_ratio_[1]:.0%})"),
                           (axes[1], mds, "MDS of normalized task scores")]:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=cmap, s=55, edgecolor="white", linewidth=0.5, zorder=3)
        for i, name in enumerate(Z.index):
            if ms.iloc[i] >= hi or ms.iloc[i] <= lo:
                ax.annotate(name, (emb[i, 0], emb[i, 1]), fontsize=6.5,
                            xytext=(4, 3), textcoords="offset points", zorder=4,
                            bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.6))
        ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, label="Mean task score")
    C.save(fig, "fig6_model_embedding")
    load = pd.DataFrame(pca.components_.T, index=cols, columns=["PC1", "PC2"])
    load.to_csv(os.path.join(C.TAB, "table_pca_loadings.csv"))
    return load


def fig_parallel(L, topn=12):
    cols = list(HEADLINE) + ["CPS", "CMDS", "CDS"]
    order = L.sort_values("Mean(task)", ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    xs = np.arange(len(cols))
    cmap = plt.cm.turbo(np.linspace(0, 0.9, len(order)))
    for (name, row), col in zip(order.iterrows(), cmap):
        ax.plot(xs, row[cols].values.astype(float), marker="o", ms=4, lw=1.4,
                color=col, label=name)
    ax.set_xticks(xs); ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(fontsize=6.5, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"Top {topn} models across task scores", pad=8)
    C.save(fig, "fig7_top_models_parallel")


def main():
    C.set_style()
    L = build_landscape()
    L.sort_values("Mean(task)", ascending=False).to_csv(
        os.path.join(C.TAB, "table_model_scores_normalized.csv"))
    A = fig_heatmap(L)
    fig_rankings(L)
    fig_params_scatter(L)
    fig_embedding(L)
    fig_parallel(L)
    print("done; models:", len(L))


if __name__ == "__main__":
    main()
