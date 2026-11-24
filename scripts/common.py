"""Shared data loading and plotting helpers for the analysis pipeline.

Paths default to this repository's ``data/raw`` and ``outputs`` directories.
They can be overridden with ``MATBENCH_DATA_DIR`` and
``MATBENCH_OUTPUT_DIR`` for CI, testing, or external data storage.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MATBENCH_PROJECT_ROOT", os.path.dirname(HERE))
DATA = os.environ.get("MATBENCH_DATA_DIR", os.path.join(ROOT, "data", "raw"))
OUTPUT = os.environ.get("MATBENCH_OUTPUT_DIR", os.path.join(ROOT, "outputs"))
FIG = os.path.join(OUTPUT, "figures")
TAB = os.path.join(OUTPUT, "tables")

MASTER = os.path.join(DATA, "matbench-discovery-unique_prototypes.csv")
TASK_FILES = {
    "Discovery": MASTER,
    "Geometry": os.path.join(DATA, "matbench-discovery-unique_prototypes_geo-opt.csv"),
    "Phonons": os.path.join(DATA, "matbench-discovery-unique_prototypes-phonons.csv"),
    "MD": os.path.join(DATA, "matbench-discovery-unique_prototypes-md.csv"),
    "Diatomics": os.path.join(DATA, "matbench-discovery-unique_prototypes-diatomics.csv"),
}

# ---------------------------------------------------------------- metric groups
# Only intrinsic performance metrics (metadata / links / targets excluded).
TASK_METRICS = {
    "Discovery": ["Acc", "Prec", "F1", "DAF", "MAE", "R2"],
    "Geometry": [
        "Σ= 10-2", "Σ↓ 10-2", "Σ↑ 10-2",
        "Σ= 10-5", "Σ↓ 10-5", "Σ↑ 10-5", "RMSD",
    ],
    "Phonons": ["κSRME", "κSRE", "κSRD", "κ failed", "Im(ω)", "W1(ω)"],
    "MD": ["ΔERMSE", "FRMSE", "ΔADF", "ΔvDOS", "PMAE", "PW1", "ΔP", "Speed"],
    "Diatomics": [
        "τ", "E flips", "E jump", "PBE Δr wall", "PBE E MAE", "PBE Δre",
        "PBE ΔDe", "PBE F MAE", "PBE Δω", "F flips", "F TV", "F jump", "Speed",
    ],
}

# Direction of each metric from the "better" perspective (+1 higher-better,
# -1 lower-better, 0 close-to-zero-better).
DIRECTION = {
    "Acc": 1, "Prec": 1, "F1": 1, "DAF": 1, "MAE": -1, "R2": 1,
    "Σ= 10-2": 1, "Σ↓ 10-2": -1, "Σ↑ 10-2": 1,
    "Σ= 10-5": 1, "Σ↓ 10-5": -1, "Σ↑ 10-5": 1, "RMSD": -1,
    "κSRME": -1, "κSRE": -1, "κSRD": -1, "κ failed": -1, "Im(ω)": -1, "W1(ω)": -1,
    "ΔERMSE": -1, "FRMSE": -1, "ΔADF": -1, "ΔvDOS": -1,
    "PMAE": -1, "PW1": -1, "ΔP": -1, "Speed": -1,
    "τ": -1, "E flips": -1, "E jump": -1, "PBE Δr wall": -1, "PBE E MAE": -1,
    "PBE Δre": -1, "PBE ΔDe": -1, "PBE F MAE": -1, "PBE Δω": -1,
    "F flips": -1, "F TV": -1, "F jump": -1,
    "CPS": 1, "CMDS": 1, "CDS": 1,
}

# Metrics whose ideal value is 0 (not the min or max of their range): these
# must be scored by |value| before a "lower-is-better" direction is applied.
CLOSE_ZERO = {"κSRD"}


def direction_score(s: pd.Series, metric: str, direction: int) -> pd.Series:
    """Min-max map a metric to [0,1] where 1 is best (direction-aware)."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if metric in CLOSE_ZERO:
        s = s.abs()
    lo, hi = s.min(), s.max()
    x = pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)
    return 1.0 - x if direction < 0 else x

# Pretty unicode/LaTeX labels for publication figures.
PRETTY = {
    "Acc": "Accuracy", "Prec": "Precision", "F1": "F1", "DAF": "DAF",
    "MAE": "MAE", "R2": r"$R^2$",
    "Σ= 10-2": r"$\Sigma_{=}\,10^{-2}$", "Σ↓ 10-2": r"$\Sigma_{\downarrow}\,10^{-2}$",
    "Σ↑ 10-2": r"$\Sigma_{\uparrow}\,10^{-2}$",
    "Σ= 10-5": r"$\Sigma_{=}\,10^{-5}$", "Σ↓ 10-5": r"$\Sigma_{\downarrow}\,10^{-5}$",
    "Σ↑ 10-5": r"$\Sigma_{\uparrow}\,10^{-5}$", "RMSD": "RMSD",
    "κSRME": r"$\kappa_{\rm SRME}$", "κSRE": r"$\kappa_{\rm SRE}$",
    "κSRD": r"$\kappa_{\rm SRD}$", "κ failed": r"$\kappa$ failed",
    "Im(ω)": r"Im($\omega$)", "W1(ω)": r"$W_1(\omega)$",
    "ΔERMSE": r"$\Delta E_{\rm RMSE}$", "FRMSE": r"$F_{\rm RMSE}$",
    "ΔADF": r"$\Delta$ADF", "ΔvDOS": r"$\Delta$vDOS",
    "PMAE": r"$P_{\rm MAE}$", "PW1": r"$P_{W1}$", "ΔP": r"$\Delta P$",
    "Speed": "Speed", "Slowdown": "Slowdown",
    "τ": r"$\tau$", "E flips": "E flips", "E jump": "E jump",
    "PBE Δr wall": r"PBE $\Delta r_{\rm wall}$", "PBE E MAE": "PBE E MAE",
    "PBE Δre": r"PBE $\Delta r_e$", "PBE ΔDe": r"PBE $\Delta D_e$",
    "PBE F MAE": "PBE F MAE", "PBE Δω": r"PBE $\Delta\omega$",
    "F flips": "F flips", "F TV": "F TV", "F jump": "F jump",
    "CPS": "CPS", "CMDS": "CMDS", "CDS": "CDS",
    "Params_M": r"Params (M)",
}

COMPOSITE = ["CPS", "CMDS", "CDS"]


def _pretty(col: str) -> str:
    return PRETTY.get(col, col)


def load_tasks() -> dict[str, pd.DataFrame]:
    """Load each subtask table indexed by model name (all columns numeric)."""
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    out = {}
    for task, path in TASK_FILES.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing input for {task}: {path}. "
                "Set MATBENCH_DATA_DIR if the CSV files live elsewhere."
            )
        df = pd.read_csv(path).set_index("Model")
        out[task] = df
    return out


def numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def combined_metrics(which: dict[str, list[str]] | None = None,
                     prefix: bool = True) -> pd.DataFrame:
    """Model x metric frame over all tasks, using the canonical metric lists.

    With ``prefix=True`` columns are named ``"<Task>::<metric>"`` so that
    metrics shared across tasks (RMSD, Speed, ...) stay unique.
    """
    which = which or TASK_METRICS
    tasks = load_tasks()
    frames = []
    for task, cols in which.items():
        df = numeric(tasks[task][cols])
        if prefix:
            df.columns = [f"{task}::{c}" for c in df.columns]
        frames.append(df)
    out = pd.concat(frames, axis=1)
    return out


def metric_label(col: str) -> str:
    """Pretty label for a ``Task::metric`` column."""
    return _pretty(col.split("::")[-1])


def params_to_millions(series: pd.Series) -> pd.Series:
    def parse(x):
        if not isinstance(x, str):
            return np.nan
        m = re.match(r"([\d.]+)\s*([MmKk]?)", x.strip())
        if not m:
            return np.nan
        v = float(m.group(1))
        unit = m.group(2)
        if unit == "M":
            return v
        if unit == "K" or unit == "k":
            return v / 1000.0
        return v / 1e6
    return series.map(parse)


# ------------------------------------------------------------------- styling
def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


DIVERGING = "RdBu_r"
SEQ = "viridis"


def corr_colormap():
    return mpl.colormaps[DIVERGING]


def save(fig, name: str, also_pdf: bool = True):
    os.makedirs(FIG, exist_ok=True)
    png = os.path.join(FIG, name + ".png")
    fig.savefig(png)
    if also_pdf:
        fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("saved", png)


def star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


# --------------------------------------------------------------- correlations
def corr_with_p(df: pd.DataFrame, method: str = "spearman"):
    """Pairwise correlation + p-values; listwise-complete per pair."""
    from scipy.stats import spearmanr, pearsonr

    cols = list(df.columns)
    n = len(cols)
    R = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    N = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        R[i, i], P[i, i], N[i, i] = 1.0, 0.0, df[cols[i]].notna().sum()
        for j in range(i + 1, n):
            sub = df[[cols[i], cols[j]]].dropna()
            if len(sub) < 3:
                continue
            fn = spearmanr if method == "spearman" else pearsonr
            r, p = fn(sub.iloc[:, 0], sub.iloc[:, 1])
            R[i, j] = R[j, i] = r
            P[i, j] = P[j, i] = p
            N[i, j] = N[j, i] = len(sub)
    return pd.DataFrame(R, index=cols, columns=cols), pd.DataFrame(P, index=cols, columns=cols), pd.DataFrame(N, index=cols, columns=cols)


def cluster_order(corr: pd.DataFrame) -> list[int]:
    """Hierarchical (1 - r) ordering of a correlation/similarity matrix."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    m = corr.values.copy()
    m = np.nan_to_num(m, nan=0.0)
    np.fill_diagonal(m, 1.0)
    d = 1.0 - m
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    d[d < 0] = 0.0
    Z = linkage(squareform(d, checks=False), method="average")
    return list(leaves_list(Z))


def draw_corr_heatmap(ax, corr, pval, order=None, annot=True, cmap=None,
                      show_stars=True, cbar=True, title=None, label_fs=None,
                      tri="lower", vmin=-1, vmax=1):
    import matplotlib.pyplot as plt
    if order is None:
        order = list(range(len(corr)))
    c = corr.iloc[order, order]
    p = pval.iloc[order, order]
    labels = [metric_label(x) if "::" in str(x) else _pretty(x) for x in c.columns]
    cmap = cmap or corr_colormap()
    mask = np.zeros_like(c.values, dtype=bool)
    if tri == "lower":
        mask[np.triu_indices_from(mask, k=1)] = True
    elif tri == "upper":
        mask[np.tril_indices_from(mask, k=-1)] = True
    C = np.ma.array(c.values, mask=mask)
    im = ax.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    fs = label_fs or (9 if len(labels) <= 8 else 7.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    if annot:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if mask[i, j]:
                    continue
                v = c.values[i, j]
                if not np.isfinite(v):
                    continue
                s = f"{v:.2f}"
                if show_stars and i != j:
                    s += star(p.values[i, j])
                col = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, s, ha="center", va="center", fontsize=fs - 1.2, color=col)
    if title:
        ax.set_title(title, pad=8)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Spearman $\rho$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return im
