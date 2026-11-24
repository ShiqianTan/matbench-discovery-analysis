"""Tables - descriptive stats, within/cross-task correlations, rankings (CSV/MD/LaTeX)."""
import os
import numpy as np
import pandas as pd
import common as C

UNITS = {
    "MAE": "eV/atom", "RMSD": "norm.", "κSRME": "-", "κSRE": "-", "κSRD": "-",
    "κ failed": "frac.", "Im(ω)": "frac.", "W1(ω)": "THz",
    "ΔERMSE": "meV/atom", "FRMSE": "meV/Å", "ΔADF": "%", "ΔvDOS": "%",
    "PMAE": "GPa", "PW1": "GPa", "ΔP": "%", "Speed": "s",
    "τ": "-", "E flips": "-", "E jump": "eV", "PBE Δr wall": "Å",
    "PBE E MAE": "eV", "PBE Δre": "Å", "PBE ΔDe": "eV", "PBE F MAE": "eV/Å",
    "PBE Δω": "cm⁻¹", "F flips": "-", "F TV": "-", "F jump": "eV/Å",
}

HEADLINE = {"Discovery": "F1", "Geometry": "RMSD", "Phonons": "κSRME",
            "MD": "CMDS", "Diatomics": "CDS"}


def to_latex(df, path, caption, label, float_format="%.3f", **kw):
    try:
        tex = df.to_latex(caption=caption, label=label, float_format=float_format,
                          escape=True, **kw)
        with open(path, "w") as f:
            f.write(tex)
    except Exception as e:  # jinja2 missing etc.
        print("LaTeX skipped:", e)


def descriptive():
    tasks = C.load_tasks()
    rows = []
    for task, cols in C.TASK_METRICS.items():
        df = C.numeric(tasks[task][cols])
        for c in cols:
            s = df[c].dropna()
            rows.append({
                "Task": task, "Metric": c, "Unit": UNITS.get(c, ""),
                "Direction": {1: "higher", -1: "lower", 0: "zero"}[C.DIRECTION[c]],
                "n": len(s), "Mean": s.mean(), "Std": s.std(),
                "Median": s.median(), "Min": s.min(), "Max": s.max(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.TAB, "table1_descriptive_stats.csv"), index=False)
    md = out.copy()
    for c in ["Mean", "Std", "Median", "Min", "Max"]:
        md[c] = md[c].map(lambda v: f"{v:.4g}")
    with open(os.path.join(C.TAB, "table1_descriptive_stats.md"), "w") as f:
        f.write(md.to_markdown(index=False))
    to_latex(out.set_index(["Task", "Metric"])[["Unit", "Direction", "n", "Mean", "Std", "Median", "Min", "Max"]],
             os.path.join(C.TAB, "table1_descriptive_stats.tex"),
             "Descriptive statistics of Matbench-Discovery metrics per subtask.",
             "tab:descriptive")
    return out


def within_top_pairs():
    tasks = C.load_tasks()
    rows = []
    for task, cols in C.TASK_METRICS.items():
        df = C.numeric(tasks[task][cols])
        corr, pval, n = C.corr_with_p(df, "spearman")
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                rows.append({"Task": task, "Metric A": cols[i], "Metric B": cols[j],
                             "rho": corr.iloc[i, j], "p": pval.iloc[i, j], "n": n.iloc[i, j]})
    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()
    out = out.sort_values(["Task", "abs_rho"], ascending=[True, False])
    out.to_csv(os.path.join(C.TAB, "table2_within_task_pairs.csv"), index=False)
    md = out.drop(columns="abs_rho").copy()
    md["rho"] = md["rho"].map(lambda v: f"{v:.3f}")
    md["p"] = md["p"].map(lambda v: f"{v:.2e}")
    with open(os.path.join(C.TAB, "table2_within_task_pairs.md"), "w") as f:
        f.write(md.to_markdown(index=False))
    return out


def cross_task_table():
    from importlib import import_module
    m = import_module("02_cross_task_corr")
    T = m.task_scores()
    corr, pval, _ = C.corr_with_p(T, "spearman")
    corr.round(4).to_csv(os.path.join(C.TAB, "table3_cross_task_corr.csv"))
    to_latex(corr.round(3), os.path.join(C.TAB, "table3_cross_task_corr.tex"),
             "Spearman correlation between per-task headline scores.",
             "tab:crosscorr")
    return corr


def model_ranking():
    from importlib import import_module
    m = import_module("03_model_landscape")
    L = m.build_landscape()
    ranks = pd.DataFrame(index=L.index)
    for task in HEADLINE:
        ranks[f"rank_{task}"] = L[task].rank(ascending=False, method="min")
        ranks[f"score_{task}"] = L[task]
    ranks["rank_overall"] = L["Mean(task)"].rank(ascending=False, method="min")
    ranks["score_mean_task"] = L["Mean(task)"]
    for c in ["CPS", "CMDS", "CDS"]:
        ranks[c] = L[c]
    ranks["Params_M"] = L["Params_M"]
    ranks = ranks.sort_values("rank_overall")
    ranks.insert(0, "Rank", np.arange(1, len(ranks) + 1))
    ranks.to_csv(os.path.join(C.TAB, "table4_model_ranking.csv"))
    md = ranks[["rank_overall"] + [f"rank_{t}" for t in HEADLINE]
               + ["score_mean_task", "CPS", "CMDS", "CDS", "Params_M"]].copy()
    for c in ["score_mean_task", "CPS", "CMDS", "CDS", "Params_M"]:
        md[c] = md[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    md = md.astype(object).where(pd.notna(md), "")
    with open(os.path.join(C.TAB, "table4_model_ranking.md"), "w") as f:
        f.write(md.to_markdown())
    to_latex(md, os.path.join(C.TAB, "table4_model_ranking.tex"),
             "Model ranking by per-task scores and composite scores.",
             "tab:ranking")
    return ranks


def composite_table():
    tasks = C.load_tasks()
    master = tasks["Discovery"]
    comp = C.numeric(master[["CPS", "CMDS", "CDS"]])
    desc = comp.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.to_csv(os.path.join(C.TAB, "table5_composite_stats.csv"))
    return desc


def main():
    descriptive()
    within_top_pairs()
    cross_task_table()
    model_ranking()
    composite_table()
    print("tables written to", C.TAB)
    print(sorted(os.listdir(C.TAB)))


if __name__ == "__main__":
    main()
