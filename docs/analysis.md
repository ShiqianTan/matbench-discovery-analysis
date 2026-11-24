# Matbench-Discovery: Cross-Task Metric & Model Analysis

Comprehensive analysis of the Matbench-Discovery leaderboard covering **43 ML
interatomic potentials** across five downstream subtasks. Three questions are
addressed:

1. How do metrics correlate **within** each subtask, and how do subtasks
   relate to each other?
2. How does **inference speed** compare across tasks, and what does the
   accuracy–speed trade-off look like (Pareto)?
3. Can a **Speed-free composite** better capture performance, and how does it
   re-rank models?
4. A dedicated study of the **GRACE model family** (`outputs/grace/`).
5. A **roadmap for developing a SOTA model** from the structural gaps found
   (three-axis leaderboard, `fig13`/`fig14`).

- **Figures:** `outputs/figures/` (PNG @300 dpi + vector PDF)
- **Tables:** `outputs/tables/` (CSV + Markdown + LaTeX)
- **Code:** `scripts/` (`common.py` + numbered scripts `01`–`09`)
- **GRACE study:** `outputs/grace/`

Reproduce everything:

```bash
python scripts/run_all.py
```

---

## 1. Data & methods

| Subtask | Table | Models | Headline metric |
|---|---|---|---|
| Discovery | `unique_prototypes.csv` (master) | 43 | F1 |
| Geometry optimization | `..._geo-opt.csv` | 42 | RMSD |
| Phonons | `...-phonons.csv` | 42 | κ_SRME |
| Molecular dynamics | `...-md.csv` | 42 | CMDS |
| Diatomics | `...-diatomics.csv` | 42 | CDS |

Composite scores **CPS** (discovery+geo-opt+phonons), **CMDS** (MD), **CDS**
(diatomics) are from the master table. `NequIP-GNoME` has discovery metrics
only.

**Statistical choices**

- **Spearman ρ** everywhere: several raw diatomics metrics are extremely
  heavy-tailed (e.g. `PBE F MAE` up to 1.8 × 10³⁰ for `EquFlash*`, `F TV` up to
  8 × 10³²); rank statistics are robust to these outliers.
- For cross-task views, metrics are **direction-corrected to a higher-is-better
  [0,1] scale**. Spearman is invariant to this monotone rescaling — it only
  fixes the *sign* of "better".
- **Near-zero-optimal metrics** (`κ_SRD`) are scored by |value| before applying
  direction (its ideal is 0, not an extreme).
- Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001.

---

## 2. Within-task metric correlations — `fig1_*`

Clustered, significance-annotated Spearman matrices: `fig1_overview_within_corr`
(all five subtasks) + one detail per subtask; full numbers in
`outputs/tables/table_within_corr_*.csv`, ranked pairs in
`table2_within_task_pairs`.

**Observations**

- **Discovery** is highly redundant: all six metrics have |ρ| ≥ 0.86; MAE
  anti-correlates with the rest (ρ ≈ −0.95 … −0.99).
- **Geometry:** symmetry metrics load on one factor; RMSD is only weakly
  related to them, so it carries extra information.
- **Phonons:** κ_SRME, κ_SRE, Im(ω), W1(ω) are tightly coupled (block ρ = 0.79);
  κ_SRD is the odd one out.
- **MD:** ΔE_RMSE, F_RMSE, ΔADF, ΔvDOS, ΔP cluster (ρ ≥ 0.8); **Speed
  anti-correlates with accuracy** (ρ ≈ −0.4) — a speed–accuracy trade-off.
- **Diatomics** is the most heterogeneous task (mean within-block ρ ≈ 0.22): a
  broad accuracy/physicality group (ρ ≈ 0.4–0.7) plus weakly-related `F TV`
  and `Speed`.

---

## 3. Cross-task correlations — `fig2_*`

- `fig2a_task_scores_corr` — task-to-task ρ of normalised headline scores.
- `fig2b_task_scores_scatter` — pairwise scatter matrix.
- `fig2c_composite_corr` — CPS / CMDS / CDS agreement.
- `fig2d_full_metric_corr` — **all** metrics, blocked by task.
- `fig2e_task_block_corr` — mean ρ between metric groups.

**Observations**

- **{Discovery, Geometry, Phonons} form one tight cluster**
  (task-score ρ = 0.73–0.91): models that classify stable structures well also
  relax them and predict their phonons well.
- **MD and Diatomics are nearly orthogonal** to that cluster (ρ ≈ 0.1) and to
  each other (ρ ≈ 0.24) — no single model is uniformly strong.
- At the composite level: CPS–CMDS ρ = 0.14, CPS–CDS ρ = 0.06, CMDS–CDS ρ = 0.24.
- Block-diagonal coherence (mean within-task ρ): Discovery 0.95, **Phonons
  0.79**, Geometry 0.63, MD 0.44, Diatomics 0.22.

---

## 4. Composite-score anatomy — `fig8_composite_anatomy`

How tightly each composite tracks its own raw components
(`table6_composite_anatomy.csv`):

- **CPS** weights F1, RMSD and κ_SRME comparably (ρ ≈ 0.9 each).
- **CMDS is dominated by Speed** (ρ = 0.76); its accuracy components
  (ΔE_RMSE, F_RMSE, ΔvDOS) contribute almost nothing (ρ ≤ 0.1). CMDS is
  effectively a **speed score** — this is why it decouples from accuracy.
- **CDS** is driven by force/energy errors and physicality (F MAE, E MAE,
  E jump, τ; ρ = 0.7–0.85), Speed ρ = 0.23.

---

## 5. Exploration 1 — Speed across tasks & Pareto fronts — `fig9_speed_pareto`

**MD speed vs Diatomics speed:** Spearman ρ = **0.61*** (n = 42, log scale).
Models that are fast on MD trajectories are also fast on diatomics — speed is a
partially transferable, model-intrinsic property.

**Pareto fronts** (minimise speed, maximise a Speed-free accuracy score):

- **MD Pareto-optimal (8):** GRACE-2L-OAM-L, GRACE-2L-OAM, GRACE-1L-OAM,
  GRACE-3L-OAM-L, TACE-OAM-L, ORB v2 MPA, EquFlashV2*, Prophet-OAME-MBD.
- **Diatomics Pareto-optimal (5):** ORB v3, Allegro-OAM-L, M3GNet,
  Nequix MP PFT, Nequix MP.

The GRACE family dominates the MD Pareto front; the diatomics front is more
diverse and includes very small models (Nequix, M3GNet) that trade accuracy for
speed. Numbers in `table7_speed_analysis.csv` and
`table7_speed_pareto_members.txt`.

---

## 6. Exploration 2 — a Speed-free composite (NSC) — `fig10`–`fig12`

**NSC (No-Speed Composite):** equal-weight mean of six direction-corrected
[0,1] components — the metrics that correlate most with each existing composite,
with Speed removed:

| Source composite | Metric | Weight |
|---|---|---|
| CPS | Discovery F1 | 1/6 |
| CPS | Geometry RMSD | 1/6 |
| CPS | Phonons κ_SRME | 1/6 |
| CMDS | MD ΔADF | 1/6 |
| CDS | Diatomics PBE E MAE | 1/6 |
| CDS | Diatomics PBE F MAE | 1/6 |

**NSC ranking agreement (Spearman) with existing orderings:**

| vs | ρ | interpretation |
|---|---|---|
| CPS | **0.90*** | NSC ≈ CPS |
| task Geometry | 0.90*** | |
| task Phonons | 0.83*** | |
| task Discovery | 0.83*** | |
| Mean(task) | 0.77*** | |
| CDS | 0.10 | independent |
| CMDS | 0.09 | independent |

**Key result:** once Speed is removed, the remaining "accuracy" parts of **all**
composites collapse onto the same discovery/geometry/phonons axis — because
ΔADF and PBE E/F MAE are themselves strongly correlated with discovery
performance. **The apparent multi-task structure of the leaderboard composites
is largely an artefact of Speed weighting.**

**Re-ranking highlights** (`table8_NSC_ranking.*`):

- **TECE-OAM-RRA-1.0**: 20th (Mean) → **1st (NSC)**.
- **EquiformerV3+DeNS-OAM**: 21st → 5th; **DPA-4.0.1-Pro-MPtrj**: 32nd → 16th.
- **EquFlash***: 12th → **39th** — its catastrophic diatomics
  (`PBE F MAE` ≈ 10³⁰) is exposed once accuracy is weighted over speed.
- **ORB v3**: 4th → 17th.
- NSC top-5: TECE-OAM-RRA-1.0, EquFlashV2*, GRACE-3L-OAM-L, Prophet-OAME-MBD,
  EquiformerV3+DeNS-OAM.

---

## 7. Exploration 3 — GRACE model family — `outputs/grace/`

Five variants (`GRACE-1L-OAM`, `GRACE-2L-OAM`, `GRACE-2L-OAM-L`,
`GRACE-3L-OAM-L`, `GRACE-2L-MPtrj`) analysed standalone in `outputs/grace/`:

- `outputs/grace/figures/grace_metric_heatmap` — every metric, normalised vs the full
  field, blocked by task.
- `outputs/grace/figures/grace_task_scores` — per-task + composite scores.
- `outputs/grace/figures/grace_radar` — task-profile radar.
- `outputs/grace/figures/grace_field_rank` — rank within the whole field per task.
- `outputs/grace/tables/grace_metrics_and_ranks.csv`; see
  `outputs/grace/README.md`.

**Observations**

- **GRACE-3L-OAM-L** is the family's best on Discovery (rank 5.5), Geometry
  (1.5) and Phonons (6) and has the best mean task score.
- **GRACE-2L-OAM-L** leads MD (rank 1) and CMDS (rank 1), and is strong on
  diatomics (rank 14); it is the balanced "large" variant.
- Scaling **1L → 2L → 3L** (fixed OAM training) monotonically improves the
  discovery/geometry/phonons axis (Discovery rank 29 → 20.5 → 19 → 5.5).
- Switching training data to **MPtrj** (`GRACE-2L-MPtrj`) sharply degrades the
  discovery/geometry axis (rank 39) while staying competitive on MD (rank 9).

---

## 8. Model landscape — `fig3`–`fig7`

- `fig3_model_landscape_heatmap` — models × task-score heatmap, ordered by mean
  score (most informative single overview).
- `fig4_composite_rankings` — rankings by CPS / CMDS / CDS.
- `fig5_params_vs_performance` — parameters vs. performance (weak positive
  trend, ρ = 0.38).
- `fig6_model_embedding` — PCA / MDS of the task-score vector.
- `fig7_top_models_parallel` — parallel-coordinates profile of the top 12.

**Observations**

- Top overall (mean task score): **GRACE-3L-OAM-L, TACE-OAM-L, GRACE-2L-OAM-L,
  ORB v3, Nequip-OAM-L** (`table4_model_ranking.*`).
- **No model wins everywhere** (e.g. ORB v3 leads diatomics but is mid-pack on
  geometry; GRACE-2L-OAM-L leads MD but ranks 19th on discovery).
- Model size explains little of the variance in the mean score.

---

## 9. Roadmap for a SOTA model — `fig13` / `fig14`

The analysis above implies *where* to invest. Define three axes:

- **Coupled axis** = mean(Discovery, Geometry, Phonons)
- **MD**
- **Diatomics**

`fig13_three_axis_top3` and `fig14_three_axis_top5` show the top-3 / top-5 per
axis and the overlap (`table9_three_axis_top3.csv`, `table9_three_axis_top5.csv`).

**Structural facts that define the opportunity**

- The three axes are **largely independent**: no model is in the top-3 of two
  axes at all; on the top-5 view only `GRACE-3L-OAM-L` bridges two axes
  (Coupled ∩ MD), and the **triple intersection is empty**.
- **Coupled axis is crowded/saturated** (CPS ≈ 0.9 for the leaders) → only
  incremental gains there.
- **Diatomics is the weakest, most orthogonal axis** (CDS spread 0.12–0.83,
  mean 0.57) and the leaders ignore it → the largest single-task headroom.
  `EquFlash*` (`PBE F MAE` ≈ 10³⁰) shows strong models can diverge on
  isolated/dissociating systems.
- **Speed is transferable** (MD↔diatomics ρ = 0.61) and **CMDS is
  speed-dominated** → cheap leaderboard gains, but fragile under the
  Speed-free (NSC) view.

**Recommended directions, ranked by expected ROI**

1. **Diatomics-specialised capability (highest ROI).** Target CDS > 0.85 without
   losing the coupled axis — currently unmet. Data: dissociation curves,
   isolated atoms/dimers, radicals and spin states, long-range/electrostatic
   corrections; add robustness training against divergence.
2. **Multi-task foundation + routing/ensemble (fastest route to "all-round
   SOTA").** Start from the best OAM checkpoint (`GRACE-3L-OAM-L` /
   `TACE-OAM-L` / `ORB v3`), continue-train on an OAM + MPtrj + MP + MatPES mix
   with task heads, or use MoE routing / distillation. Beat the current best
   all-rounder (max-min over the five tasks): **`TACE-OAM-L` = 0.85**.
3. **Long-range + symmetry inductive biases.** Simultaneously helps the coupled
   axis (Σ geometry metrics) and diatomics (Δr_wall, τ): equivariant backbones
   (eSEN / Equiformer) with explicit long-range electrostatics/dispersion.
4. **MD long-timescale accuracy × low latency.** MD is independent and the
   GRACE family owns its Pareto front; improve ΔADF/ΔvDOS/ΔE_RMSE stability at
   low cost to also top CMDS.
5. **Speed engineering last (and hedge it).** Compilation, distillation,
   quantisation, smaller equivariant nets raise both MD and diatomics Speed —
   but do not sacrifice accuracy, since the NSC view removes the speed bonus.

**Scaling is necessary but not sufficient:** parameter count correlates only
weakly with the mean score (ρ = 0.38); architecture and data quality dominate
(the GRACE 1L→3L monotone gain on the coupled axis is the exception, and worth
reusing).

**Suggested target & evaluation**

- Aim for **top-3 on all three axes simultaneously** — the clearest, currently
  empty SOTA slot.
- Report **NSC (accuracy)** *and* **CMDS / CDS (leaderboard)** to avoid
  over-fitting the speed-driven composite.
- Report the full per-task profile (`fig3`), not a single aggregate.

---

## 10. Figure & table index

**Figures** (`outputs/figures/`)

| File | Content |
|---|---|
| `fig1_overview_within_corr`, `fig1_{discovery,geometry,phonons,md,diatomics}_within_corr` | within-task Spearman matrices |
| `fig2a_task_scores_corr`, `fig2b_task_scores_scatter`, `fig2c_composite_corr`, `fig2d_full_metric_corr`, `fig2e_task_block_corr` | cross-task correlations |
| `fig3_model_landscape_heatmap` | model × task heatmap |
| `fig4_composite_rankings` | CPS/CMDS/CDS rankings |
| `fig5_params_vs_performance` | size vs performance |
| `fig6_model_embedding` | PCA / MDS |
| `fig7_top_models_parallel` | top-12 profiles |
| `fig8_composite_anatomy` | composite ↔ component ρ |
| `fig9_speed_pareto` | speed correlation + Pareto fronts |
| `fig10_NSC_ranking_stacked` | NSC ranking + component contributions |
| `fig11_NSC_rank_comparison` | NSC vs existing rankings |
| `fig12_NSC_vs_mean` | NSC vs mean task score, rank shifts |
| `fig13_three_axis_top3` | three-axis leaderboard, top-3 + overlap |
| `fig14_three_axis_top5` | three-axis leaderboard, top-5 + overlap |

**Tables** (`outputs/tables/`)

| File | Content |
|---|---|
| `table1_descriptive_stats.{csv,md,tex}` | per-metric summary stats + units |
| `table2_within_task_pairs.{csv,md}` | within-task pairs ranked by \|ρ\| |
| `table3_cross_task_corr.{csv,tex}` | task-to-task Spearman |
| `table4_model_ranking.{csv,md,tex}` | per-task ranks + composites |
| `table5_composite_stats.csv` | CPS/CMDS/CDS stats |
| `table6_composite_anatomy.csv` | composite ↔ component ρ |
| `table7_speed_analysis.csv`, `table7_speed_pareto_members.txt` | speed + Pareto |
| `table8_NSC_ranking.{csv,tex}`, `table8b_NSC_rank_corr.csv` | NSC ranking + agreement |
| `table9_three_axis_top{3,5}.csv` | top-3 / top-5 per axis + overlap |
| `table_within_corr_*.csv`, `table_within_pval_*.csv` | 5 within-task matrices + p-values |
| `table_cross_task_full_corr.csv`, `table_cross_task_block_corr.csv` | cross-task matrices |
| `table_model_scores_normalized.csv`, `table_task_scores_{corr,pval,normalized}.csv`, `table_pca_loadings.csv` | derived score matrices |

---

## 11. Key take-aways

1. **Discovery, geometry and phonons are one coupled axis** of MLIP quality.
2. **MD and diatomics are independent axes** — evaluate a model on all three to
   trust it downstream.
3. **The official CMDS is effectively a speed score**, not an accuracy score;
   composite weights should be chosen accordingly.
4. **Speed transfers across tasks** (MD↔diatomics ρ = 0.61); the GRACE family
   owns most of the MD accuracy–speed Pareto front.
5. **Removing Speed collapses the composites onto one accuracy axis** (NSC vs
   CPS ρ = 0.90; NSC vs CMDS/CDS ρ ≈ 0.1), exposing outliers such as
   `EquFlash*` (12th → 39th).
6. **Diatomics metrics are only moderately correlated**, so treat the task as
   multi-dimensional.
7. **Ranking depends on task** — report the full per-task profile (`fig3`), not
   a single aggregate.
8. **The SOTA slot is the empty triple intersection** — no model is top-3 (or
   even top-5) on all three axes; the best route is a diatomics-specialised,
   multi-task model (`fig13`/`fig14`, §9).
