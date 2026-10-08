# SE-FAHP-CD: Stress-Eigen Fuzzy AHP with Crack Detection

Code, input data, and result tables for the decision-support model that prioritizes AI features for ESG risk assessment in the European financial sector. The pipeline combines stress-eigen fuzzy AHP with crack diagnostics, fuzzy TOPSIS, a stressed DEMATEL connectedness measure, and the Stress-Resilience Priority Index (SRPI).

The scripts reproduce every table and figure of the manuscript and the supplementary materials.

## Method overview

| Stage | Script | What it does |
|---|---|---|
| 1 | `MCDM_AHP.py` | Aggregates expert fuzzy pairwise matrices (geometric mean), defuzzifies, and computes eigenvector weights under a power-stress path β ∈ {−50%, −25%, 0, +25%, +50%}. Applies crack detection (κ threshold 0.30, CR threshold 0.10) and computes the Eigenvector Stability Index (ESI) and its CR-restricted version (ESI_CR). |
| 2 | `MCDM_Topsis.py` | Fuzzy TOPSIS over 9 AI features and 11 criteria with the weights from stage 1. Primary model uses the 75/25 outcome/resource allocation. Also runs raw FAHP weights, allocation scenarios, and the polarity audit and sensitivity. |
| 3 | `MCDM_Dematel.py` | Fuzzy DEMATEL on an undirected feature interrelation matrix under symmetry-preserving stress. Computes prominence D+R, the stress sensitivity index FSI, and benchmarks D+R against weighted degree and eigenvector centrality (path-length ablation and expert bootstrap). |
| 4 | `MCDM_FINAL.py` | Fuses TOPSIS closeness (RC), DEMATEL prominence, and FSI into the SRPI. Reports absolute and min-max vulnerability variants and raw FAHP vs 75/25 agreement. |
| 5 | `MCDM_Validation.py` | Runs the validation suite (85 runs across seven experiments plus identity checks, Monte Carlo AHP benchmark, permutation invariance, crack-threshold sweep, stress-grid diagnostic, ESI vs Monte Carlo, polarity sensitivity). |
| 6 | `MCDM_Visualization.py` | Produces figures `fig02` to `fig14` and `SUPPLEMENTARY_REVIEW_TABLES.xlsx`. |

## Repository layout

```
.
├── MCDM_AHP.py
├── MCDM_Topsis.py
├── MCDM_Dematel.py
├── MCDM_FINAL.py
├── MCDM_Validation.py
├── MCDM_Visualization.py
├── dfa.xlsx                 expert input, criteria pairwise comparisons (11 x 11) - available on request
├── dft.xlsx                 expert input, feature-criterion ratings (9 x 11) - available on request
├── dfd.xlsx                 expert input, feature interrelations (9 x 9) - available on request
├── *.csv                    result tables written by the scripts
├── SUPPLEMENTARY_REVIEW_TABLES.xlsx
└── figures_v3/              figures written by MCDM_Visualization.py
```

## Input data

Each workbook has eight sheets (`E1` to `E8`, one per expert). Each cell holds a triangular fuzzy number written as a string `"l,m,u"`.

| File | Sheet shape | Content | Scale anchors |
|---|---|---|---|
| `dfa.xlsx` | 11 x 11 | Pairwise importance of criteria | Equal importance to extremely more important (Saaty 1 to 9) |
| `dft.xlsx` | 9 x 11 | Rating of each feature's contribution to each criterion | Extremely Low to Extremely High (7 levels) |
| `dfd.xlsx` | 9 x 9 | Strength of relationship between feature pairs (undirected) | No relationship to Extremely high relationship (7 levels) |

Criterion definitions shown to experts are favorable-polarity for all 11 criteria. A higher rating in `dft.xlsx` always means stronger contribution to the favorable state, including Cost Efficiency, Implementation Complexity (simpler is better), and Data Dependency (less reliance is better). Therefore `REVERSE_CODED_CRITERIA` is empty by default.

Expert responses are anonymized. No personal data are included.

## Requirements

Python 3.10 or newer (tested with 3.12).

```
numpy
pandas
scipy
matplotlib
openpyxl
```

Install:

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

## Usage

All scripts read and write in one project folder. Set it once through the environment variable `MCDM_PROJECT_DIR`. This is required because the default path differs between scripts (`MCDM_AHP.py` uses the current working directory, the others default to `/home/coder/project`).

```bash
export MCDM_PROJECT_DIR="$(pwd)"
```

Run the stages in this order. Each stage reads the output of the previous one.

```bash
python MCDM_AHP.py
python MCDM_Topsis.py
python MCDM_Dematel.py
python MCDM_FINAL.py
python MCDM_Validation.py
python MCDM_Visualization.py
```

`MCDM_Validation.py` imports `MCDM_AHP`, `MCDM_Topsis`, and `MCDM_Dematel`, so all scripts must stay in the same folder.

## Key settings

Settings are module-level constants at the top of each script.

| Setting | Value | File |
|---|---|---|
| Stress set β | −0.50, −0.25, 0, 0.25, 0.50 | `MCDM_AHP.py` |
| Crack threshold κc | 0.30 | `MCDM_AHP.py` |
| CR admissibility threshold | 0.10 | `MCDM_AHP.py` |
| ESI agreement rule | τ ≥ 0.90, max rank shift 1 | `MCDM_AHP.py` |
| Primary weighting | `"75_25"` (set `"raw"` to use raw FAHP weights) | `MCDM_Topsis.py` |
| Outcome/resource allocation | 0.75 / 0.25 | `MCDM_Topsis.py` |
| Allocation scenarios | 50/50, 60/40, 70/30, 75/25, 80/20, raw FAHP | `MCDM_Topsis.py` |
| Raw vs 75/25 agreement rule | ρ ≥ 0.95, τ ≥ 0.85, max rank shift 1 | `MCDM_Topsis.py` |
| Stress amplification γ sweep | 0, 0.5, 1.0, 1.5, 2.0 (primary 1.0) | `MCDM_Dematel.py` |
| DEMATEL crack tolerance | 0.30 | `MCDM_Dematel.py` |
| Centrality agreement rule | ρ ≥ 0.95, τ ≥ 0.85, max rank shift 1 | `MCDM_Dematel.py` |
| SRPI fusion parameters λ, μ | 1.0, 1.0 | `MCDM_Validation.py` |
| Weight perturbation | ±5% | `MCDM_Validation.py` |
| Permutation test | 200 permutations, seed 42, tolerance 1e-6 | `MCDM_Validation.py` |
| Monte Carlo AHP benchmark | 2000 draws, σ = 0.15, seed 123 | `MCDM_Validation.py` |
| ESI vs Monte Carlo | 5000 bootstrap, 20000 permutation resamples | `MCDM_Validation.py` |
| Bootstrap seeds (DEMATEL, polarity check) | 20261005 | `MCDM_Dematel.py`, `MCDM_Topsis.py` |

All random procedures use fixed seeds, so repeated runs give identical tables.

## Outputs

### Stage 1, `MCDM_AHP.py`

| File | Content |
|---|---|
| `STRESS_FAHP_RESULTS.csv` | Baseline weights w0, stable weights w*, mean κ, ESI, ESI_CR, and ranks per criterion |
| `FAHP_A0_matrix.csv` | Aggregated crisp pairwise matrix |
| `weight_trajectory.csv`, `consistency_trajectory.csv` | Weights and CR along the stress path |
| `crack_table_fahp.csv` | Step-normalized log-weight change and crack flags, all transitions |
| `crack_table_fahp_CR_admissible.csv` | Same table restricted to transitions with admissible endpoints |
| `ESI_CR_summary.csv`, `ESI_CR_restricted.csv` | Full-path vs CR-restricted ESI comparison (ρ, τ, rank shifts) |

### Stage 2, `MCDM_Topsis.py`

| File | Content |
|---|---|
| `STRESS_TOPSIS_RESULT.csv` | Primary fuzzy TOPSIS result (closeness RC, rank, weighting label) |
| `STRESS_TOPSIS_RESULT_75_25.csv`, `STRESS_TOPSIS_RESULT_raw_fahp.csv` | Results under the two weightings |
| `STRESS_TOPSIS_endogenous_shares.csv` | Endogenous outcome/resource shares implied by raw FAHP weights |
| `STRESS_TOPSIS_weights_raw_vs_7525.csv`, `STRESS_TOPSIS_rescaled_weights.csv` | Criterion weights under both schemes |
| `STRESS_TOPSIS_raw_vs_7525.csv`, `STRESS_TOPSIS_raw_vs_7525_summary.csv` | RC, ranks, ρ, τ, and decision rule outcome for raw vs 75/25 |
| `STRESS_TOPSIS_allocation_sensitivity.csv` (and `_weights`) | Allocation scenarios including raw FAHP |
| `STRESS_TOPSIS_fpis_fnis.csv` | Fuzzy positive and negative ideal solutions |
| `STRESS_TOPSIS_criterion_polarity_audit.csv` | Per-criterion polarity audit with questionnaire definition |
| `STRESS_TOPSIS_polarity_raw_ratings_by_feature.csv`, `..._by_expert.csv` | Raw rating summaries for polarity inspection |
| `STRESS_TOPSIS_polarity_empirical_check.csv` | Correlation of pooled resource-criteria ratings with outcome criteria (bootstrap CI) |
| `STRESS_TOPSIS_polarity_sensitivity.csv`, `..._summary.csv` | Rankings when Implementation Complexity and Data Dependency are reverse-coded |

### Stage 3, `MCDM_Dematel.py`

| File | Content |
|---|---|
| `DEMATEL_Z0_matrix.csv`, `DEMATEL_Zstar_matrix.csv`, `DEMATEL_Tstar_matrix.csv` | Baseline, stress-amplified, and total-relation matrices |
| `STRESS_DEMATEL_RESULTS.csv` | D, R, prominence D+R, FSI, rank per feature |
| `prominence_trajectory.csv`, `crack_table_dematel.csv` | Prominence and crack flags along the stress path |
| `DEMATEL_gamma_sensitivity.csv` | Ranking under the γ sweep |
| `DEMATEL_relationship_coverage.csv` | Coverage of the symmetrized relation matrix |
| `DEMATEL_centrality_benchmark.csv`, `..._summary.csv` | D+R vs weighted degree and eigenvector centrality |
| `DEMATEL_centrality_path_ablation.csv` | Prominence by maximum path length |
| `DEMATEL_centrality_expert_bootstrap.csv` | Expert-bootstrap stability of the benchmark |

### Stage 4, `MCDM_FINAL.py`

| File | Content |
|---|---|
| `Final_Stress_Coupled_Result.csv` | Fused RC, FSI, DR, SRPI, and ranks |
| `SRPI_SCORE_TABLE.csv` | SRPI with absolute and min-max vulnerability |
| `SRPI_VULNERABILITY_SCENARIOS.csv` | SRPI under vulnerability variants |
| `RANK_RECONCILIATION_TOPSIS_SRPI.csv` | TOPSIS vs SRPI rank agreement |
| `SRPI_RAW_vs_7525_COMPARISON.csv`, `..._SUMMARY.csv` | SRPI under raw FAHP vs 75/25 |

### Stage 5, `MCDM_Validation.py`

| File | Content |
|---|---|
| `VALIDATION_ROBUSTNESS_RESULTS.csv` | Per-run ρ and τ for all validation runs |
| `VALIDATION_ROBUSTNESS_SUMMARY.csv`, `..._SUMMARY_ALL_RUNS.csv` | Aggregated statistics, with and without identity runs |
| `VALIDATION_IMPLEMENTATION_EQUIVALENCE.csv` | Baseline-equivalent identity runs |
| `VALIDATION_PERMUTATION_INVARIANCE.csv` | Criterion-order permutation test |
| `VALIDATION_MONTE_CARLO_AHP.csv` | Monte Carlo envelopes of criterion weights |
| `VALIDATION_ESI_vs_MC_STABILITY.csv`, `..._SUMMARY.csv` | ESI vs Monte Carlo dispersion (Spearman, Kendall, bootstrap CI, permutation p, partial correlation) |
| `VALIDATION_CRACK_THRESHOLD_DIAGNOSTIC.csv` | κc sweep |
| `VALIDATION_STRESS_GRID_DIAGNOSTIC.csv`, `..._RANK_AGREEMENT.csv` | Stress step and range grid |
| `VALIDATION_STRESS_AMPLIFICATION_EFFECT.csv`, `..._SUMMARY.csv` | Effect of γ on prominence |
| `VALIDATION_SRPI_VULNERABILITY_SCENARIOS.csv` | Vulnerability scaling scenarios |
| `VALIDATION_POLARITY_SRPI_SENSITIVITY.csv` | SRPI under polarity reversal |
| `VALIDATION_PRIMARY_CONSISTENCY_CHECK.csv` | Consistency of the primary model across scripts |
| `VALIDATION_PARAMETER_GRID.csv` | All parameters used in the run |

### Stage 6, `MCDM_Visualization.py`

| File | Content |
|---|---|
| `figures_v3/fig02` to `fig14` | Figures (PNG, 300 dpi) |
| `SUPPLEMENTARY_REVIEW_TABLES.xlsx` | Supplementary tables with `MANIFEST` and `PARAMETER_GRID` sheets |
| `REPRODUCIBILITY_OUTPUT_MANIFEST.csv` | List of output files with the producing script |
| `FIG_RANK_RECONCILIATION_USED.csv` | Rank table used by the figures |

## Figure index

| File | Content |
|---|---|
| `fig02_weight_trajectory_and_threshold_sensitivity` | Weight trajectories and κc sensitivity |
| `fig03_ahp_permutation_invariance` | Criterion-order permutation test |
| `fig04_dematel_connectedness` | DEMATEL connectedness |
| `fig05_prominence_trajectory` | Prominence under stress |
| `fig06_primary_vs_exploratory_ranking` | Primary TOPSIS vs exploratory SRPI ranking |
| `fig07_robustness_summary` | Robustness across seven validation tests |
| `fig08_comparative_variants` | Comparative method variants |
| `fig09_monte_carlo_ahp_benchmark` | Monte Carlo envelopes of criterion weights |
| `fig10_benefit_cost_allocation_sensitivity` | Outcome/resource allocation sensitivity |
| `fig11_raw_vs_7525` | Raw FAHP vs 75/25 |
| `fig12_esi_vs_mc` | ESI vs Monte Carlo dispersion |
| `fig13_esi_full_vs_cr` | ESI vs CR-restricted ESI |
| `fig14_dematel_centrality` | D+R vs weighted degree and eigenvector centrality |

Figure numbers in the file names follow the script output and can differ from the numbering in the manuscript.

## Reproducibility notes

* Every figure carries a stamp with the primary weighting, the reverse-coded criteria, and the polarity verification flag, read from `VALIDATION_PARAMETER_GRID.csv`.
* To switch the primary model to raw FAHP weights, set `PRIMARY_WEIGHTING = "raw"` in `MCDM_Topsis.py` and rerun stages 2 to 6.
* Blank or malformed cells in the input workbooks trigger warnings (`MCDM_AHP.py` replaces blank cells by the neutral TFN (1,1,1), crisp values are replicated as (v,v,v)).
* `MCDM_AHP.py` requires a tabulated random index for the matrix size (n up to 15 is included).

## Citation

If you use this code, please cite the manuscript.

```

```

## License



## Contact

Widyasmoro Priatmojo, Constructor University
