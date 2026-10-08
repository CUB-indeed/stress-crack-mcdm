

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

PROJECT_DIR = os.environ.get("MCDM_PROJECT_DIR", "/home/coder/project")
# Make sure MCDM_AHP / MCDM_Topsis / MCDM_Dematel (imported below) resolve their
# own PROJECT_DIR to the SAME folder (MCDM_AHP defaults to os.getcwd()).
os.environ.setdefault("MCDM_PROJECT_DIR", PROJECT_DIR)

DFA_FILE = f"{PROJECT_DIR}/dfa.xlsx"
DFT_FILE = f"{PROJECT_DIR}/dft.xlsx"
DFD_FILE = f"{PROJECT_DIR}/dfd.xlsx"

FAHP_WEIGHTS_FILE   = f"{PROJECT_DIR}/STRESS_FAHP_RESULTS.csv"
TOPSIS_RESULT_FILE  = f"{PROJECT_DIR}/STRESS_TOPSIS_RESULT.csv"
DEMATEL_RESULTS_FILE = f"{PROJECT_DIR}/STRESS_DEMATEL_RESULTS.csv"
GAMMA_SWEEP_FILE     = f"{PROJECT_DIR}/DEMATEL_gamma_sensitivity.csv"
FUSION_RESULT_FILE   = f"{PROJECT_DIR}/Final_Stress_Coupled_Result.csv"

OUTPUT_SUMMARY_FILE      = f"{PROJECT_DIR}/VALIDATION_ROBUSTNESS_RESULTS.csv"
OUTPUT_SUMMARY_AGG_FILE  = f"{PROJECT_DIR}/VALIDATION_ROBUSTNESS_SUMMARY.csv"
OUTPUT_PERMUTATION_FILE  = f"{PROJECT_DIR}/VALIDATION_PERMUTATION_INVARIANCE.csv"
OUTPUT_MONTECARLO_FILE   = f"{PROJECT_DIR}/VALIDATION_MONTE_CARLO_AHP.csv"
OUTPUT_CRACK_DIAG_FILE   = f"{PROJECT_DIR}/VALIDATION_CRACK_THRESHOLD_DIAGNOSTIC.csv"
OUTPUT_STRESSGRID_DIAG_FILE = f"{PROJECT_DIR}/VALIDATION_STRESS_GRID_DIAGNOSTIC.csv"
OUTPUT_STRESSGRID_RANK_FILE = f"{PROJECT_DIR}/VALIDATION_STRESS_GRID_RANK_AGREEMENT.csv"
OUTPUT_SUMMARY_ALL_FILE = f"{PROJECT_DIR}/VALIDATION_ROBUSTNESS_SUMMARY_ALL_RUNS.csv"
OUTPUT_IDENTITY_FILE = f"{PROJECT_DIR}/VALIDATION_IMPLEMENTATION_EQUIVALENCE.csv"
OUTPUT_VULN_SCENARIO_FILE = f"{PROJECT_DIR}/VALIDATION_SRPI_VULNERABILITY_SCENARIOS.csv"
OUTPUT_AMP_FEATURE_FILE = f"{PROJECT_DIR}/VALIDATION_STRESS_AMPLIFICATION_EFFECT.csv"
OUTPUT_AMP_SUMMARY_FILE = f"{PROJECT_DIR}/VALIDATION_STRESS_AMPLIFICATION_SUMMARY.csv"

# --- added  ---
OUTPUT_ESI_MC_FILE = f"{PROJECT_DIR}/VALIDATION_ESI_vs_MC_STABILITY.csv"
OUTPUT_ESI_MC_SUMMARY_FILE = f"{PROJECT_DIR}/VALIDATION_ESI_vs_MC_SUMMARY.csv"
OUTPUT_POLARITY_SRPI_FILE = f"{PROJECT_DIR}/VALIDATION_POLARITY_SRPI_SENSITIVITY.csv"
OUTPUT_PRIMARY_CHECK_FILE = f"{PROJECT_DIR}/VALIDATION_PRIMARY_CONSISTENCY_CHECK.csv"
OUTPUT_PARAM_GRID_FILE = f"{PROJECT_DIR}/VALIDATION_PARAMETER_GRID.csv"

EXPECTED_K = 8
WEIGHT_PERTURBATION_PCT = 0.05

LAMBDA_AMP = 1.0
MU_DISC = 1.0

KAPPA_THRESHOLD_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)

STRESS_STEP_GRID = (0.05, 0.10, 0.25)
STRESS_RANGE_GRID = (0.25, 0.50, 0.75, 1.00)

N_PERMUTATIONS = 200
PERMUTATION_SEED = 42
PERMUTATION_TOLERANCE = 1e-6

MC_N_DRAWS = 2000
MC_SIGMA = 0.15
MC_SEED = 123

LAMBDA_MU_GRID = (0, 0.25, 0.5, 1, 1.5, 2)

ESI_MC_STRONG_RHO = -0.70          # effect-size LABEL only (not sufficient on its own, n = 11)
ESI_MC_PRIMARY_MEASURE = "CV"
ESI_MC_BOOT_B = 5000               # bootstrap resamples of the criteria (CI of rho)
ESI_MC_PERM_B = 20000              # permutation resamples (two-sided p of rho)
ESI_MC_ALPHA = 0.05                # (1 - alpha) bootstrap interval; excluding 0 is REQUIRED

PRIMARY_CONSISTENCY_TOL = 1e-8

sys.path.insert(0, PROJECT_DIR)

try:
    import MCDM_AHP as stress_fahp
    import MCDM_Topsis as stress_topsis
    import MCDM_Dematel as stress_dematel
except ImportError as e:
    raise ImportError(
        "[VALIDATION] Could not import MCDM_AHP / MCDM_Topsis / MCDM_Dematel. "
        f"Check that PROJECT_DIR ('{PROJECT_DIR}') is correct and that all "
        f"three modules are on that path. Original error: {e}"
    )

# The revised MCDM_Topsis / MCDM_AHP are required (single weighting entry point,
# polarity switch, CR-admissible ESI helpers).
for _fn in ("compute_weights", "primary_weights", "endogenous_group_shares",
            "polarity_columns"):
    if not hasattr(stress_topsis, _fn):
        raise ImportError(
            f"[VALIDATION] MCDM_Topsis has no '{_fn}': use the revised MCDM_Topsis.py "
            f"(review)."
        )
for _fn in ("admissible_transitions", "mean_kappa_over_transitions"):
    if not hasattr(stress_fahp, _fn):
        raise ImportError(
            f"[VALIDATION] MCDM_AHP has no '{_fn}': use the revised MCDM_AHP.py."
        )

# Single source of truth for the primary model: whatever MCDM_Topsis is configured with.
PRIMARY_WEIGHTING = stress_topsis.PRIMARY_WEIGHTING

NAME_MAP = {
    "NLP for ESG Reporting": "NLP",
    "Sentiment & Reputation": "Sentiment",
    "Predictive ESG Risk": "Predictive",
    "Causal AI Risk": "Causal AI",
    "Automated ESG Scoring": "Automated ESG",
    "Fraud & Greenwashing": "Fraud & Greenwashing",
    "Explainable AI (XAI)": "XAI",
    "Federated Learning": "Federated Learning",
    "Secure ESG Integration": "Secure ESG",
}

EPS = 1e-12

def minmax(x):
    x = np.asarray(x, dtype=float)
    denom = x.max() - x.min()
    if denom == 0:
        return np.zeros(len(x))
    return (x - x.min()) / (denom + EPS)

def minmax_complement(x):
    """
    (Eq. 32, mirrored from
    MCDM_FINAL.py: V_x is the instability score and must increase as
    stability (FSI) decreases, i.e. V_x = [max(FSI)-FSI_x]/[max(FSI)-min(FSI)].
    The previous minmax(FSI) penalized the MOST stable features the most,
    which is the opposite of the intended vulnerability discount in Eq. (33)
    and corrupted every downstream sensitivity/validation test below.
    """
    x = np.asarray(x, dtype=float)
    denom = x.max() - x.min()
    if denom == 0:
        return np.zeros(len(x))
    return (x.max() - x) / (denom + EPS)

def vulnerability_scores(FSI, mode="absolute"):
    FSI = np.asarray(FSI, dtype=float)
    if mode == "absolute":
        return 1.0 - FSI
    if mode == "minmax":
        return minmax_complement(FSI)
    if mode == "zero":
        return np.zeros(len(FSI))
    raise ValueError(mode)

VULNERABILITY_MODE = "absolute"

def compute_srpi(RC, DR, FSI, lambda_amp=LAMBDA_AMP, mu_disc=MU_DISC, vulnerability_mode=None):
    mode = VULNERABILITY_MODE if vulnerability_mode is None else vulnerability_mode
    Px = minmax(RC)
    Cx = minmax(DR)
    Vx = vulnerability_scores(FSI, mode)
    Phi = Px * (1 + lambda_amp * Cx) / (1 + mu_disc * Vx)
    Phi_max = Phi.max() if Phi.max() > EPS else EPS
    return Phi / Phi_max

def rank_correlation(baseline_series, new_series):
    common = baseline_series.index.intersection(new_series.index)
    if len(common) < 3:
        warnings.warn(
            f"[VALIDATION] Only {len(common)} overlapping features "
            f"found; correlation may be unreliable."
        )
    b = baseline_series.loc[common].astype(float).values
    n = new_series.loc[common].astype(float).values

    rho, p_rho = spearmanr(b, n)
    tau, p_tau = kendalltau(b, n)

    m = len(common)
    concordant = discordant = tied = 0
    for i in range(m):
        for j in range(i + 1, m):
            db = b[i] - b[j]
            dn = n[i] - n[j]
            if db == 0 or dn == 0:
                tied += 1
            elif (db > 0) == (dn > 0):
                concordant += 1
            else:
                discordant += 1

    return {
        "spearman_rho": rho, "spearman_p": p_rho,
        "kendall_tau": tau, "kendall_p": p_tau,
        "n_common": m,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
    }

def load_baseline_srpi():
    fused = pd.read_csv(FUSION_RESULT_FILE)
    fused["Feature"] = fused["Feature"].astype(str).str.strip()
    srpi_col = "SRPI"
    return fused.set_index("Feature")[srpi_col]

def load_baseline_topsis_ranking():
    """Primary TOPSIS ranking as written by MCDM_Topsis.py (weighting = PRIMARY_WEIGHTING)."""
    df = pd.read_csv(TOPSIS_RESULT_FILE)
    df["Feature"] = df["Method"].replace(NAME_MAP)
    return df.set_index("Feature")["RC"]

def weight_sensitivity_test(pert_pct=WEIGHT_PERTURBATION_PCT):
    print("[TEST 1] Weight sensitivity (+/-5% per criterion)...")

    raw_dfs = stress_topsis.load_expert_sheets(DFT_FILE)
    _, criteria = stress_topsis.validate_expert_sheets(raw_dfs, expected_k=EXPECTED_K)
    tfn_dfs = stress_topsis.parse_all_sheets(raw_dfs)
    agg_df, _K = stress_topsis.aggregate_experts_arithmetic(tfn_dfs)

    benefit_cols = stress_topsis.POLARITY_BENEFIT_COLS
    cost_cols = stress_topsis.POLARITY_COST_COLS
    norm_df, _, _ = stress_topsis.normalize_fuzzy_matrix(agg_df, benefit_cols, cost_cols)

    w_star = stress_topsis.load_fahp_weights(FAHP_WEIGHTS_FILE)

    dematel_df = pd.read_csv(DEMATEL_RESULTS_FILE)
    dematel_df["Feature"] = dematel_df["Feature"].astype(str).str.strip()
    dematel_lookup = dematel_df.set_index("Feature")[["Prominence_star", "FSI"]]

    baseline_srpi = load_baseline_srpi()

    records = []
    for crit in w_star.keys():
        for sign, label in [(+1, f"+{int(pert_pct*100)}%"), (-1, f"-{int(pert_pct*100)}%")]:
            w_pert = dict(w_star)
            w_pert[crit] = w_pert[crit] * (1 + sign * pert_pct)
            total = sum(w_pert.values())
            w_pert = {k: v / total for k, v in w_pert.items()}

            w_hat = stress_topsis.compute_weights(w_pert, PRIMARY_WEIGHTING)
            weighted_df = stress_topsis.build_weighted_matrix(norm_df, w_hat)
            fpis, fnis = stress_topsis.compute_fpis_fnis(weighted_df)
            Dp, Dm = stress_topsis.compute_separations(weighted_df, fpis, fnis)
            CC = stress_topsis.compute_closeness(Dp, Dm)

            RC_new = pd.Series(CC)
            RC_new.index = [NAME_MAP.get(m, m) for m in RC_new.index]

            merged = pd.DataFrame({"RC": RC_new}).join(dematel_lookup, how="inner")
            srpi_vals = compute_srpi(merged["RC"], merged["Prominence_star"], merged["FSI"])
            new_series = pd.Series(srpi_vals, index=merged.index)

            audit = rank_correlation(baseline_srpi, new_series)
            records.append({
                "test": "weight_sensitivity", "criterion": crit, "perturbation": label,
                "is_identity_run": False,
                **audit,
            })

    return pd.DataFrame(records)

def gamma_sensitivity_test():
    print("[TEST 2] Stress-coefficient (gamma) sensitivity...")

    gamma_df = pd.read_csv(GAMMA_SWEEP_FILE)
    gamma_df["Feature"] = gamma_df["feature"].astype(str).str.strip()

    topsis_df = pd.read_csv(TOPSIS_RESULT_FILE)
    topsis_df["Feature"] = topsis_df["Method"].replace(NAME_MAP)

    dematel_df = pd.read_csv(DEMATEL_RESULTS_FILE)
    dematel_df["Feature"] = dematel_df["Feature"].astype(str).str.strip()

    baseline_srpi = load_baseline_srpi()

    records = []
    for gamma_val, grp in gamma_df.groupby("gamma"):
        merged = grp.merge(topsis_df[["Feature", "RC"]], on="Feature", how="inner")
        merged = merged.merge(dematel_df[["Feature", "FSI"]], on="Feature", how="inner")

        srpi_vals = compute_srpi(merged["RC"], merged["Prominence_star"], merged["FSI"])
        new_series = pd.Series(srpi_vals, index=merged["Feature"])

        audit = rank_correlation(baseline_srpi, new_series)
        records.append({
            "test": "gamma_sensitivity", "gamma": gamma_val,
            "is_identity_run": bool(abs(float(gamma_val) - stress_dematel.GAMMA_AMPLIFICATION) < 1e-12),
            **audit,
        })

    return pd.DataFrame(records)

def leave_one_expert_out_test():
    print("[TEST 3] Leave-one-expert-out...")

    baseline_srpi = load_baseline_srpi()
    records = []

    fahp_raw = stress_fahp.load_expert_sheets(DFA_FILE)
    fahp_tfn = stress_fahp.parse_all_sheets(fahp_raw)
    expert_names = list(fahp_tfn.keys())

    topsis_raw = stress_topsis.load_expert_sheets(DFT_FILE)
    stress_topsis.validate_expert_sheets(topsis_raw, expected_k=len(expert_names))
    topsis_tfn = stress_topsis.parse_all_sheets(topsis_raw)

    dematel_raw = stress_dematel.load_expert_sheets(DFD_FILE)
    dematel_tfn = stress_dematel.parse_all_sheets(dematel_raw)

    benefit_cols = stress_topsis.POLARITY_BENEFIT_COLS
    cost_cols = stress_topsis.POLARITY_COST_COLS

    for expert_out in expert_names:

        fahp_subset = {k: v for k, v in fahp_tfn.items() if k != expert_out}
        agg_fuzzy, _E = stress_fahp.aggregate_experts_geometric(fahp_subset)
        agg_fuzzy = stress_fahp.enforce_unit_diagonal_fuzzy(agg_fuzzy)
        agg_fuzzy = stress_fahp.enforce_fuzzy_reciprocity(agg_fuzzy)
        A0, labels = stress_fahp.build_crisp_matrix(agg_fuzzy)
        A0 = stress_fahp.enforce_crisp_reciprocity(A0)
        n = len(labels)
        ri_n = stress_fahp.get_random_index(n)
        _lmax0, w0 = stress_fahp.principal_eigenpair(A0)

        w_star_arr = stress_fahp.compute_stable_weight(w0)
        w_star_loo = dict(zip(labels, w_star_arr))

        topsis_subset = {k: v for k, v in topsis_tfn.items() if k != expert_out}
        agg_df, _K = stress_topsis.aggregate_experts_arithmetic(topsis_subset)
        w_hat = stress_topsis.compute_weights(w_star_loo, PRIMARY_WEIGHTING)
        norm_df, _, _ = stress_topsis.normalize_fuzzy_matrix(agg_df, benefit_cols, cost_cols)
        weighted_df = stress_topsis.build_weighted_matrix(norm_df, w_hat)
        fpis, fnis = stress_topsis.compute_fpis_fnis(weighted_df)
        Dp, Dm = stress_topsis.compute_separations(weighted_df, fpis, fnis)
        CC = stress_topsis.compute_closeness(Dp, Dm)
        RC_new = pd.Series(CC)
        RC_new.index = [NAME_MAP.get(m, m) for m in RC_new.index]

        dematel_subset = {k: v for k, v in dematel_tfn.items() if k != expert_out}
        agg_fuzzy_d = stress_dematel.aggregate_experts_arithmetic(dematel_subset)
        agg_fuzzy_d = stress_dematel.enforce_zero_diagonal_fuzzy(agg_fuzzy_d)
        Z0, labels_d = stress_dematel.build_crisp_matrix(agg_fuzzy_d)
        Z0 = stress_dematel.symmetrize_matrix(Z0)
        X0, _s0 = stress_dematel.normalize_matrix(Z0)
        T0, _rho0 = stress_dematel.total_relation_matrix(X0)
        D0, R0, _prom0, _rel0 = stress_dematel.compute_D_R(T0)
        c0 = stress_dematel.compute_stress_sensitivity(
            Z0=Z0, D0=D0, R0=R0, source=stress_dematel.STRESS_SENSITIVITY_SOURCE
        )
        _D_traj, _R_traj, prom_traj, rel_traj = stress_dematel.stress_trajectory(
            Z0, stress_dematel.STRESS_SET, c0
        )
        delta_bar = stress_dematel.compute_mean_crack_index(prom_traj, rel_traj, labels_d)
        fsi = stress_dematel.compute_fsi(delta_bar)
        Z_star = stress_dematel.amplify_matrix(Z0, delta_bar, gamma=stress_dematel.GAMMA_AMPLIFICATION)
        X_star, _s_star = stress_dematel.normalize_matrix(Z_star)
        T_star, _rho_star = stress_dematel.total_relation_matrix(X_star)
        _D_star, _R_star, prom_star, _rel_star = stress_dematel.compute_D_R(T_star)

        dematel_loo = pd.DataFrame({
            "Feature": [str(f).strip() for f in labels_d],
            "Prominence_star": prom_star,
            "FSI": fsi,
        }).set_index("Feature")

        merged = pd.DataFrame({"RC": RC_new}).merge(
            dematel_loo, left_index=True, right_index=True, how="inner"
        )
        srpi_vals = compute_srpi(merged["RC"], merged["Prominence_star"], merged["FSI"])
        new_series = pd.Series(srpi_vals, index=merged.index)

        audit = rank_correlation(baseline_srpi, new_series)
        records.append({
            "test": "leave_one_expert_out", "expert_excluded": expert_out,
            "is_identity_run": False, **audit,
        })

    return pd.DataFrame(records)

def comparative_variants_test():
    print("[TEST 4] Comparative robustness vs reduced variants...")

    baseline_srpi = load_baseline_srpi()
    records = []

    fahp_results = pd.read_csv(FAHP_WEIGHTS_FILE)
    topsis_df = pd.read_csv(TOPSIS_RESULT_FILE)
    topsis_df["Feature"] = topsis_df["Method"].replace(NAME_MAP)
    dematel_df = pd.read_csv(DEMATEL_RESULTS_FILE)
    dematel_df["Feature"] = dematel_df["Feature"].astype(str).str.strip()

    raw_dfs = stress_topsis.load_expert_sheets(DFT_FILE)
    tfn_dfs = stress_topsis.parse_all_sheets(raw_dfs)
    agg_df, _K = stress_topsis.aggregate_experts_arithmetic(tfn_dfs)
    benefit_cols = stress_topsis.POLARITY_BENEFIT_COLS
    cost_cols = stress_topsis.POLARITY_COST_COLS
    norm_df, _, _ = stress_topsis.normalize_fuzzy_matrix(agg_df, benefit_cols, cost_cols)

    w0_dict = dict(zip(fahp_results["Criterion"], fahp_results["w0_baseline"]))
    w_hat_0 = stress_topsis.compute_weights(w0_dict, PRIMARY_WEIGHTING)
    weighted_df0 = stress_topsis.build_weighted_matrix(norm_df, w_hat_0)
    fpis0, fnis0 = stress_topsis.compute_fpis_fnis(weighted_df0)
    Dp0, Dm0 = stress_topsis.compute_separations(weighted_df0, fpis0, fnis0)
    CC0 = stress_topsis.compute_closeness(Dp0, Dm0)
    RC_noCrack = pd.Series(CC0)
    RC_noCrack.index = [NAME_MAP.get(m, m) for m in RC_noCrack.index]

    mergedA = pd.DataFrame({"RC": RC_noCrack}).join(
        dematel_df.set_index("Feature")[["Prominence_star", "FSI"]], how="inner"
    )
    srpi_A = compute_srpi(mergedA["RC"], mergedA["Prominence_star"], mergedA["FSI"])
    seriesA = pd.Series(srpi_A, index=mergedA.index)
    audit = rank_correlation(baseline_srpi, seriesA)
    records.append({
        "test": "comparative_variant",
        "variant": "FAHP-TOPSIS_w0_explicit (== primary under AHP)",
        "is_identity_run": True,
        **audit,
    })

    mergedB = topsis_df.set_index("Feature")[["RC"]].join(
        dematel_df.set_index("Feature")[["Prominence0", "FSI"]], how="inner"
    )
    srpi_B = compute_srpi(mergedB["RC"], mergedB["Prominence0"], mergedB["FSI"])
    seriesB = pd.Series(srpi_B, index=mergedB.index)
    audit = rank_correlation(baseline_srpi, seriesB)
    records.append({
        "test": "comparative_variant", "variant": "FAHP-DEMATEL_no_stress_amplification",
        "is_identity_run": False, **audit,
    })

    w_star = stress_topsis.load_fahp_weights(FAHP_WEIGHTS_FILE)
    w_hat_star = stress_topsis.compute_weights(w_star, PRIMARY_WEIGHTING)

    crisp_df = agg_df.apply(lambda col: col.map(lambda tfn: tfn[1])).astype(float)

    norm_crisp = crisp_df.copy()
    for col in crisp_df.columns:
        denom = np.sqrt((crisp_df[col] ** 2).sum())
        denom = denom if denom > EPS else EPS
        norm_crisp[col] = crisp_df[col] / denom

    weighted_crisp = norm_crisp.copy()
    for col in norm_crisp.columns:
        weighted_crisp[col] = norm_crisp[col] * w_hat_star[col]

    ideal_pos, ideal_neg = {}, {}
    for col in weighted_crisp.columns:
        if col in benefit_cols:
            ideal_pos[col] = weighted_crisp[col].max()
            ideal_neg[col] = weighted_crisp[col].min()
        else:
            ideal_pos[col] = weighted_crisp[col].min()
            ideal_neg[col] = weighted_crisp[col].max()

    ideal_pos_s = pd.Series(ideal_pos)
    ideal_neg_s = pd.Series(ideal_neg)

    Dp_c = np.sqrt(((weighted_crisp - ideal_pos_s) ** 2).sum(axis=1))
    Dm_c = np.sqrt(((weighted_crisp - ideal_neg_s) ** 2).sum(axis=1))
    denom_c = (Dp_c + Dm_c).replace(0, EPS)
    CC_crisp = Dm_c / denom_c

    RC_crisp = CC_crisp.copy()
    RC_crisp.index = [NAME_MAP.get(m, m) for m in RC_crisp.index]

    mergedC = pd.DataFrame({"RC": RC_crisp}).join(
        dematel_df.set_index("Feature")[["Prominence_star", "FSI"]], how="inner"
    )
    srpi_C = compute_srpi(mergedC["RC"], mergedC["Prominence_star"], mergedC["FSI"])
    seriesC = pd.Series(srpi_C, index=mergedC.index)
    audit = rank_correlation(baseline_srpi, seriesC)
    records.append({
        "test": "comparative_variant", "variant": "standard_crisp_TOPSIS",
        "is_identity_run": False, **audit,
    })

    return pd.DataFrame(records)

def _build_baseline_ahp_matrix():
    fahp_raw = stress_fahp.load_expert_sheets(DFA_FILE)
    fahp_tfn = stress_fahp.parse_all_sheets(fahp_raw)
    agg_fuzzy, _E = stress_fahp.aggregate_experts_geometric(fahp_tfn)
    agg_fuzzy = stress_fahp.enforce_unit_diagonal_fuzzy(agg_fuzzy)
    agg_fuzzy = stress_fahp.enforce_fuzzy_reciprocity(agg_fuzzy)
    A0, labels = stress_fahp.build_crisp_matrix(agg_fuzzy)
    A0 = stress_fahp.enforce_crisp_reciprocity(A0)
    return A0, labels

def _load_common_topsis_context():
    raw_dfs = stress_topsis.load_expert_sheets(DFT_FILE)
    tfn_dfs = stress_topsis.parse_all_sheets(raw_dfs)
    agg_df, _K = stress_topsis.aggregate_experts_arithmetic(tfn_dfs)
    benefit_cols = stress_topsis.POLARITY_BENEFIT_COLS
    cost_cols = stress_topsis.POLARITY_COST_COLS
    norm_df, _, _ = stress_topsis.normalize_fuzzy_matrix(agg_df, benefit_cols, cost_cols)

    dematel_df = pd.read_csv(DEMATEL_RESULTS_FILE)
    dematel_df["Feature"] = dematel_df["Feature"].astype(str).str.strip()
    dematel_lookup = dematel_df.set_index("Feature")[["Prominence_star", "FSI"]]
    return agg_df, norm_df, benefit_cols, cost_cols, dematel_lookup

def ahp_permutation_invariance_test(n_permutations=N_PERMUTATIONS, seed=PERMUTATION_SEED):
    print(f"[TEST 5] AHP criterion-permutation invariance ({n_permutations} permutations)...")

    A0, labels = _build_baseline_ahp_matrix()
    n = len(labels)

    _lmax0, w0 = stress_fahp.principal_eigenpair(A0)
    w_star0 = stress_fahp.compute_stable_weight(w0)
    w_star0_map = dict(zip(labels, w_star0))

    rng = np.random.default_rng(seed)
    records = []
    for trial in range(n_permutations):
        perm = rng.permutation(n)
        A_perm = A0[np.ix_(perm, perm)]
        labels_perm = [labels[i] for i in perm]

        _lmax_p, w0_p = stress_fahp.principal_eigenpair(A_perm)
        w_star_p = stress_fahp.compute_stable_weight(w0_p)

        w_star_p_map = dict(zip(labels_perm, w_star_p))
        w_star_p_reordered = np.array([w_star_p_map[l] for l in labels])
        w_star0_arr = np.array([w_star0_map[l] for l in labels])

        abs_dev = np.abs(w_star_p_reordered - w_star0_arr)
        records.append({
            "test": "ahp_permutation_invariance",
            "trial": trial,
            "max_abs_dev": float(abs_dev.max()),
            "mean_abs_dev": float(abs_dev.mean()),
            "invariant_within_tolerance": bool(abs_dev.max() < PERMUTATION_TOLERANCE),
        })

    df = pd.DataFrame(records)
    frac_invariant = df["invariant_within_tolerance"].mean()
    print(
        f"    -> {frac_invariant*100:.1f}% of permutations left w* unchanged "
        f"(tolerance={PERMUTATION_TOLERANCE})."
    )
    return df

def crack_threshold_sensitivity_test(kappa_grid=KAPPA_THRESHOLD_GRID):
    print("[TEST 6] Crack-threshold (kappa_threshold) sensitivity (diagnostic only)...")

    A0, labels = _build_baseline_ahp_matrix()
    n = len(labels)
    ri_n = stress_fahp.get_random_index(n)

    w_traj, _l, _CI, CR_traj = stress_fahp.stress_scenario_eigen_weights(
        A0, stress_fahp.STRESS_SET, n, ri_n
    )
    kappa_bar = stress_fahp.compute_mean_kappa_index(w_traj, labels)
    esi = stress_fahp.compute_esi(kappa_bar)

    records = []
    for kappa_threshold in kappa_grid:
        crack_table, B_star = stress_fahp.crack_detection(
            w_traj, CR_traj, labels, kappa_threshold=kappa_threshold
        )
        n_crack_events = int(crack_table["crack"].sum())
        criteria_flagged = sorted(crack_table.loc[crack_table["crack"], "criterion"].unique().tolist())

        for i, lab in enumerate(labels):
            records.append({
                "test": "crack_threshold_sensitivity",
                "kappa_threshold": kappa_threshold,
                "criterion": lab,
                "kappa_bar": float(kappa_bar[i]),
                "ESI": float(esi[i]),
                "criterion_ever_flagged_cracked": lab in criteria_flagged,
                "n_crack_events_total_at_this_threshold": n_crack_events,
                "n_crackfree_transitions_Bstar": len(B_star),
            })

    df = pd.DataFrame(records)
    n_thresholds = df["kappa_threshold"].nunique()
    events_by_threshold = df.groupby("kappa_threshold")["n_crack_events_total_at_this_threshold"].first()
    print(
        f"    -> crack events flagged per threshold ({n_thresholds} thresholds tested): "
        f"{dict(events_by_threshold)}"
    )
    print("    -> kappa_bar/ESI are threshold-independent by construction (reported for reference).")
    return df

def stress_grid_and_range_sensitivity_test(step_grid=STRESS_STEP_GRID, range_grid=STRESS_RANGE_GRID):
    print("[TEST 7] Stress-grid resolution and stress-range sensitivity (diagnostic only)...")

    A0, labels = _build_baseline_ahp_matrix()
    n = len(labels)
    ri_n = stress_fahp.get_random_index(n)

    records = []
    for step in step_grid:
        for rng_ in range_grid:
            beta_grid = np.round(np.arange(-rng_, rng_ + 1e-9, step), 10)
            if 0.0 not in beta_grid:
                beta_grid = np.sort(np.append(beta_grid, 0.0))

            w_traj, _l, _CI, CR_traj = stress_fahp.stress_scenario_eigen_weights(
                A0, beta_grid, n, ri_n
            )
            kappa_bar = stress_fahp.compute_mean_kappa_index(w_traj, labels)
            esi = stress_fahp.compute_esi(kappa_bar)
            crack_table, B_star = stress_fahp.crack_detection(w_traj, CR_traj, labels)
            n_crack_events = int(crack_table["crack"].sum())

            for i, lab in enumerate(labels):
                records.append({
                    "test": "stress_grid_range_sensitivity",
                    "step_size": step,
                    "stress_range": rng_,
                    "n_grid_points": len(beta_grid),
                    "criterion": lab,
                    "kappa_bar": float(kappa_bar[i]),
                    "ESI": float(esi[i]),
                    "n_crack_events_total": n_crack_events,
                    "n_crackfree_transitions_Bstar": len(B_star),
                })

    return pd.DataFrame(records)

def benefit_cost_allocation_sensitivity_test(scenarios=None, include_raw=True):
    print("[TEST 8] Benefit/cost allocation sensitivity (splits + raw FAHP w*)...")

    if scenarios is None:
        scenarios = stress_topsis.ALLOCATION_SCENARIOS

    baseline_srpi = load_baseline_srpi()
    agg_df, norm_df, benefit_cols, cost_cols, dematel_lookup = _load_common_topsis_context()
    w_star = stress_topsis.load_fahp_weights(FAHP_WEIGHTS_FILE)

    # (label, benefit_share, cost_share, is_raw)
    scen_list = [(f"{int(round(bs*100))}/{int(round(cs*100))}", bs, cs, False) for bs, cs in scenarios]
    if include_raw:
        q_O, q_S = stress_topsis.endogenous_group_shares(w_star)
        scen_list.append((stress_topsis.RAW_SCENARIO_LABEL, q_O, q_S, True))

    records = []
    for label, benefit_share, cost_share, is_raw in scen_list:
        if is_raw:
            w_hat = stress_topsis.compute_weights(w_star, "raw")      # Eq. (17) NOT applied
        else:
            w_hat = stress_topsis.rescale_weights(
                w_star, stress_topsis.BENEFIT_CRITERIA, stress_topsis.COST_CRITERIA,
                benefit_share=benefit_share, cost_share=cost_share,
            )
        weighted_df = stress_topsis.build_weighted_matrix(norm_df, w_hat)
        fpis, fnis = stress_topsis.compute_fpis_fnis(weighted_df)
        Dp, Dm = stress_topsis.compute_separations(weighted_df, fpis, fnis)
        CC = stress_topsis.compute_closeness(Dp, Dm)
        RC_new = pd.Series(CC)
        RC_new.index = [NAME_MAP.get(m, m) for m in RC_new.index]

        merged = pd.DataFrame({"RC": RC_new}).join(dematel_lookup, how="inner")
        srpi_vals = compute_srpi(merged["RC"], merged["Prominence_star"], merged["FSI"])
        new_series = pd.Series(srpi_vals, index=merged.index)

        audit = rank_correlation(baseline_srpi, new_series)
        # The scenario that coincides with the configured PRIMARY model is the identity run.
        if is_raw:
            is_primary = (PRIMARY_WEIGHTING == "raw")
        else:
            is_primary = (PRIMARY_WEIGHTING == "75_25"
                          and abs(benefit_share - stress_topsis.BENEFIT_SHARE) < 1e-9
                          and abs(cost_share - stress_topsis.COST_SHARE) < 1e-9)
        records.append({
            "test": "benefit_cost_allocation_sensitivity",
            "benefit_share": benefit_share, "cost_share": cost_share,
            "scenario": label, "is_raw_FAHP": bool(is_raw),
            "is_primary_baseline": bool(is_primary),
            "is_identity_run": bool(is_primary),
            **audit,
        })

    return pd.DataFrame(records)

_MC_CACHE = {}

def _mc_ahp_draws(n_draws=MC_N_DRAWS, sigma=MC_SIGMA, seed=MC_SEED):
    """The (single) set of Monte-Carlo AHP draws shared by Test 9 and the ESI-vs-MC
    validation. Same seed, same RNG call order as before -> identical draws."""
    key = (n_draws, sigma, seed)
    if key in _MC_CACHE:
        return _MC_CACHE[key]

    A0, labels = _build_baseline_ahp_matrix()
    n = len(labels)
    ri_n = stress_fahp.get_random_index(n)
    _lmax0, w0 = stress_fahp.principal_eigenpair(A0)

    rng = np.random.default_rng(seed)
    draws = np.zeros((n_draws, n))
    cr_draws = np.zeros(n_draws)
    for d in range(n_draws):
        A_pert = A0.copy()
        for i in range(n):
            for j in range(i + 1, n):
                noise = np.exp(rng.normal(0.0, sigma))
                A_pert[i, j] = A0[i, j] * noise
                A_pert[j, i] = 1.0 / A_pert[i, j]
        lmax_d, w_d = stress_fahp.principal_eigenpair(A_pert)
        draws[d, :] = w_d
        _CI_d, CR_d = stress_fahp.consistency_measures(lmax_d, n, ri_n)
        cr_draws[d] = CR_d

    out = {"A0": A0, "labels": labels, "n": n, "ri_n": ri_n, "w0": w0,
           "draws": draws, "cr_draws": cr_draws}
    _MC_CACHE[key] = out
    return out

def monte_carlo_ahp_benchmark_test(n_draws=MC_N_DRAWS, sigma=MC_SIGMA, seed=MC_SEED,
                                    cr_threshold=stress_fahp.CR_THRESHOLD):
    print(f"[TEST 9] Monte-Carlo / stochastic AHP benchmark ({n_draws} draws)...")

    mc = _mc_ahp_draws(n_draws, sigma, seed)
    labels, n, w0 = mc["labels"], mc["n"], mc["w0"]
    draws, cr_draws = mc["draws"], mc["cr_draws"]

    invalid_mask = cr_draws > cr_threshold
    n_invalid = int(invalid_mask.sum())
    invalid_rate = n_invalid / n_draws

    mean_w = draws.mean(axis=0)
    std_w = draws.std(axis=0)
    p5 = np.percentile(draws, 5, axis=0)
    p95 = np.percentile(draws, 95, axis=0)

    taus = [kendalltau(w0, draws[d, :])[0] for d in range(n_draws)]

    if n_invalid < n_draws:
        valid_draws = draws[~invalid_mask]
        mean_w_valid = valid_draws.mean(axis=0)
        taus_valid = [kendalltau(w0, valid_draws[d, :])[0] for d in range(valid_draws.shape[0])]
    else:
        mean_w_valid = np.full(n, np.nan)
        taus_valid = [np.nan]

    print(f"    -> CR-invalid draws (CR > {cr_threshold:.2f}): "
          f"{n_invalid}/{n_draws} ({invalid_rate:.1%}); "
          f"mean CR across all draws = {cr_draws.mean():.4f}, "
          f"max CR = {cr_draws.max():.4f}")

    records = []
    for i, lab in enumerate(labels):
        records.append({
            "test": "monte_carlo_ahp_benchmark", "criterion": lab,
            "w0_baseline": w0[i], "mc_mean": mean_w[i], "mc_std": std_w[i],
            "mc_p5": p5[i], "mc_p95": p95[i],
            "mc_mean_cr_valid_only": mean_w_valid[i],
        })
    records.append({
        "test": "monte_carlo_ahp_benchmark", "criterion": "__OVERALL_RANK_STABILITY_KENDALL_TAU__",
        "w0_baseline": np.nan,
        "mc_mean": float(np.mean(taus)), "mc_std": float(np.std(taus)),
        "mc_p5": float(np.percentile(taus, 5)), "mc_p95": float(np.percentile(taus, 95)),
        "mc_mean_cr_valid_only": float(np.nanmean(taus_valid)),
    })
    records.append({
        "test": "monte_carlo_ahp_benchmark", "criterion": "__CR_DIAGNOSTIC__",
        "w0_baseline": np.nan,
        "mc_mean": float(cr_draws.mean()), "mc_std": float(cr_draws.std()),
        "mc_p5": float(np.percentile(cr_draws, 5)), "mc_p95": float(np.percentile(cr_draws, 95)),
        "mc_mean_cr_valid_only": np.nan,
        "n_draws": n_draws, "n_cr_invalid_draws": n_invalid,
        "cr_invalid_rate": invalid_rate, "cr_threshold": cr_threshold,
    })
    return pd.DataFrame(records)

# ------------------------------------------------------------------
# ESI vs Monte-Carlo weight uncertainty
# ------------------------------------------------------------------

def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.isnan(a).any() or np.isnan(b).any() or np.ptp(a) < 1e-15 or np.ptp(b) < 1e-15:
        return np.nan, np.nan
    return float(spearmanr(a, b)[0]), float(kendalltau(a, b)[0])

def _partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z (rank-residual method)."""
    rx = pd.Series(np.asarray(x, dtype=float)).rank().values
    ry = pd.Series(np.asarray(y, dtype=float)).rank().values
    rz = pd.Series(np.asarray(z, dtype=float)).rank().values
    if np.isnan(rx).any() or np.isnan(ry).any():
        return np.nan
    A = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if np.ptp(ex) < 1e-12 or np.ptp(ey) < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])

def _rank(v):
    return pd.Series(np.asarray(v, dtype=float)).rank().values

def _fast_rho(rx, ry):
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else np.nan

def _boot_ci(stat_fn, n, rng, B, alpha):
    """Percentile bootstrap over the n criteria for a statistic stat_fn(idx)."""
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < 0.5 * B:
        return np.nan, np.nan
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))

def _spearman_with_inference(x, y, rng, B=ESI_MC_BOOT_B, P=ESI_MC_PERM_B, alpha=ESI_MC_ALPHA):
    """Spearman rho, percentile-bootstrap CI over criteria, two-sided permutation p."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); n = len(x)
    if n < 4 or np.isnan(x).any() or np.isnan(y).any():
        return np.nan, np.nan, np.nan, np.nan
    rx, ry = _rank(x), _rank(y)
    rho = _fast_rho(rx, ry)
    def stat(idx):
        a, b = x[idx], y[idx]
        if np.ptp(a) < 1e-14 or np.ptp(b) < 1e-14:
            return None
        return _fast_rho(_rank(a), _rank(b))
    lo, hi = _boot_ci(stat, n, rng, B, alpha)
    cnt = sum(abs(_fast_rho(rx, rng.permutation(ry))) >= abs(rho) - 1e-12 for _ in range(P))
    return rho, lo, hi, float((cnt + 1) / (P + 1))

def _partial_with_inference(x, y, z, rng, B=ESI_MC_BOOT_B, alpha=ESI_MC_ALPHA):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); z = np.asarray(z, dtype=float)
    n = len(x)
    pr = _partial_spearman(x, y, z)
    def stat(idx):
        a, b, c = x[idx], y[idx], z[idx]
        if np.ptp(a) < 1e-14 or np.ptp(b) < 1e-14 or np.ptp(c) < 1e-14:
            return None
        v = _partial_spearman(a, b, c)
        return None if np.isnan(v) else v
    lo, hi = _boot_ci(stat, n, rng, B, alpha)
    return pr, lo, hi

def _mc_dispersion(draws_subset, w0):
    if draws_subset.shape[0] < 3:
        nan = np.full(draws_subset.shape[1], np.nan)
        return {"mean": nan, "SD": nan, "CV": nan, "p5": nan, "p95": nan, "width": nan, "widthrel": nan}
    mean = draws_subset.mean(axis=0)
    sd = draws_subset.std(axis=0)
    p5 = np.percentile(draws_subset, 5, axis=0)
    p95 = np.percentile(draws_subset, 95, axis=0)
    width = p95 - p5
    return {
        "mean": mean, "SD": sd,
        "CV": sd / np.where(mean > EPS, mean, np.nan),
        "p5": p5, "p95": p95, "width": width,
        "widthrel": width / np.where(w0 > EPS, w0, np.nan),   # width / baseline weight
    }

def esi_vs_monte_carlo_test(n_draws=MC_N_DRAWS, sigma=MC_SIGMA, seed=MC_SEED,
                            cr_threshold=stress_fahp.CR_THRESHOLD):
    print("[TEST 2.3] ESI vs Monte-Carlo weight dispersion (same draws as Test 9)...")

    mc = _mc_ahp_draws(n_draws, sigma, seed)
    A0, labels, n, ri_n, w0 = mc["A0"], mc["labels"], mc["n"], mc["ri_n"], mc["w0"]
    draws, cr_draws = mc["draws"], mc["cr_draws"]
    rng_inf = np.random.default_rng(int(seed) + 7919)   # separate stream: the MC draws are unchanged
    valid_mask = ~(cr_draws > cr_threshold)

    # deterministic ESI from the existing stress path (full trajectory) and CR-admissible ESI
    w_traj, _l, _CI, CR_traj = stress_fahp.stress_scenario_eigen_weights(
        A0, stress_fahp.STRESS_SET, n, ri_n
    )
    esi = stress_fahp.compute_esi(stress_fahp.compute_mean_kappa_index(w_traj, labels))
    adm = stress_fahp.admissible_transitions(CR_traj, cr_threshold)
    esi_cr = stress_fahp.compute_esi(stress_fahp.mean_kappa_over_transitions(w_traj, labels, adm))

    subsets = {"all": np.ones(n_draws, dtype=bool), "valid": valid_mask}
    measures = ("SD", "CV", "width", "widthrel")

    table = pd.DataFrame({
        "criterion": labels, "w0_baseline": w0, "ESI": esi, "ESI_CR": esi_cr,
        "Rank_ESI": pd.Series(esi).rank(ascending=False, method="min").astype(int).values,
        "Rank_ESI_CR": (pd.Series(esi_cr).rank(ascending=False, method="min").astype(int).values
                        if not np.isnan(esi_cr).any() else np.nan),
    })

    disp = {}
    for sub, mask in subsets.items():
        d = _mc_dispersion(draws[mask], w0)
        disp[sub] = d
        table[f"MC_mean_{sub}"] = d["mean"]
        table[f"MC_p5_{sub}"] = d["p5"]
        table[f"MC_p95_{sub}"] = d["p95"]
        for m in measures:
            table[f"MC_{m}_{sub}"] = d[m]
            table[f"Rank_MCstab_{m}_{sub}"] = (
                pd.Series(d[m]).rank(ascending=True, method="min").astype("Int64").values
                if not np.isnan(d[m]).any() else np.nan
            )   # 1 = most stable (smallest dispersion)

    rows = []
    for sub in subsets:
        n_sub = int(subsets[sub].sum())
        for esi_name, esi_vec in (("ESI", esi), ("ESI_CR", esi_cr)):
            if np.isnan(esi_vec).any():
                continue
            rank_esi = pd.Series(esi_vec).rank(ascending=False, method="min").values   # 1 = most stable
            for m in measures:
                val = disp[sub][m]
                if np.isnan(val).any():
                    continue
                rank_mc = pd.Series(val).rank(ascending=True, method="min").values     # 1 = most stable
                rho_v, tau_v = _safe_corr(esi_vec, val)        # expected NEGATIVE if ESI tracks MC dispersion
                _r, ci_lo, ci_hi, p_perm = _spearman_with_inference(esi_vec, val, rng_inf)
                rho_r, tau_r = _safe_corr(rank_esi, rank_mc)   # expected POSITIVE (stability-rank agreement)
                rows.append({
                    "section": "esi_vs_mc", "draw_subset": sub, "n_draws_used": n_sub,
                    "ESI_variant": esi_name, "dispersion_measure": m,
                    "spearman_rho_ESI_vs_dispersion": rho_v, "kendall_tau_ESI_vs_dispersion": tau_v,
                    "spearman_rho_stability_ranks": rho_r, "kendall_tau_stability_ranks": tau_r,
                    "rho_ci95_lo": ci_lo, "rho_ci95_hi": ci_hi, "perm_p_two_sided": p_perm,
                    "n_criteria": n,
                })

    # confounder check: ESI (and MC dispersion) vs baseline weight magnitude
    for sub in subsets:
        n_sub = int(subsets[sub].sum())
        cv = disp[sub]["CV"]
        if np.isnan(cv).any():
            continue
        r1, t1 = _safe_corr(esi, w0)
        r2, t2 = _safe_corr(cv, w0)
        _a, c1lo, c1hi, p1 = _spearman_with_inference(esi, w0, rng_inf)
        _b, c2lo, c2hi, p2 = _spearman_with_inference(cv, w0, rng_inf)
        rows.append({"section": "confounder_w0", "draw_subset": sub, "n_draws_used": n_sub,
                     "ESI_variant": "ESI", "dispersion_measure": "w0 vs ESI",
                     "spearman_rho_ESI_vs_dispersion": r1, "kendall_tau_ESI_vs_dispersion": t1,
                     "rho_ci95_lo": c1lo, "rho_ci95_hi": c1hi, "perm_p_two_sided": p1, "n_criteria": n})
        rows.append({"section": "confounder_w0", "draw_subset": sub, "n_draws_used": n_sub,
                     "ESI_variant": "ESI", "dispersion_measure": "w0 vs CV",
                     "spearman_rho_ESI_vs_dispersion": r2, "kendall_tau_ESI_vs_dispersion": t2,
                     "rho_ci95_lo": c2lo, "rho_ci95_hi": c2hi, "perm_p_two_sided": p2, "n_criteria": n})
        pr, plo, phi = _partial_with_inference(esi, cv, w0, rng_inf)
        rows.append({"section": "partial_given_w0", "draw_subset": sub, "n_draws_used": n_sub,
                     "ESI_variant": "ESI", "dispersion_measure": "CV",
                     "spearman_rho_ESI_vs_dispersion": pr,
                     "rho_ci95_lo": plo, "rho_ci95_hi": phi, "n_criteria": n})

    # decision rule (primary measure, all draws): effect size + CI + partial given w0 (n is small)
    prim = [r for r in rows if r["section"] == "esi_vs_mc" and r["draw_subset"] == "all"
            and r["ESI_variant"] == "ESI" and r["dispersion_measure"] == ESI_MC_PRIMARY_MEASURE]
    part = [r for r in rows if r["section"] == "partial_given_w0" and r["draw_subset"] == "all"]
    rho_prim = prim[0]["spearman_rho_ESI_vs_dispersion"] if prim else np.nan
    ci_lo = prim[0]["rho_ci95_lo"] if prim else np.nan
    ci_hi = prim[0]["rho_ci95_hi"] if prim else np.nan
    p_prim = prim[0]["perm_p_two_sided"] if prim else np.nan
    prho = part[0]["spearman_rho_ESI_vs_dispersion"] if part else np.nan
    plo = part[0]["rho_ci95_lo"] if part else np.nan
    phi = part[0]["rho_ci95_hi"] if part else np.nan
    sig_neg = (not np.isnan(ci_hi)) and ci_hi < 0
    part_neg = (not np.isnan(phi)) and phi < 0
    # sign consistency across dispersion measures (all draws, plain ESI)
    meas_rows = [r for r in rows if r["section"] == "esi_vs_mc" and r["draw_subset"] == "all"
                 and r["ESI_variant"] == "ESI"]
    n_sig_neg = sum(1 for r in meas_rows if (not np.isnan(r["rho_ci95_hi"])) and r["rho_ci95_hi"] < 0)
    n_pos = sum(1 for r in meas_rows if r["spearman_rho_ESI_vs_dispersion"] > 0)
    if np.isnan(rho_prim):
        verdict, action = "NOT_COMPUTABLE", "Monte-Carlo dispersion undefined."
    elif sig_neg and part_neg:
        verdict = ("STRONG_NEGATIVE_ASSOCIATION" if rho_prim <= ESI_MC_STRONG_RHO
                   else "MODERATE_NEGATIVE_ASSOCIATION")
        action = ("ESI is supported as a stability diagnostic: the association with MC dispersion has a "
                  "CI below zero and survives control for w0.")
    elif sig_neg and not part_neg:
        verdict = "CONFOUNDED_BY_W0"
        action = ("The raw association is negative but does not survive control for baseline weight "
                  "magnitude: describe ESI as sensitivity to the power-stress path, partly driven by w0, "
                  "not as general weight uncertainty.")
    else:
        verdict = "INCONCLUSIVE_SMALL_N"
        action = ("With 11 criteria the bootstrap CI of rho(ESI, CV) includes 0, so the data do not support "
                  "an ESI-dispersion link; describe ESI specifically as sensitivity to the chosen "
                  "power-stress path, not as general weight uncertainty. Do not interpret a point "
                  "estimate against a cut-off.")
    rows.append({"section": "decision_rule", "draw_subset": "all", "n_draws_used": n_draws,
                 "ESI_variant": "ESI", "dispersion_measure": ESI_MC_PRIMARY_MEASURE,
                 "spearman_rho_ESI_vs_dispersion": rho_prim,
                 "rho_ci95_lo": ci_lo, "rho_ci95_hi": ci_hi, "perm_p_two_sided": p_prim,
                 "n_criteria": n, "verdict": verdict,
                 "note": action + f" Primary rho={rho_prim:.3f} (95% CI {ci_lo:.3f} to {ci_hi:.3f}; "
                 f"permutation p={p_prim:.3f}); partial rho given w0={prho:.3f} (95% CI {plo:.3f} to {phi:.3f}). "
                 f"Across the {len(meas_rows)} dispersion measures: {n_sig_neg} with CI below 0, {n_pos} with "
                 f"positive rho (measure-dependent sign means the link is not robust). "
                 f"n_criteria={n}; -0.70 is an effect-size label only. "
                 f"CR-valid draws: {int(valid_mask.sum())}/{n_draws}."})

    summary = pd.DataFrame(rows)
    print(f"    -> rho(ESI, {ESI_MC_PRIMARY_MEASURE}; all draws) = {rho_prim:.3f} "
          f"(95% CI {ci_lo:.3f}..{ci_hi:.3f}; perm p={p_prim:.3f}); partial|w0 = {prho:.3f}; verdict = {verdict}")
    return table, summary

def _tfn_vertex_distance(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return np.sqrt(((a1 - b1) ** 2 + (a2 - b2) ** 2 + (a3 - b3) ** 2) / 3.0)

def _tfn_euclidean_distance(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return np.sqrt((a1 - b1) ** 2 + (a2 - b2) ** 2 + (a3 - b3) ** 2)

def _generic_fpis_fnis(weighted_df, benefit_cols, cost_cols):
    fpis, fnis = {}, {}
    for col in weighted_df.columns:
        vals = weighted_df[col].tolist()
        is_tfn = isinstance(vals[0], (tuple, list, np.ndarray))
        arr = np.array(vals, dtype=float)
        if is_tfn:
            col_max = arr.max(axis=0)
            col_min = arr.min(axis=0)
        else:
            col_max = arr.max()
            col_min = arr.min()
        if col in benefit_cols:
            fpis[col], fnis[col] = col_max, col_min
        else:
            fpis[col], fnis[col] = col_min, col_max
    return fpis, fnis

def _generic_separations(weighted_df, fpis, fnis, distance_fn):
    """
    Linear accumulation: Dp/Dm = sum_j distance_fn(v_ij, ref_j).

    Valid whenever distance_fn already returns the FINAL per-criterion
    distance to be summed -- e.g. |a-b| for crisp L1 ("abs_dist"), or the
    TFN vertex/Euclidean functions above, which already take
    sqrt(sum of squares) INSIDE each per-criterion call before being
    summed linearly across criteria (the standard fuzzy-TOPSIS convention,
    Chen 1996).

    the scalar branch previously hardcoded abs(val - ref) regardless
    of distance_fn, so distance_fn was silently ignored for crisp columns.
    This made config 5 ("...euclidean_dist (=orig. comparator)") compute
    the exact same L1 distance as config 4 ("abs_dist") despite its label.
    This function is NOT valid for classic crisp Euclidean TOPSIS
    (Hwang & Yoon 1981), which needs sqrt(sum_j (v_ij-ref_j)^2) taken ONCE
    across the whole criteria vector -- use _generic_separations_l2() for
    that case (config 5 below).
    """
    Dp = pd.Series(0.0, index=weighted_df.index)
    Dm = pd.Series(0.0, index=weighted_df.index)
    for col in weighted_df.columns:
        for idx in weighted_df.index:
            val = weighted_df.loc[idx, col]
            if isinstance(val, (tuple, list, np.ndarray)):
                Dp.loc[idx] += distance_fn(tuple(val), tuple(fpis[col]))
                Dm.loc[idx] += distance_fn(tuple(val), tuple(fnis[col]))
            else:
                Dp.loc[idx] += distance_fn(val, fpis[col])
                Dm.loc[idx] += distance_fn(val, fnis[col])
    return Dp, Dm

def _generic_separations_l2(weighted_df, fpis, fnis):
    """
    (Test 10, config 5 "crisp_repr + vector_norm + euclidean_dist"):
    genuine Euclidean TOPSIS separation for crisp (scalar) criteria,
    D+ = sqrt(sum_j (v_ij - v_j+)^2) -- squared differences summed across
    ALL criteria first, square root taken once at the end. Mirrors the
    "standard_crisp_TOPSIS" variant already implemented correctly in
    comparative_variants_test() (Test 4), which config 5 is meant to
    reproduce under a different label.
    """
    fpis_s = pd.Series(fpis)
    fnis_s = pd.Series(fnis)
    Dp = np.sqrt(((weighted_df - fpis_s) ** 2).sum(axis=1))
    Dm = np.sqrt(((weighted_df - fnis_s) ** 2).sum(axis=1))
    return Dp, Dm

def _vector_normalize_fuzzy(agg_df):
    norm_df = agg_df.copy()
    for col in agg_df.columns:
        modal_vals = agg_df[col].apply(lambda tfn: tfn[1]).astype(float)
        denom = np.sqrt((modal_vals ** 2).sum())
        denom = denom if denom > EPS else EPS
        norm_df[col] = agg_df[col].apply(lambda tfn, d=denom: tuple(np.array(tfn, dtype=float) / d))
    return norm_df

def crisp_fuzzy_ablation_test():
    print("[TEST 10] Controlled crisp/fuzzy ablation (one factor at a time)...")

    agg_df, _norm_df_unused, benefit_cols, cost_cols, dematel_lookup = _load_common_topsis_context()
    w_star = stress_topsis.load_fahp_weights(FAHP_WEIGHTS_FILE)
    w_hat = stress_topsis.compute_weights(w_star, PRIMARY_WEIGHTING)
    baseline_srpi = load_baseline_srpi()

    def rc_from_weighted(weighted_df, distance_fn=None, use_l2=False, polarity_applied=False):
        # polarity_applied=True: the normalization already converted every criterion to a
        # higher-is-better scale (fuzzy_norm: x/u+ for benefit, l-/x for cost), so FPIS = max and
        # FNIS = min for ALL columns. Using the cost-type (min/max swapped) FPIS/FNIS here would
        # reverse the criterion twice. This only matters once REVERSE_CODED_CRITERIA is non-empty.
        # polarity_applied=False: vector normalization does NOT flip cost criteria, so the
        # ideal solutions must still swap min/max for cost columns.
        if polarity_applied:
            fpis, fnis = _generic_fpis_fnis(weighted_df, list(weighted_df.columns), [])
        else:
            fpis, fnis = _generic_fpis_fnis(weighted_df, benefit_cols, cost_cols)
        if use_l2:
            Dp, Dm = _generic_separations_l2(weighted_df, fpis, fnis)
        else:
            Dp, Dm = _generic_separations(weighted_df, fpis, fnis, distance_fn)
        denom = (Dp + Dm).replace(0, EPS)
        CC = Dm / denom
        RC = CC.copy()
        RC.index = [NAME_MAP.get(m, m) for m in RC.index]
        return RC

    configs = {}

    norm_fuzzy, _, _ = stress_topsis.normalize_fuzzy_matrix(agg_df, benefit_cols, cost_cols)
    weighted_fuzzy = stress_topsis.build_weighted_matrix(norm_fuzzy, w_hat)
    configs["fuzzy_repr + fuzzy_norm + vertex_dist (baseline)"] = rc_from_weighted(
        weighted_fuzzy, _tfn_vertex_distance, polarity_applied=True
    )

    configs["fuzzy_repr + fuzzy_norm + euclidean_dist"] = rc_from_weighted(
        weighted_fuzzy, _tfn_euclidean_distance, polarity_applied=True
    )

    norm_vector = _vector_normalize_fuzzy(agg_df)
    weighted_vector = stress_topsis.build_weighted_matrix(norm_vector, w_hat)
    configs["fuzzy_repr + vector_norm + vertex_dist"] = rc_from_weighted(
        weighted_vector, _tfn_vertex_distance
    )

    crisp_df = agg_df.apply(lambda col: col.map(lambda tfn: tfn[1])).astype(float)
    norm_crisp_fuzzy_style = crisp_df.copy()
    for col in crisp_df.columns:
        if col in benefit_cols:
            denom = crisp_df[col].max()
            denom = denom if abs(denom) > EPS else EPS
            norm_crisp_fuzzy_style[col] = crisp_df[col] / denom
        else:
            denom = crisp_df[col].min()
            denom = denom if abs(denom) > EPS else EPS
            norm_crisp_fuzzy_style[col] = denom / crisp_df[col]
    weighted_crisp_fuzzy_style = norm_crisp_fuzzy_style.copy()
    for col in norm_crisp_fuzzy_style.columns:
        weighted_crisp_fuzzy_style[col] = norm_crisp_fuzzy_style[col] * w_hat[col]
    configs["crisp_repr + fuzzy_norm + abs_dist"] = rc_from_weighted(
        weighted_crisp_fuzzy_style, lambda a, b: abs(a - b), polarity_applied=True
    )

    norm_vec_crisp = crisp_df.copy()
    for col in crisp_df.columns:
        denom = np.sqrt((crisp_df[col] ** 2).sum())
        denom = denom if denom > EPS else EPS
        norm_vec_crisp[col] = crisp_df[col] / denom
    weighted_vec_crisp = norm_vec_crisp.copy()
    for col in norm_vec_crisp.columns:
        weighted_vec_crisp[col] = norm_vec_crisp[col] * w_hat[col]
    configs["crisp_repr + vector_norm + euclidean_dist (=orig. comparator)"] = rc_from_weighted(
        weighted_vec_crisp, use_l2=True
    )

    records = []
    for cfg_name, RC in configs.items():
        merged = pd.DataFrame({"RC": RC}).join(dematel_lookup, how="inner")
        srpi_vals = compute_srpi(merged["RC"], merged["Prominence_star"], merged["FSI"])
        new_series = pd.Series(srpi_vals, index=merged.index)
        audit = rank_correlation(baseline_srpi, new_series)
        records.append({
            "test": "crisp_fuzzy_ablation", "configuration": cfg_name,
            "is_identity_run": "(baseline)" in cfg_name, **audit,
        })

    return pd.DataFrame(records)

def fusion_lambda_mu_grid_test(grid=LAMBDA_MU_GRID):
    print("[TEST 11] SRPI fusion lambda_amp x mu_disc sensitivity grid (exploratory only)...")

    topsis_df = pd.read_csv(TOPSIS_RESULT_FILE)
    topsis_df["Feature"] = topsis_df["Method"].replace(NAME_MAP)
    dematel_df = pd.read_csv(DEMATEL_RESULTS_FILE)
    dematel_df["Feature"] = dematel_df["Feature"].astype(str).str.strip()

    merged = topsis_df.set_index("Feature")[["RC"]].join(
        dematel_df.set_index("Feature")[["Prominence_star", "FSI"]], how="inner"
    )
    baseline_srpi = load_baseline_srpi()

    records = []
    for lam in grid:
        for mu in grid:
            srpi_vals = compute_srpi(
                merged["RC"], merged["Prominence_star"], merged["FSI"],
                lambda_amp=lam, mu_disc=mu,
            )
            new_series = pd.Series(srpi_vals, index=merged.index)
            audit = rank_correlation(baseline_srpi, new_series)
            records.append({
                "test": "lambda_mu_grid_sensitivity",
                "lambda_amp": lam, "mu_disc": mu,
                "is_identity_run": bool(abs(lam - LAMBDA_AMP) < 1e-12 and abs(mu - MU_DISC) < 1e-12),
                **audit,
            })

    return pd.DataFrame(records)

def stress_grid_rank_agreement(stressgrid_df):
    from scipy.stats import spearmanr as _sp, kendalltau as _kt
    ref_step, ref_range = 0.25, 0.50
    ref = stressgrid_df[
        (np.isclose(stressgrid_df["step_size"], ref_step))
        & (np.isclose(stressgrid_df["stress_range"], ref_range))
    ].set_index("criterion")["ESI"]
    records = []
    for (step, rng_), grp in stressgrid_df.groupby(["step_size", "stress_range"]):
        esi = grp.set_index("criterion")["ESI"].reindex(ref.index)
        rho, _ = _sp(ref.values, esi.values)
        tau, _ = _kt(ref.values, esi.values)
        ordered = esi.sort_values(ascending=False)
        records.append({
            "step_size": step, "stress_range": rng_,
            "is_reference_grid": bool(np.isclose(step, ref_step) and np.isclose(rng_, ref_range)),
            "n_criteria_ranked": int(len(esi)),
            "ranked_object": "criterion ESI",
            "spearman_rho_vs_reference": rho, "kendall_tau_vs_reference": tau,
            "ESI_min": float(esi.min()), "ESI_max": float(esi.max()),
            "n_crack_events_total": int(grp["n_crack_events_total"].iloc[0]),
            "n_crackfree_transitions_Bstar": int(grp["n_crackfree_transitions_Bstar"].iloc[0]),
            "most_stable_criterion": ordered.index[0],
            "least_stable_criterion": ordered.index[-1],
        })
    out = pd.DataFrame(records)
    non_ref = out[~out["is_reference_grid"]]
    print(
        f"    -> stress-grid ESI rank agreement vs reference grid ({len(non_ref)} non-reference grids): "
        f"rho_min={non_ref['spearman_rho_vs_reference'].min():.3f}, "
        f"tau_min={non_ref['kendall_tau_vs_reference'].min():.3f}, "
        f"ESI range=[{out['ESI_min'].min():.3f}, {out['ESI_max'].max():.3f}], "
        f"max B*={int(out['n_crackfree_transitions_Bstar'].max())}"
    )
    return out

def srpi_vulnerability_scenarios_test():
    print("[DIAG] SRPI vulnerability scaling scenarios (absolute vs min-max vs V=0)...")
    topsis_df = pd.read_csv(TOPSIS_RESULT_FILE)
    topsis_df["Feature"] = topsis_df["Method"].replace(NAME_MAP)
    dematel_df = pd.read_csv(DEMATEL_RESULTS_FILE)
    dematel_df["Feature"] = dematel_df["Feature"].astype(str).str.strip()
    merged = topsis_df.set_index("Feature")[["RC"]].join(
        dematel_df.set_index("Feature")[["Prominence_star", "FSI"]], how="inner"
    )
    srpi = {
        mode: pd.Series(
            compute_srpi(merged["RC"], merged["Prominence_star"], merged["FSI"], vulnerability_mode=mode),
            index=merged.index,
        )
        for mode in ("absolute", "minmax", "zero")
    }
    rc_norm = merged["RC"] / merged["RC"].max()
    records = []
    for mode, ser in srpi.items():
        for ref_name, ref in (("fuzzy_TOPSIS_RC", merged["RC"]), ("SRPI_absolute", srpi["absolute"])):
            audit = rank_correlation(ref, ser)
            records.append({
                "scenario": mode, "reference": ref_name,
                "spearman_rho": audit["spearman_rho"], "kendall_tau": audit["kendall_tau"],
                "discordant_pairs": audit["discordant_pairs"],
                "max_abs_score_diff_vs_reference": float(
                    np.abs(ser - (rc_norm if ref_name == "fuzzy_TOPSIS_RC" else ref)).max()
                ),
            })
    return pd.DataFrame(records)

def stress_amplification_effect_test():
    print("[DIAG] Score-level effect of DEMATEL stress amplification (gamma=0 vs reference gamma)...")
    gamma_df = pd.read_csv(GAMMA_SWEEP_FILE)
    gamma_df["Feature"] = gamma_df["feature"].astype(str).str.strip()
    g_ref = stress_dematel.GAMMA_AMPLIFICATION
    pr0 = gamma_df[np.isclose(gamma_df["gamma"], 0.0)].set_index("Feature")["Prominence_star"]
    pr1 = gamma_df[np.isclose(gamma_df["gamma"], g_ref)].set_index("Feature")["Prominence_star"].reindex(pr0.index)

    topsis_df = pd.read_csv(TOPSIS_RESULT_FILE)
    topsis_df["Feature"] = topsis_df["Method"].replace(NAME_MAP)
    dematel_df = pd.read_csv(DEMATEL_RESULTS_FILE)
    dematel_df["Feature"] = dematel_df["Feature"].astype(str).str.strip()
    base = topsis_df.set_index("Feature")[["RC"]].join(
        dematel_df.set_index("Feature")[["FSI"]], how="inner"
    ).reindex(pr0.index)

    out = pd.DataFrame(index=pr0.index)
    out["PR_gamma0"] = pr0
    out["PR_gamma_ref"] = pr1
    out["delta_PR_rel"] = (pr1 - pr0) / pr0
    out["abs_delta_PR_rel"] = out["delta_PR_rel"].abs()
    for mode in ("absolute", "minmax"):
        s0 = compute_srpi(base["RC"], pr0.values, base["FSI"], vulnerability_mode=mode)
        s1 = compute_srpi(base["RC"], pr1.values, base["FSI"], vulnerability_mode=mode)
        out[f"SRPI_{mode}_gamma0"] = s0
        out[f"SRPI_{mode}_gamma_ref"] = s1
        out[f"abs_SRPI_{mode}_change"] = np.abs(s1 - s0)
        out[f"SRPI_{mode}_rank_gamma0"] = pd.Series(s0, index=out.index).rank(ascending=False, method="min").astype(int)
        out[f"SRPI_{mode}_rank_gamma_ref"] = pd.Series(s1, index=out.index).rank(ascending=False, method="min").astype(int)
    out["PR_rank_gamma0"] = pr0.rank(ascending=False, method="min").astype(int)
    out["PR_rank_gamma_ref"] = pr1.rank(ascending=False, method="min").astype(int)
    out = out.reset_index().rename(columns={"index": "Feature"})

    summary = {
        "gamma_ref": g_ref,
        "abs_delta_PR_rel_min": float(out["abs_delta_PR_rel"].min()),
        "abs_delta_PR_rel_max": float(out["abs_delta_PR_rel"].max()),
        "abs_delta_PR_rel_mean": float(out["abs_delta_PR_rel"].mean()),
        "abs_SRPI_absolute_change_max": float(out["abs_SRPI_absolute_change"].max()),
        "abs_SRPI_absolute_change_mean": float(out["abs_SRPI_absolute_change"].mean()),
        "abs_SRPI_minmax_change_max": float(out["abs_SRPI_minmax_change"].max()),
        "abs_SRPI_minmax_change_mean": float(out["abs_SRPI_minmax_change"].mean()),
        "PR_rank_changed_features": int((out["PR_rank_gamma0"] != out["PR_rank_gamma_ref"]).sum()),
        "SRPI_absolute_rank_changed_features": int((out["SRPI_absolute_rank_gamma0"] != out["SRPI_absolute_rank_gamma_ref"]).sum()),
        "SRPI_minmax_rank_changed_features": int((out["SRPI_minmax_rank_gamma0"] != out["SRPI_minmax_rank_gamma_ref"]).sum()),
        "delta_bar_spread": float(dematel_df["Delta_bar_D"].max() - dematel_df["Delta_bar_D"].min()),
        "delta_bar_mean": float(dematel_df["Delta_bar_D"].mean()),
    }
    print("    -> " + ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    return out, pd.DataFrame([summary])

def verify_primary_consistency(tol=PRIMARY_CONSISTENCY_TOL):
    """Alignment guard: recompute the PRIMARY TOPSIS ranking from the configured weighting and
    polarity (stress_topsis.PRIMARY_WEIGHTING / REVERSE_CODED_CRITERIA) and compare it with the
    files that Validation takes as its baseline. A mismatch means the pipeline is stale."""
    agg_df, norm_df, _b, _c, _d = _load_common_topsis_context()
    w_star = stress_topsis.load_fahp_weights(FAHP_WEIGHTS_FILE)
    w_hat = stress_topsis.primary_weights(w_star)
    weighted_df = stress_topsis.build_weighted_matrix(norm_df, w_hat)
    fpis, fnis = stress_topsis.compute_fpis_fnis(weighted_df)
    Dp, Dm = stress_topsis.compute_separations(weighted_df, fpis, fnis)
    CC = stress_topsis.compute_closeness(Dp, Dm)
    rc = pd.Series(CC)
    rc.index = [NAME_MAP.get(m, m) for m in rc.index]

    topsis_file = load_baseline_topsis_ranking()
    d_topsis = float((rc - topsis_file.reindex(rc.index)).abs().max())

    fused = pd.read_csv(FUSION_RESULT_FILE)
    fused["Feature"] = fused["Feature"].astype(str).str.strip()
    d_final = float((rc - fused.set_index("Feature")["TOPSIS_RC"].reindex(rc.index)).abs().max())

    topsis_raw = pd.read_csv(TOPSIS_RESULT_FILE)
    weighting_in_file = (str(topsis_raw["Weighting"].iloc[0]).strip()
                         if "Weighting" in topsis_raw.columns else "unknown")

    ok = (d_topsis < tol and d_final < tol and weighting_in_file in (PRIMARY_WEIGHTING, "unknown"))
    rec = pd.DataFrame([{
        "PRIMARY_WEIGHTING_config": PRIMARY_WEIGHTING,
        "Weighting_in_STRESS_TOPSIS_RESULT": weighting_in_file,
        "REVERSE_CODED_CRITERIA": "; ".join(sorted(stress_topsis.REVERSE_CODED_CRITERIA)) or "(none)",
        "POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE": bool(stress_topsis.POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE),
        "max_abs_RC_diff_vs_TOPSIS_file": d_topsis,
        "max_abs_RC_diff_vs_FINAL_file": d_final,
        "consistent": bool(ok),
    }])
    if not ok:
        warnings.warn(
            "[VALIDATION] Recomputed PRIMARY TOPSIS RC does not match STRESS_TOPSIS_RESULT.csv / "
            f"Final_Stress_Coupled_Result.csv (diff {d_topsis:.2e} / {d_final:.2e}; file weighting "
            f"'{weighting_in_file}' vs config '{PRIMARY_WEIGHTING}'). Rerun MCDM_Topsis.py and "
            "MCDM_FINAL.py before interpreting the validation results."
        )
    else:
        print(f"[CHECK] Primary model consistent across Topsis/FINAL/Validation "
              f"(weighting={PRIMARY_WEIGHTING}, max RC diff {max(d_topsis, d_final):.1e}).")
    if not stress_topsis.POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE:
        print("[CHECK] NOTE: criterion polarity (Implementation Complexity / Data Dependency) is "
              "still NOT verified against the questionnaire.")
    return rec

def polarity_srpi_sensitivity_diag():
    print("[DIAG] SRPI under hypothetical criterion-polarity reversal; outside the seven experiments...")
    baseline_srpi = load_baseline_srpi()
    agg_df, _norm, _b, _c, dematel_lookup = _load_common_topsis_context()
    w_star = stress_topsis.load_fahp_weights(FAHP_WEIGHTS_FILE)
    w_hat = stress_topsis.primary_weights(w_star)

    scenarios = {"configured_primary": set(stress_topsis.REVERSE_CODED_CRITERIA)}
    scenarios.update(stress_topsis.POLARITY_SCENARIOS)

    base_top = baseline_srpi.sort_values(ascending=False).index[0]
    records = []
    for name, rev in scenarios.items():
        b_cols, c_cols = stress_topsis.polarity_columns(rev)
        norm_df, _, _ = stress_topsis.normalize_fuzzy_matrix(agg_df, b_cols, c_cols)
        weighted_df = stress_topsis.build_weighted_matrix(norm_df, w_hat)
        fpis, fnis = stress_topsis.compute_fpis_fnis(weighted_df)
        Dp, Dm = stress_topsis.compute_separations(weighted_df, fpis, fnis)
        CC = stress_topsis.compute_closeness(Dp, Dm)
        RC = pd.Series(CC)
        RC.index = [NAME_MAP.get(m, m) for m in RC.index]
        merged = pd.DataFrame({"RC": RC}).join(dematel_lookup, how="inner")
        ser = pd.Series(compute_srpi(merged["RC"], merged["Prominence_star"], merged["FSI"]),
                        index=merged.index)
        audit = rank_correlation(baseline_srpi, ser)
        records.append({
            "diagnostic": "polarity_srpi_sensitivity", "scenario": name,
            "reverse_coded": "; ".join(sorted(rev)) or "(none)",
            "PRIMARY_WEIGHTING": PRIMARY_WEIGHTING,
            "top1_baseline_srpi": base_top, "top1_scenario_srpi": ser.sort_values(ascending=False).index[0],
            "top1_changed": bool(ser.sort_values(ascending=False).index[0] != base_top),
            **audit,
        })
    return pd.DataFrame(records)

def build_parameter_grid():
    """Parameter grid of every validation/diagnostic setting (review 2.8, item 9)."""
    rows = [
        ("PRIMARY_WEIGHTING", PRIMARY_WEIGHTING, "all tests (single weighting entry point)"),
        ("REVERSE_CODED_CRITERIA", "; ".join(sorted(stress_topsis.REVERSE_CODED_CRITERIA)) or "(none)", "all TOPSIS-based tests"),
        ("POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE", stress_topsis.POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE, "item 2.5"),
        ("BENEFIT_SHARE/COST_SHARE (imposed)", f"{stress_topsis.BENEFIT_SHARE}/{stress_topsis.COST_SHARE}", "Eq. (17)"),
        ("ALLOCATION_SCENARIOS", "; ".join(f"{b}/{c}" for b, c in stress_topsis.ALLOCATION_SCENARIOS) + "; raw_FAHP", "Test 8"),
        ("WEIGHT_PERTURBATION_PCT", WEIGHT_PERTURBATION_PCT, "Test 1"),
        ("GAMMA_SWEEP_VALUES", stress_dematel.GAMMA_SWEEP_VALUES, "Test 2"),
        ("GAMMA_AMPLIFICATION (reference)", stress_dematel.GAMMA_AMPLIFICATION, "Test 2 / DEMATEL"),
        ("EXPECTED_K (experts)", EXPECTED_K, "Test 3"),
        ("AHP STRESS_SET", [float(b) for b in stress_fahp.STRESS_SET], "AHP / Tests 5-7, 2.3"),
        ("KAPPA_THRESHOLD (reference)", stress_fahp.KAPPA_THRESHOLD, "AHP / Test 6"),
        ("KAPPA_THRESHOLD_GRID", KAPPA_THRESHOLD_GRID, "Test 6"),
        ("CR_THRESHOLD", stress_fahp.CR_THRESHOLD, "AHP ESI_CR / Test 9 / item 2.3"),
        ("STRESS_STEP_GRID", STRESS_STEP_GRID, "Test 7"),
        ("STRESS_RANGE_GRID", STRESS_RANGE_GRID, "Test 7"),
        ("N_PERMUTATIONS / PERMUTATION_SEED", f"{N_PERMUTATIONS} / {PERMUTATION_SEED}", "Test 5"),
        ("MC_N_DRAWS / MC_SIGMA / MC_SEED", f"{MC_N_DRAWS} / {MC_SIGMA} / {MC_SEED}", "Test 9 / item 2.3"),
        ("ESI_MC_STRONG_RHO / primary measure", f"{ESI_MC_STRONG_RHO} / {ESI_MC_PRIMARY_MEASURE}", "item 2.3 decision rule"),
        ("LAMBDA_AMP / MU_DISC", f"{LAMBDA_AMP} / {MU_DISC}", "SRPI reference"),
        ("LAMBDA_MU_GRID", LAMBDA_MU_GRID, "Test 11"),
        ("VULNERABILITY_MODE (reference)", VULNERABILITY_MODE, "SRPI reference"),
        ("DEMATEL STRESS_SET / MODE", f"{list(stress_dematel.STRESS_SET)} / {stress_dematel.STRESS_MODE}", "DEMATEL"),
        ("DEMATEL TAU_C", stress_dematel.TAU_C_DEMATEL, "DEMATEL crack"),
    ]
    return pd.DataFrame([{"parameter": k, "value": str(v), "used_in": u} for k, v, u in rows])

def _aggregate(frame):
    return (
        frame.groupby("test")[["spearman_rho", "kendall_tau", "discordant_pairs"]]
        .agg(["mean", "min", "max"])
    )

def run_all_validation_tests():
    print(f"[INFO] PRIMARY_WEIGHTING = '{PRIMARY_WEIGHTING}'; reverse-coded criteria = "
          f"{sorted(stress_topsis.REVERSE_CODED_CRITERIA) or '(none)'}")
    primary_check_df = verify_primary_consistency()
    primary_check_df.to_csv(OUTPUT_PRIMARY_CHECK_FILE, index=False)

    rank_based_frames = [
        weight_sensitivity_test(),
        gamma_sensitivity_test(),
        leave_one_expert_out_test(),
        comparative_variants_test(),
        benefit_cost_allocation_sensitivity_test(),
        crisp_fuzzy_ablation_test(),
        fusion_lambda_mu_grid_test(),
    ]
    summary = pd.concat(rank_based_frames, ignore_index=True, sort=False)
    summary["is_identity_run"] = summary["is_identity_run"].fillna(False).astype(bool)
    summary.to_csv(OUTPUT_SUMMARY_FILE, index=False)

    non_identity = summary[~summary["is_identity_run"]]
    agg = _aggregate(non_identity)
    agg.to_csv(OUTPUT_SUMMARY_AGG_FILE)
    agg_all = _aggregate(summary)
    agg_all.to_csv(OUTPUT_SUMMARY_ALL_FILE)

    identity = summary[summary["is_identity_run"]].copy()
    identity.to_csv(OUTPUT_IDENTITY_FILE, index=False)

    permutation_df = ahp_permutation_invariance_test()
    permutation_df.to_csv(OUTPUT_PERMUTATION_FILE, index=False)

    montecarlo_df = monte_carlo_ahp_benchmark_test()
    montecarlo_df.to_csv(OUTPUT_MONTECARLO_FILE, index=False)

    crack_diag_df = crack_threshold_sensitivity_test()
    crack_diag_df.to_csv(OUTPUT_CRACK_DIAG_FILE, index=False)

    stressgrid_diag_df = stress_grid_and_range_sensitivity_test()
    stressgrid_diag_df.to_csv(OUTPUT_STRESSGRID_DIAG_FILE, index=False)
    stressgrid_rank_df = stress_grid_rank_agreement(stressgrid_diag_df)
    stressgrid_rank_df.to_csv(OUTPUT_STRESSGRID_RANK_FILE, index=False)

    vuln_df = srpi_vulnerability_scenarios_test()
    vuln_df.to_csv(OUTPUT_VULN_SCENARIO_FILE, index=False)

    amp_feature_df, amp_summary_df = stress_amplification_effect_test()
    amp_feature_df.to_csv(OUTPUT_AMP_FEATURE_FILE, index=False)
    amp_summary_df.to_csv(OUTPUT_AMP_SUMMARY_FILE, index=False)

    esi_mc_df, esi_mc_summary_df = esi_vs_monte_carlo_test()
    esi_mc_df.to_csv(OUTPUT_ESI_MC_FILE, index=False)
    esi_mc_summary_df.to_csv(OUTPUT_ESI_MC_SUMMARY_FILE, index=False)

    polarity_srpi_df = polarity_srpi_sensitivity_diag()
    polarity_srpi_df.to_csv(OUTPUT_POLARITY_SRPI_FILE, index=False)

    build_parameter_grid().to_csv(OUTPUT_PARAM_GRID_FILE, index=False)

    print(f"\n[INFO] Total rank-based runs: {len(summary)} "
          f"({len(identity)} implementation-equivalence runs, {len(non_identity)} comparative/sensitivity runs)")
    print("\n===== VALIDATION SUMMARY, identity runs excluded (mean / min / max) =====\n")
    print(agg.to_string())
    print("\n===== VALIDATION SUMMARY, all runs (mean / min / max) =====\n")
    print(agg_all.to_string())
    print("\n===== IMPLEMENTATION-EQUIVALENCE RUNS =====\n")
    print(identity[["test", "spearman_rho", "kendall_tau"]].to_string(index=False))
    print(f"\n[INFO] Rank-based test results:          {OUTPUT_SUMMARY_FILE}")
    print(f"[INFO] Aggregated summary (no identity):  {OUTPUT_SUMMARY_AGG_FILE}")
    print(f"[INFO] Aggregated summary (all runs):     {OUTPUT_SUMMARY_ALL_FILE}")
    print(f"[INFO] Implementation equivalence:        {OUTPUT_IDENTITY_FILE}")
    print(f"[INFO] Permutation-invariance results:    {OUTPUT_PERMUTATION_FILE}")
    print(f"[INFO] Monte-Carlo AHP benchmark results: {OUTPUT_MONTECARLO_FILE}")
    print(f"[INFO] Crack-threshold diagnostic:        {OUTPUT_CRACK_DIAG_FILE}")
    print(f"[INFO] Stress-grid diagnostic:            {OUTPUT_STRESSGRID_DIAG_FILE}")
    print(f"[INFO] Stress-grid rank agreement:        {OUTPUT_STRESSGRID_RANK_FILE}")
    print(f"[INFO] SRPI vulnerability scenarios:      {OUTPUT_VULN_SCENARIO_FILE}")
    print(f"[INFO] Stress-amplification effect:       {OUTPUT_AMP_FEATURE_FILE}")
    print(f"[INFO] Stress-amplification summary:      {OUTPUT_AMP_SUMMARY_FILE}")
    print(f"[INFO] ESI vs Monte-Carlo (2.3):          {OUTPUT_ESI_MC_FILE}")
    print(f"[INFO] ESI vs Monte-Carlo summary:        {OUTPUT_ESI_MC_SUMMARY_FILE}")
    print(f"[INFO] Polarity -> SRPI sensitivity:      {OUTPUT_POLARITY_SRPI_FILE}")
    print(f"[INFO] Primary-consistency check:         {OUTPUT_PRIMARY_CHECK_FILE}")
    print(f"[INFO] Validation parameter grid:         {OUTPUT_PARAM_GRID_FILE}")

    return summary, agg, permutation_df, montecarlo_df, crack_diag_df, stressgrid_diag_df

if __name__ == "__main__":
    run_all_validation_tests()
