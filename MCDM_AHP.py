import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

PROJECT_DIR = os.environ.get("MCDM_PROJECT_DIR", os.getcwd())
INPUT_FILE = os.path.join(PROJECT_DIR, "dfa.xlsx")

STRESS_SET = np.array([-0.50, -0.25, 0.0, 0.25, 0.50])

KAPPA_THRESHOLD = 0.30
CR_THRESHOLD = 0.10

ESI_AGREEMENT_TAU = 0.90
ESI_AGREEMENT_MAX_SHIFT = 1

EPS = 1e-9
CR_TOL = 1e-12

RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}


def get_random_index(n):
    if n not in RI_TABLE:
        raise ValueError(
            f"[FAHP] No Saaty Random Index (RI_n) tabulated for n={n}. "
            f"Extend RI_TABLE or verify the criteria matrix dimension."
        )
    return RI_TABLE[n]


def out_path(name):
    return os.path.join(PROJECT_DIR, name)


def load_expert_sheets(path):
    with pd.ExcelFile(path) as xls:
        dfs = {
            sheet: pd.read_excel(xls, sheet_name=sheet, index_col=0)
            for sheet in xls.sheet_names
        }
    return dfs


def parse_tfn(cell, context=""):
    if pd.isna(cell) or cell == "":
        warnings.warn(
            f"[FAHP] Missing/blank cell encountered ({context}); "
            f"treated as neutral TFN (1,1,1). Verify source data."
        )
        return (1.0, 1.0, 1.0)

    parts = [float(x.strip()) for x in str(cell).split(",")]

    if len(parts) == 3:
        return tuple(parts)
    elif len(parts) == 1:
        v = parts[0]
        warnings.warn(
            f"[FAHP] Crisp (non-TFN) value encountered ({context}); "
            f"replicated as ({v},{v},{v})."
        )
        return (v, v, v)
    else:
        raise ValueError(
            f"[FAHP] Cell ({context}) has {len(parts)} comma-separated "
            f"values; expected 1 (crisp) or 3 (l,m,u). Value: {cell!r}"
        )


def parse_all_sheets(dfs):
    parsed = {}
    for name, df in dfs.items():
        parsed[name] = df.apply(
            lambda col: [parse_tfn(v, context=f"sheet={name}") for v in col]
        )
    return parsed


def aggregate_experts_geometric(dfs):
    sheets = list(dfs.values())
    E = len(sheets)
    agg = sheets[0].copy()
    n_rows, n_cols = agg.shape

    for i in range(n_rows):
        for j in range(n_cols):
            l = np.prod([df.iloc[i, j][0] for df in sheets]) ** (1.0 / E)
            m = np.prod([df.iloc[i, j][1] for df in sheets]) ** (1.0 / E)
            u = np.prod([df.iloc[i, j][2] for df in sheets]) ** (1.0 / E)
            agg.iloc[i, j] = (l, m, u)

    return agg, E


def enforce_unit_diagonal_fuzzy(df):
    n = df.shape[0]
    out = df.copy()
    for i in range(n):
        out.iloc[i, i] = (1.0, 1.0, 1.0)
    return out


def fuzzy_reciprocal(tfn):
    l, m, u = tfn
    return (1.0 / u, 1.0 / m, 1.0 / l)


def enforce_fuzzy_reciprocity(df):
    n = df.shape[0]
    out = df.copy()
    for i in range(n):
        for j in range(n):
            if i < j:
                out.iloc[j, i] = fuzzy_reciprocal(out.iloc[i, j])
    return out


def defuzzify(tfn):
    l, m, u = tfn
    return (l + m + u) / 3.0


def build_crisp_matrix(fuzzy_df):
    n = fuzzy_df.shape[0]
    labels = list(fuzzy_df.index)
    A0 = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                A0[i, j] = defuzzify(fuzzy_df.iloc[i, j])

    return A0, labels


def enforce_crisp_reciprocity(A0):
    n = A0.shape[0]
    A0_fixed = A0.copy()
    for i in range(n):
        for j in range(n):
            if i < j:
                A0_fixed[j, i] = 1.0 / A0_fixed[i, j]
    return A0_fixed


def principal_eigenpair(A):
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    lambda_max = eigvals[idx].real
    w = eigvecs[:, idx].real
    if np.sum(w) < 0:
        w = -w
    w = w / np.sum(w)
    return lambda_max, w


def consistency_measures(lambda_max, n, ri_n):
    CI = (lambda_max - n) / (n - 1)
    CR = CI / ri_n if ri_n > EPS else np.inf
    return CI, CR


def apply_stress_fahp(A0, beta):
    exponent = 1.0 + beta
    A_beta = np.power(A0, exponent)
    np.fill_diagonal(A_beta, 1.0)
    return A_beta


def stress_scenario_eigen_weights(A0, stress_set, n, ri_n):
    w_traj, lambda_traj, CI_traj, CR_traj = {}, {}, {}, {}

    for beta in stress_set:
        A_beta = apply_stress_fahp(A0, beta)
        lambda_max_beta, w_beta = principal_eigenpair(A_beta)
        CI_beta, CR_beta = consistency_measures(lambda_max_beta, n, ri_n)

        w_traj[beta] = w_beta
        lambda_traj[beta] = lambda_max_beta
        CI_traj[beta] = CI_beta
        CR_traj[beta] = CR_beta

    return w_traj, lambda_traj, CI_traj, CR_traj


def all_adjacent_transitions(w_traj):
    betas_sorted = sorted(w_traj.keys())
    return [(betas_sorted[k], betas_sorted[k + 1]) for k in range(len(betas_sorted) - 1)]


def admissible_betas(CR_traj, cr_threshold=CR_THRESHOLD):
    return [b for b in sorted(CR_traj.keys()) if CR_traj[b] <= cr_threshold + CR_TOL]


def admissible_transitions(CR_traj, cr_threshold=CR_THRESHOLD):
    betas_sorted = sorted(CR_traj.keys())
    ok = [CR_traj[b] <= cr_threshold + CR_TOL for b in betas_sorted]
    return [
        (betas_sorted[k], betas_sorted[k + 1])
        for k in range(len(betas_sorted) - 1)
        if ok[k] and ok[k + 1]
    ]


def step_log_change(w_a, w_b, step):
    wa = np.where(np.asarray(w_a) > EPS, w_a, EPS)
    wb = np.where(np.asarray(w_b) > EPS, w_b, EPS)
    s = step if step > EPS else EPS
    return np.abs(np.log(wb) - np.log(wa)) / s


def crack_detection(w_traj, CR_traj, labels, kappa_threshold=KAPPA_THRESHOLD,
                    cr_threshold=CR_THRESHOLD, transitions=None):
    if transitions is None:
        transitions = all_adjacent_transitions(w_traj)

    n = len(labels)
    crack_records = []
    B_star = []

    for b_k, b_k1 in transitions:
        w_k, w_k1 = w_traj[b_k], w_traj[b_k1]
        cr_k = CR_traj[b_k]
        cr_k1 = CR_traj[b_k1]
        cr_warning = cr_k > cr_threshold
        transition_admissible = (cr_k <= cr_threshold + CR_TOL) and (cr_k1 <= cr_threshold + CR_TOL)
        step = abs(b_k1 - b_k)
        kappa_all = step_log_change(w_k, w_k1, step)

        cracked = False
        for i in range(n):
            kappa_i = float(kappa_all[i])
            criterion_crack = bool(kappa_i > kappa_threshold)

            crack_records.append({
                "beta_k": b_k,
                "beta_k1": b_k1,
                "criterion": labels[i],
                "kappa_A": kappa_i,
                "CR_beta_k": cr_k,
                "CR_beta_k1": cr_k1,
                "criterion_crack": criterion_crack,
                "cr_warning": cr_warning,
                "transition_admissible": transition_admissible,
                "crack": criterion_crack,
            })

            if criterion_crack:
                cracked = True

        if not cracked:
            B_star.append(b_k)

    columns = [
        "beta_k", "beta_k1", "criterion", "kappa_A", "CR_beta_k", "CR_beta_k1",
        "criterion_crack", "cr_warning", "transition_admissible", "crack",
    ]
    crack_table = pd.DataFrame(crack_records, columns=columns)
    return crack_table, sorted(set(B_star))


def compute_stable_weight(w0):
    return w0.copy()


def mean_kappa_over_transitions(w_traj, labels, transitions):
    n = len(labels)
    if len(transitions) == 0:
        return np.full(n, np.nan)

    kappa_sum = np.zeros(n)
    for b_k, b_k1 in transitions:
        step = abs(b_k1 - b_k)
        kappa_sum += step_log_change(w_traj[b_k], w_traj[b_k1], step)

    return kappa_sum / len(transitions)


def compute_mean_kappa_index(w_traj, labels):
    return mean_kappa_over_transitions(w_traj, labels, all_adjacent_transitions(w_traj))


def compute_mean_kappa_index_restricted(w_traj, CR_traj, labels, cr_threshold=CR_THRESHOLD):
    transitions = admissible_transitions(CR_traj, cr_threshold)
    return mean_kappa_over_transitions(w_traj, labels, transitions)


def compute_esi(kappa_bar):
    return np.exp(-np.asarray(kappa_bar, dtype=float))


def rank_stability_desc(values):
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        return np.full(len(arr), np.nan)
    return pd.Series(arr).rank(ascending=False, method="min").astype(int).values


def safe_rank_correlations(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan, np.nan
    if np.ptp(a) < EPS or np.ptp(b) < EPS:
        return np.nan, np.nan
    rho = spearmanr(a, b)[0]
    tau = kendalltau(a, b)[0]
    return float(rho), float(tau)


def compare_esi_full_vs_restricted(labels, kappa_full, esi_full, kappa_cr, esi_cr, n_admissible_transitions):
    rank_full = rank_stability_desc(esi_full)
    rank_cr = rank_stability_desc(esi_cr)

    table = pd.DataFrame({
        "Criterion": labels,
        "kappa_bar_full": np.asarray(kappa_full, dtype=float),
        "ESI_full": np.asarray(esi_full, dtype=float),
        "kappa_bar_CR": np.asarray(kappa_cr, dtype=float),
        "ESI_CR": np.asarray(esi_cr, dtype=float),
        "ESI_abs_diff": np.abs(np.asarray(esi_full, dtype=float) - np.asarray(esi_cr, dtype=float)),
        "Rank_ESI_full": rank_full,
        "Rank_ESI_CR": rank_cr,
        "rank_shift_CR_minus_full": rank_cr - rank_full,
        "n_admissible_transitions": n_admissible_transitions,
    })

    rho, tau = safe_rank_correlations(esi_full, esi_cr)
    return table, rho, tau


def esi_cr_verdict(n_admissible_transitions, tau, max_abs_shift):
    if n_admissible_transitions == 0:
        return "NOT_COMPUTABLE: no admissible transition with CR <= threshold at both endpoints"
    if np.isnan(tau) or np.isnan(max_abs_shift):
        return "NOT_COMPUTABLE: rank correlation undefined (constant ESI vector)"
    if tau >= ESI_AGREEMENT_TAU and max_abs_shift <= ESI_AGREEMENT_MAX_SHIFT:
        return ("NEARLY_IDENTICAL: ESI ranking is not driven by the CR-violating stress matrices")
    return ("MATERIAL_DIFFERENCE: report full-trajectory ESI as an adversarial stress diagnostic, "
            "not as a reliability measure")


def esi_cr_analysis(w_traj, CR_traj, labels, kappa_threshold=KAPPA_THRESHOLD,
                    cr_threshold=CR_THRESHOLD):
    adm_betas = admissible_betas(CR_traj, cr_threshold)
    adm_trans = admissible_transitions(CR_traj, cr_threshold)
    all_trans = all_adjacent_transitions(w_traj)

    kappa_full = compute_mean_kappa_index(w_traj, labels)
    esi_full = compute_esi(kappa_full)
    kappa_cr = mean_kappa_over_transitions(w_traj, labels, adm_trans)
    esi_cr = compute_esi(kappa_cr)

    comparison, rho, tau = compare_esi_full_vs_restricted(
        labels, kappa_full, esi_full, kappa_cr, esi_cr, len(adm_trans)
    )

    crack_full, B_full = crack_detection(
        w_traj, CR_traj, labels, kappa_threshold=kappa_threshold, cr_threshold=cr_threshold
    )
    crack_cr, B_cr = crack_detection(
        w_traj, CR_traj, labels, kappa_threshold=kappa_threshold, cr_threshold=cr_threshold,
        transitions=adm_trans,
    )

    shifts = comparison["rank_shift_CR_minus_full"].values.astype(float)
    if np.isnan(shifts).any():
        max_abs_shift = np.nan
        n_shifted = np.nan
    else:
        max_abs_shift = float(np.max(np.abs(shifts)))
        n_shifted = int((shifts != 0).sum())

    diffs = comparison["ESI_abs_diff"].values
    verdict = esi_cr_verdict(len(adm_trans), tau, max_abs_shift)

    def fmt_trans(tr):
        return "; ".join(f"({a:+.2f},{b:+.2f})" for a, b in tr) if tr else "none"

    summary = pd.DataFrame([{
        "cr_threshold": cr_threshold,
        "kappa_threshold": kappa_threshold,
        "n_stress_states_total": len(CR_traj),
        "n_stress_states_admissible": len(adm_betas),
        "admissible_betas": "; ".join(f"{b:+.2f}" for b in adm_betas) if adm_betas else "none",
        "n_transitions_total": len(all_trans),
        "n_transitions_admissible": len(adm_trans),
        "admissible_transitions": fmt_trans(adm_trans),
        "spearman_rho_ESI_full_vs_CR": rho,
        "kendall_tau_ESI_full_vs_CR": tau,
        "max_abs_rank_shift": max_abs_shift,
        "n_criteria_rank_shifted": n_shifted,
        "mean_abs_ESI_diff": float(np.nanmean(diffs)) if not np.isnan(diffs).all() else np.nan,
        "max_abs_ESI_diff": float(np.nanmax(diffs)) if not np.isnan(diffs).all() else np.nan,
        "crack_events_full": int(crack_full["crack"].sum()),
        "crack_events_CR_admissible": int(crack_cr["crack"].sum()),
        "criteria_flagged_full": int(crack_full.loc[crack_full["crack"], "criterion"].nunique()),
        "criteria_flagged_CR_admissible": int(crack_cr.loc[crack_cr["crack"], "criterion"].nunique()),
        "crackfree_transitions_full": len(B_full),
        "crackfree_transitions_CR_admissible": len(B_cr),
        "agreement_rule_tau_min": ESI_AGREEMENT_TAU,
        "agreement_rule_max_rank_shift": ESI_AGREEMENT_MAX_SHIFT,
        "verdict": verdict,
    }])

    return {
        "comparison": comparison,
        "summary": summary,
        "crack_table_cr": crack_cr,
        "B_star_cr": B_cr,
        "kappa_bar_cr": kappa_cr,
        "esi_cr": esi_cr,
        "admissible_betas": adm_betas,
        "admissible_transitions": adm_trans,
        "rho": rho,
        "tau": tau,
        "verdict": verdict,
    }


def run_pipeline(input_file=INPUT_FILE, stress_set=STRESS_SET,
                 kappa_threshold=KAPPA_THRESHOLD, cr_threshold=CR_THRESHOLD):

    raw_dfs = load_expert_sheets(input_file)
    tfn_dfs = parse_all_sheets(raw_dfs)

    agg_fuzzy, E = aggregate_experts_geometric(tfn_dfs)
    agg_fuzzy = enforce_unit_diagonal_fuzzy(agg_fuzzy)
    agg_fuzzy = enforce_fuzzy_reciprocity(agg_fuzzy)
    print(f"[INFO] Aggregated {E} expert judgment matrices via geometric mean.")

    A0, labels = build_crisp_matrix(agg_fuzzy)
    A0 = enforce_crisp_reciprocity(A0)
    n = len(labels)
    ri_n = get_random_index(n)
    print(f"[INFO] Baseline crisp matrix A0 built for n={n} criteria (RI_n={ri_n}).")

    lambda_max0, w0 = principal_eigenpair(A0)
    CI0, CR0 = consistency_measures(lambda_max0, n, ri_n)
    print(f"[INFO] Baseline (beta=0): lambda_max0={lambda_max0:.4f}, "
          f"CI0={CI0:.4f}, CR0={CR0:.4f}")
    if CR0 > cr_threshold:
        warnings.warn(
            f"[FAHP] Baseline CR0={CR0:.4f} exceeds the conventional AHP "
            f"threshold of {cr_threshold}. Expert judgments may be inconsistent."
        )

    w_traj, lambda_traj, CI_traj, CR_traj = stress_scenario_eigen_weights(
        A0, stress_set, n, ri_n
    )
    print(f"[INFO] Stress trajectory computed for beta in {[float(b) for b in stress_set]}.")

    crack_table, B_star = crack_detection(
        w_traj, CR_traj, labels, kappa_threshold=kappa_threshold, cr_threshold=cr_threshold
    )
    print(f"[INFO] Crack-free stress set B* = {[float(b) for b in B_star]} "
          f"(kappa_threshold={kappa_threshold}, CR threshold={cr_threshold}) "
          f"- diagnostic only, does not affect the weights used downstream.")

    w_star = compute_stable_weight(w0)

    kappa_bar = compute_mean_kappa_index(w_traj, labels)
    esi = compute_esi(kappa_bar)

    cr_analysis = esi_cr_analysis(
        w_traj, CR_traj, labels, kappa_threshold=kappa_threshold, cr_threshold=cr_threshold
    )
    esi_cr = cr_analysis["esi_cr"]
    kappa_bar_cr = cr_analysis["kappa_bar_cr"]

    print(f"[INFO] CR-admissible stress states: {[float(b) for b in cr_analysis['admissible_betas']]}")
    print(f"[INFO] CR-admissible transitions: {[(float(a), float(b)) for a, b in cr_analysis['admissible_transitions']]}")
    print(f"[INFO] ESI full vs ESI_CR: Spearman rho={cr_analysis['rho']}, "
          f"Kendall tau={cr_analysis['tau']}")
    print(f"[INFO] Verdict: {cr_analysis['verdict']}")

    results = pd.DataFrame({
        "Criterion": labels,
        "w0_baseline": w0,
        "w_star": w_star,
        "kappa_bar": kappa_bar,
        "ESI": esi,
        "kappa_bar_CR": kappa_bar_cr,
        "ESI_CR": esi_cr,
    })
    results["Rank_w_star"] = results["w_star"].rank(
        ascending=False, method="min"
    ).astype(int)
    results["Rank_ESI"] = rank_stability_desc(esi)
    results["Rank_ESI_CR"] = rank_stability_desc(esi_cr)
    results = results.sort_values("w_star", ascending=False)

    weight_traj_df = pd.DataFrame(
        {beta: w_traj[beta] for beta in sorted(w_traj)}, index=labels
    ).T
    weight_traj_df.index.name = "beta"

    betas_sorted = sorted(CR_traj.keys())
    adm_set = set(admissible_betas(CR_traj, cr_threshold))
    consistency_traj_df = pd.DataFrame({
        "beta": betas_sorted,
        "lambda_max": [lambda_traj[b] for b in betas_sorted],
        "CI": [CI_traj[b] for b in betas_sorted],
        "CR": [CR_traj[b] for b in betas_sorted],
        "CR_admissible": [b in adm_set for b in betas_sorted],
    })

    results.to_csv(out_path("STRESS_FAHP_RESULTS.csv"), index=False)
    pd.DataFrame(A0, index=labels, columns=labels).to_csv(out_path("FAHP_A0_matrix.csv"))
    weight_traj_df.to_csv(out_path("weight_trajectory.csv"))
    consistency_traj_df.to_csv(out_path("consistency_trajectory.csv"), index=False)
    crack_table.to_csv(out_path("crack_table_fahp.csv"), index=False)
    cr_analysis["crack_table_cr"].to_csv(out_path("crack_table_fahp_CR_admissible.csv"), index=False)
    cr_analysis["comparison"].to_csv(out_path("ESI_CR_restricted.csv"), index=False)
    cr_analysis["summary"].to_csv(out_path("ESI_CR_summary.csv"), index=False)

    print("\n===== STRESS-EIGEN FAHP RESULTS =====\n")
    print(results[["Criterion", "w_star", "ESI", "ESI_CR", "Rank_w_star"]].to_string(index=False))

    return {
        "results": results,
        "A0": A0,
        "w0": w0,
        "w_star": w_star,
        "labels": labels,
        "lambda_max0": lambda_max0,
        "CI0": CI0,
        "CR0": CR0,
        "weight_trajectory": weight_traj_df,
        "consistency_trajectory": consistency_traj_df,
        "crack_table": crack_table,
        "B_star": B_star,
        "kappa_bar": kappa_bar,
        "esi": esi,
        "kappa_bar_cr": kappa_bar_cr,
        "esi_cr": esi_cr,
        "esi_cr_analysis": cr_analysis,
    }


if __name__ == "__main__":
    run_pipeline()
