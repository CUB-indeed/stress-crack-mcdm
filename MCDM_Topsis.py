
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

PROJECT_DIR = os.environ.get("MCDM_PROJECT_DIR", "/home/coder/project")


def _p(name):
    return os.path.join(PROJECT_DIR, name)


INPUT_FILE = _p("dft.xlsx")                       # K = 8 sheets (one per expert)
FAHP_WEIGHTS_FILE = _p("STRESS_FAHP_RESULTS.csv")  # produced by MCDM_AHP.py

OUTPUT_RESULT_FILE = _p("STRESS_TOPSIS_RESULT.csv")  # PRIMARY model; read by MCDM_FINAL / Validation
OUTPUT_SENSITIVITY_FILE = _p("STRESS_TOPSIS_allocation_sensitivity.csv")

# --- new outputs  ---
OUTPUT_RESULT_7525_FILE = _p("STRESS_TOPSIS_RESULT_75_25.csv")
OUTPUT_RESULT_RAW_FILE = _p("STRESS_TOPSIS_RESULT_raw_fahp.csv")
OUTPUT_ENDOGENOUS_FILE = _p("STRESS_TOPSIS_endogenous_shares.csv")
OUTPUT_WEIGHTS_COMPARE_FILE = _p("STRESS_TOPSIS_weights_raw_vs_7525.csv")
OUTPUT_RAW_VS_7525_FILE = _p("STRESS_TOPSIS_raw_vs_7525.csv")
OUTPUT_RAW_VS_7525_SUMMARY_FILE = _p("STRESS_TOPSIS_raw_vs_7525_summary.csv")

# --- new outputs  ---
OUTPUT_POLARITY_AUDIT_FILE = _p("STRESS_TOPSIS_criterion_polarity_audit.csv")
OUTPUT_RATING_AUDIT_FEATURE_FILE = _p("STRESS_TOPSIS_polarity_raw_ratings_by_feature.csv")
OUTPUT_RATING_AUDIT_EXPERT_FILE = _p("STRESS_TOPSIS_polarity_raw_ratings_by_expert.csv")
OUTPUT_POLARITY_EMPIRICAL_FILE = _p("STRESS_TOPSIS_polarity_empirical_check.csv")
OUTPUT_POLARITY_SENS_FILE = _p("STRESS_TOPSIS_polarity_sensitivity.csv")
OUTPUT_POLARITY_SENS_SUMMARY_FILE = _p("STRESS_TOPSIS_polarity_sensitivity_summary.csv")

EPS = 1e-9

BENEFIT_CRITERIA = {          # outcome group O (|O| = 8)
    "Effectiveness": "Effectiveness",
    "Regulatory Compliance": "Regulatory Compliance",
    "Data Privacy": "Data Privacy",
    "Integration": "Integration",
    "Scalability": "Scalability",
    "Stakeholder Acceptance": "Stakeholder",
    "Innovation Potential": "Innovation",
    "Time-to-Insight": "Time",
}

BENEFIT_CRITERIA_ALLOCATION_GROUP = BENEFIT_CRITERIA  # kept as an alias for clarity at call sites
COST_CRITERIA = {             # resource group S (|S| = 3)
    "Cost Efficiency": "Cost",
    "Implementation Complexity": "Implementation",
    "Data Dependency": "Data Dependency",
}
COST_CRITERIA_ALLOCATION_GROUP = COST_CRITERIA  # kept as an alias for clarity at call sites

# full criterion name -> column name in dft.xlsx (all 11, group O first, then S)
ALL_CRITERIA = {**BENEFIT_CRITERIA, **COST_CRITERIA}

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

# ------------------------------------------------------------------
# POLARITY SWITCH
# ------------------------------------------------------------------

REVERSE_CODED_CRITERIA = set()
POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE = True

SOURCE_CODING_CONFIRMED_BY_AUTHOR = False
EMPIRICAL_CHECK_BOOT_B = 2000
EMPIRICAL_CHECK_SEED = 20261005

QUESTIONNAIRE_ITEM_STEM = ("B2: Please rate how well each AI Feature (Rows) contributes to achieving "
                           "each Assessment Criterion (Column).")
QUESTIONNAIRE_EVIDENCE = {   # verbatim criterion definitions shown to the experts
    "Cost Efficiency": "Lower cost and effort to implement or run the solution is preferred.",
    "Implementation Complexity": "Simpler systems that are easier to deploy and manage are better.",
    "Data Dependency": "Less reliance on rare or hard-to-access ESG data makes a solution more practical.",
}

# "what if reversed" scenarios (absolute sets; compared with the configured primary)
POLARITY_SCENARIOS = {
    "IMP_reversed": {"Implementation Complexity"},
    "DEP_reversed": {"Data Dependency"},
    "IMP_DEP_reversed": {"Implementation Complexity", "Data Dependency"},
}
POLARITY_AUDIT_CRITERIA = ["Cost Efficiency", "Implementation Complexity", "Data Dependency"]

# What the criterion LABEL suggests to a reader (documentation for the audit table).
LABEL_DIRECTION = {
    "Cost Efficiency": "favorable (higher = more cost-efficient)",
    "Implementation Complexity": "unfavorable-sounding label (rated construct: implementation simplicity)",
    "Data Dependency": "unfavorable-sounding label (rated construct: low data dependence)",
}


def polarity_columns(reverse_coded=None):
    """Return (benefit_cols, cost_cols) in dft.xlsx column names.
    Criteria in `reverse_coded` (full names) are normalized as cost-type."""
    rc = set(REVERSE_CODED_CRITERIA if reverse_coded is None else reverse_coded)
    unknown = rc - set(ALL_CRITERIA)
    if unknown:
        raise ValueError(f"[TOPSIS] REVERSE_CODED_CRITERIA has unknown names: {sorted(unknown)}")
    cost_cols = [col for full, col in ALL_CRITERIA.items() if full in rc]
    benefit_cols = [col for full, col in ALL_CRITERIA.items() if full not in rc]
    return benefit_cols, cost_cols


# Used ONLY by normalize_fuzzy_matrix() (TOPSIS polarity). Validation imports these.
POLARITY_BENEFIT_COLS, POLARITY_COST_COLS = polarity_columns(REVERSE_CODED_CRITERIA)

# ------------------------------------------------------------------
# WEIGHTING SWITCH
# ------------------------------------------------------------------

PRIMARY_WEIGHTING = "75_25"

# 75/25 is the PRE-SPECIFIED model adopted from prior literature.
BENEFIT_SHARE = 0.75
COST_SHARE = 0.25

ALLOCATION_SCENARIOS = [
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.75, 0.25),  # duplicates the 75/25 model on purpose, as an internal check
    (0.80, 0.20),
]
RAW_SCENARIO_LABEL = "raw_FAHP"

RAW_VS_7525_RHO_MIN = 0.95
RAW_VS_7525_TAU_MIN = 0.85
RAW_VS_7525_MAX_RANK_SHIFT = 1
RAW_VS_7525_TOP_K = 3
RANK_SHIFT_FLAG = 2   # |rank shift| >= this is flagged

assert len(BENEFIT_CRITERIA) == 8, "[TOPSIS] |O| must be 8 per the manuscript."
assert len(COST_CRITERIA) == 3, "[TOPSIS] |S| must be 3 per the manuscript."
assert PRIMARY_WEIGHTING in ("75_25", "raw"), "[TOPSIS] PRIMARY_WEIGHTING must be '75_25' or 'raw'."


# ------------------------------------------------------------------
# STEP 1: Load + parse expert TFN ratings
# ------------------------------------------------------------------

def load_expert_sheets(path):
    xls = pd.ExcelFile(path)
    dfs = {
        sheet: pd.read_excel(xls, sheet_name=sheet, index_col=0)
        for sheet in xls.sheet_names
    }
    return dfs


def parse_tfn(cell, context=""):
    if pd.isna(cell) or cell == "":
        warnings.warn(
            f"[TOPSIS] Missing/blank cell encountered ({context}); "
            f"treated as neutral TFN (1,1,1). Verify source data."
        )
        return (1.0, 1.0, 1.0)

    parts = [float(x.strip()) for x in str(cell).split(",")]

    if len(parts) == 3:
        return tuple(parts)
    elif len(parts) == 1:
        v = parts[0]
        warnings.warn(
            f"[TOPSIS] Crisp (non-TFN) value encountered ({context}); "
            f"replicated as ({v},{v},{v})."
        )
        return (v, v, v)
    else:
        raise ValueError(
            f"[TOPSIS] Cell ({context}) has {len(parts)} comma-separated "
            f"values; expected 1 (crisp) or 3 (l,m,u). Value: {cell!r}"
        )


def parse_all_sheets(dfs):
    parsed = {}
    for name, df in dfs.items():
        parsed[name] = df.apply(
            lambda col: [parse_tfn(v, context=f"sheet={name}") for v in col]
        )
    return parsed


def validate_expert_sheets(dfs, expected_k=8):
    k = len(dfs)
    if k != expected_k:
        warnings.warn(
            f"[TOPSIS] Expected K={expected_k} expert sheets per the "
            f"manuscript, found K={k}. Proceeding with K={k}."
        )

    ref_name = list(dfs.keys())[0]
    ref_index = list(dfs[ref_name].index)
    ref_columns = list(dfs[ref_name].columns)

    for name, df in dfs.items():
        if list(df.index) != ref_index:
            raise ValueError(
                f"[TOPSIS] Alternative set in sheet '{name}' does not match "
                f"sheet '{ref_name}'. All expert sheets must rate the same "
                f"alternatives in the same order."
            )
        if list(df.columns) != ref_columns:
            raise ValueError(
                f"[TOPSIS] Criteria set in sheet '{name}' does not match "
                f"sheet '{ref_name}'."
            )

    return ref_index, ref_columns


# ------------------------------------------------------------------
# STEP 2: Arithmetic-mean aggregation across experts
# ------------------------------------------------------------------

def aggregate_experts_arithmetic(dfs):
    sheets = list(dfs.values())
    K = len(sheets)
    agg = sheets[0].copy()
    n_rows, n_cols = agg.shape

    for i in range(n_rows):
        for j in range(n_cols):
            l = np.mean([df.iloc[i, j][0] for df in sheets])
            m = np.mean([df.iloc[i, j][1] for df in sheets])
            u = np.mean([df.iloc[i, j][2] for df in sheets])
            agg.iloc[i, j] = (l, m, u)

    return agg, K


# ------------------------------------------------------------------
# STEP 3: Weights  (75/25 rescaling | raw FAHP) -- ONE entry point
# ------------------------------------------------------------------

def load_fahp_weights(path):
    df = pd.read_csv(path)
    if "Criterion" not in df.columns or "w_star" not in df.columns:
        raise ValueError(
            "[TOPSIS] STRESS_FAHP_RESULTS.csv must contain 'Criterion' and "
            "'w_star' columns."
        )
    return dict(zip(df["Criterion"], df["w_star"]))


def _check_missing(w_star, benefit_criteria, cost_criteria):
    missing = [
        c for c in list(benefit_criteria.keys()) + list(cost_criteria.keys())
        if c not in w_star
    ]
    if missing:
        raise ValueError(
            f"[TOPSIS] Criteria missing from STRESS_FAHP_RESULTS.csv: {missing}"
        )


def endogenous_group_shares(w_star, benefit_criteria=BENEFIT_CRITERIA,
                            cost_criteria=COST_CRITERIA):
    """q_O = sum_{i in O} w_i*, q_S = sum_{i in S} w_i*  (normalized over the 11 criteria)."""
    _check_missing(w_star, benefit_criteria, cost_criteria)
    sum_O = sum(w_star[c] for c in benefit_criteria)
    sum_S = sum(w_star[c] for c in cost_criteria)
    tot = sum_O + sum_S
    return sum_O / tot, sum_S / tot


def raw_weights(w_star, benefit_criteria=BENEFIT_CRITERIA, cost_criteria=COST_CRITERIA):
    """Raw FAHP weights w* WITHOUT Eq. (17); keyed by dft column name, sum = 1."""
    _check_missing(w_star, benefit_criteria, cost_criteria)
    names = {**benefit_criteria, **cost_criteria}
    total = sum(w_star[c] for c in names)
    if total <= EPS:
        raise ValueError("[TOPSIS] Degenerate raw weight sum.")
    return {col: w_star[full] / total for full, col in names.items()}


def rescale_weights(w_star, benefit_criteria, cost_criteria,
                    benefit_share=BENEFIT_SHARE, cost_share=COST_SHARE):
    """Eq. (17): group-level rescaling (benefit_share to O, cost_share to S)."""
    _check_missing(w_star, benefit_criteria, cost_criteria)

    sum_B = sum(w_star[c] for c in benefit_criteria)
    sum_C = sum(w_star[c] for c in cost_criteria)

    if sum_B <= EPS or sum_C <= EPS:
        raise ValueError(
            "[TOPSIS] Degenerate benefit/cost weight sum encountered "
            "while rescaling w* (Eq. rescale)."
        )

    w_hat = {}
    for full_name, col_name in benefit_criteria.items():
        w_hat[col_name] = benefit_share * (w_star[full_name] / sum_B)
    for full_name, col_name in cost_criteria.items():
        w_hat[col_name] = cost_share * (w_star[full_name] / sum_C)

    total = sum(w_hat.values())
    if abs(total - 1.0) > 1e-6:
        warnings.warn(
            f"[TOPSIS] Rescaled weights sum to {total:.6f}, not 1.0; "
            f"renormalizing for numerical safety."
        )
        w_hat = {k: v / total for k, v in w_hat.items()}

    return w_hat


def compute_weights(w_star, weighting="75_25",
                    benefit_share=BENEFIT_SHARE, cost_share=COST_SHARE):
    """Single entry point for criterion weights used by TOPSIS (and by Validation)."""
    if weighting == "75_25":
        return rescale_weights(w_star, BENEFIT_CRITERIA, COST_CRITERIA,
                               benefit_share=benefit_share, cost_share=cost_share)
    if weighting == "raw":
        return raw_weights(w_star)
    raise ValueError(f"[TOPSIS] Unknown weighting '{weighting}' (use '75_25' or 'raw').")


def primary_weights(w_star):
    """Weights of the PRIMARY model, controlled by PRIMARY_WEIGHTING.
    Validation/SRPI must use this function instead of calling rescale_weights directly."""
    return compute_weights(w_star, PRIMARY_WEIGHTING)


# ------------------------------------------------------------------
# STEP 4: Normalization with benefit/cost polarity
# ------------------------------------------------------------------

def normalize_fuzzy_matrix(agg_df, benefit_cols, cost_cols):
    alternatives = list(agg_df.index)
    criteria = list(agg_df.columns)
    norm_df = agg_df.copy()

    # u_i^+ = max_p u_pi, for benefit-type (higher is better) criteria
    u_plus = {}
    for i in benefit_cols:
        u_plus[i] = max(agg_df.loc[p, i][2] for p in alternatives)

    # l_i^- = min_p l_pi, for cost-type (reverse-coded) criteria
    l_minus = {}
    for i in cost_cols:
        l_minus[i] = min(agg_df.loc[p, i][0] for p in alternatives)

    for p in alternatives:
        for i in criteria:
            l, m, u = agg_df.loc[p, i]
            if i in benefit_cols:
                denom = u_plus[i] if u_plus[i] > EPS else EPS
                norm_df.loc[p, i] = (l / denom, m / denom, u / denom)
            elif i in cost_cols:
                num = l_minus[i]
                l_safe = l if abs(l) > EPS else EPS
                m_safe = m if abs(m) > EPS else EPS
                u_safe = u if abs(u) > EPS else EPS
                norm_df.loc[p, i] = (num / u_safe, num / m_safe, num / l_safe)
            else:
                raise ValueError(
                    f"[TOPSIS] Criterion '{i}' is neither in the benefit "
                    f"nor the cost set."
                )

    return norm_df, u_plus, l_minus


# ------------------------------------------------------------------
# STEP 5-8: weighted matrix, FPIS/FNIS, separations, closeness
# ------------------------------------------------------------------

def build_weighted_matrix(norm_df, w_hat):
    weighted_df = norm_df.copy()
    for p in norm_df.index:
        for i in norm_df.columns:
            l, m, u = norm_df.loc[p, i]
            w_i = w_hat[i]
            weighted_df.loc[p, i] = (w_i * l, w_i * m, w_i * u)
    return weighted_df


def compute_fpis_fnis(weighted_df):
    criteria = list(weighted_df.columns)
    alternatives = list(weighted_df.index)

    fpis, fnis = {}, {}
    for i in criteria:
        ls = [weighted_df.loc[p, i][0] for p in alternatives]
        ms = [weighted_df.loc[p, i][1] for p in alternatives]
        us = [weighted_df.loc[p, i][2] for p in alternatives]
        fpis[i] = (max(ls), max(ms), max(us))
        fnis[i] = (min(ls), min(ms), min(us))

    return fpis, fnis


def vertex_distance(tfn_a, tfn_b):
    l_a, m_a, u_a = tfn_a
    l_b, m_b, u_b = tfn_b
    return np.sqrt(((l_a - l_b) ** 2 + (m_a - m_b) ** 2 + (u_a - u_b) ** 2) / 3.0)


def compute_separations(weighted_df, fpis, fnis):
    alternatives = list(weighted_df.index)
    criteria = list(weighted_df.columns)

    D_plus, D_minus = {}, {}
    for p in alternatives:
        d_plus_sum = 0.0
        d_minus_sum = 0.0
        for i in criteria:
            v_pi = weighted_df.loc[p, i]
            d_plus_sum += vertex_distance(v_pi, fpis[i])
            d_minus_sum += vertex_distance(v_pi, fnis[i])
        D_plus[p] = d_plus_sum
        D_minus[p] = d_minus_sum

    return D_plus, D_minus


def compute_closeness(D_plus, D_minus):
    CC = {}
    for p in D_plus:
        denom = D_plus[p] + D_minus[p]
        CC[p] = D_minus[p] / denom if denom > EPS else 0.0
    return CC


# ------------------------------------------------------------------
# HELPER: weighting -> ranking for ONE weighting configuration
# ------------------------------------------------------------------

def run_allocation(w_star, norm_df, alternatives, criteria,
                   benefit_share=BENEFIT_SHARE, cost_share=COST_SHARE,
                   w_hat=None):
    """Backward compatible. If `w_hat` is given (e.g. raw weights) it is used as is,
    otherwise Eq. (17) with (benefit_share, cost_share) is applied."""
    if w_hat is None:
        w_hat = rescale_weights(
            w_star, BENEFIT_CRITERIA, COST_CRITERIA,
            benefit_share=benefit_share, cost_share=cost_share,
        )
    weighted_df = build_weighted_matrix(norm_df, w_hat)
    fpis, fnis = compute_fpis_fnis(weighted_df)
    D_plus, D_minus = compute_separations(weighted_df, fpis, fnis)
    CC = compute_closeness(D_plus, D_minus)

    results = pd.DataFrame({
        "Method": alternatives,
        "D_plus": [D_plus[p] for p in alternatives],
        "D_minus": [D_minus[p] for p in alternatives],
        "RC": [CC[p] for p in alternatives],
    })
    results["Rank"] = results["RC"].rank(ascending=False, method="min").astype(int)
    results = results.sort_values("RC", ascending=False).reset_index(drop=True)

    return {
        "results": results,
        "w_hat": w_hat,
        "weighted_matrix": weighted_df,
        "fpis": fpis,
        "fnis": fnis,
        "D_plus": D_plus,
        "D_minus": D_minus,
        "CC": CC,
    }


def run_weighting(w_star, norm_df, alternatives, criteria, weighting):
    """Run TOPSIS for weighting in {'75_25','raw'} (via compute_weights)."""
    return run_allocation(w_star, norm_df, alternatives, criteria,
                          w_hat=compute_weights(w_star, weighting))


def _rank_vector(results, alternatives):
    r = dict(zip(results["Method"], results["Rank"]))
    return [r[m] for m in alternatives]


def _rank_corr(a, b):
    rho, _ = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    return float(rho), float(tau)


# ------------------------------------------------------------------
# ENDOGENOUS SHARES + RAW vs 75/25
# ------------------------------------------------------------------

def build_endogenous_share_tables(w_star):
    q_O, q_S = endogenous_group_shares(w_star)
    shares = pd.DataFrame([
        {"Group": "O (outcome, |O|=8)", "q_endogenous_from_w_star": q_O,
         "q_imposed_75_25": BENEFIT_SHARE, "difference_endogenous_minus_imposed": q_O - BENEFIT_SHARE},
        {"Group": "S (resource, |S|=3)", "q_endogenous_from_w_star": q_S,
         "q_imposed_75_25": COST_SHARE, "difference_endogenous_minus_imposed": q_S - COST_SHARE},
    ])

    w_raw = raw_weights(w_star)
    w_7525 = compute_weights(w_star, "75_25")
    rows = []
    for full, col in ALL_CRITERIA.items():
        rows.append({
            "Criterion": full, "Column": col,
            "Group": "O" if full in BENEFIT_CRITERIA else "S",
            "w_star": w_star[full],
            "w_raw": w_raw[col],
            "w_75_25": w_7525[col],
            "ratio_75_25_over_raw": w_7525[col] / w_raw[col] if w_raw[col] > EPS else np.nan,
        })
    return shares, pd.DataFrame(rows), q_O, q_S


def decide_raw_vs_7525(rho, tau, max_abs_shift, top1_same, topk_same):
    almost = (
        rho >= RAW_VS_7525_RHO_MIN and tau >= RAW_VS_7525_TAU_MIN
        and max_abs_shift <= RAW_VS_7525_MAX_RANK_SHIFT and top1_same and topk_same
    )
    return "ALMOST_IDENTICAL" if almost else "MATERIAL_DIFFERENCE"


def compare_raw_vs_7525(run_raw, run_7525, alternatives, q_O, q_S):
    r_raw = run_raw["results"].set_index("Method")
    r_75 = run_7525["results"].set_index("Method")

    rows = []
    for m in alternatives:
        rk_raw, rk_75 = int(r_raw.loc[m, "Rank"]), int(r_75.loc[m, "Rank"])
        rows.append({
            "Feature": NAME_MAP.get(m, m), "Method": m,
            "RC_raw_FAHP": r_raw.loc[m, "RC"], "Rank_raw_FAHP": rk_raw,
            "RC_75_25": r_75.loc[m, "RC"], "Rank_75_25": rk_75,
            "RC_diff_raw_minus_75_25": r_raw.loc[m, "RC"] - r_75.loc[m, "RC"],
            "RC_abs_diff": abs(r_raw.loc[m, "RC"] - r_75.loc[m, "RC"]),
            # positive = feature ranks HIGHER under raw FAHP than under 75/25
            "Rank_shift_75_25_minus_raw": rk_75 - rk_raw,
            "Rank_shift_abs": abs(rk_75 - rk_raw),
            "Flag_shift_ge_threshold": abs(rk_75 - rk_raw) >= RANK_SHIFT_FLAG,
        })
    detail = pd.DataFrame(rows).sort_values(["Rank_75_25", "Rank_raw_FAHP"]).reset_index(drop=True)

    rho, tau = _rank_corr(detail["Rank_75_25"].values, detail["Rank_raw_FAHP"].values)
    top1_75 = detail.sort_values("Rank_75_25").iloc[0]["Feature"]
    top1_raw = detail.sort_values("Rank_raw_FAHP").iloc[0]["Feature"]
    k = RAW_VS_7525_TOP_K
    topk_75 = set(detail.sort_values("Rank_75_25").head(k)["Feature"])
    topk_raw = set(detail.sort_values("Rank_raw_FAHP").head(k)["Feature"])
    max_shift = int(detail["Rank_shift_abs"].max())
    top1_same = bool(top1_75 == top1_raw)
    topk_same = bool(topk_75 == topk_raw)
    verdict = decide_raw_vs_7525(rho, tau, max_shift, top1_same, topk_same)

    if verdict == "ALMOST_IDENTICAL":
        action = ("RETAIN 75/25 as primary and report that it differs from the "
                  f"endogenous allocation but preserves the ranking up to one adjacent exchange (q_O={q_O:.3f}, q_S={q_S:.3f}).")
    else:
        action = ("Raw FAHP and 75/25 differ materially: make raw FAHP the primary TOPSIS model "
                  "(set PRIMARY_WEIGHTING='raw', rerun TOPSIS/FINAL/Validation/Visualization) and "
                  "treat 75/25 as a policy scenario; keep 75/25 as primary only with an independent "
                  "expert/policy justification.")

    summary_rows = [
        ("q_O_endogenous", q_O), ("q_S_endogenous", q_S),
        ("q_O_imposed", BENEFIT_SHARE), ("q_S_imposed", COST_SHARE),
        ("spearman_rho_raw_vs_75_25", rho), ("kendall_tau_raw_vs_75_25", tau),
        ("mean_abs_RC_diff", float(detail["RC_abs_diff"].mean())),
        ("max_abs_RC_diff", float(detail["RC_abs_diff"].max())),
        ("n_features_rank_changed", int((detail["Rank_shift_abs"] > 0).sum())),
        ("max_abs_rank_shift", max_shift),
        ("n_features_flagged_shift_ge_%d" % RANK_SHIFT_FLAG, int(detail["Flag_shift_ge_threshold"].sum())),
        ("top1_75_25", top1_75), ("top1_raw_FAHP", top1_raw), ("top1_same", top1_same),
        ("top%d_set_same" % k, topk_same),
        ("threshold_rho_min", RAW_VS_7525_RHO_MIN), ("threshold_tau_min", RAW_VS_7525_TAU_MIN),
        ("threshold_max_rank_shift", RAW_VS_7525_MAX_RANK_SHIFT),
        ("PRIMARY_WEIGHTING_current", PRIMARY_WEIGHTING),
        ("decision_rule_verdict", verdict), ("recommended_action", action),
    ]
    summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    return detail, summary, verdict, action


# ------------------------------------------------------------------
# SENSITIVITY ANALYSIS: group allocation (splits + raw FAHP)
# ------------------------------------------------------------------

def _is_primary_scenario(label_is_raw, benefit_share, cost_share):
    if PRIMARY_WEIGHTING == "raw":
        return label_is_raw
    return (not label_is_raw) and abs(benefit_share - BENEFIT_SHARE) < 1e-9 \
        and abs(cost_share - COST_SHARE) < 1e-9


def run_allocation_sensitivity(w_star, norm_df, alternatives, criteria,
                               baseline_results,
                               scenarios=ALLOCATION_SCENARIOS,
                               output_file=OUTPUT_SENSITIVITY_FILE,
                               primary_results=None, include_raw=True):
    """baseline_results = the 75/25 run (kept for backward compatibility of the
    '*_vs_75_25' columns). primary_results = run selected by PRIMARY_WEIGHTING
    (defaults to baseline_results)."""
    if primary_results is None:
        primary_results = baseline_results

    base_rank = dict(zip(baseline_results["Method"], baseline_results["Rank"]))
    base_vec = [base_rank[m] for m in alternatives]
    prim_vec = _rank_vector(primary_results, alternatives)

    q_O, q_S = endogenous_group_shares(w_star)
    scen_list = [(bs, cs, f"{int(round(bs*100))}/{int(round(cs*100))}", False) for bs, cs in scenarios]
    if include_raw:
        scen_list.append((q_O, q_S, RAW_SCENARIO_LABEL, True))

    rows, weight_rows = [], []
    for benefit_share, cost_share, label, is_raw in scen_list:
        if is_raw:
            run = run_allocation(w_star, norm_df, alternatives, criteria,
                                 w_hat=compute_weights(w_star, "raw"))
        else:
            run = run_allocation(w_star, norm_df, alternatives, criteria,
                                 benefit_share=benefit_share, cost_share=cost_share)
        scen = run["results"]
        scen_vec = _rank_vector(scen, alternatives)
        rho, tau = _rank_corr(base_vec, scen_vec)
        rho_p, tau_p = _rank_corr(prim_vec, scen_vec)
        is_primary = _is_primary_scenario(is_raw, benefit_share, cost_share)

        for _, row in scen.iterrows():
            rows.append({
                "Benefit_Share": benefit_share,
                "Cost_Share": cost_share,
                "Method": row["Method"],
                "RC": row["RC"],
                "Rank": row["Rank"],
                "Spearman_rho_vs_75_25": rho,
                "Kendall_tau_vs_75_25": tau,
                # ---- columns added in this revision ----
                "Scenario": label,
                "Is_raw_FAHP": is_raw,
                "Is_primary": is_primary,
                "Feature": NAME_MAP.get(row["Method"], row["Method"]),
                "Spearman_rho_vs_primary": rho_p,
                "Kendall_tau_vs_primary": tau_p,
            })
        for c in criteria:
            weight_rows.append({
                "Benefit_Share": benefit_share, "Cost_Share": cost_share,
                "Criterion": c, "w_hat": run["w_hat"][c],
                "Scenario": label, "Is_raw_FAHP": is_raw,
            })

        print(f"[INFO] Allocation {label:>9s}: rho={rho:.4f}, tau={tau:.4f} vs 75/25 | "
              f"rho={rho_p:.4f}, tau={tau_p:.4f} vs primary ({PRIMARY_WEIGHTING}).")

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(output_file, index=False)
    weights_df = pd.DataFrame(weight_rows)
    weights_df.to_csv(output_file.replace(".csv", "_weights.csv"), index=False)
    return sensitivity_df, weights_df


# ------------------------------------------------------------------
# POLARITY: explicit table, raw-rating audit, "if reversed"
# ------------------------------------------------------------------

def ratings_tensor(tfn_dfs, n_alt, n_crit):
    """K x P x C x 3 array of expert TFN ratings (l, m, u)."""
    sheets = list(tfn_dfs.keys())
    R = np.zeros((len(sheets), n_alt, n_crit, 3))
    for k, s in enumerate(sheets):
        df = tfn_dfs[s]
        for p in range(n_alt):
            for i in range(n_crit):
                R[k, p, i, :] = df.iloc[p, i]
    return sheets, R


def build_polarity_audit(tfn_dfs, alternatives, criteria):
    sheets, R = ratings_tensor(tfn_dfs, len(alternatives), len(criteria))
    col_idx = {c: j for j, c in enumerate(criteria)}
    reverse = set(REVERSE_CODED_CRITERIA)

    table = []
    for full, col in ALL_CRITERIA.items():
        j = col_idx[col]
        x = R[:, :, j, :]
        in_audit = full in POLARITY_AUDIT_CRITERIA
        table.append({
            "Criterion": full, "Column": col,
            "Allocation_Group": "O" if full in BENEFIT_CRITERIA else "S",
            "Label_direction": LABEL_DIRECTION.get(full, "favorable"),
            "TOPSIS_polarity_in_use": "reverse-coded (cost-type)" if full in reverse
                                      else "higher_is_better (benefit-type)",
            "Rating_mean_m": float(x[:, :, 1].mean()),
            "Rating_min_l": float(x[:, :, 0].min()),
            "Rating_max_u": float(x[:, :, 2].max()),
            "Needs_questionnaire_check": in_audit,
            "Questionnaire_semantics_verified": bool(POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE),
            "Questionnaire_item_stem": QUESTIONNAIRE_ITEM_STEM,
            "Questionnaire_definition": QUESTIONNAIRE_EVIDENCE.get(full, ""),
            "Verified_rating_direction": ("high = more favorable (simpler / cheaper / less data-dependent)"
                                          if in_audit else "high = more favorable"),
            "Action": (("verified against questionnaire: retain higher-is-better, no reversal"
                        if POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE else
                        "Check anchors in questionnaire + coded cells in dft.xlsx; "
                        "if high = more complexity/dependency/cost, add to REVERSE_CODED_CRITERIA")
                       if in_audit else "none (favorable by label)"),
        })
    polarity_table = pd.DataFrame(table)

    by_feat, by_exp = [], []
    for full in POLARITY_AUDIT_CRITERIA:
        j = col_idx[ALL_CRITERIA[full]]
        for p, alt in enumerate(alternatives):
            v = R[:, p, j, :]   # K x 3
            by_feat.append({
                "Criterion": full, "Feature": NAME_MAP.get(alt, alt), "Method": alt,
                "mean_m": float(v[:, 1].mean()), "min_m": float(v[:, 1].min()),
                "max_m": float(v[:, 1].max()), "sd_m": float(v[:, 1].std(ddof=0)),
                "mean_l": float(v[:, 0].mean()), "mean_u": float(v[:, 2].mean()),
            })
        for k, s in enumerate(sheets):
            v = R[k, :, j, :]   # P x 3
            by_exp.append({
                "Criterion": full, "Expert_sheet": s,
                "mean_m": float(v[:, 1].mean()), "min_m": float(v[:, 1].min()),
                "max_m": float(v[:, 1].max()),
                "mean_l": float(v[:, 0].mean()), "mean_u": float(v[:, 2].mean()),
            })
    return polarity_table, pd.DataFrame(by_feat), pd.DataFrame(by_exp)


def build_empirical_coding_check(tfn_dfs, alternatives, criteria):
    """Supporting evidence that the coded ratings in dft.xlsx follow the questionnaire direction.
    If a criterion were stored reversed (high = worse), its ratings would correlate NEGATIVELY with
    the other favorable-framed criteria. For every criterion we correlate its centroid ratings with the
    mean of the OTHER eight outcome criteria (leave-one-out for outcome criteria), pooled over
    expert x feature cells, per expert, and over the nine feature means. The 95% CI resamples experts.
    This cannot prove the coding (a reversed item could still correlate positively by chance), so the
    verdict is worded CONSISTENT / INCONCLUSIVE / INCONSISTENT and complements, not replaces, the
    questionnaire evidence and the author's confirmation of the source file."""
    from scipy.stats import spearmanr
    sheets, R = ratings_tensor(tfn_dfs, len(alternatives), len(criteria))
    X = R.mean(axis=-1)                                  # K x P x C centroid
    col_idx = {c: j for j, c in enumerate(criteria)}
    out_cols = [col_idx[ALL_CRITERIA[f]] for f in BENEFIT_CRITERIA]
    rng = np.random.default_rng(EMPIRICAL_CHECK_SEED)
    K = X.shape[0]

    def rho_pooled(idx, j, ref_cols):
        x = X[idx][:, :, j].ravel()
        r = X[idx][:, :, ref_cols].mean(axis=2).ravel()
        if np.ptp(x) < 1e-12 or np.ptp(r) < 1e-12:
            return np.nan
        return float(spearmanr(x, r)[0])

    rows = []
    for full, col in ALL_CRITERIA.items():
        j = col_idx[col]
        ref_cols = [c for c in out_cols if c != j]
        pooled = rho_pooled(np.arange(K), j, ref_cols)
        boots = [v for v in (rho_pooled(rng.integers(0, K, K), j, ref_cols)
                             for _ in range(EMPIRICAL_CHECK_BOOT_B)) if not np.isnan(v)]
        lo, hi = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
                  if len(boots) > 0.5 * EMPIRICAL_CHECK_BOOT_B else (np.nan, np.nan))
        per_exp = []
        for k in range(K):
            x = X[k][:, j]; r = X[k][:, ref_cols].mean(axis=1)
            per_exp.append(np.nan if (np.ptp(x) < 1e-12 or np.ptp(r) < 1e-12) else float(spearmanr(x, r)[0]))
        per_exp = np.array(per_exp, dtype=float)
        fx = X[:, :, j].mean(axis=0); fr = X[:, :, ref_cols].mean(axis=(0, 2))
        rho_feat = float(spearmanr(fx, fr)[0])
        if np.isnan(lo):
            verdict = "NOT_COMPUTABLE"
        elif hi < 0:
            verdict = "INCONSISTENT_POSSIBLE_REVERSAL"
        elif lo > 0:
            verdict = "CONSISTENT_WITH_FAVORABLE_CODING"
        else:
            verdict = "INCONCLUSIVE"
        rows.append({
            "Criterion": full, "Column": col, "In_polarity_audit": full in POLARITY_AUDIT_CRITERIA,
            "Reference": "mean of the other outcome criteria" if full in BENEFIT_CRITERIA
                         else "mean of the 8 outcome criteria",
            "Pooled_spearman_rho": pooled, "Pooled_rho_CI95_lo": lo, "Pooled_rho_CI95_hi": hi,
            "Share_experts_positive_rho": float(np.nanmean(per_exp > 0)),
            "N_experts": K, "Feature_level_spearman_rho": rho_feat,
            "Empirical_verdict": verdict,
            "Source_coding_confirmed_by_author": bool(SOURCE_CODING_CONFIRMED_BY_AUTHOR),
        })
    return pd.DataFrame(rows)


def run_polarity_sensitivity(w_star, agg_df, alternatives, criteria, primary_results):
    """TOPSIS under hypothetical reversal sets, vs. the configured primary.
    Uses the PRIMARY weighting. Reversal sets are absolute (not toggles)."""
    w_hat = primary_weights(w_star)
    prim_rank = dict(zip(primary_results["Method"], primary_results["Rank"]))
    prim_rc = dict(zip(primary_results["Method"], primary_results["RC"]))
    prim_vec = [prim_rank[m] for m in alternatives]

    scenarios = {"configured_primary": set(REVERSE_CODED_CRITERIA)}
    scenarios.update(POLARITY_SCENARIOS)

    rows, summ = [], []
    for name, rev in scenarios.items():
        b_cols, c_cols = polarity_columns(rev)
        norm_s, _, _ = normalize_fuzzy_matrix(agg_df, b_cols, c_cols)
        run = run_allocation(w_star, norm_s, alternatives, criteria, w_hat=w_hat)
        res = run["results"]
        vec = _rank_vector(res, alternatives)
        rho, tau = _rank_corr(prim_vec, vec)
        rk = dict(zip(res["Method"], res["Rank"]))
        rc = dict(zip(res["Method"], res["RC"]))
        shifts = []
        for m in alternatives:
            sh = int(prim_rank[m] - rk[m])
            shifts.append(abs(sh))
            rows.append({
                "Scenario": name, "Reverse_coded": "; ".join(sorted(rev)) or "(none)",
                "Feature": NAME_MAP.get(m, m), "Method": m,
                "RC": rc[m], "Rank": int(rk[m]),
                "RC_primary": prim_rc[m], "Rank_primary": int(prim_rank[m]),
                "RC_diff_vs_primary": rc[m] - prim_rc[m],
                "Rank_shift_primary_minus_scenario": sh,
                "Flag_shift_ge_threshold": abs(sh) >= RANK_SHIFT_FLAG,
            })
        top1_p = min(prim_rank, key=prim_rank.get)
        top1_s = min(rk, key=rk.get)
        summ.append({
            "Scenario": name, "Reverse_coded": "; ".join(sorted(rev)) or "(none)",
            "Spearman_rho_vs_primary": rho, "Kendall_tau_vs_primary": tau,
            "max_abs_rank_shift": max(shifts), "n_features_rank_changed": int(sum(s > 0 for s in shifts)),
            "n_features_flagged": int(sum(s >= RANK_SHIFT_FLAG for s in shifts)),
            "top1_primary": NAME_MAP.get(top1_p, top1_p), "top1_scenario": NAME_MAP.get(top1_s, top1_s),
            "top1_changed": bool(top1_p != top1_s),
            "PRIMARY_WEIGHTING": PRIMARY_WEIGHTING,
        })
    return pd.DataFrame(rows), pd.DataFrame(summ)


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------

def run_pipeline(input_file=INPUT_FILE, fahp_weights_file=FAHP_WEIGHTS_FILE):

    if not POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE:
        warnings.warn(
            "[TOPSIS] POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE is False: the rating "
            "direction of 'Cost Efficiency', 'Implementation Complexity' and "
            "'Data Dependency' has not been confirmed against the original "
            "questionnaire anchors / dft.xlsx coding. Inspect "
            "STRESS_TOPSIS_criterion_polarity_audit.csv, set REVERSE_CODED_CRITERIA "
            "if needed, then set the flag to True."
        )

    # 1. Load + parse expert sheets
    raw_dfs = load_expert_sheets(input_file)
    alternatives, criteria = validate_expert_sheets(raw_dfs, expected_k=8)
    tfn_dfs = parse_all_sheets(raw_dfs)

    # 2. Aggregate via arithmetic mean -> X~
    agg_df, K = aggregate_experts_arithmetic(tfn_dfs)
    print(f"[INFO] Aggregated {K} expert fuzzy rating sheets via arithmetic mean.")

    # 3. Load w*
    w_star = load_fahp_weights(fahp_weights_file)

    missing_cols = [c for c in ALL_CRITERIA.values() if c not in criteria]
    if missing_cols:
        raise ValueError(
            f"[TOPSIS] Columns declared as benefit/cost not found in "
            f"dft.xlsx: {missing_cols}"
        )

    # 3b. (2.5) polarity table + raw rating audit (always written)
    pol_table, pol_feat, pol_exp = build_polarity_audit(tfn_dfs, alternatives, criteria)
    pol_table.to_csv(OUTPUT_POLARITY_AUDIT_FILE, index=False)
    pol_feat.to_csv(OUTPUT_RATING_AUDIT_FEATURE_FILE, index=False)
    pol_exp.to_csv(OUTPUT_RATING_AUDIT_EXPERT_FILE, index=False)
    emp_df = build_empirical_coding_check(tfn_dfs, alternatives, criteria)
    emp_df.to_csv(OUTPUT_POLARITY_EMPIRICAL_FILE, index=False)
    pol_table = pol_table.merge(emp_df[["Criterion", "Pooled_spearman_rho", "Pooled_rho_CI95_lo",
                                        "Pooled_rho_CI95_hi", "Empirical_verdict",
                                        "Source_coding_confirmed_by_author"]],
                                on="Criterion", how="left")
    pol_table.to_csv(OUTPUT_POLARITY_AUDIT_FILE, index=False)
    print("[INFO] Empirical coding check (audited criteria): " + "; ".join(
        f"{r.Criterion}: rho={r.Pooled_spearman_rho:.2f} [{r.Pooled_rho_CI95_lo:.2f},{r.Pooled_rho_CI95_hi:.2f}] {r.Empirical_verdict}"
        for r in emp_df[emp_df.In_polarity_audit].itertuples()))
    print(f"[INFO] Criterion polarity in use: reverse-coded = "
          f"{sorted(REVERSE_CODED_CRITERIA) or '(none; all higher-is-better)'}")
    print(f"[INFO] Saved polarity audit to {OUTPUT_POLARITY_AUDIT_FILE}")

    # 4. Normalize with the CONFIGURED polarity -> R~
    benefit_cols, cost_cols = polarity_columns(REVERSE_CODED_CRITERIA)
    norm_df, u_plus, l_minus = normalize_fuzzy_matrix(agg_df, benefit_cols, cost_cols)

    # 5-8. Both weightings are always computed; PRIMARY_WEIGHTING selects the primary.
    run_7525 = run_weighting(w_star, norm_df, alternatives, criteria, "75_25")
    run_raw = run_weighting(w_star, norm_df, alternatives, criteria, "raw")
    primary = run_raw if PRIMARY_WEIGHTING == "raw" else run_7525
    primary_label = "raw FAHP (no Eq. 17)" if PRIMARY_WEIGHTING == "raw" else "75/25"

    results = primary["results"]
    w_hat = primary["w_hat"]
    fpis, fnis = primary["fpis"], primary["fnis"]
    D_plus, D_minus, CC = primary["D_plus"], primary["D_minus"], primary["CC"]

    print(f"[INFO] Criterion weights, PRIMARY model ({primary_label}):")
    for c in criteria:
        print(f"    {c:28s} {w_hat[c]:.6f}")

    # SAVE PRIMARY OUTPUTS (file names/columns unchanged; 'Weighting' column added)
    out_primary = results.copy()
    out_primary["Weighting"] = PRIMARY_WEIGHTING
    out_primary.to_csv(OUTPUT_RESULT_FILE, index=False)
    run_7525["results"].to_csv(OUTPUT_RESULT_7525_FILE, index=False)
    run_raw["results"].to_csv(OUTPUT_RESULT_RAW_FILE, index=False)

    pd.DataFrame(
        {c: [w_hat[c]] for c in criteria}
    ).T.rename(columns={0: "w_hat"}).to_csv(_p("STRESS_TOPSIS_rescaled_weights.csv"))

    fpis_fnis_df = pd.DataFrame({
        "Criterion": criteria,
        "FPIS_l": [fpis[c][0] for c in criteria],
        "FPIS_m": [fpis[c][1] for c in criteria],
        "FPIS_u": [fpis[c][2] for c in criteria],
        "FNIS_l": [fnis[c][0] for c in criteria],
        "FNIS_m": [fnis[c][1] for c in criteria],
        "FNIS_u": [fnis[c][2] for c in criteria],
    })
    fpis_fnis_df.to_csv(_p("STRESS_TOPSIS_fpis_fnis.csv"), index=False)

    print(f"\n===== STRESS-AWARE FUZZY TOPSIS RESULTS (PRIMARY: {primary_label}) =====\n")
    print(results.to_string(index=False))
    print(f"\n[INFO] Saved ranking to {OUTPUT_RESULT_FILE}")

    # 9. (2.1) endogenous shares + raw vs 75/25
    print("\n===== (2.1) ENDOGENOUS GROUP SHARES AND RAW-FAHP vs 75/25 =====\n")
    shares_df, wcmp_df, q_O, q_S = build_endogenous_share_tables(w_star)
    shares_df.to_csv(OUTPUT_ENDOGENOUS_FILE, index=False)
    wcmp_df.to_csv(OUTPUT_WEIGHTS_COMPARE_FILE, index=False)
    print(f"[INFO] Endogenous shares from w*: q_O={q_O:.4f}, q_S={q_S:.4f} "
          f"(imposed {BENEFIT_SHARE:.2f}/{COST_SHARE:.2f}).")

    detail, summary, verdict, action = compare_raw_vs_7525(run_raw, run_7525, alternatives, q_O, q_S)
    detail.to_csv(OUTPUT_RAW_VS_7525_FILE, index=False)
    summary.to_csv(OUTPUT_RAW_VS_7525_SUMMARY_FILE, index=False)
    print(detail[["Feature", "RC_raw_FAHP", "Rank_raw_FAHP", "RC_75_25", "Rank_75_25",
                  "Rank_shift_75_25_minus_raw"]].to_string(index=False))
    print(f"\n[VERDICT] {verdict}")
    print(f"[DECISION RULE] {action}")
    print("[NOTE] The script does NOT switch the primary model automatically; "
          f"current PRIMARY_WEIGHTING = '{PRIMARY_WEIGHTING}'.")

    # 10. Allocation sensitivity: 5 fixed splits + raw FAHP (6 scenarios)
    print("\n===== BENEFIT/COST ALLOCATION SENSITIVITY ANALYSIS (+ raw FAHP) =====\n")
    sensitivity_df, weights_sensitivity_df = run_allocation_sensitivity(
        w_star, norm_df, alternatives, criteria,
        baseline_results=run_7525["results"], primary_results=results,
    )
    print(f"[INFO] Saved allocation sensitivity results to {OUTPUT_SENSITIVITY_FILE}")

    # 11. (2.5) "what if reversed" sensitivity
    print("\n===== (2.5) POLARITY SENSITIVITY (hypothetical reversal) =====\n")
    pol_sens_df, pol_sum_df = run_polarity_sensitivity(w_star, agg_df, alternatives, criteria, results)
    pol_sens_df.to_csv(OUTPUT_POLARITY_SENS_FILE, index=False)
    pol_sum_df.to_csv(OUTPUT_POLARITY_SENS_SUMMARY_FILE, index=False)
    print(pol_sum_df.to_string(index=False))

    return {
        "results": results,
        "aggregated_matrix": agg_df,
        "normalized_matrix": norm_df,
        "weighted_matrix": primary["weighted_matrix"],
        "w_hat": w_hat,
        "fpis": fpis,
        "fnis": fnis,
        "D_plus": D_plus,
        "D_minus": D_minus,
        "CC": CC,
        "allocation_sensitivity": sensitivity_df,
        "allocation_sensitivity_weights": weights_sensitivity_df,
        # ---- added in this revision ----
        "primary_weighting": PRIMARY_WEIGHTING,
        "results_75_25": run_7525["results"],
        "results_raw_fahp": run_raw["results"],
        "endogenous_shares": shares_df,
        "raw_vs_7525": detail,
        "raw_vs_7525_summary": summary,
        "raw_vs_7525_verdict": verdict,
        "polarity_audit": pol_table,
        "polarity_sensitivity": pol_sens_df,
        "polarity_sensitivity_summary": pol_sum_df,
    }


if __name__ == "__main__":
    run_pipeline()
