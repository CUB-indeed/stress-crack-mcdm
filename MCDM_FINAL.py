
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

PROJECT_DIR = os.environ.get("MCDM_PROJECT_DIR", "/home/coder/project")

# decision-rule thresholds for "SRPI ranking is (not) materially affected by the weighting"
SRPI_RHO_MIN = 0.95
SRPI_TAU_MIN = 0.85
SRPI_MAX_RANK_SHIFT = 1
RANK_SHIFT_FLAG = 2

topsis = pd.read_csv(os.path.join(PROJECT_DIR, "STRESS_TOPSIS_RESULT.csv"))
PRIMARY_WEIGHTING_USED = (str(topsis["Weighting"].iloc[0]).strip()
                          if "Weighting" in topsis.columns else "unknown")
print(f"[INFO] Primary TOPSIS weighting read from STRESS_TOPSIS_RESULT.csv: {PRIMARY_WEIGHTING_USED}")
dematel = pd.read_csv(os.path.join(PROJECT_DIR, "STRESS_DEMATEL_RESULTS.csv"))

topsis.columns = topsis.columns.str.strip()
dematel.columns = dematel.columns.str.strip()

name_map = {
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

topsis["Feature"] = topsis["Method"].replace(name_map)
dematel["Feature"] = dematel["Feature"].str.strip()
dematel = dematel.rename(columns={"Prominence_star": "DR"})

required_topsis = ["Feature", "RC"]
required_dematel = ["Feature", "FSI", "DR"]

missing_topsis = [c for c in required_topsis if c not in topsis.columns]
missing_dematel = [c for c in required_dematel if c not in dematel.columns]
if missing_topsis:
    raise ValueError(f"Missing TOPSIS columns: {missing_topsis}")
if missing_dematel:
    raise ValueError(f"Missing DEMATEL columns: {missing_dematel}")

topsis = topsis[required_topsis]
dematel = dematel[required_dematel]

print("Unique TOPSIS Features :", topsis["Feature"].nunique())
print("Unique DEMATEL Features:", dematel["Feature"].nunique())
print("Matched Features      :", len(set(topsis["Feature"]) & set(dematel["Feature"])))

primary = topsis.copy()
primary = primary.sort_values("RC", ascending=False).reset_index(drop=True)
primary["Primary_Rank"] = np.arange(len(primary)) + 1
primary = primary.rename(columns={"RC": "TOPSIS_RC"})

df = pd.merge(topsis, dematel, on="Feature", how="inner")
if df.empty:
    raise ValueError("Merge failed: feature names do not match")

def minmax(x):
    x = np.asarray(x, dtype=float)
    denom = x.max() - x.min()
    if denom == 0:
        return np.zeros(len(x))
    return (x - x.min()) / (denom + 1e-12)

def vulnerability(fsi, mode):
    fsi = np.asarray(fsi, dtype=float)
    if mode == "absolute":
        return 1.0 - fsi
    if mode == "minmax":
        denom = fsi.max() - fsi.min()
        if denom == 0:
            return np.zeros(len(fsi))
        return (fsi.max() - fsi) / (denom + 1e-12)
    if mode == "zero":
        return np.zeros(len(fsi))
    raise ValueError(mode)

def srpi(rc, dr, fsi, mode, lambda_amp=1.0, mu_disc=1.0):
    p = minmax(rc)
    c = minmax(dr)
    v = vulnerability(fsi, mode)
    phi = p * (1 + lambda_amp * c) / (1 + mu_disc * v)
    return phi / phi.max()

def rank_desc(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(ascending=False, method="min").astype(int).values

df["V_abs"] = vulnerability(df["FSI"], "absolute")
df["V_minmax"] = vulnerability(df["FSI"], "minmax")
df["SRPI_abs"] = srpi(df["RC"], df["DR"], df["FSI"], "absolute")
df["SRPI_minmax"] = srpi(df["RC"], df["DR"], df["FSI"], "minmax")
df["SRPI_V0"] = srpi(df["RC"], df["DR"], df["FSI"], "zero")
df["Rank_TOPSIS"] = rank_desc(df["RC"])
df["Rank_SRPI_abs"] = rank_desc(df["SRPI_abs"])
df["Rank_SRPI_minmax"] = rank_desc(df["SRPI_minmax"])
df["Rank_SRPI_V0"] = rank_desc(df["SRPI_V0"])

secondary = df[[
    "Feature", "RC", "FSI", "DR",
    "SRPI_abs", "Rank_SRPI_abs", "SRPI_minmax", "Rank_SRPI_minmax",
]].rename(columns={
    "SRPI_abs": "SRPI_exploratory",
    "Rank_SRPI_abs": "Exploratory_SRPI_Rank",
})

result = pd.merge(primary, secondary, on="Feature", how="left")
result = result[[
    "Primary_Rank", "Feature", "TOPSIS_RC", "FSI", "DR",
    "SRPI_exploratory", "Exploratory_SRPI_Rank",
    "SRPI_minmax", "Rank_SRPI_minmax",
]]
result = result.rename(columns={
    "SRPI_exploratory": "SRPI",
    "Exploratory_SRPI_Rank": "SRPI_Rank",
    "Rank_SRPI_minmax": "SRPI_minmax_Rank",
})
result = result.sort_values("Primary_Rank").reset_index(drop=True)

output_file = os.path.join(PROJECT_DIR, "Final_Stress_Coupled_Result.csv")
result.to_csv(output_file, index=False)

recon = result[["Feature", "Primary_Rank", "SRPI_Rank", "SRPI_minmax_Rank"]].copy()
recon = recon.rename(columns={
    "Primary_Rank": "TOPSIS_rank",
    "SRPI_Rank": "SRPI_abs_rank",
    "SRPI_minmax_Rank": "SRPI_minmax_rank",
})
recon["rank_shift_abs"] = recon["TOPSIS_rank"] - recon["SRPI_abs_rank"]
recon["rank_shift_minmax"] = recon["TOPSIS_rank"] - recon["SRPI_minmax_rank"]
recon.to_csv(os.path.join(PROJECT_DIR, "RANK_RECONCILIATION_TOPSIS_SRPI.csv"), index=False)

rows = []
scenarios = {
    "SRPI_abs": df["SRPI_abs"],
    "SRPI_minmax": df["SRPI_minmax"],
    "SRPI_V0": df["SRPI_V0"],
}
for name, s in scenarios.items():
    rho_t, _ = spearmanr(df["RC"], s)
    tau_t, _ = kendalltau(df["RC"], s)
    rho_a, _ = spearmanr(df["SRPI_abs"], s)
    tau_a, _ = kendalltau(df["SRPI_abs"], s)
    rows.append({
        "scenario": name,
        "rho_vs_topsis": rho_t, "tau_vs_topsis": tau_t,
        "rho_vs_srpi_abs": rho_a, "tau_vs_srpi_abs": tau_a,
        "max_abs_score_diff_vs_topsis_rc_normalized": float(
            np.abs(np.asarray(s) - df["RC"].values / df["RC"].max()).max()
        ),
        "max_abs_score_diff_vs_srpi_abs": float(np.abs(np.asarray(s) - df["SRPI_abs"].values).max()),
        "fsi_min": float(df["FSI"].min()),
        "fsi_max": float(df["FSI"].max()),
        "fsi_spread": float(df["FSI"].max() - df["FSI"].min()),
        "v_abs_min": float(df["V_abs"].min()),
        "v_abs_max": float(df["V_abs"].max()),
    })
pd.DataFrame(rows).to_csv(os.path.join(PROJECT_DIR, "SRPI_VULNERABILITY_SCENARIOS.csv"), index=False)

score_table = df[[
    "Feature", "RC", "FSI", "DR", "V_abs", "V_minmax",
    "SRPI_abs", "SRPI_minmax", "SRPI_V0",
    "Rank_TOPSIS", "Rank_SRPI_abs", "Rank_SRPI_minmax", "Rank_SRPI_V0",
]].sort_values("Rank_TOPSIS")
score_table.to_csv(os.path.join(PROJECT_DIR, "SRPI_SCORE_TABLE.csv"), index=False)

print(result)
print(recon)
print(pd.DataFrame(rows))
print("Saved:", output_file)


# ==================================================================
# SRPI under BOTH weightings: 75/25 (Eq. 17) and raw FAHP w*
# ==================================================================

def _srpi_under(path, label):
    t = pd.read_csv(path)
    t.columns = t.columns.str.strip()
    t["Feature"] = t["Method"].replace(name_map)
    m = pd.merge(t[["Feature", "RC"]], dematel, on="Feature", how="inner")
    if len(m) != len(t):
        raise ValueError(f"[FINAL] Feature merge failed for {label}: {len(m)} of {len(t)} rows matched.")
    m["SRPI_abs"] = srpi(m["RC"], m["DR"], m["FSI"], "absolute")
    m["SRPI_minmax"] = srpi(m["RC"], m["DR"], m["FSI"], "minmax")
    m["Rank_TOPSIS"] = rank_desc(m["RC"])
    m["Rank_SRPI_abs"] = rank_desc(m["SRPI_abs"])
    m["Rank_SRPI_minmax"] = rank_desc(m["SRPI_minmax"])
    return m.set_index("Feature")


def _corr(a, b):
    rho, _ = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    return float(rho), float(tau)


path_7525 = os.path.join(PROJECT_DIR, "STRESS_TOPSIS_RESULT_75_25.csv")
path_raw = os.path.join(PROJECT_DIR, "STRESS_TOPSIS_RESULT_raw_fahp.csv")

if not (os.path.exists(path_7525) and os.path.exists(path_raw)):
    warnings.warn("[FINAL] STRESS_TOPSIS_RESULT_75_25.csv / _raw_fahp.csv not found; run the revised "
                  "MCDM_Topsis.py first. Skipping the raw-vs-75/25 SRPI comparison.")
else:
    a = _srpi_under(path_7525, "75/25")
    b = _srpi_under(path_raw, "raw FAHP")
    b = b.loc[a.index]

    # integrity check: the primary file must equal the run its 'Weighting' column names
    ref = {"75_25": a, "raw": b}.get(PRIMARY_WEIGHTING_USED)
    if ref is not None:
        prim_rc = topsis.set_index("Feature")["RC"].reindex(ref.index)
        if not np.allclose(prim_rc.values, ref["RC"].values, atol=1e-9):
            warnings.warn("[FINAL] STRESS_TOPSIS_RESULT.csv does not match the '%s' run: files may be "
                          "stale. Rerun MCDM_Topsis.py before FINAL." % PRIMARY_WEIGHTING_USED)

    cmp_df = pd.DataFrame({
        "Feature": a.index,
        "RC_75_25": a["RC"].values, "Rank_TOPSIS_75_25": a["Rank_TOPSIS"].values,
        "SRPI_75_25": a["SRPI_abs"].values, "Rank_SRPI_75_25": a["Rank_SRPI_abs"].values,
        "SRPI_minmax_75_25": a["SRPI_minmax"].values, "Rank_SRPI_minmax_75_25": a["Rank_SRPI_minmax"].values,
        "RC_raw_FAHP": b["RC"].values, "Rank_TOPSIS_raw_FAHP": b["Rank_TOPSIS"].values,
        "SRPI_raw_FAHP": b["SRPI_abs"].values, "Rank_SRPI_raw_FAHP": b["Rank_SRPI_abs"].values,
        "SRPI_minmax_raw_FAHP": b["SRPI_minmax"].values, "Rank_SRPI_minmax_raw_FAHP": b["Rank_SRPI_minmax"].values,
        "FSI": a["FSI"].values, "DR": a["DR"].values,
    })
    cmp_df["TOPSIS_rank_shift_75_25_minus_raw"] = cmp_df["Rank_TOPSIS_75_25"] - cmp_df["Rank_TOPSIS_raw_FAHP"]
    cmp_df["SRPI_rank_shift_75_25_minus_raw"] = cmp_df["Rank_SRPI_75_25"] - cmp_df["Rank_SRPI_raw_FAHP"]
    cmp_df["SRPI_abs_diff_raw_minus_75_25"] = cmp_df["SRPI_raw_FAHP"] - cmp_df["SRPI_75_25"]
    cmp_df["Flag_SRPI_shift_ge_threshold"] = cmp_df["SRPI_rank_shift_75_25_minus_raw"].abs() >= RANK_SHIFT_FLAG
    cmp_df = cmp_df.sort_values("Rank_SRPI_75_25").reset_index(drop=True)
    cmp_df.to_csv(os.path.join(PROJECT_DIR, "SRPI_RAW_vs_7525_COMPARISON.csv"), index=False)

    rho_t, tau_t = _corr(cmp_df["Rank_TOPSIS_75_25"], cmp_df["Rank_TOPSIS_raw_FAHP"])
    rho_s, tau_s = _corr(cmp_df["Rank_SRPI_75_25"], cmp_df["Rank_SRPI_raw_FAHP"])
    rho_m, tau_m = _corr(cmp_df["Rank_SRPI_minmax_75_25"], cmp_df["Rank_SRPI_minmax_raw_FAHP"])
    rho_w1, tau_w1 = _corr(cmp_df["Rank_TOPSIS_75_25"], cmp_df["Rank_SRPI_75_25"])
    rho_w2, tau_w2 = _corr(cmp_df["Rank_TOPSIS_raw_FAHP"], cmp_df["Rank_SRPI_raw_FAHP"])
    max_shift_s = int(cmp_df["SRPI_rank_shift_75_25_minus_raw"].abs().max())
    top1_75 = cmp_df.sort_values("Rank_SRPI_75_25").iloc[0]["Feature"]
    top1_raw = cmp_df.sort_values("Rank_SRPI_raw_FAHP").iloc[0]["Feature"]
    top1_t75 = cmp_df.sort_values("Rank_TOPSIS_75_25").iloc[0]["Feature"]
    top1_traw = cmp_df.sort_values("Rank_TOPSIS_raw_FAHP").iloc[0]["Feature"]

    stable = (rho_s >= SRPI_RHO_MIN and tau_s >= SRPI_TAU_MIN
              and max_shift_s <= SRPI_MAX_RANK_SHIFT and top1_75 == top1_raw)
    verdict = "SRPI_RANKING_ROBUST_TO_WEIGHTING" if stable else "SRPI_RANKING_DEPENDS_ON_WEIGHTING"
    note = ("SRPI ranking is preserved under raw FAHP: report that the weighting choice does not drive SRPI."
            if stable else
            "SRPI ranking changes with the weighting: all SRPI statements (Section 5.1-5.3, Table 6, "
            "Abstract, Conclusion) must be reported under the chosen primary weighting "
            f"('{PRIMARY_WEIGHTING_USED}') and the other weighting as a scenario.")

    summary = pd.DataFrame([
        ("primary_weighting_used", PRIMARY_WEIGHTING_USED),
        ("TOPSIS_rho_raw_vs_75_25", rho_t), ("TOPSIS_tau_raw_vs_75_25", tau_t),
        ("SRPI_abs_rho_raw_vs_75_25", rho_s), ("SRPI_abs_tau_raw_vs_75_25", tau_s),
        ("SRPI_minmax_rho_raw_vs_75_25", rho_m), ("SRPI_minmax_tau_raw_vs_75_25", tau_m),
        ("TOPSIS_vs_SRPI_rho_under_75_25", rho_w1), ("TOPSIS_vs_SRPI_tau_under_75_25", tau_w1),
        ("TOPSIS_vs_SRPI_rho_under_raw", rho_w2), ("TOPSIS_vs_SRPI_tau_under_raw", tau_w2),
        ("SRPI_max_abs_rank_shift", max_shift_s),
        ("SRPI_n_features_flagged_shift_ge_%d" % RANK_SHIFT_FLAG, int(cmp_df["Flag_SRPI_shift_ge_threshold"].sum())),
        ("SRPI_top1_75_25", top1_75), ("SRPI_top1_raw_FAHP", top1_raw),
        ("TOPSIS_top1_75_25", top1_t75), ("TOPSIS_top1_raw_FAHP", top1_traw),
        ("threshold_rho_min", SRPI_RHO_MIN), ("threshold_tau_min", SRPI_TAU_MIN),
        ("threshold_max_rank_shift", SRPI_MAX_RANK_SHIFT),
        ("verdict", verdict), ("note", note),
    ], columns=["Metric", "Value"])
    summary.to_csv(os.path.join(PROJECT_DIR, "SRPI_RAW_vs_7525_SUMMARY.csv"), index=False)

    print("\n===== (2.1) SRPI UNDER 75/25 vs RAW FAHP =====\n")
    print(cmp_df[["Feature", "Rank_TOPSIS_75_25", "Rank_TOPSIS_raw_FAHP", "Rank_SRPI_75_25",
                  "Rank_SRPI_raw_FAHP", "SRPI_rank_shift_75_25_minus_raw"]].to_string(index=False))
    print(f"\nSRPI raw vs 75/25: rho={rho_s:.4f}, tau={tau_s:.4f}, max |rank shift|={max_shift_s}")
    print(f"[VERDICT] {verdict}")
    print(f"[NOTE] {note}")
    print("Saved: SRPI_RAW_vs_7525_COMPARISON.csv, SRPI_RAW_vs_7525_SUMMARY.csv")
