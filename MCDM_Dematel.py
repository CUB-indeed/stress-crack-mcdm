
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


INPUT_FILE = _p("dfd.xlsx")

# --- new outputs  ---
OUTPUT_CENTRALITY_FILE = _p("DEMATEL_centrality_benchmark.csv")
OUTPUT_CENTRALITY_SUMMARY_FILE = _p("DEMATEL_centrality_benchmark_summary.csv")

# Decision rule for 2.4 ("nearly identical" => reduce the novelty claim).
# NEARLY_IDENTICAL only if ALL hold for the D+R-vs-centrality pairs; thresholds are
# author-adjustable and are written verbatim into the summary file.
CENTRALITY_RHO_MIN = 0.95
CENTRALITY_TAU_MIN = 0.85
CENTRALITY_MAX_RANK_SHIFT = 1
CENTRALITY_RANK_SHIFT_FLAG = 2   # |rank shift| >= this is flagged
CENTRALITY_PATH_K_MAX = 6        # longest path length in the path-length ablation
CENTRALITY_BOOT_B = 2000         # expert-bootstrap resamples
CENTRALITY_BOOT_SEED = 20261005
OUTPUT_CENTRALITY_ABLATION_FILE = _p("DEMATEL_centrality_path_ablation.csv")
OUTPUT_CENTRALITY_BOOT_FILE = _p("DEMATEL_centrality_expert_bootstrap.csv")

STRESS_SET = np.array([-50, -25, 0, 25, 50])
STRESS_MODE = "cell_sym"   # symmetry-preserving (see apply_stress). Legacy
                            # 'row'/'cell' modes break symmetry -- avoid.

STRESS_SENSITIVITY_SOURCE = "centrality"  # 'T0'/'Z0' degenerate to all-zero
                                           # once Z0 is symmetrized (row==col
                                           # identically); use connectedness
                                           # deviation instead (see function).

TAU_C_DEMATEL = 0.30      # crack tolerance for Delta_p^{(D)} (no CR term here)
GAMMA_AMPLIFICATION = 1.0  # amplification coefficient (Eq. stress_amplification)
GAMMA_SWEEP_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]  # for calibration/sensitivity

DELTA_UNIFIED_SCORE = 0.5  # delta in Score_p = RC_p * (1 + delta * Prominence_norm_p)

EPS = 1e-9


# ------------------------------------------------------------------
# STEP 1: Load + parse (TFN, with NaN warning)
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
        return (0.0, 0.0, 0.0)

    parts = [float(x.strip()) for x in str(cell).split(",")]

    if len(parts) == 3:
        return tuple(parts)
    elif len(parts) == 1:
        v = parts[0]
        warnings.warn(
            f"[DEMATEL] Crisp (non-TFN) value encountered ({context}); "
            f"replicated as ({v},{v},{v})."
        )
        return (v, v, v)
    else:
        raise ValueError(
            f"[DEMATEL] Cell ({context}) has {len(parts)} comma-separated "
            f"values; expected 1 (crisp) or 3 (l,m,u). Value: {cell!r}"
        )


def parse_all_sheets(dfs):
    parsed = {}
    for name, df in dfs.items():
        parsed[name] = df.apply(
            lambda col: [parse_tfn(v, context=f"sheet={name}") for v in col]
        )
    return parsed


# ------------------------------------------------------------------
# STEP 2: Arithmetic-mean aggregation across experts
# ------------------------------------------------------------------

def aggregate_experts_arithmetic(dfs):
    sheets = list(dfs.values())
    agg = sheets[0].copy()
    n_rows, n_cols = agg.shape

    for i in range(n_rows):
        for j in range(n_cols):
            l = np.mean([df.iloc[i, j][0] for df in sheets])
            m = np.mean([df.iloc[i, j][1] for df in sheets])
            u = np.mean([df.iloc[i, j][2] for df in sheets])
            agg.iloc[i, j] = (l, m, u)

    return agg


def enforce_zero_diagonal_fuzzy(df):
    n = df.shape[0]
    out = df.copy()
    for i in range(n):
        out.iloc[i, i] = (0.0, 0.0, 0.0)
    return out


# ------------------------------------------------------------------
# STEP 3: Centroid defuzzification -> crisp Z0, diagonal forced to 0
# ------------------------------------------------------------------

def defuzzify(tfn):
    l, m, u = tfn
    return (l + m + u) / 3.0


def build_crisp_matrix(fuzzy_df):
    n = fuzzy_df.shape[0]
    labels = list(fuzzy_df.index)
    Z0 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                Z0[i, j] = defuzzify(fuzzy_df.iloc[i, j])
            # diagonal stays 0.0 (self-influence excluded, standard DEMATEL)

    return Z0, labels


def symmetrize_matrix(Z):
    n = Z.shape[0]
    Z_sym = Z.copy()
    conflicts = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = Z[i, j], Z[j, i]
            if a > EPS and b > EPS and abs(a - b) > EPS:
                conflicts.append((i, j, a, b))
                val = (a + b) / 2.0
            else:
                val = max(a, b)  # the populated cell; the other is 0
            Z_sym[i, j] = val
            Z_sym[j, i] = val
    if conflicts:
        warnings.warn(
            f"[DEMATEL] {len(conflicts)} pair(s) had BOTH Z[i,j] and "
            f"Z[j,i] populated with different non-zero values; averaged "
            f"when symmetrizing. Check the raw {INPUT_FILE} for these pairs."
        )
    np.fill_diagonal(Z_sym, 0.0)
    return Z_sym


def report_relationship_coverage(Z0_raw, labels):
    n = Z0_raw.shape[0]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            val = Z0_raw[i, j] if abs(Z0_raw[i, j]) > EPS else Z0_raw[j, i]
            rows.append({
                "feature_i": labels[i],
                "feature_j": labels[j],
                "recorded_value": val,
                "has_relationship": bool(abs(val) > EPS),
            })

    coverage_df = pd.DataFrame(rows)
    total_pairs = len(coverage_df)
    n_connected = int(coverage_df["has_relationship"].sum())
    n_unconnected = total_pairs - n_connected
    pct = 100.0 * n_connected / total_pairs if total_pairs else 0.0

    print(
        f"[INFO] Interrelationship coverage: {n_connected}/{total_pairs} "
        f"unordered feature pairs ({pct:.1f}%) carry a nonzero recorded "
        f"relationship; {n_unconnected} pair(s) have NO recorded "
        f"relationship ('tidak terhubung') and contribute 0 to Z0."
    )
    return coverage_df


# ------------------------------------------------------------------
# STEP 4: Stress perturbation (nonlinear, HETEROGENEOUS)
# ------------------------------------------------------------------

def compute_stress_sensitivity_from_Z0(Z0):
    row_sums = Z0.sum(axis=1)
    col_sums = Z0.sum(axis=0)
    return (row_sums - col_sums) / (row_sums + col_sums + EPS)


def compute_stress_sensitivity_from_T0(D0, R0):
    return (D0 - R0) / (D0 + R0 + EPS)


def compute_stress_sensitivity_from_centrality(Z0):
    s = Z0.sum(axis=1)  # == Z0.sum(axis=0) for symmetric Z0
    s_bar = s.mean()
    return (s - s_bar) / (s + s_bar + EPS)


def compute_stress_sensitivity(Z0=None, D0=None, R0=None,
                                source=STRESS_SENSITIVITY_SOURCE):
    if source == "T0":
        if D0 is None or R0 is None:
            raise ValueError(
                "source='T0' requires D0 and R0 (baseline total-relation "
                "row/col sums); pass them in or use source='Z0'."
            )
        return compute_stress_sensitivity_from_T0(D0, R0)
    elif source == "Z0":
        if Z0 is None:
            raise ValueError("source='Z0' requires Z0.")
        return compute_stress_sensitivity_from_Z0(Z0)
    elif source == "centrality":
        if Z0 is None:
            raise ValueError("source='centrality' requires Z0.")
        return compute_stress_sensitivity_from_centrality(Z0)
    else:
        raise ValueError(f"Unknown STRESS_SENSITIVITY_SOURCE: {source!r}")


def apply_stress(Z0, beta, c, mode=STRESS_MODE):
    n = Z0.shape[0]
    if mode == "row":
        warnings.warn(
            "[DEMATEL] STRESS_MODE='row' scales rows only and BREAKS "
            "matrix symmetry, silently reintroducing a fake direction "
            "into a matrix that was deliberately symmetrized. Use "
            "mode='cell_sym' (the default) unless have a specific "
            "reason not to."
        )
        factor = np.exp((beta / 100.0) * c).reshape(n, 1)
        Z_beta = Z0 * factor
    elif mode == "cell":
        warnings.warn(
            "[DEMATEL] STRESS_MODE='cell' uses (C_i - C_j), which is "
            "antisymmetric and BREAKS matrix symmetry by construction "
            "(it injects a synthetic direction). Use mode='cell_sym' "
            "(the default) to keep the perturbed matrix symmetric."
        )
        C_i = c.reshape(n, 1)
        C_j = c.reshape(1, n)
        Z_beta = Z0 * np.exp((beta / 100.0) * (C_i - C_j) / 2.0)
    elif mode == "cell_sym":
        C_i = c.reshape(n, 1)
        C_j = c.reshape(1, n)
        Z_beta = Z0 * np.exp((beta / 100.0) * (C_i + C_j) / 2.0)
    else:
        raise ValueError(
            f"Unknown STRESS_MODE: {mode!r} (expected 'cell_sym' "
            f"[recommended for symmetric Z], or legacy 'row'/'cell')"
        )

    np.fill_diagonal(Z_beta, 0.0)
    return Z_beta


# ------------------------------------------------------------------
# STEP 5: Normalization + total relation matrix (closed form)
# ------------------------------------------------------------------

def normalize_matrix(Z):
    row_sums = Z.sum(axis=1)
    col_sums = Z.sum(axis=0)
    s = max(row_sums.max(), col_sums.max())
    s = s if s > EPS else EPS
    X = Z / s
    return X, s


def spectral_radius(X):
    eigvals = np.linalg.eigvals(X)
    return np.max(np.abs(eigvals))


def total_relation_matrix(X):
    n = X.shape[0]
    rho = spectral_radius(X)
    if rho >= 1.0:
        warnings.warn(
            f"[DEMATEL] Spectral radius of X = {rho:.4f} >= 1: "
            f"(I - X) may be singular/ill-conditioned. Total relation "
            f"matrix may be unreliable; consider rescaling normalization."
        )
    I = np.eye(n)
    T = np.linalg.solve((I - X).T, X.T).T
    return T, rho


def compute_D_R(T):
    D = T.sum(axis=1)  # row sums (kept for symmetry sanity check only --
    R = T.sum(axis=0)  # col sums   NOT "dispatched"/"received" influence;
                       # Z0 is symmetric, so D == R for every feature.
    prominence = D + R  # interrelationship/connectedness strength of i
    relation = D - R    # ~0 by construction; symmetry check, NOT a
                        # cause(+)/effect(-) signal (see module header)
    return D, R, prominence, relation


# ------------------------------------------------------------------
# STEP 6: Stress trajectory (unamplified baseline matrix Z0)
# ------------------------------------------------------------------

def stress_trajectory(Z0, stress_set, c, mode=STRESS_MODE):
    D_traj, R_traj, prom_traj, rel_traj = {}, {}, {}, {}
    for beta in stress_set:
        Z_beta = apply_stress(Z0, beta, c, mode=mode)
        X_beta, _ = normalize_matrix(Z_beta)
        T_beta, _ = total_relation_matrix(X_beta)
        D, R, prom, rel = compute_D_R(T_beta)
        D_traj[beta] = D
        R_traj[beta] = R
        prom_traj[beta] = prom
        rel_traj[beta] = rel
    return D_traj, R_traj, prom_traj, rel_traj


# ------------------------------------------------------------------
# STEP 7: Crack detection on P & C trajectory (RMS combination) 
# ------------------------------------------------------------------

def compute_global_ranges(prom_traj, rel_traj):
    all_P = np.concatenate(list(prom_traj.values()))
    all_C = np.concatenate(list(rel_traj.values()))
    range_P = all_P.max() - all_P.min()
    range_C = all_C.max() - all_C.min()
    range_P = range_P if range_P > EPS else EPS
    range_C = range_C if range_C > EPS else EPS
    return range_P, range_C


def crack_detection(prom_traj, rel_traj, labels, tau_c=TAU_C_DEMATEL,
                     range_P=None, range_C=None):
    if range_P is None or range_C is None:
        range_P, range_C = compute_global_ranges(prom_traj, rel_traj)

    betas_sorted = sorted(prom_traj.keys())
    n = len(labels)

    crack_records = []
    B_star = []

    for k in range(len(betas_sorted) - 1):
        b_k, b_k1 = betas_sorted[k], betas_sorted[k + 1]
        p_k, p_k1 = prom_traj[b_k], prom_traj[b_k1]
        c_k, c_k1 = rel_traj[b_k], rel_traj[b_k1]

        cracked = False
        for p in range(n):
            dP = abs(p_k1[p] - p_k[p]) / range_P
            dC = abs(c_k1[p] - c_k[p]) / range_C  # diagnostic only; ~0 by construction
            delta_p = dP  # prominence-only (see docstring)
            is_crack = delta_p > tau_c

            crack_records.append({
                "beta_k": b_k,
                "beta_k1": b_k1,
                "feature": labels[p],
                "delta_P_component": dP,
                "delta_C_component_diagnostic_only": dC,
                "delta_D": delta_p,   # Delta_p^{(D)}, DEMATEL-specific, prominence-only
                "crack": is_crack,
            })

            if is_crack:
                cracked = True

        if not cracked:
            B_star.append(b_k)

    crack_table = pd.DataFrame(crack_records)
    return crack_table, sorted(set(B_star))


def compute_mean_crack_index(prom_traj, rel_traj, labels, range_P=None, range_C=None):
    """Mean Delta_bar_i^(D) -- prominence-only, see crack_detection()."""
    if range_P is None or range_C is None:
        range_P, range_C = compute_global_ranges(prom_traj, rel_traj)

    betas_sorted = sorted(prom_traj.keys())
    n = len(labels)
    n_transitions = len(betas_sorted) - 1

    delta_sum = np.zeros(n)
    for k in range(n_transitions):
        p_k, p_k1 = prom_traj[betas_sorted[k]], prom_traj[betas_sorted[k + 1]]
        dP = np.abs(p_k1 - p_k) / range_P
        delta_sum += dP  # Eq. (26): prominence-only

    delta_bar = delta_sum / max(n_transitions, 1)
    return delta_bar


def compute_fsi(delta_bar):
    return np.exp(-delta_bar)


# ------------------------------------------------------------------
# STEP 8: Stress amplification of Z0 using Delta_bar_p^{(D)}
# ------------------------------------------------------------------

def amplify_matrix(Z0, delta_bar, gamma=GAMMA_AMPLIFICATION):
    n = Z0.shape[0]
    factor = 1.0 + gamma * delta_bar
    factor = np.clip(factor, EPS, None)  # guard sqrt() against negative factors
    mult = np.sqrt(np.outer(factor, factor))
    Z_star = Z0 * mult
    np.fill_diagonal(Z_star, 0.0)
    return Z_star


# ------------------------------------------------------------------
# STEP 9: Gamma sensitivity sweep (calibration helper)
# ------------------------------------------------------------------

def gamma_sensitivity_sweep(Z0, delta_bar, labels, gamma_values=GAMMA_SWEEP_VALUES):
    records = []
    for gamma in gamma_values:
        Z_star = amplify_matrix(Z0, delta_bar, gamma=gamma)
        X_star, _ = normalize_matrix(Z_star)
        T_star, rho = total_relation_matrix(X_star)
        D_star, R_star, prom_star, rel_star = compute_D_R(T_star)

        for p, feat in enumerate(labels):
            records.append({
                "gamma": gamma,
                "feature": feat,
                "D_star": D_star[p],
                "R_star": R_star[p],
                "Prominence_star": prom_star[p],
                "Relation_star_sanity_check": rel_star[p],  # ~0 by construction
                "spectral_radius": rho,
            })

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# STEP 10: Unified score combining TOPSIS RC with DEMATEL Prominence*
# ------------------------------------------------------------------

def compute_unified_score(topsis_result_df, dematel_result_df,
                           topsis_id_col, dematel_id_col="Feature",
                           delta=DELTA_UNIFIED_SCORE):
    merged = topsis_result_df.merge(
        dematel_result_df, left_on=topsis_id_col, right_on=dematel_id_col,
        how="inner", suffixes=("_topsis", "_dematel")
    )

    if len(merged) != len(topsis_result_df):
        warnings.warn(
            f"[UNIFIED] Merge produced {len(merged)} rows, expected "
            f"{len(topsis_result_df)}. Check that feature IDs match "
            f"exactly between TOPSIS and DEMATEL outputs."
        )

    prom = merged["Prominence_star"]
    prom_min, prom_max = prom.min(), prom.max()
    prom_range = prom_max - prom_min
    prom_range = prom_range if prom_range > EPS else EPS
    merged["Prominence_norm"] = (prom - prom_min) / prom_range

    merged["Score"] = merged["RC"] * (1.0 + delta * merged["Prominence_norm"])
    merged["RC_Rank"] = merged["RC"].rank(ascending=False, method="min").astype(int)
    merged["Unified_Rank"] = merged["Score"].rank(ascending=False, method="min").astype(int)
    merged["Rank_Shift"] = merged["RC_Rank"] - merged["Unified_Rank"]
    merged = merged.sort_values("Score", ascending=False)

    return merged


# ------------------------------------------------------------------
# BENCHMARK: DEMATEL prominence vs. simple network centrality
# ------------------------------------------------------------------

def weighted_degree(Z0):
    """WD_i = sum_j z_ij on the symmetric baseline matrix Z0."""
    return Z0.sum(axis=1)


def _n_components(Z0):
    n = Z0.shape[0]
    seen, comps = set(), 0
    for s0 in range(n):
        if s0 in seen:
            continue
        comps += 1
        stack = [s0]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(int(v) for v in np.where(Z0[u] > EPS)[0] if v not in seen)
    return comps


def eigenvector_centrality(Z0):
    """Eigenvector centrality of the weighted undirected graph Z0 (symmetric, >= 0):
    principal eigenvector, sign-fixed to be non-negative, scaled to sum 1.
    Returns (centrality, lambda_1, spectral_gap, n_components)."""
    if not np.allclose(Z0, Z0.T, atol=1e-9):
        raise ValueError("[DEMATEL] Z0 must be symmetric for the centrality benchmark.")
    vals, vecs = np.linalg.eigh(Z0)           # ascending eigenvalues
    v = vecs[:, -1]
    if v.sum() < 0:
        v = -v
    if v.min() < -1e-8:
        warnings.warn("[DEMATEL] Leading eigenvector has negative entries; graph may be "
                      "disconnected (non-unique Perron vector). Interpret centrality with care.")
    v = np.clip(v, 0.0, None)
    v = v / v.sum() if v.sum() > EPS else v
    gap = float(vals[-1] - vals[-2]) if len(vals) > 1 else float("nan")
    comps = _n_components(Z0)
    if comps > 1:
        warnings.warn(f"[DEMATEL] Z0 graph has {comps} connected components; eigenvector "
                      f"centrality is concentrated on the dominant component.")
    return v, float(vals[-1]), gap, comps


def direct_indirect_decomposition(Z0):
    """Split baseline prominence D+R = (direct) + (indirect).
    direct   = rowsum(X)+colsum(X) = 2*WD/s0  (proportional to weighted degree)
    indirect = rowsum(T-X)+colsum(T-X)        (paths of length >= 2 through T = X + X^2 + ...)
    Returns (direct, indirect, total, s0)."""
    X0, s0 = normalize_matrix(Z0)
    T0, _ = total_relation_matrix(X0)
    direct = X0.sum(axis=1) + X0.sum(axis=0)
    total = T0.sum(axis=1) + T0.sum(axis=0)
    indirect = total - direct
    return direct, indirect, total, s0


def _rank_desc(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(ascending=False, method="min").astype(int).values


def _pair_stats(a_score, b_score, a_rank, b_rank):
    rho, _ = spearmanr(a_score, b_score)
    tau, _ = kendalltau(a_score, b_score)
    shifts = np.abs(np.asarray(a_rank) - np.asarray(b_rank))
    return float(rho), float(tau), int(shifts.max()), int((shifts >= CENTRALITY_RANK_SHIFT_FLAG).sum())


def path_length_prominence(Z0, k_max=CENTRALITY_PATH_K_MAX):
    """Counterfactual prominence using ONLY paths up to length k: D+R^(k) from
    S_k = X + X^2 + ... + X^k (k=1 is the direct part, proportional to weighted degree).
    Key 'inf' is the full DEMATEL total-relation prominence."""
    X0, _ = normalize_matrix(Z0)
    n = X0.shape[0]
    P, S, out = np.eye(n), np.zeros((n, n)), {}
    for k in range(1, k_max + 1):
        P = P @ X0
        S = S + P
        out[k] = S.sum(axis=1) + S.sum(axis=0)
    T0, _ = total_relation_matrix(X0)
    out["inf"] = T0.sum(axis=1) + T0.sum(axis=0)
    return out


def indirect_path_ablation(Z0, labels, r_wd, r_ec, r_dr):
    """Causal check : is a rank shift of D+R vs simple centrality CREATED by the
    indirect paths? Counterfactual = delete all paths of length >= 2 (k=1, rank == WD rank).
    For a flagged feature and the reference that triggered the flag (WD first, else EC):
        gap_direct = |rank(k=1) - rank(ref)|, gap_final = |rank(D+R) - rank(ref)|.
    The shift is attributed to indirect paths only if it disappears or shrinks when they are
    removed (gap_final >= flag AND gap_final > gap_direct). Also returns the smallest path length
    k whose truncated ranking already reproduces the final D+R rank of the feature."""
    pl = path_length_prominence(Z0)
    ranks = {k: _rank_desc(v) for k, v in pl.items()}
    rows = []
    for i, lab in enumerate(labels):
        sh_wd = abs(int(r_wd[i] - r_dr[i])); sh_ec = abs(int(r_ec[i] - r_dr[i]))
        flagged = max(sh_wd, sh_ec) >= CENTRALITY_RANK_SHIFT_FLAG
        row = {"Feature": lab}
        for k in range(1, CENTRALITY_PATH_K_MAX + 1):
            row[f"Rank_paths_len_le_{k}"] = int(ranks[k][i])
        row["Rank_paths_all"] = int(ranks["inf"][i])
        ks = [k for k in range(2, CENTRALITY_PATH_K_MAX + 1) if ranks[k][i] == ranks["inf"][i]
              and all(ranks[kk][i] == ranks["inf"][i] for kk in range(k, CENTRALITY_PATH_K_MAX + 1))]
        row["Min_path_length_reproducing_final_rank"] = (min(ks) if ks else
                                                         (1 if ranks[1][i] == ranks["inf"][i] else np.nan))
        if not flagged:
            row.update({"Reference_measure": "", "Gap_direct_only": np.nan, "Gap_final": np.nan,
                        "Indirect_path_ablation_verdict": "NO_SHIFT_TO_EXPLAIN"})
        else:
            if sh_wd >= CENTRALITY_RANK_SHIFT_FLAG:
                ref, r_ref = "weighted degree", r_wd[i]
            else:
                ref, r_ref = "eigenvector centrality", r_ec[i]
            gd = abs(int(ranks[1][i] - r_ref)); gf = abs(int(r_dr[i] - r_ref))
            caused = (gf >= CENTRALITY_RANK_SHIFT_FLAG) and (gf > gd)
            row.update({"Reference_measure": ref, "Gap_direct_only": gd, "Gap_final": gf,
                        "Indirect_path_ablation_verdict": ("CAUSED_BY_INDIRECT_PATHS" if caused
                                                           else "NOT_CAUSED_BY_INDIRECT_PATHS")})
        rows.append(row)
    return pd.DataFrame(rows)


def expert_bootstrap_stability(tfn_dfs, labels, B=CENTRALITY_BOOT_B, seed=CENTRALITY_BOOT_SEED):
    """Resample the experts with replacement, rebuild Z0, and recompute WD / EC / D+R ranks.
    Reports how often a flagged shift appears and the spread of rho -- i.e. whether the
    'nearly identical' (or 'different') finding is a property of the expert panel sample."""
    rng = np.random.default_rng(seed)
    mats = []
    for name, df in tfn_dfs.items():
        Zk, _ = build_crisp_matrix(enforce_zero_diagonal_fuzzy(df))
        mats.append(Zk)
    mats = np.stack(mats)
    K = mats.shape[0]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(B):
            idx = rng.integers(0, K, K)
            Zb = symmetrize_matrix(mats[idx].mean(axis=0))
            try:
                WD = weighted_degree(Zb)
                EC = eigenvector_centrality(Zb)[0]
                _, _, DR, _ = direct_indirect_decomposition(Zb)
            except Exception:
                continue
            rw, re_, rd = _rank_desc(WD), _rank_desc(EC), _rank_desc(DR)
            sh = max(np.abs(rw - rd).max(), np.abs(re_ - rd).max())
            rows.append({"rho_WD_DR": spearmanr(WD, DR)[0], "rho_EC_DR": spearmanr(EC, DR)[0],
                         "max_abs_shift": int(sh),
                         "n_flagged": int((np.maximum(np.abs(rw - rd), np.abs(re_ - rd))
                                           >= CENTRALITY_RANK_SHIFT_FLAG).sum())})
    d = pd.DataFrame(rows)
    return d


def summarize_bootstrap(d, K):
    if d.empty:
        return pd.DataFrame([{"Comparison": "Expert bootstrap stability", "Role": "robustness",
                              "Note": "bootstrap failed"}])
    return pd.DataFrame([{
        "Comparison": "Expert bootstrap stability", "Role": "robustness",
        "Spearman_rho": float(d["rho_WD_DR"].median()),
        "Max_abs_rank_shift": int(d["max_abs_shift"].max()),
        "Note": (f"{len(d)} resamples of the {K} experts (with replacement): median rho(WD, D+R)="
                 f"{d['rho_WD_DR'].median():.3f} (5th pct {d['rho_WD_DR'].quantile(0.05):.3f}); "
                 f"median rho(EC, D+R)={d['rho_EC_DR'].median():.3f}; share of resamples with at least one "
                 f"feature shifting >= {CENTRALITY_RANK_SHIFT_FLAG} ranks = "
                 f"{(d['n_flagged'] > 0).mean() * 100:.1f}%; max shift observed {int(d['max_abs_shift'].max())}.")}])


def build_centrality_benchmark(Z0, labels, prom_star=None):
    """Rank the nine features by weighted degree, eigenvector centrality and DEMATEL
    prominence D+R (all from the SAME baseline Z0). `prom_star` (optional) adds the
    stress-amplified prominence used downstream as a secondary comparison."""
    WD = weighted_degree(Z0)
    EC, lam1, gap, comps = eigenvector_centrality(Z0)
    direct, indirect, DR0, s0 = direct_indirect_decomposition(Z0)

    r_wd, r_ec, r_dr, r_ind = _rank_desc(WD), _rank_desc(EC), _rank_desc(DR0), _rank_desc(indirect)

    det = pd.DataFrame({
        "Feature": labels,
        "WeightedDegree": WD, "Rank_WeightedDegree": r_wd,
        "EigenvectorCentrality": EC, "Rank_EigenvectorCentrality": r_ec,
        "DplusR_baseline": DR0, "Rank_DplusR_baseline": r_dr,
        "Direct_part": direct, "Indirect_part": indirect,
        "Indirect_share": np.where(DR0 > EPS, indirect / np.where(DR0 > EPS, DR0, 1.0), np.nan),
        "Rank_IndirectPart": r_ind,
        # positive = feature ranks HIGHER under D+R than under the simpler measure
        "Rank_shift_WD_minus_DR": r_wd - r_dr,
        "Rank_shift_EC_minus_DR": r_ec - r_dr,
    })
    det["Max_abs_shift_vs_centrality"] = np.maximum(det["Rank_shift_WD_minus_DR"].abs(),
                                                    det["Rank_shift_EC_minus_DR"].abs())
    det["Flag_shift_ge_threshold"] = det["Max_abs_shift_vs_centrality"] >= CENTRALITY_RANK_SHIFT_FLAG

    # (a) heuristic kept for reference only: direction of the shift vs the indirect-part ranking.
    #     This is a CORRELATIONAL consistency check, NOT evidence of causation.
    heur, notes = [], []
    for k in range(len(labels)):
        sh_wd = int(det.loc[k, "Rank_shift_WD_minus_DR"])
        sh_ec = int(det.loc[k, "Rank_shift_EC_minus_DR"])
        if abs(sh_wd) < CENTRALITY_RANK_SHIFT_FLAG and abs(sh_ec) < CENTRALITY_RANK_SHIFT_FLAG:
            heur.append(np.nan)
            notes.append("rank essentially unchanged (|shift| < %d)" % CENTRALITY_RANK_SHIFT_FLAG)
            continue
        if abs(sh_wd) >= CENTRALITY_RANK_SHIFT_FLAG:
            ref_name, sh, r_ref = "weighted degree", sh_wd, r_wd[k]
        else:
            ref_name, sh, r_ref = "eigenvector centrality", sh_ec, r_ec[k]
        ok = (np.sign(sh) == np.sign(r_ref - r_ind[k])) and (r_ind[k] != r_ref)
        heur.append(bool(ok))
        direction = "higher" if sh > 0 else "lower"
        notes.append(f"D+R rank {direction} than {ref_name} by {abs(sh)}; indirect part rank {int(r_ind[k])} "
                     f"vs {ref_name} rank {int(r_ref)} (indirect share {det.loc[k,'Indirect_share']:.2f})")
    det["Heuristic_indirect_rank_consistent"] = heur

    # (b) path-length ablation = counterfactual test of the causal claim
    abl = indirect_path_ablation(Z0, labels, r_wd, r_ec, r_dr)
    for c in abl.columns:
        if c != "Feature":
            det[c] = abl[c].values
    det["Shift_explained_by_indirect_paths"] = det["Indirect_path_ablation_verdict"].map(
        {"CAUSED_BY_INDIRECT_PATHS": True, "NOT_CAUSED_BY_INDIRECT_PATHS": False})
    det["Interpretation"] = [
        (n_ + " -> ablation: " + v) for n_, v in zip(notes, det["Indirect_path_ablation_verdict"])]

    if prom_star is not None:
        r_star = _rank_desc(prom_star)
        det["DplusR_star_amplified"] = prom_star
        det["Rank_DplusR_star_amplified"] = r_star
        det["Rank_shift_WD_minus_DRstar"] = r_wd - r_star
        det["Rank_shift_EC_minus_DRstar"] = r_ec - r_star
        det["Flag_shift_star_ge_threshold"] = (det[["Rank_shift_WD_minus_DRstar",
                                                    "Rank_shift_EC_minus_DRstar"]].abs().max(axis=1)
                                               >= CENTRALITY_RANK_SHIFT_FLAG)
    det = det.sort_values("Rank_DplusR_baseline").reset_index(drop=True)

    # ---- pairwise agreement ----
    pairs = [
        ("WeightedDegree vs EigenvectorCentrality", WD, EC, r_wd, r_ec, "reference (both simple)"),
        ("WeightedDegree vs DEMATEL D+R (baseline)", WD, DR0, r_wd, r_dr, "primary"),
        ("EigenvectorCentrality vs DEMATEL D+R (baseline)", EC, DR0, r_ec, r_dr, "primary"),
    ]
    if prom_star is not None:
        pairs += [
            ("WeightedDegree vs DEMATEL D+R* (stress-amplified)", WD, prom_star, r_wd, r_star, "secondary"),
            ("EigenvectorCentrality vs DEMATEL D+R* (stress-amplified)", EC, prom_star, r_ec, r_star, "secondary"),
            ("DEMATEL D+R (baseline) vs D+R* (stress-amplified)", DR0, prom_star, r_dr, r_star, "secondary"),
        ]
    rows = []
    for name, a, b, ra, rb, role in pairs:
        rho, tau, mx, nfl = _pair_stats(a, b, ra, rb)
        almost = (rho >= CENTRALITY_RHO_MIN and tau >= CENTRALITY_TAU_MIN
                  and mx <= CENTRALITY_MAX_RANK_SHIFT)
        rows.append({
            "Comparison": name, "Role": role,
            "Spearman_rho": rho, "Kendall_tau": tau,
            "Max_abs_rank_shift": mx, "N_features_shift_ge_%d" % CENTRALITY_RANK_SHIFT_FLAG: nfl,
            "Pair_verdict": "NEARLY_IDENTICAL" if almost else "RANKING_CHANGES",
        })
    summ = pd.DataFrame(rows)

    prim = summ[summ["Role"] == "primary"]
    n_flag = int(det["Flag_shift_ge_threshold"].sum())
    n_expl = int((det["Indirect_path_ablation_verdict"] == "CAUSED_BY_INDIRECT_PATHS").sum())
    if (prim["Pair_verdict"] == "NEARLY_IDENTICAL").all():
        overall = "NEARLY_IDENTICAL"
        action = ("DEMATEL D+R reproduces the weighted-degree / eigenvector ordering: reduce the novelty "
                  "claim and describe DEMATEL as a connectedness transformation, not a distinctive "
                  "causal-analysis contribution (Section 3.4.2, 5.3, 6.1; Table 2 wording). No feature "
                  "shifts, so there is no shift for indirect paths to explain; see the expert-bootstrap row.")
    else:
        overall = "MEANINGFUL_CHANGE"
        action = (f"DEMATEL D+R changes the ranking relative to simple centrality ({n_flag} feature(s) "
                  f"shift >= {CENTRALITY_RANK_SHIFT_FLAG}; {n_expl} of them shown by path-length ablation to be "
                  f"CAUSED by indirect paths): describe only those as an indirect-path effect.")
    extra = pd.DataFrame([
        {"Comparison": "OVERALL (primary pairs)", "Role": "decision rule", "Pair_verdict": overall,
         "Note": action},
        {"Comparison": "Graph diagnostics", "Role": "info",
         "Note": f"lambda_1(Z0)={lam1:.4f}; spectral gap={gap:.4f}; connected components={comps}; "
                 f"normalization s0={s0:.4f}; thresholds rho>={CENTRALITY_RHO_MIN}, tau>={CENTRALITY_TAU_MIN}, "
                 f"max shift<={CENTRALITY_MAX_RANK_SHIFT}; flag |shift|>={CENTRALITY_RANK_SHIFT_FLAG}"},
    ])
    summ = pd.concat([summ, extra], ignore_index=True)
    return det, summ, overall, action


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------

def run_pipeline(input_file=INPUT_FILE, stress_set=STRESS_SET,
                  tau_c=TAU_C_DEMATEL, gamma=GAMMA_AMPLIFICATION,
                  stress_mode=STRESS_MODE,
                  sensitivity_source=STRESS_SENSITIVITY_SOURCE):

    # 1. Load + parse
    raw_dfs = load_expert_sheets(input_file)
    tfn_dfs = parse_all_sheets(raw_dfs)

    # 2. Aggregate across experts, enforce zero diagonal
    agg_fuzzy = aggregate_experts_arithmetic(tfn_dfs)
    agg_fuzzy = enforce_zero_diagonal_fuzzy(agg_fuzzy)

    # 3. Defuzzify -> baseline crisp Z0
    Z0_raw, labels = build_crisp_matrix(agg_fuzzy)
    n = len(labels)

    # 3b. Report how many feature pairs actually carry a recorded
    coverage_df = report_relationship_coverage(Z0_raw, labels)

    Z0 = symmetrize_matrix(Z0_raw)

    # 4. Baseline (beta=0) total relation, explicit (no arbitrary index pick)
    X0, s0 = normalize_matrix(Z0)
    T0, rho0 = total_relation_matrix(X0)
    D0, R0, prom0, rel0 = compute_D_R(T0)
    print(f"[INFO] Baseline (beta=0): normalization s0={s0:.4f}, "
          f"spectral radius rho0={rho0:.4f}")

    # 4b. Stress sensitivity 
    c0 = compute_stress_sensitivity(Z0=Z0, D0=D0, R0=R0, source=sensitivity_source)
    print(f"[INFO] Stress sensitivity c_i computed from source='{sensitivity_source}'.")

    # 5. Stress trajectory on the UNAMPLIFIED baseline matrix (heterogeneous)
    D_traj, R_traj, prom_traj, rel_traj = stress_trajectory(Z0, stress_set, c0, mode=stress_mode)

    # sanity check: trajectory must actually vary with beta now
    betas_sorted = sorted(prom_traj.keys())
    spread = np.max([np.max(np.abs(prom_traj[b] - prom_traj[betas_sorted[0]])) for b in betas_sorted])
    if spread < 1e-8:
        warnings.warn(
            "[DEMATEL] Prominence trajectory is (near-)constant across beta "
            "-- stress perturbation may be degenerate for this Z0 (e.g. all "
            "baseline net-roles c_i nearly equal). Check compute_stress_sensitivity()."
        )
    print(f"[INFO] Prominence trajectory spread across beta: {spread:.4f} (should be > 0)")

    range_P, range_C = compute_global_ranges(prom_traj, rel_traj)
    print(f"[INFO] Global ranges for dP/dC normalization: "
          f"range_P={range_P:.4f}, range_C={range_C:.4f}")

    # 6. Crack detection using RMS(P, C) 
    crack_table, B_star = crack_detection(prom_traj, rel_traj, labels, tau_c=tau_c,
                                           range_P=range_P, range_C=range_C)
    print(f"[INFO] Crack-free stress set B*_DEMATEL = {B_star} (diagnostic only)")

    # 7. Mean crack index (averaged over ALL transitions) 
    delta_bar = compute_mean_crack_index(prom_traj, rel_traj, labels,
                                          range_P=range_P, range_C=range_C)

    # 8. FSI derived directly from delta_bar (consistent with crack metric).
    fsi = compute_fsi(delta_bar)

    # 9. Amplify Z0 using Delta_bar_p^{(D)}, recompute final Prominence/Relation
    Z_star = amplify_matrix(Z0, delta_bar, gamma=gamma)
    X_star, s_star = normalize_matrix(Z_star)
    T_star, rho_star = total_relation_matrix(X_star)
    D_star, R_star, prom_star, rel_star = compute_D_R(T_star)

    # 9b. Gamma sensitivity sweep -- (invoked + saved)
    gamma_sweep_df = gamma_sensitivity_sweep(Z0, delta_bar, labels, gamma_values=GAMMA_SWEEP_VALUES)
    gamma_sweep_df.to_csv(_p("DEMATEL_gamma_sensitivity.csv"), index=False)
    print(f"[INFO] Gamma sensitivity sweep saved ({len(GAMMA_SWEEP_VALUES)} gamma values tested).")

    max_abs_relation = float(np.max(np.abs(rel_star)))
    if max_abs_relation > 1e-6:
        warnings.warn(
            f"[DEMATEL] max|Relation_star| = {max_abs_relation:.2e} is not "
            f"~0 -- symmetry was NOT preserved through the pipeline. This "
            f"should not happen with STRESS_MODE='cell_sym'; check for a "
            f"stray 'row'/'cell' mode call or an asymmetric Z0."
        )
    else:
        print(f"[INFO] Sanity check passed: max|Relation_star| = "
              f"{max_abs_relation:.2e} (~0, as expected under symmetrization).")

    results = pd.DataFrame({
        "Feature": labels,
        "D0_baseline": D0,
        "R0_baseline": R0,
        "Prominence0": prom0,
        "Relation0_sanity_check": rel0,   # ~0 by construction; not a Cause/Effect signal
        "Delta_bar_D": delta_bar,
        "FSI": fsi,
        "D_star": D_star,
        "R_star": R_star,
        "Prominence_star": prom_star,
        "Relation_star_sanity_check": rel_star,  # ~0 by construction; not a Cause/Effect signal
    })
    results["Rank_Prominence"] = results["Prominence_star"].rank(
        ascending=False, method="min"
    ).astype(int)
    results = results.sort_values("Prominence_star", ascending=False)

    # 11. Trajectory tables
    prom_traj_df = pd.DataFrame(
        {beta: prom_traj[beta] for beta in sorted(prom_traj)}, index=labels
    ).T
    prom_traj_df.index.name = "beta"

    # ------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------
    results.to_csv(_p("STRESS_DEMATEL_RESULTS.csv"), index=False)
    coverage_df.to_csv(_p("DEMATEL_relationship_coverage.csv"), index=False)
    pd.DataFrame(Z0, index=labels, columns=labels).to_csv(_p("DEMATEL_Z0_matrix.csv"))
    pd.DataFrame(Z_star, index=labels, columns=labels).to_csv(_p("DEMATEL_Zstar_matrix.csv"))
    pd.DataFrame(T_star, index=labels, columns=labels).to_csv(_p("DEMATEL_Tstar_matrix.csv"))
    prom_traj_df.to_csv(_p("prominence_trajectory.csv"))
    crack_table.to_csv(_p("crack_table_dematel.csv"), index=False)

    print("\n===== STRESS-AWARE DEMATEL RESULTS (undirected interrelationship / connectedness index) =====\n")
    print(results[["Feature", "Prominence_star", "FSI", "Rank_Prominence"]].to_string(index=False))

    # 12. benchmark vs. weighted degree / eigenvector centrality (same Z0)
    print("\n===== (2.4) DEMATEL PROMINENCE vs NETWORK CENTRALITY BENCHMARK =====\n")
    bench_df, bench_summary, bench_verdict, bench_action = build_centrality_benchmark(
        Z0, labels, prom_star=prom_star)
    boot_df = expert_bootstrap_stability(tfn_dfs, labels)
    boot_df.to_csv(OUTPUT_CENTRALITY_BOOT_FILE, index=False)
    bench_summary = pd.concat([bench_summary, summarize_bootstrap(boot_df, len(tfn_dfs))], ignore_index=True)
    abl_cols = ["Feature"] + [c for c in bench_df.columns if c.startswith("Rank_paths_")] + [
        "Min_path_length_reproducing_final_rank", "Reference_measure", "Gap_direct_only", "Gap_final",
        "Indirect_path_ablation_verdict"]
    bench_df[abl_cols].to_csv(OUTPUT_CENTRALITY_ABLATION_FILE, index=False)
    bench_df.to_csv(OUTPUT_CENTRALITY_FILE, index=False)
    bench_summary.to_csv(OUTPUT_CENTRALITY_SUMMARY_FILE, index=False)
    print(bench_df[["Feature", "Rank_WeightedDegree", "Rank_EigenvectorCentrality",
                    "Rank_DplusR_baseline", "Indirect_share", "Flag_shift_ge_threshold"]].to_string(index=False))
    print(bench_summary[bench_summary["Role"].isin(["primary", "secondary", "reference (both simple)"])][
        ["Comparison", "Spearman_rho", "Kendall_tau", "Max_abs_rank_shift", "Pair_verdict"]].to_string(index=False))
    print(bench_summary[bench_summary["Comparison"] == "Expert bootstrap stability"]["Note"].to_string(index=False))
    print(f"\n[VERDICT] {bench_verdict}")
    print(f"[DECISION RULE] {bench_action}")
    print(f"[INFO] Saved benchmark to {OUTPUT_CENTRALITY_FILE}")

    return {
        "results": results,
        "Z0": Z0,
        "Z_star": Z_star,
        "T_star": T_star,
        "prominence_trajectory": prom_traj_df,
        "crack_table": crack_table,
        "B_star": B_star,
        "delta_bar": delta_bar,
        "gamma_sweep": gamma_sweep_df,
        "relationship_coverage": coverage_df,
        "labels": labels,
        # ---- added in this revision (2.4) ----
        "centrality_benchmark": bench_df,
        "centrality_benchmark_summary": bench_summary,
        "centrality_benchmark_verdict": bench_verdict,
    }


if __name__ == "__main__":
    run_pipeline()
