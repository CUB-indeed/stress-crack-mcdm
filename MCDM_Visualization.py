
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr, kendalltau

PROJECT_DIR = os.environ.get("MCDM_PROJECT_DIR", "/home/coder/project")
FIG_DIR = os.path.join(PROJECT_DIR, "figures_v3")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.titlesize": 12,
    "axes.titlepad": 10,
})

PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#000000", "#999999", "#8B008B", "#7B3F00",
]

TEST_LABELS = {
    "weight_sensitivity": "Criterion-weight ±5%",
    "gamma_sensitivity": "Stress amplification (gamma)",
    "leave_one_expert_out": "Leave-one-expert-out",
    "comparative_variant": "Comparative methods",
    "benefit_cost_allocation_sensitivity": "Outcome/resource allocation",
    "crisp_fuzzy_ablation": "Methodological configuration",
    "lambda_mu_grid_sensitivity": "Fusion parameters (lambda, mu)",
}

TEST_ORDER = list(TEST_LABELS.keys())

def _color_cycle(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]

def _read(name, **kw):
    return pd.read_csv(os.path.join(PROJECT_DIR, name), **kw)

def _run_context():
    """PRIMARY_WEIGHTING / REVERSE_CODED_CRITERIA / polarity status as written by MCDM_Validation."""
    ctx = {"primary": "75_25", "reverse": "(none)", "verified": "False"}
    try:
        g = _read("VALIDATION_PARAMETER_GRID.csv").set_index("parameter")["value"]
        ctx["primary"] = str(g.get("PRIMARY_WEIGHTING", ctx["primary"]))
        ctx["reverse"] = str(g.get("REVERSE_CODED_CRITERIA", ctx["reverse"]))
        ctx["verified"] = str(g.get("POLARITY_VERIFIED_AGAINST_QUESTIONNAIRE", ctx["verified"]))
    except Exception:
        pass
    return ctx

def _stamp(fig):
    c = _run_context()
    txt = (f"Primary TOPSIS weighting: {c['primary']}  |  reverse-coded: {c['reverse']}  |  "
           f"polarity verified vs questionnaire: {c['verified']}")
    fig.text(0.995, 0.002, txt, ha="right", va="bottom", fontsize=6.5, color="#777777")

def _save(fig, name, stamp=False):
    if stamp:
        _stamp(fig)
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png_path}")

def _rho_tau(a, b):
    r, _ = spearmanr(a, b)
    t, _ = kendalltau(a, b)
    return float(r), float(t)

def _verdict_short(v):
    return str(v).split(":")[0].strip()

def _bar_value_labels(ax, bars, fmt="{:.2f}", fontsize=7.5, dy=0.012, upper_err=None):
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, b in enumerate(bars):
        h = b.get_height()
        if np.isnan(h):
            continue
        top = h + (upper_err[i] if upper_err is not None else 0.0)
        ax.text(
            b.get_x() + b.get_width() / 2, top + dy * y_span,
            fmt.format(h), ha="center", va="bottom",
            fontsize=fontsize, color="#333333",
        )

def fig02_weight_trajectory_and_threshold_sensitivity():
    traj_path = os.path.join(PROJECT_DIR, "weight_trajectory.csv")
    crackdiag_path = os.path.join(PROJECT_DIR, "VALIDATION_CRACK_THRESHOLD_DIAGNOSTIC.csv")

    traj = pd.read_csv(traj_path, index_col=0)
    betas = traj.index.astype(float).values
    criteria = list(traj.columns)
    colors = _color_cycle(len(criteria))

    fig, (ax_traj, ax_thresh) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for c, color in zip(criteria, colors):
        ax_traj.plot(betas, traj[c].values, marker="o", markersize=4,
                     linewidth=1.6, color=color, label=c)
    ax_traj.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_traj.set_xlabel(r"Stress level $\beta$")
    ax_traj.set_ylabel("Normalized criterion weight")
    ax_traj.set_title("(a) Stress-Eigen Weight Trajectories")
    ax_traj.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5,
                    borderaxespad=0.0)

    if os.path.exists(crackdiag_path):
        diag = pd.read_csv(crackdiag_path)
        n_criteria = diag["criterion"].nunique()

        per_thresh = (
            diag.groupby("kappa_threshold")
            .agg(
                n_criteria_flagged=("criterion_ever_flagged_cracked", "sum"),
                n_crack_events=("n_crack_events_total_at_this_threshold", "first"),
            )
            .reset_index()
            .sort_values("kappa_threshold")
        )

        crack_table_path = os.path.join(PROJECT_DIR, "crack_table_fahp.csv")
        if os.path.exists(crack_table_path):
            ctab = pd.read_csv(crack_table_path)
            n_transitions = ctab.groupby(["beta_k", "beta_k1"]).ngroups
            max_events = n_criteria * n_transitions
        else:
            max_events = int(per_thresh["n_crack_events"].max())

        x_labels = per_thresh["kappa_threshold"].astype(str)
        bars = ax_thresh.bar(
            x_labels, per_thresh["n_crack_events"],
            width=0.6, color=PALETTE[1], edgecolor="white", linewidth=0.6,
        )
        _bar_value_labels(ax_thresh, bars, fmt="{:.0f}")

        ax_thresh.axhline(max_events, color="grey", linewidth=0.8,
                           linestyle="--", alpha=0.6,
                           label=f"Max possible ({n_criteria} criteria "
                                 f"x transitions = {max_events})")
        ax_thresh.set_xlabel(r"Crack threshold $\kappa_{\mathrm{threshold}}$")
        ax_thresh.set_ylabel("Total crack events flagged\n(summed over all criteria & $\\beta$-transitions)")
        ax_thresh.set_ylim(0, max_events * 1.22)
        ax_thresh.set_title("(b) Crack-Event Sensitivity to Threshold")
        ax_thresh.legend(loc="upper right", fontsize=7.5)

        n_flagged_const = per_thresh["n_criteria_flagged"].iloc[0]
        same_everywhere = per_thresh["n_criteria_flagged"].nunique() == 1
        if same_everywhere:
            note = (
                f"Note: {int(n_flagged_const)}/{n_criteria} criteria are \"ever flagged\"\n"
                f"at every threshold shown (0.1-0.5) — at least one\n"
                f"$\\beta$-transition exceeds even 0.5 for every criterion.\n"
                f"Event *count* still declines, shown above."
            )
            ax_thresh.text(
                0.02, 0.02, note, transform=ax_thresh.transAxes,
                ha="left", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFF7E6", ec="#CC9900", alpha=0.95),
            )
    else:
        ax_thresh.text(0.5, 0.5, "VALIDATION_CRACK_THRESHOLD_DIAGNOSTIC.csv not found\n(run MCDM_Validation.py first)",
                        ha="center", va="center", transform=ax_thresh.transAxes, fontsize=9, color="#888888")
        ax_thresh.axis("off")

    fig.tight_layout()
    _save(fig, "fig02_weight_trajectory_and_threshold_sensitivity")

def fig03_ahp_permutation_invariance():
    path = os.path.join(PROJECT_DIR, "VALIDATION_PERMUTATION_INVARIANCE.csv")
    if not os.path.exists(path):
        warnings.warn(f"[FIG03] {path} not found; run MCDM_Validation.py first. Skipping.")
        return

    df = pd.read_csv(path)
    n_trials = len(df)
    frac_invariant = df["invariant_within_tolerance"].mean() * 100.0

    scale = 1e-16
    vals_scaled = df["max_abs_dev"].values / scale

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(vals_scaled, bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"Max $|\Delta w^*|$ across the 11 criteria, original vs. permuted labeling"
                  r"  ($\times 10^{-16}$)")
    ax.set_ylabel(f"Number of random permutations (of {n_trials} trials)")
    ax.set_title("AHP Stress-Transform Invariance to Criterion Ordering")

    ax.text(
        0.98, 0.95,
        f"{frac_invariant:.1f}% of trials invariant\n(tolerance = 1e-6)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", alpha=0.9),
    )

    fig.tight_layout()
    _save(fig, "fig03_ahp_permutation_invariance")

def fig04_dematel_connectedness():
    path = os.path.join(PROJECT_DIR, "STRESS_DEMATEL_RESULTS.csv")
    df = pd.read_csv(path).sort_values("Prominence_star", ascending=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    colors = _color_cycle(len(df))

    bars = ax.barh(df["Feature"], df["Prominence_star"], color=colors,
                    edgecolor="black", linewidth=0.5, zorder=3)
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0] if ax.get_xlim()[1] else 1.0
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.012 * x_span, b.get_y() + b.get_height() / 2,
                f"{w:.1f}", ha="left", va="center", fontsize=8, color="#333333")

    ax.set_xlabel(r"Prominence  $D^* + R^*$  (connectedness / co-influence)")
    ax.set_title("DEMATEL Connectedness Ranking (undirected)\n"
                  "Direction not reported: instrument elicits one magnitude "
                  "per pair; no reliable cause/effect signal")

    fig.tight_layout()
    _save(fig, "fig04_dematel_connectedness")

def fig05_prominence_trajectory():
    import MCDM_Dematel as stress_dematel

    dfd_path = os.path.join(PROJECT_DIR, "dfd.xlsx")

    raw_dfs = stress_dematel.load_expert_sheets(dfd_path)
    tfn_dfs = stress_dematel.parse_all_sheets(raw_dfs)
    agg_fuzzy = stress_dematel.aggregate_experts_arithmetic(tfn_dfs)
    agg_fuzzy = stress_dematel.enforce_zero_diagonal_fuzzy(agg_fuzzy)
    Z0_raw, labels = stress_dematel.build_crisp_matrix(agg_fuzzy)
    Z0 = stress_dematel.symmetrize_matrix(Z0_raw)

    X0, _s0 = stress_dematel.normalize_matrix(Z0)
    T0, _rho0 = stress_dematel.total_relation_matrix(X0)
    D0, R0, _prom0, _rel0 = stress_dematel.compute_D_R(T0)

    c0 = stress_dematel.compute_stress_sensitivity(
        Z0=Z0, D0=D0, R0=R0, source=stress_dematel.STRESS_SENSITIVITY_SOURCE
    )

    D_traj, R_traj, prom_traj, _rel_traj = stress_dematel.stress_trajectory(
        Z0, stress_dematel.STRESS_SET, c0, mode=stress_dematel.STRESS_MODE
    )

    betas = np.array(sorted(D_traj.keys()), dtype=float)
    P_df = pd.DataFrame({b: prom_traj[b] for b in betas}, index=labels).T

    features = list(labels)
    colors = _color_cycle(len(features))

    fig, (ax_P, ax_legend) = plt.subplots(1, 2, figsize=(9.5, 4.5),
                                           gridspec_kw={"width_ratios": [2.2, 1]})
    for f, color in zip(features, colors):
        ax_P.plot(betas, P_df[f].values, marker="o", markersize=4,
                  linewidth=1.6, color=color, label=f)
    ax_P.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_P.set_ylabel(r"Prominence  $D^* + R^*$  (connectedness)")
    ax_P.set_xlabel(r"Stress level $\beta$")

    ax_legend.axis("off")
    handles, labels_ = ax_P.get_legend_handles_labels()
    ax_legend.legend(handles, labels_, loc="center", fontsize=9, frameon=False)

    fig.suptitle("DEMATEL Prominence (Connectedness) Trajectory\n"
                  "Undirected: D=R by construction, no cause/effect claim")
    fig.tight_layout()
    _save(fig, "fig05_prominence_trajectory")

def fig06_primary_vs_exploratory_ranking():
    path = os.path.join(PROJECT_DIR, "Final_Stress_Coupled_Result.csv")
    df = pd.read_csv(path)

    df = df.copy()
    df["Rank_RC"] = df["Primary_Rank"]
    df["Rank_SRPI"] = df["SRPI_Rank"]
    has_minmax = "SRPI_minmax_Rank" in df.columns
    if has_minmax:
        df["Rank_SRPI_mm"] = df["SRPI_minmax_Rank"]

    df = df.sort_values("Rank_RC").reset_index(drop=True)

    recon = df[["Feature", "Rank_RC", "Rank_SRPI"]].copy()
    recon["rank_shift"] = recon["Rank_RC"] - recon["Rank_SRPI"]
    recon.to_csv(os.path.join(PROJECT_DIR, "FIG_RANK_RECONCILIATION_USED.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    colors = _color_cycle(len(df))
    ys = np.arange(len(df))
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        r_rc, r_srpi = row["Rank_RC"], row["Rank_SRPI"]
        ax.plot([r_rc, r_srpi], [i, i],
                color=colors[i], linewidth=1.6, alpha=0.85, zorder=2)
        ax.scatter(r_rc, i, color=colors[i], marker="o", s=60, zorder=3)
        ax.scatter(r_srpi, i, color=colors[i], marker="s", s=55,
                   facecolors="none", edgecolors=colors[i], linewidths=1.6, zorder=3)
        if has_minmax:
            ax.scatter(row["Rank_SRPI_mm"], i, color=colors[i], marker="^", s=45,
                       facecolors="none", edgecolors=colors[i], linewidths=1.1,
                       alpha=0.7, zorder=3)

        shift = int(r_rc - r_srpi)
        if shift != 0:
            mid_x = (r_rc + r_srpi) / 2.0
            ax.text(mid_x, i + 0.22, f"{shift:+d}", ha="center", va="bottom",
                    fontsize=7.5, color="#555555")

    ax.set_yticks(ys)
    ax.set_yticklabels(df["Feature"])
    ax.set_xlabel("Rank (1 = top)")
    ax.set_xlim(n + 1.4, -0.4)
    ax.set_title("Primary and Exploratory Rankings")

    ax.scatter([], [], color="grey", marker="o", s=60, label="PRIMARY: rank by RC (fuzzy TOPSIS)")
    ax.scatter([], [], color="grey", marker="s", s=55, facecolors="none",
               edgecolors="grey", linewidths=1.6, label="EXPLORATORY: rank by SRPI (absolute V)")
    if has_minmax:
        ax.scatter([], [], color="grey", marker="^", s=45, facecolors="none",
                   edgecolors="grey", linewidths=1.1, label="SENSITIVITY: rank by SRPI (min-max V)")
    ax.plot([], [], color="#555555", linewidth=0, marker="$\\pm n$", markersize=9,
            label="rank shift (RC $-$ SRPI absolute V)")
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8,
              borderaxespad=0.0)

    fig.tight_layout()
    _save(fig, "fig06_primary_vs_exploratory_ranking", stamp=True)

def fig07_robustness_summary():
    path = os.path.join(PROJECT_DIR, "VALIDATION_ROBUSTNESS_SUMMARY.csv")
    agg = pd.read_csv(path, header=[0, 1], index_col=0)

    order = [t for t in TEST_ORDER if t in agg.index] + [t for t in agg.index if t not in TEST_ORDER]
    agg = agg.loc[order]

    tests = [TEST_LABELS.get(t, t) for t in agg.index]
    rho_mean = agg[("spearman_rho", "mean")].values
    rho_min = agg[("spearman_rho", "min")].values
    rho_max = agg[("spearman_rho", "max")].values
    tau_mean = agg[("kendall_tau", "mean")].values
    tau_min = agg[("kendall_tau", "min")].values
    tau_max = agg[("kendall_tau", "max")].values

    rho_err = np.vstack([np.clip(rho_mean - rho_min, 0, None), np.clip(rho_max - rho_mean, 0, None)])
    tau_err = np.vstack([np.clip(tau_mean - tau_min, 0, None), np.clip(tau_max - tau_mean, 0, None)])

    x = np.arange(len(tests))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.8))
    b1 = ax.bar(x - width / 2, rho_mean, width, yerr=rho_err, capsize=4,
                color=PALETTE[0], label=r"Spearman $\rho$ (mean, min-max range)")
    b2 = ax.bar(x + width / 2, tau_mean, width, yerr=tau_err, capsize=4,
                color=PALETTE[1], label=r"Kendall $\tau$ (mean, min-max range)")

    ax.set_ylim(0, 1.32)
    _bar_value_labels(ax, b1, fmt="{:.2f}", dy=0.025, upper_err=rho_err[1])
    _bar_value_labels(ax, b2, fmt="{:.2f}", dy=0.025, upper_err=tau_err[1])

    ax.set_xticks(x)
    ax.set_xticklabels(tests, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Rank correlation vs. baseline EXPLORATORY SRPI ranking")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Robustness Summary Across Seven Validation Experiments\n"
                 "(implementation-equivalence runs excluded from means and ranges)")
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8,
              borderaxespad=0.0)

    fig.tight_layout()
    _save(fig, "fig07_robustness_summary", stamp=True)

def fig08_comparative_variants():
    path = os.path.join(PROJECT_DIR, "VALIDATION_ROBUSTNESS_RESULTS.csv")
    df = pd.read_csv(path)
    sub = df[df["test"] == "comparative_variant"].copy()

    if sub.empty:
        warnings.warn("[FIG08] No comparative_variant rows found; skipping.")
        return

    if "is_identity_run" in sub.columns:
        sub["is_identity_run"] = sub["is_identity_run"].astype(bool)
    else:
        sub["is_identity_run"] = sub["variant"].astype(str).str.contains("w0_explicit")

    sub["order_key"] = sub["is_identity_run"].astype(int)
    sub = sub.sort_values(["order_key", "spearman_rho"]).reset_index(drop=True)
    x = np.arange(len(sub))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.8))
    b1 = ax.bar(x - width / 2, sub["spearman_rho"], width,
                color=PALETTE[0], label=r"Spearman $\rho$")
    b2 = ax.bar(x + width / 2, sub["kendall_tau"], width,
                color=PALETTE[1], label=r"Kendall $\tau$")

    for i, is_id in enumerate(sub["is_identity_run"]):
        if is_id:
            b1[i].set_hatch("//")
            b2[i].set_hatch("//")
            b1[i].set_alpha(0.55)
            b2[i].set_alpha(0.55)

    ax.set_ylim(0, 1.32)
    _bar_value_labels(ax, b1, fmt="{:.2f}", dy=0.03)
    _bar_value_labels(ax, b2, fmt="{:.2f}", dy=0.03)

    ax.set_xticks(x)
    display_labels = {
        "standard_crisp_TOPSIS": "Standard crisp TOPSIS",
        "FAHP-DEMATEL_no_stress_amplification": r"FAHP–DEMATEL, $\gamma = 0$",
        "FAHP-TOPSIS_w0_explicit (== primary under AHP)": r"Explicit $w_0$"
    }
    ax.set_xticklabels(
        [display_labels.get(v, v) for v in sub["variant"]],
        rotation=20,
        ha="right",
        fontsize=7.5
    )
    
    ax.set_ylabel("Agreement with baseline EXPLORATORY SRPI ranking")
    ax.set_title("Comparison of Method Variants")
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8,
              borderaxespad=0.0)

    comp = sub[~sub["is_identity_run"]]
    if len(comp):
        ax.text(0.01, 0.97,
                f"Comparative-method mean (n={len(comp)}):\n"
                f"$\\rho$ = {comp['spearman_rho'].mean():.3f}, $\\tau$ = {comp['kendall_tau'].mean():.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.95))

    for i, is_id in enumerate(sub["is_identity_run"]):
        if is_id:
            top = max(sub["spearman_rho"].iloc[i], sub["kendall_tau"].iloc[i])
            ax.annotate("Reference run",
                        (x[i], top), textcoords="offset points", xytext=(0, 22),
                        ha="center", fontsize=7, color="#B22222")

    fig.tight_layout()
    _save(fig, "fig08_comparative_variants", stamp=True)

def fig09_monte_carlo_ahp_benchmark():
    path = os.path.join(PROJECT_DIR, "VALIDATION_MONTE_CARLO_AHP.csv")
    if not os.path.exists(path):
        warnings.warn(f"[FIG09] {path} not found; run MCDM_Validation.py first. Skipping.")
        return

    df = pd.read_csv(path)
    df = df[~df["criterion"].isin(
        ["__OVERALL_RANK_STABILITY_KENDALL_TAU__", "__CR_DIAGNOSTIC__"]
    )].copy()
    df = df.sort_values("w0_baseline", ascending=True)

    y = np.arange(len(df))
    lower_err = (df["mc_mean"] - df["mc_p5"]).clip(lower=0)
    upper_err = (df["mc_p95"] - df["mc_mean"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.errorbar(df["mc_mean"], y, xerr=[lower_err, upper_err],
                fmt="o", color=PALETTE[1], ecolor=PALETTE[1], elinewidth=1.4,
                capsize=3, markersize=5, label="Monte-Carlo mean (p5–p95 envelope)")
    ax.scatter(df["w0_baseline"], y, marker="D", s=45, color=PALETTE[0],
               zorder=5, label=r"Baseline eigenweight $w_0$")

    ax.set_yticks(y)
    ax.set_yticklabels(df["criterion"])
    ax.set_ylim(-1.0, len(df))
    ax.set_xlabel("Criterion weight")
    ax.set_title(r"Baseline $w_0$ vs. Monte-Carlo Judgment-Perturbation Envelope"
                 "\n(external robustness benchmark)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              borderaxespad=0.0)

    tau_row = pd.read_csv(path)
    tau_row = tau_row[tau_row["criterion"] == "__OVERALL_RANK_STABILITY_KENDALL_TAU__"]
    if not tau_row.empty:
        tau_mean = tau_row["mc_mean"].iloc[0]
        ax.text(0.02, 0.97,
                f"Criterion-weight ordering stability:\nmean Kendall $\\tau$ = {tau_mean:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", alpha=0.95))

    cr_row = pd.read_csv(path)
    cr_row = cr_row[cr_row["criterion"] == "__CR_DIAGNOSTIC__"]
    if not cr_row.empty:
        n_draws = int(cr_row["n_draws"].iloc[0])
        n_invalid = int(cr_row["n_cr_invalid_draws"].iloc[0])
        invalid_rate = cr_row["cr_invalid_rate"].iloc[0]
        cr_threshold = cr_row["cr_threshold"].iloc[0]
        mean_cr = cr_row["mc_mean"].iloc[0]
        ax.text(0.98, 0.03,
                f"CR-invalid draws (CR > {cr_threshold:.2f}):\n"
                f"{n_invalid}/{n_draws} ({invalid_rate:.2%}); mean CR = {mean_cr:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", alpha=0.95))

    fig.tight_layout()
    _save(fig, "fig09_monte_carlo_ahp_benchmark")

_ALLOC_ORDER = ["50/50", "60/40", "70/30", "75/25", "80/20", "raw_FAHP"]

def _alloc_label(s, q_o=None):
    if s == "raw_FAHP":
        return "raw $w^*$\n" + (f"({q_o*100:.0f}/{(1-q_o)*100:.0f})" if q_o is not None else "(no Eq. 17)")
    return s

def _alloc_panel(ax, df, title, ylabel, q_o):
    df = df.copy()
    df["_o"] = df["Scenario"].map({s: i for i, s in enumerate(_ALLOC_ORDER)})
    df = df.sort_values("_o").reset_index(drop=True)
    x = np.arange(len(df)); width = 0.36
    b1 = ax.bar(x - width/2, df["rho"], width, color=PALETTE[0], label=r"Spearman $\rho$", zorder=3)
    b2 = ax.bar(x + width/2, df["tau"], width, color=PALETTE[1], label=r"Kendall $\tau$", zorder=3)
    for i, raw in enumerate(df["Scenario"] == "raw_FAHP"):
        if raw:
            for bb in (b1[i], b2[i]):
                bb.set_hatch("//"); bb.set_edgecolor("white")
    ax.set_ylim(0, 1.32)
    _bar_value_labels(ax, b1, fmt="{:.2f}", dy=0.02, fontsize=7)
    _bar_value_labels(ax, b2, fmt="{:.2f}", dy=0.02, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([_alloc_label(s, q_o) for s in df["Scenario"]], fontsize=8)
    ax.set_xlabel("Outcome / resource allocation (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    for i, p in enumerate(df["primary"]):
        if bool(p):
            ax.annotate("primary", (x[i], max(df["rho"].iloc[i], df["tau"].iloc[i])),
                        textcoords="offset points", xytext=(0, 20), ha="center",
                        fontsize=8, color=PALETTE[0])

def fig10_benefit_cost_allocation_sensitivity():
    """Compare allocation scenarios for TOPSIS and SRPI rankings."""
    q_O = None
    try:
        q_O = float(_read("STRESS_TOPSIS_endogenous_shares.csv").iloc[0]["q_endogenous_from_w_star"])
    except Exception:
        pass

    panels = []
    try:
        t = _read("STRESS_TOPSIS_allocation_sensitivity.csv")
        g = t.groupby("Scenario", sort=False).first().reset_index()
        panels.append((pd.DataFrame({"Scenario": g["Scenario"],
                                     "rho": g["Spearman_rho_vs_primary"],
                                     "tau": g["Kendall_tau_vs_primary"],
                                     "primary": g["Is_primary"].astype(bool)}),
                       "(a) TOPSIS ranking ($RC$) vs primary", "Agreement with primary TOPSIS ranking"))
    except FileNotFoundError:
        warnings.warn("[FIG10] STRESS_TOPSIS_allocation_sensitivity.csv not found; panel (a) skipped.")

    v = _read("VALIDATION_ROBUSTNESS_RESULTS.csv")
    v = v[v["test"] == "benefit_cost_allocation_sensitivity"].copy()
    if not v.empty:
        if "scenario" not in v.columns or v["scenario"].isna().all():
            v["scenario"] = v.apply(lambda r: f"{int(round(r['benefit_share']*100))}/{int(round(r['cost_share']*100))}", axis=1)
        panels.append((pd.DataFrame({"Scenario": v["scenario"], "rho": v["spearman_rho"],
                                     "tau": v["kendall_tau"],
                                     "primary": v["is_primary_baseline"].astype(bool)}),
                       "(b) SRPI ranking (Validation Test 8) vs primary", "Agreement with primary SRPI ranking"))
    if not panels:
        warnings.warn("[FIG10] no allocation data found; skipping."); return

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 5.6), squeeze=False)
    for ax, (d, ttl, yl) in zip(axes[0], panels):
        _alloc_panel(ax, d, ttl, yl, q_O)
    axes[0][0].legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("Sensitivity to Outcome/Resource Weight Allocation", y=1.02)
    fig.tight_layout()
    _save(fig, "fig10_benefit_cost_allocation_sensitivity", stamp=True)

def fig11_raw_vs_7525():
    """Compare raw FAHP and 75/25 TOPSIS rankings and group shares."""
    df = _read("STRESS_TOPSIS_raw_vs_7525.csv").sort_values("Rank_75_25", ascending=False)
    try:
        sh = _read("STRESS_TOPSIS_endogenous_shares.csv")
    except FileNotFoundError:
        sh = None
    rho, tau = _rho_tau(df["Rank_raw_FAHP"], df["Rank_75_25"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [2.3, 1]})
    y = np.arange(len(df)); h = 0.38
    ax.barh(y + h/2, df["RC_75_25"], h, color=PALETTE[0], label="75/25 (Eq. 17)", zorder=3)
    ax.barh(y - h/2, df["RC_raw_FAHP"], h, color=PALETTE[1], label=r"raw FAHP $w^*$", zorder=3)
    xmax = max(df["RC_75_25"].max(), df["RC_raw_FAHP"].max())
    for i, r in enumerate(df.itertuples()):
        ax.text(max(r.RC_75_25, r.RC_raw_FAHP) + 0.012 * xmax / 0.5, i,
                f"#{r.Rank_75_25} vs #{r.Rank_raw_FAHP}" + ("  (shift %+d)" % r.Rank_shift_75_25_minus_raw
                                                           if r.Rank_shift_75_25_minus_raw != 0 else ""),
                va="center", fontsize=8,
                color=("#B00020" if abs(r.Rank_shift_75_25_minus_raw) >= 2 else "#333333"))
    ax.set_yticks(y); ax.set_yticklabels(df["Feature"])
    ax.set_xlim(0, xmax * 1.45)
    ax.set_xlabel(r"Closeness coefficient $RC$   (labels: rank 75/25 vs rank raw)")
    ax.set_title(f"(a) Fuzzy TOPSIS: 75/25 vs raw FAHP\nSpearman $\\rho$={rho:.3f}, Kendall $\\tau$={tau:.3f}")
    ax.legend(loc="lower right", fontsize=8)

    if sh is not None:
        grp = ["Outcome O\n(|O|=8)", "Resource S\n(|S|=3)"]
        end = sh["q_endogenous_from_w_star"].values; imp = sh["q_imposed_75_25"].values
        xx = np.arange(2); w = 0.36
        b1 = ax2.bar(xx - w/2, end, w, color=PALETTE[1], label=r"endogenous (from $w^*$)", zorder=3)
        b2 = ax2.bar(xx + w/2, imp, w, color=PALETTE[0], label="imposed 75/25", zorder=3)
        ax2.set_ylim(0, 1.1)
        _bar_value_labels(ax2, b1, fmt="{:.3f}", dy=0.015, fontsize=8)
        _bar_value_labels(ax2, b2, fmt="{:.2f}", dy=0.015, fontsize=8)
        ax2.set_xticks(xx); ax2.set_xticklabels(grp)
        ax2.set_ylabel("Group weight share")
        ax2.set_title("(b) Endogenous vs imposed group shares")
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.axis("off")
    fig.tight_layout()
    _save(fig, "fig11_raw_vs_7525", stamp=True)

def fig12_esi_vs_mc():
    """Compare ESI with Monte-Carlo dispersion and baseline weights."""
    d = _read("VALIDATION_ESI_vs_MC_STABILITY.csv")
    cv, wr = "MC_CV_all", "MC_widthrel_all"
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    specs = [
        (axes[0, 0], cv, "ESI", "Monte-Carlo CV = SD/mean", "(a) ESI vs Monte-Carlo CV"),
        (axes[0, 1], wr, "ESI", r"MC (p95$-$p5)/$w_0$", "(b) ESI vs relative percentile width"),
        (axes[1, 0], "w0_baseline", "ESI", r"Baseline weight $w_0$", r"(c) Confounder: ESI vs $w_0$"),
        (axes[1, 1], "w0_baseline", cv, r"Baseline weight $w_0$", r"(d) Confounder: MC CV vs $w_0$"),
    ]
    ylab = {"ESI": "ESI", cv: "Monte-Carlo CV"}
    for ax, xc, yc, xl, ttl in specs:
        ax.scatter(d[xc], d[yc], s=46, color=PALETTE[0], edgecolor="black", linewidth=0.5, zorder=3)
        for r in d.itertuples():
            ax.annotate(getattr(r, "criterion"), (getattr(r, xc), getattr(r, yc)),
                        textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="#333333")
        if len(d) > 2:
            k = np.polyfit(d[xc], d[yc], 1); xs = np.linspace(d[xc].min(), d[xc].max(), 50)
            ax.plot(xs, np.polyval(k, xs), color="grey", linestyle="--", linewidth=1, zorder=2)
        rho, tau = _rho_tau(d[xc], d[yc])
        _low = (xc == "w0_baseline" and yc == cv)
        ax.margins(x=0.18)
        ax.text(0.03, 0.04 if _low else 0.97, f"Spearman $\\rho$={rho:.2f}\nKendall $\\tau$={tau:.2f}", transform=ax.transAxes,
                va="bottom" if _low else "top", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9))
        ax.set_xlabel(xl); ax.set_ylabel(ylab.get(yc, yc)); ax.set_title(ttl, fontsize=10.5)
    verdict = ""
    try:
        sm = _read("VALIDATION_ESI_vs_MC_SUMMARY.csv")
        dr = sm[sm["section"] == "decision_rule"].iloc[-1]
        pt = sm[(sm["section"] == "partial_given_w0") & (sm["draw_subset"] == "all")].iloc[0]
        verdict = (f"Decision rule: {_verdict_short(dr['verdict'])}\n"
                   f"rho(ESI, CV) = {dr['spearman_rho_ESI_vs_dispersion']:.2f} "
                   f"[95% CI {dr['rho_ci95_lo']:.2f}, {dr['rho_ci95_hi']:.2f}], permutation p = {dr['perm_p_two_sided']:.3f}; "
                   f"partial given $w_0$ = {pt['spearman_rho_ESI_vs_dispersion']:.2f} "
                   f"[{pt['rho_ci95_lo']:.2f}, {pt['rho_ci95_hi']:.2f}]; n = {int(dr['n_criteria'])} criteria")
    except Exception:
        pass
    fig.suptitle("ESI and Monte-Carlo Weight Uncertainty\n" + verdict, y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig12_esi_vs_mc")

def fig13_esi_full_vs_cr():
    """Compare full-trajectory ESI with CR-admissible ESI."""
    d = _read("ESI_CR_restricted.csv").sort_values("Rank_ESI_full").reset_index(drop=True)
    try:
        s = _read("ESI_CR_summary.csv").iloc[0]
    except Exception:
        s = None
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w/2, d["ESI_full"], w, color=PALETTE[0], label="ESI (full adversarial trajectory)", zorder=3)
    ax.bar(x + w/2, d["ESI_CR"], w, color=PALETTE[2], label=r"ESI$_{CR}$ (CR$\leq$0.10 transitions only)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(d["Criterion"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Elasticity-stability index"); ax.set_ylim(0, max(d["ESI_full"].max(), d["ESI_CR"].max()) * 1.18)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) ESI per criterion (sorted by full-trajectory rank)")

    for r in d.itertuples():
        big = abs(r.rank_shift_CR_minus_full) >= 2
        ax2.plot([0, 1], [r.Rank_ESI_full, r.Rank_ESI_CR], marker="o", linewidth=1.8 if big else 1.0,
                 color="#B00020" if big else "#888888", zorder=3)
        ax2.text(-0.04, r.Rank_ESI_full, r.Criterion, ha="right", va="center", fontsize=7.5)
        ax2.text(1.04, r.Rank_ESI_CR, f"{r.Criterion}", ha="left", va="center", fontsize=7.5)
    ax2.set_xlim(-1.1, 2.1); ax2.set_ylim(len(d) + 0.7, 0.3)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["rank: full ESI", r"rank: ESI$_{CR}$"])
    ax2.set_yticks([]); ax2.grid(False)
    for sp in ("left", "bottom"): ax2.spines[sp].set_visible(False)
    ttl = "(b) Rank shift (red = |shift| $\\geq$ 2)"
    if s is not None:
        ttl += (f"\n$\\rho$={s['spearman_rho_ESI_full_vs_CR']:.3f}, $\\tau$={s['kendall_tau_ESI_full_vs_CR']:.3f}; "
                f"admissible $\\beta$: {s['admissible_betas']}")
    ax2.set_title(ttl, fontsize=10)
    if s is not None:
        fig.suptitle(f"Stress-Induced Inconsistency Check: {_verdict_short(s['verdict'])} "
                     f"({int(s['n_transitions_admissible'])} of {int(s['n_transitions_total'])} transitions admissible)",
                     y=1.01, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig13_esi_full_vs_cr")

def fig14_dematel_centrality():
    """Compare DEMATEL prominence with network centrality measures."""
    d = _read("DEMATEL_centrality_benchmark.csv").sort_values("Rank_DplusR_baseline")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.2, 1.4]})
    cols = [("Rank_WeightedDegree", "Weighted degree", "o", PALETTE[0], -0.12),
            ("Rank_EigenvectorCentrality", "Eigenvector centrality", "s", PALETTE[1], 0.0),
            ("Rank_DplusR_baseline", "DEMATEL $D+R$", "^", PALETTE[2], 0.12)]
    for col, lab, mk, c, off in cols:
        ax.scatter(d[col] + off * 0, np.arange(len(d)) + off, marker=mk, s=60, color=c, label=lab,
                   edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["Feature"]); ax.invert_yaxis()
    ax.set_xticks(range(1, len(d) + 1)); ax.set_xlabel("Rank (1 = most connected)")
    flagged = d[d["Flag_shift_ge_threshold"].astype(bool)]["Feature"].tolist()
    for i, r in enumerate(d.itertuples()):
        if bool(r.Flag_shift_ge_threshold):
            ax.text(len(d) + 0.6, i, "shift $\\geq$ 2", va="center", fontsize=8, color="#B00020")
    r1 = _rho_tau(d["Rank_WeightedDegree"], d["Rank_EigenvectorCentrality"])
    r2 = _rho_tau(d["Rank_WeightedDegree"], d["Rank_DplusR_baseline"])
    r3 = _rho_tau(d["Rank_EigenvectorCentrality"], d["Rank_DplusR_baseline"])
    ax.text(0.02, 0.03,
            f"WD vs EC: $\\rho$={r1[0]:.2f}, $\\tau$={r1[1]:.2f}\nWD vs D+R: $\\rho$={r2[0]:.2f}, $\\tau$={r2[1]:.2f}\n"
            f"EC vs D+R: $\\rho$={r3[0]:.2f}, $\\tau$={r3[1]:.2f}\nflagged features: {len(flagged)}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.95))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(a) Ranking by three connectedness measures", fontsize=10.5)

    yy = np.arange(len(d))
    ax2.barh(yy, d["Direct_part"], color=PALETTE[0], label="direct part (proportional to WD)", zorder=3)
    ax2.barh(yy, d["Indirect_part"], left=d["Direct_part"], color=PALETTE[4], label="indirect part ($T-X$)", zorder=3)
    for i, r in enumerate(d.itertuples()):
        ax2.text(r.Direct_part + r.Indirect_part + 0.3, i, f"{r.Indirect_share*100:.0f}% indirect", va="center", fontsize=8)
    ax2.set_yticks(yy); ax2.set_yticklabels(d["Feature"]); ax2.invert_yaxis()
    ax2.set_xlim(0, (d["Direct_part"] + d["Indirect_part"]).max() * 1.22)
    ax2.set_xlabel(r"DEMATEL prominence $D+R$ (baseline $Z_0$)")
    ax2.set_title("(b) Direct vs indirect contribution to $D+R$", fontsize=10.5)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    verdict = ""
    try:
        sm = _read("DEMATEL_centrality_benchmark_summary.csv")
        pr = sm[sm["Role"] == "primary"]
        verdict = _verdict_short(pr["Pair_verdict"].iloc[0]) if len(pr) else ""
    except Exception:
        pass
    extra = ""
    try:
        bt = _read("DEMATEL_centrality_expert_bootstrap.csv")
        share = float((bt["n_flagged"] > 0).mean() * 100)
        n_caused = int((d.get("Indirect_path_ablation_verdict", pd.Series(dtype=str)) == "CAUSED_BY_INDIRECT_PATHS").sum())
        extra = (f"\nexpert bootstrap ({len(bt)} resamples): shift $\\geq$ 2 in {share:.1f}% of resamples; "
                 f"path-length ablation: {n_caused} feature(s) with shift caused by indirect paths")
    except Exception:
        pass
    fig.suptitle(f"DEMATEL Prominence and Network Centrality: {verdict}{extra}",
                 y=1.03, fontsize=10.5)
    fig.tight_layout()
    _save(fig, "fig14_dematel_centrality")
OUTPUT_REGISTRY = [
    ("STRESS_FAHP_RESULTS.csv", "MCDM_AHP.py", "Sec 3.3, 5.1; Table (FAHP weights, ESI, ESI_CR)"),
    ("ESI_CR_restricted.csv", "MCDM_AHP.py", "Sec 3.3.3, 5.1, 6.1; Suppl."),
    ("ESI_CR_summary.csv", "MCDM_AHP.py", "Sec 3.3.3, 5.1, 6.1; Suppl."),
    ("crack_table_fahp.csv", "MCDM_AHP.py", "Sec 3.3.3, 5.1 (full adversarial trajectory)"),
    ("crack_table_fahp_CR_admissible.csv", "MCDM_AHP.py", "Suppl. (CR-admissible transitions)"),
    ("weight_trajectory.csv", "MCDM_AHP.py", "Fig. 2 / Sec 5.1", "-"),
    ("consistency_trajectory.csv", "MCDM_AHP.py", "Sec 3.3.2 (CR vs beta)"),
    ("STRESS_DEMATEL_RESULTS.csv", "MCDM_Dematel.py", "Sec 3.4.2, 5.3; Fig. 4"),
    ("DEMATEL_centrality_benchmark.csv", "MCDM_Dematel.py", "Sec 3.4.2, 5.3, 6.1; Table 2 wording; Suppl."),
    ("DEMATEL_centrality_benchmark_summary.csv", "MCDM_Dematel.py", "Sec 5.3, 6.1; Suppl."),
    ("DEMATEL_centrality_path_ablation.csv", "MCDM_Dematel.py", "Sec 5.3; Suppl. (path-length ablation)"),
    ("DEMATEL_centrality_expert_bootstrap.csv", "MCDM_Dematel.py", "Sec 5.3; Suppl. (expert bootstrap)"),
    ("DEMATEL_gamma_sensitivity.csv", "MCDM_Dematel.py", "Sec 5.3 / Suppl.", "-"),
    ("STRESS_TOPSIS_RESULT.csv", "MCDM_Topsis.py", "Sec 5.1 primary ranking; Table 6"),
    ("STRESS_TOPSIS_RESULT_75_25.csv", "MCDM_Topsis.py", "Sec 5.1; Table 6"),
    ("STRESS_TOPSIS_RESULT_raw_fahp.csv", "MCDM_Topsis.py", "Sec 3.4.1 p4, 5.1; Suppl."),
    ("STRESS_TOPSIS_endogenous_shares.csv", "MCDM_Topsis.py", "Sec 3.4.1 p4 / Eq. (17)"),
    ("STRESS_TOPSIS_raw_vs_7525.csv", "MCDM_Topsis.py", "Sec 3.4.1, 5.1, 5.3; Table 6; Suppl."),
    ("STRESS_TOPSIS_raw_vs_7525_summary.csv", "MCDM_Topsis.py", "Abstract; Sec 5.1, 6.1, 6.2, 7"),
    ("STRESS_TOPSIS_allocation_sensitivity.csv", "MCDM_Topsis.py", "Fig. 10 / Sec 5.3"),
    ("STRESS_TOPSIS_criterion_polarity_audit.csv", "MCDM_Topsis.py", "Sec 3.4.1 p5; Suppl. (polarity)"),
    ("STRESS_TOPSIS_polarity_raw_ratings_by_feature.csv", "MCDM_Topsis.py", "Suppl. (raw-rating audit)"),
    ("STRESS_TOPSIS_polarity_raw_ratings_by_expert.csv", "MCDM_Topsis.py", "Suppl. (raw-rating audit)"),
    ("STRESS_TOPSIS_polarity_empirical_check.csv", "MCDM_Topsis.py", "Sec 3.4.1 p5; Suppl. (empirical coding check)"),
    ("STRESS_TOPSIS_polarity_sensitivity.csv", "MCDM_Topsis.py", "Suppl. (what-if reversed)"),
    ("STRESS_TOPSIS_polarity_sensitivity_summary.csv", "MCDM_Topsis.py", "Suppl. / Sec 3.4.1 p5"),
    ("Final_Stress_Coupled_Result.csv", "MCDM_FINAL.py", "Sec 5.2 SRPI; Fig. 6"),
    ("SRPI_RAW_vs_7525_COMPARISON.csv", "MCDM_FINAL.py", "Sec 5.2 / Suppl."),
    ("SRPI_RAW_vs_7525_SUMMARY.csv", "MCDM_FINAL.py", "Sec 5.2, 6.1"),
    ("VALIDATION_ROBUSTNESS_RESULTS.csv", "MCDM_Validation.py", "Sec 5.3-5.4; Fig. 7, 8, 10"),
    ("VALIDATION_ROBUSTNESS_SUMMARY.csv", "MCDM_Validation.py", "Sec 5.3; Fig. 7"),
    ("VALIDATION_MONTE_CARLO_AHP.csv", "MCDM_Validation.py", "Sec 5.4; Fig. 9"),
    ("VALIDATION_ESI_vs_MC_STABILITY.csv", "MCDM_Validation.py", "Sec 3.3.3, 5.4, 6.1; Suppl."),
    ("VALIDATION_ESI_vs_MC_SUMMARY.csv", "MCDM_Validation.py", "Sec 5.4, 6.1; Suppl."),
    ("VALIDATION_POLARITY_SRPI_SENSITIVITY.csv", "MCDM_Validation.py", "Suppl. (polarity -> SRPI)"),
    ("VALIDATION_PRIMARY_CONSISTENCY_CHECK.csv", "MCDM_Validation.py", "Suppl. (reproducibility check)"),
    ("VALIDATION_PARAMETER_GRID.csv", "MCDM_Validation.py", "Sec 4.2, 4.3; Suppl. parameter grid"),
]

FIGURE_REGISTRY = [
    ("fig02_weight_trajectory_and_threshold_sensitivity.png", "Fig. 2 (script numbering)"),
    ("fig03_ahp_permutation_invariance.png", "script Fig. 3"),
    ("fig04_dematel_connectedness.png", "script Fig. 4"),
    ("fig05_prominence_trajectory.png", "script Fig. 5"),
    ("fig06_primary_vs_exploratory_ranking.png", "script Fig. 6"),
    ("fig07_robustness_summary.png", "script Fig. 7"),
    ("fig08_comparative_variants.png", "script Fig. 8"),
    ("fig09_monte_carlo_ahp_benchmark.png", "manuscript Fig. 10"),
    ("fig10_benefit_cost_allocation_sensitivity.png", "manuscript Fig. 11"),
    ("fig11_raw_vs_7525.png", "Supplementary / Sec 5.1"),
    ("fig12_esi_vs_mc.png", "Supplementary / Sec 5.4"),
    ("fig13_esi_full_vs_cr.png", "Supplementary / Sec 5.1"),
    ("fig14_dematel_centrality.png", "Supplementary / Sec 5.3"),
]

def _sheet_name(fname, used):
    base = os.path.splitext(fname)[0]
    if base.startswith("STRESS_TOPSIS_"):
        base = "TOPSIS_" + base[len("STRESS_TOPSIS_"):]
    elif base.startswith("STRESS_"):
        base = base[len("STRESS_"):]
    elif base.startswith("VALIDATION_"):
        base = "VAL_" + base[len("VALIDATION_"):]
    base = base[:28]
    name, k = base, 1
    while name in used:
        k += 1; name = f"{base[:26]}_{k}"
    used.add(name)
    return name

def build_supplementary_and_manifest():
    import datetime
    rows, used = [], set()
    xlsx_path = os.path.join(PROJECT_DIR, "SUPPLEMENTARY_REVIEW_TABLES.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        index_rows = []
        for fname, script, loc in OUTPUT_REGISTRY:
            path = os.path.join(PROJECT_DIR, fname)
            exists = os.path.exists(path)
            nrows = ncols = None
            sheet = ""
            if exists:
                try:
                    df = pd.read_csv(path)
                    nrows, ncols = df.shape
                    if len(df) <= 1000:
                        sheet = _sheet_name(fname, used)
                        df.to_excel(xw, sheet_name=sheet, index=False)
                        ws = xw.sheets[sheet]
                        for ci, col in enumerate(df.columns, start=1):
                            width = min(48, max(10, len(str(col)) + 2))
                            ws.column_dimensions[ws.cell(row=1, column=ci).column_letter].width = width
                        ws.freeze_panes = "A2"
                except Exception as e:
                    warnings.warn(f"[XLSX] {fname}: {e}")
            rows.append({"Output_file": fname, "Type": "csv", "Produced_by": script,
                         "Manuscript_location": loc, "Exists": exists,
                         "Rows": nrows, "Columns": ncols, "Sheet_in_xlsx": sheet,
                         "Modified_UTC": (datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
                                          if exists else "")})
        for fname, loc in FIGURE_REGISTRY:
            path = os.path.join(FIG_DIR, fname)
            exists = os.path.exists(path)
            rows.append({"Output_file": "figures_v3/" + fname, "Type": "png", "Produced_by": "MCDM_Visualization.py",
                         "Manuscript_location": loc, "Exists": exists, "Rows": None,
                         "Columns": None, "Sheet_in_xlsx": "",
                         "Modified_UTC": (datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
                                          if exists else "")})
        man = pd.DataFrame(rows)
        man.to_excel(xw, sheet_name="MANIFEST", index=False)
        try:
            _read("VALIDATION_PARAMETER_GRID.csv").to_excel(xw, sheet_name="PARAMETER_GRID", index=False)
        except FileNotFoundError:
            pass
        wb = xw.book
        order = ["MANIFEST", "PARAMETER_GRID"] + [s for s in wb.sheetnames if s not in ("MANIFEST", "PARAMETER_GRID")]
        wb._sheets = [wb[s] for s in order if s in wb.sheetnames]
    man_path = os.path.join(PROJECT_DIR, "REPRODUCIBILITY_OUTPUT_MANIFEST.csv")
    man.to_csv(man_path, index=False)
    missing = man[~man["Exists"]]["Output_file"].tolist()
    print(f"[SAVED] {xlsx_path}")
    print(f"[SAVED] {man_path}  ({len(man)} entries; missing: {missing if missing else 'none'})")


def main():
    generators = [
        ("Figure 2 — Weight trajectory + crack-event sensitivity", fig02_weight_trajectory_and_threshold_sensitivity),
        ("Figure 3 — AHP permutation invariance", fig03_ahp_permutation_invariance),
        ("Figure 4 — DEMATEL connectedness ranking (undirected)", fig04_dematel_connectedness),
        ("Figure 5 — Prominence trajectory", fig05_prominence_trajectory),
        ("Figure 6 — Primary (RC) vs exploratory (SRPI) ranking", fig06_primary_vs_exploratory_ranking),
        ("Figure 7 — Robustness summary", fig07_robustness_summary),
        ("Figure 8 — Comparative variants", fig08_comparative_variants),
        ("Figure 9 — Monte-Carlo AHP benchmark", fig09_monte_carlo_ahp_benchmark),
        ("Figure 10 — Benefit/cost allocation sensitivity (+ raw FAHP)", fig10_benefit_cost_allocation_sensitivity),
        ("Figure 11 — Raw FAHP vs 75/25", fig11_raw_vs_7525),
        ("Figure 12 — ESI vs Monte-Carlo", fig12_esi_vs_mc),
        ("Figure 13 — ESI full vs CR-admissible", fig13_esi_full_vs_cr),
        ("Figure 14 — DEMATEL vs centrality", fig14_dematel_centrality),
        ("Supplementary tables and reproducibility manifest", build_supplementary_and_manifest),
    ]

    for label, fn in generators:
        try:
            print(f"\n[RUNNING] {label}")
            fn()
        except FileNotFoundError as e:
            print(f"[SKIPPED] {label}: required file not found -> {e}")
        except Exception as e:
            print(f"[ERROR] {label}: {e}")

    print(f"\nAll available figures saved to: {FIG_DIR}")

if __name__ == "__main__":
    main()
