#!/usr/bin/env python3
"""
Part-1 (theory) publication figures for ScatterUMAP.

Produces THREE figures, all rendered ONLY from real saved pipeline outputs
(never fabricate, jitter, trim, or alter data):

  (a) figures/part1_correlation_vs_N.{pdf,png}
      Four correlation series (variance/alpha x Spearman/Pearson) vs N on a
      log-x axis, read verbatim from the repo-root domain_time_correlations.tsv.
      N=1 is EXCLUDED (invalid single-domain run).
      N=50 (the selected best N) is annotated.

  (b) figures/part1_scatter_alpha_N50.{pdf,png}
  (c) figures/part1_scatter_var_N50.{pdf,png}
      Representative density scatters of the summed domain distance D_ij(N=50)
      vs TimeTree patristic distance - (b) alpha-weighted, (c) variance-weighted
      - reproduced with domain_time_scatter.py's OWN functions. The top-50
      domain panel is ranked by |correlation| (metric-independent), so it is
      IDENTICAL for both; only the per-domain weight differs. Log-density hexbin
      plus a LOWESS trend (colour by log point density; smooth trend line).

      FIDELITY ANCHOR: the recomputed Spearman/Pearson of (D, patristic) MUST
      match domain_time_correlations.tsv row N=50 (spearman_{alpha,var} /
      pearson_{alpha,var}) to ~3 decimals. If they do not, the script STOPS and
      reports rather than shipping an unfaithful figure.

Run:
    MPLCONFIGDIR=$PWD/.cache/matplotlib python3 figures/render_part1.py
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd

# --- Paths -----------------------------------------------------------------
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FIG_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")

CORR_TSV = os.path.join(PROJECT_ROOT, "domain_time_correlations.tsv")
DOMAIN_STATS_TSV = os.path.join(PROJECT_ROOT, "domain_time_domainStats.tsv")
MANTEL_JSON = os.path.join(PROJECT_ROOT, "results", "mantel_significance.json")


def _mantel_pstr(p, n_perm=9999):
    """Format a Mantel empirical p; at the floor 1/(n_perm+1) report '< 1e-4'."""
    return "≤ 1e-4" if p <= 1.0 / (n_perm + 1) + 1e-12 else f"= {p:.1e}"

# pubstyle + the live pipeline functions
sys.path.insert(0, FIG_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import pubstyle  # noqa: E402

# Import the OWN functions of the pipeline so the recompute is faithful.
from domain_time_scatter import (  # noqa: E402
    load_species_list,
    load_domain_counts,
    load_phylogeny_and_compute_patristic,
    select_ranked_domains,
    compute_summed_distances,
)

import matplotlib.pyplot as plt  # noqa: E402  (after pubstyle import; backend set in apply())
import matplotlib.colors as mcolors  # noqa: E402

N_BEST = 50  # the selected best N (annotated on figure a, used for figure b)
FIDELITY_TOL = 1e-3  # "~3 decimals"


# ---------------------------------------------------------------------------
# Figure (a): correlation vs N
# ---------------------------------------------------------------------------
def render_correlation_vs_N() -> str:
    """Plot the four correlation series vs N (log-x), excluding N=1, marking N=50."""
    df = pd.read_csv(CORR_TSV, sep="\t")

    # EXCLUDE N=1 (invalid single-domain run). Do not alter any value.
    plot_df = df[df["N"] != 1].sort_values("N").reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(pubstyle.SINGLE_COL, 0.82 * pubstyle.SINGLE_COL)
    )

    # alpha is the primary metric -> drawn solid, on top, in the Okabe-Ito
    # primary colors; variance is secondary -> dashed, muted.
    c_alpha_s = pubstyle.OKABE_ITO[0]  # blue   - alpha Spearman
    c_alpha_p = pubstyle.OKABE_ITO[1]  # orange - alpha Pearson
    c_var_s = pubstyle.OKABE_ITO[2]    # green  - variance Spearman
    c_var_p = pubstyle.OKABE_ITO[3]    # rose   - variance Pearson

    ax.plot(plot_df["N"], plot_df["pearson_alpha"], "-o", color=c_alpha_p,
            markersize=3, lw=1.1, label="alpha, Pearson (primary)", zorder=6)
    ax.plot(plot_df["N"], plot_df["spearman_alpha"], "-s", color=c_alpha_s,
            markersize=3, lw=1.1, label="alpha, Spearman (primary)", zorder=6)
    ax.plot(plot_df["N"], plot_df["pearson_var"], "--o", color=c_var_p,
            markersize=2.5, lw=0.9, label="variance, Pearson", zorder=4)
    ax.plot(plot_df["N"], plot_df["spearman_var"], "--s", color=c_var_s,
            markersize=2.5, lw=0.9, label="variance, Spearman", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("number of top domains (N)")
    ax.set_ylabel("correlation")
    ax.set_ylim(0.0, 0.85)

    # Mark/annotate N=50 (the selected best N). Use the actual table values.
    row50 = df[df["N"] == N_BEST].iloc[0]
    ax.axvline(N_BEST, color="0.5", lw=0.7, ls=":", zorder=1)
    y_top = max(row50["pearson_alpha"], row50["pearson_var"])
    ax.annotate(
        "N = 50\n(selected)",
        xy=(N_BEST, y_top),
        xytext=(N_BEST * 1.7, 0.50),
        fontsize=6,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7),
    )

    # Legend OUTSIDE the axes so it never overlaps the plotted correlation lines.
    pubstyle.legend_outside(ax, fontsize=5.5, handlelength=2.0)
    ax.set_title("Domain-distance vs TimeTree correlation")

    fig.tight_layout()
    base = os.path.join(FIG_DIR, "part1_correlation_vs_N")
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    print(f"[render_part1] wrote {pdf}")
    print(f"[render_part1] wrote {png}")
    return base


# ---------------------------------------------------------------------------
# Recompute D(N=50, alpha) vs patristic with the pipeline's OWN functions
# ---------------------------------------------------------------------------
def recompute_distance_N50(use_alpha=True):
    """Faithfully reproduce D_ij(N=50) and the patristic T_vec.

    Replays the exact data-loading + species-intersection + domain-frequency
    filter from domain_time_scatter.main(), then uses the SAVED per-domain
    stats (domain_time_domainStats.tsv) to rank domains and weight distances.
    The top-50 domain panel is ranked by |correlation| (metric-independent), so
    it is IDENTICAL for both weightings; only the per-domain weight differs
    (|alpha| if use_alpha else variance). Returns (T_vec, D, ranked_top50).
    """
    species_order_orig = load_species_list(os.path.join(DATA_RAW, "MammalsList.txt"))
    X_raw = load_domain_counts(
        os.path.join(DATA_RAW, "MammalDomainCount.tsv"), species_order_orig
    )
    species_order = list(X_raw.index)

    _, _, species_in_tree = load_phylogeny_and_compute_patristic(
        os.path.join(DATA_RAW, "MammalsPhylogeny.nwk"), species_order
    )

    # Final intersection across all three sources, minus the platypus outgroup
    # (Ornithorhynchus) - the canonical cohort shared with Parts 2/3.
    species_final = [sp for sp in species_order if sp in species_in_tree]
    species_final = [sp for sp in species_final if "Ornithorhynchus" not in sp]
    X_raw = X_raw.loc[species_final]

    T_vec, pairs, _ = load_phylogeny_and_compute_patristic(
        os.path.join(DATA_RAW, "MammalsPhylogeny.nwk"), species_final
    )

    # Domain frequency filter: keep domains present in >= 5% of species.
    presence_fraction = (X_raw > 0).sum(axis=0) / X_raw.shape[0]
    X_raw = X_raw.loc[:, (presence_fraction >= 0.05)]

    # Use the SAVED per-domain stats (per task instruction).
    domain_stats = pd.read_csv(DOMAIN_STATS_TSV, sep="\t")

    # Rank domains by |correlation| (select_ranked_domains; no TopDomains.txt
    # present -> filters pval<0.05 & alpha>0, sorts by abs_r), take top 50.
    top_domains_path = os.path.join(DATA_RAW, "TopDomains.txt")
    ranked = select_ranked_domains(domain_stats, top_domains_path)
    panel = ranked[:N_BEST]

    # Summed distance via the script's own function (alpha- or variance-weighted;
    # same numerator Sum ΔC^2, weight = |alpha_k| or var_k).
    D = compute_summed_distances(
        X_raw, panel, domain_stats, pairs, use_alpha=use_alpha
    )

    return np.asarray(T_vec, dtype=float), np.asarray(D, dtype=float), panel


# ---------------------------------------------------------------------------
# Figures (b)/(c): representative density scatter, N=50, alpha- or variance-weighted
# ---------------------------------------------------------------------------
def render_scatter_N50(use_alpha=True) -> str:
    from scipy.stats import pearsonr, spearmanr
    from statsmodels.nonparametric.smoothers_lowess import lowess

    weight = "alpha" if use_alpha else "var"           # TSV column suffix + filename
    label = "alpha-weighted" if use_alpha else "variance-weighted"
    T_vec, D, panel = recompute_distance_N50(use_alpha=use_alpha)

    # ---- FIDELITY ANCHOR (required) ----
    sp_computed, _ = spearmanr(D, T_vec)
    pe_computed, _ = pearsonr(D, T_vec)

    corr = pd.read_csv(CORR_TSV, sep="\t")
    row50 = corr[corr["N"] == N_BEST].iloc[0]
    sp_table = float(row50[f"spearman_{weight}"])
    pe_table = float(row50[f"pearson_{weight}"])

    print(f"[render_part1] FIDELITY ANCHOR (D vs patristic, N=50, {weight}):")
    print(f"    Spearman: computed={sp_computed:.6f}  table={sp_table:.6f}  "
          f"|diff|={abs(sp_computed - sp_table):.2e}")
    print(f"    Pearson : computed={pe_computed:.6f}  table={pe_table:.6f}  "
          f"|diff|={abs(pe_computed - pe_table):.2e}")

    if (abs(sp_computed - sp_table) > FIDELITY_TOL or
            abs(pe_computed - pe_table) > FIDELITY_TOL):
        raise SystemExit(
            "FIDELITY ANCHOR FAILED: recomputed correlations do not match "
            f"domain_time_correlations.tsv N={N_BEST} ({weight}) to ~3 decimals. "
            "The recompute is unfaithful; refusing to ship a mismatched figure."
        )
    print("[render_part1] FIDELITY ANCHOR PASSED (match to <1e-3).")

    # Guard: a log y-scale would silently drop any D<=0 point. The numerator is
    # Sum ΔC^2 (identical panel for both weightings), so D>0 for every pair unless
    # all 50 panel domains are identical between a species pair - assert it holds.
    if not (D > 0).all():
        raise SystemExit(
            f"{int((D <= 0).sum())} pair(s) have D<=0 at N=50/{weight}; a log "
            "y-axis would drop them. Refusing to ship a figure that hides data."
        )

    # ---- Figure ----
    fig, ax = plt.subplots(
        figsize=(pubstyle.SINGLE_COL, 0.85 * pubstyle.SINGLE_COL)
    )

    # Log-density hexbin (colour by log point density, not raw density).
    # y is right-skewed over several decades -> log y-axis spreads the dense
    # low-distance region. All 5671 pairs are non-zero (asserted above), so a
    # log y-scale drops no points.
    hb = ax.hexbin(
        T_vec, D,
        gridsize=42,
        cmap=pubstyle.SEQUENTIAL_CMAP,
        bins="log",
        mincnt=1,
        yscale="log",
        linewidths=0.0,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("log$_{10}$(pair count)", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    # LOWESS trend: high frac so it reads as a smooth curve. Computed in
    # log-y space (matching the rendered axis) on the real, untrimmed data.
    sm = lowess(np.log10(D), T_vec, frac=0.6, return_sorted=True)
    ax.plot(sm[:, 0], 10 ** sm[:, 1], "-", color=pubstyle.OKABE_ITO[5],
            lw=1.4, label="LOWESS", zorder=8)

    ax.set_xlabel("TimeTree patristic distance (MY)")
    ax.set_ylabel(f"domain distance D(N=50, {label})")
    ax.set_title(f"{label.capitalize()} domain distance vs TimeTree distance (N=50)")

    # Open a little headroom at the top of the (log) y-axis so the stats box and
    # the LOWESS legend have clear whitespace and never sit over hexbins. This is
    # a pure axis-limit (display) change; no plotted value is touched.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.2)

    # Report the official (table) statistics on the figure. These EQUAL the
    # saved values (verified by the fidelity anchor above). Significance is the
    # MANTEL species-permutation p - NOT the invalid i.i.d. p over 5,671
    # non-independent pairs; the independent unit is the 107 species.
    mantel_key = "part1_alpha_N50" if use_alpha else "part1_variance_N50"
    mantel = json.load(open(MANTEL_JSON))[mantel_key]
    sp_p = _mantel_pstr(mantel["spearman_p_mantel"], mantel["n_perm"])
    pe_p = _mantel_pstr(mantel["pearson_p_mantel"], mantel["n_perm"])
    stat_txt = (f"Spearman = {sp_table:.3f} (P {sp_p})\n"
                f"Pearson = {pe_table:.3f} (P {pe_p})\n"
                f"{len(T_vec):,} pairs · {mantel['n_species']} species (Mantel)")
    pubstyle.annotate(ax, stat_txt, loc="upper left", fontsize=6)

    # LOWESS legend in the empty vertical strip between the two data clusters
    # (TimeTree distance ~215-305 MY has no pairs), high up where there is clear
    # whitespace, with an opaque frame so it never overlaps markers or the curve.
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.66, 0.99),
                    fontsize=6, frameon=True, framealpha=0.92,
                    edgecolor="0.85", borderaxespad=0.0)
    leg.set_zorder(20)
    fig.tight_layout()

    base = os.path.join(FIG_DIR, f"part1_scatter_{weight}_N50")
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    print(f"[render_part1] wrote {pdf}")
    print(f"[render_part1] wrote {png}")
    return base


def main() -> None:
    pubstyle.apply()
    render_correlation_vs_N()
    render_scatter_N50(use_alpha=True)    # (b) part1_scatter_alpha_N50
    render_scatter_N50(use_alpha=False)   # (c) part1_scatter_var_N50
    print("[render_part1] done.")


if __name__ == "__main__":
    main()
