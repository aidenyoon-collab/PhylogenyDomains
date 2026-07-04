#!/usr/bin/env python3
"""
Part 3 (biological interpretation) publication figures for ScatterUMAP.

Renders THREE figures, each ONLY from real saved pipeline outputs
(never fabricate, jitter, trim, or alter data):

  (a) figures/part3_rf_importance.{pdf,png}
      Random-Forest feature importance, top-20 by Gini importance_score, with a
      paired panel of the held-out permutation importance for the SAME domains.
      Source: results/domain_analysis/rf_feature_importance.tsv
      (cols: domain_name, importance_score [Gini], permutation_importance
       [held-out], rank, perm_rank).

  (b) figures/part3_anova_boxplots.{pdf,png}
      Per-species RELATIVE ABUNDANCE grouped by clade for the top-8 ANOVA domains
      (smallest adjusted_p_value), in a 2x4 grid. Relative
      abundance is the pipeline's OWN per-species row-normalization
      (count / per-species total), i.e. the X_norm matrix returned by
      classification_cv_tree_reconstruction.load_and_preprocess_data() - the same
      relative-frequency matrix the ANOVA itself was computed on. Each species row
      of X_norm sums to ~1; the resulting per-domain values are small (~1e-5),
      which is expected and correct. RAW integer counts are NOT used (they make
      mostly-0/1 domains collapse to a flat line). The clade mapping is the
      canonical pipeline mapping IMPORTED from
      scripts/elastic_net_regression_stratified_species_cv.py (build_clade_mapping)
      via the same resolution used by the official analysis
      (assign_species_to_clades), restricted to the same 7 clades (n>=3) the
      enrichment/ANOVA used. NOTHING is invented; the species set (107) and clade
      assignment are reproduced from the live pipeline functions.
      Top-8 domains read from results/domain_analysis/anova_domains.tsv.

  (c) figures/part3_rf_anova_concordance.{pdf,png}
      RF-vs-ANOVA concordance: scatter of anova_rank vs rf_rank for the domains in
      the top-100 intersection, annotating the top concordant domain (PF02885).
      Source: results/domain_analysis/anova_rf_top100_intersection_by_anova.tsv
      (and ..._by_rf.tsv, which is the same 50-domain set in RF order).

INTEGRITY: every statistic printed on a figure equals the saved official value, or
is a transparent re-computation from a saved TSV (e.g. the Spearman rank
concordance of the saved intersection, the n per clade from the live mapping). The
clade mapping is the imported canonical one. No data is altered.

Usage:
    MPLCONFIGDIR=$PWD/.cache/matplotlib python3 figures/render_part3_interpretation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# --- Paths -------------------------------------------------------------------
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FIG_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "domain_analysis")
DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
DOMAIN_COUNTS_PATH = os.path.join(DATA_RAW, "MammalDomainCount.tsv")

RF_TSV = os.path.join(RESULTS_DIR, "rf_feature_importance.tsv")
ANOVA_TSV = os.path.join(RESULTS_DIR, "anova_domains.tsv")
INTER_ANOVA_TSV = os.path.join(RESULTS_DIR, "anova_rf_top100_intersection_by_anova.tsv")
INTER_RF_TSV = os.path.join(RESULTS_DIR, "anova_rf_top100_intersection_by_rf.tsv")

# Default headless config dir (the runner is expected to set MPLCONFIGDIR too).
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".cache", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# pubstyle (typography / sizing / palette / save) - import from figures/.
sys.path.insert(0, FIG_DIR)
import pubstyle  # noqa: E402
from pfam_name_fixes import fix_domain_label  # noqa: E402

# Canonical pipeline imports (clade mapping + the exact species/clade resolution the
# official Part-3 analysis uses). Importing these guarantees we use the SAME 7-clade
# set and the SAME 107-species set - we do NOT invent any mapping.
sys.path.insert(0, SCRIPTS_DIR)
from scipy.stats import spearmanr  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def short_domain_label(name: str, maxlen: int = 34) -> str:
    """'PF02885:Glycosyl transferase ...' -> 'PF02885 Glycosyl transferase ...'."""
    name = fix_domain_label(name)   # restore source-corrupted Pfam descriptions
    if ":" in name:
        acc, desc = name.split(":", 1)
        desc = desc.strip()
        label = f"{acc}  {desc}"
    else:
        label = name
    if len(label) > maxlen:
        label = label[: maxlen - 1].rstrip() + "..."
    return label


# Canonical clade name (as produced by build_clade_mapping) -> pubstyle.CLADE_COLORS key.
# These are identical strings for all 7 ANOVA clades; the dict is explicit so the match
# is auditable rather than implicit.
CANON_TO_COLORKEY = {
    "Afrotheria": "Afrotheria",
    "Glires": "Glires",
    "Primates": "Primates",
    "Marsupials": "Marsupials",
    "Laurasiatheria_Carnivora": "Laurasiatheria_Carnivora",
    "Laurasiatheria_Cetartiodactyla": "Laurasiatheria_Cetartiodactyla",
    "Laurasiatheria_Chiroptera": "Laurasiatheria_Chiroptera",
}

# Short x-axis labels (mirror scripts/domain_clade_interpretation.shorten_clade_label).
CLADE_SHORT = {
    "Afrotheria": "Afrotheria",
    "Glires": "Glires",
    "Primates": "Primates",
    "Marsupials": "Marsupials",
    "Laurasiatheria_Carnivora": "Carnivora",
    "Laurasiatheria_Cetartiodactyla": "Cetartiodactyla",
    "Laurasiatheria_Chiroptera": "Chiroptera",
}


# -----------------------------------------------------------------------------
# (a) RF feature importance: top-20 Gini + paired held-out permutation importance
# -----------------------------------------------------------------------------
def render_rf_importance() -> list:
    import matplotlib.pyplot as plt

    df = pd.read_csv(RF_TSV, sep="\t")
    # Top-20 by Gini importance_score (the saved 'rank' column is the Gini rank).
    top = df.sort_values("importance_score", ascending=False).head(20).reset_index(drop=True)

    # Order so the largest Gini bar is at the TOP of the horizontal axis.
    top = top.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top))
    labels = [short_domain_label(n) for n in top["domain_name"]]

    gini_color = pubstyle.OKABE_ITO[0]   # blue
    perm_color = pubstyle.OKABE_ITO[1]   # orange

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True,
        figsize=(pubstyle.DOUBLE_COL, 0.62 * pubstyle.DOUBLE_COL),
    )

    # Panel 1: Gini importance (the ranking variable).
    ax1.barh(y, top["importance_score"].values, color=gini_color, height=0.72,
             edgecolor="none")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=5.2)
    ax1.set_xlabel("Gini importance")
    ax1.set_title("RF Gini importance (top 20)")
    ax1.margins(y=0.01)

    # Panel 2: held-out permutation importance for the SAME domains (paired).
    ax2.barh(y, top["permutation_importance"].values, color=perm_color, height=0.72,
             edgecolor="none")
    ax2.axvline(0.0, color="0.5", linewidth=0.5)
    ax2.set_xlabel("Permutation importance (held-out)")
    ax2.set_title("Held-out permutation importance")

    # No panel letter here: letters are assigned by the composite (render_composites.py)
    # so a single panel never carries a stale/conflicting letter when re-laid-out.

    # Caption note: permutation importance UNDER-estimates under correlated features;
    # the Gini + ANOVA concordance is therefore the model-independent check.
    fig.text(
        0.5, -0.02,
        "Domains ordered by RF Gini importance (left). Held-out permutation importance "
        "for the same domains (right) UNDER-estimates\nimportance when domains are "
        "correlated (an important domain's score is shared with its correlates); hence "
        "Gini+ANOVA concordance\n(Fig. c) is the model-independent check rather than "
        "permutation importance alone.",
        ha="center", va="top", fontsize=5.4,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    base = os.path.join(FIG_DIR, "part3_rf_importance")
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    return [pdf, png]


# -----------------------------------------------------------------------------
# (b) ANOVA top-8 domains: per-species RELATIVE ABUNDANCE grouped by clade
# -----------------------------------------------------------------------------
def render_anova_boxplots(
    X_norm: pd.DataFrame, species_to_clade: dict, species_final: list
) -> list:
    """
    Boxplots of per-species RELATIVE ABUNDANCE for the top-8 ANOVA domains, grouped
    by clade, in a 2x4 grid.

    Relative abundance = (domain count for a species) / (total domain count for that
    species). This is exactly the pipeline's OWN per-species row-normalization: the
    `X_norm` matrix returned by
    classification_cv_tree_reconstruction.load_and_preprocess_data(), which is also
    the matrix the ANOVA was computed on. We do NOT re-derive or alter it - each row
    sums to ~1 and the per-domain values are small (~1e-5), which is expected and
    correct. RAW integer counts are deliberately NOT plotted (they collapse the
    mostly-0/1 domains to a flat line).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    anova = pd.read_csv(ANOVA_TSV, sep="\t")
    anova = anova.sort_values(["adjusted_p_value", "F_statistic"],
                              ascending=[True, False]).reset_index(drop=True)
    top8 = anova.head(8).reset_index(drop=True)
    top_domains = top8["domain_name"].tolist()

    # Integrity guard: every top domain must exist as a column of the pipeline's
    # relative-abundance matrix (no silent re-mapping / fabrication).
    missing = [d for d in top_domains if d not in X_norm.columns]
    if missing:
        raise SystemExit(f"ANOVA top domains missing from X_norm: {missing}")

    # 7 ANOVA clades (n>=3), in the same sorted order the pipeline uses.
    valid_clades = sorted(
        {c for c in species_to_clade.values() if c in CANON_TO_COLORKEY}
    )
    clade_to_species = {
        c: [s for s in species_final if species_to_clade.get(s) == c]
        for c in valid_clades
    }

    fig, axes = plt.subplots(
        2, 4, figsize=(pubstyle.DOUBLE_COL, 0.62 * pubstyle.DOUBLE_COL),
    )
    axes_flat = axes.ravel()

    box_w = 0.62
    for idx, dom in enumerate(top_domains):
        ax = axes_flat[idx]
        data_by_clade = []
        positions = []
        colors = []
        ticklabels = []
        for j, clade in enumerate(valid_clades):
            sp_c = clade_to_species[clade]
            # Relative abundance straight from the pipeline's X_norm (unaltered).
            vals = X_norm.loc[sp_c, dom].values.astype(float)
            data_by_clade.append(vals)
            positions.append(j)
            colors.append(pubstyle.CLADE_COLORS[CANON_TO_COLORKEY[clade]])
            # Single-line clade name; the per-clade n is shown ONCE in the legend
            # (it is identical across all 8 panels - repeating it here only crowds
            # the axis). rotation_mode="anchor" pins each label's end to its tick.
            ticklabels.append(CLADE_SHORT[clade])

        bp = ax.boxplot(
            data_by_clade, positions=positions, widths=box_w,
            showfliers=False, patch_artist=True,
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.85)
            patch.set_edgecolor("0.2")
            patch.set_linewidth(0.5)
        for element in ("whiskers", "caps"):
            for ln in bp[element]:
                ln.set_color("0.3")
                ln.set_linewidth(0.5)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(0.8)

        dom_fixed = fix_domain_label(dom)   # restore source-corrupted Pfam descriptions
        acc = dom_fixed.split(":", 1)[0]
        desc = dom_fixed.split(":", 1)[1].strip() if ":" in dom_fixed else ""
        title = f"{acc}: {desc}"
        if len(title) > 42:
            title = title[:41].rstrip() + "..."
        padj = float(top8.loc[idx, "adjusted_p_value"])
        ax.set_title(f"{title}\nADJ. P = {padj:.1e}", fontsize=5.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels, fontsize=5.2, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel("relative abundance", fontsize=6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(4.8)
        ax.margins(x=0.04)

    # Panel letter assigned by the composite (render_composites.py), not here.

    # Legend carries the per-clade n (relocated off the x-axis); 4 columns -> 2 tidy
    # rows so the longer "name (n=NN)" entries never run off the figure edge.
    legend_handles = [
        Patch(facecolor=pubstyle.CLADE_COLORS[CANON_TO_COLORKEY[c]], edgecolor="0.2",
              label=f"{CLADE_SHORT[c]} (n={len(clade_to_species[c])})")
        for c in valid_clades
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=5.6,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        "Top-8 ANOVA domains (smallest adjusted P): per-species relative abundance by clade",
        fontsize=8, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    base = os.path.join(FIG_DIR, "part3_anova_boxplots")
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    return [pdf, png]


# -----------------------------------------------------------------------------
# (c) RF-vs-ANOVA concordance: anova_rank vs rf_rank for the top-100 intersection
# -----------------------------------------------------------------------------
def render_concordance() -> list:
    import matplotlib.pyplot as plt

    inter = pd.read_csv(INTER_ANOVA_TSV, sep="\t")
    # Sanity: the by_rf file must contain the same domain set.
    inter_rf = pd.read_csv(INTER_RF_TSV, sep="\t")
    assert set(inter["domain_name"]) == set(inter_rf["domain_name"]), (
        "by_anova and by_rf intersection files disagree on the domain set."
    )

    n = len(inter)
    # Re-compute the rank concordance directly from the saved intersection table.
    rho, p = spearmanr(inter["anova_rank"], inter["rf_rank"])
    print(f"[part3_interp] RF-ANOVA concordance: n={n}  Spearman rho={rho:.4f}  P={p:.4g}")

    fig, ax = plt.subplots(
        figsize=(pubstyle.SINGLE_COL, 0.95 * pubstyle.SINGLE_COL)
    )

    # Color points by Gini importance_score (saved value) for context.
    sc = ax.scatter(
        inter["anova_rank"], inter["rf_rank"],
        c=inter["importance_score"], cmap=pubstyle.SEQUENTIAL_CMAP,
        s=16, edgecolors="white", linewidths=0.3, zorder=3,
    )
    lim = max(inter["anova_rank"].max(), inter["rf_rank"].max())
    ax.plot([1, lim], [1, lim], linestyle="--", linewidth=0.7, color="0.45",
            label="rank parity (y = x)", zorder=1)

    # Annotate the top concordant domain PF02885 (anova_rank=1, rf_rank=1). The
    # label is placed in the open band BELOW the diagonal (low RF rank, moderate
    # ANOVA rank) so neither the text nor its leader line crosses the point cloud.
    pf = inter[inter["domain_name"].str.startswith("PF02885")]
    if len(pf):
        ax_, rf_ = int(pf["anova_rank"].iloc[0]), int(pf["rf_rank"].iloc[0])
        ax.scatter([ax_], [rf_], s=42, facecolors="none", edgecolors="black",
                   linewidths=1.0, zorder=4)
        # Compact, stacked label placed in the verified-empty lower-right pocket
        # (ANOVA ~70-99, RF ~11-24 contains no points); the leader to the (1,1) point
        # stays in the empty sub-diagonal band so neither box nor line hits a marker.
        # (Identity 'Glycosyl transferase' is already shown in panels a/b + caption.)
        ax.annotate(
            "PF02885\nANOVA #1, RF #1",
            xy=(ax_, rf_), xytext=(70, 13),
            fontsize=5.4, ha="left", va="center",
            bbox=pubstyle.TEXTBBOX, zorder=20,
            arrowprops=dict(arrowstyle="-", linewidth=0.5, color="black"),
        )

    ax.set_xlabel("ANOVA rank (smaller = stronger)")
    ax.set_ylabel("RF Gini rank (smaller = stronger)")
    # pubstyle default title size (8, bold); explicit pad so it never crowds the axes.
    ax.set_title("RF vs ANOVA concordance", pad=8)
    # Open a clear empty band ALONG THE TOP (data max rf = 99) so the stats box sits
    # in genuine whitespace instead of over the point cloud. x keeps a small margin.
    ax.set_xlim(0, lim + 5)
    ax.set_ylim(0, lim + 28)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("RF Gini importance", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # Stats text in an opaque box pinned to the UPPER-LEFT corner, which now lies in
    # the empty band opened above the cloud, so neither the box nor the points collide.
    pubstyle.annotate(
        ax,
        f"n = {n} domains in top-100 ∩\nSpearman ρ = {rho:.2f} (P = {p:.3g})",
        loc="upper left", fontsize=5.8,
    )
    # Parity-line legend in the UPPER-RIGHT corner - the empty band opened above the
    # cloud (mirrors the stats box on the upper-left), so it clears both the points
    # and the PF02885 callout near the bottom. Opaque box for safety over the guide.
    leg = ax.legend(loc="upper right", fontsize=5.4)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_alpha(0.92)
    leg.set_zorder(20)

    # Panel letter assigned by the composite (render_composites.py), not here.
    fig.tight_layout()
    base = os.path.join(FIG_DIR, "part3_rf_anova_concordance")
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    return [pdf, png]


# -----------------------------------------------------------------------------
def main() -> None:
    pubstyle.apply()

    # Derive the canonical species set + clade assignment from the LIVE pipeline
    # functions (no invented mapping). This reproduces the official 107-species set.
    from domain_clade_interpretation import (  # noqa: E402
        load_and_preprocess_data, assign_species_to_clades,
    )
    X_norm, _dist, species_final = load_and_preprocess_data()
    species_to_clade = assign_species_to_clades(list(X_norm.index))

    written = []
    written += render_rf_importance()
    written += render_anova_boxplots(X_norm, species_to_clade, list(species_final))
    written += render_concordance()

    print("WROTE:")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
