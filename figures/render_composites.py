#!/usr/bin/env python3
"""
Assemble paper-ready MULTI-PANEL composite figures from the already-rendered,
integrity-verified individual panels (figures/part1_*, part2_*, part3_*) and the
externally-rendered tree images (figures/paper_assets/).

This script does NOT recompute or alter any data - it only LAYS OUT existing
panel images (matplotlib imshow, native aspect preserved) with panel letters and
shared elements. The scientific content of every panel is exactly what its own
renderer produced (each of which carries its own fidelity/integrity check).

Composites (Nature double-column width; panels kept at native aspect):
  fig2_part1_correlation     (a) correlation-vs-N  (b) alpha N=50  (c) variance N=50
  fig3_part2_classifier      (a) confusion matrix  (b) distance recovery
  fig4_part2_trees           (a) reference TimeTree  (b) predicted domain UPGMA  [+ color-sync note]
  fig5_part3_domains         (a) z-score heatmap (b) ANOVA boxplots
                             (c) RF importance (d) RF-ANOVA concordance (e) volcano (Primates)

NOTE: composite panels are raster (the source PNGs at 600 dpi) embedded in a vector
PDF. The individual vector PDFs remain the high-res source panels; the trees are
high-resolution radial iTOL exports (vector source in figures/paper_assets/*.pdf,
rasterized to ~4800 px PNG for the montage).

Run:
    MPLCONFIGDIR=$PWD/.cache/matplotlib python3 figures/render_composites.py
"""
from __future__ import annotations
import os
import sys

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(FIG_DIR, "paper_assets")
sys.path.insert(0, FIG_DIR)
import pubstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

DOUBLE = pubstyle.DOUBLE_COL  # ~7.2 in


def _aspect(path: str) -> float:
    w, h = Image.open(path).size
    return w / h


def _panel(ax, path: str, label: str | None):
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    if label:
        ax.text(-0.01, 1.0, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", ha="right", clip_on=False)


def compose(rows, base, title=None, fig_width=DOUBLE,
            wspace=0.04, hspace=0.12, title_in=0.32):
    """rows: list of rows; each row a list of (path, label). Cell aspects match
    the panel images so imshow fills with minimal letterboxing."""
    row_h = [fig_width / sum(_aspect(p) for p, _ in row) for row in rows]
    fig_h = sum(row_h) * (1 + hspace) + (title_in if title else 0.05)
    fig = plt.figure(figsize=(fig_width, fig_h))
    top = 1 - (title_in / fig_h) if title else 0.995
    outer = fig.add_gridspec(len(rows), 1, height_ratios=row_h, hspace=hspace,
                             top=top, bottom=0.01, left=0.01, right=0.99)
    for r, row in enumerate(rows):
        asps = [_aspect(p) for p, _ in row]
        inner = outer[r].subgridspec(1, len(row), width_ratios=asps, wspace=wspace)
        for c, (p, lab) in enumerate(row):
            _panel(fig.add_subplot(inner[0, c]), p, lab)
    if title:
        fig.suptitle(title, fontsize=8, fontweight="bold", y=0.997)
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    print(f"[composites] wrote {png}")
    return base


def f(name):  # figures/<name>
    return os.path.join(FIG_DIR, name)


def fig_part1():
    compose(
        [[(f("part1_correlation_vs_N.png"), "a"),
          (f("part1_scatter_alpha_N50.png"), "b"),
          (f("part1_scatter_var_N50.png"), "c")]],
        f("fig2_part1_correlation"),
        title="Domain-distance vs phylogenetic-distance correlation and its dependence on N",
    )


def fig_part2_classifier():
    compose(
        [[(f("part2_confusion_matrix.png"), "a"),
          (f("part2_distance_recovery.png"), "b")]],
        f("fig3_part2_classifier"),
        title="Random Forest classifier recovers pairwise divergence-distance classes",
    )


def fig_part2_trees():
    # Two trees STACKED VERTICALLY (a over b): each panel spans the full figure
    # width, so both trees print substantially larger / more legible than a
    # side-by-side layout. No fabricated clade legend:
    # the iTOL tree colours are not yet synced to pubstyle.CLADE_COLORS, so a
    # pubstyle-coloured legend would mislabel them. A short note flags the sync +
    # resolution caveat.
    pa = (f("paper_assets/tree_reference_timetree.png"))
    pb = (f("paper_assets/tree_predicted_domain.png"))
    a_a, a_b = _aspect(pa), _aspect(pb)
    fig_w = DOUBLE
    h_a, h_b = fig_w / a_a, fig_w / a_b       # full-width panel heights (in)
    title_in, note_in, gap_in = 0.42, 0.26, 0.40
    fig_h = h_a + h_b + title_in + note_in + gap_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[h_a, h_b],
                          hspace=gap_in / ((h_a + h_b) / 2),
                          top=1 - title_in / fig_h, bottom=note_in / fig_h,
                          left=0.01, right=0.99)
    ax0 = fig.add_subplot(gs[0, 0]); _panel(ax0, pa, "a")
    ax0.set_title("Reference (TimeTree)", fontsize=7)
    ax1 = fig.add_subplot(gs[1, 0]); _panel(ax1, pb, "b")
    ax1.set_title("Predicted (domain UPGMA)", fontsize=7)
    fig.suptitle("Reference and Predicted Tree",
                 fontsize=8, fontweight="bold", y=0.998)
    fig.text(0.5, 0.012,
             "Same 107-tip taxon set; 7 clades (n≥3) + grey 'Other (n<3)'. Broad clades "
             "recovered but fine topology is not (normalized RF ≈ 0.885). Predicted "
             "distances take 4 ordinal class values, so distance 'recovery' is bounded by "
             "the binning (a perfect 4-class oracle ceilings at Pearson ≈0.85; model 0.83) "
             "- the confusion matrix (Fig 3) is the primary Part-2 result.",
             ha="center", va="bottom", fontsize=5.0, style="italic", color="0.35")
    pubstyle.save(fig, f("fig4_part2_trees"))
    plt.close(fig)
    print("[composites] wrote fig4_part2_trees.png")


def fig_part3_variation():
    # Split of the old combined domains figure: the top two
    # panels (ANOVA cross-clade variation) as their own figure, stacked vertically.
    compose(
        [[(f("part3_domain_heatmap.png"), "a")],
         [(f("part3_anova_boxplots.png"), "b")]],
        f("fig5_part3_variation"),
        title="Domains that vary across clades",
        fig_width=5.5,
        hspace=0.12,
    )


def fig_part3_importance():
    # Split of the old combined domains figure: the bottom
    # panels as their own figure - RF importance (a) full-width on top, with the
    # RF-ANOVA concordance (b) and per-clade enrichment volcano (c) side by side
    # below. Re-lettered a/b/c. (Balanced layout chosen over a pure 3-panel
    # vertical stack, which prints as an unreadably tall/narrow strip.)
    compose(
        [[(f("part3_rf_importance.png"), "a")],
         [(f("part3_rf_anova_concordance.png"), "b"),
          (f("part3_volcano_Primates.png"), "c")]],
        f("fig6_part3_importance"),
        title="Domains that drive the predictions and distinguish clades",
        fig_width=6.5,
        hspace=0.14,
    )


def main():
    pubstyle.apply()
    fig_part1()
    fig_part2_classifier()
    fig_part2_trees()
    fig_part3_variation()
    fig_part3_importance()
    print("[composites] done.")


if __name__ == "__main__":
    main()
