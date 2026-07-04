#!/usr/bin/env python3
"""
Overview / methods schematic for the ScatterUMAP paper (new Figure 1).

A workflow diagram - NOT a data figure. It draws boxes + arrows describing the
pipeline; the only "data" are the fixed dataset sizes (107 species, 8,020
variance-filtered domains, 5,671 pairs, 4 distance classes), which are the
project's documented dataset invariants. No analysis output is read or altered.

Run:
    MPLCONFIGDIR=$PWD/.cache/matplotlib python3 figures/render_schematic.py
"""
from __future__ import annotations
import os
import sys

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG_DIR)
import pubstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


def _box(ax, x, y, w, h, text, fc, ec="0.25", fs=7, tc="black", weight="normal"):
    """Rounded box centered at (x, y) in axes-fraction coords, with wrapped text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=3, fontweight=weight, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="0.35"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.0, color=color, zorder=1,
        shrinkA=2, shrinkB=2,
    ))


def render_schematic() -> str:
    pubstyle.apply()
    fig, ax = plt.subplots(figsize=(pubstyle.DOUBLE_COL, 0.52 * pubstyle.DOUBLE_COL))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # palette
    c_data = "#DCE6F1"     # inputs (light blue)
    c_feat = "#FBEAD2"     # features (light orange)
    c_part1 = "#FFF1B8"    # Part-1 analytical metric (light yellow; distinct lane)
    c_part2 = "#D7E9DC"    # reconstruction (light green)
    c_part3 = "#EAD9E6"    # interpretation (light purple)
    c_ref = "#EFEFEF"      # reference / ground truth (grey)

    # ---- Inputs (central-left, branch up to Part 1/2 and down to Part 3) ----
    _box(ax, 0.11, 0.60, 0.18, 0.16,
         "PFAM domain counts\n(UMAP of Life)\n107 species\n× 8,020 domains",
         c_data, weight="bold", fs=6.6)
    _box(ax, 0.11, 0.22, 0.18, 0.14,
         "TimeTree reference\nphylogeny\n(patristic dist., MY)",
         c_ref, weight="bold", fs=6.6)

    # ---- Top lane: Parts 1 & 2 (features -> classifier -> tree) ----
    _box(ax, 0.37, 0.80, 0.18, 0.14,
         "Pairwise features\n|ΔC| per domain\n(5,671 pairs)", c_feat, fs=6.6)
    _box(ax, 0.37, 0.575, 0.18, 0.13,
         "Part 1 - analytical\nmetric Σ ΔC²/weight\nvs time (effect of N)",
         c_part1, ec="0.45", fs=6.2)
    _box(ax, 0.61, 0.80, 0.17, 0.14,
         "Random Forest\nclassifier\n(species-blocked CV)", c_part2, weight="bold", fs=6.6)
    _box(ax, 0.61, 0.575, 0.17, 0.13,
         "4 distance classes\n-> mapped to MY\n(train folds only)", c_part2, fs=6.4)
    _box(ax, 0.85, 0.685, 0.18, 0.16,
         "UPGMA tree\nvs TimeTree\n(RF distance,\ndistance recovery)", c_part2, weight="bold", fs=6.6)

    # ---- Bottom lane: Part 3 (interpretation) ----
    _box(ax, 0.49, 0.22, 0.20, 0.16,
         "Part 3 - which domains?\nper-clade enrichment\n(Mann-Whitney) · ANOVA\n· RF importance",
         c_part3, weight="bold", fs=6.3)
    _box(ax, 0.76, 0.22, 0.18, 0.14,
         "Biological\ninterpretation\n(diet, immunity,\nchemosensory ...)", c_part3, fs=6.4)

    # ---- arrows ----
    _arrow(ax, 0.19, 0.65, 0.30, 0.77)            # counts -> features (up-right)
    _arrow(ax, 0.19, 0.55, 0.40, 0.30)            # counts -> Part 3 (down-right)
    _arrow(ax, 0.46, 0.80, 0.525, 0.80)           # features -> RF
    _arrow(ax, 0.205, 0.585, 0.28, 0.578)         # counts -> Part 1 (raw counts, not the |ΔC| feature)
    _arrow(ax, 0.61, 0.73, 0.61, 0.64)            # RF -> classes
    _arrow(ax, 0.695, 0.61, 0.78, 0.665)          # classes -> UPGMA/compare
    _arrow(ax, 0.59, 0.22, 0.67, 0.22)            # enrichment -> interpretation
    _arrow(ax, 0.20, 0.26, 0.78, 0.62, color="0.6")  # TimeTree -> tree comparison (reference)

    # ---- two questions banner ----
    ax.text(0.5, 0.045,
            "Q1: How much phylogenetic signal is in domain content?     "
            "Q2: Which domains carry it, and what do they do?",
            ha="center", va="center", fontsize=6.5, style="italic", color="0.25")

    ax.set_title("Workflow: phylogenetic signal in mammalian protein-domain copy number",
                 fontsize=8, fontweight="bold")
    fig.tight_layout()
    base = os.path.join(FIG_DIR, "fig1_overview_schematic")
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    print(f"[schematic] wrote {pdf}")
    print(f"[schematic] wrote {png}")
    return base


if __name__ == "__main__":
    render_schematic()
