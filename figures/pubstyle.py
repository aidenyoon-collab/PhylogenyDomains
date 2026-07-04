"""
Reusable publication figure style for ScatterUMAP - import this from every figure
script so typography, sizing, and export are consistent (consistency is half of why
a figure set reads as "publication quality").

Usage:
    import sys, os; sys.path.insert(0, "<repo>/figures")
    import pubstyle
    pubstyle.apply()
    fig, ax = plt.subplots(figsize=(pubstyle.SINGLE_COL, 0.85*pubstyle.SINGLE_COL))
    ...
    pubstyle.save(fig, "<repo>/figures/<name>")   # writes .pdf (vector) + .png (600 dpi)

INTEGRITY: this module only controls *appearance*. Never use it as cover to alter,
trim, or jitter data to make a plot look better. Figures are regenerated from the
real pipeline outputs only.

JOURNAL SPECS: the dimensions/fonts below are the de-facto Nature-family standard.
**Confirm the exact numbers against the chosen journal's current artwork guide**
before final submission (column widths, min font size, line weights, color mode).
"""
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature column widths (inches). Never stretch a figure beyond these.
MM = 1.0 / 25.4
SINGLE_COL = 89 * MM     # ~3.50 in
ONEHALF_COL = 120 * MM   # ~4.72 in
DOUBLE_COL = 183 * MM    # ~7.20 in

# Colorblind-safe categorical palette (Okabe-Ito). Use for discrete categories.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442", "#000000"]
SEQUENTIAL_CMAP = "viridis"   # sequential / density (colorblind-safe, perceptually uniform)

# Canonical clade -> colour (colorblind-safe, Paul Tol 'muted'). A clade is the SAME
# colour in EVERY figure. The keys are the project's working clade set (the 7 enrichment
# clades + the Monotreme outgroup).
# iTOL sync: paste these hex codes into the iTOL tree annotation so the tree panels
# match the matplotlib panels - or, if the tree already uses a different palette, adopt
# those hues here instead (consistency matters more than which exact hues).
CLADE_COLORS = {
    "Monotremes":                     "#000000",  # outgroup (platypus); usually excluded
    "Marsupials":                     "#882255",  # wine
    "Afrotheria":                     "#88CCEE",  # cyan
    "Glires":                         "#117733",  # green
    "Primates":                       "#CC6677",  # rose
    "Laurasiatheria_Carnivora":       "#332288",  # indigo
    "Laurasiatheria_Cetartiodactyla": "#DDCC77",  # sand
    "Laurasiatheria_Chiroptera":      "#44AA99",  # teal
}

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.titleweight": "bold",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.9,
    "figure.dpi": 150,        # on-screen preview
    "savefig.dpi": 600,       # raster export
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,       # editable text in Illustrator/Inkscape
    "ps.fonttype": 42,
}


def apply() -> None:
    """Apply the publication rcParams (Agg backend so it runs headless)."""
    mpl.use("Agg")
    mpl.rcParams.update(_RC)


def save(fig, basepath: str, dpi: int = 600) -> tuple[str, str]:
    """Save BOTH a vector PDF and a high-dpi PNG. Returns (pdf_path, png_path)."""
    os.makedirs(os.path.dirname(os.path.abspath(basepath)), exist_ok=True)
    pdf, png = basepath + ".pdf", basepath + ".png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    return pdf, png


def panel_label(ax, letter: str) -> None:
    """Bold lowercase panel letter (a/b/c...) at the top-left, for multipanels."""
    ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="right")


# Opaque-ish white box drawn BEHIND in-axes text so it never visually collides with
# markers or guide lines - a standard publication-figure fix for overlapping text.
TEXTBBOX = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.85",
                linewidth=0.5, alpha=0.92)

_CORNERS = {
    "upper left": (0.025, 0.975, "top", "left"),
    "upper right": (0.975, 0.975, "top", "right"),
    "lower left": (0.025, 0.025, "bottom", "left"),
    "lower right": (0.975, 0.025, "bottom", "right"),
}


def annotate(ax, text, loc="upper left", fontsize=6, **kw):
    """In-axes text pinned to a corner with an opaque white box, so it reads cleanly
    over any data points / guide lines. `loc` ∈ {upper,lower}×{left,right}."""
    x, y, va, ha = _CORNERS[loc]
    return ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha,
                   fontsize=fontsize, bbox=TEXTBBOX, zorder=20, **kw)


def legend_outside(ax, **kw):
    """Place the legend just OUTSIDE the axes (upper-right) so it never overlaps the
    data cloud or guide lines."""
    kw.setdefault("loc", "upper left")
    kw.setdefault("bbox_to_anchor", (1.01, 1.0))
    kw.setdefault("frameon", False)
    kw.setdefault("borderaxespad", 0.0)
    return ax.legend(**kw)
