#!/usr/bin/env python3
"""
Part 3 (biology) - informative domain heatmap.

TOP-40 domains (by ANOVA F_statistic) x SPECIES, coloured by each domain's
Z-SCORE of relative frequency (z-scored PER DOMAIN / per row across species),
species ordered BY CLADE with separators between clade blocks.

Species are ordered phylogenetically/by clade and z-score normalized for
visible contrast, which is far more informative than the clade-level log2FC
summary in part3_enrichment_heatmap (which is left untouched as the
enrichment-summary alternative).

Nothing here is fabricated, jittered, trimmed, or
altered. The relative-frequency matrix is recomputed via the PIPELINE'S OWN
function `load_and_preprocess_data()` (so it is byte-for-byte the same
normalization the classifier/ANOVA use: per-species row-normalized counts,
platypus excluded, zero-variance domains dropped). The top-40 selection comes
verbatim from results/domain_analysis/anova_domains.tsv (sorted by F_statistic).
The only transforms are:
  - per-domain z-score: (x - mean) / std across species  (the figure's stated
    normalization; verified mean~0, std~1 per row), and
  - the clade ordering, which only PERMUTES columns, never changes any value.
All of these are spot-checked at the end (see _integrity_checks()).

Run from project root (set the matplotlib cache dir):
  MPLCONFIGDIR=$PWD/.cache/matplotlib python3 figures/render_part3_heatmap.py

Writes (NEW; does NOT touch part3_enrichment_heatmap):
  figures/part3_domain_heatmap.pdf   (vector)
  figures/part3_domain_heatmap.png   (600 dpi)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "figures"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pubstyle  # noqa: E402
from pfam_name_fixes import fix_domain_label  # noqa: E402

pubstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Canonical pipeline functions - the SAME normalization the classifier/ANOVA use.
from classification_cv_tree_reconstruction import load_and_preprocess_data  # noqa: E402
from elastic_net_regression_stratified_species_cv import (  # noqa: E402
    CLADE_TO_SPECIES_UNDERSCORE,
    build_clade_mapping,
)

ANOVA_TSV = os.path.join(REPO, "results", "domain_analysis", "anova_domains.tsv")
OUT_DIR = os.path.join(REPO, "figures")
OUT_BASE = os.path.join(OUT_DIR, "part3_domain_heatmap")

N_TOP = 40  # top-40 domains by ANOVA F_statistic

# Canonical clade block order = the order clades appear in the pipeline's
# CLADE_TO_SPECIES_UNDERSCORE dict (monotreme outgroup first, then deep-to-shallow).
# Platypus (Monotreme_outgroup) is excluded upstream by the pipeline, so it never
# appears here; we keep the order otherwise canonical.
CLADE_BLOCK_ORDER = list(CLADE_TO_SPECIES_UNDERSCORE.keys())

# Map the pipeline clade keys -> pubstyle.CLADE_COLORS keys (canonical, one colour
# per clade in EVERY figure). The two clades too small to test (Eulipotyphla n=1,
# Perissodactyla n=2; the n<3 clades excluded from ANOVA/enrichment) are shown as a
# neutral grey "Other (n<3)" so the palette matches Fig 4 + the boxplots - NOT folded
# into a neighbouring clade's colour, which would mislabel them taxonomically.
GREY_OTHER = "#BBBBBB"
CLADE_COLOR_KEY = {
    "Monotreme_outgroup": "Monotremes",
    "Marsupials": "Marsupials",
    "Afrotheria": "Afrotheria",
    "Primates": "Primates",
    "Glires": "Glires",
    "Laurasiatheria_Eulipotyphla": "Other",
    "Laurasiatheria_Chiroptera": "Laurasiatheria_Chiroptera",
    "Laurasiatheria_Carnivora": "Laurasiatheria_Carnivora",
    "Laurasiatheria_Perissodactyla": "Other",
    "Laurasiatheria_Cetartiodactyla": "Laurasiatheria_Cetartiodactyla",
}


def _short_clade(name: str) -> str:
    """Compact display label for a clade block."""
    return name.replace("Laurasiatheria_", "Laura. ").replace("_", " ")


def _short_domain(label: str, maxlen: int = 46) -> str:
    """'PFxxxxx:Some long name' -> keep accession + truncated name for the row tick."""
    label = fix_domain_label(label)   # restore source-corrupted Pfam descriptions
    if len(label) <= maxlen:
        return label
    return label[: maxlen - 1] + "..."


def select_top_domains() -> pd.DataFrame:
    """Top-40 domains by ANOVA F_statistic, verbatim from anova_domains.tsv."""
    anova = pd.read_csv(ANOVA_TSV, sep="\t")
    top = anova.sort_values("F_statistic", ascending=False).head(N_TOP).reset_index(drop=True)
    return top


def order_species_by_clade(species_final: list[str]) -> tuple[list[str], list[str], list[int]]:
    """
    Order species by clade block (canonical dict order). Within a block, species
    are sorted alphabetically for a stable, reproducible layout.

    Returns:
      ordered_species : species names in clade-block order
      ordered_clades  : per-species clade label (parallel to ordered_species)
      block_clades    : the clade label of each non-empty block, in display order
    """
    mapping = build_clade_mapping()
    species_to_clade = {sp: mapping[sp] for sp in species_final}  # all map (verified)

    ordered_species: list[str] = []
    ordered_clades: list[str] = []
    block_clades: list[str] = []
    for clade in CLADE_BLOCK_ORDER:
        members = sorted(sp for sp in species_final if species_to_clade[sp] == clade)
        if not members:
            continue
        block_clades.append(clade)
        ordered_species.extend(members)
        ordered_clades.extend([clade] * len(members))
    return ordered_species, ordered_clades, block_clades


def build_heatmap() -> dict:
    # --- 1. Faithful recompute of relative frequencies via the pipeline's own fn ---
    X_norm, _dist, species_final = load_and_preprocess_data()  # species x domains, row-normalized

    # --- 2. Top-40 domains by ANOVA F_statistic (verbatim selection) ---
    top = select_top_domains()
    domains = list(top["domain_name"])
    missing = [d for d in domains if d not in X_norm.columns]
    if missing:
        raise SystemExit(
            f"{len(missing)} top-ANOVA domains absent from the normalized matrix "
            f"(naming mismatch): {missing[:3]} ... - aborting rather than guessing."
        )

    # --- 3. Order species by clade ---
    ordered_species, ordered_clades, block_clades = order_species_by_clade(species_final)

    # Submatrix: rows = top-40 domains, cols = species (clade-ordered). Values are
    # the verbatim relative frequencies from the pipeline.
    relfreq = X_norm.loc[ordered_species, domains].to_numpy().T  # shape (40, n_species)

    # --- 4. Per-domain (per-row) z-score across species ---
    row_mean = relfreq.mean(axis=1, keepdims=True)
    row_std = relfreq.std(axis=1, ddof=0, keepdims=True)
    # All top-40 domains are ANOVA-significant -> variable across species -> std>0.
    if np.any(row_std == 0):
        zero_rows = [domains[i] for i in np.where(row_std.ravel() == 0)[0]]
        raise SystemExit(f"Zero-variance row(s) among top-40 (unexpected): {zero_rows}")
    Z = (relfreq - row_mean) / row_std

    # --- 5. Figure ---
    n_dom, n_sp = Z.shape
    fig_w = pubstyle.DOUBLE_COL
    fig_h = 5.6  # tall enough for 40 row labels at small font
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Symmetric-ish colour limits from the data (no clipping of values; just the
    # display range). Use a robust cap at the 99th percentile of |z| so a single
    # extreme cell does not wash out the rest, but never below 2.5.
    vmax = max(2.5, float(np.percentile(np.abs(Z), 99)))
    im = ax.imshow(Z, cmap="viridis", aspect="auto", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    # Domain (row) labels - accession + name, small font fits 40.
    ax.set_yticks(range(n_dom))
    ax.set_yticklabels([_short_domain(d) for d in domains], fontsize=3.4)
    ax.tick_params(axis="y", length=1.5, pad=1.5)

    # Species columns: too many to label individually; show clade blocks instead.
    ax.set_xticks([])

    # --- thin separator lines between clade blocks + block labels ---
    boundaries: list[int] = []  # right edge (exclusive) of each block
    start = 0
    block_spans: list[tuple[str, int, int]] = []  # (clade, start, end-exclusive)
    for clade in block_clades:
        size = sum(1 for c in ordered_clades if c == clade)
        block_spans.append((clade, start, start + size))
        start += size
        boundaries.append(start)
    # vertical separators between blocks (skip the final right edge)
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="white", lw=0.8)
        ax.axvline(b - 0.5, color="0.15", lw=0.35)

    # Block labels under the heatmap + a coloured swatch strip.
    # Geometry (data coords): matrix bottom edge is at y = n_dom - 0.5. Put a small
    # gap, then the clade colour band, then the rotated block labels below it.
    ax.set_xlim(-0.5, n_sp - 0.5)
    band_top = n_dom - 0.5 + 0.6   # small gap below the matrix
    band_h = 1.1                   # colour-band thickness
    label_y = band_top + band_h    # labels start just below the band
    for clade, s, e in block_spans:
        mid = (s + e - 1) / 2.0
        key = CLADE_COLOR_KEY[clade]
        col = GREY_OTHER if key == "Other" else pubstyle.CLADE_COLORS[key]
        block_label = "Other (n<3)" if key == "Other" else f"{_short_clade(clade)} (n={e - s})"
        # coloured swatch strip just below the matrix (clade colour band)
        ax.add_patch(
            plt.Rectangle(
                (s - 0.5, band_top), e - s, band_h,
                facecolor=col, edgecolor="white", lw=0.4,
                clip_on=False, zorder=3,
            )
        )
        # short rotated label below the colour band
        ax.annotate(
            block_label,
            xy=(mid, label_y),
            xytext=(0, -2),
            textcoords="offset points",
            ha="center", va="top", rotation=90, fontsize=4.4,
            color="0.1", annotation_clip=False,
        )

    # labelpad clears the rotated clade-block labels below the colour band (the long
    # "Laura. Cetartiodactyla (n=23)" label extends well below the band) so the
    # horizontal axis label sits in clear space and never crosses the clade labels.
    ax.set_xlabel(f"Species (n={n_sp}), ordered by clade", fontsize=6, labelpad=92)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.012, extend="both")
    cb.set_label("z-score (per domain)", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    ax.set_title(
        "Top 40 domains (by ANOVA F): per-domain z-score of relative frequency; "
        "species by clade",
        fontsize=8,
    )

    fig.tight_layout()
    pdf, png = pubstyle.save(fig, OUT_BASE)
    plt.close(fig)

    return {
        "pdf": pdf,
        "png": png,
        "Z": Z,
        "relfreq": relfreq,
        "domains": domains,
        "top": top,
        "ordered_species": ordered_species,
        "ordered_clades": ordered_clades,
        "block_spans": block_spans,
        "X_norm": X_norm,
        "species_final": species_final,
    }


def _integrity_checks(res: dict) -> None:
    """Independent re-derivation of every load-bearing transform; aborts on failure."""
    print("=== INTEGRITY CHECKS (part3_domain_heatmap) ===")
    ok = True

    # (1) Top-40 == anova_domains.tsv sorted by F_statistic, ranks 1..40.
    anova = pd.read_csv(ANOVA_TSV, sep="\t")
    ref_top = anova.sort_values("F_statistic", ascending=False).head(N_TOP)
    same_domains = list(ref_top["domain_name"]) == res["domains"]
    f_monotone = bool(np.all(np.diff(res["top"]["F_statistic"].to_numpy()) <= 0))
    rank_ok = list(ref_top["rank"]) == list(range(1, N_TOP + 1))
    print(f"[top40] domains == anova sorted-by-F: {same_domains}; "
          f"F descending: {f_monotone}; ranks 1..40: {rank_ok}")
    ok &= same_domains and f_monotone and rank_ok

    # (2) Relative frequency == count / per-species total. Recompute from RAW counts
    #     for a couple of (species, domain) cells, independently of the matrix used.
    X_norm = res["X_norm"]
    rowsum = X_norm.sum(axis=1)
    rowsum_ok = bool(np.allclose(rowsum.to_numpy(), 1.0, atol=1e-9))
    print(f"[relfreq] every species row of X_norm sums to 1: {rowsum_ok} "
          f"(min={rowsum.min():.10f}, max={rowsum.max():.10f})")
    ok &= rowsum_ok

    # Independent raw recompute: load raw counts, transpose, strip parenthetical,
    # and confirm relfreq[sp, dom] == raw_count / raw_species_total for samples.
    raw = pd.read_csv(
        os.path.join(REPO, "data_raw", "MammalDomainCount.tsv"),
        sep="\t", skiprows=1, index_col=0, low_memory=False,
    ).transpose()
    raw.index = [str(n).split("(")[0].strip() for n in raw.index]
    sample_sp = [res["ordered_species"][0], res["ordered_species"][-1]]
    sample_dom = [res["domains"][0], res["domains"][-1]]
    rf_match = True
    for sp in sample_sp:
        # raw per-species total over the SAME 8020 retained domains (X_norm cols)
        raw_row = raw.loc[sp, X_norm.columns].astype(float)
        total = raw_row.sum()
        for dom in sample_dom:
            expect = float(raw_row[dom]) / float(total)
            got = float(X_norm.loc[sp, dom])
            close = np.isclose(expect, got, rtol=1e-6, atol=1e-12)
            rf_match &= bool(close)
            print(f"[relfreq] {sp[:22]:22s} / {dom.split(':')[0]}: "
                  f"raw {expect:.3e} vs X_norm {got:.3e} -> {close}")
    ok &= rf_match

    # (3) Per-domain z-score: each plotted row has mean ~0, std ~1, and recomputing
    #     z from the (clade-ordered) relfreq submatrix reproduces the plotted Z.
    Z = res["Z"]
    row_means = Z.mean(axis=1)
    row_stds = Z.std(axis=1, ddof=0)
    mean_ok = bool(np.allclose(row_means, 0.0, atol=1e-9))
    std_ok = bool(np.allclose(row_stds, 1.0, atol=1e-9))
    print(f"[zscore] per-row mean~0: {mean_ok} (max|mean|={np.abs(row_means).max():.2e}); "
          f"per-row std~1: {std_ok} (range {row_stds.min():.6f}..{row_stds.max():.6f})")
    relfreq = res["relfreq"]
    Z_re = (relfreq - relfreq.mean(axis=1, keepdims=True)) / relfreq.std(
        axis=1, ddof=0, keepdims=True
    )
    z_recompute_ok = bool(np.allclose(Z_re, Z, atol=1e-12))
    print(f"[zscore] recompute of plotted Z matches: {z_recompute_ok}")
    ok &= mean_ok and std_ok and z_recompute_ok

    # (4) Clade ordering only permutes columns (same multiset of species, every
    #     species mapped, blocks contiguous and in canonical order).
    perm_ok = sorted(res["ordered_species"]) == sorted(res["species_final"])
    blocks = [c for c, _, _ in res["block_spans"]]
    canonical_subseq = [c for c in CLADE_BLOCK_ORDER if c in blocks]
    block_order_ok = blocks == canonical_subseq
    print(f"[clade] columns are a permutation of species_final: {perm_ok}; "
          f"blocks in canonical order: {block_order_ok} ({blocks})")
    ok &= perm_ok and block_order_ok

    print(f"=== ALL INTEGRITY CHECKS {'PASSED' if ok else 'FAILED'} ===")
    if not ok:
        raise SystemExit("Integrity checks FAILED - not emitting the figure as valid.")


def main() -> None:
    res = build_heatmap()
    _integrity_checks(res)
    print("\nWROTE:")
    print(" ", res["pdf"])
    print(" ", res["png"])


if __name__ == "__main__":
    main()
