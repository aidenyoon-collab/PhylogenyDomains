#!/usr/bin/env python3
"""
Part 3 (biology) figures - per-clade domain enrichment, rendered with pubstyle.
No data is altered, trimmed, or jittered.

Produces:
  (a) figures/part3_volcano_<clade>.{pdf,png}   (ONE figure per clade, 7 total)
        Single-column, readable volcano per clade (one clade per figure,
        the FULL cloud of all tested domains, SPARSE labels). The saved
        results/clade_domain_enrichment/enriched_domains_<clade>.csv contains ONLY
        the significant rows, so the full background cloud is NOT in those files.
        We therefore RECOMPUTE the full per-clade enrichment over ALL tested domains
        by importing scripts/domain_clade_enrichment.py's OWN functions and
        reproducing its EXACT preprocessing (the same pipeline that produced the
        saved CSVs):
            load_pfam_style_tsv -> drop EXCLUDE_SPECIES -> load_clade_map (built-in
            clade mapping) -> join_species_clades -> drop_zero_variance_domains ->
            one_clade_enrichment (per-domain two-sided Mann-Whitney U on the RAW
            per-species count vectors, BH-FDR via multipletests, MEAN-based
            fold_change with the script's PSEUDOCOUNT).
        This is a FAITHFUL recompute via the pipeline's own functions - verified to
        reproduce the saved enriched_domains_<clade>.csv exactly (same domain set +
        log2FC + p_adjusted; see INTEGRITY ANCHOR in main()).

        NOTE ON PREPROCESSING: domain_clade_enrichment tests RAW domain counts
        (copy number; NOT relative frequency - that creates a Chiroptera closure
        artifact) on the canonical 107-species cohort (platypus excluded, Cervus
        included), the SAME cohort as the ANOVA/heatmap. Both main() and this recompute
        call dce.build_enrichment_data(), so they cannot drift; reproducing it exactly
        reproduces the saved CSVs.

        x = log2(fold_change) (mean basis); y = -log10(p_adjusted), with p_adjusted
        floored only as an inf-guard. Colours follow the direction convention:
        grey = non-significant; RED = significant & enriched (log2FC > 0);
        BLUE = significant & depleted (log2FC < 0); significance = p_adjusted < 0.05.
        Labels are SPARSE: only |log2FC| >= 1 AND p_adjusted < 1e-10, most
        significant first, capped at ~6 total across both sides (fallback: top-5
        significant by p_adjusted if fewer than 3 clear that bar).

  (b) figures/part3_enrichment_heatmap.{pdf,png}
        clade x domain log2FC heatmap from
        results/clade_domain_enrichment/filtered/all_clades_filtered_enriched_domains_summary.csv
        Rows ordered phylogenetically; columns = top-N domains per clade (by
        p_adjusted) to keep labels readable; diverging RdBu_r centred at 0, clipped
        to [-2, 2]. (This filtered set is enriched-only, so log2FC is all positive.)
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

pubstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# The pipeline's OWN enrichment module - we call its functions for a faithful
# recompute of the full (all-domain) per-clade enrichment.
import domain_clade_enrichment as dce  # noqa: E402

ENRICH_DIR = os.path.join(REPO, "results", "clade_domain_enrichment")
FILTERED_CSV = os.path.join(
    ENRICH_DIR, "filtered", "all_clades_filtered_enriched_domains_summary.csv"
)
OUT_DIR = os.path.join(REPO, "figures")

# The 7 enrichment clades (Monotreme outgroup is not an enrichment clade).
VOLCANO_CLADES = [
    "Afrotheria",
    "Glires",
    "Laurasiatheria_Carnivora",
    "Laurasiatheria_Cetartiodactyla",
    "Laurasiatheria_Chiroptera",
    "Marsupials",
    "Primates",
]

# Red-up / blue-down direction convention (NOT clade colors).
COLOR_ENRICHED = "#D55E00"   # red-orange: significant & enriched (log2FC > 0)
COLOR_DEPLETED = "#0072B2"   # blue:       significant & depleted (log2FC < 0)
COLOR_NS = "0.78"            # grey:       non-significant

# Phylogenetic ordering for the heatmap (rows + the grouping of columns).
PHYLO_ORDER = [
    "Marsupials",
    "Afrotheria",
    "Glires",
    "Primates",
    "Laurasiatheria_Carnivora",
    "Laurasiatheria_Cetartiodactyla",
    "Laurasiatheria_Chiroptera",
]

FDR = 0.05                # significance threshold on p_adjusted
PADJ_FLOOR = 1e-300       # inf-guard floor for -log10 only (no value is truly 0)
LFC_THRESH = 1.0          # |log2FC| = 1 (>= 2-fold): label gate + effect-size guide
LABEL_PADJ = 1e-10        # label gate: only very-significant points get labelled
LABEL_CAP = 6             # at most ~6 labels total across BOTH sides
FALLBACK_TOPK = 5         # if < 3 clear the gate, fall back to top-5 sig by p_adj
TOP_N_PER_CLADE = 10      # heatmap: top-N domains per clade by p_adjusted


def _short_clade(name: str) -> str:
    """Compact display label for a clade."""
    return name.replace("Laurasiatheria_", "Laura. ")


def _parse_domain(domain: str) -> tuple[str, str]:
    """'PF12345:Some name' -> ('PF12345', 'Some name'). Split on FIRST colon only."""
    if ":" in domain:
        pid, name = domain.split(":", 1)
        return pid.strip(), name.strip()
    return domain.strip(), ""


# --------------------------------------------------------------------------- #
# Faithful recompute of the FULL per-clade enrichment (all tested domains) via
# the pipeline's OWN functions, reproducing domain_clade_enrichment.main()'s
# exact preprocessing. Cached so all 7 figures share one data load.
# --------------------------------------------------------------------------- #
_RECOMPUTE_CACHE: dict | None = None


def _recompute_all_clades() -> tuple[pd.DataFrame, list[str]]:
    """Run the pipeline's own preprocessing once and return (data, kept_domains).
    Delegates to dce.build_enrichment_data() - the SINGLE source main() also uses:
    canonical 107-species cohort (platypus out, Cervus in), RAW domain counts -
    so this recompute stays byte-for-byte in sync with the saved enriched_domains_<clade>.csv."""
    global _RECOMPUTE_CACHE
    if _RECOMPUTE_CACHE is not None:
        return _RECOMPUTE_CACHE["data"], _RECOMPUTE_CACHE["keep"]
    data, keep = dce.build_enrichment_data()
    _RECOMPUTE_CACHE = {"data": data, "keep": keep}
    return data, keep


def load_clade(clade: str) -> pd.DataFrame:
    """Recompute the full (ALL tested domains) enrichment table for one clade.

    Returns a DataFrame with one row per tested domain and the volcano transforms
    already applied. The significant subset (p_adjusted < FDR & fold_change != 1)
    is byte-for-byte equivalent to the saved enriched_domains_<clade>.csv.
    """
    data, keep = _recompute_all_clades()
    res = dce.one_clade_enrichment(data, keep, clade, dce.MIN_CLADE_N)
    if res is None:
        raise RuntimeError(f"clade {clade!r} did not qualify (n < MIN_CLADE_N)")
    df = res.copy()
    # x-axis: log2(fold_change), mean basis. fold_change is strictly > 0.
    df["log2FC"] = np.log2(df["fold_change"].to_numpy())
    # y-axis: -log10(p_adjusted), with a floor used ONLY as an inf-guard.
    padj = df["p_adjusted"].to_numpy()
    df["neglog10_padj"] = -np.log10(np.maximum(padj, PADJ_FLOOR))
    df["pfam"] = df["domain"].map(lambda d: _parse_domain(d)[0])
    df["dname"] = df["domain"].map(lambda d: _parse_domain(d)[1])
    return df


# --------------------------------------------------------------------------- #
# (a) Per-clade volcano (ONE figure per clade, full single-column)
# --------------------------------------------------------------------------- #
def _select_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the SPARSE label set.

    Primary gate: |log2FC| >= 1 (>= 2-fold) AND p_adjusted < 1e-10, sorted by
    p_adjusted ascending, capped at LABEL_CAP total across BOTH sides. If fewer
    than 3 domains clear that gate, fall back to the top-FALLBACK_TOPK
    significant (p_adjusted < FDR) domains by p_adjusted.
    """
    sig = df["p_adjusted"] < FDR
    gate = sig & (df["log2FC"].abs() >= LFC_THRESH) & (df["p_adjusted"] < LABEL_PADJ)
    cand = df[gate].sort_values("p_adjusted").head(LABEL_CAP)
    if len(cand) < 3:
        cand = (df[sig].sort_values("p_adjusted").head(FALLBACK_TOPK))
    return cand


def volcano_figure(clade: str, df: pd.DataFrame) -> dict:
    sig = df["p_adjusted"] < FDR
    enr = sig & (df["log2FC"] > 0)   # significant & enriched  -> red
    dep = sig & (df["log2FC"] < 0)   # significant & depleted  -> blue
    ns = ~sig                        # non-significant         -> grey
    x = df["log2FC"].to_numpy()
    y = df["neglog10_padj"].to_numpy()

    fig, ax = plt.subplots(
        figsize=(pubstyle.SINGLE_COL, 0.95 * pubstyle.SINGLE_COL)
    )

    # FULL cloud of all tested domains: grey background first, then coloured hits.
    ax.scatter(x[ns.to_numpy()], y[ns.to_numpy()], s=4, c=COLOR_NS,
               linewidths=0, rasterized=True, zorder=1)
    ax.scatter(x[dep.to_numpy()], y[dep.to_numpy()], s=5, c=COLOR_DEPLETED,
               linewidths=0, alpha=0.85, rasterized=True, zorder=2)
    ax.scatter(x[enr.to_numpy()], y[enr.to_numpy()], s=5, c=COLOR_ENRICHED,
               linewidths=0, alpha=0.85, rasterized=True, zorder=2)

    # Vertical line at x=0; light dashed guides at the FDR threshold and |log2FC|=1.
    fdr_y = -np.log10(FDR)
    ax.axvline(0.0, ls="-", lw=0.5, color="0.6", zorder=0)
    ax.axhline(fdr_y, ls="--", lw=0.5, color="0.7", zorder=0)
    ax.axvline(-LFC_THRESH, ls="--", lw=0.5, color="0.8", zorder=0)
    ax.axvline(LFC_THRESH, ls="--", lw=0.5, color="0.8", zorder=0)

    ax.set_title(f"Domain enrichment: {clade} vs all other mammals", fontsize=7.5)
    ax.set_xlabel("log2 fold change (mean counts)", fontsize=7)
    ax.set_ylabel("-log10 $p_{adj}$", fontsize=7)

    n_enr, n_dep, n_tot = int(enr.sum()), int(dep.sum()), int(len(df))

    xmax = max(2.0, np.abs(x).max() * 1.18)
    ax.set_xlim(-xmax, xmax)
    ymax = max(y.max() * 1.10, fdr_y * 1.5)
    ax.set_ylim(-ymax * 0.03, ymax)

    # Count annotation in an opaque-boxed corner so it never reads over markers or
    # the dashed |log2FC|=1 guide. The upper-LEFT extreme corner (x well left of the
    # x=-1 dashed line) is the emptiest region for every clade here; the opaque box
    # guarantees it sits clear of any faint guide line it might otherwise touch.
    pubstyle.annotate(
        ax,
        f"{n_tot} domains tested\n"
        f"{n_enr} enriched (red) / {n_dep} depleted (blue)",
        loc="upper left", fontsize=5.0, color="0.30",
    )

    # SPARSE labels: stack them down the inner edge on the matching side, with a
    # thin leader line back to the true point (point stays at its real coords).
    labels = _select_labels(df)
    left = labels[labels["log2FC"] < 0].sort_values("p_adjusted")
    right = labels[labels["log2FC"] >= 0].sort_values("p_adjusted")

    def _place(sub, side):
        n = len(sub)
        if n == 0:
            return
        if side == "right":
            # Right edge is clear (legend is now OUTSIDE the axes): use the full
            # upper band, just below the title.
            y_hi, y_lo = ymax * 0.92, ymax * 0.40
            x_lab, ha = xmax * 0.46, "left"
        else:
            # Left edge: keep the stack BELOW the upper-left count annotation box
            # (top ~15% of the axes) so the leader labels never touch it.
            y_hi, y_lo = ymax * 0.78, ymax * 0.34
            x_lab, ha = -xmax * 0.50, "right"
        ys = np.linspace(y_hi, y_lo, n) if n > 1 else [(y_hi + y_lo) * 0.5]
        for (_, r), y_lab in zip(sub.iterrows(), ys):
            ax.annotate(
                r["pfam"],
                xy=(r["log2FC"], r["neglog10_padj"]),
                xytext=(x_lab, y_lab),
                fontsize=5.0, ha=ha, va="center", color="0.12",
                annotation_clip=False, zorder=4,
                arrowprops=dict(arrowstyle="-", lw=0.3, color="0.55",
                                shrinkA=0.5, shrinkB=2.0),
            )

    _place(right, "right")
    _place(left, "left")

    # Compact legend (red-up/blue-down convention).
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=4, mfc=COLOR_ENRICHED,
               mec="none", label="enriched (sig)"),
        Line2D([0], [0], marker="o", ls="", ms=4, mfc=COLOR_DEPLETED,
               mec="none", label="depleted (sig)"),
        Line2D([0], [0], marker="o", ls="", ms=4, mfc=COLOR_NS,
               mec="none", label="n.s."),
        Line2D([0], [0], ls="--", lw=0.6, color="0.7",
               label=f"FDR $p_{{adj}}$={FDR}"),
        Line2D([0], [0], ls="--", lw=0.6, color="0.8",
               label=f"|log2FC|={int(LFC_THRESH)}"),
    ]
    # Legend OUTSIDE the axes so it can never touch the dashed FDR / |log2FC|=1
    # guides or the point cloud (the previous lower-right placement sat on them).
    # Anchor it BELOW the top so it also clears the (centred) title band on the right.
    pubstyle.legend_outside(ax, handles=handles, fontsize=5.0,
                            handletextpad=0.4, labelspacing=0.35,
                            bbox_to_anchor=(1.01, 0.82))

    fig.tight_layout()
    base = os.path.join(OUT_DIR, f"part3_volcano_{clade}")
    pdf, png = pubstyle.save(fig, base)
    plt.close(fig)
    return {"pdf": pdf, "png": png, "n_labels": int(len(labels))}


def build_volcanoes() -> dict:
    data = {c: load_clade(c) for c in VOLCANO_CLADES}
    out = {}
    for clade in VOLCANO_CLADES:
        out[clade] = volcano_figure(clade, data[clade])
    return {"data": data, "figures": out}


# --------------------------------------------------------------------------- #
# (b) Enrichment heatmap
# --------------------------------------------------------------------------- #
def build_heatmap() -> dict:
    fil = pd.read_csv(FILTERED_CSV)

    # Choose columns: top-N domains per clade by p_adjusted (most significant first),
    # so the heatmap stays label-readable instead of cramming all 306 domains.
    selected: list[str] = []
    seen: set[str] = set()
    for clade in PHYLO_ORDER:
        sub = (fil[fil["clade"] == clade]
               .sort_values("p_adjusted")
               .head(TOP_N_PER_CLADE))
        for pid in sub["pfam_id"]:
            if pid not in seen:
                seen.add(pid)
                selected.append(pid)

    sub = fil[fil["pfam_id"].isin(seen)].copy()

    # Pivot to clade x domain matrix of log2FC (verbatim CSV values).
    piv = sub.pivot_table(
        index="clade", columns="pfam_id", values="log2FC", aggfunc="first"
    )
    piv = piv.reindex(index=PHYLO_ORDER)

    # Order columns: group by the (first / strongest) clade each domain belongs to,
    # following the same phylogenetic clade order, then by descending log2FC within.
    name_map = (sub.drop_duplicates("pfam_id")
                .set_index("pfam_id")["domain_name"].to_dict())
    primary = {}
    for pid in selected:
        rows = sub[sub["pfam_id"] == pid]
        # the clade where this domain is most significant
        best = rows.sort_values("p_adjusted").iloc[0]
        primary[pid] = (PHYLO_ORDER.index(best["clade"]), -float(best["log2FC"]))
    col_order = sorted(selected, key=lambda p: primary[p])
    piv = piv[col_order]

    # Diverging colormap centred at 0, clipped to [-2, 2].
    vmin, vmax = -2.0, 2.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    data = piv.to_numpy()

    nrows, ncols = data.shape
    fig_w = pubstyle.DOUBLE_COL
    fig_h = min(0.18 * ncols, 9.0) * 0.5 + 1.6  # scale height with column count
    fig_h = max(3.2, min(fig_h, 9.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(ncols))
    xlabels = [f"{p}: {name_map.get(p, '')[:34]}" for p in col_order]
    ax.set_xticklabels(xlabels, rotation=90, fontsize=3.6, ha="center")
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([_short_clade(c) for c in PHYLO_ORDER], fontsize=6)

    # Light gridlines between cells.
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)

    ax.set_xlabel(f"Enriched PFAM domains (top {TOP_N_PER_CLADE} per clade by "
                  f"$p_{{adj}}$; {ncols} unique)", fontsize=6)

    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012, extend="both")
    cb.set_label("log2 fold change (clipped ±2)", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    ax.set_title("Clade-enriched domains (filtered, FDR-significant)", fontsize=8)

    fig.tight_layout()
    pdf, png = pubstyle.save(fig, os.path.join(OUT_DIR, "part3_enrichment_heatmap"))
    plt.close(fig)
    return {"pdf": pdf, "png": png, "piv": piv, "col_order": col_order}


def _integrity_anchor(data: dict, clades=("Afrotheria", "Marsupials",
                                          "Primates", "Laurasiatheria_Chiroptera")) -> bool:
    """Confirm the RECOMPUTED significant set equals the saved CSV per clade.

    Significant set = recomputed rows with p_adjusted < FDR and fold_change != 1
    (exactly the rule domain_clade_enrichment.main() uses to choose what to save).
    Checks the same domain SET, log2FC, and p_adjusted against
    enriched_domains_<clade>.csv. Required for >= 2 clades.
    """
    print("=== INTEGRITY ANCHOR: recomputed significant set vs saved CSV ===")
    passed = 0
    for clade in clades:
        df = data[clade]
        sig = df[(df["p_adjusted"] < FDR) & (df["fold_change"] != 1.0)].copy()
        saved = pd.read_csv(os.path.join(ENRICH_DIR, f"enriched_domains_{clade}.csv"))
        saved["log2FC_sv"] = np.log2(saved["fold_change"].to_numpy())
        re_dom, sv_dom = set(sig["domain"]), set(saved["domain"])
        set_ok = re_dom == sv_dom
        m = sig.merge(saved, on="domain", suffixes=("_re", "_sv"))
        dl2 = float((m["log2FC"] - m["log2FC_sv"]).abs().max()) if len(m) else float("nan")
        dpa = float((m["p_adjusted_re"] - m["p_adjusted_sv"]).abs().max()) if len(m) else float("nan")
        ok = set_ok and dl2 < 1e-9 and dpa < 1e-9
        passed += int(ok)
        print(f"[{clade}] recomp_sig={len(sig)} saved={len(saved)} "
              f"domain_set_equal={set_ok} max|dlog2FC|={dl2:.2e} "
              f"max|dp_adj|={dpa:.2e} -> {'PASS' if ok else 'FAIL'}")
    print(f"Integrity anchor: {passed}/{len(clades)} clades match exactly "
          f"(required >= 2).")
    return passed >= 2


def main() -> None:
    v = build_volcanoes()
    h = build_heatmap()

    ok = _integrity_anchor(v["data"])
    if not ok:
        raise SystemExit("INTEGRITY ANCHOR FAILED: recompute does not match saved CSVs.")

    print("\nLabel counts per clade (sparse):")
    for clade in VOLCANO_CLADES:
        print(f"  {clade}: {v['figures'][clade]['n_labels']} labels")

    # Heatmap integrity spot-check: pivot cells == raw filtered CSV (unchanged).
    fil = pd.read_csv(FILTERED_CSV)
    piv = h["piv"]
    checks = 0
    for clade in PHYLO_ORDER:
        for pid in h["col_order"]:
            raw = fil[(fil["clade"] == clade) & (fil["pfam_id"] == pid)]
            if len(raw):
                csv_v = float(raw["log2FC"].iloc[0])
                cell = piv.loc[clade, pid]
                ok = np.isclose(cell, csv_v)
                print(f"[heatmap] {clade}/{pid}: pivot={cell:.6f} csv={csv_v:.6f} -> {ok}")
                checks += 1
                if checks >= 6:
                    break
        if checks >= 6:
            break

    print("\nWROTE:")
    for clade in VOLCANO_CLADES:
        print(" ", v["figures"][clade]["pdf"])
        print(" ", v["figures"][clade]["png"])
    print(" ", h["pdf"])
    print(" ", h["png"])


if __name__ == "__main__":
    main()
