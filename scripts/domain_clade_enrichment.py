#!/usr/bin/env python3
"""
One-vs-rest domain count enrichment by mammal clade (Mann-Whitney U, BH-FDR).

Loads a PFAM-style TSV (domains as rows, species as columns), transposes to
species x domains, joins clade labels, and tests each clade against all others.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# Paths (edit these)
# -----------------------------------------------------------------------------

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

# Native MammalDomainCount-style TSV: row 0 = #Domains metadata; row 1 = species headers; body = domain counts
INPUT_TSV_PATH = os.path.join(PROJECT_ROOT, "data_raw", "MammalDomainCount.tsv")
# If USE_BUILTIN_CLADES is False, set path to a two-column CSV: species, clade
CLADE_MAP_PATH: Optional[str] = None
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "clade_domain_enrichment")

# Analysis parameters
USE_BUILTIN_CLADES = True
METADATA_SKIPROWS = 1  # skip "#Domains ..." line in project TSV
MIN_CLADE_N = 3
FDR_ALPHA = 0.05
# For small focal clades the normal-approx Mann-Whitney (scipy 'auto' under ties)
# reports p below the EXACT achievable floor 2/C(n1+n2, n1) - e.g. n=4 vs 103 floor is
# 3.88e-7, yet asymptotic returns p_adj ~6e-22, which is mathematically impossible and
# inflates significance. Use the EXACT null when the focal group is this small.
MWU_EXACT_MAX_N = 8
PSEUDOCOUNT = 1.0
# Volcano: avoid infinite -log10 when p_adj == 0
P_ADJ_FLOOR = 1e-300
# Volcano labels: |log2FC| magnitude past which a significant point is "large-effect"
# (matches the downstream LOG2FC_THRESHOLD in filter_clade_enrichment_secondary.py).
LOG2FC_LABEL_THRESHOLD = 0.5
# Max labeled points per corner (enriched/right and depleted/left), most significant first.
N_VOLCANO_LABELS = 10
# Note: no longer used by main() - the cohort is now defined by
# build_enrichment_data() (the canonical 107-species set, which INCLUDES Cervus
# hanglu, matching the ANOVA/heatmap). Kept only for the legacy load path.
EXCLUDE_SPECIES: Tuple[str, ...] = (
    "Cervus hanglu yarkandensis",
    "Cervus_hanglu_yarkandensis",
)


def log(msg: str) -> None:
    print(f"[domain_clade_enrichment] {msg}")


def clean_species_name(name: str) -> str:
    """Strip parenthetical common names / database noise (match domain_time_scatter)."""
    if not isinstance(name, str):
        name = str(name)
    return name.split("(")[0].strip()


def safe_filename(label: str) -> str:
    """Sanitize clade (or other) labels for use in file names."""
    s = re.sub(r"[^\w\-.]+", "_", str(label).strip(), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "clade"


def load_pfam_style_tsv(path: str, metadata_skiprows: int = 1) -> pd.DataFrame:
    """
    Load domains x species matrix, return species x domains.
    First column = domain IDs (index), remaining columns = species, values = counts.
    """
    log(f"Loading PFAM-style TSV: {path}")
    raw = pd.read_csv(
        path,
        sep="\t",
        skiprows=metadata_skiprows,
        index_col=0,
        low_memory=False,
    )
    # Transpose: rows = species, columns = domain IDs
    x = raw.transpose()
    x.index = [clean_species_name(s) for s in x.index]
    x.index.name = "species"
    if x.index.duplicated().any():
        log("Warning: duplicate species after cleaning; averaging rows")
        x = x.groupby(x.index).mean()
    return x


def load_clade_map() -> pd.DataFrame:
    """Return DataFrame with columns ['species', 'clade'] (species names cleaned)."""
    if USE_BUILTIN_CLADES or not (CLADE_MAP_PATH and os.path.isfile(CLADE_MAP_PATH)):
        sys.path.insert(0, SCRIPTS_DIR)
        from elastic_net_regression_stratified_species_cv import (  # type: ignore
            build_clade_mapping,
        )

        m: Dict[str, str] = build_clade_mapping()
        df = pd.DataFrame(
            [(clean_species_name(s), c) for s, c in m.items()],
            columns=["species", "clade"],
        )
        log(
            "Using built-in clade mapping from "
            "elastic_net_regression_stratified_species_cv"
        )
    else:
        log(f"Loading clade map from {CLADE_MAP_PATH}")
        df = pd.read_csv(CLADE_MAP_PATH)
        if "species" not in df.columns or "clade" not in df.columns:
            raise ValueError("Clade CSV must have columns 'species' and 'clade'")
        df = df.copy()
        df["species"] = df["species"].map(clean_species_name)
    if df["species"].duplicated().any():
        log("Warning: duplicate species in clade table; keeping first occurrence")
        df = df.drop_duplicates(subset=["species"], keep="first")
    return df


def join_species_clades(
    x: pd.DataFrame, clades: pd.DataFrame
) -> Tuple[pd.DataFrame, int, int, int, int]:
    """
    Inner-join matrix rows to clade labels.
    Returns (df, n_matched, n_drop_no_clade, n_clade_unmatched, n_matrix_unmatched).
    """
    n_species_x = x.shape[0]
    merged = x.reset_index().merge(clades, on="species", how="inner")
    n_matched = merged.shape[0]
    n_drop = n_species_x - n_matched
    in_x = set(x.index)
    in_c = set(clades["species"])
    n_clade_unmatched = len(in_c - in_x)
    n_matrix_unmatched = len(in_x - in_c)
    return merged, n_matched, n_drop, n_clade_unmatched, n_matrix_unmatched


def drop_zero_variance_domains(x: pd.DataFrame, domain_cols: List[str]) -> List[str]:
    v = x[domain_cols].var(axis=0, ddof=0)
    keep = v[v > 0].index.tolist()
    n_drop = len(domain_cols) - len(keep)
    if n_drop:
        log(f"Dropping {n_drop} zero-variance domain columns (no information)")
    return keep


def mann_whitney_p(
    focal: np.ndarray, background: np.ndarray
) -> Tuple[float, float]:
    """
    Two-sided Mann-Whitney U; return (U_stat, p). On failure, (nan, 1.0).
    """
    focal = np.asarray(focal, dtype=float)
    background = np.asarray(background, dtype=float)
    if focal.size < 1 or background.size < 1:
        return float("nan"), 1.0
    # EXACT null for small focal clades: caps p at what the rank test can actually
    # produce; the normal approximation otherwise reports impossibly-small p for n=4.
    method = "exact" if min(focal.size, background.size) <= MWU_EXACT_MAX_N else "asymptotic"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u, p = mannwhitneyu(
                focal, background, alternative="two-sided", method=method
            )
            if not np.isfinite(p):
                return float(u), 1.0
            return float(u), float(p)
        except (ValueError, TypeError):
            return float("nan"), 1.0


def one_clade_enrichment(
    data: pd.DataFrame,
    domain_cols: List[str],
    clade: str,
    min_clade_n: int,
) -> Optional[pd.DataFrame]:
    """One-vs-rest test for a single clade. Returns a table for all domains."""
    focal_idx = data["clade"] == clade
    n_f = focal_idx.sum()
    if n_f < min_clade_n:
        log(
            f"  Skipping clade '{clade}': n={n_f} < min_clade_n={min_clade_n}"
        )
        return None
    if (data.shape[0] - n_f) < 1:
        log(f"  Skipping clade '{clade}': no background species")
        return None

    rows = []
    for d in domain_cols:
        a = data.loc[focal_idx, d].values
        b = data.loc[~focal_idx, d].values
        u_stat, p_val = mann_whitney_p(a, b)
        # Fold-change basis = MEAN counts (log2 fold change of mean counts; matches
        # domain_clade_interpretation.marker_domains_analysis).
        # Pseudocount retained to avoid divide-by-zero / infinite log2FC.
        mean_f = float(np.mean(a))
        mean_b = float(np.mean(b))
        fc = (mean_f + PSEUDOCOUNT) / (mean_b + PSEUDOCOUNT)
        rows.append(
            {
                "domain": d,
                "mean_focal": mean_f,
                "mean_background": mean_b,
                "fold_change": fc,
                "U_statistic": u_stat,
                "p_value": p_val,
            }
        )
    res = pd.DataFrame(rows)
    pvals = res["p_value"].values
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    pvals = np.clip(pvals, 1e-300, 1.0)
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh", alpha=FDR_ALPHA)
    res["p_adjusted"] = p_adj
    return res


def plot_volcano(
    res: pd.DataFrame,
    clade: str,
    out_path: str,
) -> None:
    """Volcano: all tested domains, colored by significant direction.

    Labels the significant-AND-large-effect "corners" on BOTH sides (standard
    volcano convention: label enriched and depleted hits):
    points with p_adjusted < FDR_ALPHA AND |log2FC| past LOG2FC_LABEL_THRESHOLD.
    Within each side, label the most-significant points (lowest p_adjusted).
    """
    res = res.copy()
    res["log2_fc"] = np.log2(
        np.maximum(res["fold_change"], np.finfo(float).tiny)
    )
    p_plot = res["p_adjusted"].values.copy()
    p_plot = np.maximum(p_plot, P_ADJ_FLOOR)
    y = -np.log10(p_plot)
    sig_hi = (res["p_adjusted"] < FDR_ALPHA) & (res["fold_change"] > 1.0)
    sig_lo = (res["p_adjusted"] < FDR_ALPHA) & (res["fold_change"] < 1.0)
    colors = np.where(sig_hi, "red", np.where(sig_lo, "blue", "gray"))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        res["log2_fc"],
        y,
        c=colors,
        s=10,
        alpha=0.6,
        edgecolors="none",
    )

    def _annotate(sub: pd.DataFrame, ha: str) -> None:
        for _, r in sub.iterrows():
            px = np.log2(max(r["fold_change"], np.finfo(float).tiny))
            ppy = -np.log10(max(r["p_adjusted"], P_ADJ_FLOOR))
            ax.annotate(
                str(r["domain"])[:50],
                (px, ppy),
                fontsize=6,
                alpha=0.85,
                ha=ha,
            )

    # Large-effect corner masks: significant + magnitude past threshold, both sides.
    sig_hi_large = sig_hi & (res["log2_fc"] >= LOG2FC_LABEL_THRESHOLD)
    sig_lo_large = sig_lo & (res["log2_fc"] <= -LOG2FC_LABEL_THRESHOLD)
    # Most-significant first within each corner (lowest p_adjusted).
    top_enriched = (
        res[sig_hi_large].sort_values("p_adjusted", ascending=True).head(N_VOLCANO_LABELS)
    )
    top_depleted = (
        res[sig_lo_large].sort_values("p_adjusted", ascending=True).head(N_VOLCANO_LABELS)
    )
    _annotate(top_enriched, ha="left")
    _annotate(top_depleted, ha="right")

    ax.axvline(
        LOG2FC_LABEL_THRESHOLD, color="k", linewidth=0.5, alpha=0.2, linestyle="--"
    )
    ax.axvline(
        -LOG2FC_LABEL_THRESHOLD, color="k", linewidth=0.5, alpha=0.2, linestyle="--"
    )
    ax.set_xlabel("log2(fold change)", fontsize=11)
    ax.set_ylabel(r"$-\log_{10}$(adjusted p-value)", fontsize=11)
    ax.set_title(
        f"Domain Enrichment: {clade} vs. All Other Mammals", fontsize=12
    )
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cross_clade_dotplot(
    all_enriched: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Top 20 domains by number of clades in which they are enriched; one point per
    (clade, domain) in the enriched set.
    """
    if all_enriched.empty:
        log("No enriched domains; skipping cross-clade dot plot")
        return
    counts = all_enriched.groupby("domain")["clade"].nunique().sort_values(ascending=False)
    top20 = counts.head(20).index.tolist()
    sub = all_enriched[all_enriched["domain"].isin(top20)].copy()
    sub["neglog10_p"] = -np.log10(sub["p_adjusted"].clip(lower=P_ADJ_FLOOR))
    clades = sorted(sub["clade"].unique())
    d_to_y = {d: i for i, d in enumerate(top20)}
    c_to_x = {c: i for i, c in enumerate(clades)}
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(clades)), 10))
    sc = ax.scatter(
        [c_to_x[c] for c in sub["clade"]],
        [d_to_y[d] for d in sub["domain"]],
        s=np.clip(sub["fold_change"].values * 8, 20, 400),
        c=sub["neglog10_p"].values,
        cmap="viridis",
        alpha=0.85,
        edgecolors="k",
        linewidths=0.2,
    )
    ax.set_xticks(range(len(clades)))
    ax.set_xticklabels(clades, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20, fontsize=7)
    ax.set_xlabel("clade", fontsize=11)
    ax.set_ylabel("domain (top 20 by recurrence in enriched set)", fontsize=10)
    plt.colorbar(sc, ax=ax, label=r"$-\log_{10}$(adjusted p-value)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_enrichment_data() -> Tuple[pd.DataFrame, List[str]]:
    """Canonical Part-3 cohort, ABSOLUTE (raw) domain counts.

    Uses the SAME 107-species cohort and 8,020-domain set as the ANOVA / boxplots /
    heatmap (`load_and_preprocess_data`: platypus excluded, Cervus hanglu INCLUDED),
    but tests RAW copy number - domain enrichment/expansion is a copy-number question.
    (Relative frequency was rejected: dividing each species by its total domain count
    inflates ALL relative frequencies for smaller-proteome clades - e.g. Chiroptera,
    ~5% fewer total domains - manufacturing thousands of spurious 'enriched' hits, a
    compositional closure artifact. CLR is likewise flag-happy and hard to interpret.)
    Clades from `assign_species_to_clades`. Headline 'enriched' set = the effect-size-
    filtered subset (FDR & |log2FC| >= 0.5; filter_clade_enrichment_secondary.py).
    Returns (data, keep_domains): columns 'species','clade' + raw-count domain columns."""
    sys.path.insert(0, SCRIPTS_DIR)
    from classification_cv_tree_reconstruction import load_and_preprocess_data
    from domain_clade_interpretation import assign_species_to_clades
    X_norm, _dist, _species = load_and_preprocess_data()            # canonical cohort + domain set
    raw = load_pfam_style_tsv(INPUT_TSV_PATH, metadata_skiprows=METADATA_SKIPROWS)
    if any(s not in raw.index for s in X_norm.index) or any(d not in raw.columns for d in X_norm.columns):
        raise SystemExit("Canonical cohort/domain names not found in raw matrix - name mismatch.")
    data = raw.loc[list(X_norm.index), list(X_norm.columns)].copy()  # 107 x 8020, RAW counts
    s2c = assign_species_to_clades(list(data.index))
    data.insert(0, "clade", [s2c[s] for s in data.index])
    data.insert(0, "species", list(data.index))
    data = data.reset_index(drop=True)
    keep_domains = [c for c in data.columns if c not in ("species", "clade")]
    keep_domains = drop_zero_variance_domains(data, keep_domains)
    log(f"Enrichment cohort (canonical 107, RAW counts): {data.shape[0]} species, "
        f"{len(keep_domains)} domains; clade n: "
        f"{dict(sorted(data['clade'].value_counts().items()))}")
    return data, keep_domains


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # One consistent Part-3 cohort (canonical 107; platypus out, Cervus in);
    # RAW counts (relative-frequency rejected - Chiroptera closure artifact).
    data, keep_domains = build_enrichment_data()
    clade_n = data.groupby("clade")["species"].nunique().to_dict()
    qualifying = [c for c, n in clade_n.items() if n >= MIN_CLADE_N]
    for c, n in sorted(clade_n.items(), key=lambda x: x[0]):
        if n < MIN_CLADE_N:
            log(
                f"Will skip clade '{c}': n={n} < {MIN_CLADE_N} (not enough species)"
            )

    all_significant_list: List[pd.DataFrame] = []
    count_summary: List[Dict[str, object]] = []
    for clade in sorted(qualifying):
        log(f"Processing clade: {clade} ...")
        res = one_clade_enrichment(data, keep_domains, clade, MIN_CLADE_N)
        if res is None:
            count_summary.append({"clade": clade, "n_enriched": 0})
            continue
        res_out = res.copy()
        out_csv = os.path.join(
            OUTPUT_DIR, f"enriched_domains_{safe_filename(clade)}.csv"
        )
        # Retain BOTH significant directions (examine contractions/depletions,
        # not only expansions). Direction = sign of the
        # fold change; significance = FDR. fold_change == 1.0 carries no direction.
        sig = res_out[res_out["p_adjusted"] < FDR_ALPHA]
        significant = sig[sig["fold_change"] != 1.0].sort_values(
            "fold_change", ascending=False
        )
        n_enriched = int((significant["fold_change"] > 1.0).sum())
        n_depleted = int((significant["fold_change"] < 1.0).sum())
        count_summary.append(
            {
                "clade": clade,
                "n_enriched": n_enriched,
                "n_depleted": n_depleted,
                "n_significant": len(significant),
            }
        )
        to_save = significant[
            [
                "domain",
                "mean_focal",
                "mean_background",
                "fold_change",
                "U_statistic",
                "p_value",
                "p_adjusted",
            ]
        ]
        to_save.to_csv(out_csv, index=False)
        log(
            f"  Wrote {len(to_save)} significant domains "
            f"({n_enriched} enriched, {n_depleted} depleted) -> {out_csv}"
        )

        vpath = os.path.join(OUTPUT_DIR, f"volcano_{safe_filename(clade)}.png")
        plot_volcano(res_out, clade, vpath)
        log(f"  Volcano -> {vpath}")

        edf = to_save.copy()
        edf.insert(0, "clade", clade)
        all_significant_list.append(edf)

    if all_significant_list:
        summary_df = pd.concat(all_significant_list, ignore_index=True)
        summ_path = os.path.join(OUTPUT_DIR, "all_clades_enriched_domains_summary.csv")
        summary_df.to_csv(summ_path, index=False)
        log(f"Wrote master summary (enriched + depleted): {summ_path}")
    else:
        summary_df = pd.DataFrame()
        log("No clade produced significant output")

    ssum = pd.DataFrame(count_summary)
    ssum_path = os.path.join(OUTPUT_DIR, "enriched_domain_counts_by_clade.csv")
    ssum.to_csv(ssum_path, index=False)
    log("Significant domain counts per clade:\n" + ssum.to_string(index=False))
    log(f"Wrote {ssum_path}")

    # Cross-clade dot plot is an ENRICHED-only view (bubble size = fold_change > 1);
    # restrict to the enriched direction so its semantics are unchanged.
    if not summary_df.empty:
        enriched_only = summary_df[summary_df["fold_change"] > 1.0].copy()
    else:
        enriched_only = summary_df
    cross_path = os.path.join(OUTPUT_DIR, "cross_clade_enrichment_dotplot.png")
    plot_cross_clade_dotplot(enriched_only, cross_path)
    if not summary_df.empty:
        log(f"Cross-clade plot -> {cross_path}")

    log("Done.")


if __name__ == "__main__":
    main()
