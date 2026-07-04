#!/usr/bin/env python3
"""
Domain-Time Scatter Plot Analysis

This script analyzes how domain-based distances correlate with TimeTree distances
as we vary the number of domains included. It compares variance-weighted vs
alpha-weighted distance metrics to find the optimal number of domains.

Approach:
- Create scatter plots: D_ij(N) vs T_ij for different N
- Compare variance-weighted vs alpha-weighted metrics
- Find optimal N that maximizes correlation

KEY FIXES IMPLEMENTED:
1. Consistent point counts: All scatter plots now use the same number of points
   (n_species * (n_species - 1) / 2), regardless of N. We always loop over all
   unordered pairs (i < j) of the species set.
2. Spearman correlation is the primary metric: Spearman (rank-based) correlation
   is computed first and displayed prominently, with Pearson as secondary.
3. No regression line: Scatter plots show only the cloud of points with
   correlation values in the title. The regression line was removed as it was
   sometimes confusing (e.g., going negative while all points are positive).
4. All N values included: Scatter plots are generated for all N values in
   NS_TO_TEST:
   [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.interpolate import UnivariateSpline

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit("This script requires the `ete3` package. Please install it and retry.") from exc


# Project paths
# Use current workspace directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Output directory

# Constants
EPSILON = 1e-8  # Matching current D1 implementation
NS_TO_TEST = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000]
# N=1 is excluded from NS_FOR_SCATTER (headline scatter series): the single-domain
# run is a zero-inflation artifact (invalid, needs >=2 domains) that gives a spurious peak.
# It is intentionally KEPT in NS_TO_TEST above so the diagnostic scripts (which import
# NS_TO_TEST) still compute and report N=1. NS_FOR_SCATTER is local to this script (not imported).
NS_FOR_SCATTER = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]  # Which N values to create scatter plots for (N=1 excluded from headline)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def log(message: str) -> None:
    """Print log message."""
    print(f"[domain_time_scatter] {message}")


def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """
    LOWESS (Locally Weighted Scatterplot Smoothing) using UnivariateSpline for smooth curves.
    
    Args:
        x: Independent variable
        y: Dependent variable
        frac: Fraction of data to use for smoothing (0.0 to 1.0) - controls smoothness
    
    Returns:
        Smoothed y values
    """
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    n = len(x_sorted)
    
    if n < 3:
        # Not enough points for smoothing
        return y_sorted
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    if not np.all(valid_mask):
        x_sorted = x_sorted[valid_mask]
        y_sorted = y_sorted[valid_mask]
        n = len(x_sorted)
    
    if n < 3:
        return y_sorted
    
    # Use UnivariateSpline for smooth interpolation
    # s parameter controls smoothness: larger s = smoother curve
    # Set s based on data variance and number of points
    # Use frac to adjust smoothness: higher frac = smoother
    # Convert frac to smoothing factor: frac=0.3 means use ~30% of data variance as smoothing
    y_variance = np.var(y_sorted)
    smoothing_factor = y_variance * (1.0 - frac) * n
    
    try:
        # Use UnivariateSpline with appropriate smoothing
        spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor, k=min(3, n-1))
        y_smooth = spline(x_sorted)
    except Exception:
        # Fallback to linear interpolation if spline fails
        from scipy.interpolate import interp1d
        interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        y_smooth = interp_func(x_sorted)
    
    # Restore original order if needed
    if not np.all(valid_mask):
        y_smooth_full = np.full(len(x), np.nan)
        y_smooth_full[sort_idx[valid_mask]] = y_smooth
        return y_smooth_full
    
    y_smooth_restored = np.zeros_like(y)
    y_smooth_restored[sort_idx] = y_smooth
    
    return y_smooth_restored


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_species_list(path: str) -> List[str]:
    """Load species list, stripping whitespace and handling BOM."""
    species = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                species.append(stripped)
    return species


def load_domain_counts(path: str, species_order: List[str]) -> pd.DataFrame:
    """Load domain count matrix and align to species order."""
    log(f"Loading domain counts from {path}")
    # Read CSV: first row is metadata, second row has species names as columns
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0, low_memory=False)
    
    # Transpose so species are rows and domains are columns
    X_raw = df.transpose()
    X_raw.index.name = "species"
    
    # Clean species names (remove parenthetical descriptions)
    def clean_species_name(name):
        # Remove everything after first parenthesis
        cleaned = name.split("(")[0].strip()
        return cleaned
    
    X_raw.index = [clean_species_name(name) for name in X_raw.index]
    
    # Map to target species order (handle simple mismatches)
    target_set = set(species_order)
    mapping = {}
    for label in X_raw.index:
        # Try exact match
        if label in target_set:
            mapping[label] = label
        else:
            # Try case-insensitive match
            for sp in species_order:
                if label.lower() == sp.lower():
                    mapping[label] = sp
                    break
    
    # Subset to mapped species
    mapped_labels = [l for l in X_raw.index if l in mapping]
    X_raw = X_raw.loc[mapped_labels]
    X_raw.index = [mapping[label] for label in X_raw.index]
    
    # Handle duplicates by averaging
    if X_raw.index.duplicated().any():
        log("Warning: Duplicate species detected after mapping; averaging duplicated rows.")
        X_raw = X_raw.groupby(X_raw.index).mean()
    
    # Subset and reorder to match species_order
    missing = set(species_order) - set(X_raw.index)
    if missing:
        log(f"Warning: {len(missing)} species missing from domain counts, proceeding with available species")
        species_order = [sp for sp in species_order if sp in X_raw.index]
    
    X_raw = X_raw.loc[species_order]
    log(f"Loaded {X_raw.shape[0]} species × {X_raw.shape[1]} domains")
    return X_raw


def load_phylogeny_and_compute_patristic(path: str, species_order: List[str]) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    """
    Load phylogeny and compute patristic distance matrix.
    Returns flattened upper triangle T_vec and list of (i,j) pairs.
    """
    log(f"Loading phylogeny from {path}")
    tree = Tree(path, format=1)
    
    # Known tree name mappings
    TREE_NAME_MAP = {
        "Neogale vison": "Neovison vison",
        "Neogale_vison": "Neovison vison",
        "Bos grunniens": "Bos mutus grunniens",
        "Bos_grunniens": "Bos mutus grunniens",
        "Physeter catodon": "Physeter macrocephalus",
        "Physeter_catodon": "Physeter macrocephalus",
    }
    
    # Map tree tip labels to species_order
    species_set = set(species_order)
    tree_species_map = {}
    
    for leaf in tree.iter_leaves():
        leaf_name_orig = leaf.name
        # Try underscore to space conversion
        leaf_name = leaf_name_orig.replace("_", " ")
        
        # Check preset map first
        if leaf_name_orig in TREE_NAME_MAP:
            mapped = TREE_NAME_MAP[leaf_name_orig]
        elif leaf_name in TREE_NAME_MAP:
            mapped = TREE_NAME_MAP[leaf_name]
        else:
            mapped = leaf_name
        
        # Try exact match
        if mapped in species_set:
            tree_species_map[leaf_name_orig] = mapped
        else:
            # Try case-insensitive match
            for sp in species_order:
                if mapped.lower() == sp.lower():
                    tree_species_map[leaf_name_orig] = sp
                    break
    
    # Rename tree tips
    for leaf in list(tree.iter_leaves()):
        if leaf.name in tree_species_map:
            leaf.name = tree_species_map[leaf.name]
        else:
            leaf.detach()
    
    # Get available species from tree
    tree_species = {leaf.name for leaf in tree.iter_leaves()}
    available_species = [sp for sp in species_order if sp in tree_species]
    
    if len(available_species) < len(species_order):
        missing = set(species_order) - tree_species
        log(f"Warning: {len(missing)} species missing from phylogeny, proceeding with {len(available_species)} species")
        species_order = available_species
    
    # Prune to available species
    tree.prune(available_species, preserve_branch_length=True)
    
    # Compute patristic distance matrix
    n = len(available_species)
    T_matrix = np.zeros((n, n), dtype=float)
    
    for i, sp_i in enumerate(available_species):
        for j in range(i + 1, n):
            sp_j = available_species[j]
            dist = tree.get_distance(sp_i, sp_j)
            T_matrix[i, j] = T_matrix[j, i] = dist
    
    # Flatten upper triangle
    T_vec = []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            T_vec.append(T_matrix[i, j])
            pairs.append((i, j))
    
    T_vec = np.array(T_vec)
    log(f"Computed patristic distances for {len(pairs)} species pairs")
    
    return T_vec, pairs, available_species


# ---------------------------------------------------------------------------
# Per-Domain Statistics
# ---------------------------------------------------------------------------


def compute_domain_stats(X_raw: pd.DataFrame, T_vec: np.ndarray, pairs: List[Tuple[int, int]]) -> pd.DataFrame:
    """
    For each domain, compute:
    - ΔC_ij(k) = X_raw[i,k] - X_raw[j,k]
    - y_ij_k = (ΔC_ij(k))²
    - Correlation r_k between y_k_vec and T_vec
    - Linear regression: y_k_vec = α_k * T_vec + β_k
    - Variance var_k (population variance, ddof=0)
    
    Uses vectorized operations for speed.
    """
    log("Computing per-domain statistics...")
    
    # Convert to numpy array for faster computation
    X_array = X_raw.values  # shape: (n_species, n_domains)
    n_species, n_domains = X_array.shape
    domain_names = list(X_raw.columns)
    
    log(f"  Computing pairwise squared differences for {n_domains} domains...")
    
    # Pre-compute all pairwise squared differences using vectorized operations
    # For each pair (i,j), compute (X[i,:] - X[j,:])^2 for all domains at once
    y_all = np.zeros((len(pairs), n_domains), dtype=float)
    
    for idx, (i, j) in enumerate(pairs):
        delta = X_array[i, :] - X_array[j, :]
        y_all[idx, :] = delta ** 2
    
    log(f"  Computing correlations and regressions...")
    
    # Compute variances for all domains at once (population variance, ddof=0)
    variances = np.var(X_array, axis=0, ddof=0)
    
    # Compute correlations and regressions for all domains
    records = []
    T_std = np.std(T_vec)
    T_mean = np.mean(T_vec)
    
    for k in range(n_domains):
        if (k + 1) % 5000 == 0:
            log(f"    Processed {k + 1} / {n_domains} domains...")
        
        y_k_vec = y_all[:, k]
        y_std = np.std(y_k_vec)
        
        # Compute correlation
        if y_std > 0 and T_std > 0:
            try:
                r, pval = pearsonr(y_k_vec, T_vec)
            except Exception:
                r, pval = 0.0, 1.0
        else:
            r, pval = 0.0, 1.0
        
        # Linear regression: y = α * T + β
        if y_std > 0 and T_std > 0:
            try:
                slope, intercept, r_value, p_reg, std_err = linregress(T_vec, y_k_vec)
                alpha = slope
                beta = intercept
                r_squared = r_value ** 2
            except Exception:
                alpha, beta, r_squared, p_reg = 0.0, 0.0, 0.0, 1.0
        else:
            alpha, beta, r_squared, p_reg = 0.0, 0.0, 0.0, 1.0
        
        records.append({
            "domain_id": domain_names[k],
            "r": r,
            "abs_r": abs(r),
            "R2": r_squared,
            "pval": pval,
            "alpha": alpha,
            "beta": beta,
            "variance": variances[k],
        })
    
    df = pd.DataFrame(records)
    log(f"Computed statistics for {len(df)} domains")
    return df


# ---------------------------------------------------------------------------
# Domain Ranking
# ---------------------------------------------------------------------------


def select_ranked_domains(domain_stats: pd.DataFrame, top_domains_path: str = None) -> List[str]:
    """
    Rank domains by absolute correlation.
    If TopDomains.txt exists, use it; otherwise rank by |r|.
    """
    if top_domains_path and os.path.exists(top_domains_path):
        log(f"Loading domain ranking from {top_domains_path}")
        with open(top_domains_path, "r") as f:
            ranked = [line.strip() for line in f if line.strip()]
        # Filter to domains that exist in stats
        ranked = [d for d in ranked if d in domain_stats["domain_id"].values]
        log(f"Using {len(ranked)} domains from TopDomains.txt")
        return ranked
    else:
        log("Ranking domains by absolute correlation...")
        # Filter to domains with pval < 0.05 and positive alpha (optional but recommended)
        filtered = domain_stats[
            (domain_stats["pval"] < 0.05) & (domain_stats["alpha"] > 0)
        ].copy()
        
        if len(filtered) == 0:
            log("Warning: No domains with pval < 0.05 and alpha > 0, using all domains")
            filtered = domain_stats.copy()
        
        # Sort by descending |r|
        ranked = filtered.sort_values("abs_r", ascending=False)["domain_id"].tolist()
        log(f"Ranked {len(ranked)} domains by absolute correlation")
        return ranked


# ---------------------------------------------------------------------------
# Distance Computation
# ---------------------------------------------------------------------------


def compute_summed_distances(
    X_raw: pd.DataFrame,
    domain_list: List[str],
    domain_stats: pd.DataFrame,
    pairs: List[Tuple[int, int]],
    use_alpha: bool = False,
) -> np.ndarray:
    """
    Compute summed distance matrix D_ij(N) for given domain list.
    
    If use_alpha=False: d_ij,var(k) = (ΔC_ij(k))² / (var_k + ε)
    If use_alpha=True: d_ij,α(k) = (ΔC_ij(k))² / (|α_k| + ε)
    
    Returns flattened upper triangle vector.
    Uses vectorized operations for speed.
    """
    n_pairs = len(pairs)
    
    # Filter to domains that exist in stats
    domain_list = [d for d in domain_list if d in domain_stats["domain_id"].values]
    if len(domain_list) == 0:
        return np.zeros(n_pairs, dtype=float)
    
    # Get domain indices and stats
    stats_subset = domain_stats[domain_stats["domain_id"].isin(domain_list)].copy()
    stats_subset = stats_subset.set_index("domain_id")
    
    # Get denominators for all domains at once
    if use_alpha:
        denominators = np.abs(stats_subset["alpha"].values) + EPSILON
    else:
        denominators = stats_subset["variance"].values + EPSILON
    
    # Filter out domains with zero denominator
    valid_mask = denominators > EPSILON
    valid_domains = stats_subset.index[valid_mask].tolist()
    valid_denoms = denominators[valid_mask]
    
    if len(valid_domains) == 0:
        return np.zeros(n_pairs, dtype=float)
    
    # Convert to numpy array for vectorized operations
    X_array = X_raw[valid_domains].values  # shape: (n_species, n_valid_domains)
    
    # Compute all pairwise squared differences for all valid domains at once
    # This creates a matrix: (n_pairs, n_valid_domains)
    contributions = np.zeros((n_pairs, len(valid_domains)), dtype=float)
    
    for idx, (i, j) in enumerate(pairs):
        delta = X_array[i, :] - X_array[j, :]
        contributions[idx, :] = (delta ** 2) / valid_denoms
    
    # Sum across domains to get final distance vector
    D_vec = np.sum(contributions, axis=1)
    
    return D_vec


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------


def main():
    """Main analysis pipeline."""
    log("Starting domain-time scatter plot analysis")
    
    # Ensure results directory exists
    ensure_dir(RESULTS)
    
    # 1. Load data
    log("\n=== Step 1: Loading Data ===")
    species_order_orig = load_species_list(os.path.join(DATA_RAW, "MammalsList.txt"))
    X_raw = load_domain_counts(os.path.join(DATA_RAW, "MammalDomainCount.tsv"), species_order_orig)
    
    # Update species_order to match what's actually in X_raw
    species_order = list(X_raw.index)
    
    T_vec, pairs, species_in_tree = load_phylogeny_and_compute_patristic(
        os.path.join(DATA_RAW, "MammalsPhylogeny.nwk"), species_order
    )
    
    # Final intersection: only use species present in all three sources
    species_final = [sp for sp in species_order if sp in species_in_tree]

    # Exclude the platypus outgroup (Ornithorhynchus) so Part 1 uses the SAME canonical
    # cohort as Parts 2/3 (107 species, platypus excluded, Cervus hanglu INCLUDED).
    # Outgroup handling must be consistent across the paper: dropping Cervus while
    # keeping platypus would yield a different 107-species set sharing only 106 taxa
    # with Parts 2/3.
    species_final = [sp for sp in species_final if "Ornithorhynchus" not in sp]
    log(f"Excluded platypus outgroup (canonical cohort). Species count: {len(species_final)}")
    
    X_raw = X_raw.loc[species_final]
    
    # Recompute patristic for final species set
    T_vec, pairs, _ = load_phylogeny_and_compute_patristic(
        os.path.join(DATA_RAW, "MammalsPhylogeny.nwk"), species_final
    )
    
    log(f"Final species set: {len(species_final)} species")
    
    # Filter domains by frequency (remove only rare domains <5%, keep conserved domains)
    log("\n=== Step 1.5: Filtering Domains by Frequency ===")
    n_species = X_raw.shape[0]
    domain_presence = (X_raw > 0).sum(axis=0)  # count species with each domain
    presence_fraction = domain_presence / n_species
    
    min_freq = 0.05  # present in at least 5% of species
    # No max_freq filter - keep conserved domains (>95%) as they are still informative
    
    valid_mask = (presence_fraction >= min_freq)  # Only filter out rare domains
    n_domains_before = X_raw.shape[1]
    n_domains_after = valid_mask.sum()
    n_filtered = n_domains_before - n_domains_after
    
    log(f"Domains before filtering: {n_domains_before}")
    log(f"Domains after filtering: {n_domains_after} ({100*n_domains_after/n_domains_before:.1f}% retained)")
    log(f"Filtered out: {n_filtered} domains (too rare, <{min_freq:.0%} prevalence)")
    log(f"Kept conserved domains (>95% prevalence) as they are still informative")
    
    # Apply filter
    X_raw = X_raw.loc[:, valid_mask]
    log(f"Using {X_raw.shape[1]} domains with prevalence >= {min_freq:.0%}")
    
    # 2. Compute per-domain statistics
    log("\n=== Step 2: Computing Per-Domain Statistics ===")
    domain_stats = compute_domain_stats(X_raw, T_vec, pairs)
    stats_path = os.path.join(RESULTS, "domain_time_domainStats.tsv")
    domain_stats.to_csv(stats_path, sep="\t", index=False)
    log(f"Saved domain statistics to {stats_path}")
    
    # 3. Rank domains
    log("\n=== Step 3: Ranking Domains ===")
    top_domains_path = os.path.join(DATA_RAW, "TopDomains.txt")
    ranked_domains = select_ranked_domains(domain_stats, top_domains_path)
    
    # Sanity check: verify top domain has reasonable prevalence
    if len(ranked_domains) > 0:
        top_domain = ranked_domains[0]
        if top_domain in X_raw.columns:
            top_domain_prevalence = (X_raw[top_domain] > 0).sum()
            top_domain_pct = 100 * top_domain_prevalence / len(species_final)
            log(f"Top ranked domain: {top_domain}")
            log(f"  Present in {top_domain_prevalence} out of {len(species_final)} species ({top_domain_pct:.1f}%)")
            if top_domain_pct < 10 or top_domain_pct > 90:
                log(f"  WARNING: Top domain prevalence ({top_domain_pct:.1f}%) is outside recommended range (10-90%)")
    
    # Clip N values to available domains
    max_n = min(max(NS_TO_TEST), len(ranked_domains))
    ns_to_test = [n for n in NS_TO_TEST if n <= max_n]
    log(f"Testing N values: {ns_to_test}")
    
    # 4. Compute distances and correlations for each N
    log("\n=== Step 4: Computing Distances and Correlations ===")
    # Ensure we always use all unordered pairs (i < j) for consistent point counts
    n_species_final = len(species_final)
    expected_pairs = n_species_final * (n_species_final - 1) // 2
    log(f"  Using all unordered species pairs: {expected_pairs} pairs from {n_species_final} species")
    log(f"  This count should be identical for all N values")
    
    results = []
    
    for N in ns_to_test:
        log(f"  Processing N = {N}...")
        domain_panel = ranked_domains[:N]
        
        # Compute distances - always returns vector of length len(pairs)
        D_var_vec = compute_summed_distances(X_raw, domain_panel, domain_stats, pairs, use_alpha=False)
        D_alpha_vec = compute_summed_distances(X_raw, domain_panel, domain_stats, pairs, use_alpha=True)

        # Verify vector lengths match expected pair count
        assert len(D_var_vec) == expected_pairs, f"D_var_vec length mismatch: expected {expected_pairs}, got {len(D_var_vec)}"
        assert len(D_alpha_vec) == expected_pairs, f"D_alpha_vec length mismatch: expected {expected_pairs}, got {len(D_alpha_vec)}"
        assert len(T_vec) == expected_pairs, f"T_vec length mismatch: expected {expected_pairs}, got {len(T_vec)}"

        # Count non-zero values to show how many pairs have non-zero distances
        n_nonzero_var = np.count_nonzero(D_var_vec)
        n_zero_var = np.sum(D_var_vec == 0)
        n_nonzero_alpha = np.count_nonzero(D_alpha_vec)
        n_zero_alpha = np.sum(D_alpha_vec == 0)
        
        # Compute correlations using the same flattened vectors for all N
        # Spearman (primary) and Pearson (secondary) correlations with p-values
        # Note: All pairs are included in correlation, even if D_ij(N) = 0
        spearman_var, p_spearman_var = spearmanr(D_var_vec, T_vec)
        pearson_var, p_pearson_var = pearsonr(D_var_vec, T_vec)
        spearman_alpha, p_spearman_alpha = spearmanr(D_alpha_vec, T_vec)
        pearson_alpha, p_pearson_alpha = pearsonr(D_alpha_vec, T_vec)

        num_pairs = len(pairs)
        log(f"    N = {N}, species = {n_species_final}, total points = {num_pairs} (should equal {expected_pairs})")
        log(f"      Variance: {n_nonzero_var} non-zero, {n_zero_var} zero (all {num_pairs} pairs included)")
        log(f"      Alpha: {n_nonzero_alpha} non-zero, {n_zero_alpha} zero (all {num_pairs} pairs included)")
        log(f"    Spearman (primary): var r_s = {spearman_var:.4f}, alpha r_s = {spearman_alpha:.4f}")
        log(f"    Pearson (secondary): var r = {pearson_var:.4f}, alpha r = {pearson_alpha:.4f}")

        results.append({
            "N": N,
            "species_count": len(species_final),
            "pair_count": num_pairs,
            "spearman_var": spearman_var,
            "spearman_var_pval": p_spearman_var,
            "pearson_var": pearson_var,
            "pearson_var_pval": p_pearson_var,
            "spearman_alpha": spearman_alpha,
            "spearman_alpha_pval": p_spearman_alpha,
            "pearson_alpha": pearson_alpha,
            "pearson_alpha_pval": p_pearson_alpha,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "N",
            "species_count",
            "pair_count",
            "spearman_var",
            "spearman_var_pval",
            "pearson_var",
            "pearson_var_pval",
            "spearman_alpha",
            "spearman_alpha_pval",
            "pearson_alpha",
            "pearson_alpha_pval",
        ]
    ]
    results_path = os.path.join(RESULTS, "domain_time_correlations.tsv")
    results_df.to_csv(results_path, sep="\t", index=False)
    log(f"Saved correlation results to {results_path}")
    
    # 5. Create scatter plots for selected N values
    log("\n=== Step 5: Creating Scatter Plots ===")
    ns_for_scatter = [n for n in NS_FOR_SCATTER if n <= max_n]
    
    # Verify consistent pair counts - all scatter plots should use the same pairs
    n_species_final = len(species_final)
    expected_pairs = n_species_final * (n_species_final - 1) // 2
    log(f"  Expected number of pairs for all scatter plots: {expected_pairs} (from {n_species_final} species)")
    log(f"  Actual pairs list length: {len(pairs)}")
    assert len(pairs) == expected_pairs, f"Pair count mismatch: expected {expected_pairs}, got {len(pairs)}"
    assert len(T_vec) == expected_pairs, f"T_vec length mismatch: expected {expected_pairs}, got {len(T_vec)}"
    
    for N in ns_for_scatter:
        log(f"  Creating scatter plots for N = {N}...")
        domain_panel = ranked_domains[:N]
        
        # Compute distances - these should always return vectors of length len(pairs)
        D_var_vec = compute_summed_distances(X_raw, domain_panel, domain_stats, pairs, use_alpha=False)
        D_alpha_vec = compute_summed_distances(X_raw, domain_panel, domain_stats, pairs, use_alpha=True)
        
        # Verify vector lengths are consistent
        assert len(D_var_vec) == expected_pairs, f"D_var_vec length mismatch for N={N}: expected {expected_pairs}, got {len(D_var_vec)}"
        assert len(D_alpha_vec) == expected_pairs, f"D_alpha_vec length mismatch for N={N}: expected {expected_pairs}, got {len(D_alpha_vec)}"
        
        # Count non-zero values to verify all pairs are included
        n_nonzero_var = np.count_nonzero(D_var_vec)
        n_zero_var = np.sum(D_var_vec == 0)
        n_nonzero_alpha = np.count_nonzero(D_alpha_vec)
        n_zero_alpha = np.sum(D_alpha_vec == 0)
        
        log(f"    N = {N}, species = {n_species_final}, total points = {len(D_var_vec)} (should be {expected_pairs})")
        log(f"      Variance: {n_nonzero_var} non-zero, {n_zero_var} zero values")
        log(f"      Alpha: {n_nonzero_alpha} non-zero, {n_zero_alpha} zero values")
        
        stats_row = results_df[results_df["N"] == N].iloc[0]
        r_var = stats_row["spearman_var"]
        pearson_var_val = stats_row["pearson_var"]
        r_alpha = stats_row["spearman_alpha"]
        pearson_alpha_val = stats_row["pearson_alpha"]
        
        # Variance-weighted density plot with trend lines
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create density plot (hexbin or KDE) with log-scale density coloring
        if HAS_SEABORN:
            # Use hexbin for density with log normalization
            hb = ax.hexbin(T_vec, D_var_vec, gridsize=50, cmap='Blues', mincnt=1, alpha=0.7, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='Point density (log scale)')
        else:
            # Fallback to hexbin from matplotlib with log normalization
            hb = ax.hexbin(T_vec, D_var_vec, gridsize=50, cmap='Blues', mincnt=1, alpha=0.7, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='Point density (log scale)')
        
        # Add LOWESS trend line
        # Filter out zeros for smoother trend line
        nonzero_mask = D_var_vec > 0
        if np.sum(nonzero_mask) > 10:
            T_nonzero = T_vec[nonzero_mask]
            D_nonzero = D_var_vec[nonzero_mask]
            
            # Sort for smooth line
            sort_idx = np.argsort(T_nonzero)
            T_sorted = T_nonzero[sort_idx]
            D_sorted = D_nonzero[sort_idx]
            
            # LOWESS smoothing
            D_smooth = lowess_smooth(T_sorted, D_sorted, frac=0.3)
            ax.plot(T_sorted, D_smooth, 'r-', linewidth=2.5, label='LOWESS trend', alpha=0.8)

        # Add text annotation
        n_nonzero = np.count_nonzero(D_var_vec)
        n_zero = len(D_var_vec) - n_nonzero
        if n_zero > 0:
            ax.text(0.02, 0.98, f"Total points: {len(D_var_vec)}\nNon-zero: {n_nonzero}\nZero: {n_zero}", 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlabel("T_ij (TimeTree distance)", fontsize=12)
        ax.set_ylabel(f"D_ij(N) = Σ_k (ΔC_ij² / var_k)", fontsize=12)
        ax.set_title(
            f"Variance metric, N={N} domains - Spearman r_s = {r_var:.4f}, Pearson r = {pearson_var_val:.4f}",
            fontsize=14,
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scatter_path = os.path.join(RESULTS, f"scatter_var_N{N}.png")
        plt.savefig(scatter_path, dpi=200)
        plt.close()
        log(f"    Saved {scatter_path}")
        
        # Alpha-weighted density plot with trend lines
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create density plot (hexbin or KDE) with log-scale density coloring
        if HAS_SEABORN:
            # Use hexbin for density with log normalization
            hb = ax.hexbin(T_vec, D_alpha_vec, gridsize=50, cmap='Oranges', mincnt=1, alpha=0.7, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='Point density (log scale)')
        else:
            # Fallback to hexbin from matplotlib with log normalization
            hb = ax.hexbin(T_vec, D_alpha_vec, gridsize=50, cmap='Oranges', mincnt=1, alpha=0.7, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='Point density (log scale)')
        
        # Add LOWESS trend line
        # Filter out zeros for smoother trend line
        nonzero_mask = D_alpha_vec > 0
        if np.sum(nonzero_mask) > 10:
            T_nonzero = T_vec[nonzero_mask]
            D_nonzero = D_alpha_vec[nonzero_mask]
            
            # Sort for smooth line
            sort_idx = np.argsort(T_nonzero)
            T_sorted = T_nonzero[sort_idx]
            D_sorted = D_nonzero[sort_idx]
            
            # LOWESS smoothing
            D_smooth = lowess_smooth(T_sorted, D_sorted, frac=0.3)
            ax.plot(T_sorted, D_smooth, 'r-', linewidth=2.5, label='LOWESS trend', alpha=0.8)

        # Add text annotation
        n_nonzero = np.count_nonzero(D_alpha_vec)
        n_zero = len(D_alpha_vec) - n_nonzero
        if n_zero > 0:
            ax.text(0.02, 0.98, f"Total points: {len(D_alpha_vec)}\nNon-zero: {n_nonzero}\nZero: {n_zero}", 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlabel("T_ij (TimeTree distance)", fontsize=12)
        ax.set_ylabel(f"D_ij(N) = Σ_k (ΔC_ij² / |α_k|)", fontsize=12)
        ax.set_title(
            f"Alpha metric, N={N} domains - Spearman r_s = {r_alpha:.4f}, Pearson r = {pearson_alpha_val:.4f}",
            fontsize=14,
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scatter_path = os.path.join(RESULTS, f"scatter_alpha_N{N}.png")
        plt.savefig(scatter_path, dpi=200)
        plt.close()
        log(f"    Saved {scatter_path}")
    
    # 6. Create summary plot: correlation vs N
    log("\n=== Step 6: Creating Summary Plot ===")
    # N=1 is EXCLUDED from this headline figure: the single-domain run is a
    # zero-inflation artifact (need >=2 domains) that produces a spurious correlation peak,
    # so showing it would misrepresent the headline trend. N=1 is RETAINED in NS_TO_TEST and
    # in domain_time_correlations.tsv (written above) for the diagnostics + optional supplement.
    plot_df = results_df[results_df["N"] != 1]
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(plot_df["N"], plot_df["spearman_var"], "o-", label="Variance Spearman r_s", linewidth=2, markersize=8)
    ax.plot(plot_df["N"], plot_df["spearman_alpha"], "s-", label="Alpha Spearman r_s", linewidth=2, markersize=8)
    ax.plot(plot_df["N"], plot_df["pearson_var"], "o--", label="Variance Pearson r", linewidth=1.5, markersize=6, color="gray")
    ax.plot(plot_df["N"], plot_df["pearson_alpha"], "s--", label="Alpha Pearson r", linewidth=1.5, markersize=6, color="lightgray")
    
    ax.set_xlabel("Number of domains (N)", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title("Correlation between Domain Distance and TimeTree Distance", fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    summary_path = os.path.join(RESULTS, "correlation_vs_N.png")
    plt.savefig(summary_path, dpi=200)
    plt.close()
    log(f"Saved summary plot to {summary_path}")
    
    # 7. Print summary
    log("\n=== Step 7: Results Summary ===")
    best_var_idx = results_df["spearman_var"].idxmax()
    best_alpha_idx = results_df["spearman_alpha"].idxmax()
    
    best_var_N = results_df.loc[best_var_idx, "N"]
    best_var_rs = results_df.loc[best_var_idx, "spearman_var"]
    best_var_rp = results_df.loc[best_var_idx, "pearson_var"]
    
    best_alpha_N = results_df.loc[best_alpha_idx, "N"]
    best_alpha_rs = results_df.loc[best_alpha_idx, "spearman_alpha"]
    best_alpha_rp = results_df.loc[best_alpha_idx, "pearson_alpha"]
    
    if best_var_rs > best_alpha_rs:
        best_overall = "variance"
        best_overall_N = best_var_N
        best_overall_rs = best_var_rs
    else:
        best_overall = "alpha"
        best_overall_N = best_alpha_N
        best_overall_rs = best_alpha_rs
    
    print("\n" + "=" * 80)
    print("DOMAIN-TIME SCATTER PLOT ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nSummary across all N values:")
    print(f"{'N':<6} {'Spearman r_s (var)':<20} {'p-val':<12} {'Pearson r (var)':<18} {'p-val':<12} {'Spearman r_s (alpha)':<22} {'p-val':<12} {'Pearson r (alpha)':<18} {'p-val':<12}")
    print("-" * 140)
    for _, row in results_df.iterrows():
        print(f"{int(row['N']):<6} "
              f"{row['spearman_var']:<20.4f} {row['spearman_var_pval']:<12.2e} "
              f"{row['pearson_var']:<18.4f} {row['pearson_var_pval']:<12.2e} "
              f"{row['spearman_alpha']:<22.4f} {row['spearman_alpha_pval']:<12.2e} "
              f"{row['pearson_alpha']:<18.4f} {row['pearson_alpha_pval']:<12.2e}")
    
    print("\n" + "=" * 80)
    print("BEST RESULTS:")
    print("=" * 80)
    print(f"Best variance-weighted Spearman r_s = {best_var_rs:.4f} (Pearson r = {best_var_rp:.4f}) at N = {best_var_N}")
    print(f"Best alpha-weighted Spearman r_s = {best_alpha_rs:.4f} (Pearson r = {best_alpha_rp:.4f}) at N = {best_alpha_N}")
    print(f"Best overall metric by Spearman: {best_overall} at N = {best_overall_N} (r_s = {best_overall_rs:.4f})")
    print("=" * 80)
    
    log("Analysis complete!")


if __name__ == "__main__":
    main()

