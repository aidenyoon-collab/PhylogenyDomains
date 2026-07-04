#!/usr/bin/env python3
"""
Sanity Checks for LOOCV Domain-Phylogeny Pipeline

This standalone script performs comprehensive sanity checks on the LOOCV pipeline
to catch bugs, data alignment issues, distance scale problems, and feature scaling issues.

It does NOT run the actual model training - only validates the data pipeline.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available - plots will be skipped")

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit(
        "This script requires the `ete3` package. Please install it and retry."
    ) from exc

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files (relative to script directory or absolute paths)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAIN_TSV = os.path.join(PROJECT_ROOT, "data_raw", "MammalDomainCount.tsv")
TREE_NWK = os.path.join(PROJECT_ROOT, "MammalsPhylogeny.nwk")
SPECIES_LIST_TXT = os.path.join(PROJECT_ROOT, "data_raw", "MammalsList.txt")  # Optional

# Configuration flags
EXCLUDE_OUTGROUP = True
OUTGROUP_NAME = "Ornithorhynchus_anatinus"  # Will check both underscore and space versions
RANDOM_SEED = 0
CHECK_HELDOUT_SPECIES = None  # If None, pick 3 random + first + last

# Output files
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "sanity_checks_loocv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "sanity_report.txt")
HIST_PLOT = os.path.join(OUTPUT_DIR, "sanity_distance_hist.png")
BOX_PLOT = os.path.join(OUTPUT_DIR, "sanity_test_distance_box.png")

# Constants
NORMALIZATION_EPS = 1e-12
SYMMETRY_TOL = 1e-8
DIAGONAL_TOL = 1e-10
DISTANCE_VERIFY_TOL = 1e-6

# Known tree name mappings (to reconcile labels)
TREE_NAME_MAP = {
    "Neogale vison": "Neovison vison",
    "Neogale_vison": "Neovison vison",
    "Bos grunniens": "Bos mutus grunniens",
    "Bos_grunniens": "Bos mutus grunniens",
    "Physeter catodon": "Physeter macrocephalus",
    "Physeter_catodon": "Physeter macrocephalus",
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


class ReportWriter:
    """Context manager for writing report to both stdout and file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_handle = None
        
    def __enter__(self):
        ensure_dir(os.path.dirname(self.filepath))
        self.file_handle = open(self.filepath, "w")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
            
    def write(self, message: str):
        """Write message to both stdout and file."""
        print(message)
        if self.file_handle:
            self.file_handle.write(message + "\n")
            self.file_handle.flush()


def clean_species_name(name: str) -> str:
    """Remove parenthetical descriptions from species names."""
    return name.split("(")[0].strip()


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def load_domain_counts(domain_path: str) -> pd.DataFrame:
    """
    Load domain count matrix from TSV file.
    
    Format: TSV with skiprows=1, index_col=0, then transposed.
    Returns: DataFrame with species as rows, domains as columns.
    """
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Domain counts file not found: {domain_path}")
    
    # Read CSV: first row is metadata, second row has species names as columns
    df = pd.read_csv(domain_path, sep="\t", skiprows=1, index_col=0, low_memory=False)
    
    # Transpose so species are rows and domains are columns
    X_raw = df.transpose()
    X_raw.index.name = "species"
    
    # Clean species names (remove parenthetical descriptions)
    X_raw.index = [clean_species_name(name) for name in X_raw.index]
    
    return X_raw


def load_tree_and_compute_distances(tree_path: str, species_list: List[str]) -> Tuple[pd.DataFrame, Tree]:
    """
    Load Newick tree and compute full patristic distance matrix.
    
    Returns:
        dist_matrix: DataFrame of shape (n_species, n_species) with symmetric distances
        tree: ete3 Tree object (pruned to available species)
    """
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    
    tree = Tree(tree_path, format=1)
    
    species_set = set(species_list)
    tree_species_map: dict[str, str] = {}
    
    # Map tree leaf names to species list
    for leaf in tree.iter_leaves():
        leaf_name_orig = leaf.name
        leaf_name = leaf_name_orig.replace("_", " ")
        
        # Check preset mappings
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
            # Fallback: case-insensitive match
            for sp in species_list:
                if mapped.lower() == sp.lower():
                    tree_species_map[leaf_name_orig] = sp
                    break
    
    # Rename and prune tree
    for leaf in list(tree.iter_leaves()):
        if leaf.name in tree_species_map:
            leaf.name = tree_species_map[leaf.name]
        else:
            leaf.detach()
    
    tree_species = {leaf.name for leaf in tree.iter_leaves()}
    available_species = [sp for sp in species_list if sp in tree_species]
    
    if len(available_species) < len(species_list):
        missing = set(species_list) - tree_species
        warnings.warn(f"{len(missing)} species missing from phylogeny: {sorted(missing)[:5]}...")
    
    tree.prune(available_species, preserve_branch_length=True)
    
    # Compute distance matrix
    n = len(available_species)
    T_matrix = np.zeros((n, n), dtype=float)
    
    for i, sp_i in enumerate(available_species):
        for j in range(i + 1, n):
            sp_j = available_species[j]
            dist = tree.get_distance(sp_i, sp_j)
            T_matrix[i, j] = T_matrix[j, i] = dist
    
    T_df = pd.DataFrame(T_matrix, index=available_species, columns=available_species)
    
    return T_df, tree


def preprocess_data(
    domain_path: str,
    tree_path: str,
    exclude_outgroup: bool = True,
    outgroup_name: str = "Ornithorhynchus_anatinus",
) -> Tuple[np.ndarray, np.ndarray, List[str], Tree]:
    """
    Load and preprocess domain counts and tree distances.
    
    Returns:
        X_norm: Normalized domain matrix (n_species × n_domains, float32)
        D: Distance matrix (n_species × n_species, float64)
        species: Final aligned species list
        tree: ete3 Tree object
    """
    # Load domain counts
    X_counts = load_domain_counts(domain_path)
    domain_species = list(X_counts.index)
    
    # Load tree and compute distances
    dist_matrix_full, tree = load_tree_and_compute_distances(tree_path, domain_species)
    tree_species = list(dist_matrix_full.index)
    
    # Intersect species sets
    species_intersection = sorted(set(domain_species) & set(tree_species))
    
    # Subset matrices to common species
    X_counts = X_counts.loc[species_intersection]
    dist_matrix = dist_matrix_full.loc[species_intersection, species_intersection]
    
    # Exclude outgroup if requested
    outgroup_space = outgroup_name.replace("_", " ")
    outgroup_underscore = outgroup_name
    
    if exclude_outgroup:
        if outgroup_space in X_counts.index:
            X_counts = X_counts.drop(index=outgroup_space)
            dist_matrix = dist_matrix.drop(index=outgroup_space, columns=outgroup_space)
        elif outgroup_underscore in X_counts.index:
            X_counts = X_counts.drop(index=outgroup_underscore)
            dist_matrix = dist_matrix.drop(index=outgroup_underscore, columns=outgroup_underscore)
    
    species_final = list(X_counts.index)
    
    # Drop zero-variance domains
    domain_variance = X_counts.var(axis=0)
    zero_var_mask = domain_variance == 0
    n_zero_var = int(zero_var_mask.sum())
    if n_zero_var > 0:
        X_counts = X_counts.loc[:, ~zero_var_mask]
    
    # Verify no remaining zero-variance domains
    if (X_counts.var(axis=0) == 0).any():
        raise ValueError("Zero-variance domains remain after filtering")
    
    # Normalize to relative frequencies per species
    row_sums = X_counts.sum(axis=1).values.reshape(-1, 1)
    X_array = X_counts.values.astype(float)
    X_norm_array = X_array / (row_sums + NORMALIZATION_EPS)
    
    # Convert to aligned NumPy arrays
    X_norm = X_norm_array.astype(np.float32)
    D = dist_matrix.values.astype(np.float64)
    
    # Ensure species order matches
    assert list(X_counts.index) == list(dist_matrix.index), "Species order mismatch"
    
    return X_norm, D, species_final, tree


# =============================================================================
# SANITY CHECKS
# =============================================================================


def check_distance_scale(D: np.ndarray, report: ReportWriter) -> None:
    """Check distance matrix scale and distribution."""
    report.write("\n" + "=" * 80)
    report.write("CHECK 1: Distance Scale")
    report.write("=" * 80)
    
    n = D.shape[0]
    report.write(f"Distance matrix shape: {D.shape}")
    
    # Extract off-diagonal values
    i_idx, j_idx = np.triu_indices(n, k=1)
    off_diag = D[i_idx, j_idx]
    
    # Statistics
    min_dist = float(np.min(off_diag))
    median_dist = float(np.median(off_diag))
    max_dist = float(np.max(off_diag))
    mean_dist = float(np.mean(off_diag))
    std_dist = float(np.std(off_diag))
    
    # Quantiles
    quantiles = np.percentile(off_diag, [1, 5, 50, 95, 99])
    
    report.write(f"\nOff-diagonal distance statistics:")
    report.write(f"  Number of pairs: {len(off_diag)}")
    report.write(f"  Min:    {min_dist:.6f}")
    report.write(f"  1%:     {quantiles[0]:.6f}")
    report.write(f"  5%:     {quantiles[1]:.6f}")
    report.write(f"  Median: {median_dist:.6f} (50%)")
    report.write(f"  Mean:   {mean_dist:.6f}")
    report.write(f"  Std:    {std_dist:.6f}")
    report.write(f"  95%:    {quantiles[3]:.6f}")
    report.write(f"  99%:    {quantiles[4]:.6f}")
    report.write(f"  Max:    {max_dist:.6f}")
    
    # Check for exact zeros
    n_zeros = int(np.sum(off_diag == 0))
    report.write(f"\nExact zeros off-diagonal: {n_zeros}")
    if n_zeros > 0:
        warnings.warn(f"Found {n_zeros} exact zero distances off-diagonal (should be 0)")
    
    # Check for spikes
    if max_dist > 10 * quantiles[4]:  # max > 10× 99th percentile
        warnings.warn(
            f"Potential spike detected: max={max_dist:.6f} is >10× 99th percentile "
            f"({quantiles[4]:.6f})"
        )
        report.write(
            f"\nWARNING: Max distance ({max_dist:.6f}) is >10× 99th percentile "
            f"({quantiles[4]:.6f}) - possible outlier"
        )


def check_species_alignment(D: np.ndarray, species: List[str], tree: Tree, report: ReportWriter) -> None:
    """Check species alignment, symmetry, and distance consistency."""
    report.write("\n" + "=" * 80)
    report.write("CHECK 2: Species Alignment")
    report.write("=" * 80)
    
    n = len(species)
    report.write(f"\nNumber of species: {n}")
    report.write(f"\nFirst 10 species in aligned order:")
    for i, sp in enumerate(species[:10], 1):
        report.write(f"  {i:2d}. {sp}")
    
    # Check symmetry
    max_diff = float(np.max(np.abs(D - D.T)))
    report.write(f"\nSymmetry check: max|D - D.T| = {max_diff:.2e}")
    if max_diff > SYMMETRY_TOL:
        raise AssertionError(f"Distance matrix is not symmetric: max|D - D.T| = {max_diff:.2e} > {SYMMETRY_TOL}")
    report.write("  [OK] Symmetry verified")
    
    # Check diagonal
    diag_max = float(np.max(np.abs(np.diag(D))))
    report.write(f"\nDiagonal check: max|diag(D)| = {diag_max:.2e}")
    if diag_max > DIAGONAL_TOL:
        warnings.warn(f"Diagonal not all zeros: max|diag(D)| = {diag_max:.2e}")
    else:
        report.write("  [OK] Diagonal verified (~0)")
    
    # Random verification: recompute 5 random pairs directly from tree
    np.random.seed(RANDOM_SEED)
    n_verify = min(5, n * (n - 1) // 2)
    random_pairs = np.random.choice(len(species) * (len(species) - 1) // 2, n_verify, replace=False)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    report.write(f"\nRandom verification (recomputing {n_verify} pairs from tree):")
    max_error = 0.0
    for pair_idx in random_pairs:
        i = i_idx[pair_idx]
        j = j_idx[pair_idx]
        sp_i = species[i]
        sp_j = species[j]
        stored_dist = D[i, j]
        recomputed_dist = tree.get_distance(sp_i, sp_j)
        error = abs(stored_dist - recomputed_dist)
        max_error = max(max_error, error)
        report.write(
            f"  Pair ({sp_i[:20]:20s}, {sp_j[:20]:20s}): "
            f"stored={stored_dist:.6f}, recomputed={recomputed_dist:.6f}, "
            f"error={error:.2e}"
        )
    
    report.write(f"\nMax verification error: {max_error:.2e}")
    if max_error > DISTANCE_VERIFY_TOL:
        raise AssertionError(
            f"Distance verification failed: max error = {max_error:.2e} > {DISTANCE_VERIFY_TOL}"
        )
    report.write("  [OK] Random verification passed")


def check_loocv_pair_masks(
    n_species: int,
    held_out_indices: List[int],
    report: ReportWriter,
) -> None:
    """Check LOOCV train/test pair masks for correctness."""
    report.write("\n" + "=" * 80)
    report.write("CHECK 3: LOOCV Pair Masks")
    report.write("=" * 80)
    
    # Precompute all pair indices
    i_idx, j_idx = np.triu_indices(n_species, k=1)
    n_pairs = len(i_idx)
    expected_train_pairs = n_species * (n_species - 1) // 2 - (n_species - 1)
    expected_test_pairs = n_species - 1
    
    report.write(f"\nTotal pairs: {n_pairs}")
    report.write(f"Expected train pairs (per held-out): {expected_train_pairs}")
    report.write(f"Expected test pairs (per held-out): {expected_test_pairs}")
    
    for s_idx in held_out_indices:
        report.write(f"\n--- Held-out species index: {s_idx} ---")
        
        # Build masks
        train_mask = (i_idx != s_idx) & (j_idx != s_idx)
        test_mask = (i_idx == s_idx) | (j_idx == s_idx)
        
        n_train = int(np.sum(train_mask))
        n_test = int(np.sum(test_mask))
        
        report.write(f"  Train pairs: {n_train} (expected: {expected_train_pairs})")
        report.write(f"  Test pairs:  {n_test} (expected: {expected_test_pairs})")
        
        # Assertions
        # 1. Masks are disjoint
        overlap = np.sum(train_mask & test_mask)
        if overlap > 0:
            raise AssertionError(f"Masks overlap: {overlap} pairs in both train and test")
        report.write("  [OK] Masks are disjoint")
        
        # 2. No s appears in train pairs
        train_has_s = np.any((i_idx[train_mask] == s_idx) | (j_idx[train_mask] == s_idx))
        if train_has_s:
            raise AssertionError(f"Held-out species {s_idx} appears in training pairs")
        report.write(f"  [OK] Held-out species {s_idx} not in training pairs")
        
        # 3. Every other species appears exactly once in test pairs with s
        test_i = i_idx[test_mask]
        test_j = j_idx[test_mask]
        # Test pairs are either (s, j) or (i, s)
        test_species = np.concatenate([test_i[test_i != s_idx], test_j[test_j != s_idx]])
        unique_test_species, counts = np.unique(test_species, return_counts=True)
        
        if len(unique_test_species) != n_species - 1:
            raise AssertionError(
                f"Expected {n_species - 1} unique species in test pairs, "
                f"got {len(unique_test_species)}"
            )
        
        if not np.all(counts == 1):
            raise AssertionError(
                f"Some species appear multiple times in test pairs: {unique_test_species[counts != 1]}"
            )
        report.write(f"  [OK] Each other species appears exactly once in test pairs")
    
    report.write("\n[OK] All LOOCV pair mask checks passed")


def check_feature_scaling(
    X_norm: np.ndarray,
    D: np.ndarray,
    species: List[str],
    held_out_idx: int,
    report: ReportWriter,
) -> None:
    """Check feature scaling and out-of-distribution issues."""
    report.write("\n" + "=" * 80)
    report.write("CHECK 4: Feature Magnitude and Scaling")
    report.write("=" * 80)
    
    n_species, n_domains = X_norm.shape
    held_out_species = species[held_out_idx]
    report.write(f"\nHeld-out species: {held_out_species} (index {held_out_idx})")
    
    # Training species: all except held-out
    train_indices = [i for i in range(n_species) if i != held_out_idx]
    train_species = [species[i] for i in train_indices]
    n_train = len(train_species)
    
    # Build training pairs using pdist (same as LOOCV script)
    X_train_subset = X_norm[train_indices, :]  # (n_train × n_domains)
    n_train_pairs = n_train * (n_train - 1) // 2
    
    X_train = np.zeros((n_train_pairs, n_domains), dtype=float)
    for domain_idx in range(n_domains):
        domain_values = X_train_subset[:, domain_idx].reshape(-1, 1)
        squared_diffs = pdist(domain_values, metric="sqeuclidean")
        X_train[:, domain_idx] = squared_diffs
    
    # Build test pairs: (held_out, train_species) for all train_species
    X_held_out = X_norm[held_out_idx, :]  # (n_domains,)
    X_train_species = X_norm[train_indices, :]  # (n_train × n_domains)
    
    # Vectorized: (X_held_out - X_train_species)^2
    diff = X_held_out[np.newaxis, :] - X_train_species  # (n_train, n_domains)
    X_test = (diff ** 2)  # (n_train, n_domains)
    
    report.write(f"  Training pairs: {n_train_pairs}")
    report.write(f"  Test pairs: {len(X_test)}")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Report statistics
    max_train_abs = float(np.max(np.abs(X_train_scaled)))
    max_test_abs = float(np.max(np.abs(X_test_scaled)))
    
    report.write(f"\nScaled feature statistics:")
    report.write(f"  max|scaled_X_train|: {max_train_abs:.2f}")
    report.write(f"  max|scaled_X_test|:  {max_test_abs:.2f}")
    
    # Check for extreme z-scores in test set
    test_abs = np.abs(X_test_scaled)
    pct_gt_10 = 100.0 * np.sum(test_abs > 10) / test_abs.size
    pct_gt_25 = 100.0 * np.sum(test_abs > 25) / test_abs.size
    pct_gt_50 = 100.0 * np.sum(test_abs > 50) / test_abs.size
    
    report.write(f"\nTest set extreme z-scores:")
    report.write(f"  |z| > 10:  {pct_gt_10:.2f}%")
    report.write(f"  |z| > 25:  {pct_gt_25:.2f}%")
    report.write(f"  |z| > 50:  {pct_gt_50:.2f}%")
    
    if pct_gt_10 > 5.0:
        warnings.warn(
            f"High percentage ({pct_gt_10:.2f}%) of test features have |z| > 10 - "
            "possible out-of-distribution issue"
        )
        report.write(
            f"\nWARNING: {pct_gt_10:.2f}% of test features have |z| > 10 - "
            "test set may be out-of-distribution relative to training"
        )
    
    # Per-pair L2 norm quantiles
    train_l2 = np.linalg.norm(X_train_scaled, axis=1)
    test_l2 = np.linalg.norm(X_test_scaled, axis=1)
    
    train_l2_quantiles = np.percentile(train_l2, [5, 25, 50, 75, 95])
    test_l2_quantiles = np.percentile(test_l2, [5, 25, 50, 75, 95])
    
    report.write(f"\nPer-pair L2 norm quantiles:")
    report.write(f"  Training: 5%={train_l2_quantiles[0]:.2f}, 25%={train_l2_quantiles[1]:.2f}, "
                 f"50%={train_l2_quantiles[2]:.2f}, 75%={train_l2_quantiles[3]:.2f}, "
                 f"95%={train_l2_quantiles[4]:.2f}")
    report.write(f"  Test:     5%={test_l2_quantiles[0]:.2f}, 25%={test_l2_quantiles[1]:.2f}, "
                 f"50%={test_l2_quantiles[2]:.2f}, 75%={test_l2_quantiles[3]:.2f}, "
                 f"95%={test_l2_quantiles[4]:.2f}")
    
    if np.median(test_l2) > 2 * np.median(train_l2):
        warnings.warn(
            f"Test L2 norms are much larger than training: "
            f"median(test)={np.median(test_l2):.2f} vs median(train)={np.median(train_l2):.2f}"
        )
        report.write(
            f"\nWARNING: Test L2 norms are much larger than training - "
            "possible distribution shift"
        )


def create_plots(
    D: np.ndarray,
    species: List[str],
    held_out_indices: List[int],
    report: ReportWriter,
) -> None:
    """Create optional diagnostic plots."""
    if not HAS_MATPLOTLIB:
        report.write("\n" + "=" * 80)
        report.write("PLOTS: Skipped (matplotlib not available)")
        report.write("=" * 80)
        return
    
    report.write("\n" + "=" * 80)
    report.write("PLOTS: Generating diagnostic plots")
    report.write("=" * 80)
    
    n = D.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    off_diag = D[i_idx, j_idx]
    
    # Histogram of off-diagonal distances
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(off_diag, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Patristic Distance", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Off-Diagonal Distances", fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HIST_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    report.write(f"\nSaved histogram to: {HIST_PLOT}")
    
    # Boxplot of test pair distances for held-out species
    test_distances_by_species = []
    species_labels = []
    
    for s_idx in held_out_indices:
        train_mask = (i_idx != s_idx) & (j_idx != s_idx)
        test_mask = (i_idx == s_idx) | (j_idx == s_idx)
        test_dists = D[i_idx[test_mask], j_idx[test_mask]]
        test_distances_by_species.append(test_dists)
        species_labels.append(f"{species[s_idx][:20]}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(test_distances_by_species, labels=species_labels, patch_artist=True)
    ax.set_ylabel("Patristic Distance", fontsize=12)
    ax.set_xlabel("Held-Out Species", fontsize=12)
    ax.set_title("Test Pair Distance Distributions (per Held-Out Species)", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(BOX_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    report.write(f"Saved boxplot to: {BOX_PLOT}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main() -> None:
    """Run all sanity checks."""
    import datetime
    
    # Initialize report writer
    with ReportWriter(REPORT_FILE) as report:
        report.write("=" * 80)
        report.write("LOOCV SANITY CHECKS")
        report.write("=" * 80)
        report.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.write("")
        
        # Configuration summary
        report.write("CONFIGURATION:")
        report.write(f"  Domain TSV: {DOMAIN_TSV}")
        report.write(f"  Tree NWK:   {TREE_NWK}")
        report.write(f"  Exclude outgroup: {EXCLUDE_OUTGROUP}")
        report.write(f"  Outgroup name: {OUTGROUP_NAME}")
        report.write(f"  Random seed: {RANDOM_SEED}")
        report.write("")
        
        # Load and preprocess data
        report.write("=" * 80)
        report.write("DATA LOADING AND PREPROCESSING")
        report.write("=" * 80)
        report.write("Loading domain counts and tree...")
        
        try:
            X_norm, D, species, tree = preprocess_data(
                DOMAIN_TSV,
                TREE_NWK,
                exclude_outgroup=EXCLUDE_OUTGROUP,
                outgroup_name=OUTGROUP_NAME,
            )
        except Exception as e:
            report.write(f"\nERROR during data loading: {e}")
            raise
        
        n_species, n_domains = X_norm.shape
        report.write(f"\nLoaded data:")
        report.write(f"  Species: {n_species}")
        report.write(f"  Domains: {n_domains}")
        report.write(f"  Distance matrix shape: {D.shape}")
        
        # Select held-out species for checks
        if CHECK_HELDOUT_SPECIES is None:
            np.random.seed(RANDOM_SEED)
            random_indices = np.random.choice(n_species, min(3, n_species), replace=False).tolist()
            held_out_indices = sorted(random_indices + [0, n_species - 1])  # 3 random + first + last
            held_out_indices = sorted(list(set(held_out_indices)))  # Remove duplicates
        else:
            held_out_indices = CHECK_HELDOUT_SPECIES
        
        report.write(f"\nHeld-out species indices for checks: {held_out_indices}")
        
        # Run sanity checks
        try:
            check_distance_scale(D, report)
            check_species_alignment(D, species, tree, report)
            check_loocv_pair_masks(n_species, held_out_indices, report)
            check_feature_scaling(X_norm, D, species, held_out_indices[0], report)
            create_plots(D, species, held_out_indices, report)
        except AssertionError as e:
            report.write(f"\n\nASSERTION FAILED: {e}")
            raise
        except Exception as e:
            report.write(f"\n\nERROR during sanity checks: {e}")
            raise
        
        # Final summary
        report.write("\n" + "=" * 80)
        report.write("SUMMARY")
        report.write("=" * 80)
        report.write("All sanity checks completed successfully!")
        report.write(f"Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.write(f"\nReport saved to: {REPORT_FILE}")
        if HAS_MATPLOTLIB:
            report.write(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
