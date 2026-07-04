#!/usr/bin/env python3
"""
Manual Distance-Binning Search for Timetree Classifier
=========================================================

Standalone script that replaces quantile-based binning with manually defined
patristic-distance cutoffs and evaluates each cutoff scheme using
species-blocked 5-fold cross-validation with a Random Forest classifier.

Key idea:
- Quantile binning failed because many patristic distances are tied around ~188 MY,
  which collapses bins under qcut().
- Manual cutoffs let us place bin boundaries to better match the empirical
  distance distribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit(
        "This script requires the `ete3` package. Please install it and retry."
    ) from exc

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
TREE_PATH = os.path.join(PROJECT_ROOT, "MammalsPhylogeny.nwk")
DOMAIN_COUNTS_PATH = os.path.join(DATA_RAW, "MammalDomainCount.tsv")
SPECIES_LIST_PATH = os.path.join(DATA_RAW, "MammalsList.txt")

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "manual_binning_search")
os.makedirs(OUT_DIR, exist_ok=True)

NORMALIZATION_EPS = 1e-12
EXCLUDE_PLATYPUS = True

RANDOM_STATE = 42
N_FOLDS = 5
N_ESTIMATORS = 500

FEATURE_METRIC = "abs_diff"  # plan requirement: same as best q=4 abs_diff run

# Skip schemes that create degenerate classes
MIN_CLASS_COUNT_THRESHOLD = 25


def log(msg: str) -> None:
    print(f"[manual_binning_search] {msg}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_cutoffs_for_csv(cutoffs: List[float]) -> str:
    # Keep readable: replace inf with string.
    parts = []
    for c in cutoffs:
        if np.isinf(c):
            parts.append("inf")
        else:
            parts.append(f"{float(c):.6g}")
    return "[" + ", ".join(parts) + "]"


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------


def load_species_list(path: str) -> List[str]:
    species = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                species.append(stripped)
    return species


def load_domain_counts(path: str, species_order: List[str]) -> pd.DataFrame:
    log(f"Loading domain counts from {path}")
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0, low_memory=False)
    X_raw = df.transpose()
    X_raw.index.name = "species"
    # Remove parenthetical descriptions after the first "("
    X_raw.index = [name.split("(")[0].strip() for name in X_raw.index]

    target_set = set(species_order)
    mapping: Dict[str, str] = {}
    for label in X_raw.index:
        if label in target_set:
            mapping[label] = label
        else:
            for sp in species_order:
                if label.lower() == sp.lower():
                    mapping[label] = sp
                    break

    mapped_labels = [l for l in X_raw.index if l in mapping]
    X_raw = X_raw.loc[mapped_labels]
    X_raw.index = [mapping[l] for l in X_raw.index]

    if X_raw.index.duplicated().any():
        log("  Warning: Duplicate species after mapping; averaging.")
        X_raw = X_raw.groupby(X_raw.index).mean()

    missing = set(species_order) - set(X_raw.index)
    if missing:
        log(f"  Warning: {len(missing)} species missing from domain counts (ignored).")
        species_order = [sp for sp in species_order if sp in X_raw.index]

    X_raw = X_raw.loc[species_order]
    log(f"  Loaded {X_raw.shape[0]} species x {X_raw.shape[1]} domains")
    return X_raw


def compute_patristic_matrix(tree_path: str, species_list: List[str]) -> pd.DataFrame:
    log("Computing full patristic distance matrix...")
    tree = Tree(tree_path, format=1)

    # Reconcile known tip naming mismatches
    tree_name_map = {
        "Neogale vison": "Neovison vison",
        "Neogale_vison": "Neovison vison",
        "Bos grunniens": "Bos mutus grunniens",
        "Bos_grunniens": "Bos mutus grunniens",
        "Physeter catodon": "Physeter macrocephalus",
        "Physeter_catodon": "Physeter macrocephalus",
    }

    species_set = set(species_list)
    tree_species_map: Dict[str, str] = {}

    for leaf in tree.iter_leaves():
        leaf_name_orig = leaf.name
        leaf_name = leaf_name_orig.replace("_", " ")

        if leaf_name_orig in tree_name_map:
            mapped = tree_name_map[leaf_name_orig]
        elif leaf_name in tree_name_map:
            mapped = tree_name_map[leaf_name]
        else:
            mapped = leaf_name

        if mapped in species_set:
            tree_species_map[leaf_name_orig] = mapped
        else:
            # case-insensitive fallback
            for sp in species_list:
                if mapped.lower() == sp.lower():
                    tree_species_map[leaf_name_orig] = sp
                    break

    # Rename and prune
    for leaf in list(tree.iter_leaves()):
        if leaf.name in tree_species_map:
            leaf.name = tree_species_map[leaf.name]
        else:
            leaf.detach()

    available_species = [sp for sp in species_list if sp in {leaf.name for leaf in tree.iter_leaves()}]
    tree.prune(available_species, preserve_branch_length=True)

    n = len(available_species)
    T_matrix = np.zeros((n, n), dtype=float)
    for i, sp_i in enumerate(available_species):
        for j in range(i + 1, n):
            sp_j = available_species[j]
            T_matrix[i, j] = T_matrix[j, i] = tree.get_distance(sp_i, sp_j)

    T_df = pd.DataFrame(T_matrix, index=available_species, columns=available_species)
    log(f"  Computed distance matrix for {n} species")
    return T_df


def load_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    log("=" * 80)
    log("Step 1: Load and preprocess")
    log("=" * 80)

    species_from_list = load_species_list(SPECIES_LIST_PATH)
    X_counts = load_domain_counts(DOMAIN_COUNTS_PATH, species_from_list)
    domain_species = list(X_counts.index)

    dist_matrix_full = compute_patristic_matrix(TREE_PATH, domain_species)
    tree_species = list(dist_matrix_full.index)

    species_intersection = sorted(set(domain_species) & set(tree_species))
    X_counts = X_counts.loc[species_intersection]
    dist_matrix = dist_matrix_full.loc[species_intersection, species_intersection]

    if EXCLUDE_PLATYPUS:
        outgroup_name = "Ornithorhynchus anatinus"
        if outgroup_name in X_counts.index:
            log(f"  Excluding {outgroup_name} (outgroup)")
            X_counts = X_counts.drop(index=outgroup_name)
            dist_matrix = dist_matrix.drop(index=outgroup_name, columns=outgroup_name)

    species_final = list(X_counts.index)

    # Drop zero-variance domains
    domain_variance = X_counts.var(axis=0)
    zero_var_mask = domain_variance == 0
    if zero_var_mask.any():
        log(f"  Dropping {int(zero_var_mask.sum())} zero-variance domains")
        X_counts = X_counts.loc[:, ~zero_var_mask]

    # Normalize to relative frequencies
    row_sums = X_counts.sum(axis=1).values.reshape(-1, 1)
    X_norm = X_counts.values.astype(float) / (row_sums + NORMALIZATION_EPS)
    X_norm = pd.DataFrame(X_norm, index=X_counts.index, columns=X_counts.columns)

    log(f"  Final: {len(species_final)} species x {X_norm.shape[1]} domains")
    return X_norm, dist_matrix, species_final


# -----------------------------------------------------------------------------
# Pairwise dataset
# -----------------------------------------------------------------------------


def build_pairwise_abs_diff_dataset(
    X_norm: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    species_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n_pairs, n_domains) features = |freq_i - freq_j|
      y_dist: (n_pairs,) patristic distances
      pair_sp1, pair_sp2: (n_pairs,) species name for each pair
    """
    species_list = sorted(species_list)
    n_species = len(species_list)
    n_domains = X_norm.shape[1]

    X_subset = X_norm.loc[species_list].values  # (n_species, n_domains)
    dist_subset = dist_matrix.loc[species_list, species_list].values

    n_pairs = n_species * (n_species - 1) // 2
    X_pairs = np.zeros((n_pairs, n_domains), dtype=float)

    for d in range(n_domains):
        # For a single domain, pdist with cityblock equals absolute difference
        col = X_subset[:, d].reshape(-1, 1)
        X_pairs[:, d] = pdist(col, metric="cityblock")

    i_idx, j_idx = np.triu_indices(n_species, k=1)
    y_dist = dist_subset[i_idx, j_idx]

    pair_sp1 = np.array([species_list[i] for i in i_idx])
    pair_sp2 = np.array([species_list[j] for j in j_idx])

    if np.isnan(X_pairs).any() or np.isinf(X_pairs).any():
        raise ValueError("NaN/inf detected in feature matrix")
    if np.isnan(y_dist).any() or np.isinf(y_dist).any():
        raise ValueError("NaN/inf detected in distance targets")

    return X_pairs, y_dist, pair_sp1, pair_sp2


def build_species_blocked_fold_indices(
    species_list: List[str],
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    x_shape: Tuple[int, ...],
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns list of (train_pair_idx, test_pair_idx) for species-blocked CV.
    Mixed pairs (train species vs test species) are excluded.
    """
    species_arr = np.array(species_list)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for _, (train_idx, test_idx) in enumerate(kf.split(species_arr)):
        train_species = set(species_arr[train_idx])
        test_species = set(species_arr[test_idx])
        if not train_species.isdisjoint(test_species):
            raise RuntimeError("Train/test species overlap detected; leakage would occur.")

        # Boolean masks over pairs
        train_mask = np.array(
            [(sp1 in train_species and sp2 in train_species) for sp1, sp2 in zip(pair_sp1, pair_sp2)],
            dtype=bool,
        )
        test_mask = np.array(
            [(sp1 in test_species and sp2 in test_species) for sp1, sp2 in zip(pair_sp1, pair_sp2)],
            dtype=bool,
        )

        train_pair_idx = np.where(train_mask)[0]
        test_pair_idx = np.where(test_mask)[0]
        folds.append((train_pair_idx, test_pair_idx))

    return folds


# -----------------------------------------------------------------------------
# Manual binning schemes
# -----------------------------------------------------------------------------


def build_manual_schemes() -> Dict[str, Dict[str, Any]]:
    schemes: Dict[str, Dict[str, Any]] = {
        # 3-bin candidates
        "SCHEME_3A": {"cutoffs": [0, 150, 188, np.inf], "labels": ["D0", "D1", "D2"]},
        "SCHEME_3B": {"cutoffs": [0, 160, 190, np.inf], "labels": ["D0", "D1", "D2"]},
        "SCHEME_3C": {"cutoffs": [0, 140, 185, np.inf], "labels": ["D0", "D1", "D2"]},
        # 4-bin candidates
        "SCHEME_4A": {"cutoffs": [0, 140, 170, 190, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4B": {"cutoffs": [0, 150, 175, 190, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4C": {"cutoffs": [0, 150, 180, 195, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4D": {"cutoffs": [0, 130, 160, 188, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        # Optional 5-bin candidates
        "SCHEME_5A": {"cutoffs": [0, 120, 150, 175, 190, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5B": {"cutoffs": [0, 140, 160, 180, 195, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        # Variant emphasizing shorter-distance resolution.
        "SCHEME_4E": {"cutoffs": [0, 50, 100, 150, 2000], "labels": ["D0", "D1", "D2", "D3"]},
        # Additional short-distance-focused 4-bin variants
        "SCHEME_4F": {"cutoffs": [0, 60, 120, 170, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4G": {"cutoffs": [0, 70, 130, 175, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4H": {"cutoffs": [0, 80, 140, 180, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4I": {"cutoffs": [0, 90, 145, 185, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4J": {"cutoffs": [0, 100, 150, 188, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        # Additional short-distance-focused 5-bin variants
        "SCHEME_5C": {"cutoffs": [0, 50, 100, 140, 175, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5D": {"cutoffs": [0, 60, 110, 150, 180, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5E": {"cutoffs": [0, 70, 120, 160, 188, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5F": {"cutoffs": [0, 80, 130, 165, 190, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5G": {"cutoffs": [0, 90, 140, 170, 195, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        # 6-bin variants for finer granularity under 200 MY
        "SCHEME_6A": {"cutoffs": [0, 50, 90, 130, 160, 188, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4", "D5"]},
        "SCHEME_6B": {"cutoffs": [0, 60, 100, 140, 170, 190, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4", "D5"]},
    }
    return schemes


def apply_manual_binning(
    y_dist: np.ndarray,
    cutoffs: List[float],
    labels: List[str],
) -> Tuple[np.ndarray, int, List[int]]:
    """
    Returns:
      y_class: (n,) integer labels 0..n_classes-1
      n_classes: int
      class_counts: list of length n_classes
    """
    n_classes = len(labels)
    if len(cutoffs) != n_classes + 1:
        raise ValueError("cutoffs length must equal n_classes + 1")

    y_class = pd.cut(
        y_dist,
        bins=cutoffs,
        right=False,  # left-inclusive
        include_lowest=True,
        labels=False,
    )
    # pandas may return either a Series or ndarray depending on version/settings
    y_class_arr = np.asarray(y_class)
    if pd.isna(y_class_arr).any():
        # Some values did not fall in provided edges
        raise ValueError("Some y_dist values did not fall into manual bin edges.")

    y_class_arr = y_class_arr.astype(np.intp)
    class_counts = [(y_class_arr == k).sum() for k in range(n_classes)]
    return y_class_arr, n_classes, class_counts


def plot_class_counts(y_class: np.ndarray, scheme_name: str, labels: List[str], out_dir: str) -> None:
    counts = [(y_class == k).sum() for k in range(len(labels))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), counts, edgecolor="black", alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Count")
    ax.set_title(f"Class counts: {scheme_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"class_counts_{scheme_name}.png"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Evaluation per scheme
# -----------------------------------------------------------------------------


def evaluate_scheme_species_blocked_cv(
    X: np.ndarray,
    y_class: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    scheme_name: str,
    class_labels: List[str],
    out_dir: str,
) -> Dict[str, Any]:
    """
    Returns dict with metrics + confusion matrix + y/pred pooled.
    """
    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    for fold_idx, (train_pair_idx, test_pair_idx) in enumerate(folds, start=1):
        X_train = X[train_pair_idx]
        y_train = y_class[train_pair_idx]
        X_test = X[test_pair_idx]
        y_test = y_class[test_pair_idx]

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        all_y_true.append(y_test)
        all_y_pred.append(pred)

        log(f"    {scheme_name} fold {fold_idx}: train_pairs={len(y_train)}, test_pairs={len(y_test)}")

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # majority baseline (pooled)
    values, counts = np.unique(y_true, return_counts=True)
    majority_baseline = float(counts.max() / len(y_true)) if len(y_true) else 0.0
    acc_minus_baseline = float(acc - majority_baseline)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        n_c = len(class_labels)
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(class_labels)
        ax.set_yticks(range(n_c))
        ax.set_yticklabels(class_labels)
        for i in range(n_c):
            for j in range(n_c):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {scheme_name}")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{scheme_name}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Save classification report
    report_path = os.path.join(out_dir, f"classification_report_{scheme_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Species-blocked 5-fold CV classification report\nScheme: {scheme_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n")
        f.write(f"Majority baseline: {majority_baseline:.4f}\n")
        f.write(f"Accuracy - baseline: {acc_minus_baseline:.4f}\n\n")
        f.write(report_str)

    return {
        "scheme_name": scheme_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "majority_baseline": majority_baseline,
        "accuracy_minus_baseline": acc_minus_baseline,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "report_str": report_str,
    }


# -----------------------------------------------------------------------------
# Ranking, overlays, summaries
# -----------------------------------------------------------------------------


def select_best_schemes(
    comparison_rows: List[Dict[str, Any]],
    target_bins: int,
) -> Optional[Dict[str, Any]]:
    valid = [r for r in comparison_rows if r.get("is_valid") and r.get("n_classes") == target_bins]
    if not valid:
        return None
    # rank by balanced_accuracy then accuracy_minus_baseline
    valid.sort(key=lambda r: (r["balanced_accuracy"], r["accuracy_minus_baseline"]), reverse=True)
    return valid[0]


def plot_distance_histogram_with_cutoffs(
    y_dist: np.ndarray,
    best_3: Optional[Dict[str, Any]],
    best_4: Optional[Dict[str, Any]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_dist, bins=60, edgecolor="black", alpha=0.65)
    ax.set_xlabel("Patristic distance (MY)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distance histogram with best manual cutoffs")

    def add_scheme_lines(scheme_row: Dict[str, Any], color: str, label: str) -> None:
        # For plotting we want numeric cutoffs. In the comparison rows we store
        # both a CSV-friendly string and a numeric helper list.
        cutoffs = scheme_row.get("cutoffs_raw", scheme_row.get("cutoffs"))
        if not isinstance(cutoffs, list):
            raise TypeError(f"Expected numeric cutoffs list for plotting, got: {type(cutoffs)}")
        # internal boundaries are all cutoffs except the first and last
        internal = [c for c in cutoffs[1:-1] if not np.isinf(c)]
        for c in internal:
            ax.axvline(c, linestyle="--", linewidth=2.0, color=color, label=label)

    handles = []
    if best_3 is not None:
        add_scheme_lines(best_3, color="red", label=f"best {best_3['scheme_name']} (3-bin)")
    if best_4 is not None:
        add_scheme_lines(best_4, color="green", label=f"best {best_4['scheme_name']} (4-bin)")

    # Build a legend (unique labels)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_dir(OUT_DIR)

    # Load data
    X_norm, dist_matrix, species_final = load_and_preprocess()
    X, y_dist, pair_sp1, pair_sp2 = build_pairwise_abs_diff_dataset(X_norm, dist_matrix, species_final)

    # Sanity info
    log(f"Step 0: built pairwise dataset: X={X.shape}, y_dist={y_dist.shape}")

    # Save base distance histogram
    base_hist_path = os.path.join(OUT_DIR, "distance_histogram_base.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_dist, bins=60, edgecolor="black", alpha=0.65)
    ax.set_xlabel("Patristic distance (MY)")
    ax.set_ylabel("Frequency")
    ax.set_title("Off-diagonal pairwise distance histogram")
    plt.tight_layout()
    plt.savefig(base_hist_path, dpi=150)
    plt.close()
    log(f"  Saved {base_hist_path}")

    # Build species-blocked folds once (indices only; y_class differs by scheme)
    folds = build_species_blocked_fold_indices(
        species_list=sorted(species_final),
        pair_sp1=pair_sp1,
        pair_sp2=pair_sp2,
        x_shape=X.shape,
    )

    # Candidate manual schemes
    schemes = build_manual_schemes()

    comparison_rows: List[Dict[str, Any]] = []

    log("=" * 80)
    log(f"Step 1: Evaluating {len(schemes)} manual binning schemes")
    log("=" * 80)

    for scheme_name, spec in schemes.items():
        cutoffs = [float(x) for x in spec["cutoffs"]]
        labels = list(spec["labels"])

        log("-" * 80)
        log(f"Evaluating {scheme_name}")
        log(f"  Cutoffs: {cutoffs}")

        # Apply manual binning
        try:
            y_class, n_classes, class_counts = apply_manual_binning(y_dist, cutoffs, labels)
        except Exception as e:
            log(f"  Skipping {scheme_name}: failed to bin with error: {e}")
            comparison_rows.append(
                {
                    "scheme_name": scheme_name,
                    "cutoffs": format_cutoffs_for_csv(cutoffs),
                    "n_classes": len(labels),
                    "class_counts": str(class_counts) if "class_counts" in locals() else "",
                    "min_class_count": np.nan,
                    "max_class_count": np.nan,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "majority_baseline": np.nan,
                    "accuracy_minus_baseline": np.nan,
                    "is_valid": False,
                    "skip_reason": f"binning_failed: {e}",
                }
            )
            continue

        min_count = int(min(class_counts))
        max_count = int(max(class_counts))
        proportions = [c / len(y_class) for c in class_counts]

        log(f"  Class counts: {class_counts}")
        log(f"  Class proportions: {[round(p, 4) for p in proportions]}")

        # Save class-counts plot
        plot_class_counts(y_class, scheme_name, labels, OUT_DIR)

        # Validate class counts
        if min_count < MIN_CLASS_COUNT_THRESHOLD:
            reason = f"class_count_below_threshold (min={min_count}, threshold={MIN_CLASS_COUNT_THRESHOLD})"
            log(f"  Skipping {scheme_name}: {reason}")
            comparison_rows.append(
                {
                    "scheme_name": scheme_name,
                    "cutoffs": format_cutoffs_for_csv(cutoffs),
                    "n_classes": n_classes,
                    "class_counts": str(class_counts),
                    "min_class_count": min_count,
                    "max_class_count": max_count,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "majority_baseline": np.nan,
                    "accuracy_minus_baseline": np.nan,
                    "is_valid": False,
                    "skip_reason": reason,
                }
            )
            continue

        # Evaluate
        metrics = evaluate_scheme_species_blocked_cv(
            X=X,
            y_class=y_class,
            folds=folds,
            scheme_name=scheme_name,
            class_labels=labels,
            out_dir=OUT_DIR,
        )

        comparison_rows.append(
            {
                "scheme_name": scheme_name,
                "cutoffs": format_cutoffs_for_csv(cutoffs),
                "n_classes": n_classes,
                "class_counts": str(class_counts),
                "min_class_count": min_count,
                "max_class_count": max_count,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "majority_baseline": metrics["majority_baseline"],
                "accuracy_minus_baseline": metrics["accuracy_minus_baseline"],
                "is_valid": True,
                "skip_reason": "",
                # Keep raw cutoffs for plotting
                "cutoffs_raw": cutoffs,
            }
        )

    # Master CSV
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(OUT_DIR, "manual_binning_comparison.csv")
    # Drop helper column cutoffs_raw from CSV if present
    if "cutoffs_raw" in comp_df.columns:
        comp_df = comp_df.drop(columns=["cutoffs_raw"])
    comp_df.to_csv(comp_csv_path, index=False)
    log(f"Saved {comp_csv_path}")

    # Rank valid schemes
    valid_rows = [r for r in comparison_rows if r.get("is_valid")]
    valid_rows.sort(key=lambda r: (r["balanced_accuracy"], r["accuracy_minus_baseline"]), reverse=True)

    best = valid_rows[0] if valid_rows else None
    top3 = valid_rows[:3]

    log("=" * 80)
    log("Step 2: Ranking manual schemes")
    log("=" * 80)
    if best is None:
        log("No valid schemes passed the minimum class-count threshold.")
    else:
        log(f"Best scheme: {best['scheme_name']} with balanced_accuracy={best['balanced_accuracy']:.4f}")
        for i, r in enumerate(top3, start=1):
            log(f"  Top {i}: {r['scheme_name']} bal_acc={r['balanced_accuracy']:.4f} acc-b={r['accuracy_minus_baseline']:.4f}")

    # Choose best 3-bin and 4-bin for histogram overlay
    best_3 = None
    best_4 = None
    for r in valid_rows:
        if r.get("n_classes") == 3 and best_3 is None:
            best_3 = r
        if r.get("n_classes") == 4 and best_4 is None:
            best_4 = r
    # Ensure they are actually best within their group (valid_rows already sorted globally by ranking key)
    best_3 = select_best_schemes(valid_rows, 3)  # type: ignore[arg-type]
    best_4 = select_best_schemes(valid_rows, 4)  # type: ignore[arg-type]

    overlay_path = os.path.join(OUT_DIR, "distance_histogram_with_manual_cutoffs.png")
    plot_distance_histogram_with_cutoffs(y_dist, best_3, best_4, overlay_path)
    log(f"Saved {overlay_path}")

    # Write summary text
    summary_path = os.path.join(OUT_DIR, "manual_binning_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Manual distance-binning search summary\n")
        f.write("=========================================\n\n")
        f.write("Why quantile binning failed:\n")
        f.write("- qcut/pd.qcut creates equal-frequency bins.\n")
        f.write("- Your timetree pairwise distances have a large tied mass around ~188 MY.\n")
        f.write("- When multiple quantile boundaries land on the same tied value, pandas collapses bins\n")
        f.write("  (duplicates='drop'), resulting in fewer effective classes than requested.\n\n")

        f.write("Why manual binning was used:\n")
        f.write("- To place explicit distance cutoffs at empirically meaningful locations in the histogram.\n")
        f.write("- Manual cutoffs avoid equal-frequency constraints and let classes reflect visible peaks/valleys.\n\n")

        if best is None:
            f.write("No valid schemes passed thresholds.\n")
            f.write(f"Threshold used: MIN_CLASS_COUNT_THRESHOLD={MIN_CLASS_COUNT_THRESHOLD}\n")
        else:
            f.write("Best-performing schemes (species-blocked 5-fold RF):\n")
            f.write(f"- Best overall: {best['scheme_name']}\n")
            f.write(f"  Balanced accuracy: {best['balanced_accuracy']:.4f}\n")
            f.write(f"  Accuracy - majority baseline: {best['accuracy_minus_baseline']:.4f}\n")
            f.write(f"  Cutoffs: {best['cutoffs']}\n\n")

            f.write("Top 3 schemes:\n")
            for i, r in enumerate(top3, start=1):
                f.write(f"{i}. {r['scheme_name']} | n_classes={r['n_classes']} | bal_acc={r['balanced_accuracy']:.4f} | acc-b={r['accuracy_minus_baseline']:.4f}\n")
                f.write(f"   cutoffs={r['cutoffs']} | min_class_count={r['min_class_count']} | max_class_count={r['max_class_count']}\n")

            f.write("\nRecommended scheme for downstream tree reconstruction:\n")
            f.write("- Use the best scheme above (best overall by balanced_accuracy then accuracy_minus_baseline).\n")
            if best_3 is not None:
                f.write(f"- Best 3-bin scheme: {best_3['scheme_name']} ({best_3['cutoffs']})\n")
            if best_4 is not None:
                f.write(f"- Best 4-bin scheme: {best_4['scheme_name']} ({best_4['cutoffs']})\n")

    log(f"Saved {summary_path}")

    log("=" * 80)
    log("Done.")
    log("=" * 80)


if __name__ == "__main__":
    main()

