#!/usr/bin/env python3
"""
Pairwise evolutionary-distance classification with class-imbalance handling
==========================================================================

Standalone script: species-blocked cross-validation, manual distance binning
schemes, pairwise features from absolute differences |freq_i - freq_j| per domain
(same construction as manual_distance_binning_search.py / cityblock pdist), and
comparison of baseline vs class-weighting vs oversampling
(RandomOverSampler / optional SMOTE) applied only inside each training fold
(after scaling), with test data never resampled.

Outputs go to: results/pairwise_classification_oversampling/

Run from project root:
  python3 scripts/pairwise_distance_classification_oversampling.py [options]

Runtime (roughly proportional to: schemes × training modes × folds × n_estimators):
  - Default uses n_estimators=200 (faster than manual_binning_search.py at 500).
  - Use --match-manual-search for n_estimators=500 parity with manual binning search.
  - Use --quick-schemes to evaluate only 5 representative schemes instead of all 22.
  - Use --n-folds 3 for a quicker CV (less stable estimates than 5).
  - Omit --enable-smote unless you need SMOTE; it is slow on high-dimensional features.

Design vs manual_distance_binning_search.py: same species list, domain matrix, patristic distances,
manual cutoffs, and species-blocked KFold. By default, each fold tests on all pairs involving at
least one held-out species (test-test and test-train), trains only on train-train pairs, then
aggregates to one prediction per unordered pair (mean predict_proba across folds, then argmax).
Use --legacy-test-cv for the older evaluation (test-test pairs only). This script adds per-fold
StandardScaler and optional imbalanced-learn resampling on training data only (as specified for the
oversampling study).
"Balanced accuracy" in CSV/JSON is macro-averaged recall over all classes with a fixed label
list (sklearn's multiclass balanced-accuracy definition); needed because some folds' test splits
omit a class entirely.

Dependencies: pandas, numpy, scipy, scikit-learn, matplotlib, ete3;
optional but recommended: imbalanced-learn (RandomOverSampler, SMOTE),
seaborn (nicer heatmaps).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

matplotlib = __import__("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    RandomOverSampler = None  # type: ignore[misc, assignment]
    SMOTE = None  # type: ignore[misc, assignment]

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
TREE_PATH = os.path.join(PROJECT_ROOT, "MammalsPhylogeny.nwk")
DOMAIN_COUNTS_PATH = os.path.join(DATA_RAW, "MammalDomainCount.tsv")
SPECIES_LIST_PATH = os.path.join(DATA_RAW, "MammalsList.txt")

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "pairwise_classification_oversampling")

NORMALIZATION_EPS = 1e-12
EXCLUDE_PLATYPUS = True

RANDOM_STATE = 42
N_FOLDS = 5
# manual_distance_binning_search.py uses 500 trees; default below is lower for reasonable runtime.
DEFAULT_N_ESTIMATORS = 200
MANUAL_SEARCH_N_ESTIMATORS = 500

# Used with --quick-schemes: 3-, 4-, and 5-bin representatives (skip long full grid).
QUICK_SCHEME_NAMES: Tuple[str, ...] = (
    "SCHEME_3A",
    "SCHEME_4B",
    "SCHEME_4C",
    "SCHEME_4E",
    "SCHEME_5A",
)

# Match manual binning search: skip schemes with very small global classes
MIN_CLASS_COUNT_THRESHOLD = 25

# Training modes (logical names used in outputs and summary)
MODE_BASELINE = "baseline"  # no oversampling, class_weight=None
MODE_BALANCED_WEIGHT = "class_weight_balanced"  # no oversampling, class_weight="balanced"
MODE_RANDOM_OS = "random_oversample"  # RandomOverSampler + class_weight=None
MODE_SMOTE = "smote"  # SMOTE + class_weight=None (optional)


def log(msg: str) -> None:
    print(f"[pairwise_oversampling] {msg}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Data loading and preprocessing (aligned with manual_distance_binning_search)
# -----------------------------------------------------------------------------


def load_species_list(path: str) -> List[str]:
    species: List[str] = []
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

    mapped_labels = [lab for lab in X_raw.index if lab in mapping]
    X_raw = X_raw.loc[mapped_labels]
    X_raw.index = [mapping[lab] for lab in X_raw.index]

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
            for sp in species_list:
                if mapped.lower() == sp.lower():
                    tree_species_map[leaf_name_orig] = sp
                    break

    for leaf in list(tree.iter_leaves()):
        if leaf.name in tree_species_map:
            leaf.name = tree_species_map[leaf.name]
        else:
            leaf.detach()

    available_species = [sp for sp in species_list if sp in {x.name for x in tree.iter_leaves()}]
    tree.prune(available_species, preserve_branch_length=True)

    n = len(available_species)
    T_matrix = np.zeros((n, n), dtype=float)
    for i, sp_i in enumerate(available_species):
        for j in range(i + 1, n):
            sp_j = available_species[j]
            dist = tree.get_distance(sp_i, sp_j)
            T_matrix[i, j] = T_matrix[j, i] = dist

    T_df = pd.DataFrame(T_matrix, index=available_species, columns=available_species)
    log(f"  Computed distance matrix for {n} species")
    return T_df


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load domain counts and phylogeny, intersect species, optionally drop platypus,
    drop zero-variance domains, normalize rows to relative frequencies.
    Returns X_norm, dist_matrix, species_final (same pipeline as manual binning search).
    """
    log("=" * 80)
    log("Load and preprocess")
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

    domain_variance = X_counts.var(axis=0)
    zero_var_mask = domain_variance == 0
    if zero_var_mask.any():
        log(f"  Dropping {int(zero_var_mask.sum())} zero-variance domains")
        X_counts = X_counts.loc[:, ~zero_var_mask]

    row_sums = X_counts.sum(axis=1).values.reshape(-1, 1)
    X_norm_arr = X_counts.values.astype(float) / (row_sums + NORMALIZATION_EPS)
    X_norm = pd.DataFrame(X_norm_arr, index=X_counts.index, columns=X_counts.columns)

    log(f"  Final: {len(species_final)} species x {X_norm.shape[1]} domains")
    return X_norm, dist_matrix, species_final


# -----------------------------------------------------------------------------
# Pairwise features: absolute differences per domain (upper triangle order)
# -----------------------------------------------------------------------------


def construct_pairwise_features(
    X_norm: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    species_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each domain d, feature value for pair (i,j) is |freq_i - freq_j|
    (pdist with cityblock on a single column - identical to build_pairwise_abs_diff
    in manual_distance_binning_search.py). Same pair order as np.triu_indices.

    Returns:
        X_pairs: (n_pairs, n_domains)
        y_dist: (n_pairs,) patristic distances
        pair_sp1, pair_sp2: species names per row
    """
    species_list = sorted(species_list)
    n_species = len(species_list)
    n_domains = X_norm.shape[1]

    if set(species_list) - set(dist_matrix.index):
        raise ValueError("Species missing from distance matrix")

    X_subset = X_norm.loc[species_list].values
    dist_subset = dist_matrix.loc[species_list, species_list].values

    n_pairs = n_species * (n_species - 1) // 2
    X_pairs = np.zeros((n_pairs, n_domains), dtype=float)

    for d in range(n_domains):
        col = X_subset[:, d].reshape(-1, 1)
        # Single dimension: cityblock distance = |a - b|
        X_pairs[:, d] = pdist(col, metric="cityblock")

    i_idx, j_idx = np.triu_indices(n_species, k=1)
    y_dist = dist_subset[i_idx, j_idx]
    pair_sp1 = np.array([species_list[i] for i in i_idx])
    pair_sp2 = np.array([species_list[j] for j in j_idx])

    if np.isnan(X_pairs).any() or np.isinf(X_pairs).any():
        raise ValueError("NaN/inf in feature matrix")
    if np.isnan(y_dist).any() or np.isinf(y_dist).any():
        raise ValueError("NaN/inf in distance vector")

    return X_pairs, y_dist, pair_sp1, pair_sp2


# -----------------------------------------------------------------------------
# Species-blocked CV (same design as manual_distance_binning_search)
# -----------------------------------------------------------------------------


def build_species_blocked_fold_indices(
    species_list: List[str],
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
    extended_test_cv: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, frozenset, frozenset]]:
    """
    Return list of (train_pair_idx, test_pair_idx, train_species, test_species).

    Training pairs are always train-train (both species in the fold's training set).

    If extended_test_cv (default): test pairs are test-test ∪ test-train (any pair with at least
    one species in the fold's test set). If False (legacy): test pairs are test-test only.
    """
    species_arr = np.array(sorted(species_list))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds: List[Tuple[np.ndarray, np.ndarray, frozenset, frozenset]] = []

    for _, (tr_sp_idx, te_sp_idx) in enumerate(kf.split(species_arr)):
        train_species = frozenset(species_arr[tr_sp_idx])
        test_species = frozenset(species_arr[te_sp_idx])
        if train_species & test_species:
            raise RuntimeError("Train/test species overlap; leakage would occur.")

        train_mask = np.array(
            [(a in train_species and b in train_species) for a, b in zip(pair_sp1, pair_sp2)],
            dtype=bool,
        )
        if extended_test_cv:
            test_mask = np.array(
                [(a in test_species or b in test_species) for a, b in zip(pair_sp1, pair_sp2)],
                dtype=bool,
            )
        else:
            test_mask = np.array(
                [(a in test_species and b in test_species) for a, b in zip(pair_sp1, pair_sp2)],
                dtype=bool,
            )
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0], train_species, test_species))

    return folds


def canonical_pair_key(sp1: str, sp2: str) -> Tuple[str, str]:
    return (sp1, sp2) if sp1 <= sp2 else (sp2, sp1)


def pair_cv_role(train_species: Set[str], test_species: Set[str], sp1: str, sp2: str) -> str:
    """Label for extended CV: test_test or test_train."""
    t1, t2 = sp1 in test_species, sp2 in test_species
    if t1 and t2:
        return "test_test"
    return "test_train"


def mean_patristic_per_class(y_class: np.ndarray, y_dist: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class mean true patristic distance (global, all pairs)."""
    means = np.full(n_classes, np.nan, dtype=float)
    y_class = y_class.astype(int)
    for k in range(n_classes):
        mask = y_class == k
        if np.any(mask):
            means[k] = float(np.mean(y_dist[mask]))
    return means


def align_proba_to_full_matrix(
    clf: RandomForestClassifier,
    proba: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Map clf.predict_proba columns to full n_classes (zeros for classes absent from training)."""
    n_samples = proba.shape[0]
    out = np.zeros((n_samples, n_classes), dtype=float)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = proba[:, j]
    return out


# -----------------------------------------------------------------------------
# Manual binning schemes (same definitions as manual_distance_binning_search)
# -----------------------------------------------------------------------------


def build_manual_schemes() -> Dict[str, Dict[str, Any]]:
    return {
        "SCHEME_3A": {"cutoffs": [0, 150, 188, np.inf], "labels": ["D0", "D1", "D2"]},
        "SCHEME_3B": {"cutoffs": [0, 160, 190, np.inf], "labels": ["D0", "D1", "D2"]},
        "SCHEME_3C": {"cutoffs": [0, 140, 185, np.inf], "labels": ["D0", "D1", "D2"]},
        "SCHEME_4A": {"cutoffs": [0, 140, 170, 190, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4B": {"cutoffs": [0, 150, 175, 190, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4C": {"cutoffs": [0, 150, 180, 195, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4D": {"cutoffs": [0, 130, 160, 188, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_5A": {"cutoffs": [0, 120, 150, 175, 190, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5B": {"cutoffs": [0, 140, 160, 180, 195, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_4E": {"cutoffs": [0, 50, 100, 150, 2000], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4F": {"cutoffs": [0, 60, 120, 170, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4G": {"cutoffs": [0, 70, 130, 175, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4H": {"cutoffs": [0, 80, 140, 180, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4I": {"cutoffs": [0, 90, 145, 185, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4J": {"cutoffs": [0, 100, 150, 188, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_5C": {"cutoffs": [0, 50, 100, 140, 175, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5D": {"cutoffs": [0, 60, 110, 150, 180, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5E": {"cutoffs": [0, 70, 120, 160, 188, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5F": {"cutoffs": [0, 80, 130, 165, 190, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_5G": {"cutoffs": [0, 90, 140, 170, 195, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4"]},
        "SCHEME_6A": {"cutoffs": [0, 50, 90, 130, 160, 188, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4", "D5"]},
        "SCHEME_6B": {"cutoffs": [0, 60, 100, 140, 170, 190, np.inf], "labels": ["D0", "D1", "D2", "D3", "D4", "D5"]},
    }


def assign_classes(
    y_dist: np.ndarray,
    cutoffs: List[float],
    labels: List[str],
) -> Tuple[np.ndarray, int, List[int]]:
    """Integer labels 0..K-1 from manual cutoffs (left-inclusive bins)."""
    n_classes = len(labels)
    if len(cutoffs) != n_classes + 1:
        raise ValueError("cutoffs must have length n_classes + 1")

    y_class = pd.cut(
        y_dist,
        bins=cutoffs,
        right=False,
        include_lowest=True,
        labels=False,
    )
    y_arr = np.asarray(y_class)
    if pd.isna(y_arr).any():
        raise ValueError("Some distances did not fall into manual bin edges.")

    y_arr = y_arr.astype(np.intp)
    class_counts = [int((y_arr == k).sum()) for k in range(n_classes)]
    return y_arr, n_classes, class_counts


# -----------------------------------------------------------------------------
# Oversamplers (training fold only, after scaling)
# -----------------------------------------------------------------------------


def get_oversampler(
    method: str,
    random_state: int,
) -> Optional[Any]:
    """
    Return None for modes without resampling, or an imblearn sampler instance.
    method: MODE_RANDOM_OS or MODE_SMOTE
    """
    if method == MODE_RANDOM_OS:
        if not HAS_IMBLEARN:
            raise RuntimeError("imbalanced-learn required for RandomOverSampler")
        return RandomOverSampler(random_state=random_state)
    if method == MODE_SMOTE:
        if not HAS_IMBLEARN:
            raise RuntimeError("imbalanced-learn required for SMOTE")
        # High-dimensional pairwise features: default k_neighbors; failures handled per fold
        return SMOTE(random_state=random_state, k_neighbors=5)
    return None


def class_counts_vector(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Count per class 0..n_classes-1."""
    return np.bincount(y.astype(int), minlength=n_classes)


def plot_train_counts_before_after(
    counts_before: np.ndarray,
    counts_after: np.ndarray,
    class_labels: List[str],
    title: str,
    out_path: str,
) -> None:
    """Grouped bar chart: original vs after resampling (after may equal before for no-op)."""
    x = np.arange(len(class_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(class_labels) * 0.9), 4.5))
    ax.bar(x - w / 2, counts_before, width=w, label="train (before)", edgecolor="black")
    ax.bar(x + w / 2, counts_after, width=w, label="train (after resample)", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# One fold: scale, optional resample, train RF, predict test
# -----------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold_idx: int
    train_counts_original: np.ndarray
    train_counts_after: np.ndarray
    test_counts: np.ndarray
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    y_true: np.ndarray
    y_pred: np.ndarray
    skip_reason: str = ""
    test_global_indices: Optional[np.ndarray] = None
    y_pred_proba_full: Optional[np.ndarray] = None


def run_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_mode: str,
    n_classes: int,
    random_state: int,
    n_estimators: int,
    test_global_indices: Optional[np.ndarray] = None,
    return_proba: bool = False,
) -> Optional[FoldResult]:
    """
    Fit StandardScaler on X_train; transform train/test.
    Apply oversampling only to scaled training data when mode requests it.
    """
    if len(np.unique(y_train)) < 2:
        log(f"    Fold skipped ({training_mode}): training subset has < 2 classes.")
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    counts_orig = class_counts_vector(y_train, n_classes)
    counts_after = counts_orig.copy()
    y_tr = y_train
    X_tr_s = X_train_s

    cw = None
    if training_mode == MODE_BALANCED_WEIGHT:
        cw = "balanced"

    sampler: Optional[Any] = None
    if training_mode == MODE_RANDOM_OS:
        sampler = get_oversampler(MODE_RANDOM_OS, random_state=random_state)
    elif training_mode == MODE_SMOTE:
        sampler = get_oversampler(MODE_SMOTE, random_state=random_state)

    skip_reason = ""

    if sampler is not None:
        try:
            X_tr_s, y_tr = sampler.fit_resample(X_train_s, y_train)
            counts_after = class_counts_vector(y_tr, n_classes)
        except Exception as e:
            skip_reason = f"{type(e).__name__}: {e}"
            log(f"    Resampling failed ({training_mode}): {skip_reason}")
            log(f"    Traceback:\n{traceback.format_exc()}")
            return None

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        class_weight=cw,
        n_jobs=-1,
    )
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_test_s)
    proba_full: Optional[np.ndarray] = None
    if return_proba:
        proba_raw = clf.predict_proba(X_test_s)
        proba_full = align_proba_to_full_matrix(clf, proba_raw, n_classes)

    label_idx = list(range(n_classes))
    # Macro recall with fixed label set matches multiclass balanced accuracy used in practice
    # when some test folds omit a class (species-blocked CV).
    bal = float(
        recall_score(y_test, y_pred, labels=label_idx, average="macro", zero_division=0)
    )
    macro = float(f1_score(y_test, y_pred, labels=label_idx, average="macro", zero_division=0))
    wtd = float(f1_score(y_test, y_pred, labels=label_idx, average="weighted", zero_division=0))

    return FoldResult(
        fold_idx=0,  # set by caller
        train_counts_original=counts_orig,
        train_counts_after=counts_after,
        test_counts=class_counts_vector(y_test, n_classes),
        balanced_accuracy=bal,
        macro_f1=macro,
        weighted_f1=wtd,
        y_true=y_test.copy(),
        y_pred=y_pred.copy(),
        skip_reason=skip_reason,
        test_global_indices=None if test_global_indices is None else test_global_indices.copy(),
        y_pred_proba_full=proba_full,
    )


# -----------------------------------------------------------------------------
# CV loop for one scheme + one training mode
# -----------------------------------------------------------------------------


def run_cv_for_scheme(
    X: np.ndarray,
    y_class: np.ndarray,
    y_dist: np.ndarray,
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    species_list: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray, frozenset, frozenset]],
    scheme_name: str,
    class_labels: List[str],
    training_mode: str,
    out_dir: str,
    n_estimators: int,
    extended_test_cv: bool = True,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Species-blocked CV with per-fold scaling and train-only oversampling.
    Extended mode: test = test-test ∪ test-train; pool with mean predict_proba then argmax per pair.
    Saves per-fold metrics CSV, count plots, pooled confusion matrix and report, aggregated CSVs,
    and predicted distance matrix.
    """
    n_classes = len(class_labels)
    n_pairs_expected = X.shape[0]
    fold_rows: List[Dict[str, Any]] = []
    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []
    fold_summaries: List[FoldResult] = []
    per_fold_pred_rows: List[Dict[str, Any]] = []

    proba_sum: Dict[Tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(n_classes, dtype=float))
    proba_count: Dict[Tuple[str, str], int] = defaultdict(int)
    y_true_by_key: Dict[Tuple[str, str], int] = {}

    ensure_dir(out_dir)

    for fold_idx, (tr_idx, te_idx, train_species, test_species) in enumerate(folds, start=1):
        X_train, y_train = X[tr_idx], y_class[tr_idx]
        X_test, y_test = X[te_idx], y_class[te_idx]

        c_train_orig = class_counts_vector(y_train, n_classes)
        c_test = class_counts_vector(y_test, n_classes)

        log(
            f"  [{scheme_name}] [{training_mode}] fold {fold_idx}: "
            f"train class counts (original) = {c_train_orig.tolist()}, "
            f"test class counts = {c_test.tolist()}"
        )

        fr = run_one_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            training_mode,
            n_classes,
            random_state=random_state + fold_idx,
            n_estimators=n_estimators,
            test_global_indices=te_idx,
            return_proba=extended_test_cv,
        )

        if fr is None:
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "status": "skipped",
                    "reason": "resample_failed_or_insufficient_classes",
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                }
            )
            continue

        fr.fold_idx = fold_idx
        fold_summaries.append(fr)

        if extended_test_cv:
            assert fr.test_global_indices is not None and fr.y_pred_proba_full is not None
            gidx = fr.test_global_indices
            prob = fr.y_pred_proba_full
            for row_i, gi in enumerate(gidx):
                sp1, sp2 = str(pair_sp1[gi]), str(pair_sp2[gi])
                key = canonical_pair_key(sp1, sp2)
                proba_sum[key] += prob[row_i]
                proba_count[key] += 1
                yt = int(y_class[gi])
                if key in y_true_by_key and y_true_by_key[key] != yt:
                    raise RuntimeError(f"Inconsistent y_true for pair {key}")
                y_true_by_key[key] = yt
                role = pair_cv_role(train_species, test_species, sp1, sp2)
                per_fold_pred_rows.append(
                    {
                        "fold": fold_idx,
                        "sp1": sp1,
                        "sp2": sp2,
                        "pair_type": role,
                        "y_true": yt,
                        "y_pred": int(fr.y_pred[row_i]),
                    }
                )
        else:
            all_y_true.append(fr.y_true)
            all_y_pred.append(fr.y_pred)

        log(
            f"  [{scheme_name}] [{training_mode}] fold {fold_idx}: "
            f"train counts after resample = {fr.train_counts_after.tolist()}, "
            f"balanced_acc={fr.balanced_accuracy:.4f}, macro_f1={fr.macro_f1:.4f}"
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "status": "ok",
                "train_counts_original": fr.train_counts_original.tolist(),
                "train_counts_after": fr.train_counts_after.tolist(),
                "test_counts": fr.test_counts.tolist(),
                "balanced_accuracy": fr.balanced_accuracy,
                "macro_f1": fr.macro_f1,
                "weighted_f1": fr.weighted_f1,
            }
        )

        plot_title = f"{scheme_name} | {training_mode} | fold {fold_idx}"
        plot_path = os.path.join(out_dir, f"class_counts_fold{fold_idx}.png")
        plot_train_counts_before_after(
            fr.train_counts_original.astype(float),
            fr.train_counts_after.astype(float),
            class_labels,
            plot_title,
            plot_path,
        )

    folds_csv = os.path.join(out_dir, "fold_metrics.csv")
    pd.DataFrame(fold_rows).to_csv(folds_csv, index=False)

    if per_fold_pred_rows:
        pd.DataFrame(per_fold_pred_rows).to_csv(
            os.path.join(out_dir, "predictions_per_fold.csv"), index=False
        )

    if not fold_summaries:
        log(f"  [{scheme_name}] [{training_mode}]: no successful folds; skipping pooled metrics.")
        return {
            "scheme_name": scheme_name,
            "training_mode": training_mode,
            "n_successful_folds": 0,
            "pooled_balanced_accuracy": np.nan,
            "pooled_macro_f1": np.nan,
            "pooled_weighted_f1": np.nan,
            "mean_fold_balanced_accuracy": np.nan,
            "mean_fold_macro_f1": np.nan,
            "report_dict": {},
        }

    methodology_note = (
        "Species-blocked CV; train pairs = train-train only; StandardScaler fit per fold on train; "
        "oversampling (if any) on train only after scaling. "
        "Test pairs = all pairs with ≥1 held-out species (test-test ∪ test-train). "
        "Pooled metrics: one row per unordered species pair; duplicate fold predictions merged by "
        "averaging predict_proba then argmax."
        if extended_test_cv
        else (
            "Species-blocked CV (legacy test-test only); StandardScaler fit per fold on train; "
            "oversampling (if any) on train only after scaling. "
            "Pooled metrics concatenate per-fold test predictions (no duplicate pairs across folds)."
        )
    )

    if extended_test_cv:
        if len(y_true_by_key) != n_pairs_expected:
            raise RuntimeError(
                f"Aggregation coverage: got {len(y_true_by_key)} unique pairs, expected {n_pairs_expected}"
            )
        for key, cnt in proba_count.items():
            if cnt < 1:
                raise RuntimeError(f"Invalid proba count for {key}")
        y_true_agg = np.empty(n_pairs_expected, dtype=int)
        y_pred_agg = np.empty(n_pairs_expected, dtype=int)
        agg_rows: List[Dict[str, Any]] = []
        mean_d_per_class = mean_patristic_per_class(y_class, y_dist, n_classes)

        species_ordered = sorted(species_list)
        species_to_i = {s: i for i, s in enumerate(species_ordered)}
        n_sp = len(species_ordered)
        dist_mat = np.zeros((n_sp, n_sp), dtype=float)

        pair_idx = 0
        for i in range(n_sp):
            for j in range(i + 1, n_sp):
                sp_i, sp_j = species_ordered[i], species_ordered[j]
                key = canonical_pair_key(sp_i, sp_j)
                mean_p = proba_sum[key] / float(proba_count[key])
                pred_c = int(np.argmax(mean_p))
                yt = y_true_by_key[key]
                y_true_agg[pair_idx] = yt
                y_pred_agg[pair_idx] = pred_c
                pair_idx += 1
                pred_dist = float(mean_d_per_class[pred_c]) if not np.isnan(mean_d_per_class[pred_c]) else 0.0
                dist_mat[i, j] = dist_mat[j, i] = pred_dist
                agg_rows.append(
                    {
                        "sp1": key[0],
                        "sp2": key[1],
                        "n_folds_averaged": proba_count[key],
                        "y_true": yt,
                        "y_pred": pred_c,
                        "y_true_label": class_labels[yt],
                        "y_pred_label": class_labels[pred_c],
                        "predicted_distance_MY": pred_dist,
                    }
                )

        pd.DataFrame(agg_rows).to_csv(os.path.join(out_dir, "predictions_aggregated.csv"), index=False)
        dist_df = pd.DataFrame(dist_mat, index=species_ordered, columns=species_ordered)
        dist_df.to_csv(os.path.join(out_dir, "distance_matrix_predicted.csv"))
        np.save(os.path.join(out_dir, "distance_matrix_predicted.npy"), dist_mat)

        metrics_extra = {
            "extended_test_cv": True,
            "aggregation_rule": "mean_predict_proba_then_argmax",
            "n_aggregated_pairs": int(n_pairs_expected),
            "mean_patristic_MY_per_class": [float(x) if not np.isnan(x) else None for x in mean_d_per_class],
        }

        y_true, y_pred = y_true_agg, y_pred_agg
    else:
        for fname in (
            "predictions_aggregated.csv",
            "predictions_per_fold.csv",
            "distance_matrix_predicted.csv",
            "distance_matrix_predicted.npy",
        ):
            stale = os.path.join(out_dir, fname)
            if os.path.isfile(stale):
                os.remove(stale)
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        metrics_extra = {
            "extended_test_cv": False,
            "aggregation_rule": None,
            "n_aggregated_pairs": int(len(y_true)),
            "mean_patristic_MY_per_class": None,
        }

    label_idx = list(range(n_classes))
    pooled_bal = float(
        recall_score(y_true, y_pred, labels=label_idx, average="macro", zero_division=0)
    )
    pooled_macro = float(f1_score(y_true, y_pred, labels=label_idx, average="macro", zero_division=0))
    pooled_wtd = float(f1_score(y_true, y_pred, labels=label_idx, average="weighted", zero_division=0))

    mean_bal = float(np.mean([f.balanced_accuracy for f in fold_summaries]))
    mean_macro = float(np.mean([f.macro_f1 for f in fold_summaries]))

    report_str = classification_report(
        y_true,
        y_pred,
        labels=label_idx,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_idx,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    save_metrics_and_plots(
        scheme_name=scheme_name,
        training_mode=training_mode,
        out_dir=out_dir,
        class_labels=class_labels,
        y_true=y_true,
        y_pred=y_pred,
        cm=cm,
        report_str=report_str,
        report_dict=report_dict,
        pooled_balanced_accuracy=pooled_bal,
        pooled_macro_f1=pooled_macro,
        pooled_weighted_f1=pooled_wtd,
        mean_fold_balanced_accuracy=mean_bal,
        mean_fold_macro_f1=mean_macro,
        fold_summaries=fold_summaries,
        methodology_note=methodology_note,
        metrics_extra=metrics_extra,
    )

    return {
        "scheme_name": scheme_name,
        "training_mode": training_mode,
        "n_successful_folds": len(fold_summaries),
        "pooled_balanced_accuracy": pooled_bal,
        "pooled_macro_f1": pooled_macro,
        "pooled_weighted_f1": pooled_wtd,
        "mean_fold_balanced_accuracy": mean_bal,
        "mean_fold_macro_f1": mean_macro,
        "report_dict": report_dict,
        "confusion_matrix": cm,
    }


def save_metrics_and_plots(
    scheme_name: str,
    training_mode: str,
    out_dir: str,
    class_labels: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    report_str: str,
    report_dict: Dict[str, Any],
    pooled_balanced_accuracy: float,
    pooled_macro_f1: float,
    pooled_weighted_f1: float,
    mean_fold_balanced_accuracy: float,
    mean_fold_macro_f1: float,
    fold_summaries: List[FoldResult],
    methodology_note: str = "",
    metrics_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write classification report, JSON metrics, confusion matrix heatmap, aggregate count plot."""
    n_classes = len(class_labels)
    if not methodology_note:
        methodology_note = (
            "Species-blocked CV; StandardScaler fit per fold on train only; "
            "oversampling (if any) on train only after scaling."
        )

    # Text report
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Scheme: {scheme_name}\nTraining mode: {training_mode}\n")
        f.write(f"{methodology_note}\n")
        f.write(
            "Balanced accuracy = macro-averaged recall over all classes (fixed label set); "
            "folds with no test samples for a class contribute 0 recall for that class.\n"
        )
        f.write("=" * 72 + "\n\n")
        f.write(f"Pooled balanced accuracy: {pooled_balanced_accuracy:.6f}\n")
        f.write(f"Pooled macro F1: {pooled_macro_f1:.6f}\n")
        f.write(f"Pooled weighted F1: {pooled_weighted_f1:.6f}\n")
        f.write(f"Mean fold balanced accuracy: {mean_fold_balanced_accuracy:.6f}\n")
        f.write(f"Mean fold macro F1: {mean_fold_macro_f1:.6f}\n\n")
        f.write("Classification report (pooled test predictions):\n")
        f.write(report_str)

    # JSON: scalar metrics + per-class precision/recall/f1/support
    per_class: Dict[str, Any] = {}
    for name in class_labels:
        if name in report_dict:
            per_class[name] = report_dict[name]

    metrics_json: Dict[str, Any] = {
        "scheme_name": scheme_name,
        "training_mode": training_mode,
        "pooled_balanced_accuracy": pooled_balanced_accuracy,
        "pooled_macro_f1": pooled_macro_f1,
        "pooled_weighted_f1": pooled_weighted_f1,
        "mean_fold_balanced_accuracy": mean_fold_balanced_accuracy,
        "mean_fold_macro_f1": mean_fold_macro_f1,
        "per_class": per_class,
        "folds": [
            {
                "fold": fr.fold_idx,
                "train_counts_original": fr.train_counts_original.tolist(),
                "train_counts_after": fr.train_counts_after.tolist(),
                "test_counts": fr.test_counts.tolist(),
                "balanced_accuracy": fr.balanced_accuracy,
                "macro_f1": fr.macro_f1,
                "weighted_f1": fr.weighted_f1,
            }
            for fr in fold_summaries
        ],
    }
    if metrics_extra:
        for k, v in metrics_extra.items():
            metrics_json[k] = v
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as jf:
        json.dump(metrics_json, jf, indent=2)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
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
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_labels, rotation=20, ha="right")
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_labels)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    cm_sub = "pooled, one prediction per pair" if metrics_extra and metrics_extra.get("extended_test_cv") else "pooled"
    ax.set_title(f"Confusion matrix ({cm_sub})\n{scheme_name} | {training_mode}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_labels], columns=[f"pred_{c}" for c in class_labels])
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # Aggregate: mean train counts before/after across folds (for oversampling visualization)
    if fold_summaries:
        bef = np.mean([f.train_counts_original for f in fold_summaries], axis=0)
        aft = np.mean([f.train_counts_after for f in fold_summaries], axis=0)
        plot_train_counts_before_after(
            bef,
            aft,
            class_labels,
            f"Mean train class counts across folds\n{scheme_name} | {training_mode}",
            os.path.join(out_dir, "class_counts_mean_across_folds.png"),
        )


def format_cutoffs_for_csv(cutoffs: List[float]) -> str:
    parts = []
    for c in cutoffs:
        parts.append("inf" if np.isinf(c) else f"{float(c):.6g}")
    return "[" + ", ".join(parts) + "]"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairwise distance classification with train-fold oversampling (species-blocked CV)."
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"RandomForest tree count. Default: {DEFAULT_N_ESTIMATORS} (faster). "
            f"Use --match-manual-search for {MANUAL_SEARCH_N_ESTIMATORS} (same as manual_binning_search)."
        ),
    )
    p.add_argument(
        "--match-manual-search",
        action="store_true",
        help=f"Set n_estimators={MANUAL_SEARCH_N_ESTIMATORS} to match manual_distance_binning_search.py (overrides default).",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        metavar="K",
        help=f"Species-blocked KFold splits (default {N_FOLDS}). Use 3 for faster runs.",
    )
    p.add_argument(
        "--quick-schemes",
        action="store_true",
        help=(
            "Evaluate only a small set of representative schemes "
            f"{list(QUICK_SCHEME_NAMES)} instead of all manual schemes (large time save)."
        ),
    )
    p.add_argument(
        "--enable-smote",
        action="store_true",
        help="Also run SMOTE (slow on high-D features; omit for faster runs unless you need it).",
    )
    p.add_argument(
        "--schemes",
        nargs="*",
        default=None,
        help="Optional subset of scheme names (e.g. SCHEME_4B). Default: all manual schemes.",
    )
    p.add_argument(
        "--legacy-test-cv",
        action="store_true",
        help=(
            "Use legacy test split: only pairs with both species in the held-out set (test-test). "
            "Default is extended CV: test-test ∪ test-train, with mean predict_proba aggregation per pair."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_estimators is not None:
        n_estimators = args.n_estimators
    elif args.match_manual_search:
        n_estimators = MANUAL_SEARCH_N_ESTIMATORS
    else:
        n_estimators = DEFAULT_N_ESTIMATORS

    n_folds = args.n_folds
    if n_folds < 2:
        raise SystemExit("--n-folds must be at least 2.")

    if not HAS_IMBLEARN:
        log("=" * 80)
        log(
            "Package `imbalanced-learn` is not installed. "
            "Install with: pip install imbalanced-learn"
        )
        log("Continuing with baseline and class_weight='balanced' only (no RandomOverSampler/SMOTE).")
        log("=" * 80)

    ensure_dir(OUTPUT_ROOT)

    X_norm, dist_matrix, species_final = load_and_preprocess_data()
    X, y_dist, pair_sp1, pair_sp2 = construct_pairwise_features(
        X_norm, dist_matrix, species_final
    )
    log(f"Pairwise dataset: X={X.shape}, y_dist={y_dist.shape} (abs-diff features)")

    n_species = len(species_final)
    if n_folds > n_species:
        raise SystemExit(f"--n-folds ({n_folds}) cannot exceed number of species ({n_species}).")

    extended_test_cv = not bool(args.legacy_test_cv)
    folds = build_species_blocked_fold_indices(
        species_final,
        pair_sp1,
        pair_sp2,
        n_folds=n_folds,
        random_state=RANDOM_STATE,
        extended_test_cv=extended_test_cv,
    )
    if extended_test_cv:
        log(
            "Species-blocked CV: extended test (test-test ∪ test-train); "
            "pooled metrics use one prediction per pair (mean predict_proba -> argmax)."
        )
    else:
        log("Species-blocked CV: legacy test-test only.")

    schemes = build_manual_schemes()
    if args.schemes:
        wanted = set(args.schemes)
        schemes = {k: v for k, v in schemes.items() if k in wanted}
        missing = wanted - set(schemes.keys())
        if missing:
            log(f"Warning: unknown scheme names ignored: {sorted(missing)}")
    elif args.quick_schemes:
        schemes = {k: v for k, v in schemes.items() if k in QUICK_SCHEME_NAMES}
        missing_q = set(QUICK_SCHEME_NAMES) - set(schemes.keys())
        if missing_q:
            log(f"Warning: quick-schemes missing from manual dict: {sorted(missing_q)}")

    enable_smote = bool(args.enable_smote and HAS_IMBLEARN)
    run_cfg = {
        "n_estimators": n_estimators,
        "n_folds": n_folds,
        "extended_test_cv": extended_test_cv,
        "legacy_test_cv": bool(args.legacy_test_cv),
        "quick_schemes": bool(args.quick_schemes) and not bool(args.schemes),
        "enable_smote_requested": bool(args.enable_smote),
        "enable_smote_ran": enable_smote,
        "scheme_count": len(schemes),
        "random_state": RANDOM_STATE,
    }
    cfg_path = os.path.join(OUTPUT_ROOT, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as cf:
        json.dump(run_cfg, cf, indent=2)
    log(
        f"Run config: n_estimators={n_estimators}, n_folds={n_folds}, "
        f"{len(schemes)} scheme(s), SMOTE={'on' if enable_smote else 'off'} "
        f"(saved {cfg_path})"
    )

    training_modes: List[str] = [MODE_BASELINE, MODE_BALANCED_WEIGHT]
    if HAS_IMBLEARN:
        training_modes.append(MODE_RANDOM_OS)
        if enable_smote:
            training_modes.append(MODE_SMOTE)
    elif args.enable_smote:
        log("SMOTE requested but imbalanced-learn is missing; skipping SMOTE.")

    summary_rows: List[Dict[str, Any]] = []

    for scheme_name, spec in schemes.items():
        cutoffs = [float(x) for x in spec["cutoffs"]]
        labels = list(spec["labels"])

        try:
            y_class, n_classes, class_counts = assign_classes(y_dist, cutoffs, labels)
        except Exception as e:
            log(f"Skipping {scheme_name}: binning failed: {e}")
            for tm in training_modes:
                summary_rows.append(
                    {
                        "scheme_name": scheme_name,
                        "training_mode": tm,
                        "n_estimators": n_estimators,
                        "n_folds": n_folds,
                        "smote_enabled_run": enable_smote,
                        "cutoffs": format_cutoffs_for_csv(cutoffs),
                        "n_classes": len(labels),
                        "skipped": True,
                        "skip_reason": f"binning: {e}",
                        "pooled_balanced_accuracy": np.nan,
                        "pooled_macro_f1": np.nan,
                        "pooled_weighted_f1": np.nan,
                        "mean_fold_balanced_accuracy": np.nan,
                        "mean_fold_macro_f1": np.nan,
                    }
                )
            continue

        min_c = min(class_counts)
        if min_c < MIN_CLASS_COUNT_THRESHOLD:
            reason = f"global min class count {min_c} < {MIN_CLASS_COUNT_THRESHOLD}"
            log(f"Skipping {scheme_name}: {reason}")
            for tm in training_modes:
                summary_rows.append(
                    {
                        "scheme_name": scheme_name,
                        "training_mode": tm,
                        "n_estimators": n_estimators,
                        "n_folds": n_folds,
                        "smote_enabled_run": enable_smote,
                        "cutoffs": format_cutoffs_for_csv(cutoffs),
                        "n_classes": n_classes,
                        "skipped": True,
                        "skip_reason": reason,
                        "pooled_balanced_accuracy": np.nan,
                        "pooled_macro_f1": np.nan,
                        "pooled_weighted_f1": np.nan,
                        "mean_fold_balanced_accuracy": np.nan,
                        "mean_fold_macro_f1": np.nan,
                    }
                )
            continue

        scheme_dir = os.path.join(OUTPUT_ROOT, scheme_name)
        ensure_dir(scheme_dir)

        for tm in training_modes:
            mode_dir = os.path.join(scheme_dir, tm)
            ensure_dir(mode_dir)

            log("=" * 80)
            log(f"Running {scheme_name} | {tm}")
            log("=" * 80)

            result = run_cv_for_scheme(
                X=X,
                y_class=y_class,
                y_dist=y_dist,
                pair_sp1=pair_sp1,
                pair_sp2=pair_sp2,
                species_list=species_final,
                folds=folds,
                scheme_name=scheme_name,
                class_labels=labels,
                training_mode=tm,
                out_dir=mode_dir,
                n_estimators=n_estimators,
                extended_test_cv=extended_test_cv,
                random_state=RANDOM_STATE,
            )

            summary_rows.append(
                {
                    "scheme_name": scheme_name,
                    "training_mode": tm,
                    "n_estimators": n_estimators,
                    "n_folds": n_folds,
                    "smote_enabled_run": enable_smote,
                    "cutoffs": format_cutoffs_for_csv(cutoffs),
                    "n_classes": n_classes,
                    "skipped": result["n_successful_folds"] == 0,
                    "skip_reason": ""
                    if result["n_successful_folds"] > 0
                    else "all_folds_failed_or_skipped",
                    "n_successful_folds": result["n_successful_folds"],
                    "pooled_balanced_accuracy": result["pooled_balanced_accuracy"],
                    "pooled_macro_f1": result["pooled_macro_f1"],
                    "pooled_weighted_f1": result["pooled_weighted_f1"],
                    "mean_fold_balanced_accuracy": result["mean_fold_balanced_accuracy"],
                    "mean_fold_macro_f1": result["mean_fold_macro_f1"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_ROOT, "summary_all_schemes_and_modes.csv")
    summary_df.to_csv(summary_path, index=False)
    log(f"Saved summary table: {summary_path}")

    # Pivot-style comparison: schemes as rows, modes as columns (balanced acc)
    if not summary_df.empty:
        pivot_bal = summary_df.pivot_table(
            index="scheme_name",
            columns="training_mode",
            values="pooled_balanced_accuracy",
            aggfunc="first",
        )
        pivot_macro = summary_df.pivot_table(
            index="scheme_name",
            columns="training_mode",
            values="pooled_macro_f1",
            aggfunc="first",
        )
        pivot_bal.to_csv(os.path.join(OUTPUT_ROOT, "summary_pivot_pooled_balanced_accuracy.csv"))
        pivot_macro.to_csv(os.path.join(OUTPUT_ROOT, "summary_pivot_pooled_macro_f1.csv"))
        log(f"Saved pivot tables under {OUTPUT_ROOT}")

    log("Done.")


if __name__ == "__main__":
    main()
