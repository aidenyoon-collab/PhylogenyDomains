#!/usr/bin/env python3
"""
Species-Blocked 5-Fold Random Forest Classifier + Quantile->Time Mapping + Tree Reconstruction

GOAL
1) Ensure cross-validation is SPECIES-LEVEL (no leakage across folds).
2) Use quantile binning of patristic distances (q=5) to create class labels.
3) Report and plot the MY range for each quantile.
4) Train RandomForestClassifier and evaluate with species-blocked folds.
5) Reconstruct a tree from predicted distances and compare reference.

Inputs data_raw/MammalDomainCount.tsv, data_raw/MammalsList.txt, MammalsPhylogeny.nwk
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import KFold

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit("This script requires the `ete3` package. Please install it and retry.") from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "species_blocked_rf_quantile_tree")

NORMALIZATION_EPS = 1e-12
EXCLUDE_PLATYPUS = True
RANDOM_STATE = 42
N_ESTIMATORS = 500
N_FOLDS = 5

# Both metrics are run and compared; outputs saved with _abs_diff and _sq_diff suffixes.
# abs_diff => |freq_i - freq_j|; sq_diff => (freq_i - freq_j)^2
FEATURE_METRICS = ("abs_diff", "sq_diff")


def log(message: str) -> None:
    print(f"[species_blocked_rf] {message}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_species_label_for_newick(species_name: str) -> str:
    s = species_name.replace(" ", "_").replace("(", "").replace(")", "")
    return s.replace(",", "").replace(";", "")


# -----------------------------------------------------------------------------
# Part A - Load and preprocess
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
    X_raw.index = [name.split("(")[0].strip() for name in X_raw.index]
    target_set = set(species_order)
    mapping = {}
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
    tree_species_map = {}
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
    tree_species = {leaf.name for leaf in tree.iter_leaves()}
    available_species = [sp for sp in species_list if sp in tree_species]
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


def load_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    log("=" * 80)
    log("Part A: Load and preprocess")
    log("=" * 80)
    species_from_list = load_species_list(SPECIES_LIST_PATH)
    X_counts = load_domain_counts(DOMAIN_COUNTS_PATH, species_from_list)
    domain_species = list(X_counts.index)
    dist_matrix_full = compute_patristic_matrix(TREE_PATH, domain_species)
    tree_species = list(dist_matrix_full.index)
    species_intersection = sorted(set(domain_species) & set(tree_species))
    X_counts = X_counts.loc[species_intersection]
    dist_matrix = dist_matrix_full.loc[species_intersection, species_intersection]
    outgroup_name = "Ornithorhynchus anatinus"
    if EXCLUDE_PLATYPUS and outgroup_name in X_counts.index:
        log(f"  Excluding {outgroup_name} (outgroup)")
        X_counts = X_counts.drop(index=outgroup_name)
        dist_matrix = dist_matrix.drop(index=outgroup_name, columns=outgroup_name)
    species_final = list(X_counts.index)
    log(f"  Final species: {len(species_final)}")
    domain_variance = X_counts.var(axis=0)
    zero_var_mask = domain_variance == 0
    if zero_var_mask.any():
        X_counts = X_counts.loc[:, ~zero_var_mask]
    log(f"  Domains after zero-var drop: {X_counts.shape[1]}")
    row_sums = X_counts.sum(axis=1).values.reshape(-1, 1)
    X_norm = X_counts.values.astype(float) / (row_sums + NORMALIZATION_EPS)
    X_norm = pd.DataFrame(X_norm, index=X_counts.index, columns=X_counts.columns)
    return X_norm, dist_matrix, species_final


# -----------------------------------------------------------------------------
# Part B - Pairwise dataset with metadata
# -----------------------------------------------------------------------------


def build_pairwise_features_and_metadata(
    X_norm: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    species_list: List[str],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X (n_pairs, n_domains), y_dist (n_pairs,), pair_sp1, pair_sp2. metric is 'abs_diff' or 'sq_diff'."""
    species_list = sorted(species_list)
    n_species = len(species_list)
    n_domains = X_norm.shape[1]
    X_subset = X_norm.loc[species_list].values
    dist_subset = dist_matrix.loc[species_list, species_list].values
    n_pairs = n_species * (n_species - 1) // 2
    X_pairs = np.zeros((n_pairs, n_domains), dtype=float)
    if metric == "abs_diff":
        for d in range(n_domains):
            col = X_subset[:, d].reshape(-1, 1)
            X_pairs[:, d] = pdist(col, metric="cityblock")
    else:
        for d in range(n_domains):
            col = X_subset[:, d].reshape(-1, 1)
            X_pairs[:, d] = pdist(col, metric="sqeuclidean")
    i_idx, j_idx = np.triu_indices(n_species, k=1)
    y_dist = dist_subset[i_idx, j_idx]
    pair_sp1 = np.array([species_list[i] for i in i_idx])
    pair_sp2 = np.array([species_list[j] for j in j_idx])
    if np.isnan(X_pairs).any() or np.isinf(X_pairs).any():
        raise ValueError("NaN/inf in feature matrix")
    if np.isnan(y_dist).any() or np.isinf(y_dist).any():
        raise ValueError("NaN/inf in target vector")
    return X_pairs, y_dist, pair_sp1, pair_sp2


# -----------------------------------------------------------------------------
# Part C - Quantile labels and time mapping
# -----------------------------------------------------------------------------


def build_quantile_labels_and_time_ranges(
    y_dist: np.ndarray,
    out_dir: str,
    q: int,
    suffix: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float], pd.DataFrame]:
    """Returns y_quant, quant_edges, mean_MY_per_quantile, and per-quantile stats DataFrame."""
    y_quant, bin_edges = pd.qcut(y_dist, q=q, labels=False, duplicates="drop", retbins=True)
    y_quant = y_quant.astype(np.intp)
    quant_edges = np.asarray(bin_edges)
    n_classes = int(y_quant.max()) + 1
    log(f"Quantile bin edges (q={q}):")
    for i in range(len(quant_edges) - 1):
        log(f"  Bin {i}: [{quant_edges[i]:.4f}, {quant_edges[i+1]:.4f}]")
    if n_classes < q:
        log(f"  Warning: qcut with duplicates='drop' yielded {n_classes} classes (requested {q})")
    rows = []
    mean_MY_per_quantile = {}
    for k in range(n_classes):
        mask = y_quant == k
        vals = y_dist[mask]
        n_pairs = int(mask.sum())
        min_my = float(np.min(vals))
        max_my = float(np.max(vals))
        mean_my = float(np.mean(vals))
        median_my = float(np.median(vals))
        mean_MY_per_quantile[k] = mean_my
        rows.append({"quantile": k, "n_pairs": n_pairs, "min_MY": min_my, "max_MY": max_my, "mean_MY": mean_my, "median_MY": median_my})
    df = pd.DataFrame(rows)
    csv_name = f"quantile_time_ranges{suffix}.csv"
    box_name = f"quantile_time_ranges_boxplot{suffix}.png"
    df.to_csv(os.path.join(out_dir, csv_name), index=False)
    log(f"  Saved {csv_name}")
    fig, ax = plt.subplots(figsize=(7, 4))
    data_by_q = [y_dist[y_quant == k] for k in range(n_classes)]
    ax.boxplot(data_by_q, tick_labels=[f"Q{k}" for k in range(n_classes)])
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Distance (MY)")
    ax.set_title(f"Distance distribution by quantile (q={q})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, box_name), dpi=150)
    plt.close()
    log(f"  Saved {box_name}")
    return y_quant, quant_edges, mean_MY_per_quantile, df


# -----------------------------------------------------------------------------
# Part D - Species-blocked 5-fold CV
# -----------------------------------------------------------------------------


def get_species_blocked_folds(
    species_list: List[str],
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Returns list of (X_train, y_train, X_test, y_test) per fold."""
    species_arr = np.array(species_list)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(species_arr)):
        train_species = set(species_arr[train_idx])
        test_species = set(species_arr[test_idx])
        assert train_species.isdisjoint(test_species), "Train and test species must not overlap"
        train_pair_mask = np.array([
            (sp1 in train_species and sp2 in train_species)
            for sp1, sp2 in zip(pair_sp1, pair_sp2)
        ])
        test_pair_mask = np.array([
            (sp1 in test_species and sp2 in test_species)
            for sp1, sp2 in zip(pair_sp1, pair_sp2)
        ])
        X_train = X[train_pair_mask]
        y_train = y[train_pair_mask]
        X_test = X[test_pair_mask]
        y_test = y[test_pair_mask]
        log(f"  Fold {fold_idx + 1}: train_pairs={len(y_train)}, test_pairs={len(y_test)}")
        folds.append((X_train, y_train, X_test, y_test))
    return folds


# -----------------------------------------------------------------------------
# Part E - Train and evaluate
# -----------------------------------------------------------------------------


def run_species_blocked_cv(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    class_labels: List[str],
    out_dir: str,
    suffix: str = "",
) -> Tuple[float, float, float]:
    """Returns accuracy, balanced_accuracy, majority_baseline. suffix e.g. '_abs_diff' or '_sq_diff' for filenames."""
    all_y_true = []
    all_y_pred = []
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    for X_train, y_train, X_test, y_test in folds:
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        all_y_true.append(y_test)
        all_y_pred.append(pred)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    from collections import Counter
    counts = Counter(y_true)
    majority_class = max(counts, key=counts.get)
    majority_baseline = counts[majority_class] / len(y_true) if y_true.size else 0.0
    report_str = classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report_fname = f"classification_report_species_blocked{suffix}.txt"
    cm_fname = f"confusion_matrix_species_blocked{suffix}.png"
    with open(os.path.join(out_dir, report_fname), "w") as f:
        f.write(f"Species-blocked 5-fold CV{suffix}\n")
        f.write("=======================\n\n")
        f.write(f"Feature metric: {suffix.strip('_').replace('_', ' ')}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n")
        f.write(f"Majority-class baseline: {majority_baseline:.4f}\n\n")
        f.write(report_str)
    log(f"  Accuracy: {acc:.4f}, Balanced accuracy: {bal_acc:.4f}, Baseline: {majority_baseline:.4f}")
    fig, ax = plt.subplots(figsize=(7, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    else:
        im = ax.imshow(cm, cmap="Blues")
        nc = len(class_labels)
        ax.set_xticks(range(nc))
        ax.set_xticklabels(class_labels)
        ax.set_yticks(range(nc))
        ax.set_yticklabels(class_labels)
        for i in range(nc):
            for j in range(nc):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (species-blocked 5-fold CV){suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, cm_fname), dpi=150)
    plt.close()
    log(f"  Saved {cm_fname} and {report_fname}")
    return acc, bal_acc, majority_baseline


# -----------------------------------------------------------------------------
# Part G - Tree reconstruction and comparison
# -----------------------------------------------------------------------------


def build_upgma_newick(distance_df: pd.DataFrame) -> str:
    """Build UPGMA tree from symmetric distance matrix; return Newick string."""
    species = list(distance_df.index)
    labels = [format_species_label_for_newick(s) for s in species]
    condensed = squareform(distance_df.values, checks=False)
    linkage_matrix = hierarchy.average(condensed)
    tree = hierarchy.to_tree(linkage_matrix, rd=False)

    def build_newick(node) -> Tuple[str, float]:
        if node.is_leaf():
            return labels[node.id], 0.0
        left_str, left_height = build_newick(node.left)
        right_str, right_height = build_newick(node.right)
        branch_left = max(node.dist - left_height, 0.0)
        branch_right = max(node.dist - right_height, 0.0)
        newick = f"({left_str}:{branch_left:.10f},{right_str}:{branch_right:.10f})"
        return newick, node.dist

    newick_str, _ = build_newick(tree)
    return f"{newick_str};"


def compute_patristic_distances_ete3(tree: Tree, species_list: List[str]) -> np.ndarray:
    """Flattened upper-triangle distances (i < j) for species_list order."""
    name_to_node = {leaf.name: leaf for leaf in tree.iter_leaves()}
    n = len(species_list)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            node1 = name_to_node.get(species_list[i])
            node2 = name_to_node.get(species_list[j])
            if node1 is None or node2 is None:
                distances.append(np.nan)
            else:
                distances.append(tree.get_distance(node1, node2))
    return np.array(distances)


def run_tree_reconstruction(
    X: np.ndarray,
    y_quant: np.ndarray,
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    species_list: List[str],
    mean_MY_per_quantile: Dict[int, float],
    out_dir: str,
    suffix: str = "",
) -> Tuple[float, float]:
    """Returns (pearson_r, spearman_r). Saves tree and comparison report with optional suffix."""
    log("=" * 80)
    log(f"Part G: Tree reconstruction from predicted distances{suffix}")
    log("=" * 80)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X, y_quant)
    pred_quant = clf.predict(X)
    n_species = len(species_list)
    D_pred = np.zeros((n_species, n_species))
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    for p, (sp1, sp2) in enumerate(zip(pair_sp1, pair_sp2)):
        q = int(pred_quant[p])
        mean_my = mean_MY_per_quantile.get(q, 0.0)
        i, j = species_to_idx[sp1], species_to_idx[sp2]
        if i > j:
            i, j = j, i
        D_pred[i, j] = D_pred[j, i] = mean_my
    D_df = pd.DataFrame(D_pred, index=species_list, columns=species_list)
    newick_str = build_upgma_newick(D_df)
    tree_fname = f"tree_from_predicted_distances{suffix}.nwk"
    tree_path = os.path.join(out_dir, tree_fname)
    with open(tree_path, "w") as f:
        f.write(newick_str)
    log(f"  Saved {tree_path}")

    ref_tree = Tree(TREE_PATH, format=1)
    tree_name_map = {
        "Neogale vison": "Neovison vison",
        "Neogale_vison": "Neovison vison",
        "Bos grunniens": "Bos mutus grunniens",
        "Bos_grunniens": "Bos mutus grunniens",
        "Physeter catodon": "Physeter macrocephalus",
        "Physeter_catodon": "Physeter macrocephalus",
    }
    species_set = set(species_list)
    for leaf in list(ref_tree.iter_leaves()):
        orig = leaf.name
        name = orig.replace("_", " ")
        if orig in tree_name_map:
            name = tree_name_map[orig]
        elif name in tree_name_map:
            name = tree_name_map[name]
        if name in species_set:
            leaf.name = name
        else:
            leaf.detach()
    ref_tree.prune(species_list, preserve_branch_length=True)
    ref_tree_species = sorted(species_list)
    recon_tree_species = sorted([format_species_label_for_newick(s) for s in species_list])
    recon_tree = Tree(newick_str, format=1)
    ref_dists = compute_patristic_distances_ete3(ref_tree, ref_tree_species)
    recon_dists = compute_patristic_distances_ete3(recon_tree, recon_tree_species)
    valid = ~(np.isnan(ref_dists) | np.isnan(recon_dists))
    if valid.sum() < 3:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
    else:
        pearson_r, pearson_p = pearsonr(ref_dists[valid], recon_dists[valid])
        spearman_r, spearman_p = spearmanr(ref_dists[valid], recon_dists[valid])
    report_fname = f"tree_comparison_report{suffix}.txt"
    report_path = os.path.join(out_dir, report_fname)
    with open(report_path, "w") as f:
        f.write(f"Reference tree vs tree from predicted distances (UPGMA){suffix}\n")
        f.write("======================================================\n\n")
        f.write(f"Pearson r:  {pearson_r:.4f}  p: {pearson_p:.4e}\n")
        f.write(f"Spearman r: {spearman_r:.4f}  p: {spearman_p:.4e}\n")
    log(f"  Pearson r: {pearson_r:.4f}, Spearman r: {spearman_r:.4f}")
    log(f"  Saved {report_path}")
    return (float(pearson_r) if not np.isnan(pearson_r) else np.nan, float(spearman_r) if not np.isnan(spearman_r) else np.nan)


def run_tree_reconstruction_probability_weighted(
    X: np.ndarray,
    y_quant: np.ndarray,
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    species_list: List[str],
    mean_MY_per_quantile: Dict[int, float],
    out_dir: str,
    suffix: str = "",
) -> Tuple[float, float]:
    """Expected-distance reconstruction using RF class probabilities. Additive outputs only."""
    log("=" * 80)
    log(f"Part G: Probability-weighted tree reconstruction{suffix}")
    log("=" * 80)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X, y_quant)
    proba = clf.predict_proba(X)  # shape: n_pairs x n_classes
    n_species = len(species_list)
    D_pred = np.zeros((n_species, n_species))
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    class_values = np.array(
        [mean_MY_per_quantile.get(k, 0.0) for k in range(proba.shape[1])],
        dtype=float,
    )
    expected_dist = proba @ class_values
    for p, (sp1, sp2) in enumerate(zip(pair_sp1, pair_sp2)):
        i, j = species_to_idx[sp1], species_to_idx[sp2]
        if i > j:
            i, j = j, i
        D_pred[i, j] = D_pred[j, i] = expected_dist[p]

    D_df = pd.DataFrame(D_pred, index=species_list, columns=species_list)
    newick_str = build_upgma_newick(D_df)
    tree_fname = f"tree_from_predicted_distances{suffix}.nwk"
    tree_path = os.path.join(out_dir, tree_fname)
    with open(tree_path, "w") as f:
        f.write(newick_str)
    log(f"  Saved {tree_path}")

    ref_tree = Tree(TREE_PATH, format=1)
    tree_name_map = {
        "Neogale vison": "Neovison vison",
        "Neogale_vison": "Neovison vison",
        "Bos grunniens": "Bos mutus grunniens",
        "Bos_grunniens": "Bos mutus grunniens",
        "Physeter catodon": "Physeter macrocephalus",
        "Physeter_catodon": "Physeter macrocephalus",
    }
    species_set = set(species_list)
    for leaf in list(ref_tree.iter_leaves()):
        orig = leaf.name
        name = orig.replace("_", " ")
        if orig in tree_name_map:
            name = tree_name_map[orig]
        elif name in tree_name_map:
            name = tree_name_map[name]
        if name in species_set:
            leaf.name = name
        else:
            leaf.detach()
    ref_tree.prune(species_list, preserve_branch_length=True)
    ref_tree_species = sorted(species_list)
    recon_tree_species = sorted([format_species_label_for_newick(s) for s in species_list])
    recon_tree = Tree(newick_str, format=1)
    ref_dists = compute_patristic_distances_ete3(ref_tree, ref_tree_species)
    recon_dists = compute_patristic_distances_ete3(recon_tree, recon_tree_species)
    valid = ~(np.isnan(ref_dists) | np.isnan(recon_dists))
    if valid.sum() < 3:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
    else:
        pearson_r, pearson_p = pearsonr(ref_dists[valid], recon_dists[valid])
        spearman_r, spearman_p = spearmanr(ref_dists[valid], recon_dists[valid])
    report_fname = f"tree_comparison_report{suffix}.txt"
    report_path = os.path.join(out_dir, report_fname)
    with open(report_path, "w") as f:
        f.write(f"Reference tree vs probability-weighted predicted distances (UPGMA){suffix}\n")
        f.write("====================================================================\n\n")
        f.write(f"Pearson r:  {pearson_r:.4f}  p: {pearson_p:.4e}\n")
        f.write(f"Spearman r: {spearman_r:.4f}  p: {spearman_p:.4e}\n")
    log(f"  Pearson r: {pearson_r:.4f}, Spearman r: {spearman_r:.4f}")
    log(f"  Saved {report_path}")
    return (
        float(pearson_r) if not np.isnan(pearson_r) else np.nan,
        float(spearman_r) if not np.isnan(spearman_r) else np.nan,
    )


# -----------------------------------------------------------------------------
# Part F - Output summary
# -----------------------------------------------------------------------------


def print_summary(
    mean_MY_per_quantile: Dict[int, float],
    acc: float,
    bal_acc: float,
    majority_baseline: float,
) -> None:
    log("=" * 80)
    log("OUTPUT SUMMARY")
    log("=" * 80)
    log("Species overlap between train and test: none (species-blocked folds).")
    log("Quantile -> time (mean MY): " + ", ".join(f"Q{k}={mean_MY_per_quantile[k]:.1f}" for k in sorted(mean_MY_per_quantile)))
    log(f"Accuracy: {acc:.4f}, Balanced accuracy: {bal_acc:.4f}, Majority baseline: {majority_baseline:.4f}")
    log("")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    X_norm, dist_matrix, species_final = load_and_preprocess()

    # Build pairwise data once for both feature metrics.
    X_abs, y_dist, pair_sp1, pair_sp2 = build_pairwise_features_and_metadata(
        X_norm, dist_matrix, species_final, "abs_diff"
    )
    X_sq, _, _, _ = build_pairwise_features_and_metadata(
        X_norm, dist_matrix, species_final, "sq_diff"
    )
    n_domains = X_abs.shape[1]
    log(f"  Pairwise: n_pairs={len(y_dist)}, n_domains={n_domains}. Running both abs_diff and sq_diff.")

    # Method clarification
    log("")
    log("METHOD CLARIFICATION")
    log("  Pairwise feature variants run separately:")
    log("     -> abs_diff: feature_d = |freq_i - freq_j|")
    log("     -> sq_diff:  feature_d = (freq_i - freq_j)^2")
    log("  Domains used: ALL domains after zero-variance filtering only.")
    log(f"     -> Number of domain features per pair: {n_domains}")
    method_path = os.path.join(OUTPUT_DIR, "methodology_clarification.txt")
    with open(method_path, "w") as f:
        f.write("Methodology clarification\n")
        f.write("=======================\n\n")
        f.write("Both feature metrics are run separately:\n")
        f.write("- abs_diff: |freq_i,d - freq_j,d|\n")
        f.write("- sq_diff:  (freq_i,d - freq_j,d)^2\n\n")
        f.write("All domains are used after dropping zero-variance domains.\n")
        f.write(f"Number of domain features per pair: {n_domains}\n")
    log(f"  Saved {method_path}")

    # Quantile sensitivity run: q=4,5,6
    quantile_settings = (4, 5, 6)
    metric_data = {"abs_diff": X_abs, "sq_diff": X_sq}
    all_results = []
    q4_abs_hard: Dict[str, float] | None = None
    q4_abs_prob: Dict[str, float] | None = None

    for q in quantile_settings:
        q_suffix = f"_q{q}"
        log("")
        log("=" * 80)
        log(f"Quantile run q={q}")
        log("=" * 80)

        y_quant, _, mean_MY_per_quantile, q_df = build_quantile_labels_and_time_ranges(
            y_dist, OUTPUT_DIR, q=q, suffix=q_suffix
        )
        n_classes = int(y_quant.max()) + 1
        class_labels = [f"Q{k}" for k in range(n_classes)]
        class_counts = q_df["n_pairs"].tolist()
        class_cv = float(np.std(class_counts) / np.mean(class_counts)) if class_counts and np.mean(class_counts) > 0 else np.nan

        for metric_name, X in metric_data.items():
            suffix = f"{q_suffix}_{metric_name}"
            log("=" * 80)
            log(f"Part D/E: Species-blocked CV + RF ({metric_name}, q={q})")
            log("=" * 80)
            folds = get_species_blocked_folds(species_final, pair_sp1, pair_sp2, X, y_quant)
            acc, bal_acc, majority_baseline = run_species_blocked_cv(folds, class_labels, OUTPUT_DIR, suffix=suffix)

            log("=" * 80)
            pearson_r, spearman_r = run_tree_reconstruction(
                X, y_quant, pair_sp1, pair_sp2, species_final, mean_MY_per_quantile, OUTPUT_DIR, suffix=suffix
            )

            all_results.append(
                {
                    "q": q,
                    "metric": metric_name,
                    "n_classes": n_classes,
                    "class_counts": class_counts,
                    "class_count_cv": class_cv,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "majority_baseline": majority_baseline,
                    "tree_pearson_r": pearson_r,
                    "tree_spearman_r": spearman_r,
                }
            )

            # Additional path: q=4 + abs_diff probability-weighted tree
            if q == 4 and metric_name == "abs_diff":
                q4_abs_hard = {"pearson_r": pearson_r, "spearman_r": spearman_r}
                prob_suffix = "_q4_abs_diff_prob_weighted"
                pw_pearson, pw_spearman = run_tree_reconstruction_probability_weighted(
                    X, y_quant, pair_sp1, pair_sp2, species_final, mean_MY_per_quantile, OUTPUT_DIR, suffix=prob_suffix
                )
                q4_abs_prob = {"pearson_r": pw_pearson, "spearman_r": pw_spearman}

        # Per-q summary print (abs vs sq)
        q_rows = [r for r in all_results if r["q"] == q]
        log(f"Summary q={q}:")
        for r in q_rows:
            log(
                f"  {r['metric']}: acc={r['accuracy']:.4f}, bal_acc={r['balanced_accuracy']:.4f}, "
                f"pearson={r['tree_pearson_r']:.4f}, spearman={r['tree_spearman_r']:.4f}"
            )

    # Global comparison files
    comp_metrics_path = os.path.join(OUTPUT_DIR, "comparison_quantiles_q4_q5_q6.txt")
    with open(comp_metrics_path, "w") as f:
        f.write("Comparison across quantiles q=4,5,6 (species-blocked CV)\n")
        f.write("====================================================\n\n")
        f.write("q\tmetric\tn_classes\tclass_counts\tclass_count_cv\taccuracy\tbalanced_accuracy\ttree_pearson_r\ttree_spearman_r\n")
        for r in all_results:
            f.write(
                f"{r['q']}\t{r['metric']}\t{r['n_classes']}\t{r['class_counts']}\t{r['class_count_cv']:.4f}\t"
                f"{r['accuracy']:.4f}\t{r['balanced_accuracy']:.4f}\t{r['tree_pearson_r']:.4f}\t{r['tree_spearman_r']:.4f}\n"
            )

        # Recommendation between q=4 and q=6 based on abs_diff (primary)
        abs_rows = [r for r in all_results if r["metric"] == "abs_diff" and r["q"] in (4, 6)]
        if len(abs_rows) == 2:
            r4 = next(r for r in abs_rows if r["q"] == 4)
            r6 = next(r for r in abs_rows if r["q"] == 6)
            score4 = r4["balanced_accuracy"] + 0.2 * r4["tree_spearman_r"] - 0.1 * r4["class_count_cv"]
            score6 = r6["balanced_accuracy"] + 0.2 * r6["tree_spearman_r"] - 0.1 * r6["class_count_cv"]
            recommended = 4 if score4 >= score6 else 6
            f.write("\nRecommendation between q=4 and q=6 (using abs_diff):\n")
            f.write(
                f"- q=4: bal_acc={r4['balanced_accuracy']:.4f}, tree_spearman={r4['tree_spearman_r']:.4f}, class_count_cv={r4['class_count_cv']:.4f}\n"
            )
            f.write(
                f"- q=6: bal_acc={r6['balanced_accuracy']:.4f}, tree_spearman={r6['tree_spearman_r']:.4f}, class_count_cv={r6['class_count_cv']:.4f}\n"
            )
            f.write(f"- Recommended quantile count: q={recommended}\n")

    log(f"Saved {comp_metrics_path}")

    # Additive side-by-side note for q=4 abs_diff hard vs probability-weighted reconstruction
    if q4_abs_hard is not None and q4_abs_prob is not None:
        pw_note = os.path.join(OUTPUT_DIR, "prob_weighted_vs_hard_q4_abs_diff.txt")
        with open(pw_note, "w") as f:
            f.write("Q4 abs_diff tree reconstruction: hard label vs probability-weighted\n")
            f.write("===============================================================\n\n")
            f.write("Hard assignment (class -> mean distance):\n")
            f.write(f"  Pearson r:  {q4_abs_hard['pearson_r']:.4f}\n")
            f.write(f"  Spearman r: {q4_abs_hard['spearman_r']:.4f}\n\n")
            f.write("Probability-weighted expected distance:\n")
            f.write(f"  Pearson r:  {q4_abs_prob['pearson_r']:.4f}\n")
            f.write(f"  Spearman r: {q4_abs_prob['spearman_r']:.4f}\n\n")
            better = "probability-weighted" if q4_abs_prob["spearman_r"] >= q4_abs_hard["spearman_r"] else "hard assignment"
            f.write(f"Better by Spearman: {better}\n")
        log(f"Saved {pw_note}")

    # Keep prior summary style (show q=5 abs_diff if present)
    q5_abs = [r for r in all_results if r["q"] == 5 and r["metric"] == "abs_diff"]
    if q5_abs:
        r = q5_abs[0]
        q5_rows = [x for x in all_results if x["q"] == 5 and x["metric"] == "abs_diff"]
        mean_map = {}
        q5_csv = os.path.join(OUTPUT_DIR, "quantile_time_ranges_q5.csv")
        if os.path.exists(q5_csv):
            qdf = pd.read_csv(q5_csv)
            mean_map = {int(row.quantile): float(row.mean_MY) for row in qdf.itertuples(index=False)}
        if mean_map:
            print_summary(mean_map, r["accuracy"], r["balanced_accuracy"], r["majority_baseline"])

    log("Done.")


if __name__ == "__main__":
    main()
