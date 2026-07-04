#!/usr/bin/env python3
"""
Clade-Stratified Species-Level Elastic Net Regression for Domain-Phylogeny Prediction

This script implements a STRICT species-level evaluation of evolutionary
distance prediction with an 80/20 clade-stratified split. The goal is to:

- Avoid pairwise leakage (no species appears in both train and test).
- Reduce distribution shift by stratifying by major mammal clades and
  Laurasiatheria subclades.

High-level steps:
- Load domain-by-species matrix and phylogeny (MammalsPhylogeny.nwk).
- Align species between domain data and phylogeny; drop mismatches.
- Exclude Ornithorhynchus_anatinus (outgroup) from ML.
- Drop zero-variance domains and normalize counts to relative frequencies.
- Perform clade-stratified 80/20 species split (species-level).
- Build pairwise features within train/test species only:
  (D_i - D_j)^2 for each domain.
- Predict log-distance (log(d + eps)) with an ElasticNetCV model in a
  StandardScaler + ElasticNetCV pipeline.
- Evaluate on held-out species pairs (Pearson, Spearman, R^2, RMSE, MAE).
- Save scatter plot, residual plot, metrics, model, coefficient rankings,
  and a split summary.
"""

from __future__ import annotations

import os
import sys
import math
import pickle
import itertools
import time
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit(
        "This script requires the `ete3` package. Please install it and retry."
    ) from exc

# Add scripts directory to path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

# Import data loading functions and paths
from domain_time_scatter import (  # type: ignore
    load_species_list,
    load_domain_counts,
    log,
    ensure_dir,
    DATA_RAW,
)


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "elastic_net_stratified_species_cv")
ensure_dir(OUTPUT_DIR)

TREE_PATH = os.path.join(PROJECT_ROOT, "MammalsPhylogeny.nwk")
DOMAIN_COUNTS_PATH = os.path.join(PROJECT_ROOT, "data_raw", "MammalDomainCount.tsv")
SPECIES_LIST_PATH = os.path.join(PROJECT_ROOT, "data_raw", "MammalsList.txt")

RANDOM_STATE = 42
TARGET_EPS = 1e-6
NORMALIZATION_EPS = 1e-12


# -----------------------------------------------------------------------------
# Clade mapping (underscore names as provided by user)
# -----------------------------------------------------------------------------

CLADE_TO_SPECIES_UNDERSCORE: Dict[str, List[str]] = {
    "Monotreme_outgroup": [
        "Ornithorhynchus_anatinus",
    ],
    "Marsupials": [
        "Monodelphis_domestica",
        "Sarcophilus_harrisii",
        "Vombatus_ursinus",
        "Phascolarctos_cinereus",
    ],
    "Afrotheria": [
        "Loxodonta_africana",
        "Trichechus_manatus_latirostris",
        "Chrysochloris_asiatica",
        "Orycteropus_afer_afer",
    ],
    "Primates": [
        "Homo_sapiens",
        "Pan_troglodytes",
        "Pan_paniscus",
        "Gorilla_gorilla_gorilla",
        "Pongo_abelii",
        "Nomascus_leucogenys",
        "Macaca_mulatta",
        "Macaca_fascicularis",
        "Macaca_nemestrina",
        "Papio_anubis",
        "Theropithecus_gelada",
        "Mandrillus_leucophaeus",
        "Cercocebus_atys",
        "Chlorocebus_sabaeus",
        "Rhinopithecus_roxellana",
        "Rhinopithecus_bieti",
        "Colobus_angolensis_palliatus",
        "Piliocolobus_tephrosceles",
        "Sapajus_apella",
        "Cebus_imitator",
        "Callithrix_jacchus",
        "Aotus_nancymaae",
        "Microcebus_murinus",
        "Propithecus_coquereli",
        "Otolemur_garnettii",
        "Prolemur_simus",
        "Carlito_syrichta",
    ],
    "Glires": [
        "Oryctolagus_cuniculus",
        "Sciurus_vulgaris",
        "Marmota_monax",
        "Marmota_marmota",
        "Marmota_marmota_marmota",
        "Spermophilus_dauricus",
        "Urocitellus_parryii",
        "Ictidomys_tridecemlineatus",
        "Heterocephalus_glaber",
        "Cavia_porcellus",
        "Chinchilla_lanigera",
        "Octodon_degus",
        "Dipodomys_ordii",
        "Castor_canadensis",
        "Jaculus_jaculus",
        "Nannospalax_galili",
        "Peromyscus_maniculatus_bairdii",
        "Rattus_norvegicus",
        "Mus_musculus",
        "Mus_spicilegus",
    ],
    "Laurasiatheria_Eulipotyphla": [
        "Galemys_pyrenaicus",
    ],
    "Laurasiatheria_Chiroptera": [
        "Phyllostomus_discolor",
        "Molossus_molossus",
        "Pipistrellus_kuhlii",
        "Myotis_myotis",
        "Myotis_lucifugus",
        "Rhinolophus_ferrumequinum",
        "Hipposideros_armiger",
        "Rousettus_aegyptiacus",
        "Pteropus_vampyrus",
        "Pteropus_alecto",
    ],
    "Laurasiatheria_Carnivora": [
        "Suricata_suricatta",
        "Panthera_leo",
        "Puma_concolor",
        "Felis_catus",
        "Lynx_canadensis",
        "Lynx_pardinus",
        "Vulpes_vulpes",
        "Nyctereutes_procyonoides",
        "Ailuropoda_melanoleuca",
        "Ursus_maritimus",
        "Enhydra_lutris_kenyoni",
        "Neogale_vison",
        "Neovison_vison",
        "Mustela_putorius_furo",
        "Odobenus_rosmarus_divergens",
        "Callorhinus_ursinus",
        "Leptonychotes_weddellii",
        "Neomonachus_schauinslandi",
    ],
    "Laurasiatheria_Perissodactyla": [
        "Equus_caballus",
        "Equus_asinus",
    ],
    "Laurasiatheria_Cetartiodactyla": [
        "Vicugna_pacos",
        "Camelus_bactrianus",
        "Camelus_ferus",
        "Moschus_moschiferus",
        "Physeter_catodon",
        "Physeter_macrocephalus",
        "Lipotes_vexillifer",
        "Delphinapterus_leucas",
        "Phocoena_sinus",
        "Tursiops_truncatus",
        "Balaenoptera_musculus",
        "Bos_taurus",
        "Bos_indicus",
        "Bos_mutus",
        "Bos_mutus_grunniens",
        "Bison_bison_bison",
        "Capra_hircus",
        "Ovis_aries",
        "Ovis_ammon_polii",
        "Cervus_hanglu_yarkandensis",
        "Muntiacus_reevesi",
        "Odocoileus_virginianus_texanus",
        "Sus_scrofa",
        "Catagonus_wagneri",
    ],
}


def underscore_to_space(name: str) -> str:
    """Convert underscore_name to space name."""

    return name.replace("_", " ")


def build_clade_mapping() -> Dict[str, str]:
    """
    Build mapping from canonical species names (with spaces) to clade labels.

    Species names are provided in underscore form; we convert to spaces
    for comparison with the domain/phylogeny species names.
    """

    species_to_clade: Dict[str, str] = {}
    for clade, species_list in CLADE_TO_SPECIES_UNDERSCORE.items():
        for sp_us in species_list:
            sp_name = underscore_to_space(sp_us)
            species_to_clade[sp_name] = clade
    return species_to_clade


def audit_clade_mapping(species_list: List[str]) -> None:
    """Print any species that will fall to Other so they can be reclassified."""
    mapping = build_clade_mapping()
    unmapped = []
    for sp in sorted(species_list):
        sp_us = sp.replace(" ", "_")
        if sp not in mapping:
            found = any(sp_us in slist for slist in CLADE_TO_SPECIES_UNDERSCORE.values())
            if not found:
                unmapped.append(sp)
    print(f"[audit] {len(unmapped)} species will fall to 'Other':")
    for s in unmapped:
        print(f"  {s}")


def compute_patristic_matrix(tree_path: str, species_list: List[str]) -> pd.DataFrame:
    """
    Compute full patristic distance matrix for all species pairs using ete3.

    Args:
        tree_path: Path to Newick tree file
        species_list: List of species names (with spaces)

    Returns:
        DataFrame of shape (n_species, n_species) with symmetric distances.
    """

    log("Computing full patristic distance matrix...")

    tree = Tree(tree_path, format=1)

    # Known tree name mappings (to reconcile labels)
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
        log(f"  Warning: {len(missing)} species missing from phylogeny")

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


def align_and_normalize_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Load domain counts and phylogeny, align species, filter, and normalize.

    Returns:
        X_norm: normalized domain matrix (species × domains)
        dist_matrix: symmetric distance matrix (species × species)
        species_to_clade: mapping from species to clade label
    """

    log("=" * 80)
    log("Step 1: Loading and aligning data")
    log("=" * 80)

    # Load species order and domain counts
    species_from_list = load_species_list(SPECIES_LIST_PATH)
    X_counts = load_domain_counts(DOMAIN_COUNTS_PATH, species_from_list)

    # Species actually present in domain counts
    domain_species = list(X_counts.index)
    log(f"  Domain matrix species: {len(domain_species)}")

    # Build full patristic distance matrix from phylogeny
    dist_matrix_full = compute_patristic_matrix(TREE_PATH, domain_species)
    tree_species = list(dist_matrix_full.index)
    log(f"  Phylogeny species: {len(tree_species)}")

    # Intersect species sets
    species_intersection = sorted(set(domain_species) & set(tree_species))
    dropped_from_domain = sorted(set(domain_species) - set(species_intersection))
    dropped_from_tree = sorted(set(tree_species) - set(species_intersection))

    if dropped_from_domain:
        log(f"  Dropping {len(dropped_from_domain)} species missing from phylogeny")
    if dropped_from_tree:
        log(f"  Dropping {len(dropped_from_tree)} species missing from domain counts")

    # Subset matrices to common species and align order
    X_counts = X_counts.loc[species_intersection]
    dist_matrix = dist_matrix_full.loc[species_intersection, species_intersection]

    log(f"  After intersection: {len(species_intersection)} species")

    # Exclude Ornithorhynchus_anatinus (outgroup) from ML
    outgroup_name = underscore_to_space("Ornithorhynchus_anatinus")
    if outgroup_name in X_counts.index:
        log("  Excluding Ornithorhynchus_anatinus from ML (outgroup)")
        X_counts = X_counts.drop(index=outgroup_name)
        dist_matrix = dist_matrix.drop(index=outgroup_name, columns=outgroup_name)

    species_final = list(X_counts.index)
    log(f"  Final species after outgroup exclusion: {len(species_final)}")

    # Drop zero-variance domains
    log("Step 2: Dropping zero-variance domains")
    domain_variance = X_counts.var(axis=0)
    zero_var_mask = domain_variance == 0
    n_zero_var = int(zero_var_mask.sum())
    if n_zero_var > 0:
        log(f"  Dropping {n_zero_var} zero-variance domains (zero variance across species)")
        X_counts = X_counts.loc[:, ~zero_var_mask]

    # Sanity check: confirm no remaining zero-variance domains
    if (X_counts.var(axis=0) == 0).any():
        raise ValueError("Zero-variance domains remain after filtering step.")

    log(f"  Remaining domains after variance filter: {X_counts.shape[1]}")

    # Normalize to relative frequencies per species
    log("Step 3: Normalizing domain counts to relative frequencies")
    row_sums = X_counts.sum(axis=1).values.reshape(-1, 1)
    X_array = X_counts.values.astype(float)
    X_norm_array = X_array / (row_sums + NORMALIZATION_EPS)
    X_norm = pd.DataFrame(X_norm_array, index=X_counts.index, columns=X_counts.columns)

    # Build clade mapping
    species_to_clade_full = build_clade_mapping()
    species_to_clade: Dict[str, str] = {}

    log("Step 4: Assigning clades to species")
    for sp in species_final:
        if sp in species_to_clade_full:
            species_to_clade[sp] = species_to_clade_full[sp]
        else:
            # Try underscore form
            sp_us = sp.replace(" ", "_")
            # Reverse lookup in provided mapping
            found_clade = None
            for clade, us_list in CLADE_TO_SPECIES_UNDERSCORE.items():
                if sp_us in us_list:
                    found_clade = clade
                    break
            if found_clade is not None:
                species_to_clade[sp] = found_clade
            else:
                species_to_clade[sp] = "Other"

    # Report clade assignment statistics
    clade_counts: Dict[str, int] = {}
    for sp, clade in species_to_clade.items():
        clade_counts[clade] = clade_counts.get(clade, 0) + 1

    total_species = len(species_final)
    log("  Species per clade after filtering and alignment:")
    for clade, count in sorted(clade_counts.items()):
        frac = 100.0 * count / max(1, total_species)
        log(f"    {clade}: {count} species ({frac:.1f}%)")

    n_other = clade_counts.get("Other", 0)
    if n_other > 0:
        frac_other = 100.0 * n_other / max(1, total_species)
        log(
            f"  WARNING: {n_other} species ({frac_other:.1f}%) assigned to 'Other' clade. "
            "This may indicate naming or mapping issues."
        )
        # Log which species are in Other to avoid silent misassignment
        others = sorted([sp for sp, clade in species_to_clade.items() if clade == "Other"])
        for sp in others:
            log(f"    Other-clade species: {sp}")

    # Check which clade-defined species are missing after filtering
    all_clade_species_space = {
        underscore_to_space(sp_us)
        for species_list_us in CLADE_TO_SPECIES_UNDERSCORE.values()
        for sp_us in species_list_us
    }
    missing_clade_species = sorted(all_clade_species_space - set(species_final) - {outgroup_name})
    if missing_clade_species:
        log(
            f"  NOTE: {len(missing_clade_species)} species from clade definitions are "
            "not present in the aligned/filtered dataset:"
        )
        for sp in missing_clade_species:
            log(f"    Missing clade-defined species: {sp}")

    return X_norm, dist_matrix, species_to_clade


def stratified_species_split(
    species_to_clade: Dict[str, str],
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Perform clade-stratified 80/20 species split (species-level).

    For each clade (except Monotreme_outgroup):
      - If n >= 5: 80/20 split with test_size=0.2.
      - If 2 <= n <= 4: 1 species in test, rest in train.
      - If n == 1: species in train only.

    Returns:
        train_species, test_species, split_summary_df
    """

    log("=" * 80)
    log("Step 4: Clade-stratified 80/20 species split")
    log("=" * 80)

    # Organize species by clade
    clade_to_species: Dict[str, List[str]] = {}
    for sp, clade in species_to_clade.items():
        clade_to_species.setdefault(clade, []).append(sp)

    rng_state = RANDOM_STATE
    train_species: List[str] = []
    test_species: List[str] = []
    summary_rows: List[Dict[str, str]] = []

    for clade, clade_species in sorted(clade_to_species.items()):
        if clade == "Monotreme_outgroup":
            # Should already be excluded; skip for safety
            log(f"  Skipping clade {clade} (outgroup)")
            continue

        n = len(clade_species)
        if n == 0:
            continue

        if n >= 5:
            # 80/20 split with at least 1 in test
            n_test = max(1, int(round(0.2 * n)))
        elif 2 <= n <= 4:
            # Force 1 in test
            n_test = 1
        else:  # n == 1
            n_test = 0

        if n_test > 0:
            clade_train, clade_test = train_test_split(
                clade_species,
                test_size=n_test,
                random_state=rng_state,
                shuffle=True,
            )
        else:
            clade_train = list(clade_species)
            clade_test = []

        # Update RNG seed per clade to avoid deterministic identical splits
        rng_state += 1

        train_species.extend(clade_train)
        test_species.extend(clade_test)

        log(
            f"  Clade {clade}: total={n}, train={len(clade_train)}, "
            f"test={len(clade_test)}"
        )

        for sp in clade_train:
            summary_rows.append({"species": sp, "clade": clade, "set": "train"})
        for sp in clade_test:
            summary_rows.append({"species": sp, "clade": clade, "set": "test"})

    # Final sets
    train_species = sorted(set(train_species))
    test_species = sorted(set(test_species))

    overlap = set(train_species) & set(test_species)
    if overlap:
        raise ValueError(f"Species overlap between train and test sets: {overlap}")

    log(
        f"  Total train species: {len(train_species)}, "
        f"test species: {len(test_species)}"
    )

    split_summary_df = pd.DataFrame(summary_rows)

    # Save summary
    split_summary_path = os.path.join(
        OUTPUT_DIR, "stratified_species_cv_split_summary.tsv"
    )
    split_summary_df.to_csv(split_summary_path, sep="\t", index=False)
    log(f"  Saved split summary to {split_summary_path}")

    return train_species, test_species, split_summary_df


def construct_pairwise_features_and_targets(
    X_norm: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    species_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build pairwise feature matrix and log-distance targets for a species set.

    Features: (D_i - D_j)^2 per domain (normalized domain frequencies).
    Target: log(d_ij + TARGET_EPS).

    OPTIMIZED: Uses vectorized numpy operations instead of Python loops.

    Returns:
        X_pairs: (n_pairs × n_domains)
        y_log_pairs: (n_pairs,)
        pairs: list of (species_i, species_j)
    """

    species_list = sorted(species_list)
    n_species = len(species_list)
    domain_names = list(X_norm.columns)
    n_domains = len(domain_names)

    log(
        f"  Constructing pairwise features for {n_species} species "
        f"and {n_domains} domains (VECTORIZED)"
    )

    # Sanity: dist_matrix should have all species
    missing_in_dist = set(species_list) - set(dist_matrix.index)
    if missing_in_dist:
        raise ValueError(f"Species missing from distance matrix: {missing_in_dist}")

    # Pre-extract subset matrices as numpy arrays for fast indexing
    X_subset = X_norm.loc[species_list].values  # (n_species × n_domains)
    dist_subset = dist_matrix.loc[species_list, species_list].values  # (n_species × n_species)

    # Verify distance matrix symmetry and non-negativity (once, not in loop)
    if not np.allclose(dist_subset, dist_subset.T, rtol=1e-8, atol=1e-8):
        raise ValueError("Distance matrix is not symmetric")
    if np.any(dist_subset < 0):
        raise ValueError("Distance matrix contains negative values")

    # Build pairs list (for return value and target extraction)
    pairs: List[Tuple[str, str]] = []
    for i in range(n_species):
        for j in range(i + 1, n_species):
            pairs.append((species_list[i], species_list[j]))

    n_pairs = len(pairs)
    log(f"  Number of pairs: {n_pairs}")

    # VECTORIZED: Compute all squared differences at once using pdist
    # pdist computes pairwise distances in condensed form (upper triangle, i < j)
    # For each domain, we compute (X_i - X_j)^2 for all pairs
    pair_start_time = time.time()
    X_pairs = np.zeros((n_pairs, n_domains), dtype=float)
    
    for domain_idx in range(n_domains):
        # Extract domain values for all species: shape (n_species,)
        domain_values = X_subset[:, domain_idx].reshape(-1, 1)  # Reshape for pdist
        
        # Compute squared Euclidean distance = (X_i - X_j)^2
        # pdist with 'sqeuclidean' metric computes sum of squared differences
        # For 1D vectors, this is just (X_i - X_j)^2
        squared_diffs = pdist(domain_values, metric='sqeuclidean')
        X_pairs[:, domain_idx] = squared_diffs

    # VECTORIZED: Extract distance values using integer indexing
    # Build index pairs for upper triangle (i < j) using numpy operations
    # This is faster than building Python lists
    i_indices, j_indices = np.triu_indices(n_species, k=1)
    
    # Extract distances using integer indexing (much faster than .loc[])
    dist_values = dist_subset[i_indices, j_indices]
    
    # Compute log distances
    y_log_pairs = np.log(dist_values + TARGET_EPS)

    elapsed_time = time.time() - pair_start_time
    log(f"  Vectorized feature construction completed in {elapsed_time:.2f}s")

    # Sanity checks
    if np.isnan(X_pairs).any() or np.isinf(X_pairs).any():
        raise ValueError("NaN or inf detected in feature matrix")
    if np.isnan(y_log_pairs).any() or np.isinf(y_log_pairs).any():
        raise ValueError("NaN or inf detected in target vector")

    log(
        f"  Features shape: {X_pairs.shape}, "
        f"target range (log): [{y_log_pairs.min():.6f}, {y_log_pairs.max():.6f}]"
    )

    return X_pairs, y_log_pairs, pairs


def train_elastic_net(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
) -> Pipeline:
    """
    Train ElasticNetCV model on log-distance targets.

    Returns:
        Fitted sklearn Pipeline (StandardScaler + ElasticNetCV).
    """

    log("=" * 80)
    log("Step 5: Training Elastic Net model")
    log("=" * 80)

    log(f"  Training pairs: {len(y_train_log)}")
    log(f"  Features (domains): {X_train.shape[1]}")

    # OPTIMIZED: Reduced search space for faster training while maintaining coverage
    # Original: 7 l1_ratios × 30 alphas × 3 CV = 630 fits
    # Previous: 5 l1_ratios × 15 alphas × 3 CV = 225 fits
    # New: 3 l1_ratios × 10 alphas × 3 CV = 90 fits (~2.5x faster than 225, ~7x faster than 630)
    # Rationale: 3 l1_ratios (0.1, 0.5, 0.9) cover Lasso (0.9), Ridge (0.1), and balanced (0.5)
    l1_ratios = [0.1, 0.5, 0.9]  # Reduced from 5 to 3 - covers L1/L2 spectrum
    n_l1_ratios = len(l1_ratios)
    n_cv_folds = 3
    
    # Reduced alpha grid: 10 values (still covers important regularization range)
    alphas = np.logspace(-4, 1, 10)  # 10 values from 0.0001 to 10
    n_alphas = len(alphas)
    total_fits = n_l1_ratios * n_alphas * n_cv_folds

    log(f"  Hyperparameter search space (OPTIMIZED for speed):")
    log(f"    - l1_ratios: {n_l1_ratios} values {l1_ratios} (covers Lasso->Ridge spectrum)")
    log(f"    - alphas: {n_alphas} values (logspace from 1e-4 to 10)")
    log(f"    - CV folds: {n_cv_folds}")
    log(f"    - Total model fits: {total_fits:,} (reduced from 225 -> ~2.5x faster)")

    elastic_net = ElasticNetCV(
        alphas=alphas,  # Explicit alpha grid for faster search
        l1_ratio=l1_ratios,
        cv=n_cv_folds,
        max_iter=50000,
        tol=1e-3,  # Slightly relaxed tolerance for faster convergence
        n_jobs=-1,
        random_state=RANDOM_STATE,
        selection='random',  # Faster than 'cyclic' for large datasets
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("elastic_net", elastic_net),
        ]
    )

    log("  Fitting ElasticNetCV (this may take a while)...")
    log(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    pipeline.fit(X_train, y_train_log)
    
    elapsed_time = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    log(f"  Training completed in: {elapsed_str} ({elapsed_time:.1f} seconds)")
    log(f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    enet = pipeline.named_steps["elastic_net"]
    log(f"  Best alpha: {enet.alpha_:.6e}")
    log(f"  Best l1_ratio: {float(enet.l1_ratio_):.4f}")
    log(f"  Nonzero coefficients: {(enet.coef_ != 0).sum()} / {len(enet.coef_)}")

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test_log: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate model on held-out species pairs and generate plots.

    Returns:
        metrics dict with Pearson, Spearman, R^2, RMSE, MAE, etc.
    """

    log("=" * 80)
    log("Step 6: Evaluating model on held-out species pairs")
    log("=" * 80)

    y_pred_log = pipeline.predict(X_test)

    # Inverse transform: log(d + eps) -> d
    y_test = np.exp(y_test_log) - TARGET_EPS
    y_pred = np.exp(y_pred_log) - TARGET_EPS

    # Enforce nonnegativity
    y_test = np.clip(y_test, a_min=0.0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    pearson_r, pearson_p = pearsonr(y_test, y_pred)
    spearman_rho, spearman_p = spearmanr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    log(f"  Test pairs: {len(y_test)}")
    log(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    log(f"  Spearman rho: {spearman_rho:.4f} (p={spearman_p:.2e})")
    log(f"  R^2: {r2:.4f}")
    log(f"  RMSE: {rmse:.6f}")
    log(f"  MAE: {mae:.6f}")

    # Scatter plot true vs predicted
    scatter_path = os.path.join(
        OUTPUT_DIR, "stratified_species_cv_prediction_scatter.png"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, s=12, alpha=0.5, edgecolors="none")

    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")

    ax.set_xlabel("True evolutionary distance", fontsize=12)
    ax.set_ylabel("Predicted evolutionary distance", fontsize=12)
    ax.set_title(
        (
            "Clade-stratified species-level CV: predicted vs true\n"
            f"Pearson r={pearson_r:.3f}, Spearman rho={spearman_rho:.3f}, R^2={r2:.3f}"
        ),
        fontsize=12,
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved scatter plot to {scatter_path}")

    # Residual plot: true vs residual (pred - true)
    residuals = y_pred - y_test
    residual_path = os.path.join(
        OUTPUT_DIR, "stratified_species_cv_residuals.png"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, residuals, s=12, alpha=0.5, edgecolors="none")
    ax.axhline(0.0, color="r", linestyle="--", label="residual = 0")
    ax.set_xlabel("True evolutionary distance", fontsize=12)
    ax.set_ylabel("Residual (pred - true)", fontsize=12)
    ax.set_title("Clade-stratified species-level CV: residuals", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(residual_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved residual plot to {residual_path}")

    metrics = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_test_pairs": int(len(y_test)),
    }

    return metrics


def extract_feature_importance(
    pipeline: Pipeline, domain_names: List[str]
) -> pd.DataFrame:
    """
    Extract and rank feature importances from the trained ElasticNet model.
    """

    log("=" * 80)
    log("Step 7: Extracting feature importance")
    log("=" * 80)

    enet = pipeline.named_steps["elastic_net"]
    coefs = enet.coef_

    if len(coefs) != len(domain_names):
        raise ValueError(
            f"Coefficient length {len(coefs)} does not match domain count "
            f"{len(domain_names)}"
        )

    df = pd.DataFrame(
        {
            "domain": domain_names,
            "coefficient": coefs,
        }
    )
    df["abs_coefficient"] = df["coefficient"].abs()
    df["rank"] = df["abs_coefficient"].rank(
        ascending=False, method="min"
    ).astype(int)
    df = df.sort_values("rank")

    top_n = min(10, len(df))
    log("  Top domains by absolute coefficient:")
    for _, row in df.head(top_n).iterrows():
        log(
            f"    rank {row['rank']}: {row['domain']} "
            f"(coef={row['coefficient']:.6e})"
        )

    return df


def save_artifacts(
    pipeline: Pipeline,
    rankings_df: pd.DataFrame,
    metrics: Dict[str, float],
    split_summary_df: pd.DataFrame,
    train_species: List[str],
    test_species: List[str],
    X_train_shape: Tuple[int, int],
    X_test_shape: Tuple[int, int],
) -> None:
    """
    Save model, rankings, metrics, and split summary.
    """

    log("=" * 80)
    log("Step 8: Saving artifacts")
    log("=" * 80)

    # Model
    model_path = os.path.join(OUTPUT_DIR, "stratified_species_cv_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    log(f"  Saved model to {model_path}")

    # Rankings
    rankings_path = os.path.join(
        OUTPUT_DIR, "stratified_species_cv_domain_rankings.tsv"
    )
    rankings_df.to_csv(rankings_path, sep="\t", index=False)
    log(f"  Saved domain rankings to {rankings_path}")

    # Split summary (already saved once, but write again here for completeness)
    split_summary_path = os.path.join(
        OUTPUT_DIR, "stratified_species_cv_split_summary.tsv"
    )
    split_summary_df.to_csv(split_summary_path, sep="\t", index=False)

    # Metrics text file
    metrics_path = os.path.join(OUTPUT_DIR, "stratified_species_cv_metrics.txt")
    enet = pipeline.named_steps["elastic_net"]

    with open(metrics_path, "w") as f:
        f.write(
            "Clade-Stratified Species-Level Elastic Net Regression "
            "(log-distance target)\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write("Species split:\n")
        f.write(f"  Train species: {len(train_species)}\n")
        f.write(f"  Test species:  {len(test_species)}\n")
        f.write("  Zero overlap confirmed: yes\n\n")

        f.write("Pair counts:\n")
        f.write(
            f"  Train pairs: {X_train_shape[0]} "
            f"(features per pair: {X_train_shape[1]})\n"
        )
        f.write(
            f"  Test  pairs: {X_test_shape[0]} "
            f"(features per pair: {X_test_shape[1]})\n\n"
        )

        f.write("Test-set performance (held-out species pairs):\n")
        f.write(f"  Pearson r:       {metrics['pearson_r']:.6f}\n")
        f.write(f"  Pearson p-value: {metrics['pearson_p']:.2e}\n")
        f.write(f"  Spearman rho:    {metrics['spearman_rho']:.6f}\n")
        f.write(f"  Spearman p-val:  {metrics['spearman_p']:.2e}\n")
        f.write(f"  R^2:             {metrics['r2']:.6f}\n")
        f.write(f"  RMSE:            {metrics['rmse']:.6f}\n")
        f.write(f"  MAE:             {metrics['mae']:.6f}\n\n")

        f.write("ElasticNet hyperparameters:\n")
        f.write(f"  Best alpha:      {enet.alpha_:.6e}\n")
        f.write(f"  Best l1_ratio:   {float(enet.l1_ratio_):.4f}\n")
        f.write(
            f"  Nonzero coeffs:  {(enet.coef_ != 0).sum()} / "
            f"{len(enet.coef_)}\n\n"
        )

        f.write("Clade-wise train/test species counts:\n")
        for clade, group in split_summary_df.groupby("clade"):
            n_train = (group["set"] == "train").sum()
            n_test = (group["set"] == "test").sum()
            f.write(
                f"  {clade}: total={len(group)}, "
                f"train={n_train}, test={n_test}\n"
            )

    log(f"  Saved metrics summary to {metrics_path}")


def main() -> None:
    """
    Main entry point: run full clade-stratified species-level CV pipeline.
    """

    script_start_time = time.time()
    log("=" * 80)
    log("Clade-stratified species-level Elastic Net regression (log-distance)")
    log("=" * 80)
    log(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    # 1-3. Load, align, filter, and normalize data
    step_start = time.time()
    X_norm, dist_matrix, species_to_clade = align_and_normalize_data()
    step_elapsed = time.time() - step_start
    log(f"  [Time: {step_elapsed:.1f}s] Data loading, alignment, and normalization complete")
    log("")
    species_all = list(X_norm.index)

    # 4. Clade-stratified species split
    step_start = time.time()
    train_species, test_species, split_summary_df = stratified_species_split(
        species_to_clade
    )
    step_elapsed = time.time() - step_start
    log(f"  [Time: {step_elapsed:.1f}s] Clade-stratified split complete")
    log("")

    if not train_species or not test_species:
        raise ValueError(
            "Train or test species set is empty after stratified split."
        )

    # Sanity: ensure all train/test species are in domain and distance matrices
    for sp in train_species + test_species:
        if sp not in species_all:
            raise ValueError(
                f"Species {sp} from split not found in domain matrix index."
            )
        if sp not in dist_matrix.index:
            raise ValueError(
                f"Species {sp} from split not found in distance matrix."
            )

    # 5. Construct pairwise features and targets for train and test species
    log("=" * 80)
    log("Step 6: Constructing pairwise features and log-distance targets")
    log("=" * 80)
    step_start = time.time()

    X_train, y_train_log, train_pairs = construct_pairwise_features_and_targets(
        X_norm, dist_matrix, train_species
    )
    X_test, y_test_log, test_pairs = construct_pairwise_features_and_targets(
        X_norm, dist_matrix, test_species
    )

    step_elapsed = time.time() - step_start
    log(
        f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}, "
        f"domains: {X_train.shape[1]}"
    )
    log(f"  [Time: {step_elapsed:.1f}s] Feature construction complete")
    log("")

    # 6. Train ElasticNet model on log-distance targets
    pipeline = train_elastic_net(X_train, y_train_log)
    log("")

    # 7. Evaluate on held-out species pairs
    step_start = time.time()
    metrics = evaluate_model(pipeline, X_test, y_test_log)
    step_elapsed = time.time() - step_start
    log(f"  [Time: {step_elapsed:.1f}s] Model evaluation complete")
    log("")

    # 8. Feature importance and artifacts
    step_start = time.time()
    domain_names = list(X_norm.columns)
    rankings_df = extract_feature_importance(pipeline, domain_names)

    save_artifacts(
        pipeline,
        rankings_df,
        metrics,
        split_summary_df,
        train_species,
        test_species,
        X_train.shape,
        X_test.shape,
    )
    step_elapsed = time.time() - step_start
    log(f"  [Time: {step_elapsed:.1f}s] Artifact saving complete")
    log("")

    # Final summary
    total_elapsed = time.time() - script_start_time
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))
    log("=" * 80)
    log("Clade-stratified species-level Elastic Net regression complete.")
    log(f"Results saved under: {OUTPUT_DIR}")
    log("=" * 80)
    log(f"Total execution time: {total_elapsed_str} ({total_elapsed:.1f} seconds)")
    log(f"Script finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()


