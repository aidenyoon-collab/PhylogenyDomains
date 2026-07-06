#!/usr/bin/env python3
"""
Downstream biological interpretation: domain importance vs mammal clades.

Uses the SAME preprocessing and pairwise RF setup as classification_cv_tree_reconstruction:
relative frequencies per species, platypus excluded, zero-variance domains dropped,
pairwise L1 (cityblock) per domain, StandardScaler & RandomOverSampler & RandomForest.

Run from project root:
  python3 scripts/domain_clade_interpretation.py [--scheme SCHEME_4B] [--n-folds 5] [--n-estimators 200] [--random-state 42]
  python3 scripts/domain_clade_interpretation.py --rf-importance-mode full_data
  python3 scripts/domain_clade_interpretation.py --perm-max-samples 0   # slow: score on all held-out test rows
  python3 scripts/domain_clade_interpretation.py --skip-rf --skip-permutation-importance
      # Fast refresh: reuse rf_feature_importance.tsv; redo ANOVA/markers/heatmap/concordance/overlap

Outputs (TSV + figures) under results/domain_analysis/
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd

matplotlib = __import__("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, mannwhitneyu
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError as exc:
    raise SystemExit(
        "This script requires imbalanced-learn (RandomOverSampler). pip install imbalanced-learn"
    ) from exc

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None  # type: ignore

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from classification_cv_tree_reconstruction import (  # noqa: E402
    RANDOM_STATE,
    assign_classes,
    build_scheme_definitions,
    construct_pairwise_features,
    fit_fold_random_oversample_rf,
    load_and_preprocess_data,
)

try:  # compatibility with renamed fold builder in classification_cv_tree_reconstruction
    from classification_cv_tree_reconstruction import (  # noqa: E402
        build_species_blocked_fold_indices,
    )
except ImportError:  # pragma: no cover
    from classification_cv_tree_reconstruction import (  # noqa: E402
        build_stratified_deep_species_fold_indices as build_species_blocked_fold_indices,
    )
from elastic_net_regression_stratified_species_cv import (  # noqa: E402
    CLADE_TO_SPECIES_UNDERSCORE,
    build_clade_mapping,
)

PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "results", "domain_analysis")
DOMAIN_COUNTS_PATH_LOCAL = os.path.join(PROJECT_ROOT, "data_raw", "MammalDomainCount.tsv")

EPS_FOLDCHANGE = 1e-8
N_TOP_RF_PLOT = 20
N_TOP_ANOVA_PRINT = 20
# Clades smaller than this are NOT tested as a marker group: with n<3 the Mann-Whitney
# U is degenerate and its ties/asymptotic approximation reports FDR-"significant" hits
# that are exact-test-impossible (e.g. n=1 vs 106 has a two-sided p floor ~0.019, which
# cannot survive FDR). Matches MIN_CLADE_N_ANOVA. Excluded clades stay in the "rest"
# comparison set; they are simply not assigned markers of their own.
MIN_CLADE_N_MARKER = 3
N_TOP_MARKERS = 10
N_TOP_ANOVA_BOXPLOT = 8
N_HEATMAP_DOMAINS = 40
N_TOP_ANOVA_RF_INTERSECTION = 100
OVERLAP_PLOT_ROWS = 3
OVERLAP_PLOT_COLS = 4
PF_DOMAIN_LABEL_RE = re.compile(r"^(PF\d{5}):(.*)$")

# Permutation importance is computed on HELD-OUT CV test folds (sklearn guidance: must be measured on
# held-out data, not the train rows the RF memorized). Scoring each shuffled column on every test row is
# O(n_test_rows * n_features * n_repeats); capping rows for the *scoring* phase (max_samples) keeps the
# same fitted RF and all features. 0 = use all held-out test rows (slowest).
PERMUTATION_N_REPEATS_DEFAULT = 10
PERMUTATION_MAX_SAMPLES_DEFAULT = 12000


def log(msg: str) -> None:
    print(f"[domain_interp] {msg}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rf_importance_from_tsv(out_dir: str) -> pd.DataFrame:
    """
    Load RF feature importance table from a prior run (domain_name, importance_score, rank, ...).
    Used when --skip-rf avoids retraining; permutation columns optional.
    """
    path = os.path.join(out_dir, "rf_feature_importance.tsv")
    if not os.path.isfile(path):
        raise SystemExit(
            f"--skip-rf requires existing {path} from a previous run. Run the script without --skip-rf first."
        )
    df = pd.read_csv(path, sep="\t")
    if "domain_name" not in df.columns or "importance_score" not in df.columns:
        raise SystemExit(f"{path} must contain columns domain_name and importance_score.")
    if "rank" not in df.columns:
        df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    if "permutation_importance" not in df.columns:
        df["permutation_importance"] = np.nan
    if "perm_rank" not in df.columns:
        df["perm_rank"] = np.nan
    log(f"Loaded RF table from {path} ({len(df)} domains).")
    return df


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values (two-sided tests)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = np.empty(n, dtype=float)
    prev_min = 1.0
    for i in range(n - 1, -1, -1):
        j = i + 1  # 1-based rank from smallest p
        val = ranked[i] * n / j
        prev_min = min(prev_min, val)
        q_sorted[i] = min(prev_min, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    return q


def apply_fdr(p_values: np.ndarray) -> np.ndarray:
    if multipletests is not None:
        _, q, _, _ = multipletests(p_values, method="fdr_bh")
        return np.asarray(q, dtype=float)
    return benjamini_hochberg(p_values)


def assign_species_to_clades(species_final: List[str]) -> Dict[str, str]:
    """Same resolution logic as elastic_net align_and_normalize_data Step 4."""
    species_to_clade_full = build_clade_mapping()
    species_to_clade: Dict[str, str] = {}
    for sp in species_final:
        if sp in species_to_clade_full:
            species_to_clade[sp] = species_to_clade_full[sp]
        else:
            sp_us = sp.replace(" ", "_")
            found_clade = None
            for clade, us_list in CLADE_TO_SPECIES_UNDERSCORE.items():
                if sp_us in us_list:
                    found_clade = clade
                    break
            species_to_clade[sp] = found_clade if found_clade is not None else "Other"
    return species_to_clade


def sanitize_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/inf with 0 for ANOVA/markers (should be rare after preprocess)."""
    arr = X.values.astype(float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def load_authoritative_domain_headers(path: str) -> List[str]:
    """
    Load authoritative PF domain headers from MammalDomainCount.tsv first column.
    """
    first_col = pd.read_csv(path, sep="\t", skiprows=1, usecols=[0], low_memory=False).iloc[:, 0]
    labels = [str(v) for v in first_col.tolist() if isinstance(v, str) or pd.notna(v)]
    return [x for x in labels if PF_DOMAIN_LABEL_RE.match(x)]


def remap_to_authoritative_domain_headers(
    current_labels: List[str], authoritative_labels: List[str]
) -> List[str]:
    """
    Remap current domain labels to authoritative labels via exact match first, then
    unique PF accession match fallback.
    """
    authoritative_set = set(authoritative_labels)
    by_accession: Dict[str, List[str]] = {}
    for lbl in authoritative_labels:
        acc = lbl.split(":", 1)[0]
        by_accession.setdefault(acc, []).append(lbl)

    remapped: List[str] = []
    n_exact = 0
    n_accession = 0
    n_unresolved = 0
    for lbl in current_labels:
        if lbl in authoritative_set:
            remapped.append(lbl)
            n_exact += 1
            continue
        acc = lbl.split(":", 1)[0]
        candidates = by_accession.get(acc, [])
        if len(candidates) == 1:
            remapped.append(candidates[0])
            n_accession += 1
        else:
            remapped.append(lbl)
            n_unresolved += 1
    log(
        "Domain header remap: "
        f"exact={n_exact}, accession_repaired={n_accession}, unresolved={n_unresolved}"
    )
    return remapped


def compute_rf_importance_cv_mean(
    X: np.ndarray,
    y_class: np.ndarray,
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    species_ordered: List[str],
    n_folds: int,
    random_state: int,
    n_estimators: int,
    n_classes: int,
    domain_names: List[str],
) -> np.ndarray:
    """
    Mean feature_importances_ across species-blocked folds (train-only fit each fold).
    Each feature index maps 1:1 to a domain (pairwise L1 on that domain); averaging is
    across CV models, not aggregating multiple features per domain.
    """
    folds_result = build_species_blocked_fold_indices(
        species_ordered, pair_sp1, pair_sp2, n_folds, random_state
    )
    folds = folds_result[0] if isinstance(folds_result, tuple) else folds_result
    accum = np.zeros(len(domain_names), dtype=float)
    n_ok = 0
    for fold_i, (tr_idx, _te_idx, _tr_sp, _te_sp) in enumerate(folds, start=1):
        X_train, y_train = X[tr_idx], y_class[tr_idx]
        X_dummy = X_train[:1].copy()
        y_dummy = y_train[:1].copy()
        result = fit_fold_random_oversample_rf(
            X_train,
            y_train,
            X_dummy,
            y_dummy,
            n_classes,
            random_state=random_state + fold_i,
            n_estimators=n_estimators,
        )
        if result is None:
            raise RuntimeError(f"Fold {fold_i} failed (RF training).")
        _proba, clf = result
        accum += clf.feature_importances_
        n_ok += 1
    return accum / max(n_ok, 1)


def compute_rf_importance_full_data(
    X: np.ndarray,
    y_class: np.ndarray,
    random_state: int,
    n_estimators: int,
    n_classes: int,
) -> np.ndarray:
    """Single scaler on all pairs, ROS on all scaled rows, one RF fit."""
    if len(np.unique(y_class)) < 2:
        raise RuntimeError("Need at least 2 classes for RF.")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    ros = RandomOverSampler(random_state=random_state)
    X_r, y_r = ros.fit_resample(X_s, y_class)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        class_weight=None,
        n_jobs=-1,
    )
    clf.fit(X_r, y_r)
    return clf.feature_importances_


def _run_permutation_importance(
    clf: RandomForestClassifier,
    X_score: np.ndarray,
    y_score: np.ndarray,
    n_repeats: int,
    random_state: int,
    perm_max_samples: int,
) -> np.ndarray:
    """Call sklearn permutation_importance on already-scaled scoring rows; honor max_samples cap."""
    perm_kw: Dict[str, Any] = {
        "n_repeats": n_repeats,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if (
        perm_max_samples > 0
        and len(X_score) > perm_max_samples
        and "max_samples" in inspect.signature(permutation_importance).parameters
    ):
        perm_kw["max_samples"] = int(perm_max_samples)
        log(
            f"  Permutation importance: max_samples={perm_max_samples} for scoring "
            f"(held-out test rows in fold={len(X_score)})."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            result = permutation_importance(clf, X_score, y_score, **perm_kw)
        except PermissionError:
            # Some environments disallow loky's semaphore checks; use threads to keep parallelism.
            log("  Permutation importance loky backend blocked; retrying with threading backend.")
            with parallel_backend("threading"):
                result = permutation_importance(clf, X_score, y_score, **perm_kw)
    return result.importances_mean


def compute_permutation_importance(
    X: np.ndarray,
    y_class: np.ndarray,
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    species_ordered: List[str],
    n_folds: int,
    random_state: int,
    n_estimators: int,
    n_classes: int,
    domain_names: List[str],
    n_repeats: int = PERMUTATION_N_REPEATS_DEFAULT,
    perm_max_samples: int = PERMUTATION_MAX_SAMPLES_DEFAULT,
) -> np.ndarray:
    """
    Permutation importance averaged over the SAME species-blocked CV folds used for the Gini
    (cv_mean) path. Per fold: fit StandardScaler+RandomOverSampler+RF on TRAIN ONLY (exactly as
    fit_fold_random_oversample_rf does), then call sklearn permutation_importance on that fold's
    HELD-OUT, scaled TEST rows. Permutation importance MUST be measured on held-out data - scoring
    on the train rows lets the unconstrained (max_depth=None) RF memorize them (acc~1.0), which made
    every feature's permutation importance ~0 (the prior full-data bug). See
    https://scikit-learn.org/stable/modules/permutation_importance.html

    NOTE: permutation importance is a SUPPORTING signal here. With ~8020 highly correlated domain
    features it UNDER-estimates per-feature importance (shuffling one of several correlated columns
    barely moves accuracy), just as Gini/MDI OVER-estimates. ANOVA concordance is the
    model-independent cross-check; treat this column as diffuse/small but non-degenerate.
    """
    if len(np.unique(y_class)) < 2:
        raise RuntimeError("Need at least 2 classes for RF.")
    folds_result = build_species_blocked_fold_indices(
        species_ordered, pair_sp1, pair_sp2, n_folds, random_state
    )
    folds = folds_result[0] if isinstance(folds_result, tuple) else folds_result

    accum = np.zeros(len(domain_names), dtype=float)
    n_ok = 0
    t0 = time.time()
    for fold_i, (tr_idx, te_idx, _tr_sp, _te_sp) in enumerate(folds, start=1):
        X_train, y_train = X[tr_idx], y_class[tr_idx]
        X_test, y_test = X[te_idx], y_class[te_idx]
        if len(np.unique(y_train)) < 2 or len(te_idx) == 0:
            log(f"  Permutation fold {fold_i}: skipped (train<2 classes or empty test).")
            continue
        if len(np.unique(y_test)) < 2:
            log(f"  Permutation fold {fold_i}: skipped (held-out test has <2 classes).")
            continue
        # Fit train-only pipeline exactly as the Gini path (fit_fold_random_oversample_rf):
        # StandardScaler(train) -> RandomOverSampler(train) -> RF(train).
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        ros = RandomOverSampler(random_state=random_state + fold_i)
        X_tr_s, y_tr = ros.fit_resample(X_train_s, y_train)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=random_state + fold_i,
            class_weight=None,
            n_jobs=-1,
        )
        clf.fit(X_tr_s, y_tr)
        # Permutation importance on the fold's HELD-OUT test rows (scaled with train scaler).
        imp = _run_permutation_importance(
            clf,
            X_test_s,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state + fold_i,
            perm_max_samples=perm_max_samples,
        )
        accum += imp
        n_ok += 1
        log(f"  Permutation fold {fold_i}: scored {len(X_test_s)} held-out rows.")
    if n_ok == 0:
        raise RuntimeError("No valid CV fold produced held-out permutation importance.")
    log(
        f"Permutation importance (held-out CV mean over {n_ok} folds) completed in "
        f"{(time.time() - t0):.1f}s."
    )
    return accum / n_ok


def run_rf_importance_section(
    args: argparse.Namespace,
    X: np.ndarray,
    y_class: np.ndarray,
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    species_ordered: List[str],
    n_classes: int,
    domain_names: List[str],
    out_dir: str,
) -> pd.DataFrame:
    if args.rf_importance_mode == "cv_mean":
        imp = compute_rf_importance_cv_mean(
            X,
            y_class,
            pair_sp1,
            pair_sp2,
            species_ordered,
            args.n_folds,
            args.random_state,
            args.n_estimators,
            n_classes,
            domain_names,
        )
    else:
        imp = compute_rf_importance_full_data(
            X, y_class, args.random_state, args.n_estimators, n_classes
        )

    if args.skip_permutation_importance:
        log("Skipping permutation importance (--skip-permutation-importance).")
        perm_imp = np.full(len(domain_names), np.nan, dtype=float)
    else:
        log(
            "Computing permutation importance on held-out CV test folds "
            "(supporting signal; this may take a few minutes)..."
        )
        perm_imp = compute_permutation_importance(
            X,
            y_class,
            pair_sp1,
            pair_sp2,
            species_ordered,
            args.n_folds,
            args.random_state,
            args.n_estimators,
            n_classes,
            domain_names,
            n_repeats=args.perm_n_repeats,
            perm_max_samples=args.perm_max_samples,
        )
    df = pd.DataFrame(
        {
            "domain_name": domain_names,
            "importance_score": imp,
            "permutation_importance": perm_imp,
        }
    )
    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    if args.skip_permutation_importance:
        df["perm_rank"] = np.nan
    else:
        df["perm_rank"] = df["permutation_importance"].rank(ascending=False).astype(int)
    path = os.path.join(out_dir, "rf_feature_importance.tsv")
    df.to_csv(path, sep="\t", index=False)
    log(f"Saved {path}")

    top = df.head(N_TOP_RF_PLOT).iloc[::-1]
    fig_path = os.path.join(out_dir, "rf_feature_importance_top20.png")
    if args.skip_permutation_importance:
        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax1.barh(range(len(top)), top["importance_score"].values, color="steelblue")
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top["domain_name"].values, fontsize=8)
        ax1.set_xlabel("Mean Gini importance (standardized L1-diff features)")
        ax1.set_title(f"Top {N_TOP_RF_PLOT} domains by RF ({args.rf_importance_mode}); permutation skipped")
        plt.tight_layout()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.barh(range(len(top)), top["importance_score"].values, color="steelblue")
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top["domain_name"].values, fontsize=8)
        ax1.set_xlabel("Mean Gini importance (standardized L1-diff features)")
        ax1.set_title("Gini importance (may favor rare domains)")
        ax2.barh(range(len(top)), top["permutation_importance"].values, color="steelblue")
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels(top["domain_name"].values, fontsize=8)
        ax2.set_xlabel("Permutation importance (held-out CV folds)")
        ax2.set_title("Permutation importance (held-out; diffuse w/ correlated features)")
        plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    log(f"Saved {fig_path}")
    return df


def plot_rf_vs_anova_concordance(rf_df: pd.DataFrame, anova_df: pd.DataFrame, out_dir: str) -> None:
    """
    Scatter of RF rank vs ANOVA rank for domains that passed ANOVA filters.
    """
    merged = pd.merge(
        anova_df[["domain_name", "rank"]],
        rf_df[["domain_name", "rank"]],
        on="domain_name",
        how="inner",
        suffixes=("_anova", "_rf"),
    )
    if merged.empty:
        log("No overlapping domains for rf_vs_anova_concordance plot; skipping.")
        return

    anova_threshold = int(np.ceil(0.25 * float(merged["rank_anova"].max())))
    rf_threshold = int(np.ceil(0.25 * float(merged["rank_rf"].max())))
    hi_anova = merged["rank_anova"] <= anova_threshold
    hi_rf = merged["rank_rf"] <= rf_threshold

    masks = {
        "Concordant important (top 25% both)": hi_anova & hi_rf,
        "RF-only signal (top 25% RF)": (~hi_anova) & hi_rf,
        "ANOVA-only signal (top 25% ANOVA)": hi_anova & (~hi_rf),
        "Low on both": (~hi_anova) & (~hi_rf),
    }
    colors = {
        "Concordant important (top 25% both)": "blue",
        "RF-only signal (top 25% RF)": "orange",
        "ANOVA-only signal (top 25% ANOVA)": "green",
        "Low on both": "grey",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for label, mask in masks.items():
        sub = merged[mask]
        ax.scatter(
            sub["rank_anova"].values,
            sub["rank_rf"].values,
            s=14,
            alpha=0.75,
            color=colors[label],
            label=label,
            edgecolors="none",
        )

    min_line = int(max(1, min(merged["rank_anova"].min(), merged["rank_rf"].min())))
    max_line = int(max(merged["rank_anova"].max(), merged["rank_rf"].max()))
    ax.plot([min_line, max_line], [min_line, max_line], linestyle="--", linewidth=1.0, color="black", label="y=x")

    top_rf = merged.nsmallest(20, "rank_rf")
    for _, row in top_rf.iterrows():
        accession = str(row["domain_name"]).split(":", 1)[0]
        ax.annotate(
            accession,
            (float(row["rank_anova"]), float(row["rank_rf"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ANOVA rank (1 = most significant)")
    ax.set_ylabel("RF importance rank (1 = highest importance)")
    ax.set_title("RF importance rank vs ANOVA rank (domain concordance)")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "rf_vs_anova_concordance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved {out_path}")


def shorten_clade_label(label: str) -> str:
    """Shorten clade label for x-axis display."""
    replacements = {
        "Laurasiatheria_Carnivora": "Carnivora",
        "Laurasiatheria_Cetartiodactyla": "Cetartiodactyla",
        "Laurasiatheria_Chiroptera": "Chiroptera",
        "Laurasiatheria_Perissodactyla": "Perissodactyla",
        "Laurasiatheria_Eulipotyphla": "Eulipotyphla",
        "Monotreme_outgroup": "Monotreme",
    }
    for long, short in replacements.items():
        label = label.replace(long, short)
    return label


def get_anova_clade_layout(
    X_norm: pd.DataFrame,
    species_to_clade: Dict[str, str],
    use_log1p: bool,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """
    Species×domain matrix and clade groupings used by ANOVA and clade-wise plots
    (same preprocessing as anova_across_clades).
    """
    X = sanitize_matrix(X_norm)
    if use_log1p:
        arr = np.log1p(X.values.astype(float))
        X = pd.DataFrame(arr, index=X.index, columns=X.columns)

    clade_counts: Dict[str, int] = {}
    for sp in X.index:
        c = species_to_clade[sp]
        clade_counts[c] = clade_counts.get(c, 0) + 1

    MIN_CLADE_N_ANOVA = 3
    valid_clades = sorted(
        [c for c, n in clade_counts.items() if n >= MIN_CLADE_N_ANOVA and c != "Monotreme_outgroup"]
    )
    excluded = sorted(
        (c, n) for c, n in clade_counts.items() if n < MIN_CLADE_N_ANOVA and c != "Monotreme_outgroup"
    )
    if excluded:
        detail = "; ".join(f"{c} (n={n})" for c, n in excluded)
        log(
            f"ANOVA: excluding clades with n < {MIN_CLADE_N_ANOVA} (not used in ANOVA or clade boxplots): {detail}"
        )
    if not valid_clades:
        raise RuntimeError(f"No clade with n>={MIN_CLADE_N_ANOVA} species for ANOVA.")

    species_keep = [sp for sp in X.index if species_to_clade[sp] in valid_clades]
    X_sub = X.loc[species_keep]
    clade_to_species: Dict[str, List[str]] = {
        clade: [sp for sp in X_sub.index if species_to_clade[sp] == clade] for clade in valid_clades
    }
    return X_sub, valid_clades, clade_to_species


def anova_across_clades(
    X_norm: pd.DataFrame,
    species_to_clade: Dict[str, str],
    out_dir: str,
    use_log1p: bool,
) -> pd.DataFrame:
    """
    One-way ANOVA per domain across clades with >=3 species per clade; omit empty Monotreme_outgroup.
    """
    X_sub, valid_clades, clade_to_species = get_anova_clade_layout(X_norm, species_to_clade, use_log1p)
    log(f"ANOVA: {len(valid_clades)} clades (each n>=3), {len(X_sub)} species.")

    # Filter A: keep domains present (>0 in at least one species) in >=3 clades.
    domains_after_a: List[str] = []
    n_removed_a = 0
    for dom in X_sub.columns:
        clades_present = 0
        for clade in valid_clades:
            vals = X_sub.loc[clade_to_species[clade], dom].values.astype(float)
            if np.any(vals > 0.0):
                clades_present += 1
        if clades_present >= 3:
            domains_after_a.append(dom)
        else:
            n_removed_a += 1
    log(f"ANOVA filter A removed {n_removed_a} domains (present in <3 clades)")

    # Filter B: remove domains where one clade contributes >90% of total clade means.
    domains_for_anova: List[str] = []
    n_removed_b = 0
    for dom in domains_after_a:
        clade_means: List[float] = []
        for clade in valid_clades:
            vals = X_sub.loc[clade_to_species[clade], dom].values.astype(float)
            clade_means.append(float(np.mean(vals)))
        total_mean = float(np.sum(clade_means))
        if total_mean <= 0.0:
            n_removed_b += 1
            continue
        dominance = float(np.max(clade_means) / total_mean)
        if dominance > 0.90:
            n_removed_b += 1
            continue
        domains_for_anova.append(dom)
    log(f"ANOVA filter B removed {n_removed_b} domains (>90% single-clade dominance)")

    rows: List[Dict[str, Any]] = []
    for dom in domains_for_anova:
        groups = []
        for clade in valid_clades:
            sp_c = clade_to_species[clade]
            groups.append(X_sub.loc[sp_c, dom].values.astype(float))
        try:
            f_stat, p_val = f_oneway(*groups)
        except Exception:
            f_stat, p_val = float("nan"), float("nan")
        rows.append(
            {
                "domain_name": dom,
                "F_statistic": f_stat,
                "p_value": p_val,
                "filter_applied": "none",
            }
        )

    df = pd.DataFrame(rows)
    p = df["p_value"].values.astype(float)
    df["adjusted_p_value"] = apply_fdr(np.nan_to_num(p, nan=1.0))
    df = df.sort_values(["adjusted_p_value", "F_statistic"], ascending=[True, False]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    qcol = df["adjusted_p_value"].astype(float)
    n_sig_05 = int((qcol < 0.05).sum())
    n_sig_01 = int((qcol < 0.01).sum())
    log(
        f"ANOVA (BH-FDR, existing adjusted_p_value): {n_sig_05} domains significant at "
        f"adjusted_p_value < 0.05; {n_sig_01} at < 0.01 (of {len(df)} tested after filters)"
    )
    path = os.path.join(out_dir, "anova_domains.tsv")
    df.to_csv(path, sep="\t", index=False)
    log(f"Saved {path}")

    log("Top 20 domains by ANOVA (lowest adjusted_p_value):")
    for _, r in df.head(N_TOP_ANOVA_PRINT).iterrows():
        log(
            f"  {r['domain_name'][:60]:60s} F={r['F_statistic']:.4g} padj={r['adjusted_p_value']:.4e}"
        )

    top_domains = df.head(N_TOP_ANOVA_BOXPLOT)["domain_name"].tolist()
    if top_domains:
        n = len(top_domains)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for idx, dom in enumerate(top_domains):
            ax = axes_flat[idx]
            data_by_clade: List[np.ndarray] = []
            labels: List[str] = []
            for clade in valid_clades:
                sp_c = clade_to_species[clade]
                data_by_clade.append(X_sub.loc[sp_c, dom].values.astype(float))
                labels.append(shorten_clade_label(f"{clade}\n(n={len(sp_c)})"))
            try:
                ax.boxplot(data_by_clade, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(data_by_clade, labels=labels, showfliers=False)
            ax.set_title(dom[:40] + ("..." if len(dom) > 40 else ""), fontsize=8)
            ax.tick_params(axis="x", labelrotation=45, labelsize=6)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        for j in range(len(top_domains), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Top ANOVA domains by clade (relative abundance)", fontsize=10)
        plt.tight_layout(rect=[0, 0.12, 1, 0.96])
        bp_path = os.path.join(out_dir, "anova_top_domains_boxplots.png")
        plt.savefig(bp_path, dpi=150)
        plt.close()
        log(f"Saved {bp_path}")

    return df


def marker_domains_analysis(
    X_norm: pd.DataFrame,
    species_to_clade: Dict[str, str],
    out_dir: str,
    use_log1p: bool,
) -> pd.DataFrame:
    """Per clade vs all others: Mann-Whitney U; global BH on all (group × domain) tests."""
    X = sanitize_matrix(X_norm)
    if use_log1p:
        arr = np.log1p(X.values.astype(float))
        X = pd.DataFrame(arr, index=X.index, columns=X.columns)

    # Only clades with >= MIN_CLADE_N_MARKER species are tested as a marker group; the
    # n<3 clades (e.g. Eulipotyphla n=1, Perissodactyla n=2) are degenerate for a
    # Mann-Whitney rank-sum and previously produced exact-test-impossible FDR hits via
    # the ties/asymptotic fallback. Their species REMAIN in the "rest" comparison set.
    clade_sizes = {g: sum(1 for sp in X.index if species_to_clade[sp] == g)
                   for g in {species_to_clade[sp] for sp in X.index}}
    groups_to_test = sorted([g for g, n in clade_sizes.items() if n >= MIN_CLADE_N_MARKER])
    excluded = sorted((g, n) for g, n in clade_sizes.items() if n < MIN_CLADE_N_MARKER)
    if excluded:
        log("Marker: excluding clades with n < "
            f"{MIN_CLADE_N_MARKER} (kept in the 'rest' set, not assigned markers): "
            + ", ".join(f"{g}(n={n})" for g, n in excluded))

    rows: List[Dict[str, Any]] = []
    for group_name in groups_to_test:
        in_sp = [sp for sp in X.index if species_to_clade[sp] == group_name]
        out_sp = [sp for sp in X.index if species_to_clade[sp] != group_name]
        if len(in_sp) == 0 or len(out_sp) == 0:
            continue
        for dom in X.columns:
            a = X.loc[in_sp, dom].values.astype(float)
            b = X.loc[out_sp, dom].values.astype(float)
            mean_in = float(np.mean(a))
            mean_out = float(np.mean(b))
            ratio = (mean_in + EPS_FOLDCHANGE) / (mean_out + EPS_FOLDCHANGE)
            log2fc = float(np.log2(ratio))
            try:
                # exact null for small focal clades: the normal approx reports
                # impossibly-small p for n=4 (floor 2/C(107,4)=3.88e-7).
                _m = "exact" if min(len(a), len(b)) <= 8 else "asymptotic"
                _, p_val = mannwhitneyu(a, b, alternative="two-sided", method=_m)
            except ValueError:
                p_val = 1.0
            rows.append(
                {
                    "group_name": group_name,
                    "domain_name": dom,
                    "mean_in_group": mean_in,
                    "mean_out_group": mean_out,
                    "log2_fold_change": log2fc,
                    "p_value": float(p_val),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        log("No marker rows produced.")
        return df
    df["adjusted_p_value"] = apply_fdr(df["p_value"].values)
    path = os.path.join(out_dir, "marker_domains.tsv")
    df.to_csv(path, sep="\t", index=False)
    log(f"Saved {path}")

    summary_rows: List[Dict[str, Any]] = []
    for g in sorted(df["group_name"].unique()):
        sub = df[df["group_name"] == g].copy()
        n_species_in_group = sum(1 for sp in X.index if species_to_clade[sp] == g)
        n_domains_tested = int(len(sub))
        n_significant_fdr05 = int((sub["adjusted_p_value"] < 0.05).sum())
        n_significant_fdr10 = int((sub["adjusted_p_value"] < 0.10).sum())

        sub_pos = sub[sub["log2_fold_change"] > 0].sort_values(
            ["adjusted_p_value", "p_value", "log2_fold_change"], ascending=[True, True, False]
        )
        sub_neg = sub[sub["log2_fold_change"] < 0].sort_values(
            ["adjusted_p_value", "p_value", "log2_fold_change"], ascending=[True, True, True]
        )
        top_enriched = str(sub_pos.iloc[0]["domain_name"]) if not sub_pos.empty else ""
        top_depleted = str(sub_neg.iloc[0]["domain_name"]) if not sub_neg.empty else ""

        summary_rows.append(
            {
                "group_name": g,
                "n_species_in_group": n_species_in_group,
                "n_domains_tested": n_domains_tested,
                "n_significant_fdr05": n_significant_fdr05,
                "n_significant_fdr10": n_significant_fdr10,
                "top_enriched_domain": top_enriched,
                "top_depleted_domain": top_depleted,
            }
        )
        log(
            f"Marker summary [{g}]: n_species={n_species_in_group}, "
            f"n_domains_tested={n_domains_tested}, FDR<0.05={n_significant_fdr05}, "
            f"FDR<0.10={n_significant_fdr10}"
        )
        if n_significant_fdr05 == 0:
            log(
                f"WARNING: group {g} has 0 marker domains at FDR<0.05 - "
                "consider per-group BH correction or relaxed threshold"
            )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "group_name",
            "n_species_in_group",
            "n_domains_tested",
            "n_significant_fdr05",
            "n_significant_fdr10",
            "top_enriched_domain",
            "top_depleted_domain",
        ],
    )
    summary_path = os.path.join(out_dir, "marker_domains_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    log(f"Saved {summary_path}")

    for g in sorted(df["group_name"].unique()):
        sub = df[df["group_name"] == g].copy()
        pos = sub[sub["log2_fold_change"] > 0].sort_values(
            ["p_value", "log2_fold_change"], ascending=[True, False]
        )
        neg = sub[sub["log2_fold_change"] < 0].sort_values(
            ["p_value", "log2_fold_change"], ascending=[True, True]
        )
        log(f"--- Group {g}: top {N_TOP_MARKERS} enriched (log2FC>0; by p_value, then log2FC) ---")
        for _, r in pos.head(N_TOP_MARKERS).iterrows():
            log(f"  {r['domain_name'][:50]:50s} log2FC={r['log2_fold_change']:.3f} p={r['p_value']:.4e}")
        if pos.empty:
            log("  (no domains with positive log2FC)")
        log(f"--- Group {g}: top {N_TOP_MARKERS} depleted (log2FC<0; by p_value, then log2FC) ---")
        for _, r in neg.head(N_TOP_MARKERS).iterrows():
            log(f"  {r['domain_name'][:50]:50s} log2FC={r['log2_fold_change']:.3f} p={r['p_value']:.4e}")
        if neg.empty:
            log("  (no domains with negative log2FC)")

    return df


def get_tree_leaf_order(tree_path: str, species_in_matrix: List[str]) -> List[str]:
    """Return species in leaf traversal order from the Newick tree."""
    try:
        from ete3 import Tree

        tree = Tree(tree_path, format=1)
        name_map = {
            "Carlito syrichta": "Tarsius syrichta",
            "Neovison vison": "Neogale vison",
            "Physeter macrocephalus": "Physeter catodon",
            "Bos mutus grunniens": "Bos grunniens",
        }
        reverse_map = {v: k for k, v in name_map.items()}
        leaf_order: List[str] = []
        for leaf in tree.iter_leaves():
            name = leaf.name.replace("_", " ")
            mapped = reverse_map.get(name, name)
            if mapped in species_in_matrix:
                leaf_order.append(mapped)
        in_tree = set(leaf_order)
        leaf_order += [s for s in species_in_matrix if s not in in_tree]
        return leaf_order
    except Exception as e:
        log(f"Could not load tree for heatmap ordering: {e}; falling back to clade sort.")
        return []


def optional_heatmap(
    X_norm: pd.DataFrame,
    species_to_clade: Dict[str, str],
    anova_df: pd.DataFrame,
    out_dir: str,
    use_log1p: bool,
) -> None:
    try:
        import seaborn as sns
    except ImportError:
        log("seaborn not installed; skipping heatmap_top_domains_by_clade.png")
        return

    X = sanitize_matrix(X_norm)
    if use_log1p:
        arr = np.log1p(X.values.astype(float))
        X = pd.DataFrame(arr, index=X.index, columns=X.columns)

    if anova_df is not None and len(anova_df) > 0:
        top_dom = anova_df.nlargest(N_HEATMAP_DOMAINS, "F_statistic")["domain_name"].tolist()
    else:
        var = X.var(axis=0).sort_values(ascending=False)
        top_dom = var.head(N_HEATMAP_DOMAINS).index.tolist()

    X_plot = X[top_dom]
    clade_series = pd.Series({sp: species_to_clade.get(sp, "Other") for sp in X_plot.index})
    tree_path = os.path.join(PROJECT_ROOT, "MammalsPhylogeny.nwk")
    leaf_order = get_tree_leaf_order(tree_path, list(X_plot.index))
    if leaf_order:
        order = [s for s in leaf_order if s in X_plot.index]
        in_order = set(order)
        order += [s for s in X_plot.index if s not in in_order]
    else:
        order = sorted(X_plot.index, key=lambda s: (clade_series[s], s))
    X_ord = X_plot.loc[order]

    Z = X_ord.values.astype(float).copy()
    for j in range(Z.shape[1]):
        col = Z[:, j]
        mu = float(np.mean(col))
        sd = float(np.std(col))
        if not np.isfinite(sd) or sd < 1e-12:
            Z[:, j] = 0.0
        else:
            Z[:, j] = (col - mu) / sd
    X_z = pd.DataFrame(Z, index=X_ord.index, columns=X_ord.columns)

    fig_h = max(8, len(X_z) * 0.12)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.heatmap(
        X_z,
        ax=ax,
        cmap="viridis",
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "z-score (per domain)"},
    )
    n_rows = len(order)
    for i in range(1, n_rows):
        if clade_series[order[i]] != clade_series[order[i - 1]]:
            ax.axhline(
                y=float(i),
                color="white",
                linewidth=2.0,
                zorder=20,
                clip_on=False,
            )
    abund_note = "log1p abundance" if use_log1p else "relative frequency"
    ax.set_title(
        f"Top {len(top_dom)} domains (by ANOVA F); per-domain z-score of {abund_note}; species by clade"
    )
    plt.yticks(rotation=0, fontsize=5)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    hpath = os.path.join(out_dir, "heatmap_top_domains_by_clade.png")
    plt.savefig(hpath, dpi=150)
    plt.close()
    log(f"Saved {hpath}")


def write_anova_rf_top100_intersection_outputs(
    X_norm: pd.DataFrame,
    species_to_clade: Dict[str, str],
    use_log1p: bool,
    anova_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    Top-N ANOVA domains ∩ top-N RF domains: two TSV orderings + multi-page clade boxplots PDF.
    """
    n = N_TOP_ANOVA_RF_INTERSECTION
    top_anova = set(anova_df.head(n)["domain_name"])
    top_rf = set(rf_df.head(n)["domain_name"])
    inter = top_anova & top_rf
    if not inter:
        log("ANOVA∩RF top-100 intersection is empty; skipping intersection TSVs and PDF.")
        return

    anova_by_dom = anova_df.set_index("domain_name")
    rf_by_dom = rf_df.set_index("domain_name")

    def overlap_rows(domains: List[str]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for d in domains:
            ar = anova_by_dom.loc[d]
            rr = rf_by_dom.loc[d]
            rows.append(
                {
                    "domain_name": d,
                    "anova_rank": int(ar["rank"]),
                    "adjusted_p_value": float(ar["adjusted_p_value"]),
                    "F_statistic": float(ar["F_statistic"]),
                    "rf_rank": int(rr["rank"]),
                    "importance_score": float(rr["importance_score"]),
                }
            )
        return pd.DataFrame(rows)

    ordered_anova = [d for d in anova_df["domain_name"] if d in inter]
    ordered_rf = [d for d in rf_df["domain_name"] if d in inter]

    tsv_anova = os.path.join(out_dir, "anova_rf_top100_intersection_by_anova.tsv")
    tsv_rf = os.path.join(out_dir, "anova_rf_top100_intersection_by_rf.tsv")
    overlap_rows(ordered_anova).to_csv(tsv_anova, sep="\t", index=False)
    overlap_rows(ordered_rf).to_csv(tsv_rf, sep="\t", index=False)
    log(f"Saved {tsv_anova}")
    log(f"Saved {tsv_rf}")

    X_sub, valid_clades, clade_to_species = get_anova_clade_layout(X_norm, species_to_clade, use_log1p)
    per_page = OVERLAP_PLOT_ROWS * OVERLAP_PLOT_COLS
    pdf_path = os.path.join(out_dir, "anova_rf_top100_intersection_by_clade.pdf")
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(ordered_anova), per_page):
            chunk = ordered_anova[start : start + per_page]
            fig, axes = plt.subplots(
                OVERLAP_PLOT_ROWS,
                OVERLAP_PLOT_COLS,
                figsize=(4.8 * OVERLAP_PLOT_COLS, 4.2 * OVERLAP_PLOT_ROWS),
            )
            axes_flat = np.atleast_1d(axes).ravel()
            for i, dom in enumerate(chunk):
                ax = axes_flat[i]
                data_by_clade: List[np.ndarray] = []
                labels: List[str] = []
                for clade in valid_clades:
                    sp_c = clade_to_species[clade]
                    data_by_clade.append(X_sub.loc[sp_c, dom].values.astype(float))
                    labels.append(shorten_clade_label(f"{clade}\n(n={len(sp_c)})"))
                try:
                    ax.boxplot(data_by_clade, tick_labels=labels, showfliers=False)
                except TypeError:
                    ax.boxplot(data_by_clade, labels=labels, showfliers=False)
                ax.set_title(dom[:35] + ("..." if len(dom) > 35 else ""), fontsize=7)
                ax.tick_params(axis="x", labelrotation=45, labelsize=6)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
            for j in range(len(chunk), len(axes_flat)):
                axes_flat[j].set_visible(False)
            abund_note = "log1p abundance" if use_log1p else "relative abundance"
            fig.suptitle(
                f"ANOVA∩RF top-{n} (ANOVA order), pages {start // per_page + 1}; {abund_note} by clade",
                fontsize=9,
            )
            fig.tight_layout(rect=[0, 0.04, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
    log(f"Saved {pdf_path}")


def validate_domain_analysis_outputs(out_dir: str, species_to_clade: Dict[str, str]) -> None:
    """Run final validation checks and print pass/fail summary."""
    failures: List[str] = []
    lower_after_colon = re.compile(r"^PF\d{5}:[a-z]")
    artifact_accessions = {
        "PF01274",
        "PF20656",
        "PF20659",
        "PF03575",
        "PF11474",
        "PF17137",
        "PF15916",
        "PF21109",
    }

    rf_tsv = os.path.join(out_dir, "rf_feature_importance.tsv")
    anova_tsv = os.path.join(out_dir, "anova_domains.tsv")
    marker_tsv = os.path.join(out_dir, "marker_domains.tsv")
    marker_summary_tsv = os.path.join(out_dir, "marker_domains_summary.tsv")
    concordance_png = os.path.join(out_dir, "rf_vs_anova_concordance.png")
    source_headers = set(load_authoritative_domain_headers(DOMAIN_COUNTS_PATH_LOCAL))

    # Check 1: no lowercase immediately after colon in domain labels.
    check1_issues: List[str] = []
    check1_bad_labels: set[str] = set()
    tsv_domain_cols = [
        (rf_tsv, ["domain_name"]),
        (anova_tsv, ["domain_name"]),
        (marker_tsv, ["domain_name"]),
        (marker_summary_tsv, ["top_enriched_domain", "top_depleted_domain"]),
    ]
    for path, cols in tsv_domain_cols:
        if not os.path.exists(path):
            check1_issues.append(f"{os.path.basename(path)} missing")
            continue
        df = pd.read_csv(path, sep="\t")
        for col in cols:
            if col not in df.columns:
                continue
            vals = df[col].fillna("").astype(str)
            bad = vals[vals.str.match(lower_after_colon)]
            if len(bad) > 0:
                check1_bad_labels.update([x for x in bad.tolist() if x])
                check1_issues.append(f"{os.path.basename(path)}:{col} bad={len(bad)} sample='{bad.iloc[0]}'")
    if check1_issues:
        check1_non_source = sorted([x for x in check1_bad_labels if x not in source_headers])
        if len(check1_non_source) == 0:
            log(
                "CHECK 1 WARN: "
                f"{len(check1_bad_labels)} labels have lowercase-after-colon - confirmed present in source "
                "data_raw/MammalDomainCount.tsv headers, not introduced by this script"
            )
            log("  CHECK 1 samples: " + "; ".join(sorted(list(check1_bad_labels))[:5]))
        else:
            failures.append("Check 1 failed (lowercase after colon): " + "; ".join(check1_issues))
            log("CHECK 1 FAIL: lowercase-after-colon labels not fully explained by source headers.")
            log("  CHECK 1 non-source samples: " + "; ".join(check1_non_source[:5]))
    else:
        log("CHECK 1 PASS: no domain label starts with lowercase immediately after colon.")

    # Check 2: artifacts absent from top 10 ANOVA rows.
    if not os.path.exists(anova_tsv):
        failures.append("Check 2 failed: anova_domains.tsv missing.")
        log("CHECK 2 FAIL: anova_domains.tsv missing.")
    else:
        adf = pd.read_csv(anova_tsv, sep="\t")
        top10 = adf.head(10)["domain_name"].fillna("").astype(str).tolist() if "domain_name" in adf.columns else []
        top10_accessions = {name.split(":", 1)[0] for name in top10 if ":" in name}
        overlap = sorted(artifact_accessions & top10_accessions)
        if overlap:
            failures.append(f"Check 2 failed: artifact accessions in ANOVA top10: {overlap}")
            log(f"CHECK 2 FAIL: artifact domains present in ANOVA top 10: {overlap}")
        else:
            log("CHECK 2 PASS: artifact domains absent from ANOVA top 10.")

    # Check 3: marker summary exists and one row per TESTED clade group (n >=
    # MIN_CLADE_N_MARKER; tiny clades are excluded from marker testing, see
    # marker_domains_analysis).
    _msizes: Dict[str, int] = {}
    for _sp, _g in species_to_clade.items():
        _msizes[_g] = _msizes.get(_g, 0) + 1
    expected_groups = sorted(g for g, n in _msizes.items() if n >= MIN_CLADE_N_MARKER)
    if not os.path.exists(marker_summary_tsv):
        failures.append("Check 3 failed: marker_domains_summary.tsv missing.")
        log("CHECK 3 FAIL: marker_domains_summary.tsv missing.")
    else:
        sdf = pd.read_csv(marker_summary_tsv, sep="\t")
        observed_groups = sorted(sdf["group_name"].astype(str).tolist()) if "group_name" in sdf.columns else []
        if len(observed_groups) != len(expected_groups) or observed_groups != expected_groups:
            failures.append(
                "Check 3 failed: marker summary rows/groups mismatch "
                f"(expected {len(expected_groups)} groups, got {len(observed_groups)})."
            )
            log(
                "CHECK 3 FAIL: marker summary row/group mismatch "
                f"(expected {len(expected_groups)} groups, got {len(observed_groups)})."
            )
        else:
            log(f"CHECK 3 PASS: marker summary has one row per clade group ({len(expected_groups)} groups).")

    # Check 4: concordance PNG exists.
    if os.path.exists(concordance_png):
        log("CHECK 4 PASS: rf_vs_anova_concordance.png exists.")
    else:
        failures.append("Check 4 failed: rf_vs_anova_concordance.png missing.")
        log("CHECK 4 FAIL: rf_vs_anova_concordance.png missing.")

    # Check 5: RF top-20 labels readable via source labels (no truncation pattern).
    if not os.path.exists(rf_tsv):
        failures.append("Check 5 failed: rf_feature_importance.tsv missing.")
        log("CHECK 5 FAIL: rf_feature_importance.tsv missing.")
    else:
        rdf = pd.read_csv(rf_tsv, sep="\t")
        top20_labels = rdf.head(N_TOP_RF_PLOT)["domain_name"].fillna("").astype(str) if "domain_name" in rdf.columns else pd.Series(dtype=str)
        bad_top20 = top20_labels[top20_labels.str.match(lower_after_colon)]
        if len(bad_top20) > 0:
            bad_top20_set = set([x for x in bad_top20.tolist() if x])
            non_source_top20 = sorted([x for x in bad_top20_set if x not in source_headers])
            if len(non_source_top20) == 0:
                log(
                    "CHECK 5 WARN: "
                    f"{len(bad_top20_set)} top-20 labels have lowercase-after-colon - confirmed present in source "
                    "data_raw/MammalDomainCount.tsv headers, not introduced by this script"
                )
                log("  CHECK 5 samples: " + "; ".join(sorted(list(bad_top20_set))[:5]))
            else:
                failures.append(
                    "Check 5 failed: top-20 RF labels include lowercase-after-colon domains "
                    f"not in source headers (sample '{non_source_top20[0]}')."
                )
                log("CHECK 5 FAIL: rf_feature_importance_top20 labels include non-source malformed domains.")
        else:
            log("CHECK 5 PASS: rf_feature_importance_top20 source labels have no truncation pattern.")

    if failures:
        log("DOMAIN ANALYSIS VALIDATION: CHECKS FAILED")
        for item in failures:
            log(f"  - {item}")
    else:
        log("DOMAIN ANALYSIS VALIDATION: ALL CHECKS PASSED")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", default="SCHEME_4B", choices=["SCHEME_4B", "SCHEME_4C"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help=f"Output directory (default: {OUTPUT_DIR_DEFAULT})",
    )
    p.add_argument(
        "--rf-importance-mode",
        choices=["cv_mean", "full_data"],
        default="cv_mean",
        help="cv_mean: average importances across species-blocked folds; full_data: one RF on all pairs.",
    )
    p.add_argument(
        "--log1p-abundance",
        action="store_true",
        help="Apply log1p to species×domain matrix for ANOVA/markers/heatmap only (RF still uses pipeline X).",
    )
    p.add_argument("--skip-heatmap", action="store_true")
    p.add_argument(
        "--skip-rf",
        action="store_true",
        help=(
            "Skip pairwise feature construction and RF training; load rf_feature_importance.tsv "
            "from output dir (must exist from a prior run). Regenerates ANOVA, markers, heatmap, "
            "concordance, and overlap artifacts that depend on clades / ANOVA."
        ),
    )
    p.add_argument(
        "--skip-permutation-importance",
        action="store_true",
        help="When running RF: skip permutation_importance (Gini-only RF section; much faster).",
    )
    p.add_argument(
        "--perm-max-samples",
        type=int,
        default=PERMUTATION_MAX_SAMPLES_DEFAULT,
        help=(
            "Permutation importance: max rows used when scoring each shuffled feature "
            f"(default {PERMUTATION_MAX_SAMPLES_DEFAULT}; RF still trained on full ROS matrix). "
            "Use 0 to disable cap (slowest, closest to former behavior)."
        ),
    )
    p.add_argument(
        "--perm-n-repeats",
        type=int,
        default=PERMUTATION_N_REPEATS_DEFAULT,
        help=f"Permutation repeats per feature (default {PERMUTATION_N_REPEATS_DEFAULT}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.output_dir)
    ensure_dir(out_dir)

    # Step A: Same species × domain matrix as the CV tree classifier (platypus out, row-normalized).
    log("Loading and preprocessing (same as classification_cv_tree_reconstruction)")
    X_norm, dist_matrix, species_final = load_and_preprocess_data()
    X_norm = sanitize_matrix(X_norm)
    species_ordered = sorted(species_final)
    if list(species_final) != species_ordered:
        log("Reordering species to sorted order for consistency with pairwise features.")
    X_norm = X_norm.loc[species_ordered]
    authoritative_headers = load_authoritative_domain_headers(DOMAIN_COUNTS_PATH_LOCAL)
    remapped_cols = remap_to_authoritative_domain_headers(list(X_norm.columns), authoritative_headers)
    if remapped_cols != list(X_norm.columns):
        X_norm = X_norm.copy()
        X_norm.columns = remapped_cols

    # Step B: Clade labels from elastic_net stratified CV mapping (Other = unlisted taxa).
    species_to_clade = assign_species_to_clades(list(X_norm.index))
    clade_counts: Dict[str, int] = {}
    for sp in X_norm.index:
        c = species_to_clade[sp]
        clade_counts[c] = clade_counts.get(c, 0) + 1
    log("Species per clade:")
    for c, n in sorted(clade_counts.items()):
        log(f"  {c}: {n}")
    n_other = clade_counts.get("Other", 0)
    if n_other > len(X_norm) * 0.15:
        log(f"WARNING: {n_other} species in Other ({100*n_other/len(X_norm):.1f}%) - check naming.")

    schemes = build_scheme_definitions()
    spec = schemes[args.scheme]
    cutoffs = [float(x) for x in spec["cutoffs"]]
    class_labels = list(spec["labels"])

    if args.skip_rf:
        log("Skipping pairwise features and RF (--skip-rf); using cached rf_feature_importance.tsv")
        rf_df = load_rf_importance_from_tsv(out_dir)
    else:
        # Step C: Pairwise L1 features + distance bins (same scheme as tree reconstruction).
        X, y_dist, pair_sp1, pair_sp2 = construct_pairwise_features(X_norm, dist_matrix, species_ordered)
        y_class, n_classes, _ = assign_classes(y_dist, cutoffs, class_labels)
        domain_names = list(X_norm.columns)

        # Step 1: RF - mean feature_importances_ over species-blocked folds (or full_data mode).
        log("RF feature importance (importances are 1:1 with domains; optional CV mean across folds)")
        rf_df = run_rf_importance_section(
            args,
            X,
            y_class,
            pair_sp1,
            pair_sp2,
            species_ordered,
            n_classes,
            domain_names,
            out_dir,
        )

    # Step 2: ANOVA on species-level abundances across clades (n≥3 per clade); BH-FDR.
    log("One-way ANOVA across clades + BH-FDR")
    anova_df = anova_across_clades(X_norm, species_to_clade, out_dir, args.log1p_abundance)
    plot_rf_vs_anova_concordance(rf_df, anova_df, out_dir)
    write_anova_rf_top100_intersection_outputs(
        X_norm, species_to_clade, args.log1p_abundance, anova_df, rf_df, out_dir
    )

    # Step 3: Each clade vs rest - Mann-Whitney, log2FC; BH-FDR over all group×domain tests.
    # Use RAW counts (same basis as the canonical per-clade enrichment in
    # domain_clade_enrichment) - relative frequency creates a Chiroptera closure
    # artifact. FDR here is GLOBAL across all group×domain tests (vs the enrichment's
    # per-clade BH), so counts differ modestly; both are raw-count and artifact-free.
    log("Marker domains (Mann-Whitney vs rest, RAW counts) + global BH")
    import domain_clade_enrichment as _dce
    X_raw_marker = _dce.load_pfam_style_tsv(
        _dce.INPUT_TSV_PATH, metadata_skiprows=_dce.METADATA_SKIPROWS
    ).loc[list(X_norm.index), list(X_norm.columns)]
    marker_domains_analysis(X_raw_marker, species_to_clade, out_dir, args.log1p_abundance)

    # Optional: heatmap of top domains by ANOVA F, species ordered by clade (seaborn if installed).
    if not args.skip_heatmap:
        optional_heatmap(
            X_norm,
            species_to_clade,
            anova_df,
            out_dir,
            args.log1p_abundance,
        )

    validate_domain_analysis_outputs(out_dir, species_to_clade)
    log("Done.")


if __name__ == "__main__":
    main()
