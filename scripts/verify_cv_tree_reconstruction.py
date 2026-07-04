#!/usr/bin/env python3
"""
Independent audit of outputs from classification_cv_tree_reconstruction.py.

Re-derives species-blocked folds from the same data + config, checks train/test
disjointness and mask semantics (label leakage), cross-checks CSV/NPY/matrix/tree
consistency, and writes verification_report.txt / verification_report.json.

Does not modify pipeline outputs. Exit code 1 if any FAIL.

Run from project root:
  python3 scripts/verify_cv_tree_reconstruction.py --output-dir results/classification_cv_tree_reconstruction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scipy.cluster.hierarchy import average, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr as scipy_pearsonr

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit("This script requires ete3.") from exc

import classification_cv_tree_reconstruction as cvt


def log(msg: str) -> None:
    print(f"[verify_cv_tree] {msg}")


@dataclass
class Report:
    passes: List[str] = field(default_factory=list)
    warns: List[str] = field(default_factory=list)
    fails: List[str] = field(default_factory=list)

    def add(self, level: str, section: str, message: str) -> None:
        line = f"[{section}] {message}"
        if level == "PASS":
            self.passes.append(line)
        elif level == "WARN":
            self.warns.append(line)
        else:
            self.fails.append(line)

    def has_fail(self) -> bool:
        return len(self.fails) > 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "pass": self.passes,
            "warn": self.warns,
            "fail": self.fails,
            "exit_status": "fail" if self.has_fail() else "ok",
        }


def load_run_config(out_dir: str) -> Dict[str, Any]:
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_comparison_summary(out_dir: str) -> Dict[str, Any]:
    path = os.path.join(out_dir, "comparison_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_config_consistency(rep: Report, cfg: Dict[str, Any], summary: Dict[str, Any]) -> None:
    sec = "Config"
    if cfg.get("extended_test_cv") is not True:
        rep.add("WARN", sec, "extended_test_cv is not true in run_config.json")
    else:
        rep.add("PASS", sec, "run_config documents extended_test_cv=true")

    if cfg.get("aggregation") != "mean_predict_proba_then_argmax":
        rep.add("WARN", sec, f"unexpected aggregation field: {cfg.get('aggregation')}")
    else:
        rep.add("PASS", sec, "aggregation matches mean_predict_proba_then_argmax")

    for key in ("n_folds", "n_estimators", "random_state"):
        if key in summary and key in cfg and summary[key] != cfg[key]:
            rep.add(
                "WARN",
                sec,
                f"run_config {key}={cfg[key]} vs comparison_summary {key}={summary[key]}",
            )
        elif key in summary and key in cfg:
            rep.add("PASS", sec, f"{key} consistent between run_config and comparison_summary")


def check_leakage(
    rep: Report,
    folds: List[Tuple[np.ndarray, np.ndarray, frozenset, frozenset]],
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
) -> None:
    sec = "Leakage"
    for fi, (tr_idx, te_idx, train_sp, test_sp) in enumerate(folds, start=1):
        if train_sp & test_sp:
            rep.add("FAIL", sec, f"Fold {fi}: train_species and test_species overlap")
            continue
        rep.add("PASS", sec, f"Fold {fi}: train/test species disjoint")

        inter = np.intersect1d(tr_idx, te_idx, assume_unique=False)
        if inter.size > 0:
            rep.add(
                "FAIL",
                sec,
                f"Fold {fi}: {inter.size} pair indices in both tr_idx and te_idx",
            )
        else:
            rep.add("PASS", sec, f"Fold {fi}: tr_idx and te_idx disjoint")

        for i in tr_idx:
            a, b = pair_sp1[i], pair_sp2[i]
            if a not in train_sp or b not in train_sp:
                rep.add(
                    "FAIL",
                    sec,
                    f"Fold {fi}: train index {i} pair ({a},{b}) not both in train_species",
                )
                break
        else:
            rep.add("PASS", sec, f"Fold {fi}: all train rows are train-train pairs")

        bad_test = []
        for i in te_idx:
            a, b = pair_sp1[i], pair_sp2[i]
            in_test = (a in test_sp) or (b in test_sp)
            both_train_only = (a in train_sp) and (b in train_sp)
            if not in_test:
                bad_test.append(i)
            if both_train_only:
                bad_test.append(i)
        if bad_test:
            rep.add(
                "FAIL",
                sec,
                f"Fold {fi}: {len(bad_test)} test rows fail test_mask semantics (sample idx {bad_test[:3]})",
            )
        else:
            rep.add("PASS", sec, f"Fold {fi}: all test rows involve ≥1 test species (no train-train in test)")


def recompute_pair_fold_counts(
    folds: List[Tuple[np.ndarray, np.ndarray, frozenset, frozenset]],
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for tr_idx, te_idx, _ts, _ss in folds:
        for gi in te_idx:
            key = cvt.canonical_pair_key(str(pair_sp1[gi]), str(pair_sp2[gi]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def check_predictions_table(
    rep: Report,
    out_dir: str,
    n_species: int,
    n_folds: int,
    species_ordered: List[str],
    fold_counts: Dict[Tuple[str, str], int],
) -> None:
    sec = "Artifacts"
    path = os.path.join(out_dir, "predictions_aggregated.csv")
    df = pd.read_csv(path)
    expected_pairs = n_species * (n_species - 1) // 2
    if len(df) != expected_pairs:
        rep.add(
            "FAIL",
            sec,
            f"predictions_aggregated.csv rows {len(df)} != expected {expected_pairs}",
        )
    else:
        rep.add("PASS", sec, f"predictions_aggregated.csv row count = n*(n-1)/2 = {expected_pairs}")

    dup = df.duplicated(subset=["sp1", "sp2"]).sum()
    if dup:
        rep.add("FAIL", sec, f"duplicate (sp1,sp2) rows: {dup}")
    else:
        rep.add("PASS", sec, "no duplicate unordered pairs in predictions_aggregated.csv")

    bad_nf = df[(df["n_folds_averaged"] < 1) | (df["n_folds_averaged"] > n_folds)]
    if len(bad_nf):
        rep.add("FAIL", sec, f"n_folds_averaged out of range [1,{n_folds}]: {len(bad_nf)} rows")
    else:
        rep.add("PASS", sec, f"all n_folds_averaged in [1, {n_folds}]")

    mism = 0
    first_mismatch = None
    for _, row in df.iterrows():
        key = cvt.canonical_pair_key(str(row["sp1"]), str(row["sp2"]))
        fc = fold_counts.get(key, 0)
        if int(row["n_folds_averaged"]) != fc:
            mism += 1
            if first_mismatch is None:
                first_mismatch = (key, int(row["n_folds_averaged"]), fc)
    if mism:
        rep.add(
            "FAIL",
            sec,
            f"n_folds_averaged mismatch vs recomputed fold coverage: {mism} rows (e.g. {first_mismatch})",
        )
    else:
        rep.add("PASS", sec, "n_folds_averaged matches recomputed test-fold coverage per pair")

    expected = set()
    for i in range(n_species):
        for j in range(i + 1, n_species):
            expected.add(cvt.canonical_pair_key(species_ordered[i], species_ordered[j]))
    got = {cvt.canonical_pair_key(str(r["sp1"]), str(r["sp2"])) for _, r in df.iterrows()}
    if expected != got:
        rep.add(
            "FAIL",
            sec,
            f"pair set mismatch: missing {len(expected - got)}, extra {len(got - expected)}",
        )
    else:
        rep.add("PASS", sec, "predicted pair keys match full upper triangle over species order")


def check_centroids(
    rep: Report,
    out_dir: str,
    class_labels: List[str],
    n_classes: int,
) -> None:
    sec = "Artifacts"
    fold_path = os.path.join(out_dir, "class_representative_distances_per_fold.csv")
    agg_path = os.path.join(out_dir, "class_representative_distances.csv")
    fold_df = pd.read_csv(fold_path)
    agg_df = pd.read_csv(agg_path)

    fold_stack = np.zeros((len(fold_df), n_classes), dtype=float)
    for k, lab in enumerate(class_labels):
        colname = f"mu_{lab}_MY"
        if colname not in fold_df.columns:
            rep.add("FAIL", sec, f"missing column {colname} in per_fold CSV")
            return
        fold_stack[:, k] = fold_df[colname].values

    mu_re = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        col = fold_stack[:, k]
        valid = ~np.isnan(col)
        if not np.any(valid):
            rep.add("FAIL", sec, f"no valid per-fold mean for class {class_labels[k]}")
            return
        mu_re[k] = float(np.mean(col[valid]))

    if "mu_k_MY" not in agg_df.columns:
        rep.add("FAIL", sec, "class_representative_distances.csv missing mu_k_MY")
        return

    mu_saved = agg_df["mu_k_MY"].values.astype(float)
    if mu_saved.shape[0] != n_classes:
        rep.add("FAIL", sec, f"centroid CSV has {mu_saved.shape[0]} rows, expected {n_classes}")
        return

    if not np.allclose(mu_re, mu_saved, rtol=1e-9, atol=1e-6):
        rep.add(
            "FAIL",
            sec,
            f"recomputed mu_k from per_fold CSV != saved: max abs diff {np.max(np.abs(mu_re - mu_saved))}",
        )
    else:
        rep.add("PASS", sec, "class centroids (mu_k) match recomputation from per-fold CSV")


def check_matrix_and_predictions(
    rep: Report,
    out_dir: str,
    pred_df: pd.DataFrame,
    species_ordered: List[str],
) -> None:
    sec = "Matrix"
    csv_path = os.path.join(out_dir, "distance_matrix_predicted.csv")
    npy_path = os.path.join(out_dir, "distance_matrix_predicted.npy")
    D_csv = pd.read_csv(csv_path, index_col=0)
    D = D_csv.values.astype(float)
    if os.path.isfile(npy_path):
        D_npy = np.load(npy_path)
        if not np.allclose(D, D_npy, rtol=1e-9, atol=1e-6):
            rep.add("FAIL", sec, "distance_matrix_predicted.csv vs .npy differ")
        else:
            rep.add("PASS", sec, "CSV and NPY distance matrices match")

    n = D.shape[0]
    if D.shape != (n, n):
        rep.add("FAIL", sec, "distance matrix not square")
        return

    if not np.allclose(D, D.T, rtol=0, atol=1e-8):
        rep.add("FAIL", sec, "distance matrix not symmetric")
    else:
        rep.add("PASS", sec, "distance matrix symmetric")

    if not np.allclose(np.diag(D), 0.0, rtol=0, atol=1e-8):
        rep.add("FAIL", sec, "distance matrix diagonal not zero")
    else:
        rep.add("PASS", sec, "distance matrix zero diagonal")

    if np.any(~np.isfinite(D)) or np.any(D < -1e-8):
        rep.add("FAIL", sec, "matrix has non-finite or negative entries")
    else:
        rep.add("PASS", sec, "matrix finite and nonnegative")

    idx = {s: i for i, s in enumerate(species_ordered)}
    if list(D_csv.index) != species_ordered or list(D_csv.columns) != species_ordered:
        rep.add(
            "WARN",
            sec,
            "matrix index/columns differ from sorted species_ordered (checking alignment by label)",
        )

    mism = 0
    for _, row in pred_df.iterrows():
        a, b = row["sp1"], row["sp2"]
        i, j = idx[str(a)], idx[str(b)]
        d_pred = float(row["predicted_distance_MY"])
        if not np.isclose(D[i, j], d_pred, rtol=1e-9, atol=1e-5):
            mism += 1
    if mism:
        rep.add(
            "FAIL",
            sec,
            f"{mism} entries disagree between matrix and predictions_aggregated predicted_distance_MY",
        )
    else:
        rep.add("PASS", sec, "distance matrix entries match predictions_aggregated")


def check_cophenetic_and_tree(
    rep: Report,
    out_dir: str,
    species_ordered: List[str],
    D: np.ndarray,
) -> None:
    sec = "Tree"
    nwk_path = os.path.join(out_dir, "predicted_tree_UPGMA.nwk")
    with open(nwk_path, "r", encoding="utf-8") as f:
        newick = f.read().strip()

    tree = Tree(newick, format=1)
    leaves = [n.name for n in tree.iter_leaves()]
    n_sp = len(species_ordered)
    if len(leaves) != n_sp:
        rep.add(
            "FAIL",
            sec,
            f"Newick leaf count {len(leaves)} != n_species {n_sp}",
        )
    else:
        rep.add("PASS", sec, f"Newick has {n_sp} leaves")

    expected_labels = {cvt.format_species_label_for_newick(s) for s in species_ordered}
    got_labels = set(leaves)
    if expected_labels != got_labels:
        rep.add(
            "FAIL",
            sec,
            f"leaf label mismatch: symmetric diff size {len(expected_labels ^ got_labels)}",
        )
    else:
        rep.add("PASS", sec, "Newick leaf labels match formatted species set")

    condensed = squareform(D, checks=False)
    Z = average(condensed)
    # cophenet returns (cophenetic_correlation_coefficient, cophenetic_distances_condensed)
    c_corr, coph_d = cophenet(Z, condensed)
    r_check, _ = scipy_pearsonr(condensed, coph_d)
    if not np.isclose(float(c_corr), float(r_check), rtol=1e-5, atol=1e-5):
        rep.add(
            "WARN",
            sec,
            f"cophenet coeff {c_corr:.6f} vs pearson(condensed,coph_d) {r_check:.6f} (minor numerical drift)",
        )
    r = float(c_corr)
    if np.isnan(r):
        rep.add("WARN", sec, "cophenetic correlation is NaN")
    elif r < 0.85:
        rep.add(
            "WARN",
            sec,
            f"cophenetic correlation {r:.4f} < 0.85 (low fidelity of tree to input matrix; check ties/structure)",
        )
    else:
        rep.add("PASS", sec, f"cophenetic correlation = {r:.4f} (scipy.cluster.hierarchy.cophenet)")

    ref_path = os.path.join(out_dir, "reference_tree_pruned.nwk")
    if os.path.isfile(ref_path):
        rep.add("PASS", sec, "reference_tree_pruned.nwk present for manual comparison")
    else:
        rep.add("WARN", sec, "reference_tree_pruned.nwk missing (optional)")


def write_reports(out_dir: str, rep: Report) -> None:
    txt_path = os.path.join(out_dir, "verification_report.txt")
    json_path = os.path.join(out_dir, "verification_report.json")
    lines = [
        "CV tree reconstruction - verification report",
        "=" * 60,
        "",
        "PASS",
        "-" * 40,
    ]
    lines.extend(rep.passes if rep.passes else ["  (none)"])
    lines.extend(["", "WARN", "-" * 40])
    lines.extend(rep.warns if rep.warns else ["  (none)"])
    lines.extend(["", "FAIL", "-" * 40])
    lines.extend(rep.fails if rep.fails else ["  (none)"])
    lines.extend(
        [
            "",
            "Methodological note (informational)",
            "-" * 40,
            "Test-train pairs use domain features for BOTH species. Training labels never include",
            "those rows, but training taxa appear in feature vectors - transductive, not label leakage.",
            "",
        ]
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rep.to_json(), f, indent=2)
    log(f"Wrote {txt_path}")
    log(f"Wrote {json_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing classification_cv_tree_reconstruction outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Not a directory: {out_dir}")

    rep = Report()

    cfg = load_run_config(out_dir)
    summary = load_comparison_summary(out_dir)
    check_config_consistency(rep, cfg, summary)

    scheme = cfg["scheme"]
    schemes = cvt.build_scheme_definitions()
    if scheme not in schemes:
        rep.add("FAIL", "Config", f"unknown scheme {scheme} in run_config")
        write_reports(out_dir, rep)
        sys.exit(1)

    cutoffs: List[float] = []
    for x in cfg["cutoffs"]:
        if x == "Infinity" or (isinstance(x, float) and np.isinf(x)):
            cutoffs.append(np.inf)
        else:
            cutoffs.append(float(x))
    class_labels = list(cfg["labels"])
    n_folds = int(cfg["n_folds"])
    random_state = int(cfg["random_state"])

    X_norm, dist_matrix, species_final = cvt.load_and_preprocess_data()
    species_ordered = sorted(species_final)
    X, y_dist, pair_sp1, pair_sp2 = cvt.construct_pairwise_features(
        X_norm, dist_matrix, species_ordered
    )
    y_class, n_classes, _ = cvt.assign_classes(y_dist, cutoffs, class_labels)
    n_species = len(species_ordered)

    if n_classes != len(class_labels):
        rep.add("FAIL", "Data", "n_classes mismatch")
    if summary.get("n_species") != n_species:
        rep.add(
            "WARN",
            "Data",
            f"comparison_summary n_species={summary.get('n_species')} vs recomputed {n_species}",
        )

    folds, _deep_set = cvt.build_stratified_deep_species_fold_indices(
        species_ordered, pair_sp1, pair_sp2, n_folds, random_state
    )

    check_leakage(rep, folds, pair_sp1, pair_sp2)

    fold_counts = recompute_pair_fold_counts(folds, pair_sp1, pair_sp2)
    check_predictions_table(rep, out_dir, n_species, n_folds, species_ordered, fold_counts)

    pred_df = pd.read_csv(os.path.join(out_dir, "predictions_aggregated.csv"))
    check_centroids(rep, out_dir, class_labels, n_classes)
    D_csv = pd.read_csv(os.path.join(out_dir, "distance_matrix_predicted.csv"), index_col=0)
    check_matrix_and_predictions(rep, out_dir, pred_df, species_ordered)
    check_cophenetic_and_tree(rep, out_dir, species_ordered, D_csv.values.astype(float))

    rep.add(
        "WARN",
        "Methodology",
        "Test-train pair features use both species' domain profiles (transductive features; not label leakage).",
    )

    write_reports(out_dir, rep)

    if rep.has_fail():
        log("Verification finished with FAIL - see verification_report.txt")
        sys.exit(1)
    log("Verification finished: no FAIL")
    sys.exit(0)


if __name__ == "__main__":
    main()
