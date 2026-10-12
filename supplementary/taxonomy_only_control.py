#!/usr/bin/env python3
"""
Taxonomy-only control for the pairwise divergence-class classifier.

TimeTree supplies both the classifier's target and the evaluation reference, which
leaves an alternative reading of a high accuracy open: the model may recover little more
than broad clade identity, and clade identity by itself largely determines a pair's
TimeTree divergence class. This script tests that alternative directly, by predicting
the same divergence classes from taxonomic information alone and comparing the result
against the domain-content model.

Every component except the feature matrix is imported from
scripts/classification_cv_tree_reconstruction.py, so the arms are strictly like for
like: the same SCHEME_4B cutoffs, the same species-blocked clade-stratified 5-fold
split, the same StandardScaler + RandomOverSampler + RandomForestClassifier (200 trees,
per-fold seed 42 + fold_i), the same test-test union test-train prediction set, the same
mean-predict_proba-then-argmax aggregation, the same train-fold-only class->MY
centroids, the same UPGMA builder and the same ete3 unrooted Robinson-Foulds. Only the
feature matrix changes.

Every encoding is carried all the way through to a reconstructed tree and scored by
normalized Robinson-Foulds rather than stopped at classification metrics, because
classification accuracy and topological recovery come apart sharply on this data: a few
distinct class centroids leave most pairs exactly tied, so good class assignment does
not imply a resolvable tree.

Feature encodings (all symmetric in the two species of a pair):
  DOMAIN    7,976 |dC| features on per-species relative frequencies (the main analysis)
  T1        1 binary feature: are both species in the same clade?
  T2        one-hot over the unordered clade-pair label, fit on training pairs only,
            with handle_unknown='ignore' so an unseen cell is an honest all-zero row
  T3        9-dim clade count vector (entry c = how many of the two species are in clade c)
  T4        2 binary features: same clade, and same superorder
  COMBINED  DOMAIN concatenated with T2, asking whether domain content adds anything
            beyond taxonomy

Each encoding is compared against DOMAIN with an exact McNemar test on the 5,565 paired
aggregated predictions, and with a species-block bootstrap confidence interval for the
accuracy and kappa differences. Pairs that share a species are not independent, so a
McNemar P alone would overstate significance; resampling species rather than pairs
handles that pseudoreplication, and is the honest uncertainty statement.

Writes only under results/taxonomy_only_control/: a Newick tree and a float64 distance
matrix per encoding, the aggregated per-pair predictions per encoding,
encoding_comparison.csv (one row per encoding) and summary.json (full metrics plus the
head-to-head comparisons). Nothing elsewhere in the repository is modified.
"""
import json
import os
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.contingency_tables import mcnemar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))  # make the main pipeline importable

import classification_cv_tree_reconstruction as cv  # noqa: E402
from elastic_net_regression_stratified_species_cv import build_clade_mapping  # noqa: E402

OUT = os.path.join(REPO, "results", "taxonomy_only_control")
SEED = cv.RANDOM_STATE
N_FOLDS = cv.N_FOLDS_DEFAULT
N_TREES = cv.N_ESTIMATORS_DEFAULT
N_BOOT = 2000

# Superorder grouping, used only by the T4 encoding.
SUPERORDER = {
    "Primates": "Euarchontoglires",
    "Glires": "Euarchontoglires",
    "Laurasiatheria_Carnivora": "Laurasiatheria",
    "Laurasiatheria_Cetartiodactyla": "Laurasiatheria",
    "Laurasiatheria_Chiroptera": "Laurasiatheria",
    "Laurasiatheria_Perissodactyla": "Laurasiatheria",
    "Laurasiatheria_Eulipotyphla": "Laurasiatheria",
    "Afrotheria": "Afrotheria",
    "Marsupials": "Marsupials",
}


def species_to_clade() -> Dict[str, str]:
    """Canonical clade map, keyed by space-separated species name."""
    mapping = build_clade_mapping()
    out: Dict[str, str] = {}
    for clade, members in mapping.items():
        if isinstance(members, (list, set, tuple)):
            for sp in members:
                out[sp.replace("_", " ")] = clade
        else:
            out[clade.replace("_", " ")] = members
    return out


def build_encoders(pair_sp1: np.ndarray, pair_sp2: np.ndarray,
                   sp2clade: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Static (fold-independent) taxonomy feature matrices."""
    c1 = np.array([sp2clade[s] for s in pair_sp1])
    c2 = np.array([sp2clade[s] for s in pair_sp2])
    clades = sorted(set(c1) | set(c2))
    idx = {c: i for i, c in enumerate(clades)}

    same = (c1 == c2).astype(float).reshape(-1, 1)

    counts = np.zeros((len(c1), len(clades)), dtype=float)
    for r, (a, b) in enumerate(zip(c1, c2)):
        counts[r, idx[a]] += 1.0
        counts[r, idx[b]] += 1.0

    s1 = np.array([SUPERORDER[c] for c in c1])
    s2 = np.array([SUPERORDER[c] for c in c2])
    hier = np.column_stack([same.ravel(), (s1 == s2).astype(float)])

    labels = np.array([f"{min(a, b)}|{max(a, b)}" for a, b in zip(c1, c2)]).reshape(-1, 1)
    return {"T1": same, "T3": counts, "T4": hier, "_pairlabels": labels, "_clades": np.array(clades)}


def run_encoding(name: str,
                 feature_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                 folds, y_class, y_dist, pair_sp1, pair_sp2, n_classes,
                 species_ordered, class_labels) -> Dict:
    """Run the cross-validation pipeline with a swapped feature matrix, then build the tree."""
    cv.log("=" * 72)
    cv.log(f"ENCODING: {name}")
    cv.log("=" * 72)
    proba_sum = defaultdict(lambda: np.zeros(n_classes, dtype=float))
    proba_count = defaultdict(int)
    fold_means: List[np.ndarray] = []

    for fi, (tr_idx, te_idx, _train_sp, _test_sp) in enumerate(folds, start=1):
        means_f = cv.compute_fold_train_class_means(tr_idx, y_class, y_dist, n_classes)
        fold_means.append(means_f)
        X_train, X_test = feature_fn(tr_idx, te_idx)
        # Per-fold model seed, as in classification_cv_tree_reconstruction.py: the forest
        # is seeded with random_state + fold_i, not a single fixed seed. A fixed seed here
        # would give the DOMAIN arm a different fit from the main analysis, so it would no
        # longer reproduce it and the other arms would not be compared like for like.
        res = cv.fit_fold_random_oversample_rf(
            X_train, y_class[tr_idx], X_test, y_class[te_idx],
            n_classes, SEED + fi, N_TREES,
        )
        if res is None:
            cv.log(f"  fold {fi}: INVALID")
            continue
        proba_full, _ = res
        for row_i, gi in enumerate(te_idx):
            key = cv.canonical_pair_key(pair_sp1[gi], pair_sp2[gi])
            proba_sum[key] += proba_full[row_i]
            proba_count[key] += 1

    # class -> MY from train-fold means only, so no test label can inform the mapping
    stack = np.vstack(fold_means)
    mu_k = np.zeros(n_classes)
    for k in range(n_classes):
        col = stack[:, k]
        valid = ~np.isnan(col)
        mu_k[k] = float(np.mean(col[valid])) if valid.any() else np.nan

    rows = []
    for gi in range(len(y_class)):
        key = cv.canonical_pair_key(pair_sp1[gi], pair_sp2[gi])
        if proba_count[key] == 0:
            continue
        mean_p = proba_sum[key] / float(proba_count[key])
        rows.append({
            # cast to plain str: pair_sp1/pair_sp2 are numpy arrays, and ete3's node
            # lookup rejects numpy.str_ when pruning the reference tree.
            "sp1": str(key[0]), "sp2": str(key[1]),
            "y_true": int(y_class[gi]), "y_pred": int(np.argmax(mean_p)),
            "true_my": float(y_dist[gi]),
        })
    pred = pd.DataFrame(rows).drop_duplicates(subset=["sp1", "sp2"]).reset_index(drop=True)
    pred["predicted_distance_MY"] = mu_k[pred.y_pred.values]

    yt, yp = pred.y_true.values, pred.y_pred.values
    lab = list(range(n_classes))
    metrics = {
        "n_pairs": int(len(pred)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "recall_per_class": [float(x) for x in recall_score(yt, yp, labels=lab, average=None, zero_division=0)],
        "precision_per_class": [float(x) for x in precision_score(yt, yp, labels=lab, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(yt, yp, labels=lab).tolist(),
        "class_centroids_MY": [None if np.isnan(v) else float(v) for v in mu_k],
    }

    # ---- reconstruction: turn the predicted distances into a tree and score it ----
    sp = sorted(set(pred.sp1) | set(pred.sp2))
    n = len(sp)
    pos = {s: i for i, s in enumerate(sp)}
    D = np.zeros((n, n))
    for r in pred.itertuples():
        D[pos[r.sp1], pos[r.sp2]] = D[pos[r.sp2], pos[r.sp1]] = r.predicted_distance_MY
    Ddf = pd.DataFrame(D, index=sp, columns=sp)
    iu = np.triu_indices(n, k=1)
    Tmat = np.zeros((n, n))
    for r in pred.itertuples():
        Tmat[pos[r.sp1], pos[r.sp2]] = Tmat[pos[r.sp2], pos[r.sp1]] = r.true_my
    r_p, _ = pearsonr(Tmat[iu], D[iu])
    r_s, _ = spearmanr(Tmat[iu], D[iu])

    newick = cv.build_upgma_newick(Ddf)
    from ete3 import Tree
    tpred = Tree(newick, format=1)
    tref = cv.load_and_prune_reference_tree(sp)
    tref_fmt = Tree(cv.reference_tree_to_formatted_newick(tref), format=1)
    rf = tref_fmt.robinson_foulds(tpred, unrooted_trees=True)
    metrics.update({
        "pearson_true_vs_pred": float(r_p),
        "spearman_true_vs_pred": float(r_s),
        "robinson_foulds_distance": int(rf[0]),
        "robinson_foulds_max": int(rf[1]),
        "robinson_foulds_normalized": float(rf[0] / rf[1]) if rf[1] else float("nan"),
    })
    with open(os.path.join(OUT, f"tree_{name}.nwk"), "w") as fh:
        fh.write(newick + "\n")
    # Archive the float64 matrix, not just the tree. These matrices take only a handful of
    # distinct values, so thousands of pairs are exactly tied, and UPGMA's merge order, with
    # it the topology and the Robinson-Foulds score, turns on ties that a decimal round-trip
    # silently breaks. A CSV is therefore not sufficient to rebuild the tree; see
    # supplementary/verify_tree_reproducibility.py.
    np.save(os.path.join(OUT, f"distance_matrix_{name}.npy"), Ddf.values)
    cv.log(f"  {name}: acc={metrics['accuracy']:.4f} bal={metrics['balanced_accuracy']:.4f} "
           f"kappa={metrics['cohen_kappa']:.4f} RFnorm={metrics['robinson_foulds_normalized']:.4f}")
    return {"metrics": metrics, "pred": pred}


def species_block_bootstrap(pred_a: pd.DataFrame, pred_b: pd.DataFrame,
                            species: List[str], n_boot: int, seed: int) -> Dict:
    """Resample species, not pairs, to respect non-independence; CI of the A-B difference."""
    rng = np.random.default_rng(seed)
    a = pred_a.set_index(["sp1", "sp2"]).sort_index()
    b = pred_b.set_index(["sp1", "sp2"]).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    sp_arr = np.array(species)
    d_acc, d_kap = [], []
    for _ in range(n_boot):
        keep = set(rng.choice(sp_arr, size=len(sp_arr), replace=True))
        m = [(s1 in keep) and (s2 in keep) for s1, s2 in common]
        if not any(m):
            continue
        aa, bb = a[m], b[m]
        if aa.y_true.nunique() < 2:
            continue
        d_acc.append(accuracy_score(aa.y_true, aa.y_pred) - accuracy_score(bb.y_true, bb.y_pred))
        d_kap.append(cohen_kappa_score(aa.y_true, aa.y_pred) - cohen_kappa_score(bb.y_true, bb.y_pred))
    return {
        "n_boot_effective": len(d_acc),
        "delta_accuracy_mean": float(np.mean(d_acc)),
        "delta_accuracy_ci95": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
        "delta_kappa_mean": float(np.mean(d_kap)),
        "delta_kappa_ci95": [float(np.percentile(d_kap, 2.5)), float(np.percentile(d_kap, 97.5))],
    }


def main() -> None:
    cv.ensure_dir(OUT)
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    species_ordered = sorted(species)
    X_dom, y_dist, pair_sp1, pair_sp2 = cv.construct_pairwise_features(
        X_norm, dist_matrix, species_ordered)

    scheme = cv.build_scheme_definitions()["SCHEME_4B"]
    y_class, n_classes, _ = cv.assign_classes(y_dist, scheme["cutoffs"], scheme["labels"])
    class_labels = scheme["labels"]
    cv.log(f"pairs={len(y_dist)} species={len(species_ordered)} domains={X_dom.shape[1]} "
           f"classes={n_classes} cutoffs={scheme['cutoffs']}")

    folds, _deep = cv.build_stratified_deep_species_fold_indices(
        species_ordered, pair_sp1, pair_sp2, N_FOLDS, SEED)

    sp2clade = species_to_clade()
    enc = build_encoders(pair_sp1, pair_sp2, sp2clade)

    def static(mat):
        return lambda tr, te: (mat[tr], mat[te])

    def t2(tr, te):
        """One-hot over clade-pair identity, fit on the training pairs of the fold only."""
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return ohe.fit_transform(enc["_pairlabels"][tr]), ohe.transform(enc["_pairlabels"][te])

    def combined(tr, te):
        a, b = t2(tr, te)
        return np.hstack([X_dom[tr], a]), np.hstack([X_dom[te], b])

    # NOISE + T2 is the control that decides how COMBINED must be interpreted. COMBINED
    # (domains + T2) scores below T2 alone, which invites the reading that domain content is
    # noise once clade identity is known. This arm is what rules that reading out: the domain
    # block is replaced by Gaussian noise of identical shape, with the T2 block and every
    # other step left untouched, so if the domain features really were noise this arm should
    # score like COMBINED. It does not; it collapses. The COMBINED deficit is instead a
    # feature-subsampling dilution artifact, since max_features='sqrt' draws about 89 of some
    # 8,065 columns per split, so most splits never see one of the roughly 44 informative
    # one-hot columns.
    rng_noise = np.random.default_rng(SEED)
    X_noise = rng_noise.standard_normal(X_dom.shape)

    def noise_plus_t2(tr, te):
        a, b = t2(tr, te)
        return np.hstack([X_noise[tr], a]), np.hstack([X_noise[te], b])

    runs = {
        "DOMAIN": static(X_dom),
        "T1_same_clade": static(enc["T1"]),
        "T2_clade_pair_onehot": t2,
        "T3_clade_counts": static(enc["T3"]),
        "T4_hierarchical": static(enc["T4"]),
        "COMBINED_domain_plus_T2": combined,
        "NOISE_plus_T2_control": noise_plus_t2,
    }

    results = {}
    for name, fn in runs.items():
        results[name] = run_encoding(name, fn, folds, y_class, y_dist,
                                     pair_sp1, pair_sp2, n_classes,
                                     species_ordered, class_labels)

    # ---- head-to-head vs DOMAIN ----
    base = results["DOMAIN"]["pred"].set_index(["sp1", "sp2"]).sort_index()
    comps = {}
    for name in runs:
        if name == "DOMAIN":
            continue
        other = results[name]["pred"].set_index(["sp1", "sp2"]).sort_index()
        common = base.index.intersection(other.index)
        bc = (base.loc[common].y_pred == base.loc[common].y_true).values
        oc = (other.loc[common].y_pred == other.loc[common].y_true).values
        tbl = [[int((bc & oc).sum()), int((bc & ~oc).sum())],
               [int((~bc & oc).sum()), int((~bc & ~oc).sum())]]
        res = mcnemar(tbl, exact=True)
        comps[name] = {
            "mcnemar_table_[[both_right,only_DOMAIN],[only_other,both_wrong]]": tbl,
            "mcnemar_p_value": float(res.pvalue),
            "mcnemar_caveat": ("Pairs sharing a species are not independent; this P is "
                               "anti-conservative. The species-block bootstrap below is the "
                               "honest uncertainty statement."),
            "species_block_bootstrap_other_minus_DOMAIN": species_block_bootstrap(
                results[name]["pred"], results["DOMAIN"]["pred"],
                species_ordered, N_BOOT, SEED),
        }

    summary = {
        "protocol": {
            "source": "scripts/classification_cv_tree_reconstruction.py (imported, unmodified)",
            "scheme": "SCHEME_4B", "cutoffs": scheme["cutoffs"],
            "n_folds": N_FOLDS, "n_estimators": N_TREES, "random_state": SEED,
            "n_species": len(species_ordered), "n_pairs": int(len(y_dist)),
            "n_domain_features": int(X_dom.shape[1]),
            "note": ("Only the feature matrix differs between encodings; folds, scaler, "
                     "oversampling, model, aggregation, class->MY mapping, UPGMA and RF "
                     "are identical to the published run."),
        },
        "encodings": {k: v["metrics"] for k, v in results.items()},
        "head_to_head_vs_DOMAIN": comps,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    tbl = pd.DataFrame([
        {"encoding": k, **{m: v["metrics"][m] for m in
                           ("n_pairs", "accuracy", "balanced_accuracy", "cohen_kappa",
                            "pearson_true_vs_pred", "spearman_true_vs_pred",
                            "robinson_foulds_normalized")}}
        for k, v in results.items()])
    tbl.to_csv(os.path.join(OUT, "encoding_comparison.csv"), index=False)
    for k, v in results.items():
        v["pred"].to_csv(os.path.join(OUT, f"predictions_{k}.csv"), index=False)

    cv.log("")
    cv.log(tbl.to_string(index=False))
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
