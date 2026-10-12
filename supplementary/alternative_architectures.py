#!/usr/bin/env python3
"""
Benchmark alternative classifier architectures against the random forest used for
pairwise divergence-class prediction, holding everything except the estimator fixed.

Data and task: the 107-species cohort (5,671 unordered species pairs), with |dC|
relative-frequency features over 8,020 variance-filtered Pfam domains as input and
the four ordinal divergence-distance classes of SCHEME_4B (manual cutoffs at 150,
175 and 190 MY) as the target.

Every architecture is run through one identical pipeline: the same species-blocked,
clade-stratified folds (fold seed 42, per-fold model seed 42 + fold index), the same
StandardScaler and RandomOverSampler fit on training pairs only, the same test-test
union test-train prediction set, and the same aggregation per unordered pair (mean
predicted probability across the folds that cover the pair, then argmax). Only the
estimator changes, so any difference between rows of the output table is attributable
to the architecture and nothing else. Species-blocking is what keeps the comparison
honest: no species in a test pair appears in any training pair of that fold.

Estimators compared: random forest, histogram gradient boosting, linear SVC,
multinomial logistic regression, a two-hidden-layer MLP, XGBoost when it is installed,
and a majority-class dummy as the floor. Margin-only models (linear SVC) get a
softmax of the decision function so that probability aggregation works for them too.

Each model is also carried through to a tree by the same route as the main analysis
(predicted class -> mean patristic distance of that class's training pairs, averaged
across folds -> UPGMA), because a classifier that scores better on the confusion
matrix but yields a worse tree is not an improvement. The class-to-distance mapping
is derived from training pairs only, never from test labels.

Non-machine-learning comparators are not duplicated here: simple domain distances
(presence/absence, Jaccard, Bray-Curtis, Euclidean, CLR) are in
supplementary/baseline_comparison_unified.py, and a taxonomy-only lookup, the
strongest such comparator, is in supplementary/taxonomy_only_control.py.

Writes under results/alternative_architectures/:
  - architecture_comparison.csv: one row per model, sorted by Cohen's kappa. Accuracy,
    balanced accuracy, kappa, per-class recall and precision, total fit seconds, and
    tree scores. rf_normalized_UPGMA is the unrooted Robinson-Foulds distance divided
    by its maximum, so 0 means the predicted topology matches the reference and 1
    means it shares no internal split with it. clades_monophyletic_rooted counts how
    many of the n_groups clade groups are recovered as rooted monophyletic.
  - predictions_<model>.csv: true and predicted class for every species pair.
  - tree_<model>.nwk: the UPGMA tree built from that model's predicted classes.
  - summary.json: the same metrics, the class centroids in MY, and an exact McNemar
    test of each architecture against the random forest on their shared pairs. Those
    P values are anti-conservative, because species pairs share species and so are
    not independent observations.
"""
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from statsmodels.stats.contingency_tables import mcnemar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import classification_cv_tree_reconstruction as cv  # noqa: E402
import continuous_distance_trees as ct  # noqa: E402

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError as exc:
    raise SystemExit("needs imbalanced-learn") from exc

OUT = os.path.join(REPO, "results", "alternative_architectures")
SEED = cv.RANDOM_STATE
N_FOLDS = cv.N_FOLDS_DEFAULT
N_TREES = cv.N_ESTIMATORS_DEFAULT


def make_models(seed: int, n_classes: int) -> Dict[str, object]:
    m = {
        "RandomForest_published": RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, random_state=seed,
            class_weight=None, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=seed, max_iter=200),
        "LinearSVC": LinearSVC(random_state=seed, C=1.0, max_iter=5000, dual="auto"),
        # sklearn >= 1.7 dropped multi_class; multinomial is the default for >2 classes
        "LogisticRegression": LogisticRegression(
            random_state=seed, max_iter=2000, n_jobs=-1),
        "MLP_neural_net": MLPClassifier(
            hidden_layer_sizes=(128, 64), random_state=seed, max_iter=300,
            early_stopping=True, n_iter_no_change=10),
        "Majority_baseline": DummyClassifier(strategy="most_frequent"),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.3, random_state=seed, n_jobs=-1,
            objective="multi:softprob", num_class=n_classes, verbosity=0,
            tree_method="hist")
    except ImportError:
        pass
    return m


def proba_of(clf, X, n_classes):
    """Class-probability matrix with one column per class label.

    Columns are indexed by the integer class label, so a class that is absent from a
    fold's training set keeps a zero column instead of shifting the other columns out
    of alignment. Estimators without predict_proba (margin models such as linear SVC)
    get a softmax over the decision function, so that every architecture can be
    aggregated across folds by the same averaging rule.
    """
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        classes = clf.classes_
    else:
        d = clf.decision_function(X)
        if d.ndim == 1:
            d = np.column_stack([-d, d])
        e = np.exp(d - d.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        classes = clf.classes_
    full = np.zeros((p.shape[0], n_classes))
    for j, c in enumerate(classes):
        full[:, int(c)] = p[:, j]
    return full


def main() -> None:
    cv.ensure_dir(OUT)
    warnings.filterwarnings("ignore")
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    sp_ord = sorted(species)
    X, y_dist, pair_sp1, pair_sp2 = cv.construct_pairwise_features(X_norm, dist_matrix, sp_ord)
    scheme = cv.build_scheme_definitions()["SCHEME_4B"]
    y_class, n_classes, _ = cv.assign_classes(y_dist, scheme["cutoffs"], scheme["labels"])
    folds, _ = cv.build_stratified_deep_species_fold_indices(
        sp_ord, pair_sp1, pair_sp2, N_FOLDS, SEED)
    sp2clade = ct.species_to_clade()
    cv.log(f"{len(y_dist)} pairs, {X.shape[1]} features, {n_classes} classes, {N_FOLDS} folds")

    names = list(make_models(SEED, n_classes).keys())
    store = {m: {"proba_sum": defaultdict(lambda: np.zeros(n_classes)),
                 "proba_cnt": defaultdict(int), "secs": 0.0} for m in names}
    fold_means: List[np.ndarray] = []

    for fold_i, (tr, te, _a, _b) in enumerate(folds, start=1):
        cv.log(f"--- fold {fold_i}/{N_FOLDS} ---")
        fold_means.append(cv.compute_fold_train_class_means(tr, y_class, y_dist, n_classes))
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ros = RandomOverSampler(random_state=SEED + fold_i)
        Xrs, yrs = ros.fit_resample(Xtr, y_class[tr])
        for name, clf in make_models(SEED + fold_i, n_classes).items():
            t0 = time.time()
            try:
                clf.fit(Xrs, yrs)
                pf = proba_of(clf, Xte, n_classes)
            except Exception as e:
                cv.log(f"    {name}: FAILED ({type(e).__name__}: {str(e)[:80]})")
                continue
            store[name]["secs"] += time.time() - t0
            for row_i, gi in enumerate(te):
                k = cv.canonical_pair_key(pair_sp1[gi], pair_sp2[gi])
                store[name]["proba_sum"][k] += pf[row_i]
                store[name]["proba_cnt"][k] += 1
            cv.log(f"    {name:24} {time.time()-t0:6.1f}s")

    stack = np.vstack(fold_means)
    mu_k = np.array([float(np.nanmean(stack[:, k])) for k in range(n_classes)])
    true_my = {cv.canonical_pair_key(pair_sp1[g], pair_sp2[g]): float(y_dist[g])
               for g in range(len(y_dist))}
    true_cls = {cv.canonical_pair_key(pair_sp1[g], pair_sp2[g]): int(y_class[g])
                for g in range(len(y_dist))}

    from ete3 import Tree
    preds, rows = {}, []
    sp_used = None
    tref = None
    for name in names:
        cnt = store[name]["proba_cnt"]
        keys = [k for k in true_my if cnt[k] > 0]
        if not keys:
            cv.log(f"  {name}: no predictions, skipped")
            continue
        yt = np.array([true_cls[k] for k in keys])
        yp = np.array([int(np.argmax(store[name]["proba_sum"][k] / cnt[k])) for k in keys])
        preds[name] = pd.DataFrame({"sp1": [str(k[0]) for k in keys],
                                    "sp2": [str(k[1]) for k in keys],
                                    "y_true": yt, "y_pred": yp})
        if sp_used is None:
            sp_used = sorted({str(s) for k in keys for s in k})
            tref = Tree(cv.reference_tree_to_formatted_newick(
                cv.load_and_prune_reference_tree(sp_used)), format=1)
        lab = list(range(n_classes))
        n = len(sp_used); pos = {s: i for i, s in enumerate(sp_used)}
        M = np.zeros((n, n))
        for k, c in zip(keys, yp):
            M[pos[str(k[0])], pos[str(k[1])]] = M[pos[str(k[1])], pos[str(k[0])]] = mu_k[c]
        nwk = cv.build_upgma_newick(pd.DataFrame(M, index=sp_used, columns=sp_used))
        sc_tree = ct.score_tree(nwk, sp_used, sp2clade, tref)
        with open(os.path.join(OUT, f"tree_{name}.nwk"), "w") as fh:
            fh.write(nwk + "\n")
        rows.append({
            "model": name, "n_pairs": len(keys),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "cohen_kappa": float(cohen_kappa_score(yt, yp)),
            "recall_per_class": [round(float(x), 3) for x in
                                 recall_score(yt, yp, labels=lab, average=None, zero_division=0)],
            "precision_per_class": [round(float(x), 3) for x in
                                    precision_score(yt, yp, labels=lab, average=None, zero_division=0)],
            "rf_normalized_UPGMA": sc_tree["robinson_foulds_normalized"],
            "clades_monophyletic_rooted": sc_tree["n_monophyletic_rooted"],
            "n_groups": sc_tree["n_groups"],
            "fit_seconds_total": round(store[name]["secs"], 1),
        })
        cv.log(f"  {name:24} acc={rows[-1]['accuracy']:.4f} bal={rows[-1]['balanced_accuracy']:.4f} "
               f"kappa={rows[-1]['cohen_kappa']:.4f} RF={rows[-1]['rf_normalized_UPGMA']:.4f} "
               f"mono={rows[-1]['clades_monophyletic_rooted']}/{rows[-1]['n_groups']} "
               f"({rows[-1]['fit_seconds_total']}s)")

    # paired per-pair comparison of each architecture against the random forest,
    # restricted to the species pairs both models predicted
    base = preds["RandomForest_published"].set_index(["sp1", "sp2"]).sort_index()
    comp = {}
    for name, df in preds.items():
        if name == "RandomForest_published":
            continue
        o = df.set_index(["sp1", "sp2"]).sort_index()
        idx = base.index.intersection(o.index)
        bc = (base.loc[idx].y_pred == base.loc[idx].y_true).values
        oc = (o.loc[idx].y_pred == o.loc[idx].y_true).values
        tab = [[int((bc & oc).sum()), int((bc & ~oc).sum())],
               [int((~bc & oc).sum()), int((~bc & ~oc).sum())]]
        comp[name] = {"mcnemar_table": tab,
                      "mcnemar_p": float(mcnemar(tab, exact=True).pvalue),
                      "caveat": "pairs are non-independent, so this P is anti-conservative"}

    tbl = pd.DataFrame(rows).sort_values("cohen_kappa", ascending=False)
    tbl.to_csv(os.path.join(OUT, "architecture_comparison.csv"), index=False)
    for nm, df in preds.items():
        df.to_csv(os.path.join(OUT, f"predictions_{nm}.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"protocol": "identical to the published run; only the estimator changes",
                   "class_centroids_MY": mu_k.tolist(), "results": rows,
                   "head_to_head_vs_published_RF": comp}, fh, indent=2)
    cv.log("")
    cv.log(tbl[["model", "accuracy", "balanced_accuracy", "cohen_kappa",
                "rf_normalized_UPGMA", "clades_monophyletic_rooted",
                "fit_seconds_total"]].to_string(index=False))
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
