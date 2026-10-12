#!/usr/bin/env python3
"""
How much of the reconstructed tree depends on the TimeTree scale of the predicted distances?

The classification pipeline predicts an ordinal divergence-class for each species pair
and then converts that class back into a distance in millions of years, using the mean
true patristic distance of the training pairs assigned to that class. The conversion
therefore borrows its numerical scale from the TimeTree reference: the predicted matrix
is not a function of domain content alone. This script quantifies what that borrowing
buys, by asking whether it changes the inferred topology or only sets the branch lengths.

Five class-to-distance definitions ("arms") are built from a single cross-validation
run, so the fitted model is identical across arms and only the mapping differs:

  CENTROID_ARGMAX    argmax class -> mean true patristic distance (MY) of that class's
                     training-training pairs, averaged over folds. This is the mapping
                     the main pipeline uses, and it is on the TimeTree scale.
  CENTROID_EXPECTED  sum_k p_k * mu_k. Continuous, still on the TimeTree MY scale.
  ORDINAL_ARGMAX     argmax class index + 1, i.e. the integers 1, 2, 3, 4. No TimeTree
                     quantity enters the distance at all.
  ORDINAL_EXPECTED   sum_k p_k * (k+1). Continuous and free of the TimeTree scale; the
                     cleanest domain-only distance this model can supply.
  STRICT_CENTROID    argmax class -> the class mean of the scoring fold's own training
                     pairs, with no averaging across folds. Averaging mu_k over all
                     folds lets a pair's own true distance contribute to the mean that
                     scores it; taking mu_k from the scoring fold alone closes that path,
                     so each test distance rests only on training information.

A rank-preserving rescaling of a distance matrix does not in general preserve a UPGMA
topology, because the merge order depends on averages rather than on rank order alone.
Whether the ordinal and centroid trees agree is therefore an empirical question rather
than an algebraic one, and the arm-versus-arm Robinson-Foulds comparison at the end of
the run answers it directly.

For every arm the script builds UPGMA, neighbor-joining and balanced minimum-evolution
trees, scores each against the TimeTree reference pruned to the cohort, and records the
rank correlation with the true patristic distances, the number of distinct distance
values, and a Robinson-Foulds band obtained by re-clustering the matrix under 200
negligible random perturbations. That band bounds how much of the topology is decided by
exact ties rather than by signal, which matters most for the ordinal arms, where many
pairs share one of only four values.

Writes only under results/ordinal_trees/: one Newick per arm and tree method, the
per-pair class probabilities and all five distances
(pair_probabilities_and_distances.csv), a tidy arm x tree-method table
(ordinal_tree_comparison.csv), and summary.json with the class centroids, the per-arm
scores and the between-arm topology agreement.
"""
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import classification_cv_tree_reconstruction as cv  # noqa: E402
import continuous_distance_trees as ct  # noqa: E402  (reuse tree scoring + FastME)

OUT = os.path.join(REPO, "results", "ordinal_trees")
SEED = cv.RANDOM_STATE
N_FOLDS = cv.N_FOLDS_DEFAULT
N_TREES = cv.N_ESTIMATORS_DEFAULT
N_JITTER = 200
JITTER_SD = 1e-10


def main() -> None:
    cv.ensure_dir(OUT)
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    species_ordered = sorted(species)
    X, y_dist, pair_sp1, pair_sp2 = cv.construct_pairwise_features(
        X_norm, dist_matrix, species_ordered)
    scheme = cv.build_scheme_definitions()["SCHEME_4B"]
    y_class, n_classes, _ = cv.assign_classes(y_dist, scheme["cutoffs"], scheme["labels"])
    folds, _ = cv.build_stratified_deep_species_fold_indices(
        species_ordered, pair_sp1, pair_sp2, N_FOLDS, SEED)
    sp2clade = ct.species_to_clade()

    proba_sum = defaultdict(lambda: np.zeros(n_classes)); proba_cnt = defaultdict(int)
    strict_sum = defaultdict(float); strict_cnt = defaultdict(int)
    fold_means: List[np.ndarray] = []

    for fold_i, (tr_idx, te_idx, _a, _b) in enumerate(folds, start=1):
        means_f = cv.compute_fold_train_class_means(tr_idx, y_class, y_dist, n_classes)
        fold_means.append(means_f)
        res = cv.fit_fold_random_oversample_rf(X[tr_idx], y_class[tr_idx], X[te_idx],
                                               y_class[te_idx], n_classes,
                                               SEED + fold_i, N_TREES)
        if res is None:
            continue
        proba_full, _ = res
        for row_i, gi in enumerate(te_idx):
            k = cv.canonical_pair_key(pair_sp1[gi], pair_sp2[gi])
            proba_sum[k] += proba_full[row_i]; proba_cnt[k] += 1
            # Strict arm: score this test row with only this fold's training class means
            c = int(np.argmax(proba_full[row_i]))
            if not np.isnan(means_f[c]):
                strict_sum[k] += float(means_f[c]); strict_cnt[k] += 1

    stack = np.vstack(fold_means)
    mu_k = np.array([float(np.nanmean(stack[:, k])) for k in range(n_classes)])
    cv.log(f"class centroids (cross-fold mean): {np.round(mu_k, 3).tolist()}")
    cv.log(f"per-fold centroids:\n{np.round(stack, 2)}")

    keys = [k for k in proba_cnt if proba_cnt[k] > 0]
    sp_used = sorted({str(s) for k in keys for s in k})
    true_my = {cv.canonical_pair_key(pair_sp1[g], pair_sp2[g]): float(y_dist[g])
               for g in range(len(y_dist))}

    P = {k: proba_sum[k] / proba_cnt[k] for k in keys}
    arms = {
        "CENTROID_ARGMAX":   {k: float(mu_k[int(np.argmax(P[k]))]) for k in keys},
        "CENTROID_EXPECTED": {k: float(np.dot(P[k], mu_k)) for k in keys},
        "ORDINAL_ARGMAX":    {k: float(int(np.argmax(P[k])) + 1) for k in keys},
        "ORDINAL_EXPECTED":  {k: float(np.dot(P[k], np.arange(n_classes) + 1.0)) for k in keys},
        "STRICT_CENTROID":   {k: (strict_sum[k] / strict_cnt[k]) if strict_cnt[k] else
                              float(mu_k[int(np.argmax(P[k]))]) for k in keys},
    }
    arms = {a: {(str(x), str(y)): v for (x, y), v in d.items()} for a, d in arms.items()}
    true_sub = {(str(x), str(y)): true_my[k] for k in keys for x, y in [k]}

    # Save the per-pair class probabilities and all five distances, so downstream
    # analyses can reuse this cross-validation run without refitting the model.
    pd.DataFrame([{"sp1": a, "sp2": b, "true_my": true_sub[(a, b)],
                   **{f"p{c}": P[(a, b)][c] for c in range(n_classes)},
                   **{arm: arms[arm][(a, b)] for arm in arms}}
                  for a, b in sorted(true_sub)]).to_csv(
        os.path.join(OUT, "pair_probabilities_and_distances.csv"), index=False)

    from ete3 import Tree
    tref = Tree(cv.reference_tree_to_formatted_newick(
        cv.load_and_prune_reference_tree(sp_used)), format=1)
    n = len(sp_used); pos = {s: i for i, s in enumerate(sp_used)}
    T = np.zeros((n, n))
    for (a, b), v in true_sub.items():
        T[pos[a], pos[b]] = T[pos[b], pos[a]] = v
    iu = np.triu_indices(n, k=1)
    labels = [cv.format_species_label_for_newick(s) for s in sp_used]

    mats, rows, results = {}, [], []
    rng = np.random.default_rng(SEED)
    for name, d in arms.items():
        D = np.zeros((n, n))
        for (a, b), v in d.items():
            D[pos[a], pos[b]] = D[pos[b], pos[a]] = v
        mats[name] = D
        r = {"arm": name,
             "spearman_vs_true": float(spearmanr(T[iu], D[iu]).statistic),
             "kendall_vs_true": float(kendalltau(T[iu], D[iu]).statistic),
             "n_distinct_values": int(np.unique(np.round(D[iu], 9)).size),
             "predicted_range": [float(D[iu].min()), float(D[iu].max())],
             "uses_timetree_scale": name.startswith("CENTROID") or name == "STRICT_CENTROID"}
        nj = ct.nj_newick(D, labels)
        me, me_src = ct.fastme_me_newick(D, labels, name)
        trees = {"UPGMA": cv.build_upgma_newick(pd.DataFrame(D, index=sp_used, columns=sp_used)),
                 "NJ": nj}
        if me:
            trees["ME"] = me
        for meth, nwk in trees.items():
            r[meth] = ct.score_tree(nwk, sp_used, sp2clade, tref)
            with open(os.path.join(OUT, f"tree_{name}_{meth}.nwk"), "w") as fh:
                fh.write(nwk + "\n")
        vals = []
        for _ in range(N_JITTER):
            E = rng.normal(0, JITTER_SD * max(D[iu].max(), 1.0), (n, n))
            E = np.triu(E, 1); E = E + E.T
            vals.append(ct.rf_of_matrix(D + E, sp_used, tref)[0])
        r["tiebreak_rf_min"] = float(np.min(vals)); r["tiebreak_rf_max"] = float(np.max(vals))
        r["tiebreak_rf_sd"] = float(np.std(vals))
        results.append(r)
        cv.log(f"  {name:18} rho={r['spearman_vs_true']:.4f} distinct={r['n_distinct_values']:5} "
               f"UPGMA RF={r['UPGMA']['robinson_foulds_normalized']:.4f} "
               f"(band {r['tiebreak_rf_min']:.4f}-{r['tiebreak_rf_max']:.4f}) "
               f"mono_unrooted={r['UPGMA']['n_monophyletic_unrooted']}/{r['UPGMA']['n_groups']} "
               f"(rooted={r['UPGMA']['n_monophyletic_rooted']}/{r['UPGMA']['n_groups']}, "
               f"arbitrary root -- no outgroup in cohort)")
        for meth in trees:
            rows.append({"arm": name, "tree_method": meth,
                         "uses_timetree_scale": r["uses_timetree_scale"],
                         "spearman_vs_true": r["spearman_vs_true"],
                         "n_distinct_values": r["n_distinct_values"],
                         "rf_normalized": r[meth]["robinson_foulds_normalized"],
                         # The unrooted count is the clade-recovery statistic to read:
                         # the cohort excludes the platypus, so there is no outgroup and
                         # the root of every predicted tree is an artifact of the last
                         # merge. The rooted count penalizes correct bipartitions for
                         # that arbitrary root placement, so it is reported alongside
                         # rather than instead of the unrooted count.
                         "mono_unrooted": r[meth]["n_monophyletic_unrooted"],
                         "mono_rooted": r[meth]["n_monophyletic_rooted"],
                         "n_groups": r[meth]["n_groups"]})

    # ---- the decisive comparison: does dropping the TimeTree scale change the tree? ----
    cv.log("\n  topology agreement between arms (RF between their UPGMA trees):")
    pairs_to_check = [("CENTROID_ARGMAX", "ORDINAL_ARGMAX"),
                      ("CENTROID_EXPECTED", "ORDINAL_EXPECTED"),
                      ("CENTROID_ARGMAX", "STRICT_CENTROID")]
    agree = {}
    for a, b in pairs_to_check:
        ta = Tree(open(os.path.join(OUT, f"tree_{a}_UPGMA.nwk")).read().strip(), format=1)
        tb = Tree(open(os.path.join(OUT, f"tree_{b}_UPGMA.nwk")).read().strip(), format=1)
        rf = ta.robinson_foulds(tb, unrooted_trees=True)
        ident = bool(rf[0] == 0)
        agree[f"{a}_vs_{b}"] = {"rf": int(rf[0]), "rf_max": int(rf[1]), "identical": ident}
        cv.log(f"    {a:18} vs {b:18}: RF={rf[0]}/{rf[1]}  "
               f"{'IDENTICAL TOPOLOGY' if ident else 'different'}")
        # Rank agreement between the two distance matrices themselves: a high value with
        # a non-zero Robinson-Foulds distance means the arms order the pairs the same way
        # but still cluster them differently.
        sa, sb = mats[a][iu], mats[b][iu]
        agree[f"{a}_vs_{b}"]["spearman_between_matrices"] = float(spearmanr(sa, sb).statistic)

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"class_centroids_MY": mu_k.tolist(),
                   "per_fold_centroids_MY": stack.tolist(),
                   "n_species": n, "n_pairs": len(keys),
                   "arms": results, "topology_agreement": agree}, fh, indent=2)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT, "ordinal_tree_comparison.csv"), index=False)
    cv.log("")
    cv.log(tbl.to_string(index=False))
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
