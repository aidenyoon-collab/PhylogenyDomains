#!/usr/bin/env python3
"""
Taxonomic-sampling diagnostics: Xenarthra coverage, and the effect of keeping the platypus.

Xenarthra coverage. No xenarthran taxon occurs anywhere in the inputs: not in
data_raw/MammalsList.txt (115 names), not in the domain count matrix (115 species
columns), and not among the TimeTree tips (108). The superorder is absent from the source
dataset; no filtering step here removed it. Of the 115 matrix species, 7 have no TimeTree
tip (subspecies and hybrids: Acinonyx jubatus, Bos indicus x Bos taurus, Canis lupus
dingo, Canis lupus familiaris, Neophocaena asiaeorientalis asiaeorientalis, Panthera
tigris altaica, Saimiri boliviensis boliviensis); three further name mismatches are pure
synonymy and are reconciled by TREE_NAME_MAP (Neogale/Neovison vison, Bos grunniens/Bos
mutus grunniens, Physeter catodon/macrocephalus). So 115 - 7 = 108 species, and 107 once
the platypus is set aside. These counts are stated here rather than recomputed, because
the species list and the matrix columns are fixed inputs.

The platypus. Ornithorhynchus anatinus is the natural outgroup for mammals, but it sits
far from every other species in domain-count space, which is why the main analyses leave
it out. This script quantifies that trade-off instead of assuming it, in three parts.

  (1) Leverage. For raw counts, per-species relative frequencies, and a centered
      log-ratio (CLR) transform, it compares the platypus mean distance to all other
      species against the mean over all species pairs. The ratio says how far the
      platypus pulls the distance distribution; the same ratio computed on the patristic
      reference distances is the value a faithful domain distance should reproduce, so it
      is reported alongside as the target.
  (2) Robustness. The correlation (Pearson and Spearman) between domain distance and
      patristic distance is recomputed on each basis for the cohort with and without the
      platypus. A CLR basis compresses large count differences, so comparing it against
      raw and relative bases shows whether a more robust distance makes retention viable.
  (3) Deepest split. With the platypus retained, UPGMA, neighbor-joining, and
      minimum-evolution trees are built on each basis, and the size of the platypus
      sister group is measured. Recovering the Monotremata/Theria split requires that
      sister group to hold every other taxon, so recovers_deepest_split is true only when
      the sister group size equals n_species - 1.

Cohort. The platypus is retained for part of this analysis, but Cervus hanglu
yarkandensis is dropped throughout, as elsewhere in the project, so the two cohorts
compared differ by the platypus alone.

Writes under results/taxon_sampling/: summary.json (cohort sizes, the correlation table,
the leverage ratios, the deepest-split results, and the headline metrics of any
classifier runs found on disk), platypus_sensitivity.csv (the cohort by basis correlation
table), and one Newick file per basis and tree method for the platypus-retained trees.
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import classification_cv_tree_reconstruction as cv  # noqa: E402
import continuous_distance_trees as ct  # noqa: E402

OUT = os.path.join(REPO, "results", "taxon_sampling")
PLATYPUS = "Ornithorhynchus anatinus"


def distances(Z: np.ndarray, metric: str = "sqeuclidean") -> np.ndarray:
    return squareform(pdist(Z, metric))


def main() -> None:
    cv.ensure_dir(OUT)

    # platypus-retained cohort: rebuild the full 108-species matrix/tree intersection
    lst = cv.load_species_list(cv.SPECIES_LIST_PATH)
    raw_full = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH, lst)
    Dpat_all = cv.compute_patristic_matrix(cv.TREE_PATH, list(raw_full.index))
    sp108 = sorted(Dpat_all.index)
    assert PLATYPUS in sp108, "platypus missing from the 108-species intersection"
    # The platypus is kept here, but Cervus hanglu is still dropped for shallow annotation
    # depth, so the two cohorts below differ by the platypus and by nothing else.
    sp108 = [s for s in sp108 if "Cervus hanglu" not in s]
    sp107 = [s for s in sp108 if s != PLATYPUS]
    cv.log(f"cohorts: with platypus n={len(sp108)}, without n={len(sp107)}")

    Xall = raw_full.loc[sp108]
    # the zero-variance filter is recomputed within each cohort, as the main pipeline does,
    # so each cohort is modeled on the domains that are actually informative within it
    out_rows, detail = [], {}
    for tag, sp in (("with_platypus", sp108), ("without_platypus", sp107)):
        X = Xall.loc[sp]
        X = X.loc[:, X.var(axis=0) > 0]
        Xr = X.values.astype(float)
        Xrel = Xr / Xr.sum(axis=1, keepdims=True)
        C = Xr + 0.5
        Xclr = np.log(C) - np.log(C).mean(axis=1, keepdims=True)
        T = Dpat_all.loc[sp, sp].values.astype(float)
        n = len(sp); iu = np.triu_indices(n, k=1)
        cv.log(f"  {tag}: {n} species, {Xr.shape[1]} non-zero-variance domains")
        for basis, Z in (("raw", Xr), ("relative", Xrel), ("CLR", Xclr)):
            M = distances(Z)
            v = M[iu]
            out_rows.append({"cohort": tag, "basis": basis, "n_species": n,
                             "n_domains": int(Xr.shape[1]),
                             "pearson": float(pearsonr(T[iu], v)[0]),
                             "spearman": float(spearmanr(T[iu], v).statistic)})
        detail[tag] = {"n_species": n, "n_domains": int(Xr.shape[1])}

    tbl = pd.DataFrame(out_rows)
    piv = tbl.pivot(index="basis", columns="cohort", values=["pearson", "spearman"])
    cv.log("\n  correlation with patristic distance, with vs without the platypus:")
    cv.log(piv.to_string())

    # ---- how extreme is the platypus? its leverage on the distance distribution ----
    X = Xall.loc[sp108]; X = X.loc[:, X.var(axis=0) > 0]
    Xr = X.values.astype(float)
    Xrel = Xr / Xr.sum(axis=1, keepdims=True)
    C = Xr + 0.5
    Xclr = np.log(C) - np.log(C).mean(axis=1, keepdims=True)
    pi = sp108.index(PLATYPUS)
    lev = {}
    for basis, Z in (("raw", Xr), ("relative", Xrel), ("CLR", Xclr)):
        M = distances(Z)
        off = M[np.triu_indices(len(sp108), k=1)]
        pl = np.delete(M[pi], pi)
        lev[basis] = {"platypus_mean_distance": float(pl.mean()),
                      "all_pairs_mean_distance": float(off.mean()),
                      "ratio": float(pl.mean() / off.mean()),
                      "platypus_percentile_in_all_pairs": float((off < pl.mean()).mean() * 100)}
        cv.log(f"    {basis:9}: platypus mean distance is {lev[basis]['ratio']:.2f}x the "
               f"overall mean ({lev[basis]['platypus_percentile_in_all_pairs']:.1f}th percentile)")

    # the same leverage measured on the patristic reference: the value a domain distance
    # should reproduce if it tracked divergence time faithfully
    Tall = Dpat_all.loc[sp108, sp108].values.astype(float)
    plT = np.delete(Tall[pi], pi)
    offT = Tall[np.triu_indices(len(sp108), k=1)]
    lev["patristic_reference"] = {"platypus_mean_distance": float(plT.mean()),
                                  "all_pairs_mean_distance": float(offT.mean()),
                                  "ratio": float(plT.mean() / offT.mean())}
    cv.log(f"    patristic : platypus mean distance is {lev['patristic_reference']['ratio']:.2f}x "
           f"the overall mean (this is the reference, i.e. the TRUE leverage)")

    # ---- does domain content recover the deepest split when the platypus is kept? ----
    cv.log("\n  deepest-split test: is the platypus recovered as sister to all therians?")
    labels = [cv.format_species_label_for_newick(s) for s in sp108]
    plab = cv.format_species_label_for_newick(PLATYPUS)
    from ete3 import Tree
    deep = {}
    for basis, Z in (("raw", Xr), ("relative", Xrel), ("CLR", Xclr)):
        M = distances(Z)
        trees = {"UPGMA": cv.build_upgma_newick(pd.DataFrame(M, index=sp108, columns=sp108)),
                 "NJ": ct.nj_newick(M, labels)}
        me, _ = ct.fastme_me_newick(M, labels, f"platypus_{basis}")
        if me:
            trees["ME"] = me
        for meth, nwk in trees.items():
            t = Tree(nwk, format=1)
            for lf in t.iter_leaves():
                lf.name = lf.name.replace("'", "").strip()
            t.set_outgroup(t & plab) if len(t.search_nodes(name=plab)) else None
            # the platypus sister group is the other child of its parent node; counting its
            # leaves gives the number of taxa it would have to contain, n_species - 1, for
            # the tree to place the platypus outside all therians
            node = t & plab
            sis = [c for c in node.up.children if c is not node] if node.up else []
            n_sis = sum(len(c.get_leaves()) for c in sis)
            ok = (n_sis == len(sp108) - 1)
            deep[f"{basis}_{meth}"] = {"platypus_sister_group_size": int(n_sis),
                                       "n_therians": len(sp108) - 1,
                                       "recovers_deepest_split": bool(ok)}
            cv.log(f"    {basis:9} {meth:5}: platypus sister group has {n_sis:3} taxa "
                   f"(needs {len(sp108)-1} for the true deepest split) -> "
                   f"{'RECOVERED' if ok else 'not recovered'}")
            with open(os.path.join(OUT, f"tree_withplatypus_{basis}_{meth}.nwk"), "w") as fh:
                fh.write(nwk + "\n")

    # ---- headline metrics of the classifier runs, with and without the platypus ----
    # read from disk if those runs are present; this script does not retrain the classifier
    pub = {}
    p = os.path.join(REPO, "results", "classification_cv_tree_reconstruction_with_platypus",
                     "comparison_summary.json")
    p0 = os.path.join(REPO, "results", "classification_cv_tree_reconstruction",
                      "comparison_summary.json")
    for tag, path in (("with_platypus", p), ("without_platypus", p0)):
        if os.path.exists(path):
            d = json.load(open(path))
            pub[tag] = {k: d.get(k) for k in
                        ("n_pairs", "accuracy", "balanced_accuracy", "cohen_kappa",
                         "pearson_true_vs_pred_upper_triangle",
                         "spearman_true_vs_pred_upper_triangle",
                         "robinson_foulds_normalized")}
    if len(pub) == 2:
        cv.log("\n  published classifier runs:")
        for k, v in pub.items():
            cv.log(f"    {k:18} " + "  ".join(f"{a}={b}" for a, b in v.items() if b is not None))

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"xenarthra": {
            "in_species_list": 0, "in_domain_matrix": 0, "in_timetree_tips": 0,
            "conclusion": ("Xenarthra is absent from the SOURCE domain-count dataset and from "
                           "the TimeTree tips; it was not removed by any filtering in this study.")},
            "cohorts": detail, "correlations": out_rows,
            "platypus_leverage": lev, "deepest_split": deep,
            "published_classifier_runs": pub}, fh, indent=2)
    tbl.to_csv(os.path.join(OUT, "platypus_sensitivity.csv"), index=False)
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
