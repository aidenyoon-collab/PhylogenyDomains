#!/usr/bin/env python3
"""
Score every domain-based distance measure against the reference phylogeny in one table.

The measures span presence/absence (Jaccard, Hamming, Dice), raw copy number (L1, L2,
Bray-Curtis), relative abundance (L1, L2, Bray-Curtis, correlation), compositional
centered log-ratio distances, the weighted squared-difference metric used in Part 1, and
the random-forest predictions. Running presence/absence beside copy number separates the
phylogenetic signal carried by which domains a species has from the signal carried by how
many copies it has, which a copy-number measure alone cannot distinguish.

Every measure is computed on the same canonical cohort (106 species, 7,976
variance-filtered domains) and scored the same way: Pearson and Spearman against
patristic distance, then UPGMA, Neighbor-Joining and balanced minimum evolution trees
scored by normalized Robinson-Foulds and by per-group rooted monophyly over the
nine-group panel. Fixing the cohort and the scoring is what makes the measures
comparable; a difference in the table is then a difference between measures and not
between runs.

Each measure carries one of three tiers, and the tier governs how its row should be read:
  UNSUPERVISED   uses no TimeTree information whatsoever
  SELECTED       unsupervised in form, but the domains and weights were chosen using
                 TimeTree (the Part-1 metric)
  SUPERVISED     trained on TimeTree-derived class labels (the random-forest arms)
A supervised method beating an unsupervised one is expected and is not evidence that its
underlying character is more informative; the meaningful comparison is within a tier.

Writes under results/baseline_comparison/: baseline_comparison.csv (one row per measure,
with both correlations, normalized Robinson-Foulds for each tree method, the best of the
three, and the rooted-monophyly count out of nine groups), summary.json (the best
unsupervised tree, the best supervised tree, and the published supervised result), and one
Newick file per measure and tree method. Robinson-Foulds is normalized, so 1 means the
tree shares no internal split with the reference and 0 means the topologies agree; the
tiebreak_rf_min and tiebreak_rf_max columns bound that value under 100 negligible
perturbations of the distance matrix, which exposes measures whose topology rests on
arbitrary resolution of tied distances.
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
import domain_time_scatter as dts  # noqa: E402

OUT = os.path.join(REPO, "results", "baseline_comparison")
SEED = cv.RANDOM_STATE
N_JITTER = 100
N_BEST = 50


def clr(X: np.ndarray, pc: float) -> np.ndarray:
    C = X.astype(float) + pc
    return np.log(C) - np.log(C).mean(axis=1, keepdims=True)


def main() -> None:
    cv.ensure_dir(OUT)
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    sp = sorted(species)
    n = len(sp)
    Xrel = X_norm.loc[sp].values.astype(float)
    raw_full = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH,
                                     cv.load_species_list(cv.SPECIES_LIST_PATH))
    Xraw = raw_full.loc[sp, list(X_norm.columns)].values.astype(float)
    Xpa = (Xraw > 0).astype(float)
    assert Xrel.shape == Xraw.shape == Xpa.shape
    cv.log(f"cohort {Xrel.shape[0]} species x {Xrel.shape[1]} domains")

    T = dist_matrix.loc[sp, sp].values.astype(float)
    iu = np.triu_indices(n, k=1)
    labels = [cv.format_species_label_for_newick(s) for s in sp]
    sp2clade = ct.species_to_clade()
    from ete3 import Tree
    tref = Tree(cv.reference_tree_to_formatted_newick(
        cv.load_and_prune_reference_tree(sp)), format=1)

    # ---------------- UNSUPERVISED distances ----------------
    D: Dict[str, Dict] = {}

    def add(name, vec, character, tier, basis):
        D[name] = {"D": squareform(vec) if vec.ndim == 1 else vec,
                   "character": character, "tier": tier, "basis": basis}

    add("presence_absence_jaccard", pdist(Xpa, "jaccard"), "presence/absence", "UNSUPERVISED", "binary")
    add("presence_absence_hamming", pdist(Xpa, "hamming"), "presence/absence", "UNSUPERVISED", "binary")
    add("presence_absence_dice", pdist(Xpa, "dice"), "presence/absence", "UNSUPERVISED", "binary")
    add("copynumber_raw_L1", pdist(Xraw, "cityblock"), "copy-number", "UNSUPERVISED", "raw counts")
    add("copynumber_raw_L2_euclidean", pdist(Xraw, "euclidean"), "copy-number", "UNSUPERVISED", "raw counts")
    add("copynumber_raw_braycurtis", pdist(Xraw, "braycurtis"), "abundance", "UNSUPERVISED", "raw counts")
    add("relabundance_L1", pdist(Xrel, "cityblock"), "relative abundance", "UNSUPERVISED", "relative freq")
    add("relabundance_L2_euclidean", pdist(Xrel, "euclidean"), "relative abundance", "UNSUPERVISED", "relative freq")
    add("relabundance_braycurtis", pdist(Xrel, "braycurtis"), "abundance", "UNSUPERVISED", "relative freq")
    add("relabundance_correlation", pdist(Xrel, "correlation"), "abundance", "UNSUPERVISED", "relative freq")
    add("aitchison_CLR_pc0.5", pdist(clr(Xraw, 0.5), "euclidean"), "compositional", "UNSUPERVISED", "CLR(raw+0.5)")
    add("aitchison_CLR_pc1.0", pdist(clr(Xraw, 1.0), "euclidean"), "compositional", "UNSUPERVISED", "CLR(raw+1.0)")

    # ---------------- SELECTED: the Part-1 weighted metric ----------------
    pf = (pd.DataFrame(Xraw, index=sp) > 0).sum(axis=0) / n
    keep = np.flatnonzero(pf.values >= 0.05)
    Xp1 = pd.DataFrame(Xraw[:, keep], index=sp)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    Tvec = T[iu]
    stats = dts.compute_domain_stats(Xp1, Tvec, pairs)
    panel = dts.select_ranked_domains(stats)[:N_BEST]
    for wtag, use_alpha in (("alpha_slope", True), ("variance", False)):
        v = np.asarray(dts.compute_summed_distances(Xp1, panel, stats, pairs, use_alpha=use_alpha), float)
        M = np.zeros((n, n)); M[iu] = v; M = M + M.T
        D[f"part1_N50_{wtag}"] = {"D": M, "character": "weighted squared difference",
                                  "tier": "SELECTED (domains+weights chosen using TimeTree)",
                                  "basis": "raw counts, top-50"}

    # ---------------- SUPERVISED: the random-forest arms ----------------
    for path, cols in [
        (os.path.join(REPO, "results", "ordinal_trees", "pair_probabilities_and_distances.csv"),
         ["CENTROID_ARGMAX", "CENTROID_EXPECTED", "ORDINAL_EXPECTED"]),
        (os.path.join(REPO, "results", "continuous_trees", "predicted_distances_all_arms.csv"),
         ["RF_REG"]),
    ]:
        if not os.path.exists(path):
            cv.log(f"  (skipping {os.path.basename(path)} -- not present)")
            continue
        df = pd.read_csv(path)
        pos = {s: i for i, s in enumerate(sp)}
        # These prediction files are produced by other scripts, so they can be left over from a
        # run on a different cohort. The matrix fill below would then raise a bare KeyError on an
        # unrecognized species name, which says nothing about the cause. Check the names first and
        # report what is wrong, so the fix (re-run the producing script) is clear.
        unknown = ({str(s) for s in df["sp1"]} | {str(s) for s in df["sp2"]}) - set(pos)
        if unknown:
            cv.log(f"  SKIPPING {os.path.basename(path)} -- built on a different cohort; it contains "
                   f"{len(unknown)} species not in this run (e.g. {sorted(unknown)[:3]}). "
                   f"Re-run the script that produces it, then re-run this one.")
            continue
        for c in cols:
            if c not in df.columns:
                continue
            M = np.zeros((n, n))
            for r in df.itertuples():
                M[pos[r.sp1], pos[r.sp2]] = M[pos[r.sp2], pos[r.sp1]] = getattr(r, c)
            D[f"rf_{c}"] = {"D": M, "character": "RF prediction",
                            "tier": "SUPERVISED (trained on TimeTree labels)",
                            "basis": "relative freq |dC|"}

    # ---------------- score everything identically ----------------
    rng = np.random.default_rng(SEED)
    rows = []
    for name, info in D.items():
        M = info["D"]
        v = M[iu]
        r_p = float(pearsonr(Tvec, v)[0]); r_s = float(spearmanr(Tvec, v).statistic)
        trees = {"UPGMA": cv.build_upgma_newick(pd.DataFrame(M, index=sp, columns=sp)),
                 "NJ": ct.nj_newick(M, labels)}
        me, _src = ct.fastme_me_newick(M, labels, name)
        if me:
            trees["ME"] = me
        sc = {k: ct.score_tree(t, sp, sp2clade, tref) for k, t in trees.items()}
        for k, t in trees.items():
            with open(os.path.join(OUT, f"tree_{name}_{k}.nwk"), "w") as fh:
                fh.write(t + "\n")
        band = []
        scale = max(abs(v).max(), 1e-12)
        for _ in range(N_JITTER):
            E = rng.normal(0, 1e-10 * scale, (n, n)); E = np.triu(E, 1); E = E + E.T
            band.append(ct.rf_of_matrix(M + E, sp, tref)[0])
        best = min(sc, key=lambda k: sc[k]["robinson_foulds_normalized"])
        rows.append({
            "distance": name, "tier": info["tier"], "character": info["character"],
            "basis": info["basis"], "pearson_vs_patristic": r_p, "spearman_vs_patristic": r_s,
            "rf_UPGMA": sc["UPGMA"]["robinson_foulds_normalized"],
            "rf_NJ": sc["NJ"]["robinson_foulds_normalized"],
            "rf_ME": sc["ME"]["robinson_foulds_normalized"] if "ME" in sc else np.nan,
            "rf_best": sc[best]["robinson_foulds_normalized"], "best_method": best,
            "mono_rooted_UPGMA": sc["UPGMA"]["n_monophyletic_rooted"],
            "n_groups": sc["UPGMA"]["n_groups"],
            "tiebreak_rf_min": float(np.min(band)), "tiebreak_rf_max": float(np.max(band)),
        })
        cv.log(f"  {name:32} rho={r_s:+.3f} UPGMA={rows[-1]['rf_UPGMA']:.4f} "
               f"NJ={rows[-1]['rf_NJ']:.4f} ME={rows[-1]['rf_ME']:.4f} "
               f"best={rows[-1]['rf_best']:.4f}({best}) mono={rows[-1]['mono_rooted_UPGMA']}/9")

    tbl = pd.DataFrame(rows).sort_values(["tier", "rf_best"])
    tbl.to_csv(os.path.join(OUT, "baseline_comparison.csv"), index=False)

    uns = tbl[tbl.tier == "UNSUPERVISED"]
    sup = tbl[tbl.tier.str.startswith("SUPERVISED")]
    best_uns = uns.loc[uns.rf_best.idxmin()]
    pub = tbl[tbl.distance == "rf_CENTROID_ARGMAX"]
    verdict = {
        "best_unsupervised": {"distance": best_uns.distance, "rf_best": float(best_uns.rf_best),
                              "method": best_uns.best_method},
        "published_supervised_rf": (float(pub.rf_UPGMA.iloc[0]) if len(pub) else None),
        "best_supervised": (float(sup.rf_best.min()) if len(sup) else None),
        "headline_note": ("A supervised method is trained on TimeTree labels and an unsupervised "
                          "one is not, so the tiers are not directly comparable. The informative "
                          "comparison is whether the supervised pipeline's tree beats the best "
                          "simple distance's tree, reported above."),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"verdict": verdict, "rows": rows}, fh, indent=2)
    cv.log("")
    show = ["distance", "tier", "spearman_vs_patristic", "rf_UPGMA", "rf_NJ", "rf_ME",
            "rf_best", "best_method", "mono_rooted_UPGMA"]
    cv.log(tbl[show].to_string(index=False))
    cv.log(f"\nbest UNSUPERVISED tree: {best_uns.distance} at RF {best_uns.rf_best:.4f} "
           f"({best_uns.best_method})")
    if len(pub):
        cv.log(f"published supervised RF->UPGMA: {float(pub.rf_UPGMA.iloc[0]):.4f}")
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
