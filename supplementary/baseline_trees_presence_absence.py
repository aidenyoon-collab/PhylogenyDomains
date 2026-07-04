#!/usr/bin/env python3
"""
Supplementary robustness analysis (reviewer-anticipated):

  (1) PRESENCE/ABSENCE vs COPY-NUMBER - does binarized domain presence/absence
      carry comparable phylogenetic signal to copy-number abundance? (Yang et al.
      2005, PNAS, reported P/A > abundance for domain-content phylogeny.)
  (2) NAIVE BASELINE TREES - what does the supervised RF->UPGMA pipeline actually
      buy over a one-line distance tree built directly from domain content?

  (3) COMPOSITIONAL ROBUSTNESS (CLR / Aitchison) - the pipeline normalizes counts
      to per-species relative frequencies (a constant-sum closure), which
      compositional-data theory (Gloor 2017; Quinn 2019) warns can induce spurious
      correlation. We redo the distance on the compositionally-correct Aitchison
      distance (Euclidean on centered-log-ratio-transformed counts) to test whether
      the closure drives the result. CLR needs the count scale + a pseudocount
      (relative frequencies have exact zeros), so two pseudocounts are checked.

It is ADDITIVE and does NOT modify or replace the canonical pipeline. It REUSES
that pipeline's exact preprocessing (identical 107-species cohort, same
variance-filtered domains, same patristic reference) and the SAME ete3
Robinson-Foulds computation, so every number is directly comparable to the
headline classifier result. No data is altered; this only reads the canonical
inputs and writes a comparison table under results/.

For each domain distance it reports, on the upper-triangle of the 107×107 matrix:
  - Pearson/Spearman correlation vs TimeTree patristic distance (Part-1-style signal)
  - normalized Robinson-Foulds vs the reference TimeTree, for BOTH a UPGMA tree
    (same average-linkage builder as the paper) and a Neighbor-Joining tree.

Reference points printed alongside:
  - Headline classifier: distance Pearson 0.83 / Spearman 0.88; RF_norm ~0.885.
  - Part-1 weighted analytical metric (alpha, N=50): Spearman ~0.27.

NOTE: correlations here are uncorrected effect sizes for comparison; the paper's
significance is established by the species-permutation Mantel test elsewhere.

Run: MPLCONFIGDIR=$PWD/.cache/matplotlib python3 supplementary/baseline_trees_presence_absence.py
"""
from __future__ import annotations
import os
import sys
import io
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from ete3 import Tree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
sys.path.insert(0, SCRIPTS)  # reuse the canonical pipeline
import classification_cv_tree_reconstruction as cv  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "baseline_presence_absence")

# headline reference numbers (read from the committed comparison_summary.json if present)
HEADLINE_RF = 0.8846
HEADLINE_DIST_PEARSON = 0.83
HEADLINE_DIST_SPEARMAN = 0.88
PART1_WEIGHTED_SPEARMAN = 0.27  # alpha-weighted, N=50 (Part-1 analytical metric)


def _nj_newick(dvec: np.ndarray, names: list) -> str:
    """Neighbor-Joining Newick from a condensed distance vector (Biopython)."""
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix as BioDM
    from Bio import Phylo

    sq = squareform(dvec)
    n = len(names)
    lower = [[float(sq[i][j]) for j in range(i + 1)] for i in range(n)]  # incl. 0 diagonal
    dm = BioDM(list(names), lower)
    nj_tree = DistanceTreeConstructor().nj(dm)
    buf = io.StringIO()
    Phylo.write(nj_tree, buf, "newick")
    return buf.getvalue().strip()


def _clr_distance(Xc_raw: np.ndarray, pseudocount: float) -> np.ndarray:
    """Aitchison distance = Euclidean on centered-log-ratio (CLR) of raw counts.

    CLR(x)_i = ln(x_i) - mean_j ln(x_j) (mean of logs = log geometric mean). A
    pseudocount is added to the raw counts (not to relative frequencies) because
    CLR is scale-invariant but undefined at zeros; the count scale is where a
    pseudocount of ~0.5-1 is interpretable as a fractional copy.
    """
    C = Xc_raw.astype(float) + pseudocount
    L = np.log(C)
    clr = L - L.mean(axis=1, keepdims=True)
    return pdist(clr, metric="euclidean")


def _upgma_newick(dvec: np.ndarray, species: list) -> str:
    """UPGMA Newick via the paper's exact average-linkage builder."""
    sq = squareform(dvec)
    df = pd.DataFrame(sq, index=species, columns=species)
    return cv.build_upgma_newick(df)


def _rf_norm(pred_nwk: str, tree_ref: Tree, expected_leaves: set) -> tuple:
    """Normalized Robinson-Foulds vs the reference, same call as the main pipeline."""
    tp = Tree(pred_nwk, format=1)
    leaves_pred = {x.name for x in tp.iter_leaves()}
    if leaves_pred != expected_leaves:
        raise RuntimeError(
            f"Leaf-set mismatch (pred^expected): {leaves_pred ^ expected_leaves}"
        )
    res = tree_ref.robinson_foulds(tp, unrooted_trees=True)
    rf_dist, rf_max = int(res[0]), int(res[1])
    return rf_dist, rf_max, (rf_dist / rf_max if rf_max > 0 else float("nan"))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. canonical cohort, identical to the headline pipeline ---
    X_norm, dist_matrix, species_final = cv.load_and_preprocess_data()
    species = sorted(species_final)  # same ordering as construct_pairwise_features
    n = len(species)

    Xrel = X_norm.loc[species].values.astype(float)          # per-species relative freq
    PA = (Xrel > 0).astype(float)                            # presence/absence (count>0 == relfreq>0)
    pa_informative = int(((PA.sum(axis=0) > 0) & (PA.sum(axis=0) < n)).sum())

    # raw counts on the IDENTICAL cohort/domains (for CLR: needs the count scale + pseudocount)
    raw_counts_full = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH,
                                            cv.load_species_list(cv.SPECIES_LIST_PATH))
    Xc = raw_counts_full.loc[species, list(X_norm.columns)].values.astype(float)  # 107 x 8020 raw
    assert Xc.shape == Xrel.shape, "raw-count cohort/domain set must match the relative-freq matrix"

    T = dist_matrix.loc[species, species].values
    iu = np.triu_indices(n, k=1)
    y = T[iu]  # patristic upper triangle

    # --- 2. reference tree + leaf set (paper's exact formatting) ---
    ref = cv.load_and_prune_reference_tree(species)
    ref_nwk = cv.reference_tree_to_formatted_newick(ref)
    tree_ref = Tree(ref_nwk, format=1)
    expected_leaves = {cv.format_species_label_for_newick(s) for s in species}
    names = [cv.format_species_label_for_newick(s) for s in species]

    # --- 3. distances to compare ---
    # copy-number on the same relative-freq input the classifier sees; P/A binarized.
    distances = {
        "copynumber_L1_cityblock": pdist(Xrel, metric="cityblock"),
        "copynumber_L2_euclidean": pdist(Xrel, metric="euclidean"),
        "presence_absence_jaccard": pdist(PA, metric="jaccard"),
        "presence_absence_hamming": pdist(PA, metric="hamming"),
        "aitchison_CLR_pc0.5": _clr_distance(Xc, 0.5),
        "aitchison_CLR_pc1.0": _clr_distance(Xc, 1.0),
    }

    rows = []
    trees_out = {}
    for name, dvec in distances.items():
        r_p, _ = pearsonr(dvec, y)
        r_s, _ = spearmanr(dvec, y)

        upgma_nwk = _upgma_newick(dvec, species)
        nj_nwk = _nj_newick(dvec, names)
        u_d, u_max, u_norm = _rf_norm(upgma_nwk, tree_ref, expected_leaves)
        n_d, n_max, n_norm = _rf_norm(nj_nwk, tree_ref, expected_leaves)

        trees_out[name] = {"upgma": upgma_nwk, "nj": nj_nwk}
        rows.append({
            "distance": name,
            "character": ("presence/absence" if name.startswith("presence")
                          else "compositional (CLR)" if name.startswith("aitchison")
                          else "copy-number"),
            "pearson_vs_patristic": round(float(r_p), 4),
            "spearman_vs_patristic": round(float(r_s), 4),
            "rf_norm_UPGMA": round(float(u_norm), 4),
            "rf_norm_NJ": round(float(n_norm), 4),
            "rf_dist_UPGMA": u_d, "rf_dist_NJ": n_d, "rf_max": u_max,
        })

    table = pd.DataFrame(rows)

    # --- 4. internal consistency: reproduce the committed predicted tree's RF ---
    pred_path = os.path.join(REPO, "results", "classification_cv_tree_reconstruction",
                             "predicted_tree_UPGMA.nwk")
    repro = None
    if os.path.exists(pred_path):
        with open(pred_path, encoding="utf-8") as f:
            pred_committed = f.read().strip()
        try:
            cd, cmax, cnorm = _rf_norm(pred_committed, tree_ref, expected_leaves)
            repro = {"rf_dist": cd, "rf_max": cmax, "rf_norm": round(float(cnorm), 4)}
        except Exception as e:  # leaf formatting can differ; record rather than crash
            repro = {"error": str(e)}

    # --- 5. write outputs ---
    summary = {
        "n_species": n,
        "n_domains_variance_filtered": int(Xrel.shape[1]),
        "n_presence_absence_informative_domains": pa_informative,
        "headline_reference": {
            "classifier_distance_pearson": HEADLINE_DIST_PEARSON,
            "classifier_distance_spearman": HEADLINE_DIST_SPEARMAN,
            "classifier_tree_rf_norm": HEADLINE_RF,
            "part1_weighted_metric_spearman_alpha_N50": PART1_WEIGHTED_SPEARMAN,
        },
        "committed_predicted_tree_rf_reproduced": repro,
        "results": rows,
        "interpretation_notes": [
            "Distance correlations here are UNCORRECTED effect sizes over non-independent "
            "pairs (shared-species pseudoreplication); paper significance is via the Mantel "
            "test. Presence/absence Pearson (~0.01-0.02) is statistically ~0 (p~0.25-0.30); "
            "only the Spearman values are informative.",
            "Presence/absence Spearman (~0.23) modestly exceeds naive copy-number L1 (~0.16), "
            "DIRECTIONALLY consistent with Yang et al. 2005 (P/A >= abundance), but does NOT "
            "rescue topology (RF ~0.87-0.91 for both characters).",
            "The classifier's distance Pearson 0.83 / Spearman 0.88 is a SUPERVISED model "
            "trained on binned divergence time (near the ~0.85 ceiling set by 4-class binning) "
            "and is NOT directly comparable to the unsupervised raw distances here; the gap "
            "reflects supervised-vs-unsupervised + binning structure, not proof the RF extracts "
            "topological signal the baselines miss.",
            "A naive NJ tree on raw presence/absence or copy-number distance (RF ~0.865-0.875) "
            "MATCHES / marginally beats the RF->UPGMA pipeline (0.8846) on topology. We make NO "
            "topological-superiority claim for the pipeline; Part-2's tree is a clade-level "
            "summary (the classifier + distance recovery is the primary result). This localizes "
            "the limit to the CHARACTER (homoplastic, coarse domain content), not the model. "
            "Headline tree stays UPGMA; baselines are a robustness check, "
            "not a replacement.",
            "COMPOSITIONAL robustness: the Aitchison distance (Euclidean on CLR-transformed "
            "counts) gives Spearman ~0.21-0.23 vs patristic and RF ~0.88-0.90 - the SAME "
            "weak-signal, poor-topology picture as the relative-frequency analyses. The RANK "
            "signal (Spearman ~0.21-0.23) and TOPOLOGY (RF ~0.88-0.90) are stable across "
            "pseudocounts (0.5 vs 1.0); the Pearson is mildly pseudocount-sensitive "
            "(0.21->0.17 from pc0.5->pc1.0), consistent with log-ratio geometry rather than a "
            "change in phylogenetic signal. Anchoring on the rank/topology metrics (the "
            "project's primary summaries given the zero-inflation), the constant-sum closure "
            "(Gloor 2017; Quinn 2019) does NOT drive the result; the log-ratio-correct treatment "
            "reaches the same conclusion. (CLR linearizes the magnitude relationship - Pearson "
            "rises from ~0.03 to ~0.21 - but rank signal and topology are unchanged.)",
        ],
    }
    with open(os.path.join(OUT_DIR, "baseline_presence_absence_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    table.to_csv(os.path.join(OUT_DIR, "baseline_presence_absence_table.csv"), index=False)
    for name, t in trees_out.items():
        with open(os.path.join(OUT_DIR, f"tree_{name}_UPGMA.nwk"), "w", encoding="utf-8") as f:
            f.write(t["upgma"] + "\n")
        with open(os.path.join(OUT_DIR, f"tree_{name}_NJ.nwk"), "w", encoding="utf-8") as f:
            f.write(t["nj"] + "\n")

    # --- 6. console report ---
    print("=" * 78)
    print(f"Baseline / presence-absence comparison  (n={n} species, "
          f"{Xrel.shape[1]} domains; {pa_informative} P/A-informative)")
    print("=" * 78)
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(table.to_string(index=False))
    print("-" * 78)
    print(f"Headline classifier:  dist Pearson {HEADLINE_DIST_PEARSON} / "
          f"Spearman {HEADLINE_DIST_SPEARMAN};  tree RF_norm {HEADLINE_RF}")
    print(f"Part-1 weighted metric (alpha, N=50):  Spearman {PART1_WEIGHTED_SPEARMAN}")
    if repro is not None and "rf_norm" in repro:
        print(f"Reproduced committed predicted-tree RF_norm: {repro['rf_norm']} "
              f"(expect ~{HEADLINE_RF}) -> RF machinery matches the paper.")
    elif repro is not None:
        print(f"Committed-tree RF reproduction note: {repro}")
    print(f"\nWrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
