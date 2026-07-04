#!/usr/bin/env python3
"""
Significance for the Part-1 and Part-2 distance correlations that respects the
non-independence of species pairs.

The pairwise correlations are computed over 5,671 species-PAIRS from 107 SPECIES.
Pairs that share a species are NOT independent, so the i.i.d. scipy p-values
(pearsonr/spearmanr -> p = 0.0) are invalid (pseudoreplication / inflated df).
The correct test is a MANTEL test: keep the observed r/rho as the effect size,
but get the p-value from a null built by permuting SPECIES LABELS (rows+cols of
one distance matrix together), which preserves the within-matrix dependence.

Reports (effect size unchanged; only the p-value is now honest):
  Part 1: domain distance D(N=50, alpha) and D(N=50, variance) vs TimeTree patristic
  Part 2: predicted distance (classifier class centroids) vs TimeTree patristic

Writes results/mantel_significance.json and prints a table. Deterministic
(SEED=42). The observed r/rho are asserted to match the saved pipeline outputs
(domain_time_correlations.tsv / comparison_summary.json) before trusting the p.

Run: MPLCONFIGDIR=$PWD/.cache/matplotlib python3 scripts/mantel_significance.py
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from domain_time_scatter import (  # noqa: E402
    load_species_list, load_domain_counts, load_phylogeny_and_compute_patristic,
    select_ranked_domains, compute_summed_distances,
)
from classification_cv_tree_reconstruction import compute_patristic_matrix  # noqa: E402

DATA = os.path.join(REPO, "data_raw")
CV = os.path.join(REPO, "results", "classification_cv_tree_reconstruction")
N_BEST = 50
N_PERM = 9999
SEED = 42


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) / d) if d > 0 else 0.0


def _matrix(n, pairs, vec):
    M = np.zeros((n, n), dtype=float)
    for (i, j), v in zip(pairs, vec):
        M[i, j] = M[j, i] = v
    return M


def mantel(M_perm, fixed_vec, i_idx, j_idx, n_perm, seed):
    """Permute species labels of M_perm; correlate its pair-vector with fixed_vec.
    Returns dict with observed pearson/spearman and two-sided empirical p for each."""
    obs_vec = M_perm[i_idx, j_idx]
    r_obs = _pearson(obs_vec, fixed_vec)
    fixed_rank = rankdata(fixed_vec)
    rho_obs = _pearson(rankdata(obs_vec), fixed_rank)
    rng = np.random.default_rng(seed)
    n = M_perm.shape[0]
    ge_r = ge_rho = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        v = M_perm[np.ix_(p, p)][i_idx, j_idx]
        if abs(_pearson(v, fixed_vec)) >= abs(r_obs) - 1e-12:
            ge_r += 1
        if abs(_pearson(rankdata(v), fixed_rank)) >= abs(rho_obs) - 1e-12:
            ge_rho += 1
    return {
        "pearson_r": r_obs, "pearson_p_mantel": (1 + ge_r) / (1 + n_perm),
        "spearman_rho": rho_obs, "spearman_p_mantel": (1 + ge_rho) / (1 + n_perm),
        "n_species": n, "n_pairs": len(fixed_vec), "n_perm": n_perm,
    }


def part1_inputs():
    sp0 = load_species_list(os.path.join(DATA, "MammalsList.txt"))
    X = load_domain_counts(os.path.join(DATA, "MammalDomainCount.tsv"), sp0)
    order = list(X.index)
    _, _, in_tree = load_phylogeny_and_compute_patristic(os.path.join(DATA, "MammalsPhylogeny.nwk"), order)
    # Canonical cohort: platypus (Ornithorhynchus) excluded, Cervus included.
    sp_final = [s for s in order if s in in_tree and "Ornithorhynchus" not in s]
    X = X.loc[sp_final]
    T_vec, pairs, _ = load_phylogeny_and_compute_patristic(os.path.join(DATA, "MammalsPhylogeny.nwk"), sp_final)
    pf = (X > 0).sum(axis=0) / X.shape[0]
    X = X.loc[:, pf >= 0.05]
    stats = pd.read_csv(os.path.join(REPO, "domain_time_domainStats.tsv"), sep="\t")
    panel = select_ranked_domains(stats, os.path.join(DATA, "TopDomains.txt"))[:N_BEST]
    D_a = np.asarray(compute_summed_distances(X, panel, stats, pairs, use_alpha=True), float)
    D_v = np.asarray(compute_summed_distances(X, panel, stats, pairs, use_alpha=False), float)
    return len(sp_final), pairs, np.asarray(T_vec, float), D_a, D_v


def main():
    out = {}
    # ---- Part 1 ----
    n, pairs, T, D_a, D_v = part1_inputs()
    i_idx = np.array([i for i, _ in pairs]); j_idx = np.array([j for _, j in pairs])
    M_T = _matrix(n, pairs, T)
    tsv = pd.read_csv(os.path.join(REPO, "domain_time_correlations.tsv"), sep="\t")
    row = tsv[tsv["N"] == N_BEST].iloc[0]
    for key, D in (("part1_alpha_N50", D_a), ("part1_variance_N50", D_v)):
        M_D = _matrix(n, pairs, D)
        res = mantel(M_D, T, i_idx, j_idx, N_PERM, SEED)
        suff = "alpha" if "alpha" in key else "var"
        assert abs(res["pearson_r"] - float(row[f"pearson_{suff}"])) < 1e-6, "Part-1 Pearson != saved TSV"
        assert abs(res["spearman_rho"] - float(row[f"spearman_{suff}"])) < 1e-6, "Part-1 Spearman != saved TSV"
        out[key] = res
        del M_D

    # ---- Part 2 ----
    pred = pd.read_csv(os.path.join(CV, "predictions_aggregated.csv"))
    species = sorted(set(pred["sp1"]).union(pred["sp2"]))
    Dpat = compute_patristic_matrix(os.path.join(REPO, "MammalsPhylogeny.nwk"), species)
    idx = {s: k for k, s in enumerate(species)}
    n2 = len(species)
    p2 = [(idx[r.sp1], idx[r.sp2]) for r in pred.itertuples()]
    i2 = np.array([a for a, _ in p2]); j2 = np.array([b for _, b in p2])
    T2 = np.array([float(Dpat.iloc[a, b]) for a, b in p2])
    P2 = pred["predicted_distance_MY"].to_numpy(float)
    M_P = _matrix(n2, p2, P2)
    res2 = mantel(M_P, T2, i2, j2, N_PERM, SEED)
    summ = json.load(open(os.path.join(CV, "comparison_summary.json")))
    assert abs(res2["pearson_r"] - summ["pearson_r"]) < 5e-3, "Part-2 Pearson != comparison_summary"
    assert abs(res2["spearman_rho"] - summ["spearman_r"]) < 5e-3, "Part-2 Spearman != comparison_summary"
    out["part2_predicted_vs_patristic"] = res2

    out["_meta"] = {"test": "Mantel (species-label permutation)", "n_perm": N_PERM,
                    "seed": SEED, "note": "p two-sided empirical; min reportable = 1/(n_perm+1)"}
    path = os.path.join(REPO, "results", "mantel_significance.json")
    json.dump(out, open(path, "w"), indent=2)

    print(f"\n{'analysis':<32} {'r':>8} {'P(Mantel)':>12} {'rho':>8} {'P(Mantel)':>12}")
    for k in ("part1_alpha_N50", "part1_variance_N50", "part2_predicted_vs_patristic"):
        r = out[k]
        print(f"{k:<32} {r['pearson_r']:>8.4f} {r['pearson_p_mantel']:>12.2e} "
              f"{r['spearman_rho']:>8.4f} {r['spearman_p_mantel']:>12.2e}")
    print(f"\n(107 species, 5671 pairs, {N_PERM} permutations) -> {path}")


if __name__ == "__main__":
    main()
