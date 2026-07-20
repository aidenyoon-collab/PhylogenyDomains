#!/usr/bin/env python3
"""
95% confidence intervals for the Part-1 domain-distance vs TimeTree patristic
correlations, via a delete-one-species jackknife.

The Part-1 correlations (Spearman rho and Pearson r of the domain-content
distance D(N=50) against patristic distance) are computed over 5,671
species-PAIRS from 107 SPECIES. Pairs that share a species are not independent,
so a precision estimate must resample at the SPECIES level, not the pair level.
This script drops each species in turn, recomputes the correlation over the
remaining pairs, and forms the jackknife standard error and a
normal-approximation 95% CI. It reuses the exact Part-1 inputs from
mantel_significance.part1_inputs (same domain panel, weighting and <5%
prevalence filter as the headline analysis), so the point estimates match the
values in domain_time_correlations.tsv at N=50.

Prerequisite: run domain_time_scatter.py first so that domain_time_domainStats.tsv,
domain_time_correlations.tsv and data_raw/TopDomains.txt exist (the same inputs
mantel_significance.py needs).

Values reported in the manuscript (N=50):
  alpha-weighted    Spearman rho = 0.27 (95% CI 0.10-0.44), Pearson r = 0.67 (0.42-0.91)
  variance-weighted Spearman rho = 0.27 (95% CI 0.10-0.44), Pearson r = 0.67 (0.44-0.91)
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr

from mantel_significance import part1_inputs  # sibling module in scripts/


def jackknife_ci(D, T, i_idx, j_idx, n):
    """Delete-one-species jackknife SE and 95% CI for Spearman rho and Pearson r."""
    full_rho = spearmanr(D, T).statistic
    full_r = pearsonr(D, T)[0]
    rho_s, r_s = [], []
    for s in range(n):
        keep = (i_idx != s) & (j_idx != s)  # drop every pair touching species s
        rho_s.append(spearmanr(D[keep], T[keep]).statistic)
        r_s.append(pearsonr(D[keep], T[keep])[0])

    def se(arr):
        arr = np.asarray(arr, float)
        return np.sqrt((n - 1) / n * np.sum((arr - arr.mean()) ** 2))

    out = {}
    for name, full, arr in (("spearman", full_rho, rho_s), ("pearson", full_r, r_s)):
        s = se(arr)
        out[name] = {
            "estimate": float(full),
            "jackknife_se": float(s),
            "ci95_low": float(full - 1.96 * s),
            "ci95_high": float(full + 1.96 * s),
        }
    return out


def main():
    n, pairs, T, D_a, D_v = part1_inputs()
    i_idx = np.array([i for i, _ in pairs])
    j_idx = np.array([j for _, j in pairs])
    T = np.asarray(T, float)
    print(f"n_species = {n}, n_pairs = {len(pairs)}")
    for label, D in (("alpha-weighted", D_a), ("variance-weighted", D_v)):
        res = jackknife_ci(np.asarray(D, float), T, i_idx, j_idx, n)
        for metric in ("spearman", "pearson"):
            r = res[metric]
            print(f"  {label:17s} {metric:8s}: {r['estimate']:.3f} "
                  f"(95% CI {r['ci95_low']:.2f} to {r['ci95_high']:.2f}; "
                  f"jackknife SE {r['jackknife_se']:.3f})")


if __name__ == "__main__":
    main()
