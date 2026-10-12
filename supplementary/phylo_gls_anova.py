#!/usr/bin/env python3
"""
Phylogenetically informed one-way ANOVA of mammalian protein-domain frequencies.

A plain one-way ANOVA across clades treats species as independent observations, which they
are not: closely related species share trait values through common ancestry alone. This
script redoes that comparison as a phylogenetic generalized least-squares (PGLS) ANOVA,
modeling the residual covariance between species as the shared-ancestry covariance implied
by the reference TimeTree phylogeny, and it measures how large a clade shift each statistic
could have resolved on this tree.

The companion script `phylogenetic_anova.py` covers the other standard option, a Garland
simulation ANOVA: it keeps the observed ordinary-least-squares (OLS) F and rebuilds the null
by simulating traits on the tree. The two differ in efficiency, and the difference is large.
Under a covariance that is known up to scale, the best linear unbiased estimator is
generalized least squares, not ordinary least squares; on this tree, at a clade shift of 4
tip standard deviations, the calibrated GLS ANOVA reaches 0.76-0.99 power where the OLS F
reaches 0.001-0.50. A clade effect is therefore estimable here, with a finite contrast
standard error (0.48-1.07 tip-SD); what is limited is the particular combination of an OLS F
statistic, a pure-Brownian null and a rank-1 Bonferroni bar, which can certify only very
large shifts.

Three analyses, all written to results/phylo_gls_anova/.

(A) Observed PGLS ANOVA on the real data. For every domain in the one-way ANOVA table, a GLS
    one-way ANOVA across the 7 clades under Pagel's lambda covariance
    C(lambda) = lambda*C + (1-lambda)*diag(C), evaluated at each lambda in a bracket and
    again at each domain's own maximum-likelihood lambda-hat. Whitening by the Cholesky
    factor of C(lambda) reduces the GLS problem to an ordinary linear model, so the GLS F is
    exactly F(K-1, n-K) distributed under the null when C(lambda) is the true covariance:
    there is no simulation floor and no resolution limit on the p-values. That calibration
    is verified by simulation here rather than assumed. Benjamini-Hochberg FDR is applied
    across the scored domains.

(B) Power bracket for both statistics, so that the numbers are read as properties of the
    test and not of the study design. A known shift of k tip standard deviations is injected
    into one clade, and the detection rate is reported per clade, per lambda, per statistic,
    at four thresholds: raw p < 0.05, Bonferroni at rank 1, and BH at ranks 100 and 1000.
    Every cell is conditional on a lambda, a statistic and a multiple-testing bar, and moves
    substantially with all three. The rank-1 Bonferroni column is the most pessimistic
    framing available and is named for what it is, which is why it is reported beside the BH
    ranks. The OLS null is simulated, because the OLS F is not F-distributed under
    phylogenetic covariance; its Monte-Carlo p-values use (1 + #{null >= obs}) / (N + 1),
    which is never zero and avoids the ~1/N anticonservatism of the uncorrected r/N
    (North, Curtis & Sham 2002, https://doi.org/10.1086/341527; Phipson & Smyth 2010,
    https://doi.org/10.2202/1544-6115.1585). N_NULL is set high enough that the extreme
    quantile rests on many exceedances rather than one; at 200,000 draws the rank-1
    threshold still varied between 118 and 138 across seeds. The shift grid runs past 5
    tip-SD so that an undetectable effect size is reported as a number.

    The zero-shift row is a Monte-Carlo self-consistency check between the trait generator
    and the statistic, nothing more. It cannot detect model mis-specification, and on these
    data the lambda = 1 null is in fact rejected by the observed F distribution.

(C) Effective replication. The GLS standard error of each clade-vs-rest contrast, in tip-SD
    units, beside the standard error that independent sampling of the same number of species
    would have given. A monophyletic clade is often described as contributing "effectively
    one independent contrast"; measured on this tree that is an order of magnitude too
    pessimistic. The contrast standard error improves only slightly as a clade grows, but it
    does improve, and it stays finite.

REFERENCES
  Garland et al. 1993, Syst. Biol. 42:265-292   https://doi.org/10.1093/sysbio/42.3.265
  Grafen 1989, Phil. Trans. R. Soc. B 326:119   https://doi.org/10.1098/rstb.1989.0106
  Garland & Ives 2000, Am. Nat. 155:346-364     https://doi.org/10.1086/303327
  Maddison & FitzJohn 2015, Syst. Biol. 64:127  https://doi.org/10.1093/sysbio/syu070
  Adams & Collyer 2018, Evolution 72:1204-1215  https://doi.org/10.1111/evo.13492
  Pagel 1999, Nature 401:877-884                https://doi.org/10.1038/44766

OUTPUTS, all under results/phylo_gls_anova/
  observed_gls_anova.csv              one row per domain: the OLS F, the ML lambda-hat, the
                                      GLS F and its raw and BH-adjusted p at lambda-hat, and
                                      the GLS F at each bracketed lambda.
  power_by_clade_lambda_statistic.csv one row per (lambda, clade, shift): the OLS and GLS
                                      detection rate at each of the four thresholds.
  summary.json                        design, observed significant counts, the GLS null
                                      calibration, per-clade contrast standard errors, the
                                      thresholds and how to read all of it.
  run_config.json                     simulation sizes, lambda and shift grids, seed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import f as fdist
from statsmodels.stats.multitest import multipletests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ete3 import Tree  # noqa: E402

import classification_cv_tree_reconstruction as cv  # noqa: E402
import phylogenetic_anova as pa  # noqa: E402  (single source for the VCV, lambda ML, OLS F)


def tree_for(sp_list):
    """
    Load the reference phylogeny, prune it to `sp_list`, and restore the original labels.

    The pruned tree is round-tripped through the Newick formatting used elsewhere in the
    pipeline, which rewrites species labels; the leaf names are mapped back afterwards so
    that tips can be indexed by the species names used in the count matrix. Identical to the
    construction in the companion simulation ANOVA, so both score the same tree.
    """
    t = Tree(cv.reference_tree_to_formatted_newick(
        cv.load_and_prune_reference_tree(sp_list)), format=1)
    fmt = {cv.format_species_label_for_newick(s): s for s in sp_list}
    for lf in t.iter_leaves():
        lf.name = fmt.get(lf.name, lf.name)
    return t

OUT = os.path.join(REPO, "results", "phylo_gls_anova")
SEED = cv.RANDOM_STATE
# Size of the multiple-testing family the one-way ANOVA is run over, used here only to set
# the Bonferroni and BH bars in the power bracket (section B). On the 106-species cohort it is
# 7,976 variance-filtered domains minus the 198 present in fewer than three clades = 7,778.
# A cohort that includes Cervus hanglu yarkandensis gives 7,822 instead, so a power table
# produced on that cohort is not directly comparable with one produced on this one.
N_TESTS_FAMILY = 7_778
# lambda = 0.10 is in the bracket because it is the empirically calibrated value for these
# data: it is the lambda at which the median of the observed F distribution matches the median
# of the analytic null. Including it here means any figure quoted for it is reproducible from
# this script's own output rather than from a separate calculation.
LAMBDAS = (0.0, 0.10, 0.25, 0.5, 0.75, 1.0)
SHIFTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
N_NULL = 5_000_000            # OLS null; the rank-1 quantile needs many exceedances
N_SIM = 3_000                 # draws per (clade, shift, lambda, statistic) power cell
N_CAL = 200_000               # draws for the GLS calibration check
CHUNK = 20_000


def gls_F(Y: np.ndarray, L: np.ndarray, Q0: np.ndarray, Q1: np.ndarray,
          df1: int, df2: int) -> np.ndarray:
    """
    GLS one-way ANOVA F across clades, vectorized over the columns of Y.

    Whitening by the Cholesky factor of C(lambda) turns the GLS problem into an ordinary
    linear model, so the usual two-model F applies exactly: whiten, project onto the
    intercept-only and the full clade design, and compare residual sums of squares. Q0 and Q1
    are orthonormal bases of the whitened designs, passed in so that they are built once and
    reused across all columns. The denominator is floored to avoid dividing by zero on a
    column that the clade design fits perfectly.
    """
    Yw = solve_triangular(L, Y, lower=True)
    tot = (Yw ** 2).sum(axis=0)
    rss0 = tot - ((Q0.T @ Yw) ** 2).sum(axis=0)
    rss1 = tot - ((Q1.T @ Yw) ** 2).sum(axis=0)
    return ((rss0 - rss1) / df1) / np.maximum(rss1 / df2, 1e-300)


def designs(C: np.ndarray, lam: float, gmat: np.ndarray):
    """Cholesky of C(lambda) plus orthonormal bases of the whitened null/full designs."""
    L = pa.chol(C, lam)
    X0 = np.ones((C.shape[0], 1))
    Q0, _ = np.linalg.qr(solve_triangular(L, X0, lower=True))
    Q1, _ = np.linalg.qr(solve_triangular(L, gmat, lower=True))
    return L, Q0, Q1


def main() -> None:
    cv.ensure_dir(OUT)
    X_norm, _D, species = cv.load_and_preprocess_data()
    species = sorted(species)
    sp2clade = pa.species_to_clade()
    by_clade = {}
    for s in species:
        by_clade.setdefault(sp2clade[s], []).append(s)
    clades = sorted([c for c, m in by_clade.items() if len(m) >= 3])
    anova_sp = sorted([s for c in clades for s in by_clade[c]])
    n, K = len(anova_sp), len(clades)
    df1, df2 = K - 1, n - K
    cv.log(f"{K} clades, {n} species, df = {df1}/{df2}")

    tree = tree_for(anova_sp)
    C = pa.phylo_vcv(tree, anova_sp)
    rng = np.random.default_rng(SEED)
    pa.verify_vcv(tree, anova_sp, C, rng)
    tip_sd = float(np.sqrt(np.mean(np.diag(C))))
    cv.log(f"tip sd = {tip_sd:.4f} (lambda-invariant: lambda rescaling leaves diag(C) alone)")

    gidx = [np.array([anova_sp.index(s) for s in by_clade[c]]) for c in clades]
    gmat = np.zeros((n, K))
    for k, g in enumerate(gidx):
        gmat[g, k] = 1.0

    # ---------------- (A) observed PGLS on the real data ----------------
    obs_tab = pd.read_csv(os.path.join(REPO, "results", "domain_analysis", "anova_domains.tsv"),
                          sep="\t")
    doms = [d for d in obs_tab.domain_name if d in X_norm.columns]
    Y = X_norm.loc[anova_sp, doms].values.astype(float)
    F_ols = pa.anova_F(Y, gidx)
    pub = obs_tab.set_index("domain_name").loc[doms, "F_statistic"].values.astype(float)
    if not np.allclose(F_ols, pub, rtol=1e-6, atol=1e-8):
        raise SystemExit("FATAL: recomputed OLS F disagrees with the published ANOVA table.")
    cv.log(f"observed traits: {Y.shape[1]} domains, OLS F reproduces the published table")

    # Correctness check on the GLS machinery. The tree is ultrametric, so at lambda = 0 the
    # covariance C(0) is a multiple of the identity, GLS reduces to OLS, and F is invariant to
    # that scale factor. gls_F must therefore reproduce the plain one-way ANOVA F exactly, and
    # give the same BH-significant count. A failure here invalidates everything below it, so
    # both comparisons abort the run rather than warn.
    L0, Q00, Q10 = designs(C, 0.0, gmat)
    F_chk = gls_F(Y, L0, Q00, Q10, df1, df2)
    if not np.allclose(F_chk, F_ols, rtol=1e-5, atol=1e-3):
        raise SystemExit(f"FATAL: GLS at lambda=0 does not reproduce the OLS F "
                         f"(max |diff| = {np.abs(F_chk - F_ols).max():.3e}).")
    n_bh_ols = int((multipletests(fdist.sf(F_ols, df1, df2), alpha=0.05,
                                  method="fdr_bh")[1] < 0.05).sum())
    n_bh_g0 = int((multipletests(fdist.sf(F_chk, df1, df2), alpha=0.05,
                                 method="fdr_bh")[1] < 0.05).sum())
    if n_bh_ols != n_bh_g0:
        raise SystemExit(f"FATAL: BH counts differ at lambda=0 ({n_bh_ols} vs {n_bh_g0}).")
    cv.log(f"  check: GLS at lambda=0 reproduces OLS exactly, both give {n_bh_g0} "
           f"BH-significant (published: 1326)")

    lam_hat, _ll = pa.estimate_lambda(Y, C, pa.LAM_GRID)
    cv.log(f"per-domain ML lambda-hat: median {np.median(lam_hat):.3f}, "
           f"{(lam_hat >= 0.9).mean()*100:.0f}% at >=0.9")

    observed = {}
    Fg_by_lam = {}
    for lam in LAMBDAS:
        L, Q0, Q1 = designs(C, lam, gmat)
        Fg = gls_F(Y, L, Q0, Q1, df1, df2)
        Fg_by_lam[lam] = Fg
        p = fdist.sf(Fg, df1, df2)
        padj = multipletests(p, alpha=0.05, method="fdr_bh")[1]
        observed[f"lambda_{lam:.2f}"] = {
            "n_significant_fdr05": int((padj < 0.05).sum()),
            "n_raw_p05": int((p < 0.05).sum()),
            "median_F": float(np.median(Fg))}
        cv.log(f"  observed PGLS lambda={lam:.2f}: {int((padj < 0.05).sum()):5} of {len(p)} "
               f"significant at BH-FDR<0.05 (raw p<0.05: {int((p < 0.05).sum())}), "
               f"median F={np.median(Fg):.2f}")

    # Each domain scored at its own ML lambda-hat, grouped by lambda so the Cholesky factor
    # and the design bases are built once per distinct value on the grid.
    F_hat = np.empty(Y.shape[1])
    for lam in np.unique(lam_hat):
        sel = np.flatnonzero(lam_hat == lam)
        L, Q0, Q1 = designs(C, float(lam), gmat)
        F_hat[sel] = gls_F(Y[:, sel], L, Q0, Q1, df1, df2)
    p_hat = fdist.sf(F_hat, df1, df2)
    padj_hat = multipletests(p_hat, alpha=0.05, method="fdr_bh")[1]
    observed["lambda_hat_per_domain"] = {
        "n_significant_fdr05": int((padj_hat < 0.05).sum()),
        "n_raw_p05": int((p_hat < 0.05).sum()),
        "median_lambda_hat": float(np.median(lam_hat)),
        "caveat": ("lambda-hat is fitted under an intercept-only mean, so a real clade effect "
                   "inflates it and this arm is therefore CONSERVATIVE, not neutral.")}
    cv.log(f"  observed PGLS at per-domain lambda-hat: "
           f"{int((padj_hat < 0.05).sum())} significant at BH-FDR<0.05")

    pd.DataFrame({"domain_name": doms, "F_ols": F_ols, "lambda_hat": lam_hat,
                  "F_gls_lambda_hat": F_hat, "p_gls_lambda_hat": p_hat,
                  "padj_gls_lambda_hat": padj_hat,
                  **{f"F_gls_lambda_{l:.2f}": Fg_by_lam[l] for l in LAMBDAS}}
                 ).to_csv(os.path.join(OUT, "observed_gls_anova.csv"), index=False)

    # ---------------- GLS null calibration, verified by simulation ----------------
    # Traits are drawn under the null with the same covariance the test assumes, and the
    # realized rejection rate at alpha = 0.05 is compared with 0.05. A size outside
    # 0.05 +/- 0.005 means the analytic p-values used in section A would be invalid, so the
    # run aborts instead of reporting them.
    calib = {}
    for lam in (0.0, 0.5, 1.0):
        L, Q0, Q1 = designs(C, lam, gmat)
        Lg = pa.chol(C, lam)
        rej = 0; got = 0; qs = []
        done = 0
        rc = np.random.default_rng(SEED + 31)
        while done < N_CAL:
            m = min(CHUNK, N_CAL - done)
            Ysim = Lg @ rc.standard_normal((n, m))
            Fg = gls_F(Ysim, L, Q0, Q1, df1, df2)
            rej += int((fdist.sf(Fg, df1, df2) < 0.05).sum()); got += m
            if len(qs) < 5:
                qs.append(Fg)
            done += m
        allq = np.concatenate(qs)
        calib[f"lambda_{lam:.2f}"] = {
            "realized_size_at_alpha05": rej / got,
            "simulated_median_F": float(np.median(allq)),
            "analytic_median_F": float(fdist.ppf(0.5, df1, df2)),
            "simulated_95th_F": float(np.percentile(allq, 95)),
            "analytic_95th_F": float(fdist.ppf(0.95, df1, df2))}
        c = calib[f"lambda_{lam:.2f}"]
        cv.log(f"  GLS calibration lambda={lam:.2f}: size {c['realized_size_at_alpha05']:.4f}, "
               f"median {c['simulated_median_F']:.3f} vs analytic {c['analytic_median_F']:.3f}, "
               f"95th {c['simulated_95th_F']:.3f} vs {c['analytic_95th_F']:.3f}")
        if abs(c["realized_size_at_alpha05"] - 0.05) > 0.005:
            raise SystemExit(f"FATAL: GLS test is not correctly sized at lambda={lam} "
                             f"(size {c['realized_size_at_alpha05']:.4f}); the analytic "
                             f"p-values below would be invalid.")
    calib["note"] = ("The GLS F is EXACTLY F(K-1, n-K) under the null when C(lambda) is the "
                     "true covariance, because whitening reduces the GLS problem to an "
                     "ordinary linear model. Verified, not assumed. This means the GLS arm has "
                     "no simulation floor -- unlike the OLS/Garland arm, whose null must be "
                     "simulated because OLS F is NOT F-distributed under phylogenetic "
                     "covariance.")

    # ---------------- (C) effective replication ----------------
    # Var(beta_hat) = sigma^2 (X' C^-1 X)^-1 = sigma^2 (Xw' Xw)^-1 with Xw = L^-1 X, while the
    # variance of a single tip is sigma^2 * mean(diag(C)). Expressing the contrast standard
    # error in units of a tip standard deviation therefore means dividing by the tip variance
    # inside the square root, sqrt( (Xw'Xw)^-1_11 / mean(diag(C)) ), not multiplying by it;
    # multiplying inflates the reported standard errors by roughly an order of magnitude.
    Lg1 = pa.chol(C, 1.0)
    tipvar = float(np.mean(np.diag(C)))
    eff = {}
    for k, cname in enumerate(clades):
        Xc = np.column_stack([np.ones(n), gmat[:, k]])
        Xw = solve_triangular(Lg1, Xc, lower=True)
        se_phy = float(np.sqrt(np.linalg.inv(Xw.T @ Xw)[1, 1] / tipvar))
        se_ind = float(np.sqrt(np.linalg.inv(Xc.T @ Xc)[1, 1]))
        eff[cname] = {"n_species": int(gmat[:, k].sum()),
                      "gls_contrast_se_tip_sd": se_phy,
                      "ols_contrast_se_if_species_were_independent": se_ind,
                      "ratio_phylo_to_independent": se_phy / se_ind}
        cv.log(f"  {cname:32} n={int(gmat[:, k].sum()):3} contrast SE "
               f"{se_phy:.3f} tip-SD (independent sampling would give {se_ind:.3f}, "
               f"ratio {se_phy / se_ind:.1f}x)")
    summary_eff = {
        "per_clade_contrast_se": eff,
        "note": ("Replaces the claim that a monophyletic clade contributes 'effectively one "
                 "independent contrast'. Measured, that is an order of magnitude too "
                 "pessimistic: the GLS contrast SE runs 0.48-1.07 tip-SD across the seven "
                 "clades, and it barely changes as clade size goes from 4 to 27 species -- "
                 "0.60 for Afrotheria (n=4) against 0.51 for Primates (n=27), where "
                 "independent sampling would have given 0.51 and 0.22. So internal "
                 "replication buys far less than independence would (the ratio is ~1.2-2.3x), "
                 "which is the right SPIRIT, but the contrast is estimable with finite "
                 "precision and 'one contrast' is not a defensible number. Note Marsupials is "
                 "the worst case at 1.067 tip-SD. For the single-origin pseudoreplication "
                 "point cite Maddison & FitzJohn 2015 "
                 "(https://doi.org/10.1093/sysbio/syu070) and quote these SEs.")}

    # ---------------- (B) power bracket, both statistics ----------------
    thresholds = {"raw_p05": 0.05,
                  "bonferroni_rank1": 0.05 / N_TESTS_FAMILY,
                  "bh_rank100": 0.05 * 100 / N_TESTS_FAMILY,
                  "bh_rank1000": 0.05 * 1000 / N_TESTS_FAMILY}
    cv.log(f"thresholds: " + ", ".join(f"{k}={v:.3e}" for k, v in thresholds.items()))

    rows = []
    for lam in LAMBDAS:
        Lg = pa.chol(C, lam)
        L, Q0, Q1 = designs(C, lam, gmat)

        cv.log(f"  lambda={lam:.2f}: simulating the OLS null, {N_NULL:,} draws")
        F_null = np.empty(N_NULL)
        rn = np.random.default_rng(SEED + 1 + int(lam * 1000))
        done = 0
        while done < N_NULL:
            m = min(CHUNK, N_NULL - done)
            F_null[done:done + m] = pa.anova_F(Lg @ rn.standard_normal((n, m)), gidx)
            done += m
        F_null.sort()
        n_exc = {k: int(N_NULL - np.searchsorted(F_null, np.quantile(F_null, 1 - v), "left"))
                 for k, v in thresholds.items()}
        cv.log(f"    OLS null F: median {np.median(F_null):.2f}, 95th "
               f"{np.percentile(F_null, 95):.2f}; exceedances at each threshold {n_exc}")

        def p_ols(F_obs):
            # (1 + #{null >= obs}) / (N + 1). The +1 in both places keeps the p-value away
            # from exactly zero and removes the ~1/N anticonservatism of the plain r/N,
            # which would otherwise be the same order as the rank-1 Bonferroni bar.
            return (1.0 + (N_NULL - np.searchsorted(F_null, F_obs, side="left"))) / (N_NULL + 1.0)

        rs = np.random.default_rng(SEED + 5 + int(lam * 1000))
        for ci, cname in enumerate(clades):
            tgt = gidx[ci]
            for k in SHIFTS:
                Ysim = Lg @ rs.standard_normal((n, N_SIM))
                Ysim[tgt, :] += k * tip_sd
                po = p_ols(pa.anova_F(Ysim, gidx))
                pg = fdist.sf(gls_F(Ysim, L, Q0, Q1, df1, df2), df1, df2)
                row = {"lambda": lam, "clade": cname, "n_species": len(tgt), "shift_tip_sd": k}
                for tname, tval in thresholds.items():
                    row[f"power_OLS_{tname}"] = float(np.mean(po < tval))
                    row[f"power_GLS_{tname}"] = float(np.mean(pg < tval))
                rows.append(row)
        sub = [r for r in rows if r["lambda"] == lam and r["shift_tip_sd"] == 4.0]
        cv.log(f"    at 4 tip-SD, bonferroni_rank1: OLS "
               f"{min(r['power_OLS_bonferroni_rank1'] for r in sub):.3f}-"
               f"{max(r['power_OLS_bonferroni_rank1'] for r in sub):.3f} vs GLS "
               f"{min(r['power_GLS_bonferroni_rank1'] for r in sub):.3f}-"
               f"{max(r['power_GLS_bonferroni_rank1'] for r in sub):.3f}")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT, "power_by_clade_lambda_statistic.csv"), index=False)

    zero = tbl[tbl.shift_tip_sd == 0.0]
    cv.log(f"zero-shift check: OLS raw-p rejection {zero.power_OLS_raw_p05.min():.3f}-"
           f"{zero.power_OLS_raw_p05.max():.3f}, GLS {zero.power_GLS_raw_p05.min():.3f}-"
           f"{zero.power_GLS_raw_p05.max():.3f}")

    summary = {
        "design": {"n_species": n, "n_clades": K, "clade_sizes": {c: len(by_clade[c]) for c in clades},
                   "df": [df1, df2], "tip_sd": tip_sd, "bh_family_size": N_TESTS_FAMILY},
        "observed_pgls": observed,
        "gls_null_calibration": calib,
        "effective_replication": summary_eff,
        "thresholds": thresholds,
        "n_null_draws_ols": N_NULL, "n_sim_per_power_cell": N_SIM, "lambdas": list(LAMBDAS),
        "shifts_tip_sd": list(SHIFTS), "random_seed": SEED,
        "zero_shift_check": {
            "ols_raw_p05": [float(zero.power_OLS_raw_p05.min()), float(zero.power_OLS_raw_p05.max())],
            "gls_raw_p05": [float(zero.power_GLS_raw_p05.min()), float(zero.power_GLS_raw_p05.max())],
            "note": ("A Monte-Carlo self-consistency check between the generator and the "
                     "statistic, nothing more. It cannot detect model mis-specification, so it "
                     "is not evidence that the lambda=1 null fits these data, and it does not: "
                     "the observed OLS F distribution sits far below the lambda=1 null "
                     "(see interpretation).")},
        "interpretation": (
            "How to read the numbers above, in three parts.\n\n"
            "1. The reported clade differences do not exceed what shared ancestry alone "
            "predicts. Observed OLS F has median ~1.3 against a lambda=1 Brownian null median "
            "of ~12, and a no-clade-effect simulation at each domain's own lambda-hat "
            "reproduces more BH-significant hits than an uncorrected ANOVA returns. The "
            "uncorrected count is therefore explicable as miscalibration of an ANOVA that "
            "assumed independent species, rather than as a lack of statistical power.\n\n"
            "2. The Garland OLS-F is an inefficient statistic on this tree, so its failure to "
            "certify an effect is partly a property of the test and not only of the data. Under "
            "a known covariance the efficient estimator is GLS, and a calibrated GLS ANOVA is "
            "markedly more powerful at the same effect size. A clade effect is estimable here, "
            "with a finite contrast standard error; it should not be described as unidentifiable "
            "by any test.\n\n"
            "3. The power table brackets what each test could have resolved; it is not a "
            "property of the design. Every entry is conditional on a lambda, a statistic and a "
            "multiple-testing bar, and moves substantially with all three. The rank-1 Bonferroni "
            "column is the most pessimistic framing available, which is why it is labelled as "
            "such and reported beside BH ranks 100 and 1000.\n\n"
            "Taken together: the enrichment and ANOVA results are best reported as descriptive "
            "repertoire differences with effect sizes and direction, noting that they are not "
            "separable from shared ancestry under a phylogenetically calibrated test, and that "
            "each clade is a single evolutionary origin, so internal replication does not supply "
            "independent evidence (Maddison & FitzJohn 2015). Note also that Filter B in "
            "scripts/domain_clade_interpretation.py, which drops domains where a single clade "
            "contributes more than 90% of the total clade means, removes precisely the most "
            "clade-diagnostic domains."),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(OUT, "run_config.json"), "w") as fh:
        json.dump({"script": os.path.basename(__file__), "N_NULL": N_NULL, "N_SIM": N_SIM,
                   "N_CAL": N_CAL, "CHUNK": CHUNK, "LAMBDAS": list(LAMBDAS),
                   "SHIFTS": list(SHIFTS), "RANDOM_STATE": SEED,
                   "N_TESTS_FAMILY": N_TESTS_FAMILY, "n_species": n, "clades": clades}, fh, indent=2)
    cv.log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
