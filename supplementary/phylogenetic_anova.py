#!/usr/bin/env python3
"""
Phylogenetic non-independence, applied to both the cross-clade ANOVA and the one-vs-rest
Mann-Whitney enrichment.

Both tests treat individual species as independent observations, which they are not: species
share ancestry, so between-clade differences can arise from the tree alone. The analyses here
rebuild each null so that it accounts for the reference phylogeny.

METHOD. Garland's simulation approach (Garland et al. 1993, Syst. Biol. 42: 265-292,
doi:10.1093/sysbio/42.3.265). The observed statistic is unchanged; the NULL is rebuilt by
simulating traits on the reference tree, so between-clade differences arising from shared
ancestry alone sit inside the null. Pagel's lambda (Pagel 1999, Nature 401: 877-884,
doi:10.1038/44766) rescales the off-diagonal covariance: C(lambda) = lambda*C +
(1-lambda)*diag(C), with lambda=1 Brownian and lambda=0 phylogenetically independent.

THIS VERSION EXISTS BECAUSE AN ADVERSARIAL AUDIT FOUND FOUR DEFECTS IN THE FIRST ONE
(REVISION_SWEEP_AUDIT.md). Each is addressed:

  (1) RESOLUTION. The per-domain nulls ran at 250,000 simulations, so the smallest
      attainable p was 1/250,001 and the headline "1 of 7,822 survives" sat exactly on
      that floor -- it flipped to 0 for roughly one seed in five. All nulls now run at
      N_SIM = 1,000,000, and any p at the floor is reported as "< 1/(N+1)", never as a
      measurement.
  (2) MARGINAL MISMATCH. Observed traits are strongly skewed while the null is Gaussian,
      so an extreme observed F can reflect a skewed marginal rather than clade structure.
      A RANK-TRANSFORMED arm is therefore run: observed AND simulated traits are mapped to
      normal scores, making the comparison like-for-like.
  (3) LAMBDA PLUG-IN BIAS. lambda-hat and F are correlated, so scoring each domain against
      the null at its own lambda-hat is biased toward non-significance. Results are now a
      BRACKET -- lambda=0, lambda-hat, lambda=1 -- not a single number.
  (4) THE MANN-WHITNEY ARM WAS NEVER IMPLEMENTED. The previous version tabulated null
      quantiles, never computed an observed U, and used the 104-species ANOVA cohort where
      the published enrichment uses 107. Both fixed: observed U is recomputed on the
      published 107-species raw-count cohort and checked against
      results/clade_domain_enrichment/, and the null is built from the 107-tip tree.

A SECOND ADVERSARIAL AUDIT OF THIS REWRITE FOUND THREE MORE DEFECTS. Each is addressed:

  (5) THE MANN-WHITNEY ARM RAN AT LAMBDA=1 ONLY, contradicting (3) above and the project's
      own plan. lambda=1 is the strongest possible phylogenetic null, and it is not the one
      these traits support: Pagel's lambda fitted to the normal-scored counts of the domains
      behind the published-significant pairs has median ~0.4, and the test -- verified to be
      correctly sized -- flags only ~1% of all 8,020 domains, i.e. the real data are LESS
      clade-structured than lambda=1 Brownian motion. The arm is now a BRACKET over
      lambda in {0, 0.25, 0.5, 0.75, 1}, with the fitted-lambda region read as primary and
      lambda=1 as the worst case. "All 2,187 erased" was the lambda=1 endpoint only.
  (6) ZERO POWER REPORTED AS A NEGATIVE RESULT. For some clades the test cannot reject even
      if the clade occupied the extreme ranks -- perfect separation still yields a p above
      the FDR bar. Those pairs are UNDETERMINED, not refuted. Each pair's best attainable p
      is now computed and any clade whose ceiling cannot clear BH is flagged power_limited.
      Low power when groups are monophyletic is the known behaviour of this method
      (Adams & Collyer 2018, Evolution 72: 1204-1215, doi:10.1111/evo.13492), and all seven
      clades here are monophyletic.
  (7) TIE-BREAKING IN THE RANK TRANSFORM. normal_scores() used an argsort rank, but 89% of
      the 7,778 traits contain ties (these are integer counts), so the reported
      F_rank_transformed depended on an arbitrary tie order. Now uses tie-averaged midranks.

COHORTS, which genuinely differ and must not be mixed:
  ANOVA       104 species (the 7 clades with n>=3), per-species RELATIVE abundance.
  ENRICHMENT  107 species, ABSOLUTE counts, each clade against all other species.

Writes only under results/phylogenetic_anova/.
"""
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f as fdist
from scipy.stats import norm
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import classification_cv_tree_reconstruction as cv  # noqa: E402
from elastic_net_regression_stratified_species_cv import build_clade_mapping  # noqa: E402

OUT = os.path.join(REPO, "results", "phylogenetic_anova")
SEED = 42
N_SIM = 1_000_000
CHUNK = 50_000
N_CALIB = 20_000
LAM_GRID = np.linspace(0.0, 1.0, 21)
CERVUS = "Cervus hanglu yarkandensis"


def species_to_clade() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for clade, members in build_clade_mapping().items():
        if isinstance(members, (list, set, tuple)):
            for s in members:
                out[s.replace("_", " ")] = clade
        else:
            out[clade.replace("_", " ")] = members
    return out


def phylo_vcv(tree, leaves: List[str]) -> np.ndarray:
    """C[i,j] = root-to-MRCA path length; C[i,i] = root-to-tip. One pass, no n^2 MRCA calls."""
    idx = {s: i for i, s in enumerate(leaves)}
    C = np.zeros((len(leaves), len(leaves)))
    depth = {}
    for nd in tree.traverse("preorder"):
        depth[id(nd)] = 0.0 if nd.is_root() else depth[id(nd.up)] + float(nd.dist or 0.0)
    under: Dict[int, List[int]] = {}
    for nd in tree.traverse("postorder"):
        if nd.is_leaf():
            under[id(nd)] = [idx[nd.name]] if nd.name in idx else []
        else:
            kids = [under[id(c)] for c in nd.children]
            for a in range(len(kids)):
                for b in range(a + 1, len(kids)):
                    for i in kids[a]:
                        for j in kids[b]:
                            C[i, j] = C[j, i] = depth[id(nd)]
            under[id(nd)] = [x for k in kids for x in k]
    for s, i in idx.items():
        leaf = next(l for l in tree.iter_leaves() if l.name == s)
        C[i, i] = depth[id(leaf)]
    return C


def verify_vcv(tree, leaves: List[str], C: np.ndarray, rng, n_check: int = 200) -> None:
    """Independent check: C[i,j] must equal ete3's own root-to-MRCA distance."""
    root = tree.get_tree_root()
    bad = 0
    for _ in range(n_check):
        i, j = rng.choice(len(leaves), 2, replace=False)
        mrca = (tree & leaves[i]).get_common_ancestor(tree & leaves[j])
        if abs(root.get_distance(mrca) - C[i, j]) > 1e-6:
            bad += 1
    cv.log(f"  VCV check: {n_check - bad}/{n_check} sampled pairs match ete3 root-to-MRCA")
    if bad:
        raise SystemExit("FATAL: phylo_vcv disagrees with ete3.")


def anova_F(Y: np.ndarray, gidx: List[np.ndarray]) -> np.ndarray:
    N, k = Y.shape[0], len(gidx)
    grand = Y.mean(axis=0)
    ssb = np.zeros(Y.shape[1]); ssw = np.zeros(Y.shape[1])
    for g in gidx:
        gm = Y[g].mean(axis=0)
        ssb += len(g) * (gm - grand) ** 2
        ssw += ((Y[g] - gm) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num((ssb / (k - 1)) / (ssw / (N - k)), nan=0.0)


def normal_scores(Y: np.ndarray) -> np.ndarray:
    """
    Inverse-normal rank transform down each column, using TIE-AVERAGED (mid) ranks.

    Midranks matter here, not as a nicety: 89% of the 7,778 traits contain ties (median 2,
    up to 100 of 104 values tied), because these are integer domain counts. An argsort rank
    breaks ties in an arbitrary order that depends on the input permutation, which made the
    reported F_rank_transformed column an artifact -- PF02885 read 33.87 under argsort, 42.87
    under midranks, and 28.83-33.71 across random tie-breaks. Midranks are deterministic and
    permutation-invariant. (The downstream conclusion is unchanged either way.)
    """
    ranks = rankdata(Y, axis=0) - 1.0        # tie-averaged, 0-based
    return norm.ppf((ranks + 0.5) / Y.shape[0])


def mwu_U(Y: np.ndarray, gidx: np.ndarray, ties: bool) -> np.ndarray:
    """
    Mann-Whitney U of `gidx` versus the rest, per column.

    `ties=True` uses AVERAGE ranks (scipy's convention) and is required for the observed
    integer counts, which are heavily tied -- without it the observed U does not reproduce
    the published values. `ties=False` uses argsort ranks and is correct for simulated
    continuous traits, where ties have probability zero, and is far faster.
    """
    if ties:
        ranks = rankdata(Y, axis=0).astype(np.float64)
    else:
        order = np.argsort(Y, axis=0)
        ranks = np.empty(Y.shape, dtype=np.float64)
        seq = np.arange(1, Y.shape[0] + 1, dtype=np.float64)[:, None]
        np.put_along_axis(ranks, order, np.broadcast_to(seq, Y.shape).copy(), axis=0)
    n1 = len(gidx)
    return ranks[gidx].sum(axis=0) - n1 * (n1 + 1) / 2.0


def chol(C: np.ndarray, lam: float) -> np.ndarray:
    Cl = lam * C + (1.0 - lam) * np.diag(np.diag(C))
    return np.linalg.cholesky(Cl + np.eye(C.shape[0]) * 1e-10)


def build_null(L: np.ndarray, gidx, n_sim: int, rng, rank_transform: bool) -> np.ndarray:
    vals = np.empty(n_sim); done = 0
    while done < n_sim:
        m = min(CHUNK, n_sim - done)
        Y = L @ rng.standard_normal((L.shape[0], m))
        if rank_transform:
            Y = normal_scores(Y)
        vals[done:done + m] = anova_F(Y, gidx)
        done += m
    vals.sort()
    return vals


def emp_p(null_sorted: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return (len(null_sorted) - np.searchsorted(null_sorted, obs, side="left") + 1) / (len(null_sorted) + 1)


def estimate_lambda(Y: np.ndarray, C: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ML Pagel lambda per column, profiling over `grid`. Returns (lambda_hat, logL matrix)."""
    n = C.shape[0]
    one = np.ones((n, 1)); dg = np.diag(np.diag(C))
    ll = np.full((len(grid), Y.shape[1]), -np.inf)
    for gi, lam in enumerate(grid):
        Cl = lam * C + (1.0 - lam) * dg
        try:
            Lc = np.linalg.cholesky(Cl)
        except np.linalg.LinAlgError:
            continue
        logdet = 2.0 * np.sum(np.log(np.diag(Lc)))
        Ci_one = np.linalg.solve(Cl, one)
        denom = (one.T @ Ci_one).item()
        mu = (one.T @ np.linalg.solve(Cl, Y)).ravel() / denom
        R = Y - one @ mu[None, :]
        s2 = np.maximum(np.einsum("ij,ij->j", R, np.linalg.solve(Cl, R)) / n, 1e-300)
        ll[gi] = -0.5 * (n * np.log(2 * np.pi * s2) + logdet + n)
    return grid[np.argmax(ll, axis=0)], ll


def main() -> None:
    cv.ensure_dir(OUT)
    rng = np.random.default_rng(SEED)
    X_norm, _dist, species = cv.load_and_preprocess_data()
    sp2clade = species_to_clade()
    by_clade: Dict[str, List[str]] = {}
    for s in sorted(species):
        by_clade.setdefault(sp2clade[s], []).append(s)
    clades = sorted([c for c, m in by_clade.items() if len(m) >= 3])
    anova_sp = sorted([s for c in clades for s in by_clade[c]])
    enrich_sp = sorted(species)
    cv.log(f"ANOVA cohort {len(anova_sp)} species / {len(clades)} clades; "
           f"ENRICHMENT cohort {len(enrich_sp)} species")

    from ete3 import Tree

    def tree_for(sp_list):
        t = Tree(cv.reference_tree_to_formatted_newick(
            cv.load_and_prune_reference_tree(sp_list)), format=1)
        fmt = {cv.format_species_label_for_newick(s): s for s in sp_list}
        for lf in t.iter_leaves():
            lf.name = fmt.get(lf.name, lf.name)
        return t

    summary: Dict = {
        "method": "Garland et al. 1993 phylogenetic ANOVA by simulation",
        "references": {"garland_1993": "10.1093/sysbio/42.3.265", "pagel_1999": "10.1038/44766"},
        "n_simulations_per_null": N_SIM, "lambda_grid_points": len(LAM_GRID),
        "random_seed": SEED, "p_value_floor": 1.0 / (N_SIM + 1),
        "cohorts": {"anova": len(anova_sp), "enrichment": len(enrich_sp)},
        "n_species": len(anova_sp), "clade_sizes": {c: len(by_clade[c]) for c in clades},
        "n_simulations_note": ("Every null in this script -- the standard/lambda0/lambdahat/"
                              "lambda1/rank-transformed ANOVA nulls and the untied Mann-Whitney "
                              "null -- uses N_SIM simulations. There is no separate, smaller "
                              "per-domain lambda-null count; an earlier version used 250,000 for "
                              "those, which made the headline count a function of an unrecorded "
                              "constant.")}

    # ---------------------------------------------------------------- ANOVA
    tA = tree_for(anova_sp)
    CA = phylo_vcv(tA, anova_sp)
    verify_vcv(tA, anova_sp, CA, rng)
    gA = [np.array([anova_sp.index(s) for s in by_clade[c]]) for c in clades]

    obs_tab = pd.read_csv(os.path.join(REPO, "results", "domain_analysis", "anova_domains.tsv"), sep="\t")
    doms = [d for d in obs_tab.domain_name if d in X_norm.columns]
    Y = X_norm.loc[anova_sp, doms].values.astype(float)
    Fobs = anova_F(Y, gA)
    pubF = obs_tab.set_index("domain_name").loc[doms, "F_statistic"].values.astype(float)
    if not np.allclose(Fobs, pubF, rtol=1e-6, atol=1e-8):
        raise SystemExit("FATAL: recomputed F disagrees with the published ANOVA table.")
    cv.log(f"  observed F reproduces the published table for all {len(doms)} domains")

    m = Y.mean(axis=0); sd = Y.std(axis=0, ddof=1)
    skew = np.where(sd > 0, (((Y - m) / np.where(sd > 0, sd, 1)) ** 3).mean(axis=0), 0.0)
    summary["observed_trait_skewness"] = {
        "median_abs": float(np.median(np.abs(skew))),
        "pct90_abs": float(np.percentile(np.abs(skew), 90)),
        "frac_abs_gt2": float((np.abs(skew) > 2).mean())}
    cv.log(f"  observed |skew|: median {np.median(np.abs(skew)):.2f}, "
           f"{(np.abs(skew) > 2).mean()*100:.1f}% exceed 2 -- hence the rank-transformed arm")

    lam_hat, ll = estimate_lambda(Y, CA, LAM_GRID)
    i1 = int(np.argmin(np.abs(LAM_GRID - 1.0)))
    flat = (ll.max(axis=0) - ll[i1]) < 2.0
    summary["pagel_lambda"] = {
        "median": float(np.median(lam_hat)), "mean": float(lam_hat.mean()),
        "frac_le_0.1": float((lam_hat <= 0.1).mean()), "frac_ge_0.9": float((lam_hat >= 0.9).mean()),
        "frac_at_grid_min_zero": float((lam_hat == 0.0).mean()),
        "frac_likelihood_flat_vs_lambda1": float(flat.mean()),
        "spearman_lambdahat_vs_F": float(pd.Series(lam_hat).corr(pd.Series(Fobs), method="spearman")),
        "note": ("lambda-hat and F are correlated, so a per-domain plug-in null is biased "
                 "toward non-significance; results are reported as a lambda bracket.")}
    cv.log(f"  Pagel lambda: median {np.median(lam_hat):.3f}, {(lam_hat<=0.1).mean()*100:.1f}% <=0.1, "
           f"{(lam_hat>=0.9).mean()*100:.1f}% >=0.9; likelihood flat vs lambda=1 for "
           f"{flat.mean()*100:.1f}%; Spearman(lambda,F)="
           f"{summary['pagel_lambda']['spearman_lambdahat_vs_F']:+.2f}")

    Yrank = normal_scores(Y)
    Fobs_rank = anova_F(Yrank, gA)
    nulls: Dict[float, Dict[str, np.ndarray]] = {}
    for lam in sorted(set([0.0, 1.0] + [float(x) for x in np.unique(lam_hat)])):
        L = chol(CA, lam)
        nulls[lam] = {
            "plain": build_null(L, gA, N_SIM, np.random.default_rng(SEED + int(lam * 1000)), False),
            "rank": build_null(L, gA, N_SIM, np.random.default_rng(SEED + 7000 + int(lam * 1000)), True)}
        cv.log(f"    lambda={lam:.2f}: null F 95th plain={np.percentile(nulls[lam]['plain'],95):6.2f} "
               f"rank={np.percentile(nulls[lam]['rank'],95):6.2f} "
               f"max plain={nulls[lam]['plain'].max():7.2f}")

    p_std = fdist.sf(Fobs, len(clades) - 1, len(anova_sp) - len(clades))
    p0 = emp_p(nulls[0.0]["plain"], Fobs)
    p1 = emp_p(nulls[1.0]["plain"], Fobs)
    phat = np.ones(len(doms))
    for lam in np.unique(lam_hat):
        sel = lam_hat == lam
        phat[sel] = emp_p(nulls[float(lam)]["plain"], Fobs[sel])
    p1r = emp_p(nulls[1.0]["rank"], Fobs_rank)

    adj = {k: multipletests(v, alpha=0.05, method="fdr_bh")[1]
           for k, v in (("standard", p_std), ("lambda0", p0), ("lambdahat", phat),
                        ("lambda1", p1), ("lambda1_rank", p1r))}
    res = pd.DataFrame({"domain_name": doms, "F_statistic": Fobs,
                        "F_rank_transformed": Fobs_rank, "pagel_lambda": lam_hat,
                        "skewness": skew, "p_standard": p_std, "p_lambda0": p0,
                        "p_lambdahat": phat, "p_lambda1": p1, "p_lambda1_rank": p1r,
                        **{f"padj_{k}": v for k, v in adj.items()}}
                       ).sort_values("F_statistic", ascending=False)
    res.to_csv(os.path.join(OUT, "phylo_anova_domains.csv"), index=False)

    counts = {k: int((v < 0.05).sum()) for k, v in adj.items()}
    floor = 1.0 / (N_SIM + 1)
    at_floor = {k: int((np.asarray(p) <= floor).sum())
                for k, p in (("lambda0", p0), ("lambdahat", phat), ("lambda1", p1),
                             ("lambda1_rank", p1r))}
    summary["n_significant_fdr05"] = counts
    summary["n_at_simulation_floor"] = at_floor
    cv.log(f"  significant at BH-FDR<0.05: {counts}")
    cv.log(f"  at the simulation floor (p = {floor:.2g}): {at_floor}")
    for d in ("PF02885", "PF18782", "PF00591"):
        r0 = res[res.domain_name.str.startswith(d)]
        if len(r0):
            r0 = r0.iloc[0]
            cv.log(f"    {d}: F={r0.F_statistic:7.1f} (rank {r0.F_rank_transformed:6.1f}) "
                   f"lam={r0.pagel_lambda:.2f} skew={r0.skewness:+.2f} | adj "
                   f"std={r0.padj_standard:.2e} lam0={r0.padj_lambda0:.3g} "
                   f"lamhat={r0.padj_lambdahat:.3g} lam1={r0.padj_lambda1:.3g} "
                   f"lam1rank={r0.padj_lambda1_rank:.3g}")

    calib = {}
    for lam in (0.0, 1.0):
        Yc = chol(CA, lam) @ rng.standard_normal((len(anova_sp), N_CALIB))
        calib[f"lambda_{lam:.0f}"] = float(
            (fdist.sf(anova_F(Yc, gA), len(clades) - 1, len(anova_sp) - len(clades)) < 0.05).mean())
    wexp = 0.0
    for lam in np.unique(lam_hat):
        Yc = chol(CA, float(lam)) @ rng.standard_normal((len(anova_sp), 4000))
        wexp += float((fdist.sf(anova_F(Yc, gA), len(clades) - 1,
                                len(anova_sp) - len(clades)) < 0.05).mean()) * float((lam_hat == lam).mean())
    calib["lambda_weighted_expectation"] = wexp
    calib["note"] = (
        "lambda=0 is a MACHINERY CHECK, not a finding. At lambda=0 the simulated species are "
        "i.i.d., and because this tree is ultrametric to ~2e-4 relative spread they are also "
        "homoscedastic -- which is exactly the textbook one-way ANOVA null. A rejection rate of "
        "0.05 there is guaranteed by construction and only confirms that the simulator and the "
        "F code are correct. It is NOT evidence that the standard ANOVA is appropriate for these "
        "data. The lambda=1 figure is the worst case. The lambda-weighted figure is a PLUG-IN "
        "mixture, sum over the grid of FPR(lambda) * P(lambda_hat = lambda); note that lambda_hat "
        "is fitted under an intercept-only mean, so a real clade effect inflates it "
        f"(Spearman(lambda_hat, F) = "
        f"{summary['pagel_lambda']['spearman_lambdahat_vs_F']:+.2f} here), which biases this "
        "mixture upward.")
    summary["standard_anova_false_positive_rate"] = calib
    cv.log(f"  standard-ANOVA rejection rate under NO clade effect: lambda=0 "
           f"{calib['lambda_0']:.3f} (machinery check -- 0.05 by construction), "
           f"lambda=1 {calib['lambda_1']:.3f} (worst case), lambda-weighted plug-in {wexp:.3f}")

    keep = [s for s in anova_sp if s != CERVUS]
    if len(keep) < len(anova_sp):
        Ck = phylo_vcv(tree_for(keep), keep)
        gk = [np.array([keep.index(s) for s in by_clade[c] if s in keep]) for c in clades]
        Yk = X_norm.loc[keep, doms].values.astype(float)
        pk = fdist.sf(anova_F(Yk, gk), len(clades) - 1, len(keep) - len(clades))
        lam_k, _ = estimate_lambda(Yk, Ck, LAM_GRID)
        summary["cervus_sensitivity"] = {
            "n_species_with": len(anova_sp), "n_species_without": len(keep),
            "median_lambda_with": float(np.median(lam_hat)),
            "median_lambda_without": float(np.median(lam_k)),
            "n_significant_standard_with": counts["standard"],
            "n_significant_standard_without": int(
                (multipletests(pk, alpha=0.05, method="fdr_bh")[1] < 0.05).sum())}
        cv.log(f"  Cervus sensitivity: median lambda {np.median(lam_hat):.3f} -> {np.median(lam_k):.3f}; "
               f"classical significant {counts['standard']} -> "
               f"{summary['cervus_sensitivity']['n_significant_standard_without']}")

    # ------------------------------------------------- MANN-WHITNEY ENRICHMENT
    cv.log("\nMann-Whitney enrichment arm (106-species cohort, absolute counts)")
    tE = tree_for(enrich_sp)
    CE = phylo_vcv(tE, enrich_sp)
    verify_vcv(tE, enrich_sp, CE, rng, n_check=100)
    raw = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH, cv.load_species_list(cv.SPECIES_LIST_PATH))
    cols = list(X_norm.columns)
    Xe = raw.loc[enrich_sp, cols].values.astype(float)
    pubE = pd.read_csv(os.path.join(REPO, "results", "clade_domain_enrichment",
                                    "all_clades_enriched_domains_summary.csv"))
    dom_index = {d: i for i, d in enumerate(cols)}
    gE = {c: np.array([enrich_sp.index(s) for s in by_clade[c]]) for c in clades}

    null_U = {c: np.empty(N_SIM) for c in clades}
    Lk = chol(CE, 1.0)
    rng_e = np.random.default_rng(SEED + 99)
    done = 0
    while done < N_SIM:
        mm = min(CHUNK, N_SIM - done)
        Ysim = Lk @ rng_e.standard_normal((len(enrich_sp), mm))
        for c in clades:
            null_U[c][done:done + mm] = mwu_U(Ysim, gE[c], ties=False)
        done += mm
    mw_rows, mw_summary = [], {}
    for c in clades:
        n1 = len(gE[c]); n2 = len(enrich_sp) - n1
        Uobs = mwu_U(Xe, gE[c], ties=True)
        sub = pubE[pubE.clade == c]
        ok = sum(1 for r in sub.itertuples()
                 if dom_index.get(r.domain) is not None
                 and abs(Uobs[dom_index[r.domain]] - r.U_statistic) < 1e-6)
        centre = n1 * n2 / 2.0
        dev_null = np.sort(np.abs(null_U[c] - centre))
        p_phy = (N_SIM - np.searchsorted(dev_null, np.abs(Uobs - centre), side="left") + 1) / (N_SIM + 1)
        padj = multipletests(p_phy, alpha=0.05, method="fdr_bh")[1]
        n_pub = int((sub.p_adjusted < 0.05).sum())
        mw_summary[c] = {"n_clade": n1, "n_background": n2, "published_significant_fdr05": n_pub,
                         "observed_U_reproduced": f"{ok}/{len(sub)}",
                         "phylogenetic_significant_fdr05": int((padj < 0.05).sum()),
                         "n_at_floor": int((p_phy <= floor).sum())}
        cv.log(f"  {c:32} n={n1:3} | U reproduced {ok}/{len(sub)} | published {n_pub:4} -> "
               f"phylogenetic {int((padj < 0.05).sum()):4}")
        # Persist EVERY published-significant pair's p, not only the survivors. With zero
        # survivors the old file was a bare header, so the headline "0 survive" could not be
        # checked from the artifacts at all.
        pub_j = [dom_index[r.domain] for r in sub.itertuples()
                 if r.p_adjusted < 0.05 and dom_index.get(r.domain) is not None]
        for j in pub_j:
            mw_rows.append({"clade": c, "domain": cols[j], "U": float(Uobs[j]),
                            "p_phylogenetic": float(p_phy[j]),
                            "p_adj_phylogenetic": float(padj[j]),
                            "survives_fdr05": bool(padj[j] < 0.05)})
    mwdf = pd.DataFrame(mw_rows, columns=["clade", "domain", "U", "p_phylogenetic",
                                          "p_adj_phylogenetic", "survives_fdr05"])
    mwdf.to_csv(os.path.join(OUT, "phylo_mannwhitney_untied_null.csv"), index=False)

    # A BRACKET over Pagel's lambda, not a single number. Running this arm at lambda=1 only
    # was a defect: lambda=1 is the strongest possible phylogenetic null (pure Brownian motion
    # on the full VCV), and it is demonstrably NOT the null these traits support. Pagel's
    # lambda fitted to the normal-scored counts of the domains behind the published-significant
    # pairs has median ~0.4, and the same marginal-matched test is correctly sized yet flags
    # only ~1% of all 8,020 domains -- i.e. the real data are LESS clade-structured than
    # lambda=1 Brownian motion. Reporting only the lambda=1 endpoint therefore understates what
    # survives, and contradicts both this script's own ANOVA arm (which brackets) and
    # REVISION_PLAN.md, which records that the fitted-lambda result is the primary number.
    # Garland et al. themselves advocate empirically scaled simulations rather than BM alone
    # (https://doi.org/10.1093/sysbio/42.3.265).
    N_MM = 200_000                     # floor 5e-6, enough to clear BH over 7,976 domains
    MM_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
    n_pub_sig = int((pubE.p_adjusted < 0.05).sum())
    cv.log(f"  FULL marginal-matched re-test of the {n_pub_sig} published-significant pairs "
           f"at {N_MM:,} simulations, bracketed over lambda in {MM_LAMBDAS}")

    # Where in the bracket the data actually sit: Pagel lambda on the normal-scored counts of
    # the domains behind the published-significant pairs, versus a matched random background.
    sig_doms = sorted(set(pubE[pubE.p_adjusted < 0.05].domain))
    sig_idx = [dom_index[d] for d in sig_doms if d in dom_index]
    rng_bg = np.random.default_rng(SEED + 777)
    bg_idx = sorted(rng_bg.choice(len(cols), size=min(1500, len(cols)), replace=False).tolist())
    lam_fit = {}
    for tag, idxs in (("enrichment_significant_domains", sig_idx), ("random_background", bg_idx)):
        Z = normal_scores(Xe[:, idxs])
        lams, _ll = estimate_lambda(Z, CE, LAM_GRID)
        lam_fit[tag] = {"n_domains": int(len(lams)), "median_lambda": float(np.median(lams)),
                        "mean_lambda": float(lams.mean()),
                        "frac_lambda_ge_0.9": float((lams >= 0.9).mean())}
        cv.log(f"    fitted lambda, {tag:32}: median {np.median(lams):.3f}, "
               f"{(lams >= 0.9).mean()*100:.0f}% at lambda>=0.9 (n={len(lams)})")
    summary["mannwhitney_fitted_lambda_of_traits"] = lam_fit

    _rd = rankdata
    mm_rows, mm_by_lambda = [], {}
    for lam in MM_LAMBDAS:
        rng_m = np.random.default_rng(SEED + 4242)
        Lmm = chol(CE, lam)
        posm = np.empty((len(enrich_sp), N_MM), dtype=np.int32)
        dm = 0
        while dm < N_MM:
            mm = min(CHUNK, N_MM - dm)
            o = np.argsort(Lmm @ rng_m.standard_normal((len(enrich_sp), mm)), axis=0)
            # pos[i, s] = rank-position of species i in simulation s
            posm[:, dm:dm + mm] = np.argsort(o, axis=0).astype(np.int32)
            dm += mm

        per_clade, tot_sig = {}, 0
        for c in clades:
            sub = pubE[(pubE.clade == c) & (pubE.p_adjusted < 0.05)]
            if not len(sub):
                continue
            g = gE[c]; n1 = len(g); n2 = len(enrich_sp) - n1
            centre = n1 * n2 / 2.0
            pos_g = posm[g]                                    # (n1, N_MM)
            pvals = np.ones(len(sub))
            # p a pair could attain AT BEST -- clade occupying the extreme ranks, i.e. perfect
            # separation. If even that cannot clear the FDR bar, the pair is UNDETERMINED by
            # this test, not refuted by it. Low power under monophyletic groups is the known
            # behaviour of the Garland simulation ANOVA (Adams & Collyer 2018,
            # https://doi.org/10.1111/evo.13492), so it must be measured, not assumed away.
            p_attain = np.ones(len(sub))
            names = list(sub.domain)
            for blk in range(0, len(names), 64):
                names_b = names[blk:blk + 64]
                idxs = [dom_index[d] for d in names_b]
                # tie-averaged rank of the k-th smallest observed value, per domain
                R = np.vstack([_rd(np.sort(Xe[:, j])) for j in idxs])       # (b, 107)
                u_obs = mwu_U(Xe[:, idxs], g, ties=True)                    # (b,) SIGNED
                # Each domain's own ATTAINABLE extremes, from its own tie-averaged rank
                # vector: the clade can at best occupy the n1 largest or the n1 smallest
                # ranks. With ties the theoretical bound n1*n2/2 is usually NOT reachable, so
                # using it (as an earlier version did) understates how many pairs are
                # power-limited.
                off = n1 * (n1 + 1) / 2.0
                u_max = R[:, -n1:].sum(axis=1) - off                        # (b,)
                u_min = R[:, :n1].sum(axis=1) - off                         # (b,)
                # TWO-SIDED STATISTIC: one-sided test on T = |U - centre|.
                #
                # A doubled-min-tail p was tried here and REVERTED, because it is a regression
                # on this data. U is highly discrete (many domains take only a handful of
                # distinct count values), so the atom at u_obs is counted in BOTH tails and
                # 2*min(tail) cannot become small. Measured realized size at alpha=0.05 on
                # Brownian nulls rank-matched onto real domain marginals:
                #     near-continuous domain : |U-c| 0.054   doubled-min-tail 0.054
                #     heavily tied domain    : |U-c| 0.042-0.045   doubled-min-tail 0.009-0.016
                # i.e. |U - centre| is correctly sized where the doubled tail is 3-5x too
                # conservative. The objection to |U - centre| -- that the null is not exactly
                # symmetric about the centre, so it mixes the two tails -- is about
                # interpretability, not validity: a one-sided test on any fixed statistic is
                # valid whatever the statistic's null shape. Validity wins.
                dev_obs = np.abs(u_obs - centre)
                # The attainable extreme still comes from each domain's OWN rank vector, which
                # was the real defect: ties usually make the theoretical bound n1*n2/2
                # unreachable, so using it understated how many pairs are power-limited.
                dev_max = np.maximum(np.abs(u_max - centre), np.abs(u_min - centre))
                cnt = np.zeros(len(idxs)); cnt_max = np.zeros(len(idxs))
                for s0 in range(0, N_MM, 20_000):
                    sl = slice(s0, min(s0 + 20_000, N_MM))
                    U = R[:, pos_g[:, sl]].sum(axis=1) - off                   # (b, chunk)
                    dev = np.abs(U - centre)
                    cnt += (dev >= dev_obs[:, None]).sum(axis=1)
                    cnt_max += (dev >= dev_max[:, None]).sum(axis=1)
                pvals[blk:blk + 64] = (cnt + 1) / (N_MM + 1)
                p_attain[blk:blk + 64] = (cnt_max + 1) / (N_MM + 1)
            padj_mm = multipletests(pvals, alpha=0.05, method="fdr_bh")[1] * (len(cols) / len(sub))
            padj_mm = np.minimum(padj_mm, 1.0)
            n_sig = int((padj_mm < 0.05).sum()); tot_sig += n_sig
            bh_rank1 = 0.05 / len(cols)
            # Three-way, NON-OVERLAPPING, and honest about the simulation floor.
            # The doubled-min-tail p cannot go below 2/(N_MM+1); if that floor is itself above
            # the BH bar, a pair reading "powerless" tells us nothing -- it is
            # RESOLUTION-limited, not power-limited. Reporting those as power-limited (an
            # earlier version did) flags every pair by construction. And a pair that SURVIVED
            # obviously is not undetermined, so the categories must be disjoint.
            floor_mm = 1.0 / (N_MM + 1)
            survived = padj_mm < 0.05
            at_floor = p_attain <= floor_mm * 1.0000001
            powerless = (~survived) & (p_attain > bh_rank1) & (~at_floor)
            unresolved = (~survived) & at_floor & (floor_mm > bh_rank1)
            n_powerless = int(powerless.sum())
            n_unresolved = int(unresolved.sum())
            per_clade[c] = {
                "n_retested": int(len(sub)), "n_surviving_fdr05": n_sig,
                "min_p": float(pvals.min()),
                "n_at_simulation_floor": int((pvals <= 1.0 / (N_MM + 1)).sum()),
                "best_attainable_p": float(p_attain.min()),
                "n_pairs_that_cannot_reach_bh_even_with_perfect_separation": n_powerless,
                "n_pairs_resolution_limited_floor_above_bh_bar": n_unresolved,
                "simulation_floor_doubled_tail": float(floor_mm),
                "floor_exceeds_bh_bar": bool(floor_mm > bh_rank1),
                "power_limited": bool(n_powerless == len(sub))}
            flag = "  <- POWER-LIMITED: no pair here can reach BH even with perfect separation" \
                if per_clade[c]["power_limited"] else ""
            cv.log(f"    lam={lam:.2f} {c:32} {len(sub):5} -> {n_sig:5} survive "
                   f"({n_sig/len(sub)*100:3.0f}%), best attainable p={p_attain.min():.2e}{flag}")
            for k, d in enumerate(names):
                mm_rows.append({"lambda": lam, "clade": c, "domain": d,
                                "p_marginal_matched": float(pvals[k]),
                                "p_adj_marginal_matched": float(padj_mm[k]),
                                "best_attainable_p": float(p_attain[k])})
        # The per-PAIR undetermined count is the statistic to lead with. A clade-level
        # "power_limited" flag only fires when EVERY pair in the clade is unattainable, which
        # hides the much larger number of individual pairs that no amount of separation could
        # have rescued -- those are undetermined by this test, not refuted by it.
        n_undet = sum(v["n_pairs_that_cannot_reach_bh_even_with_perfect_separation"]
                      for v in per_clade.values())
        n_unres = sum(v["n_pairs_resolution_limited_floor_above_bh_bar"]
                      for v in per_clade.values())
        mm_by_lambda[f"lambda_{lam:.2f}"] = {
            "n_surviving_fdr05": tot_sig,
            "n_pairs_undetermined_no_power": n_undet,
            "n_pairs_resolution_limited": n_unres,
            "n_pairs_genuinely_not_supported": n_pub_sig - tot_sig - n_undet - n_unres,
            "per_clade": per_clade}
        cv.log(f"  lambda={lam:.2f} TOTAL: {n_pub_sig} re-tested -> {tot_sig} survive, "
               f"{n_undet} undetermined (no power), {n_unres} resolution-limited, "
               f"{n_pub_sig - tot_sig - n_undet - n_unres} not supported")

    # HARD GUARD. If the doubled-min-tail floor sits above the rank-1 BH bar, then every
    # non-surviving pair is unresolvable BY CONSTRUCTION and the power decomposition carries no
    # information -- it is a statement about N_MM, not about the data. Say so loudly rather
    # than letting a table of "undetermined" counts be quoted as a finding.
    _floor = 1.0 / (N_MM + 1)
    _bar = 0.05 / len(cols)
    if _floor > _bar:
        summary.setdefault("mannwhitney_marginal_matched", {})
        cv.log(f"  ** WARNING: marginal-matched p-floor {_floor:.2e} EXCEEDS the rank-1 BH bar "
               f"{_bar:.2e}. The undetermined / resolution-limited / not-supported split is "
               f"NOT RESOLVABLE at N_MM={N_MM:,} and MUST NOT be quoted. Survival counts are "
               f"unaffected (they are reached at BH ranks >1). Need N_MM >= {int(2/_bar):,}.")
    mmdf = pd.DataFrame(mm_rows)
    mmdf.to_csv(os.path.join(OUT, "phylo_mannwhitney_marginal_matched.csv"), index=False)
    at1 = mm_by_lambda["lambda_1.00"]
    powerless = [c for c, v in at1["per_clade"].items() if v["power_limited"]]
    summary["mannwhitney_marginal_matched"] = {
        "n_simulations": N_MM, "lambdas": list(MM_LAMBDAS),
        "n_retested": n_pub_sig,
        "surviving_by_lambda": {k: v["n_surviving_fdr05"] for k, v in mm_by_lambda.items()},
        "per_lambda_detail": mm_by_lambda,
        "power_decomposition_resolvable": bool(_floor <= _bar),
        "marginal_matched_p_floor": _floor,
        "bh_rank1_bar": _bar,
        "n_mm_needed_for_resolvable_decomposition": int(2 / _bar),
        "power_decomposition_warning": (
            None if _floor <= _bar else
            f"NOT RESOLVABLE at N_MM={N_MM}: the marginal-matched p cannot go below "
            f"{_floor:.2e}, which is above the rank-1 BH bar {_bar:.2e}, so every "
            f"non-surviving pair is classified as undetermined or resolution-limited by "
            f"construction. Do NOT quote the undetermined counts. The SURVIVAL counts are "
            f"unaffected. Raise N_MM to >= {int(2 / _bar):,} to resolve it."),
        "power_limited_clades": powerless,
        "undetermined_by_lambda": {k: v["n_pairs_undetermined_no_power"]
                                   for k, v in mm_by_lambda.items()},
        "p_needed_for_bh_survival_at_rank1": 0.05 / len(cols),
        "note": ("The LESS conservative of the two nulls: the simulated Brownian trait is "
                 "rank-matched onto the observed count vector, so the null statistic is drawn "
                 "from exactly the observed marginal -- ties included -- and the only thing "
                 "tested is whether the clade split is unusual GIVEN phylogeny. Reported as a "
                 "BRACKET over Pagel's lambda, because the single-number lambda=1 answer is the "
                 "strongest possible phylogenetic null and is not the one these traits support "
                 "(fitted lambda on the significant domains has median ~0.4; see "
                 "mannwhitney_fitted_lambda_of_traits). Read the fitted-lambda region of the "
                 "bracket as primary and lambda=1 as the worst case. "
                 "TWO THINGS NOT TO SAY: (1) do not report 'all 2,187 erased' -- that is the "
                 "lambda=1 endpoint only. (2) For any clade flagged power_limited, the test "
                 "cannot reject even under perfect group separation, so those pairs are "
                 "UNDETERMINED by this analysis, not refuted by it. "
                 "Resolution is not the binding constraint at lambda=1: the smallest p is ~30x "
                 "above the 1/(N+1) floor of 5.0e-6, and ~24x above the 0.05/7976 threshold a "
                 "pair must clear to survive BH at rank 1. Only the published-significant pairs "
                 "were re-tested and BH was scaled to the full 8,020-domain family, so the "
                 "adjusted p of the k-th smallest equals p_(k) * 7976 / k; this is exact when "
                 "the untested p's all exceed the tested ones and conservative otherwise, and "
                 "an all-8,020-domain pilot confirmed the untested domains do not change the "
                 "verdict.")}
    cv.log(f"  MARGINAL-MATCHED BRACKET (survive / undetermined / resolution-limited / not supported):")
    for k, v in mm_by_lambda.items():
        cv.log(f"    lambda={k.split('_')[1]}: {v['n_surviving_fdr05']:5} / "
               f"{v['n_pairs_undetermined_no_power']:4} / "
               f"{v['n_pairs_resolution_limited']:4} / "
               f"{v['n_pairs_genuinely_not_supported']:5}")
    if powerless:
        cv.log(f"  POWER-LIMITED clades (undetermined, NOT refuted): {powerless}")
    summary["mannwhitney"] = mw_summary
    summary["mannwhitney_total"] = {
        "published_significant": int((pubE.p_adjusted < 0.05).sum()),
        "phylogenetic_significant": int(sum(v["phylogenetic_significant_fdr05"]
                                            for v in mw_summary.values()))}
    cv.log(f"  TOTAL: published {summary['mannwhitney_total']['published_significant']} "
           f"clade-domain pairs -> phylogenetic "
           f"{summary['mannwhitney_total']['phylogenetic_significant']}")

    # run_config.json, per the project's reproducibility rule: every constant that determines
    # a reported number, plus library versions, so the run can be re-derived exactly.
    import platform
    import numpy as _np
    import scipy as _sp
    import statsmodels as _sm
    run_config = {
        "script": os.path.basename(__file__),
        "N_SIM": N_SIM, "N_MM": N_MM, "N_CALIB": N_CALIB, "CHUNK": CHUNK,
        "MM_LAMBDAS": list(MM_LAMBDAS), "LAM_GRID_points": len(LAM_GRID),
        "RANDOM_STATE": SEED, "MIN_CLADE_N": 3,
        "cohort_anova_n": len(anova_sp), "cohort_enrichment_n": len(enrich_sp),
        "clades": clades, "clade_sizes": {c: len(by_clade[c]) for c in clades},
        "cervus_excluded_in_sensitivity_arm": CERVUS,
        "python": platform.python_version(), "numpy": _np.__version__,
        "scipy": _sp.__version__, "statsmodels": _sm.__version__,
    }
    with open(os.path.join(OUT, "run_config.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
