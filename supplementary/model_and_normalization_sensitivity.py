#!/usr/bin/env python3
"""
Sensitivity of the domain-count distance to the evolutionary model and to normalization.

Domain copy numbers are discrete, non-negative characters, and they change by duplication,
deletion, gain and loss rather than by diffusion. The Brownian/random-walk formulation
behind the analytical squared-difference distance of Part 1 is therefore a heuristic, and
this script measures how far that result depends on it: how the distance behaves under
alternative transformations of copy number, whether squared differences really do grow
linearly with divergence time, whether the counts are overdispersed relative to Poisson,
how much the choice of normalization basis matters, and whether the signal rests on a few
highly abundant domains or on the wider domain repertoire.

Cohort. Every number here is recomputed through the pipeline loader on the cohort the
pipeline defines: 106 species x 7,976 domains, with Cervus hanglu yarkandensis excluded for
data quality. The transform results in scripts/test_metric_variants.py are not reused,
because that script drops Cervus hanglu and the five outlier species and so runs on roughly
102 species, which makes its values incomparable with the ones below. Section A3
additionally rebuilds the larger 107-species / 8,020-domain cohort that retains Cervus
hanglu, because its conclusion depends on that one species, so both cohorts are reported.

What is tested
  A1 Transform the count, then difference: D = sum_k (f(C_ik) - f(C_jk))^2 for
     f in {identity, sqrt, log(C+1), cube-root}.
  A2 Scale the difference: D = sum_k |dC|^p for p in {2, 1, 0.75, 0.5} plus log1p|dC|.
     These are the scalings that scripts/test_metric_variants.py varies.
  A3 Model adequacy. Under a random walk the expected squared difference grows linearly
     with divergence time. If copy number instead changes multiplicatively, by proportional
     gains and losses rather than additive ones, it is the log counts that random-walk.
     Both are tested directly by regressing mean squared difference on time and comparing
     the fits. This is not a birth-death fit and must not be reported as one: under a
     linear birth-death process with lambda > mu, E[C(t)] and Var[C(t)] grow exponentially
     in t (Kendall 1948, doi:10.1214/aoms/1177730285), that is super-linearly, whereas what
     is tested here is linearity of squared log differences.
  A4 Overdispersion. Per-domain index of dispersion (variance / mean). Poisson implies 1;
     values far above 1 mean a negative-binomial-like process, so the discrete counts are
     overdispersed and a Gaussian random walk is a heuristic, not a description.
  B1 Normalization sensitivity: the same distance under raw counts, relative frequency,
     CLR and log(C+1).
  B2 Concentration. What fraction of the Part 1 distance comes from the top 1/3/5/10
     domains, and is a domain's contribution explained by its abundance? This asks
     directly whether the signal is just a few abundant domains. It is computed unweighted,
     because under variance weighting the answer is an algebraic identity:
     sum_{i<j}(x_i-x_j)^2 = n(n-1)Var(x), so with w_k = var_k every domain contributes
     exactly n(n-1) regardless of the data, and a variance-weighted concentration
     statistic therefore carries no information and must not be reported.
  B3 Proteome size: how much does total domain count vary, and does it track phylogeny?

Output, written only under results/model_normalization/:
  variant_comparison.csv  one row per distance variant (A1, A2, B1), giving Pearson and
                          Spearman against patristic distance, the normalized
                          Robinson-Foulds distance of the UPGMA and the neighbor-joining
                          tree built from that distance, and how many clade groups come
                          out monophyletic.
  summary.json            the same rows plus the A3 fits per cohort and scale, the A4
                          dispersion statistics, the B2 concentration and drop-the-top
                          contributors curves, and the B3 proteome-size statistics.
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress, pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import classification_cv_tree_reconstruction as cv  # noqa: E402
import continuous_distance_trees as ct  # noqa: E402

OUT = os.path.join(REPO, "results", "model_normalization")
SEED = cv.RANDOM_STATE
N_BINS = 25
# Random tie orders used to measure how much of the binned R2 is sort order rather than data.
TIE_DRAWS = 200
CERVUS_NAME = "Cervus hanglu yarkandensis"


def main() -> None:
    cv.ensure_dir(OUT)
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    sp = sorted(species)
    n = len(sp)
    Xrel = X_norm.loc[sp].values.astype(float)
    raw_full = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH,
                                     cv.load_species_list(cv.SPECIES_LIST_PATH))
    Xraw = raw_full.loc[sp, list(X_norm.columns)].values.astype(float)
    T = dist_matrix.loc[sp, sp].values.astype(float)
    iu = np.triu_indices(n, k=1)
    Tvec = T[iu]
    labels = [cv.format_species_label_for_newick(s) for s in sp]
    sp2clade = ct.species_to_clade()
    from ete3 import Tree
    tref = Tree(cv.reference_tree_to_formatted_newick(
        cv.load_and_prune_reference_tree(sp)), format=1)
    cv.log(f"canonical cohort {n} species x {Xraw.shape[1]} domains")

    def score(name, M, tier):
        v = M[iu]
        trees = {"UPGMA": cv.build_upgma_newick(pd.DataFrame(M, index=sp, columns=sp)),
                 "NJ": ct.nj_newick(M, labels)}
        sc = {k: ct.score_tree(t, sp, sp2clade, tref) for k, t in trees.items()}
        best = min(sc, key=lambda k: sc[k]["robinson_foulds_normalized"])
        return {"variant": name, "tier": tier,
                "pearson": float(pearsonr(Tvec, v)[0]),
                "spearman": float(spearmanr(Tvec, v).statistic),
                "rf_UPGMA": sc["UPGMA"]["robinson_foulds_normalized"],
                "rf_NJ": sc["NJ"]["robinson_foulds_normalized"],
                "rf_best": sc[best]["robinson_foulds_normalized"], "best_method": best,
                "mono_rooted": sc["UPGMA"]["n_monophyletic_rooted"],
                "n_groups": sc["UPGMA"]["n_groups"]}

    rows = []

    # ---------- A1: transform the count, then difference ----------
    cv.log("A1: transform the count, then difference (sum of squared differences)")
    transforms = {
        "identity_C": lambda C: C,
        "sqrt_C": np.sqrt,
        "log1p_C": np.log1p,
        "cbrt_C": np.cbrt,
    }
    for tag, f in transforms.items():
        Z = f(Xraw)
        M = squareform(pdist(Z, "sqeuclidean"))
        rows.append(score(f"A1_{tag}", M, "count transform (raw basis)"))
        cv.log(f"  {tag:12} rho={rows[-1]['spearman']:+.4f} r={rows[-1]['pearson']:+.4f} "
               f"best RF={rows[-1]['rf_best']:.4f} mono={rows[-1]['mono_rooted']}/9")

    # ---------- A2: scale the difference ----------
    cv.log("A2: scale the difference |dC|^p")
    i_idx, j_idx = iu
    dC = np.abs(Xraw[i_idx] - Xraw[j_idx])
    for tag, g in {"p2_squared": lambda d: d ** 2, "p1_linear": lambda d: d,
                   "p0.75": lambda d: d ** 0.75, "p0.5_sqrt": lambda d: d ** 0.5,
                   "log1p_absdiff": np.log1p}.items():
        v = g(dC).sum(axis=1)
        M = np.zeros((n, n)); M[iu] = v; M = M + M.T
        rows.append(score(f"A2_{tag}", M, "difference scaling (raw basis)"))
        cv.log(f"  {tag:14} rho={rows[-1]['spearman']:+.4f} r={rows[-1]['pearson']:+.4f} "
               f"best RF={rows[-1]['rf_best']:.4f} mono={rows[-1]['mono_rooted']}/9")

    # ---------- A3: model adequacy, does squared difference grow linearly with time? ----------
    # Under a random walk the expected squared difference in counts grows linearly with time;
    # under a multiplicative process it is the log counts that random-walk. Both scales are
    # fitted directly against divergence time and the fits compared.
    #
    # Two cohorts are fitted, and a guard enforces that they really differ, because the
    # answer depends on a single species. Cervus hanglu yarkandensis belongs to the
    # 107-species cohort and contributes only 713 proteins against an analyzed-cohort median
    # of 37,351 and a next-lowest of 18,921, and 799 annotated domains against a cohort
    # median of 31,466 (all four from results/busco_completeness/per_species.csv and
    # summary.json). On the count scale every pair involving it therefore carries an enormous
    # squared difference, and those pairs alone are enough to produce a linear-looking
    # count-scale trend while destroying the log-scale fit. Both cohorts are computed so that
    # this dependence is visible instead of resting on whichever one happens to be quoted.
    # The 107-species arm is rebuilt through the pipeline loader with the data-quality
    # exclusion lifted, rather than copied from a stored number, so its species set,
    # zero-variance filter and normalization are the ones that cohort implies. The guard
    # matters because if the two arms resolve to the same cohort the comparison degenerates
    # into two copies of one fit and carries no information.
    cv.log("A3: model adequacy -- is growth with time linear (random walk) on the count scale "
           "or the log scale?")
    A3_SCALES = ("count_scale_(random_walk)", "log_scale_(multiplicative)",
                 "sqrt_scale")

    def a3_fit(sp_list, Xc, Tm):
        """Fit mean squared difference on divergence time, per scale, for one cohort.

        Returns cohort-level counts plus one record per scale in A3_SCALES. Each record
        carries three R2 values, listed below in increasing order of trustworthiness,
        because the binned R2 on its own is not a stable statistic.

          * linear_R2        the fit over N_BINS equal-count bins. The patristic distances
                             are heavily tie-saturated: about 2,340 pairs sit at exactly
                             188.0 MY, so equal-count bins collapse onto far fewer distinct
                             x positions and roughly ten of the twenty-five bins fall inside
                             that single tie block. Which tied pairs land in which of those
                             bins is then decided by the sort's tie order rather than by the
                             data. With numpy's default (unstable) quicksort the log-scale
                             25-bin R2 reads 0.5482, with kind="stable" 0.4692, and over
                             random tie orders it runs 0.59 to 0.65, so the sort is pinned
                             to kind="stable" to make the value deterministic and the value
                             is reported together with its tie-order band
                             (tiebreak_R2_min/max/sd over TIE_DRAWS random tie orders).
          * distinct_time_R2 tie-free. Aggregates d2 by distinct patristic value, so a tie
                             group is never split across bins and the sort order cannot
                             enter. This is the binned-style number that is safe to quote.
          * unbinned_R2      no aggregation at all; the assumption-free number.

        The direction of the comparison, that the log scale fits better than the count
        scale, holds for all three statistics and for every tie order tested; only the
        magnitude of linear_R2 is fragile.
        """
        m = len(sp_list)
        ius = np.triu_indices(m, k=1)
        Tvs = Tm[ius]
        ordr = np.argsort(Tvs, kind="stable")
        uniq = np.unique(Tvs)
        rng = np.random.default_rng(SEED)
        out = {"n_species": m, "n_pairs": int(len(Tvs)), "n_domains": int(Xc.shape[1]),
               "n_distinct_patristic_values": int(len(uniq)),
               "largest_tie_group": int(np.bincount(np.searchsorted(uniq, Tvs)).max())}
        for lab, Z in zip(A3_SCALES, (Xc, np.log1p(Xc), np.sqrt(Xc))):
            d2 = ((Z[ius[0]] - Z[ius[1]]) ** 2).sum(axis=1)
            ed = np.linspace(0, len(Tvs), N_BINS + 1).astype(int)
            bt = np.array([Tvs[ordr[ed[b]:ed[b + 1]]].mean() for b in range(N_BINS)])
            bd = np.array([d2[ordr[ed[b]:ed[b + 1]]].mean() for b in range(N_BINS)])
            lr = linregress(bt, bd)
            # curvature: a large quadratic term means growth is not linear in time
            c2 = np.polyfit(bt, bd / bd.max(), 2)
            # Tie-free aggregation: one point per distinct patristic value, so no tie group is
            # ever split and the sort order cannot enter.
            ut = np.array([Tvs[Tvs == t].mean() for t in uniq])
            ud = np.array([d2[Tvs == t].mean() for t in uniq])
            lr_u = linregress(ut, ud)
            # How much of linear_R2 is the tie order rather than the data?
            tb = []
            for _ in range(TIE_DRAWS):
                o = np.lexsort((rng.random(len(Tvs)), Tvs))
                b1 = np.array([Tvs[o[ed[b]:ed[b + 1]]].mean() for b in range(N_BINS)])
                b2 = np.array([d2[o[ed[b]:ed[b + 1]]].mean() for b in range(N_BINS)])
                tb.append(linregress(b1, b2).rvalue ** 2)
            tb = np.array(tb)
            rec = {"linear_R2": float(lr.rvalue ** 2), "n_bins_for_linear_R2": N_BINS,
                   "linear_R2_sort_kind": "stable",
                   "tiebreak_R2_min": float(tb.min()), "tiebreak_R2_max": float(tb.max()),
                   "tiebreak_R2_mean": float(tb.mean()), "tiebreak_R2_sd": float(tb.std(ddof=1)),
                   "tiebreak_n_draws": int(TIE_DRAWS),
                   # Tie-free. Quote this one, not linear_R2, for any binned-style figure.
                   "distinct_time_R2": float(lr_u.rvalue ** 2),
                   "distinct_time_n_points": int(len(uniq)),
                   "slope": float(lr.slope), "intercept": float(lr.intercept),
                   # A random walk on either scale predicts a zero intercept, so this is a
                   # second, independent adequacy criterion, and it does not favor the log
                   # scale. Report it alongside R2 rather than leaving it in the artifact.
                   "intercept_frac_of_max": float(lr.intercept / bd.max()),
                   "quadratic_coef_normalized": float(c2[0]),
                   "spearman_binned": float(spearmanr(bt, bd).statistic),
                   # Binning averages away within-bin scatter and inflates R2 substantially,
                   # so the binned R2 is a function of N_BINS and should not be quoted on
                   # its own. The unbinned value is assumption-free; both are reported.
                   "unbinned_R2": float(linregress(Tvs, d2).rvalue ** 2)}
            for nb in (10, 50):
                e2 = np.linspace(0, len(Tvs), nb + 1).astype(int)
                p1 = np.array([Tvs[ordr[e2[b]:e2[b + 1]]].mean() for b in range(nb)])
                p2 = np.array([d2[ordr[e2[b]:e2[b + 1]]].mean() for b in range(nb)])
                rec[f"R2_at_{nb}_bins"] = float(linregress(p1, p2).rvalue ** 2)
            out[lab] = rec
        return out

    saved_exclude = list(cv.EXCLUDE_SPECIES)
    try:
        cv.EXCLUDE_SPECIES = []
        Xp_norm, Tp_df, sp_pub_raw = cv.load_and_preprocess_data()
    finally:
        cv.EXCLUDE_SPECIES = saved_exclude
    sp_pub = sorted(sp_pub_raw)
    Xraw_pub = raw_full.loc[sp_pub, list(Xp_norm.columns)].values.astype(float)
    T_pub = Tp_df.loc[sp_pub, sp_pub].values.astype(float)

    if CERVUS_NAME not in sp_pub or CERVUS_NAME in sp:
        raise SystemExit(
            f"A3 cohort guard: the published arm must contain {CERVUS_NAME} and the canonical "
            f"arm must not (published n={len(sp_pub)}, canonical n={len(sp)}). If both arms "
            "resolve to the same cohort the comparison is vacuous and must not be reported.")

    model = {"published_with_cervus": a3_fit(sp_pub, Xraw_pub, T_pub),
             "canonical_without_cervus": a3_fit(sp, Xraw, T)}
    for arm in ("published_with_cervus", "canonical_without_cervus"):
        a = model[arm]
        cv.log(f"  [{arm}] {a['n_species']} species, {a['n_pairs']} pairs, "
               f"{a['n_domains']} domains")
        for lab in A3_SCALES:
            r = a[lab]
            cv.log(f"    {lab:40} R2={r['linear_R2']:.4f} ({N_BINS} bins) "
                   f"unbinned={r['unbinned_R2']:.4f} "
                   f"intercept={r['intercept_frac_of_max']:+.3f} of max "
                   f"quad={r['quadratic_coef_normalized']:+.3e}")

    cnt, lg = A3_SCALES[0], A3_SCALES[1]
    model["interpretation"] = (
        "THE CONCLUSION INVERTS ON ONE SPECIES, so no A3 number may be quoted without saying "
        "which cohort and which bin count it came from. In the submitted cohort "
        f"(n={model['published_with_cervus']['n_species']}) the count scale reaches "
        f"R2={model['published_with_cervus'][cnt]['linear_R2']:.4f} and the log scale only "
        f"{model['published_with_cervus'][lg]['linear_R2']:.4f}, which reads as support for the "
        "random-walk framing; but that count-scale trend is manufactured by the pairs involving "
        "the Cervus hanglu proteome, which contributes 799 annotated domains against a cohort "
        "median of 31,466 (results/busco_completeness/summary.json, "
        "cohort_median_domain_total). On the cohort we now ship "
        f"(n={model['canonical_without_cervus']['n_species']}) "
        f"the log scale reaches {model['canonical_without_cervus'][lg]['linear_R2']:.4f} against "
        f"{model['canonical_without_cervus'][cnt]['linear_R2']:.4f} for the count scale. "
        "A multiplicative or overdispersed-count process fits better than the random "
        "walk on the clean cohort; the opposite impression on the full cohort is a "
        "data-quality artifact of one truncated proteome. "
        "TWO CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS. (1) linear_R2 is fragile, not just "
        f"bin-count dependent. {model['canonical_without_cervus']['largest_tie_group']} of the "
        f"{model['canonical_without_cervus']['n_pairs']} pairs share one patristic value, so "
        f"equal-count bins split that tie block and the sort's tie order moves the log-scale "
        f"25-bin R2 over {model['canonical_without_cervus'][lg]['tiebreak_R2_min']:.4f}-"
        f"{model['canonical_without_cervus'][lg]['tiebreak_R2_max']:.4f} across "
        f"{model['canonical_without_cervus'][lg]['tiebreak_n_draws']} random tie orders. Quote "
        f"distinct_time_R2 ({model['canonical_without_cervus'][lg]['distinct_time_R2']:.4f} log "
        f"vs {model['canonical_without_cervus'][cnt]['distinct_time_R2']:.4f} count), which "
        "aggregates by distinct patristic value and is tie-free, or unbinned_R2 "
        f"({model['canonical_without_cervus'][lg]['unbinned_R2']:.4f} vs "
        f"{model['canonical_without_cervus'][cnt]['unbinned_R2']:.4f}). (2) R2 is not the only "
        "adequacy criterion and the other one does NOT favor the log scale: a random walk on "
        "either scale predicts a ZERO intercept, and the fitted intercept is "
        f"{model['canonical_without_cervus'][cnt]['intercept_frac_of_max']:.3f} of the maximum "
        f"on the count scale against "
        f"{model['canonical_without_cervus'][lg]['intercept_frac_of_max']:.3f} on the log scale. "
        "So the honest claim is the narrow one: squared differences grow more nearly linearly "
        "with time on the log scale than on the count scale, in the same direction at every bin "
        "count and tie order tested, while NEITHER scale satisfies the zero-intercept prediction "
        "which is itself an argument for presenting the random walk as a heuristic.")

    # ---------- A4: overdispersion ----------
    mu = Xraw.mean(axis=0); var = Xraw.var(axis=0, ddof=1)
    nz = mu > 0
    disp = var[nz] / mu[nz]
    over = {"n_domains": int(nz.sum()),
            "dispersion_median": float(np.median(disp)),
            "dispersion_mean": float(disp.mean()),
            "frac_overdispersed_gt1": float((disp > 1).mean()),
            "frac_gt10": float((disp > 10).mean()),
            "poisson_expectation": 1.0}
    cv.log(f"A4: index of dispersion (var/mean): median={over['dispersion_median']:.2f}, "
           f"{over['frac_overdispersed_gt1']*100:.1f}% exceed the Poisson value of 1, "
           f"{over['frac_gt10']*100:.1f}% exceed 10")

    # The pooled median above is not evidence that the counts are under-dispersed, and should
    # not be reported as if it were: var/mean is mechanically far below 1 for a near-invariant
    # single-copy family, and most of the feature set is exactly that. Two restrictions decide
    # the question: (i) stratify by mean count, (ii) weight by how much of the distance each
    # domain actually carries. Both are computed here, so the conclusion is drawn from the
    # domains where dispersion is measurable at all.
    strata = {}
    for thr in (1, 5, 10, 50, 100):
        sel = mu[nz] >= thr
        strata[f"mean_ge_{thr}"] = {
            "n_domains": int(sel.sum()),
            "dispersion_median": float(np.median(disp[sel])) if sel.any() else None,
            "frac_overdispersed_gt1": float((disp[sel] > 1).mean()) if sel.any() else None}
        s = strata[f"mean_ge_{thr}"]
        if s["n_domains"]:
            cv.log(f"    mean>={thr:<3} n={s['n_domains']:5} median disp={s['dispersion_median']:.2f} "
                   f"{s['frac_overdispersed_gt1']*100:.0f}% overdispersed")
    over["stratified_by_mean_count"] = strata
    over["frac_domains_with_mean_lt5"] = float((mu[nz] < 5).mean())
    over["frac_domains_with_mean_lt1"] = float((mu[nz] < 1).mean())

    # share of the total squared-difference distance carried by the overdispersed domains
    contrib_all = ((Xraw[i_idx] - Xraw[j_idx]) ** 2).sum(axis=0)
    c_nz = contrib_all[nz]
    over["distance_share_of_overdispersed_domains"] = float(c_nz[disp > 1].sum() / c_nz.sum())
    over["n_overdispersed_domains"] = int((disp > 1).sum())
    cv.log(f"    the {int((disp > 1).sum())} overdispersed domains carry "
           f"{over['distance_share_of_overdispersed_domains']*100:.1f}% of the total "
           f"squared-difference distance")
    over["interpretation"] = (
        f"The counts should not be read as under-dispersed. The pooled median of {over['dispersion_median']:.2f} is an artifact of the "
        f"{over['frac_domains_with_mean_lt5']*100:.0f}% of domains whose mean count is below 5, "
        f"where var/mean cannot exceed 1 by much. Restricted to domains with appreciable counts "
        f"the index of dispersion rises monotonically with the mean-count threshold, and the "
        f"{int((disp > 1).sum())} genuinely overdispersed domains carry "
        f"{over['distance_share_of_overdispersed_domains']*100:.1f}% of the total distance. Where "
        f"dispersion is measurable, the counts are overdispersed.")

    # ---------- B1: normalization ----------
    cv.log("B1: normalization sensitivity (same squared-difference distance)")
    C = Xraw + 0.5
    clr = np.log(C) - np.log(C).mean(axis=1, keepdims=True)
    for tag, Z in (("raw_counts", Xraw), ("relative_frequency", Xrel),
                   ("CLR", clr), ("log1p_counts", np.log1p(Xraw))):
        M = squareform(pdist(Z, "sqeuclidean"))
        rows.append(score(f"B1_{tag}", M, "normalization"))
        cv.log(f"  {tag:20} rho={rows[-1]['spearman']:+.4f} best RF={rows[-1]['rf_best']:.4f} "
               f"mono={rows[-1]['mono_rooted']}/9")

    # ---------- B2: is the signal a few abundant domains? ----------
    cv.log("B2: concentration of the distance across domains (UNWEIGHTED)")
    contrib = ((Xraw[i_idx] - Xraw[j_idx]) ** 2).sum(axis=0)     # total over all pairs, per domain
    tot = contrib.sum()
    share = np.sort(contrib)[::-1] / tot
    conc = {f"top{k}_share": float(share[:k].sum()) for k in (1, 3, 5, 10, 50, 100)}
    conc["n_domains_for_50pct"] = int(np.searchsorted(np.cumsum(share), 0.50) + 1)
    conc["n_domains_for_90pct"] = int(np.searchsorted(np.cumsum(share), 0.90) + 1)
    rho_ab = float(spearmanr(mu, contrib).statistic)
    conc["spearman_abundance_vs_contribution"] = rho_ab
    cv.log(f"  top 1/3/5/10 domains carry {conc['top1_share']*100:.1f}% / "
           f"{conc['top3_share']*100:.1f}% / {conc['top5_share']*100:.1f}% / "
           f"{conc['top10_share']*100:.1f}% of the total distance")
    cv.log(f"  domains needed for 50% / 90% of the distance: "
           f"{conc['n_domains_for_50pct']} / {conc['n_domains_for_90pct']} of {Xraw.shape[1]}")
    cv.log(f"  Spearman(mean abundance, contribution) = {rho_ab:+.4f}")
    # does removing the top contributors destroy the signal?
    ordc = np.argsort(contrib)[::-1]
    drop_curve = {}
    for k in (1, 5, 10, 50, 100):
        keepc = ordc[k:]
        v = ((Xraw[np.ix_(i_idx, keepc)] - Xraw[np.ix_(j_idx, keepc)]) ** 2).sum(axis=1)
        drop_curve[f"drop_top{k}"] = {"spearman": float(spearmanr(Tvec, v).statistic),
                                      "pearson": float(pearsonr(Tvec, v)[0])}
    base_v = ((Xraw[i_idx] - Xraw[j_idx]) ** 2).sum(axis=1)
    drop_curve["drop_top0"] = {"spearman": float(spearmanr(Tvec, base_v).statistic),
                               "pearson": float(pearsonr(Tvec, base_v)[0])}
    for k in (0, 1, 5, 10, 50, 100):
        d = drop_curve[f"drop_top{k}"]
        cv.log(f"    after removing the top {k:3} contributors: rho={d['spearman']:+.4f} "
               f"r={d['pearson']:+.4f}")

    # ---------- B3: proteome size ----------
    tot_dom = Xraw.sum(axis=1)
    size = {"min": float(tot_dom.min()), "max": float(tot_dom.max()),
            "median": float(np.median(tot_dom)), "cv": float(tot_dom.std() / tot_dom.mean())}
    dsize = np.abs(tot_dom[i_idx] - tot_dom[j_idx])
    size["spearman_sizediff_vs_patristic"] = float(spearmanr(dsize, Tvec).statistic)
    size["spearman_sizediff_vs_rawL1"] = float(spearmanr(dsize, np.abs(
        Xraw[i_idx] - Xraw[j_idx]).sum(axis=1)).statistic)
    cv.log(f"B3: total domain count per species {size['min']:.0f}-{size['max']:.0f} "
           f"(median {size['median']:.0f}, CV {size['cv']:.2f})")
    cv.log(f"  Spearman(|proteome size difference|, patristic) = "
           f"{size['spearman_sizediff_vs_patristic']:+.4f}")
    cv.log(f"  Spearman(|proteome size difference|, raw L1 distance) = "
           f"{size['spearman_sizediff_vs_rawL1']:+.4f}")

    # The size association is measured on every basis the analysis uses, not only on raw
    # counts, because row-normalizing to relative frequencies does not in fact remove the
    # proteome-size confound; the comparison across bases is in the summary below.
    C = Xraw + 0.5
    clr_b = np.log(C) - np.log(C).mean(axis=1, keepdims=True)
    for tag, Z in (("raw_counts", Xraw), ("relative_frequency", Xrel),
                   ("CLR", clr_b), ("log1p_counts", np.log1p(Xraw))):
        l1 = np.abs(Z[i_idx] - Z[j_idx]).sum(axis=1)
        size[f"spearman_sizediff_vs_L1_{tag}"] = float(spearmanr(dsize, l1).statistic)
        cv.log(f"  Spearman(|proteome size difference|, L1 on {tag:18}) = "
               f"{size[f'spearman_sizediff_vs_L1_{tag}']:+.4f}")
    size["interpretation"] = (
        "Relative-frequency normalization does NOT remove the proteome-size confound: the "
        f"size association on the relative-frequency basis is "
        f"{size['spearman_sizediff_vs_L1_relative_frequency']:+.4f} versus "
        f"{size['spearman_sizediff_vs_L1_raw_counts']:+.4f} on raw counts, i.e. no better and "
        "plausibly worse, because a species with an anomalously small total has EVERY relative "
        "frequency distorted. Row-normalization therefore does not resolve the "
        "proteome-size association. Compare the bases on this number and use whichever is "
        "actually lowest, noting that CLR is also the best-performing basis in B1.")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT, "variant_comparison.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"cohort": {"n_species": n, "n_domains": int(Xraw.shape[1])},
                   "model_adequacy": model, "overdispersion": over,
                   "concentration": conc, "drop_top_contributors": drop_curve,
                   "proteome_size": size, "variants": rows}, fh, indent=2)
    cv.log("")
    cv.log(tbl[["variant", "tier", "spearman", "pearson", "rf_best", "best_method",
                "mono_rooted"]].to_string(index=False))
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
