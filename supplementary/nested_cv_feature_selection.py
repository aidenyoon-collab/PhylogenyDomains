#!/usr/bin/env python3
"""
Out-of-sample validation of the correlation-ranked domain distance, with domain selection
moved inside the cross-validation loop, plus a permutation null that includes selection.

The analytical distance in domain_time_scatter.py ranks domains by the correlation between
their squared count difference and TimeTree patristic distance, keeps those with p < 0.05
and a positive regression slope, sums the top N of them into a distance, and reports the
correlation of that distance against the same patristic distances it selected on. That
correlation is therefore an optimized in-sample association rather than an out-of-sample
estimate. A Mantel test does not remove the optimism, because it permutes labels after the
panel has been chosen and so holds the selected panel fixed.

This script computes three things on the same cohort and count matrix.
  (A) Nested, species-blocked CV. Within each outer fold, domains are ranked and selected
      using training-training pairs only, the per-domain weights (slope or count variance)
      are taken from those same training pairs, and the resulting distance is scored on the
      held-out test-test pairs, which contain no training species at all. The held-out
      correlations are pooled across folds.
  (B) FDR selection. The same nested procedure with a Benjamini-Hochberg FDR < 0.05
      threshold in place of the nominal p < 0.05 threshold, so that the selection step
      accounts for testing thousands of domains.
  (C) Full-pipeline permutation null. Species labels of the count matrix are permuted and
      the whole procedure, ranking, selection and distance construction, is re-run, so the
      null distribution contains the selection step itself. That is the relevant reference
      distribution for a panel chosen from the data.

Held-out correlations are expected to fall below the in-sample values; the size of that
gap is the quantity of interest, and both are reported.

A vectorized re-implementation of the per-domain statistics is used, because the per-domain
loop in domain_time_scatter.compute_domain_stats is too slow to run inside a permutation
null. It is checked against that function on the full data before anything else runs, and
the script aborts if the two disagree.

Writes summary.json, permutation_draws.csv and nested_vs_insample.csv under
results/nested_cv_selection/.
"""
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))  # make the pipeline modules importable

import classification_cv_tree_reconstruction as cv  # noqa: E402
import domain_time_scatter as dts  # noqa: E402

OUT = os.path.join(REPO, "results", "nested_cv_selection")
SEED = cv.RANDOM_STATE
N_FOLDS = cv.N_FOLDS_DEFAULT
N_BEST = 50
N_PERM = 200
N_PERM_NESTED = 100
MIN_PREVALENCE = 0.05
EPSILON = dts.EPSILON        # 1e-8, the denominator guard the distance metric uses
# Reference correlations read from domain_time_correlations.tsv, row N=50: (spearman,
# pearson) for each weighting, on the 106-species cohort. The in-sample arm of this script
# re-runs that same selection and weighting, so it must reproduce these values; the
# assertion in main() enforces that before the nested run is trusted. The numbers are
# cohort-specific and have to be refreshed from the correlation table if the species set,
# the prevalence filter or N_BEST changes.
PUBLISHED_N50 = {"alpha": (0.279924, 0.728203), "var": (0.260518, 0.720405)}


def fast_domain_stats(X: np.ndarray, pairs: List[Tuple[int, int]],
                      T: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per-domain Pearson r, two-sided p, regression slope (alpha) and count variance of the
    squared count difference against patristic distance, vectorized over domains.

    y_k = (C_i,k - C_j,k)^2 for every pair; r_k = corr(y_k, T); alpha_k = cov(y_k,T)/var(T).

    The returned "var" column is the population variance of the per-species counts,
    var(C_.,k) with ddof=0. That is the definition the distance metric weights by, as
    implemented in domain_time_scatter.compute_domain_stats, and it is a different and much
    larger quantity than the variance of the squared pair differences, which would change
    the variance-weighted results if substituted. verify_fast_stats compares this column
    against the reference implementation for that reason.

    The p-values come from the usual t transform of r on n - 2 degrees of freedom, with
    |r| = 1 mapped to p = 0 and undefined correlations (a domain with no variation among the
    species in the fold) mapped to r = 0, p = 1 so they cannot be selected.
    """
    i_idx = np.fromiter((p[0] for p in pairs), int, len(pairs))
    j_idx = np.fromiter((p[1] for p in pairs), int, len(pairs))
    Y = (X[i_idx] - X[j_idx]) ** 2                       # (n_pairs, n_domains)
    n = Y.shape[0]
    Tc = T - T.mean()
    Yc = Y - Y.mean(axis=0, keepdims=True)
    sT = np.sqrt((Tc ** 2).sum())
    sY = np.sqrt((Yc ** 2).sum(axis=0))
    cov = Yc.T @ Tc
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / (sY * sT)
        alpha = cov / (Tc ** 2).sum()
    r = np.nan_to_num(r, nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(np.maximum(n - 2, 1) / np.maximum(1 - r ** 2, 1e-300))
    p = 2.0 * sps.t.sf(np.abs(t), df=max(n - 2, 1))
    p = np.where(np.abs(r) >= 1.0, 0.0, p)
    return {"r": r, "p": np.nan_to_num(p, nan=1.0), "alpha": np.nan_to_num(alpha, nan=0.0),
            "var": X.var(axis=0, ddof=0)}


def verify_fast_stats(X_raw: pd.DataFrame, pairs, T) -> None:
    """
    Abort unless the vectorized statistics agree, to within tight tolerances, with
    domain_time_scatter.compute_domain_stats on the full set of pairs. Signs of r are
    compared in absolute value because the two implementations may orient r differently;
    alpha and the count variance are compared directly.
    """
    ref = dts.compute_domain_stats(X_raw, T, pairs)
    fast = fast_domain_stats(X_raw.values.astype(float), pairs, T)
    cols = {c.lower(): c for c in ref.columns}
    rcol = cols.get("r") or cols.get("pearson_r") or cols.get("abs_r")
    acol = cols.get("alpha") or cols.get("slope")
    ok_r = np.allclose(np.abs(ref[rcol].values), np.abs(fast["r"]), atol=1e-8, equal_nan=True)
    ok_a = np.allclose(ref[acol].values, fast["alpha"], rtol=1e-6, atol=1e-8, equal_nan=True)
    vcol = cols.get("variance") or cols.get("var")
    ok_v = np.allclose(ref[vcol].values, fast["var"], rtol=1e-9, atol=0.0, equal_nan=True)
    cv.log(f"  verify vectorized stats: |r| match={ok_r}, alpha match={ok_a}, "
           f"var match={ok_v} (n_domains={len(fast['r'])})")
    if not (ok_r and ok_a and ok_v):
        raise SystemExit("FATAL: vectorized domain statistics disagree with "
                         "domain_time_scatter.compute_domain_stats -- refusing to continue.")


def select(stats: Dict[str, np.ndarray], mode: str, n_best: int) -> np.ndarray:
    """
    Indices of the domains kept by the selection step: positive slope, significant under
    `mode`, ranked by |r| and truncated to the top n_best. `mode` is "nominal" for an
    uncorrected p < 0.05 or "fdr" for Benjamini-Hochberg FDR < 0.05 across every domain
    tested. The positive-slope requirement keeps only domains whose squared count
    difference grows with divergence time, as the random-walk model predicts.
    """
    if mode == "nominal":
        keep = (stats["p"] < 0.05) & (stats["alpha"] > 0)
    elif mode == "fdr":
        rej, _q, _a, _b = multipletests(stats["p"], alpha=0.05, method="fdr_bh")
        keep = rej & (stats["alpha"] > 0)
    else:
        raise ValueError(mode)
    idx = np.flatnonzero(keep)
    return idx[np.argsort(-np.abs(stats["r"][idx]))][:n_best]


# Pairwise domain distance: the sum over the selected domains `cols` of the squared count
# difference divided by that domain's weight `w`, which is either |alpha| or the count
# variance depending on the weighting arm.
def distance(X: np.ndarray, pairs, cols: np.ndarray, w: np.ndarray) -> np.ndarray:
    i_idx = np.fromiter((p[0] for p in pairs), int, len(pairs))
    j_idx = np.fromiter((p[1] for p in pairs), int, len(pairs))
    Y = (X[np.ix_(i_idx, cols)] - X[np.ix_(j_idx, cols)]) ** 2
    return (Y / (w + EPSILON)).sum(axis=1)               # guarded against a zero weight


def run_nested(X: np.ndarray, species: List[str], D_pat: pd.DataFrame, folds,
               mode: str, weight: str) -> Dict:
    """
    Run the selection inside each training fold and score the resulting distance on that
    fold's test-test pairs only, so no species contributes to both selection and scoring.

    Folds carrying fewer than two selected domains are skipped, because a one-domain
    distance is a rescaling of a single squared count difference and its correlation is not
    interpretable as a multi-domain result. Returns the per-fold correlations and the
    correlations pooled over all held-out pairs, together with the pooled pair count.
    """
    pos = {s: i for i, s in enumerate(species)}
    held_d, held_t, per_fold = [], [], []
    for fi, (_tr, _te, train_sp, test_sp) in enumerate(folds, start=1):
        tr = sorted(train_sp); te = sorted(test_sp)
        tr_i = [pos[s] for s in tr]
        tr_pairs = [(a, b) for a in range(len(tr)) for b in range(a + 1, len(tr))]
        T_tr = np.array([float(D_pat.loc[tr[a], tr[b]]) for a, b in tr_pairs])
        st = fast_domain_stats(X[tr_i], tr_pairs, T_tr)
        cols = select(st, mode, N_BEST)
        if len(cols) < 2:
            continue
        w = np.abs(st["alpha"][cols]) if weight == "alpha" else st["var"][cols]
        te_i = [pos[s] for s in te]
        te_pairs = [(a, b) for a in range(len(te)) for b in range(a + 1, len(te))]
        T_te = np.array([float(D_pat.loc[te[a], te[b]]) for a, b in te_pairs])
        d = distance(X[te_i], te_pairs, cols, w)
        held_d.append(d); held_t.append(T_te)
        per_fold.append({"fold": fi, "n_selected": int(len(cols)),
                         "n_test_species": len(te), "n_test_test_pairs": len(te_pairs),
                         "spearman": float(sps.spearmanr(d, T_te).statistic),
                         "pearson": float(sps.pearsonr(d, T_te)[0])})
    d = np.concatenate(held_d); t = np.concatenate(held_t)
    return {"mode": mode, "weight": weight,
            "pooled_spearman": float(sps.spearmanr(d, t).statistic),
            "pooled_pearson": float(sps.pearsonr(d, t)[0]),
            "n_heldout_pairs": int(len(d)), "per_fold": per_fold}


def main() -> None:
    cv.ensure_dir(OUT)
    _Xn, D_pat, species = cv.load_and_preprocess_data()
    species = sorted(species)
    X_raw_all = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH,
                                      cv.load_species_list(cv.SPECIES_LIST_PATH))
    X_raw = X_raw_all.loc[species]
    pf = (X_raw > 0).sum(axis=0) / X_raw.shape[0]
    X_raw = X_raw.loc[:, pf >= MIN_PREVALENCE]
    X = X_raw.values.astype(float)
    cv.log(f"cohort: {len(species)} species, {X.shape[1]} domains after >={MIN_PREVALENCE:.0%} prevalence")

    all_pairs = [(a, b) for a in range(len(species)) for b in range(a + 1, len(species))]
    T_all = np.array([float(D_pat.loc[species[a], species[b]]) for a, b in all_pairs])
    verify_fast_stats(X_raw, all_pairs, T_all)

    # ---- in-sample arm: select on all pairs and score on the same pairs, for reference ----
    st_all = fast_domain_stats(X, all_pairs, T_all)
    insample = {}
    for mode in ("nominal", "fdr"):
        cols = select(st_all, mode, N_BEST)
        n_pass = int(((st_all["p"] < 0.05) & (st_all["alpha"] > 0)).sum()) if mode == "nominal" else \
            int((multipletests(st_all["p"], alpha=0.05, method="fdr_bh")[0] & (st_all["alpha"] > 0)).sum())
        for weight in ("alpha", "var"):
            w = np.abs(st_all["alpha"][cols]) if weight == "alpha" else st_all["var"][cols]
            d = distance(X, all_pairs, cols, w)
            insample[f"{mode}_{weight}"] = {
                "n_domains_passing_filter": n_pass,
                "spearman": float(sps.spearmanr(d, T_all).statistic),
                "pearson": float(sps.pearsonr(d, T_all)[0])}
        cv.log(f"  in-sample [{mode}] domains passing filter: {n_pass}")

    # The in-sample nominal arm performs exactly the N=50 procedure behind
    # domain_time_correlations.tsv, so it has to reproduce that table to within 5e-6 before
    # the nested numbers mean anything. Both weightings are checked, since a wrong
    # definition of the variance weight would leave the alpha arm correct and move only the
    # variance arm, which is easy to miss.
    for weight, (rho_pub, r_pub) in PUBLISHED_N50.items():
        got = insample[f"nominal_{weight}"]
        ok = (abs(got["spearman"] - rho_pub) < 5e-6) and (abs(got["pearson"] - r_pub) < 5e-6)
        cv.log(f"  reproduce published N=50 [{weight}]: rho {got['spearman']:.6f} vs "
               f"{rho_pub:.6f}, r {got['pearson']:.6f} vs {r_pub:.6f} -> "
               f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"FATAL: in-sample {weight} arm does not reproduce the published "
                             f"N=50 values -- refusing to continue.")

    _xn, _d, _sp = None, None, None
    _X2, _y, ps1, ps2 = cv.construct_pairwise_features(_Xn, D_pat, species)
    folds, _ = cv.build_stratified_deep_species_fold_indices(species, ps1, ps2, N_FOLDS, SEED)

    # ---- nested CV ----
    nested = {}
    for mode in ("nominal", "fdr"):
        for weight in ("alpha", "var"):
            r = run_nested(X, species, D_pat, folds, mode, weight)
            nested[f"{mode}_{weight}"] = r
            ins = insample[f"{mode}_{weight}"]
            cv.log(f"  nested [{mode}/{weight}]: held-out rho={r['pooled_spearman']:+.4f} "
                   f"r={r['pooled_pearson']:+.4f}  (in-sample rho={ins['spearman']:+.4f} "
                   f"r={ins['pearson']:+.4f}; n={r['n_heldout_pairs']} test-test pairs)")

    # ---- full-pipeline permutation null (selection included) ----
    cv.log(f"  permutation null: {N_PERM} species-label permutations, full pipeline")
    rng = np.random.default_rng(SEED)
    null = {k: [] for k in ("nominal_alpha_insample", "nominal_alpha_nested")}
    n_pass_null = []
    for b in range(N_PERM):
        perm = rng.permutation(len(species))
        Xp = X[perm]
        stp = fast_domain_stats(Xp, all_pairs, T_all)
        n_pass_null.append(int(((stp["p"] < 0.05) & (stp["alpha"] > 0)).sum()))
        cols = select(stp, "nominal", N_BEST)
        if len(cols) >= 2:
            w = np.abs(stp["alpha"][cols])
            d = distance(Xp, all_pairs, cols, w)
            null["nominal_alpha_insample"].append(float(sps.spearmanr(d, T_all).statistic))
    # ---- permutation null for the held-out estimate: permute, then re-run the nesting ----
    # The in-sample null above measures how much correlation the selection step alone can
    # manufacture from unrelated data. It says nothing about the held-out estimate, which is
    # not subject to selection bias in the first place. This second null permutes the
    # species labels and re-runs the entire nested procedure, so selection is redone inside
    # every training fold of every permutation and the reference distribution is the right
    # one for the held-out correlation.
    cv.log(f"  nested permutation null: {N_PERM_NESTED} permutations x {N_FOLDS} folds")
    rng2 = np.random.default_rng(SEED + 1)
    nested_null = []
    for b in range(N_PERM_NESTED):
        perm = rng2.permutation(len(species))
        Xp = X[perm]
        try:
            r = run_nested(Xp, species, D_pat, folds, "nominal", "alpha")
            nested_null.append(r["pooled_spearman"])
        except Exception:
            continue
        if (b + 1) % 25 == 0:
            a = np.array(nested_null)
            cv.log(f"    {b+1}/{N_PERM_NESTED}: null held-out rho mean={a.mean():+.4f} "
                   f"sd={a.std():.4f} max|.|={np.abs(a).max():.4f}")
    nn = np.array(nested_null)
    obs_nested = nested["nominal_alpha"]["pooled_spearman"]
    p_nested = float((np.sum(np.abs(nn) >= abs(obs_nested)) + 1) / (len(nn) + 1))
    cv.log(f"  NESTED null: mean={nn.mean():+.4f} sd={nn.std():.4f} "
           f"max|.|={np.abs(nn).max():.4f}; observed held-out {obs_nested:+.4f}; P={p_nested:.4f}")

    obs_ins = insample["nominal_alpha"]["spearman"]
    arr = np.array(null["nominal_alpha_insample"])
    p_emp = float((np.sum(np.abs(arr) >= abs(obs_ins)) + 1) / (len(arr) + 1))
    n_pass_obs = insample["nominal_alpha"]["n_domains_passing_filter"]
    p_count = float((np.sum(np.array(n_pass_null) >= n_pass_obs) + 1) / (len(n_pass_null) + 1))
    cv.log(f"  null (selection included): in-sample rho mean={arr.mean():+.4f} "
           f"sd={arr.std():.4f} max={np.abs(arr).max():.4f}; observed {obs_ins:+.4f}; P={p_emp:.4f}")
    cv.log(f"  null domains-passing-filter: mean={np.mean(n_pass_null):.0f} "
           f"median={np.median(n_pass_null):.0f}; observed {n_pass_obs}; P={p_count:.4f}")

    summary = {
        "published_reference": {
            "alpha": {"spearman": PUBLISHED_N50["alpha"][0], "pearson": PUBLISHED_N50["alpha"][1]},
            "var": {"spearman": PUBLISHED_N50["var"][0], "pearson": PUBLISHED_N50["var"][1]},
            "source": "domain_time_correlations.tsv, row N=50 (106 species, 5,565 pairs)",
            "note": ("N=50, selection on all data. Both weightings are reproduced exactly by "
                     "the in-sample nominal arm of this script, which is asserted at run time.")},
        "in_sample": insample, "nested_cv": nested,
        "permutation_null_full_pipeline": {
            "n_permutations": len(arr),
            "insample_spearman_null_mean": float(arr.mean()),
            "insample_spearman_null_sd": float(arr.std()),
            "insample_spearman_null_abs_max": float(np.abs(arr).max()),
            "observed_insample_spearman": obs_ins, "empirical_P": p_emp,
            "domains_passing_filter_null_mean": float(np.mean(n_pass_null)),
            "domains_passing_filter_null_median": float(np.median(n_pass_null)),
            "domains_passing_filter_observed": n_pass_obs,
            "domains_passing_filter_empirical_P": p_count},
        "permutation_null_NESTED_heldout": {
            "n_permutations": int(len(nn)),
            "null_mean": float(nn.mean()), "null_sd": float(nn.std()),
            "null_abs_max": float(np.abs(nn).max()),
            "observed_heldout_spearman": obs_nested, "empirical_P": p_nested,
            "interpretation": ("This is the test that settles major point 2: the selection "
                               "is redone inside every training fold of every permutation, so "
                               "selection bias is inside the null.")},
        "n_best": N_BEST, "n_folds": N_FOLDS, "random_state": SEED,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Write out the individual permutation draws, not only their summary statistics, so the
    # null can be plotted and re-analyzed without re-running the permutations. A permutation
    # test is read as a distribution with the observed value marked on it, and a mean and an
    # SD are not enough to draw that honestly. The two nulls have different numbers of draws,
    # so they are written in long form, one row per draw, with a "null" column naming which
    # of the two it belongs to, rather than as two columns of unequal length.
    draws = ([{"null": "insample_full_pipeline", "spearman": v}
              for v in null["nominal_alpha_insample"]]
             + [{"null": "nested_heldout", "spearman": float(v)} for v in nn])
    pd.DataFrame(draws).to_csv(os.path.join(OUT, "permutation_draws.csv"), index=False)
    cv.log(f"  wrote permutation_draws.csv ({len(draws)} draws)")
    rows = []
    for k, v in nested.items():
        rows.append({"selection": v["mode"], "weight": v["weight"],
                     "in_sample_spearman": insample[k]["spearman"],
                     "in_sample_pearson": insample[k]["pearson"],
                     "heldout_spearman": v["pooled_spearman"],
                     "heldout_pearson": v["pooled_pearson"],
                     "n_heldout_pairs": v["n_heldout_pairs"],
                     "domains_passing_filter": insample[k]["n_domains_passing_filter"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "nested_vs_insample.csv"), index=False)
    cv.log("")
    cv.log(pd.DataFrame(rows).to_string(index=False))
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
