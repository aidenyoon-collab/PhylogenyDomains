#!/usr/bin/env python3
"""
Within-clade discrimination: does domain content resolve divergence times inside a clade?

A companion control (supplementary/taxonomy_only_control.py) shows that clade identity on
its own predicts the four divergence classes better than the full domain feature set does.
This script asks the narrower question that follows from it: once clade identity is held
constant, is any signal left in domain content?

Within-clade pairs occupy only the two closest classes, D0 and D1, with no D2 or D3 pairs at
all, so the four-class target is degenerate inside a clade and classification there carries
no information. The target here is therefore the continuous patristic distance in millions of
years. The exact class split is recomputed at run time from the pooled predictions and written
to summary.json rather than stated in prose, so that it cannot fall out of step with the
cohort. Two independent analyses run per clade:

Analysis 1, unsupervised. Within each clade, the domain-content distance D_ij(N=50) from the
    correlation analysis is correlated against patristic distance. Domains are ranked
    out-of-clade, using only pairs in which neither species belongs to the target clade, so
    the panel cannot be tuned to the clade it is then scored on. The circular variant, ranked
    on all pairs including the target clade's own, is computed alongside it and labeled as
    such; the gap between the two shows how much apparent signal comes from selecting domains
    on the same pairs used to score them.
    Significance comes from a within-clade Mantel test: the clade's species labels are
    permuted within its own submatrix 9,999 times, the same procedure used by
    scripts/mantel_significance.py.

Analysis 2, supervised. Within each clade, a RandomForestRegressor on the |dC|
    relative-frequency features, under two hold-out regimes. The primary regime is
    leave-both-species-out: for each held-out pair {u,v} the model is refit on only those
    pairs that contain neither u nor v, so no test species appears anywhere in training.
    Leave-one-species-out is reported alongside it and is optimistic by construction, because
    holding out only u still lets the model learn v's position from v's remaining pairs.
    Held-out predictions are pooled per regime and reported as Spearman, Pearson and R^2
    against true patristic distance. R^2 may be negative and is reported as-is: a negative
    value means the model does worse than simply predicting the clade mean, which is a
    property of the data rather than a failed run.

Clades with >= 10 species are analyzed quantitatively. Afrotheria and Marsupials (4 species,
6 pairs each) are reported descriptively, with n stated and no correlation coefficient,
because a rho on 6 non-independent pairs is not interpretable.

Results are not pooled across clades: the clades differ in n and in within-clade distance
range, so a pooled coefficient would be dominated by Primates, much the largest clade.

Writes per_clade_stats.csv, summary.json, and one loo_predictions_<clade>.csv per
quantitatively analyzed clade, all under results/within_clade/.
"""
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))  # the pipeline modules reused below live in scripts/

import classification_cv_tree_reconstruction as cv  # noqa: E402
import domain_time_scatter as dts  # noqa: E402
from elastic_net_regression_stratified_species_cv import build_clade_mapping  # noqa: E402

OUT = os.path.join(REPO, "results", "within_clade")
SEED = 42
N_BEST = 50
N_PERM = 9999
MIN_SPECIES_QUANT = 10


def species_to_clade() -> Dict[str, str]:
    mapping = build_clade_mapping()
    out: Dict[str, str] = {}
    for clade, members in mapping.items():
        if isinstance(members, (list, set, tuple)):
            for sp in members:
                out[sp.replace("_", " ")] = clade
        else:
            out[clade.replace("_", " ")] = members
    return out


def within_clade_mantel(D: np.ndarray, T: np.ndarray, n: int, rng: np.random.Generator,
                        n_perm: int) -> float:
    """Two-sided Mantel P by permuting this clade's species labels in its own submatrix.

    D and T are the n x n domain-content and patristic distance matrices for one clade, in
    matching species order. Each permutation relabels the species of D only, which preserves
    both matrices' internal structure while breaking the correspondence between them, so the
    null accounts for the non-independence of pairs sharing a species. The returned P is
    (hits + 1) / (n_perm + 1), counting permutations whose Spearman rho is at least as large
    in absolute value as the observed one; the smallest attainable value is therefore
    1 / (n_perm + 1), and NaN is returned when the observed rho is undefined.
    """
    iu = np.triu_indices(n, k=1)
    obs = spearmanr(D[iu], T[iu]).statistic
    if not np.isfinite(obs):
        return float("nan")
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        Dp = D[np.ix_(p, p)]
        s = spearmanr(Dp[iu], T[iu]).statistic
        if np.isfinite(s) and abs(s) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)



def _within_clade_class_counts():
    """{class: n} for within-clade pairs, from the pooled predictions.

    Counts the true divergence classes of same-clade pairs in the main classification run's
    aggregated predictions, so the split reported in summary.json is always derived from the
    cohort actually analyzed rather than carried over as a literal. Returns None when that
    prediction file has not been generated yet.
    """
    import sys as _s
    _s.path.insert(0, os.path.join(REPO, "scripts"))
    from elastic_net_regression_stratified_species_cv import build_clade_mapping
    pth = os.path.join(REPO, "results", "classification_cv_tree_reconstruction",
                       "predictions_aggregated.csv")
    if not os.path.exists(pth):
        return None
    pr = pd.read_csv(pth)
    cl = build_clade_mapping()
    pr = pr.assign(c1=pr.sp1.map(cl), c2=pr.sp2.map(cl))
    w = pr[(pr.c1 == pr.c2) & pr.c1.notna()]
    return {f"D{int(k)}": int(v) for k, v in sorted(w.y_true.value_counts().items())}

def main() -> None:
    cv.ensure_dir(OUT)
    rng = np.random.default_rng(SEED)

    # --- canonical cohort, loaded exactly as the main classification pipeline loads it ---
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    species_ordered = sorted(species)
    sp2clade = species_to_clade()

    # Raw counts on the same cohort, for the Part-1 metric (Part 1 uses raw copy number).
    X_raw_all = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH, cv.load_species_list(cv.SPECIES_LIST_PATH))
    X_raw = X_raw_all.loc[species_ordered]
    pf = (X_raw > 0).sum(axis=0) / X_raw.shape[0]
    X_raw = X_raw.loc[:, pf >= 0.05]           # same >=5% prevalence filter as Part 1
    cv.log(f"Part-1 pool after prevalence filter: {X_raw.shape[1]} domains")

    idx = {s: i for i, s in enumerate(species_ordered)}
    n_all = len(species_ordered)
    all_pairs = [(i, j) for i in range(n_all) for j in range(i + 1, n_all)]
    T_all = np.array([float(dist_matrix.loc[species_ordered[i], species_ordered[j]])
                      for i, j in all_pairs])

    clade_members: Dict[str, List[str]] = {}
    for s in species_ordered:
        clade_members.setdefault(sp2clade[s], []).append(s)

    # Global (circular) ranking, for the side-by-side comparison only.
    stats_global = dts.compute_domain_stats(X_raw, T_all, all_pairs)
    panel_global = dts.select_ranked_domains(stats_global)[:N_BEST]

    rows, descriptive = [], []
    for clade, members in sorted(clade_members.items()):
        members = sorted(members)
        n = len(members)
        n_pairs = n * (n - 1) // 2
        if n < 3:
            continue
        sub_idx = [idx[s] for s in members]
        Tm = dist_matrix.loc[members, members].values.astype(float)
        iu = np.triu_indices(n, k=1)

        if n < MIN_SPECIES_QUANT:
            descriptive.append({
                "clade": clade, "n_species": n, "n_pairs": n_pairs,
                "patristic_min_MY": float(Tm[iu].min()), "patristic_max_MY": float(Tm[iu].max()),
                "note": ("fewer than 10 species; no correlation reported -- a coefficient on "
                         f"{n_pairs} non-independent pairs is not interpretable"),
            })
            cv.log(f"{clade}: n={n} ({n_pairs} pairs) -- descriptive only")
            continue

        # ---- analysis 1: out-of-clade (nested) domain selection ----
        out_species = [s for s in species_ordered if sp2clade[s] != clade]
        out_idx_map = {s: k for k, s in enumerate(out_species)}
        out_pairs = [(out_idx_map[a], out_idx_map[b])
                     for a in out_species for b in out_species if a < b]
        T_out = np.array([float(dist_matrix.loc[out_species[i], out_species[j]])
                          for i, j in out_pairs])
        X_out = X_raw.loc[out_species]
        stats_out = dts.compute_domain_stats(X_out, T_out, out_pairs)
        panel_out = dts.select_ranked_domains(stats_out)[:N_BEST]

        local_pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        res_clade = {"clade": clade, "n_species": n, "n_pairs": n_pairs,
                     "patristic_min_MY": float(Tm[iu].min()),
                     "patristic_max_MY": float(Tm[iu].max()),
                     "patristic_median_MY": float(np.median(Tm[iu]))}

        for tag, panel, stats_src in (("nested", panel_out, stats_out),
                                      ("circular_global", panel_global, stats_global)):
            X_clade = X_raw.loc[members]
            D_vec = dts.compute_summed_distances(X_clade, panel, stats_src, local_pairs, use_alpha=True)
            D_vec = np.asarray(D_vec, float)
            T_vec = Tm[iu]
            rho = spearmanr(D_vec, T_vec).statistic
            r = pearsonr(D_vec, T_vec)[0]
            res_clade[f"spearman_{tag}"] = float(rho)
            res_clade[f"pearson_{tag}"] = float(r)
            if tag == "nested":
                Dm = np.zeros((n, n))
                for (a, b), v in zip(local_pairs, D_vec):
                    Dm[a, b] = Dm[b, a] = v
                res_clade["mantel_P_nested"] = float(
                    within_clade_mantel(Dm, Tm, n, rng, N_PERM))

        # ---- analysis 2: RF regressor within the clade, two hold-out regimes ----
        # The primary regime is leave-both-species-out: for a held-out pair {u,v}, every
        # training pair excludes both u and v, so neither test species appears anywhere in
        # training. That is the species-blocking criterion used throughout this project.
        # Leave-one-species-out is also reported but is optimistic: holding out only u still
        # lets the model learn v's position from v's other pairs, which is most of what a
        # pairwise model needs. The loop below computes leave-one-species-out; the block after
        # it computes leave-both-species-out.
        Xc = X_norm.loc[members].values
        feat = np.zeros((len(local_pairs), Xc.shape[1]))
        for k, (a, b) in enumerate(local_pairs):
            feat[k] = np.abs(Xc[a] - Xc[b])
        Tloc = Tm[iu]
        preds = np.full(len(local_pairs), np.nan)
        for held in range(n):
            te = [k for k, (a, b) in enumerate(local_pairs) if a == held or b == held]
            tr = [k for k in range(len(local_pairs)) if k not in te]
            if len(tr) < 5:
                continue
            m = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                      random_state=SEED, n_jobs=-1)
            m.fit(feat[tr], Tloc[tr])
            preds[te] = m.predict(feat[te])
        ok = ~np.isnan(preds)
        res_clade.update({
            "loo_rf_spearman": float(spearmanr(preds[ok], Tloc[ok]).statistic),
            "loo_rf_pearson": float(pearsonr(preds[ok], Tloc[ok])[0]),
            "loo_rf_r2": float(r2_score(Tloc[ok], preds[ok])),
            "loo_rf_n_pairs": int(ok.sum()),
        })

        # leave-both-species-out: one model per held-out pair, fit with both endpoints removed
        preds_lbso = np.full(len(local_pairs), np.nan)
        for k, (u, v) in enumerate(local_pairs):
            tr = [q for q, (a, b) in enumerate(local_pairs)
                  if a not in (u, v) and b not in (u, v)]
            if len(tr) < 5:
                continue
            mb = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                       random_state=SEED, n_jobs=-1)
            mb.fit(feat[tr], Tloc[tr])
            preds_lbso[k] = float(mb.predict(feat[[k]])[0])
        okb = ~np.isnan(preds_lbso)
        res_clade.update({
            "lbso_rf_spearman": float(spearmanr(preds_lbso[okb], Tloc[okb]).statistic),
            "lbso_rf_pearson": float(pearsonr(preds_lbso[okb], Tloc[okb])[0]),
            "lbso_rf_r2": float(r2_score(Tloc[okb], preds_lbso[okb])),
            "lbso_rf_n_pairs": int(okb.sum()),
            "lbso_rf_n_models_fit": int(okb.sum()),
        })

        pd.DataFrame({"sp1": [members[a] for a, b in local_pairs],
                      "sp2": [members[b] for a, b in local_pairs],
                      "true_my": Tloc,
                      "pred_my_leave_both_out": preds_lbso,
                      "pred_my_leave_one_out": preds}).to_csv(
            os.path.join(OUT, f"loo_predictions_{clade}.csv"), index=False)

        rows.append(res_clade)
        cv.log(f"{clade}: n={n} pairs={n_pairs} | nested rho={res_clade['spearman_nested']:+.3f} "
               f"(Mantel P={res_clade['mantel_P_nested']:.4f}) | circular rho={res_clade['spearman_circular_global']:+.3f} "
               f"| RF leave-both-out rho={res_clade['lbso_rf_spearman']:+.3f} "
               f"R2={res_clade['lbso_rf_r2']:+.3f} "
               f"(leave-one-out, optimistic: rho={res_clade['loo_rf_spearman']:+.3f} "
               f"R2={res_clade['loo_rf_r2']:+.3f})")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT, "per_clade_stats.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"quantitative": rows, "descriptive_only": descriptive,
                   "within_clade_class_counts": _within_clade_class_counts(),
                   "design_note": (f"Within-clade pairs are "
                                   f"{_within_clade_class_counts()} by true class, with zero "
                                   "D2/D3, so the four-class target is degenerate inside a "
                                   "clade; the target here is continuous patristic distance. "
                                   "Results are deliberately NOT pooled across clades."),
                   "n_permutations": N_PERM, "n_best_domains": N_BEST, "random_state": SEED},
                  fh, indent=2)
    cv.log("")
    if len(tbl):
        cols = ["clade", "n_species", "n_pairs", "spearman_nested", "mantel_P_nested",
                "spearman_circular_global", "lbso_rf_spearman", "lbso_rf_r2",
                "loo_rf_spearman", "loo_rf_r2"]
        cv.log(tbl[cols].to_string(index=False))
    cv.log(f"\nDescriptive-only clades: {[d['clade'] for d in descriptive]}")
    cv.log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
