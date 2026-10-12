#!/usr/bin/env python3
"""
Test how much of the domain-distance/divergence-time correlation is carried by the
marsupial-placental split, and how much survives within placental mammals alone.

The analytical domain distance correlates with TimeTree patristic distance across all
species pairs, but the relationship is close to flat from 0-200 MY and rises mainly where
it reaches the deepest divergences, the marsupial-placental split. A correlation produced
that way would reflect one deep split rather than a graded signal spread across the tree,
so this script measures the two contributions separately.

Selection caveat that drives the design: the top-50 domain panel is ranked using all
species pairs, so restricting that same panel to placental pairs would be circular,
because the domains were chosen partly for separating marsupials from placentals. Every
placental-only estimate here re-selects the panel from placental pairs alone, and the
nested variant re-selects it inside each training fold of placental species.

Four quantities are computed:
  1. The correlation of the all-pairs distance within each pair category
     (marsupial-marsupial, marsupial-placental, placental-placental), which shows where
     in the data the pooled correlation originates.
  2. The correlation with the panel re-selected on placental pairs only, both in sample
     and under species-blocked nested cross-validation. Each gets its own permutation
     null, because in-sample selection inflates correlation even on permuted labels and
     the two estimates therefore need different reference distributions.
  3. Gradual versus punctuated change. A continuous random-walk model predicts distance
     growing with time throughout; a punctuated model predicts a single step at the deep
     split. Variance explained by a linear function of time is compared with variance
     explained by a binary indicator for whether a pair crosses the marsupial-placental
     split, together with the increment each term adds over the other.
  4. Whether classifier accuracy, balanced accuracy and kappa hold up when the pooled
     predictions are rescored on placental pairs only.

Reads the domain count matrix and the TimeTree reference tree through the classification
pipeline's loader, plus that pipeline's aggregated predictions for part 4 if present.
Writes results/within_placental/summary.json and nothing else.
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import classification_cv_tree_reconstruction as cv  # noqa: E402
import domain_time_scatter as dts  # noqa: E402
import nested_cv_feature_selection as ncv  # noqa: E402  (fast vectorized stats + selection)
from elastic_net_regression_stratified_species_cv import build_clade_mapping  # noqa: E402

OUT = os.path.join(REPO, "results", "within_placental")
SEED = cv.RANDOM_STATE
N_BEST = 50
N_PERM = 2000
N_PERM_NESTED = 200         # held-out null; each draw re-runs the full 5-fold nested procedure


def main() -> None:
    cv.ensure_dir(OUT)
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    sp = sorted(species)
    n = len(sp)

    mapping = build_clade_mapping()
    sp2clade: Dict[str, str] = {}
    for clade, members in mapping.items():
        if isinstance(members, (list, set, tuple)):
            for s in members:
                sp2clade[s.replace("_", " ")] = clade
        else:
            sp2clade[clade.replace("_", " ")] = members
    marsup = sorted([s for s in sp if sp2clade[s] == "Marsupials"])
    plac = sorted([s for s in sp if sp2clade[s] != "Marsupials"])
    cv.log(f"marsupials n={len(marsup)}: {marsup}")
    cv.log(f"placentals n={len(plac)}")

    # Basis for the analytical distance: raw counts restricted to domains present in at
    # least 5% of species, matching the filter in domain_time_scatter.
    raw_full = cv.load_domain_counts(cv.DOMAIN_COUNTS_PATH, cv.load_species_list(cv.SPECIES_LIST_PATH))
    Xraw = raw_full.loc[sp, list(X_norm.columns)]
    pf = (Xraw > 0).sum(axis=0) / n
    Xp1 = Xraw.loc[:, pf >= 0.05]
    cv.log(f"Part-1 pool: {Xp1.shape[1]} domains")

    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    T = np.array([float(dist_matrix.loc[sp[a], sp[b]]) for a, b in pairs])
    is_m = np.array([sp2clade[s] == "Marsupials" for s in sp])
    cat = np.array(["mm" if is_m[a] and is_m[b] else
                    ("pp" if not is_m[a] and not is_m[b] else "mp") for a, b in pairs])
    cv.log(f"pair categories: mm={int((cat=='mm').sum())}, mp={int((cat=='mp').sum())}, "
           f"pp={int((cat=='pp').sum())} (total {len(pairs)})")

    # ---------- Q1: decompose the all-pairs distance by pair category ----------
    stats_all = dts.compute_domain_stats(Xp1, T, pairs)
    panel_all = dts.select_ranked_domains(stats_all)[:N_BEST]
    D_pub = np.asarray(dts.compute_summed_distances(Xp1, panel_all, stats_all, pairs, use_alpha=True), float)
    q1 = {"overall": {"spearman": float(spearmanr(D_pub, T).statistic),
                      "pearson": float(pearsonr(D_pub, T)[0]), "n": len(T)}}
    for c, lab in (("pp", "placental-placental"), ("mp", "marsupial-placental"),
                   ("mm", "marsupial-marsupial")):
        m = cat == c
        q1[lab] = {"spearman": float(spearmanr(D_pub[m], T[m]).statistic) if m.sum() > 2 else None,
                   "pearson": float(pearsonr(D_pub[m], T[m])[0]) if m.sum() > 2 else None,
                   "n": int(m.sum())}
    # Same all-pairs panel with every marsupial pair dropped. Still circular, since the
    # panel saw the marsupials during selection; reported only as a contrast to Q2.
    m_pp = cat == "pp"
    q1["placental_only_PUBLISHED_panel"] = {
        "spearman": float(spearmanr(D_pub[m_pp], T[m_pp]).statistic),
        "pearson": float(pearsonr(D_pub[m_pp], T[m_pp])[0]), "n": int(m_pp.sum()),
        "caveat": "panel selected using all pairs including marsupials -- circular, for contrast only"}
    cv.log("\nQ1 correlation of the PUBLISHED distance, by pair category:")
    for k, v in q1.items():
        if v.get("spearman") is not None:
            cv.log(f"  {k:34} n={v['n']:5}  rho={v['spearman']:+.4f}  r={v['pearson']:+.4f}")

    # ---------- Q2: re-select the panel on placental pairs only ----------
    pidx = [sp.index(s) for s in plac]
    Xpl = Xp1.iloc[pidx]
    npl = len(plac)
    ppairs = [(a, b) for a in range(npl) for b in range(a + 1, npl)]
    Tpl = np.array([float(dist_matrix.loc[plac[a], plac[b]]) for a, b in ppairs])
    stats_pl = dts.compute_domain_stats(Xpl, Tpl, ppairs)
    panel_pl = dts.select_ranked_domains(stats_pl)[:N_BEST]
    D_pl = np.asarray(dts.compute_summed_distances(Xpl, panel_pl, stats_pl, ppairs, use_alpha=True), float)
    n_pass_pl = int(((stats_pl["pval"] < 0.05) & (stats_pl["alpha"] > 0)).sum()) \
        if "pval" in stats_pl.columns else None
    q2 = {"in_sample_placental_only": {
        "spearman": float(spearmanr(D_pl, Tpl).statistic),
        "pearson": float(pearsonr(D_pl, Tpl)[0]), "n": len(Tpl),
        "n_domains_passing_filter": n_pass_pl,
        "panel_overlap_with_published": int(len(set(panel_pl) & set(panel_all)))}}
    cv.log(f"\nQ2 panel RE-SELECTED on placentals only ({npl} species, {len(ppairs)} pairs):")
    cv.log(f"  in-sample rho={q2['in_sample_placental_only']['spearman']:+.4f} "
           f"r={q2['in_sample_placental_only']['pearson']:+.4f}; "
           f"{q2['in_sample_placental_only']['panel_overlap_with_published']}/50 domains "
           f"shared with the published panel")

    # Nested cross-validation within placentals: the panel is re-selected inside each
    # training fold, so no held-out pair contributed to choosing its own domains.
    X_np = Xpl.values.astype(float)
    Dpat_pl = dist_matrix.loc[plac, plac]
    # The main pipeline's fold builder stratifies on the deep lineages (Marsupialia and
    # Afrotheria). With the marsupials removed only 4 Afrotheria remain, fewer than the 5
    # folds StratifiedKFold requires, so a plain species-blocked KFold is used here
    # instead. The species blocking, meaning no test species appears in any training pair,
    # is unchanged; only the stratification is dropped, and what it protected against is a
    # collapsing marsupial stratum that this cohort does not contain by construction.
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds_pl = []
    plac_arr = np.array(plac)
    for tr_s, te_s in kf.split(plac_arr):
        train_sp = frozenset(plac_arr[tr_s].tolist())
        test_sp = frozenset(plac_arr[te_s].tolist())
        folds_pl.append((None, None, train_sp, test_sp))
    cv.log(f"  placental folds (species-blocked, unstratified): "
           f"{[len(f[3]) for f in folds_pl]} test species per fold")
    nested = ncv.run_nested(X_np, plac, Dpat_pl, folds_pl, "nominal", "alpha")
    q2["nested_cv_placental_only"] = {"spearman": nested["pooled_spearman"],
                                      "pearson": nested["pooled_pearson"],
                                      "n_heldout_pairs": nested["n_heldout_pairs"]}
    cv.log(f"  nested CV held-out rho={nested['pooled_spearman']:+.4f} "
           f"r={nested['pooled_pearson']:+.4f} on {nested['n_heldout_pairs']} blind pairs")
    # Permutation null for the placental-only in-sample value. Species labels are shuffled
    # and the panel is re-selected on every draw, so the selection step is inside the null.
    rng = np.random.default_rng(SEED)
    null = []
    Xv = Xpl.values.astype(float)
    for _ in range(N_PERM):
        perm = rng.permutation(npl)
        st = ncv.fast_domain_stats(Xv[perm], ppairs, Tpl)
        cols = ncv.select(st, "nominal", N_BEST)
        if len(cols) < 2:
            continue
        d = ncv.distance(Xv[perm], ppairs, cols, np.abs(st["alpha"][cols]))
        null.append(float(spearmanr(d, Tpl).statistic))
    null = np.array(null)
    obs = q2["in_sample_placental_only"]["spearman"]
    q2["permutation_null_placental"] = {
        "n": int(len(null)), "mean": float(null.mean()), "sd": float(null.std()),
        "abs_max": float(np.abs(null).max()),
        "empirical_P": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))}
    cv.log(f"  permutation null (IN-SAMPLE, selection included): mean={null.mean():+.4f} "
           f"sd={null.std():.4f} max|.|={np.abs(null).max():.4f}; observed in-sample "
           f"{obs:+.4f}; P={q2['permutation_null_placental']['empirical_P']:.4f}")

    # A separate held-out null for the nested estimate. The in-sample null above has a
    # strongly positive mean, because ranking domains by |r| and keeping the top 50
    # manufactures correlation even on permuted data; that null therefore calibrates the
    # in-sample value only. The nested held-out value does not carry the same selection
    # bias, so it needs a null of its own. Without one the nested rho is uncalibrated,
    # which matters here because it lands just above the in-sample null's abs_max and so
    # is easy to misread as marginal when it is being compared against the wrong null.
    cv.log(f"  nested held-out permutation null: {N_PERM_NESTED} permutations x 5 folds")
    rng2 = np.random.default_rng(SEED + 1)
    nested_null = []
    for b in range(N_PERM_NESTED):
        perm = rng2.permutation(npl)
        try:
            nested_null.append(ncv.run_nested(X_np[perm], plac, Dpat_pl, folds_pl,
                                              "nominal", "alpha")["pooled_spearman"])
        except Exception:
            continue
        if (b + 1) % 25 == 0:
            a = np.array(nested_null)
            cv.log(f"    {b+1}/{N_PERM_NESTED}: null held-out rho mean={a.mean():+.4f} "
                   f"sd={a.std():.4f} max|.|={np.abs(a).max():.4f}")
    nn = np.array(nested_null)
    obs_n = q2["nested_cv_placental_only"]["spearman"]
    q2["permutation_null_placental_NESTED_heldout"] = {
        "n": int(len(nn)), "mean": float(nn.mean()), "sd": float(nn.std()),
        "abs_max": float(np.abs(nn).max()),
        "observed_heldout_spearman": float(obs_n),
        "empirical_P": float((np.sum(np.abs(nn) >= abs(obs_n)) + 1) / (len(nn) + 1)),
        "note": ("Calibrates the placental-only NESTED held-out rho. Species labels are "
                 "permuted and the whole nested procedure -- including per-fold domain "
                 "selection -- is re-run, so selection is inside the null. This is the "
                 "like-for-like test of whether within-placental signal survives blinding.")}
    cv.log(f"  NESTED held-out null: mean={nn.mean():+.4f} sd={nn.std():.4f} "
           f"max|.|={np.abs(nn).max():.4f}; observed held-out {obs_n:+.4f}; "
           f"P={q2['permutation_null_placental_NESTED_heldout']['empirical_P']:.4f}")

    # ---------- Q3: gradual vs punctuated ----------
    cv.log("\nQ3 gradual (linear in time) versus punctuated (a step at the deep split):")
    y = (D_pub - D_pub.mean()) / D_pub.std()
    step = (cat != "pp").astype(float)          # 1 if the pair crosses the marsupial-placental split
    tt = (T - T.mean()) / T.std()
    def r2(x):
        x = (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
        return float(np.corrcoef(x, y)[0, 1] ** 2)
    q3 = {"R2_linear_in_time": r2(tt), "R2_step_at_marsupial_split": r2(step)}
    both = np.column_stack([tt, step])
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), both]), y, rcond=None)
    pred = np.column_stack([np.ones(len(y)), both]) @ coef
    q3["R2_both"] = float(np.corrcoef(pred, y)[0, 1] ** 2)
    q3["R2_time_given_step"] = q3["R2_both"] - q3["R2_step_at_marsupial_split"]
    q3["R2_step_given_time"] = q3["R2_both"] - q3["R2_linear_in_time"]
    for k, v in q3.items():
        cv.log(f"  {k:32} = {v:.4f}")

    # ---------- Q4: classifier restricted to placental pairs ----------
    q4 = {}
    pp = os.path.join(REPO, "results", "classification_cv_tree_reconstruction",
                      "predictions_aggregated.csv")
    if os.path.exists(pp):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
        pr = pd.read_csv(pp)
        pr["is_pp"] = (~pr.sp1.map(lambda s: sp2clade.get(s) == "Marsupials")) & \
                      (~pr.sp2.map(lambda s: sp2clade.get(s) == "Marsupials"))
        for lab, m in (("all_pairs", np.ones(len(pr), bool)), ("placental_only", pr.is_pp.values)):
            s = pr[m]
            q4[lab] = {"n": int(m.sum()),
                       "accuracy": float(accuracy_score(s.y_true, s.y_pred)),
                       "balanced_accuracy": float(balanced_accuracy_score(s.y_true, s.y_pred)),
                       "cohen_kappa": float(cohen_kappa_score(s.y_true, s.y_pred)),
                       "class_counts": s.y_true.value_counts().sort_index().to_dict()}
        cv.log("\nQ4 published classifier, all pairs vs placental pairs only:")
        for k, v in q4.items():
            cv.log(f"  {k:16} n={v['n']:5} acc={v['accuracy']:.4f} bal={v['balanced_accuracy']:.4f} "
                   f"kappa={v['cohen_kappa']:.4f} classes={v['class_counts']}")

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"marsupials": marsup, "n_placental": npl,
                   "Q1_by_pair_category": q1, "Q2_placental_only_selection": q2,
                   "Q3_gradual_vs_punctuated": q3, "Q4_classifier": q4}, fh, indent=2)
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
