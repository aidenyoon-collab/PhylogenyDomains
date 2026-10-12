#!/usr/bin/env python3
"""
Reconstruct mammalian phylogenies from a continuous domain-based distance instead of
from four predicted distance classes, compare tree-building algorithms, and score clade
recovery numerically rather than by inspection.

Why a continuous distance matters here. A four-state distance matrix is saturated with
ties, which leaves branching order undetermined: of the 5,671 pairs, 2,783 share the
modal predicted value, the resulting UPGMA tree carries 58 of its 105 internal branches
at length zero, and its topology is settled by arbitrary tie-breaking. Perturbing that
matrix by 1e-14 MY moves 144 of 208 bipartitions, so a single normalized
Robinson-Foulds value of 0.8846 is one draw from a tie-breaking band rather than a
stable property of the method. This script measures the width of that band and shows
that the continuous arms are insensitive to it.

Arms, all on the canonical 106-species / 7,976-domain cohort and the species-blocked,
clade-stratified 5-fold split, with the per-fold model seed set to random_state + fold_i
exactly as scripts/classification_cv_tree_reconstruction.py sets it, so that
CLASS_ARGMAX reproduces the baseline classification run bit for bit:
  CLASS_ARGMAX   baseline: argmax class -> train-fold class centroid (4 distinct values)
  CLASS_EXPECTED sum_k p_k * mu_k from the same fitted classifier: continuous, no new model
  RF_REG         RandomForestRegressor on |dC| predicting patristic MY directly
  RF_REG_LOG     the same regressor on log10(MY + 1), back-transformed

Robustness checks:
  * tie-break sensitivity: N_JITTER replicates of each arm's matrix perturbed by
    N(0, 1e-10) MY, a perturbation with no biological meaning but a decisive one for a
    matrix full of ties
  * species-block delete-d jackknife on the Robinson-Foulds difference between arms,
    with the predictions held fixed so that only tree-level sampling variation is
    isolated
  * counts of zero-length internal branches, the most direct measure of how much of the
    topology the distance matrix leaves unresolved
  * clade recovery reported both rooted and unrooted, per clade, never as a net count.
    Under ete3's unrooted=True a taxon set and its complement are the same test, so
    "Marsupials" and "Placentalia" are one statement; the panel also includes the two
    superorders, without which the comparison would be selective.
  * error on the MY scale (RMSE / MAE), because correlation is scale-free and a
    continuous arm can raise Pearson while being worse in millions of years
  * Kendall tau alongside Spearman, because tree quality tracks rank agreement

Writes only under results/continuous_trees/.

Minimum evolution and the NJ caveat. Neither FastME nor scikit-bio is guaranteed to be
installed and Biopython offers only nj/upgma, so balanced minimum evolution is
implemented in supplementary/me_tree.py (Pauplin's tree length plus NNI hill-climbing
from the NJ start) and validated there: from a topology scrambled by 30 random NNIs it
recovers the reference tree exactly (normalized Robinson-Foulds 0.0000) on the true
patristic matrix.

Where FastME 2.1.6.3 is installed (`brew install fastme`) it serves as an independent
check, and it beats the local implementation on every arm because it adds SPR moves to
NNI. Its tree is therefore the one reported when the binary is present; me_tree.py
remains the portable fallback and the cross-check, and the Robinson-Foulds distance
between the two implementations is recorded per arm.

Both NJ and minimum evolution are consistent only for near-additive distances, so each
arm's matrix is scored on the four-point condition rather than merely caveated: for a
quartet, additivity requires the two largest of d_ij+d_kl, d_ik+d_jl, d_il+d_jk to be
equal, and the reported statistic is the mean relative gap between them (0 = perfectly
additive) over 20,000 sampled quartets.
"""
import io
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))  # make the main pipeline importable

import classification_cv_tree_reconstruction as cv  # noqa: E402
from elastic_net_regression_stratified_species_cv import build_clade_mapping  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import me_tree  # noqa: E402  (local balanced-minimum-evolution implementation)

OUT = os.path.join(REPO, "results", "continuous_trees")
SEED = cv.RANDOM_STATE
N_FOLDS = cv.N_FOLDS_DEFAULT
N_TREES = cv.N_ESTIMATORS_DEFAULT
N_JITTER = 200
JITTER_SD = 1e-10
JACKKNIFE_D = [11, 21]
N_JACK = 120

SUPERORDER = {
    "Primates": "Euarchontoglires", "Glires": "Euarchontoglires",
    "Laurasiatheria_Carnivora": "Laurasiatheria",
    "Laurasiatheria_Cetartiodactyla": "Laurasiatheria",
    "Laurasiatheria_Chiroptera": "Laurasiatheria",
    "Laurasiatheria_Perissodactyla": "Laurasiatheria",
    "Laurasiatheria_Eulipotyphla": "Laurasiatheria",
    "Afrotheria": "Afrotheria", "Marsupials": "Marsupials",
}


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


def nj_newick(D: np.ndarray, names: List[str]) -> str:
    from Bio import Phylo
    from Bio.Phylo.TreeConstruction import DistanceMatrix as BioDM
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
    n = len(names)
    lower = [[float(D[i][j]) for j in range(i + 1)] for i in range(n)]
    tree = DistanceTreeConstructor().nj(BioDM(list(names), lower))
    for cl in tree.find_clades():
        if cl.branch_length is not None and cl.branch_length < 0:
            cl.branch_length = 0.0
    buf = io.StringIO()
    Phylo.write(tree, buf, "newick")
    return buf.getvalue().strip()


def fastme_me_newick(D: np.ndarray, names: List[str], tag: str) -> Tuple[str, str]:
    """
    Balanced minimum evolution via FastME (BME + NNI + SPR), if the binary is present.

    Falls back to supplementary/me_tree.py (BME + NNI only) so the pipeline stays
    runnable without the external dependency. FastME's SPR moves escape local optima
    that NNI-only hill-climbing cannot: on these data it reaches a strictly lower
    balanced tree length on every arm (e.g. 6019.9 vs 6039.9 for CLASS_ARGMAX), so when
    it is available its tree is the one reported.
    """
    import shutil, subprocess, tempfile
    exe = shutil.which("fastme")
    if exe is None:
        return "", "fallback_nni_only"
    with tempfile.TemporaryDirectory() as td:
        phy, out = os.path.join(td, "d.phy"), os.path.join(td, "t.nwk")
        with open(phy, "w") as fh:
            fh.write(f" {len(names)}\n")
            for i, nm in enumerate(names):
                fh.write(f"{nm:<62}" + " ".join(f"{v:.10f}" for v in D[i]) + "\n")
        r = subprocess.run([exe, "-i", phy, "-o", out, "-m", "B", "-n", "B", "-s"],
                           capture_output=True, text=True, cwd=td)
        if r.returncode != 0 or not os.path.exists(out):
            cv.log(f"    FastME failed for {tag}: {r.stderr[:200]}")
            return "", "fastme_failed"
        return open(out).read().strip(), "fastme_BME_NNI_SPR"


def _groups(species: List[str], sp2clade: Dict[str, str]) -> Dict[str, List[str]]:
    """Evaluation panel: the seven clades with n >= 3, plus the two superorders."""
    fmt = {s: cv.format_species_label_for_newick(s) for s in species}
    g: Dict[str, List[str]] = defaultdict(list)
    for s in species:
        g[sp2clade[s]].append(fmt[s])
        g["SUPER_" + SUPERORDER[sp2clade[s]]].append(fmt[s])
    return {k: v for k, v in g.items() if len(v) >= 3
            and k not in ("SUPER_Afrotheria", "SUPER_Marsupials")}


def score_tree(newick: str, species: List[str], sp2clade: Dict[str, str], tref) -> Dict:
    from ete3 import Tree
    tpred = Tree(newick, format=1)
    for leaf in tpred.iter_leaves():
        leaf.name = leaf.name.replace("'", "").replace('"', "").strip()
    if {l.name for l in tpred.iter_leaves()} != {l.name for l in tref.iter_leaves()}:
        raise RuntimeError("leaf set mismatch")
    rf = tref.robinson_foulds(tpred, unrooted_trees=True)
    zero_int = sum(1 for nd in tpred.traverse()
                   if not nd.is_leaf() and not nd.is_root()
                   and (nd.dist is None or abs(nd.dist) < 1e-12))
    n_int = sum(1 for nd in tpred.traverse() if not nd.is_leaf() and not nd.is_root())
    per: Dict[str, Dict] = {}
    for name, members in sorted(_groups(species, sp2clade).items()):
        un, _t, _v = tpred.check_monophyly(values=members, target_attr="name", unrooted=True)
        ro, _t2, _v2 = tpred.check_monophyly(values=members, target_attr="name", unrooted=False)
        per[name] = {"n": len(members), "unrooted": bool(un), "rooted": bool(ro)}
    return {
        "robinson_foulds_distance": int(rf[0]), "robinson_foulds_max": int(rf[1]),
        "robinson_foulds_normalized": float(rf[0] / rf[1]) if rf[1] else float("nan"),
        "internal_branches": n_int, "zero_length_internal_branches": zero_int,
        "n_groups": len(per),
        "n_monophyletic_unrooted": sum(1 for v in per.values() if v["unrooted"]),
        "n_monophyletic_rooted": sum(1 for v in per.values() if v["rooted"]),
        "per_group": per,
    }


def rf_of_matrix(D: np.ndarray, species: List[str], tref) -> Tuple[float, int]:
    from ete3 import Tree
    nwk = cv.build_upgma_newick(pd.DataFrame(D, index=species, columns=species))
    t = Tree(nwk, format=1)
    rf = tref.robinson_foulds(t, unrooted_trees=True)
    return (float(rf[0] / rf[1]) if rf[1] else float("nan")), int(rf[0])


def main() -> None:
    cv.ensure_dir(OUT)
    X_norm, dist_matrix, species = cv.load_and_preprocess_data()
    species_ordered = sorted(species)
    X, y_dist, pair_sp1, pair_sp2 = cv.construct_pairwise_features(
        X_norm, dist_matrix, species_ordered)
    scheme = cv.build_scheme_definitions()["SCHEME_4B"]
    y_class, n_classes, _ = cv.assign_classes(y_dist, scheme["cutoffs"], scheme["labels"])
    folds, _ = cv.build_stratified_deep_species_fold_indices(
        species_ordered, pair_sp1, pair_sp2, N_FOLDS, SEED)
    sp2clade = species_to_clade()

    proba_sum = defaultdict(lambda: np.zeros(n_classes)); proba_cnt = defaultdict(int)
    reg_sum = defaultdict(float); reg_cnt = defaultdict(int); reglog_sum = defaultdict(float)
    fold_means: List[np.ndarray] = []

    for fold_i, (tr_idx, te_idx, _a, _b) in enumerate(folds, start=1):
        cv.log(f"--- fold {fold_i}/{N_FOLDS} (model seed {SEED + fold_i}) ---")
        fold_means.append(cv.compute_fold_train_class_means(tr_idx, y_class, y_dist, n_classes))
        # per-fold seed matches classification_cv_tree_reconstruction.py exactly
        res = cv.fit_fold_random_oversample_rf(X[tr_idx], y_class[tr_idx], X[te_idx],
                                               y_class[te_idx], n_classes,
                                               SEED + fold_i, N_TREES)
        if res is not None:
            proba_full, _ = res
            for row_i, gi in enumerate(te_idx):
                k = cv.canonical_pair_key(pair_sp1[gi], pair_sp2[gi])
                proba_sum[k] += proba_full[row_i]; proba_cnt[k] += 1
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr_idx]); Xte = sc.transform(X[te_idx])
        for tag, ytr, back in (("raw", y_dist[tr_idx], lambda v: v),
                               ("log", np.log10(y_dist[tr_idx] + 1.0),
                                lambda v: np.power(10.0, v) - 1.0)):
            m = RandomForestRegressor(n_estimators=N_TREES, max_features="sqrt",
                                      random_state=SEED + fold_i, n_jobs=-1)
            m.fit(Xtr, ytr)
            p = back(m.predict(Xte))
            for row_i, gi in enumerate(te_idx):
                k = cv.canonical_pair_key(pair_sp1[gi], pair_sp2[gi])
                if tag == "raw":
                    reg_sum[k] += float(p[row_i]); reg_cnt[k] += 1
                else:
                    reglog_sum[k] += float(p[row_i])

    stack = np.vstack(fold_means)
    mu_k = np.array([float(np.nanmean(stack[:, k])) for k in range(n_classes)])
    cv.log(f"class centroids: {np.round(mu_k, 4).tolist()}")

    true_my = {cv.canonical_pair_key(pair_sp1[g], pair_sp2[g]): float(y_dist[g])
               for g in range(len(y_dist))}
    keys = [k for k in true_my if proba_cnt[k] > 0 and reg_cnt[k] > 0]
    sp_used = sorted({str(s) for k in keys for s in k})
    arms = {
        "CLASS_ARGMAX":   {k: float(mu_k[int(np.argmax(proba_sum[k] / proba_cnt[k]))]) for k in keys},
        "CLASS_EXPECTED": {k: float(np.dot(proba_sum[k] / proba_cnt[k], mu_k)) for k in keys},
        "RF_REG":         {k: reg_sum[k] / reg_cnt[k] for k in keys},
        "RF_REG_LOG":     {k: reglog_sum[k] / reg_cnt[k] for k in keys},
    }
    arms = {a: {(str(x), str(y)): v for (x, y), v in d.items()} for a, d in arms.items()}
    true_sub = {(str(x), str(y)): true_my[k] for k in keys for x, y in [k]}

    # confirm CLASS_ARGMAX reproduces the baseline classification run, pair for pair
    pubp = os.path.join(REPO, "results", "classification_cv_tree_reconstruction",
                        "predictions_aggregated.csv")
    repro = None
    if os.path.exists(pubp):
        pub = pd.read_csv(pubp)
        pubd = {(r.sp1, r.sp2): float(r.predicted_distance_MY) for r in pub.itertuples()}
        same = sum(1 for k, v in arms["CLASS_ARGMAX"].items()
                   if k in pubd and abs(v - pubd[k]) < 1e-9)
        repro = {"pairs_compared": len(pubd), "identical_predicted_distance": same,
                 "fraction_identical": same / len(pubd)}
        cv.log(f"reproduction vs published: {same}/{len(pubd)} identical "
               f"({same / len(pubd) * 100:.2f}%)")

    from ete3 import Tree
    tref = Tree(cv.reference_tree_to_formatted_newick(
        cv.load_and_prune_reference_tree(sp_used)), format=1)
    n = len(sp_used); pos = {s: i for i, s in enumerate(sp_used)}
    T = np.zeros((n, n))
    for (a, b), v in true_sub.items():
        T[pos[a], pos[b]] = T[pos[b], pos[a]] = v
    iu = np.triu_indices(n, k=1)

    results, rows = [], []
    rng = np.random.default_rng(SEED)
    for name, d in arms.items():
        D = np.zeros((n, n))
        for (a, b), v in d.items():
            D[pos[a], pos[b]] = D[pos[b], pos[a]] = v
        r = {"arm": name,
             "pearson": float(pearsonr(T[iu], D[iu])[0]),
             "spearman": float(spearmanr(T[iu], D[iu]).statistic),
             "kendall_tau_b": float(kendalltau(T[iu], D[iu]).statistic),
             "rmse_MY": float(np.sqrt(np.mean((T[iu] - D[iu]) ** 2))),
             "mae_MY": float(np.mean(np.abs(T[iu] - D[iu]))),
             "median_ae_MY": float(np.median(np.abs(T[iu] - D[iu]))),
             "predicted_range_MY": [float(D[iu].min()), float(D[iu].max())],
             "n_distinct_values": int(np.unique(np.round(D[iu], 9)).size),
             "n_tied_at_mode": int(pd.Series(np.round(D[iu], 9)).value_counts().iloc[0])}
        labels = [cv.format_species_label_for_newick(s) for s in sp_used]
        nj_nwk = nj_newick(D, labels)
        # Minimum evolution: balanced (Pauplin) tree length with NNI hill-climbing from
        # the NJ tree, which is the FastME algorithm.
        from ete3 import Tree as _T
        me_start = _T(nj_nwk, format=1)
        for _lf in me_start.iter_leaves():
            _lf.name = _lf.name.replace("'", "").strip()
        L_start = me_tree.pauplin_length(me_start, D, labels)
        me_t, L_end, n_swaps, n_rounds = me_tree.bme_nni_search(me_start, D, labels)
        own_nwk = me_t.write(format=1)
        fm_nwk, me_source = fastme_me_newick(D, labels, name)
        if fm_nwk:
            _fm = _T(fm_nwk, format=1)
            for _lf in _fm.iter_leaves():
                _lf.name = _lf.name.replace("'", "").strip()
            L_fastme = me_tree.pauplin_length(_fm, D, labels)
            me_nwk = fm_nwk
            # independent cross-check: two implementations of the same objective
            _own = _T(own_nwk, format=1)
            for _lf in _own.iter_leaves():
                _lf.name = _lf.name.replace("'", "").strip()
            _agree = _own.robinson_foulds(_fm, unrooted_trees=True)
            r["me_crosscheck_rf_own_vs_fastme"] = [int(_agree[0]), int(_agree[1])]
        else:
            L_fastme, me_nwk = float("nan"), own_nwk
        r["me_source"] = me_source
        r["me_bme_length_start"] = float(L_start)
        r["me_bme_length_own_nni"] = float(L_end)
        r["me_bme_length_fastme"] = float(L_fastme)
        r["me_nni_swaps_own"] = int(n_swaps)
        r["additivity"] = me_tree.additivity_deviation(D, n_quartets=20000,
                                                       rng=np.random.default_rng(SEED))
        for meth, nwk in (("UPGMA", cv.build_upgma_newick(pd.DataFrame(D, index=sp_used, columns=sp_used))),
                          ("NJ", nj_nwk), ("ME", me_nwk)):
            r[meth] = score_tree(nwk, sp_used, sp2clade, tref)
            with open(os.path.join(OUT, f"tree_{name}_{meth}.nwk"), "w") as fh:
                fh.write(nwk + "\n")
        # Tie-break sensitivity (UPGMA only; NJ is far less tie-sensitive). The jitter is
        # 1e-10 MY, which carries no biological meaning but is decisive for a matrix
        # whose ties leave UPGMA's merge order undetermined.
        # Clade recovery is recorded per draw alongside Robinson-Foulds, because the
        # monophyly count is tie-break dependent as well: a single "6 of 9 clades
        # recovered" is a property of one draw, not of the tree.
        vals, mono_u, mono_r = [], [], []
        for _ in range(N_JITTER):
            E = rng.normal(0, JITTER_SD, (n, n)); E = np.triu(E, 1); E = E + E.T
            Jm = D + E
            vals.append(rf_of_matrix(Jm, sp_used, tref)[0])
            sj = score_tree(cv.build_upgma_newick(
                pd.DataFrame(Jm, index=sp_used, columns=sp_used)), sp_used, sp2clade, tref)
            mono_u.append(sj["n_monophyletic_unrooted"])
            mono_r.append(sj["n_monophyletic_rooted"])
        r["tiebreak_rf_mean"] = float(np.mean(vals))
        r["tiebreak_rf_sd"] = float(np.std(vals))
        r["tiebreak_rf_min"] = float(np.min(vals))
        r["tiebreak_rf_max"] = float(np.max(vals))
        # where the arm's own reported RF sits in its tie-break distribution
        own = r["UPGMA"]["robinson_foulds_normalized"]
        r["tiebreak_rf_percentile_of_reported"] = float(np.mean(np.array(vals) < own))
        mu = np.array(mono_u)
        r["tiebreak_mono_unrooted_min"] = int(mu.min())
        r["tiebreak_mono_unrooted_max"] = int(mu.max())
        r["tiebreak_mono_unrooted_mean"] = float(mu.mean())
        r["tiebreak_frac_draws_matching_reported_mono_unrooted"] = float(
            np.mean(mu == r["UPGMA"]["n_monophyletic_unrooted"]))
        r["tiebreak_mono_rooted_min"] = int(np.min(mono_r))
        r["tiebreak_mono_rooted_max"] = int(np.max(mono_r))
        cv.log(f"  {name:14} ME RF={r['ME']['robinson_foulds_normalized']:.4f} [{me_source}] "
               f"BME length NJ {L_start:.1f} -> own-NNI {L_end:.1f} -> FastME {L_fastme:.1f} | "
               f"additivity gap {r['additivity']['mean_relative_gap']:.4f}")
        cv.log(f"  {name:14} UPGMA RF={r['UPGMA']['robinson_foulds_normalized']:.4f} "
               f"(tie-break band {r['tiebreak_rf_min']:.4f}-{r['tiebreak_rf_max']:.4f}, "
               f"sd {r['tiebreak_rf_sd']:.4f}) | zero-len internal "
               f"{r['UPGMA']['zero_length_internal_branches']}/{r['UPGMA']['internal_branches']} "
               f"| mono rooted {r['UPGMA']['n_monophyletic_rooted']}/{r['UPGMA']['n_groups']}")
        results.append(r)
        for meth in ("UPGMA", "NJ", "ME"):
            rows.append({"arm": name, "tree_method": meth,
                         "additivity_mean_gap": r["additivity"]["mean_relative_gap"],
                         "frac_exactly_additive": r["additivity"]["fraction_exactly_additive"], "pearson": r["pearson"],
                         "spearman": r["spearman"], "kendall_tau_b": r["kendall_tau_b"],
                         "rmse_MY": r["rmse_MY"], "mae_MY": r["mae_MY"],
                         "n_distinct_values": r["n_distinct_values"],
                         "rf_normalized": r[meth]["robinson_foulds_normalized"],
                         "zero_len_internal": r[meth]["zero_length_internal_branches"],
                         "mono_rooted": r[meth]["n_monophyletic_rooted"],
                         "mono_unrooted": r[meth]["n_monophyletic_unrooted"],
                         "n_groups": r[meth]["n_groups"]})

    # species-block delete-d jackknife on the RF difference (predictions fixed)
    jack = {}
    base = {(a, b): v for (a, b), v in arms["CLASS_ARGMAX"].items()}
    for other in ("CLASS_EXPECTED", "RF_REG", "RF_REG_LOG"):
        for dd in JACKKNIFE_D:
            diffs = []
            rj = np.random.default_rng(SEED + dd)
            for _ in range(N_JACK):
                keep = sorted(rj.choice(sp_used, size=n - dd, replace=False).tolist())
                kpos = {s: i for i, s in enumerate(keep)}
                m = len(keep)
                A = np.zeros((m, m)); B = np.zeros((m, m))
                for (x, y), v in base.items():
                    if x in kpos and y in kpos:
                        A[kpos[x], kpos[y]] = A[kpos[y], kpos[x]] = v
                for (x, y), v in arms[other].items():
                    if x in kpos and y in kpos:
                        B[kpos[x], kpos[y]] = B[kpos[y], kpos[x]] = v
                tr2 = Tree(cv.reference_tree_to_formatted_newick(
                    cv.load_and_prune_reference_tree(keep)), format=1)
                diffs.append(rf_of_matrix(B, keep, tr2)[0] - rf_of_matrix(A, keep, tr2)[0])
            diffs = np.array(diffs)
            # These are the 2.5/97.5 percentiles of N_JACK heavily overlapping delete-d
            # subsamples of the same 107 species, so they are not a confidence interval:
            # the replicates are strongly positively correlated and the spread is far too
            # narrow to read as one. A delete-d jackknife needs the inflation factor
            # (n-d)/d (= (n-dd)/dd here), i.e. a standard error wider by about its square
            # root. The summary statistic to quote is therefore the direction (the
            # fraction of subsets favoring the other arm), reported at every d rather
            # than a percentile range at whichever d is most favorable.
            infl = (n - dd) / dd
            jack[f"{other}_vs_CLASS_ARGMAX_delete{dd}"] = {
                "mean_rf_difference": float(diffs.mean()),
                "subsample_percentile_range_2p5_97p5_NOT_a_CI": [
                    float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "jackknife_inflation_factor_n_minus_d_over_d": float(infl),
                "se_inflation_multiplier_sqrt": float(np.sqrt(infl)),
                "fraction_favoring_other": float((diffs < 0).mean()),
                "n_replicates": int(len(diffs)),
                "note": ("Percentile range is over overlapping subsamples and is NOT a "
                         "confidence interval; do not quote it as one, and do not quote a "
                         "single d in isolation -- report every d computed.")}
            cv.log(f"  jackknife d={dd} {other} vs CLASS_ARGMAX: mean {diffs.mean():+.4f}, "
                   f"favors {(diffs<0).mean()*100:.1f}% of {len(diffs)} subsets "
                   f"(percentile range [{np.percentile(diffs,2.5):+.4f},"
                   f"{np.percentile(diffs,97.5):+.4f}] is NOT a CI; "
                   f"SE inflation x{np.sqrt(infl):.1f})")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT, "continuous_tree_comparison.csv"), index=False)
    pd.DataFrame([{"sp1": a, "sp2": b, "true_my": true_sub[(a, b)],
                   **{arm: arms[arm][(a, b)] for arm in arms}} for a, b in sorted(true_sub)]
                 ).to_csv(os.path.join(OUT, "predicted_distances_all_arms.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"published_baseline_reported": {"rf_normalized": 0.884615,
                                                   "pearson": 0.833, "spearman": 0.878},
                   "reproduction_vs_published": repro,
                   "n_species": n, "n_pairs": len(keys),
                   "class_centroids_MY": mu_k.tolist(),
                   "jitter_sd_MY": JITTER_SD, "n_jitter": N_JITTER,
                   "arms": results, "jackknife": jack}, fh, indent=2)
    cv.log("")
    cv.log(tbl.to_string(index=False))
    cv.log(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
