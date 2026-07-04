#!/usr/bin/env python3
"""
Part-2 figures (classifier distance-class result + tree reconstruction).
Chart types follow render_part2_prototypes.py; this script emits the publication
versions with no "PROTO" prefix.

Rendered ONLY from the real saved pipeline outputs in
results/classification_cv_tree_reconstruction/. Nothing is re-run; no data is altered,
jittered, or trimmed. Every statistic printed on a figure is read directly from the
OFFICIAL comparison_summary.json (Pearson/Spearman) or computed by exact crosstab of
the saved predictions (cell counts / recall %). Predicted distances are the train-only
class centroids straight out of class_representative_distances.csv.

Outputs (each as vector .pdf + 600-dpi .png via pubstyle.save):
  figures/part2_confusion_matrix          confusion matrix, the honest headline for a
                                          4-class classifier (counts + row-% recall)
  figures/part2_distance_recovery         true patristic distance per PREDICTED class
                                          (box), class centroid (predicted MY) overlaid
  figures/supp_part2_true_vs_predicted    SUPPLEMENTARY scatter/hexbin true vs predicted;
                                          keeps the on-figure note that the y=x diagonal
                                          implies a continuity the 4-class model does not
                                          produce. Pearson/Spearman = OFFICIAL values.

Source: results/classification_cv_tree_reconstruction/{predictions_aggregated.csv,
        class_representative_distances.csv, comparison_summary.json} and the
        full-precision MammalsPhylogeny.nwk (repo root) for true patristic distances
        - NOT the rounded reference_tree_pruned.nwk (see load(), which deliberately
        uses the full-precision tree so the plotted stats match comparison_summary.json).
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from ete3 import Tree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "figures"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import pubstyle  # noqa: E402
pubstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from classification_cv_tree_reconstruction import format_species_label_for_newick, compute_patristic_matrix  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score  # noqa: E402

CV = os.path.join(REPO, "results", "classification_cv_tree_reconstruction")
OUT = os.path.join(REPO, "figures")
ORDER = ["D0", "D1", "D2", "D3"]


def load():
    """Read predictions, attach the true patristic MY from the reference tree, and read
    the train-only class centroids + the official comparison statistics. No mutation of
    the saved values beyond joining true distances onto each pair."""
    pred = pd.read_csv(os.path.join(CV, "predictions_aggregated.csv"))
    # Full-precision true patristic distances from the SAME tree the pipeline used
    # (MammalsPhylogeny.nwk at repo root), so figure stats match comparison_summary.json
    # exactly. The saved reference_tree_pruned.nwk is a rounded Newick round-trip whose
    # ties shift rank-Spearman (~0.878 -> ~0.86) - wrong to display next to a plot of it.
    species = sorted(set(pred["sp1"]).union(set(pred["sp2"])))
    Dpat = compute_patristic_matrix(os.path.join(REPO, "MammalsPhylogeny.nwk"), species)
    pred = pred.copy()
    pred["true_my"] = [float(Dpat.loc[r["sp1"], r["sp2"]]) for _, r in pred.iterrows()]
    mu = pd.read_csv(os.path.join(CV, "class_representative_distances.csv"))
    mu_k = dict(zip(mu["label"], mu["mu_k_MY"]))
    summ = json.load(open(os.path.join(CV, "comparison_summary.json")))
    return pred, mu_k, summ


def fig_confusion(pred):
    """Confusion matrix of pooled cross-validated predictions. Color = row-normalized
    recall (%); each cell prints the exact pair COUNT and the row %. Per-class PRECISION
    is shown under each predicted-class label, and overall accuracy / balanced accuracy /
    Cohen's kappa vs the majority-class baseline are reported below the axis."""
    yt, yp = pred["y_true_label"], pred["y_pred_label"]
    ct = pd.crosstab(yt, yp).reindex(index=ORDER, columns=ORDER, fill_value=0)
    row_pct = ct.div(ct.sum(axis=1), axis=0) * 100.0          # recall
    col_sum = ct.sum(axis=0)
    precision = {c: (ct.loc[c, c] / col_sum[c] if col_sum[c] > 0 else float("nan"))
                 for c in ORDER}
    acc = float((yt == yp).mean())
    bal_acc = float(balanced_accuracy_score(yt, yp))
    kappa = float(cohen_kappa_score(yt, yp))
    baseline = float(yt.value_counts(normalize=True).max())   # always-predict-majority

    fig, ax = plt.subplots(figsize=(pubstyle.SINGLE_COL, pubstyle.SINGLE_COL))
    im = ax.imshow(row_pct.values, cmap="viridis", vmin=0, vmax=100, aspect="equal")
    for i in range(4):
        for j in range(4):
            c, p = int(ct.values[i, j]), row_pct.values[i, j]
            ax.text(j, i, f"{c}\n{p:.0f}%", ha="center", va="center",
                    fontsize=6, color="white" if p < 55 else "black")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{c}\nprec {precision[c]:.2f}" for c in ORDER])
    ax.set_yticks(range(4)); ax.set_yticklabels(ORDER)
    ax.set_xlabel("Predicted distance class"); ax.set_ylabel("True distance class")
    ax.set_title("Pairwise distance-class confusion", fontsize=7)
    # Summary metrics below the axis (recall lives in the cells; precision under columns).
    ax.text(0.5, -0.30,
            f"accuracy {acc:.2f} · balanced accuracy {bal_acc:.2f} · "
            f"Cohen's $\\kappa$ {kappa:.2f}  (majority-class baseline {baseline:.2f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.4, color="0.25")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row %  (recall)")
    return pubstyle.save(fig, os.path.join(OUT, "part2_confusion_matrix")), plt.close(fig)


def fig_distance_recovery(pred, mu_k):
    """Distribution of TRUE patristic distance (MY) within each PREDICTED class, with the
    single class centroid (the MY value each predicted class maps to) overlaid. Outliers
    are SHOWN (showfliers=True) and each box is annotated with its n and % of all pairs,
    so the dominant, degenerate D2 class is not silently hidden."""
    fig, ax = plt.subplots(figsize=(pubstyle.SINGLE_COL, 0.8 * pubstyle.SINGLE_COL))
    data = [pred.loc[pred["y_pred_label"] == c, "true_my"].values for c in ORDER]
    total = sum(len(d) for d in data)
    bp = ax.boxplot(data, positions=range(4), widths=0.6, showfliers=True,
                    flierprops=dict(marker="o", markersize=1.2, markerfacecolor="0.4",
                                    markeredgecolor="none", alpha=0.30),
                    patch_artist=True, medianprops=dict(color="black", lw=0.9))
    for patch, col in zip(bp["boxes"], pubstyle.OKABE_ITO):
        patch.set_facecolor(col); patch.set_alpha(0.55)
        patch.set_edgecolor("0.3"); patch.set_linewidth(0.6)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_linewidth(0.6); w.set_color("0.3")
    ax.scatter(range(4), [mu_k[c] for c in ORDER], marker="D", s=14, color="black",
               zorder=5, label="class centroid (predicted MY)")
    # n and % of all pairs above each box (D2 holds ~half the pairs).
    ymax = max(d.max() for d in data)
    ax.set_ylim(top=ymax * 1.20)
    for i, d in enumerate(data):
        ax.text(i, ymax * 1.04, f"n={len(d)}\n{100 * len(d) / total:.0f}%",
                ha="center", va="bottom", fontsize=4.8, color="0.3")
    ax.set_xticks(range(4)); ax.set_xticklabels(ORDER)
    ax.set_xlabel("Predicted distance class"); ax.set_ylabel("True patristic distance (MY)")
    ax.set_title("Distance recovery by predicted class", fontsize=7)
    ax.text(0.5, -0.30,
            "D2 holds ~half of all pairs; its tight spread reflects the 15-MY bin width and "
            "the ~188-MY TimeTree tie, not resolving power.",
            transform=ax.transAxes, ha="center", va="top", fontsize=4.8, color="0.35")
    # Legend OUTSIDE the axes so it never overlaps the D0/D3 boxes or whiskers.
    pubstyle.legend_outside(ax)
    return pubstyle.save(fig, os.path.join(OUT, "part2_distance_recovery")), plt.close(fig)


def fig_supp_true_vs_predicted(pred, summ):
    """SUPPLEMENTARY: true vs predicted patristic distance (hexbin, log density). Stats
    are the OFFICIAL pipeline values; on-figure note flags that predicted distance is
    discrete (4 class centroids) so the y=x diagonal implies a false continuity."""
    t, p = pred["true_my"].values, pred["predicted_distance_MY"].values
    rp, rs = pearsonr(t, p)[0], spearmanr(t, p)[0]   # computed on the EXACT plotted vectors
    # full-precision true_my => these equal the official comparison_summary.json values
    assert abs(rp - summ["pearson_r"]) < 5e-3 and abs(rs - summ["spearman_r"]) < 5e-3, \
        f"plotted stats (P={rp:.4f}, S={rs:.4f}) diverge from official " \
        f"(P={summ['pearson_r']:.4f}, S={summ['spearman_r']:.4f})"
    fig, ax = plt.subplots(figsize=(pubstyle.SINGLE_COL, 0.9 * pubstyle.SINGLE_COL))
    lim = [0, max(t.max(), p.max()) * 1.03]
    ax.plot(lim, lim, ls="--", lw=0.7, color="0.55")
    hb = ax.hexbin(t, p, gridsize=32, cmap="viridis", norm=LogNorm(vmin=1), mincnt=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True patristic distance (MY)"); ax.set_ylabel("Predicted distance (MY)")
    # Significance = MANTEL species-permutation p, not the invalid i.i.d. p over
    # 5,671 non-independent pairs; the independent unit is the 107 species.
    mant = json.load(open(os.path.join(CV, "..", "mantel_significance.json")))["part2_predicted_vs_patristic"]
    mp = "≤ 1e-4" if mant["pearson_p_mantel"] <= 1.0 / (mant["n_perm"] + 1) + 1e-12 \
        else f"= {mant['pearson_p_mantel']:.1e}"
    pubstyle.annotate(ax, f"Pearson $r$={rp:.2f}\nSpearman $\\rho$={rs:.2f}\n"
                      f"Mantel P {mp} ({mant['n_species']} species)",
                      loc="upper left", fontsize=6)
    # Keep the false-continuity caveat BELOW the axes, well clear of the hexbins/colorbar.
    ax.text(0.5, -0.26,
            "Note: predicted distance takes only 4 values (class centroids);\n"
            "the y=x line implies a continuity the classifier does not produce.",
            transform=ax.transAxes, ha="center", va="top", fontsize=5, color="0.35")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("pairs per bin (log)")
    return pubstyle.save(fig, os.path.join(OUT, "supp_part2_true_vs_predicted")), plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pred, mu_k, summ = load()
    # Integrity guard: every printed predicted distance must equal the saved centroid.
    muk_pred = pred["y_pred_label"].map(mu_k).values
    assert np.allclose(muk_pred, pred["predicted_distance_MY"].values), \
        "predicted_distance_MY does not match class_representative_distances.csv centroids"
    assert len(pred) == summ["n_pairs"], \
        f"pair count {len(pred)} != official n_pairs {summ['n_pairs']}"
    fig_confusion(pred)
    fig_distance_recovery(pred, mu_k)
    fig_supp_true_vs_predicted(pred, summ)
    print(f"n={len(pred)} pairs (official n_pairs={summ['n_pairs']}); "
          f"pearson_r={summ['pearson_r']:.6f}, spearman_r={summ['spearman_r']:.6f}")
    print(f"wrote part2_confusion_matrix, part2_distance_recovery, "
          f"supp_part2_true_vs_predicted (.pdf + .png) to {OUT}")


if __name__ == "__main__":
    main()
