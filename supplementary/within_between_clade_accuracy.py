#!/usr/bin/env python3
"""
Split the pairwise divergence-time classifier's accuracy by whether a species pair
crosses a clade boundary or sits inside a single clade.

A classifier trained on pairwise domain-count differences can score well for either of
two reasons: it may track divergence time, or it may simply recognize clade identity,
since for most pairs of mammals the clades the two species belong to already pin down
the distance class. Splitting the predictions separates those explanations. Between-clade
pairs are the easy case, because clade identity alone nearly determines the class.
Within-clade pairs hold clade identity constant, so only divergence time varies, and
within-clade accuracy is therefore the stricter measure of how much time-related signal
the model actually carries.

The script reads the pooled cross-validated per-pair predictions produced by
classification_cv_tree_reconstruction.py and retrains nothing, so its numbers describe
exactly the model behind the main classification results. The four ordinal distance
classes D0 to D3 are that model's classes, set by cutoffs at 150, 175 and 190 million
years. Clade assignments come from build_clade_mapping() in
elastic_net_regression_stratified_species_cv.py, the same mapping used by the clade
enrichment and interpretation analyses. If any species in the prediction table is absent
from that mapping the script stops with an error rather than report a split computed over
part of the cohort.

Reading the output needs the class distribution, not the accuracies alone. Within a clade
only the closest distance classes occur at all, so the within-clade problem is effectively
binary and its accuracy is lifted by whichever class dominates; summary.json therefore
reports true_class_counts alongside the accuracy for each split. The per-class recalls are
the quantity to compare: within_clade_recall_by_class gives n_true, n_correct and recall
for each of D0 to D3, with recall null where that class never occurs within a clade, and a
recall of 1.0 on a class holding only a handful of true pairs reflects the sample size
rather than skill. within_clade_confusion.csv carries the same counts as a 4 x 4 table
whose rows are true classes and columns are predicted classes, reindexed over all four
classes and zero-filled so its shape does not depend on which classes happen to appear.

Writes results/within_between_clade/ (summary.json + within_clade_confusion.csv).
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import classification_cv_tree_reconstruction as cv  # noqa: E402
import elastic_net_regression_stratified_species_cv as en  # noqa: E402

PRED = os.path.join(REPO, "results", "classification_cv_tree_reconstruction",
                    "predictions_aggregated.csv")
OUT = os.path.join(REPO, "results", "within_between_clade")
N_CLASSES = 4


def main() -> None:
    cv.ensure_dir(OUT)
    sp2c = {k.replace("_", " "): v for k, v in en.build_clade_mapping().items()}
    d = pd.read_csv(PRED)

    c1 = d.sp1.map(lambda s: sp2c.get(str(s).replace("_", " ")))
    c2 = d.sp2.map(lambda s: sp2c.get(str(s).replace("_", " ")))
    unmapped = sorted({str(s) for s, c in zip(d.sp1, c1) if pd.isna(c)}
                      | {str(s) for s, c in zip(d.sp2, c2) if pd.isna(c)})
    if unmapped:
        raise SystemExit(f"FATAL: {len(unmapped)} species missing from the clade mapping: "
                         f"{unmapped[:10]} -- refusing to report a partial split.")
    cv.log(f"{len(d)} aggregated pairs, all species mapped to {len(set(sp2c.values()))} clades")

    within = c1 == c2
    summary = {"n_pairs_total": int(len(d)), "overall_accuracy": float((d.y_true == d.y_pred).mean())}
    for label, mask in (("between_clade", ~within), ("within_clade", within)):
        sub = d[mask]
        summary[label] = {"n_pairs": int(len(sub)),
                          "accuracy": float((sub.y_true == sub.y_pred).mean()),
                          "true_class_counts": {int(k): int(v) for k, v
                                                in sub.y_true.value_counts().sort_index().items()}}
        cv.log(f"  {label:14} n={len(sub):5} accuracy={summary[label]['accuracy']:.4f} "
               f"true classes present: {sorted(summary[label]['true_class_counts'])}")

    sub = d[within]
    cm = (pd.crosstab(sub.y_true, sub.y_pred)
          .reindex(index=range(N_CLASSES), columns=range(N_CLASSES), fill_value=0))
    cm.to_csv(os.path.join(OUT, "within_clade_confusion.csv"))
    recalls = {}
    for k in range(N_CLASSES):
        row = cm.loc[k]
        n = int(row.sum())
        recalls[f"D{k}"] = {"n_true": n, "n_correct": int(row[k]),
                            "recall": (float(row[k]) / n) if n else None}
        if n:
            cv.log(f"    within-clade D{k} recall = {int(row[k])}/{n} = {row[k] / n:.3f}")
    summary["within_clade_recall_by_class"] = recalls
    summary["interpretation"] = (
        "Accuracy falls from "
        f"{summary['between_clade']['accuracy']:.4f} on the {summary['between_clade']['n_pairs']} "
        f"pairs that cross a clade boundary to {summary['within_clade']['accuracy']:.4f} on the "
        f"{summary['within_clade']['n_pairs']} pairs inside a single clade. The within-clade "
        "figure is moreover flattered by the class distribution: within a clade only the two "
        "closest classes occur at all, so the task there is effectively binary and is dominated "
        "by the majority class, while the minority class D1 is recovered "
        f"{recalls['D1']['n_correct']}/{recalls['D1']['n_true']} of the time. Once clade identity "
        "is held constant the model retains little ability to order divergence times.")
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    cv.log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
