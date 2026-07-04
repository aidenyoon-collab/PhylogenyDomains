#!/usr/bin/env python3
"""
Plot clade-by-domain heatmap from filtered enriched PFAM domains.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILTERED_DIR = PROJECT_ROOT / "results" / "clade_domain_enrichment" / "filtered"
INPUT_FILE = FILTERED_DIR / "all_clades_filtered_enriched_domains_summary.csv"
PNG_OUT = FILTERED_DIR / "enriched_domains_heatmap_log2FC.png"
PDF_OUT = FILTERED_DIR / "enriched_domains_heatmap_log2FC.pdf"

CLADE_ORDER = [
    "Primates",
    "Glires",
    "Marsupials",
    "Afrotheria",
    "Laurasiatheria_Carnivora",
    "Laurasiatheria_Cetartiodactyla",
    "Laurasiatheria_Chiroptera",
]
P_ADJ_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.5
EXCLUDED_NAME_TERMS = ("duf", "upf", "unknown function", "uncharacterised")


def build_domain_label_map(df: pd.DataFrame) -> pd.DataFrame:
    labels = df[["pfam_id", "domain_name"]].drop_duplicates().copy()
    name_counts = labels["domain_name"].value_counts()
    duplicate_names = set(name_counts[name_counts > 1].index)
    labels["domain_label"] = labels["domain_name"]
    if duplicate_names:
        dup_mask = labels["domain_name"].isin(duplicate_names)
        labels.loc[dup_mask, "domain_label"] = (
            labels.loc[dup_mask, "domain_name"] + " (" + labels.loc[dup_mask, "pfam_id"] + ")"
        )
    return labels


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required = {"clade", "pfam_id", "domain_name", "log2FC", "p_adjusted"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input missing required columns: {sorted(missing)}")

    df = df.copy()
    df["clade"] = df["clade"].astype(str)
    df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
    df["p_adjusted"] = pd.to_numeric(df["p_adjusted"], errors="coerce")
    df = df.dropna(subset=["pfam_id", "domain_name", "clade", "log2FC"])
    df = df[
        (df["p_adjusted"] < P_ADJ_THRESHOLD) & (df["log2FC"] >= LOG2FC_THRESHOLD)
    ].copy()

    name_lower = df["domain_name"].astype(str).str.lower()
    excluded_mask = name_lower.str.contains("|".join(EXCLUDED_NAME_TERMS), regex=True)
    df = df.loc[~excluded_mask].copy()

    if df.empty:
        raise SystemExit("Filtered summary is empty after cleanup; nothing to plot.")

    domain_labels = build_domain_label_map(df)
    merged = df.merge(domain_labels, on=["pfam_id", "domain_name"], how="left")

    matrix = merged.pivot_table(
        index="domain_label",
        columns="clade",
        values="log2FC",
        aggfunc="max",
        fill_value=0.0,
    )
    matrix = matrix.reindex(columns=CLADE_ORDER, fill_value=0.0)

    # Keep only domains significant in at least one clade (non-zero after fill).
    matrix = matrix.loc[(matrix.abs().max(axis=1) > 0)]
    if matrix.empty:
        raise SystemExit("No non-zero domains available for heatmap after filtering.")

    n_domains = matrix.shape[0]
    print(f"[heatmap] final rows before plotting: {n_domains}")

    # Group rows by clade with highest log2FC.
    dominant_clade = matrix.idxmax(axis=1)
    dominant_order = dominant_clade.map({c: i for i, c in enumerate(CLADE_ORDER)})
    max_fc = matrix.max(axis=1)
    row_order = (
        pd.DataFrame(
            {"label": matrix.index, "dominant_order": dominant_order, "max_fc": max_fc}
        )
        .sort_values(
            by=["dominant_order", "max_fc", "label"],
            ascending=[True, False, True],
            kind="stable",
        )["label"]
        .tolist()
    )
    matrix = matrix.loc[row_order]

    # Fixed publication layout targets.
    fig_w = 14.0
    fig_h = max(10.0, 0.3 * n_domains)
    matrix_clipped = matrix.clip(lower=-2.0, upper=2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    hm = sns.heatmap(
        matrix_clipped,
        ax=ax,
        cmap="RdBu_r",
        vmin=-2.0,
        vmax=2.0,
        center=0.0,
        cbar_kws={"label": "log2 fold change"},
        linewidths=0.05,
        linecolor="white",
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xlabel("Mammalian clade")
    ax.set_ylabel("")
    ax.set_title("Filtered Enriched PFAM Domains Across Mammalian Clades")

    ax.tick_params(axis="x", labelrotation=30, labelsize=11)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    # Place domain labels on the right side.
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", labelsize=8)

    # Keep enough right margin for full domain names.
    fig.subplots_adjust(left=0.12, right=0.74, top=0.95, bottom=0.08)
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[heatmap] matrix shape={matrix_clipped.shape}")
    print(f"[heatmap] wrote PNG -> {PNG_OUT}")
    print(f"[heatmap] wrote PDF -> {PDF_OUT}")


if __name__ == "__main__":
    main()
