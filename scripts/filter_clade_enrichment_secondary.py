#!/usr/bin/env python3
"""
Apply secondary filtering to per-clade enriched PFAM domain outputs.

Input:
  results/clade_domain_enrichment/enriched_domains_<clade>.csv

Output:
  results/clade_domain_enrichment/filtered/filtered_enriched_domains_<clade>.csv
  results/clade_domain_enrichment/filtered/all_clades_filtered_enriched_domains_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "results" / "clade_domain_enrichment"
OUTPUT_DIR = INPUT_DIR / "filtered"

P_ADJ_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.5


@dataclass(frozen=True)
class CanonicalCols:
    domain: str
    p_adjusted: str
    log2fc: str


def pick_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def split_domain_field(value: object) -> tuple[str, str]:
    text = "" if pd.isna(value) else str(value).strip()
    if ":" in text:
        pfam_id, domain_name = text.split(":", 1)
        return pfam_id.strip(), domain_name.strip()
    return text, text


def build_canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    domain_col = pick_existing_column(df, ("domain", "domain_name", "pfam_domain"))
    if domain_col is None:
        raise ValueError("Missing domain column (expected one of: domain, domain_name)")

    p_adj_col = pick_existing_column(
        df, ("p_adjusted", "adjusted_p_value", "adj_p_value", "fdr", "q_value")
    )
    if p_adj_col is None:
        raise ValueError("Missing adjusted p-value column")

    log2fc_col = pick_existing_column(df, ("log2FC", "log2_fc", "log2foldchange"))
    out = pd.DataFrame()
    out["domain"] = df[domain_col]
    out["p_adjusted"] = pd.to_numeric(df[p_adj_col], errors="coerce")

    if log2fc_col is not None:
        out["log2FC"] = pd.to_numeric(df[log2fc_col], errors="coerce")
    else:
        fold_col = pick_existing_column(df, ("fold_change", "fc", "foldchange"))
        if fold_col is None:
            raise ValueError("Missing log2FC and fold-change columns")
        fold = pd.to_numeric(df[fold_col], errors="coerce")
        out["log2FC"] = np.where(fold > 0, np.log2(fold), np.nan)

    pfam_parts = out["domain"].map(split_domain_field)
    out["pfam_id"] = pfam_parts.map(lambda x: x[0])
    out["domain_name"] = pfam_parts.map(lambda x: x[1])

    out = out[["pfam_id", "domain_name", "log2FC", "p_adjusted"]]
    out = out.dropna(subset=["pfam_id", "domain_name", "log2FC", "p_adjusted"])
    return out


def clade_from_filename(path: Path) -> str:
    prefix = "enriched_domains_"
    stem = path.stem
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob("enriched_domains_*.csv"))
    if not input_files:
        raise SystemExit(f"No per-clade files found in: {INPUT_DIR}")

    combined_frames: list[pd.DataFrame] = []
    for in_path in input_files:
        clade = clade_from_filename(in_path)
        raw = pd.read_csv(in_path)
        canonical = build_canonical_frame(raw)
        filtered = canonical[
            (canonical["p_adjusted"] < P_ADJ_THRESHOLD)
            & (canonical["log2FC"] >= LOG2FC_THRESHOLD)
        ].copy()
        filtered = filtered.sort_values(
            by=["p_adjusted", "domain_name"], ascending=[True, True], kind="stable"
        ).reset_index(drop=True)

        out_path = OUTPUT_DIR / f"filtered_enriched_domains_{clade}.csv"
        filtered.to_csv(out_path, index=False)
        print(
            f"[secondary_filter] {clade}: input={len(raw)} "
            f"passing={len(filtered)} -> {out_path}"
        )

        if not filtered.empty:
            with_clade = filtered.copy()
            with_clade.insert(0, "clade", clade)
            combined_frames.append(with_clade)

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined = combined.sort_values(
            by=["clade", "p_adjusted", "domain_name"],
            ascending=[True, True, True],
            kind="stable",
        ).reset_index(drop=True)
    else:
        combined = pd.DataFrame(
            columns=["clade", "pfam_id", "domain_name", "log2FC", "p_adjusted"]
        )

    combined_path = OUTPUT_DIR / "all_clades_filtered_enriched_domains_summary.csv"
    combined.to_csv(combined_path, index=False)
    print(
        f"[secondary_filter] combined: rows={len(combined)} "
        f"-> {combined_path}"
    )


if __name__ == "__main__":
    main()
