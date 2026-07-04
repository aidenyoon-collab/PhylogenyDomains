#!/usr/bin/env python3
"""
Verify TimeTree Distance Scale

This script verifies what scale the TimeTree distances represent by comparing
tree distances to known divergence times from the TimeTree database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit("This script requires the `ete3` package. Please install it and retry.") from exc

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(message: str) -> None:
    """Print log message."""
    print(f"[verify_timetree] {message}")


def load_phylogeny_and_get_distances(path: str, species_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """
    Load phylogeny and get distances for specific species pairs.
    
    Returns dictionary mapping (species1, species2) -> distance
    """
    log(f"Loading phylogeny from {path}")
    tree = Tree(path, format=1)
    
    # Known tree name mappings
    TREE_NAME_MAP = {
        "Neogale vison": "Neovison vison",
        "Neogale_vison": "Neovison vison",
        "Bos grunniens": "Bos mutus grunniens",
        "Bos_grunniens": "Bos mutus grunniens",
        "Physeter catodon": "Physeter macrocephalus",
        "Physeter_catodon": "Physeter macrocephalus",
    }
    
    # Get all leaf names
    leaf_names = {leaf.name for leaf in tree.iter_leaves()}
    
    # Map species pairs to tree names
    distances = {}
    for sp1, sp2 in species_pairs:
        # Try to find species in tree
        sp1_tree = None
        sp2_tree = None
        
        # Try different name formats
        sp1_variants = [sp1, sp1.replace(" ", "_"), sp1.replace("_", " ")]
        sp2_variants = [sp2, sp2.replace(" ", "_"), sp2.replace("_", " ")]
        
        # Try exact match first
        for variant in sp1_variants:
            if variant in leaf_names:
                sp1_tree = variant
                break
        else:
            # Try case-insensitive
            for leaf in leaf_names:
                for variant in sp1_variants:
                    if variant.lower() == leaf.lower():
                        sp1_tree = leaf
                        break
                if sp1_tree:
                    break
        
        for variant in sp2_variants:
            if variant in leaf_names:
                sp2_tree = variant
                break
        else:
            # Try case-insensitive
            for leaf in leaf_names:
                for variant in sp2_variants:
                    if variant.lower() == leaf.lower():
                        sp2_tree = leaf
                        break
                if sp2_tree:
                    break
        
        if sp1_tree and sp2_tree:
            try:
                dist = tree.get_distance(sp1_tree, sp2_tree)
                distances[(sp1, sp2)] = dist
                log(f"  {sp1} - {sp2}: {dist:.2f}")
            except Exception as e:
                log(f"  ERROR: Could not compute distance for {sp1} - {sp2}: {e}")
        else:
            log(f"  WARNING: Could not find species in tree: {sp1}={sp1_tree}, {sp2}={sp2_tree}")
    
    return distances


def get_known_divergence_times() -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Return known divergence times from TimeTree database.
    Times are in millions of years (Myr).
    
    Returns dict mapping (species1, species2) -> {'divergence_time': float, 'source': str}
    """
    # Known divergence times (in millions of years)
    # These are approximate values - user should verify with TimeTree.org
    known_times = {
        ("Bos taurus", "Capra hircus"): {
            "divergence_time": 25.0,  # Cow-Goat (Bovidae split)
            "source": "TimeTree (approximate)"
        },
        ("Bos taurus", "Cavia porcellus"): {
            "divergence_time": 75.0,  # Cow-Guinea pig (more distant)
            "source": "TimeTree (approximate)"
        },
        ("Canis lupus familiaris", "Felis catus"): {
            "divergence_time": 55.0,  # Dog-Cat (Carnivora split)
            "source": "TimeTree (approximate)"
        },
        ("Mus musculus", "Rattus norvegicus"): {
            "divergence_time": 12.0,  # Mouse-Rat
            "source": "TimeTree (approximate)"
        },
        ("Callithrix jacchus", "Cebus imitator"): {
            "divergence_time": 20.0,  # New World monkeys
            "source": "TimeTree (approximate)"
        },
    }
    
    return known_times


def main():
    """Main verification pipeline."""
    log("Starting TimeTree distance verification")
    
    # Key species pairs to verify (using species that are likely in the tree)
    # Note: Using underscore format as tree uses underscores
    species_pairs = [
        ("Bos taurus", "Capra hircus"),  # Cow-Goat (closely related)
        ("Bos taurus", "Cavia porcellus"),  # Cow-Guinea pig (more distant)
        ("Canis lupus familiaris", "Felis catus"),  # Dog-Cat (if available)
        ("Mus musculus", "Rattus norvegicus"),  # Mouse-Rat (if available)
    ]
    
    # Also try to find any primate pairs if available
    primate_species = ["Callithrix jacchus", "Cebus imitator", "Cercocebus atys", "Chlorocebus sabaeus"]
    if len(primate_species) >= 2:
        species_pairs.append((primate_species[0], primate_species[1]))
    
    # Get distances from tree
    tree_path = os.path.join(DATA_RAW, "MammalsPhylogeny.nwk")
    tree_distances = load_phylogeny_and_get_distances(tree_path, species_pairs)
    
    # Get known divergence times
    known_times = get_known_divergence_times()
    
    # Compare
    log("\n=== Comparison: Tree Distances vs Known Divergence Times ===")
    results = []
    
    for pair in species_pairs:
        if pair in tree_distances and pair in known_times:
            tree_dist = tree_distances[pair]
            known_time = known_times[pair]["divergence_time"]
            
            # Calculate ratio
            ratio = tree_dist / known_time if known_time > 0 else np.nan
            
            # Check if it's approximately double (2×)
            is_double = 1.8 <= ratio <= 2.2
            
            results.append({
                "species1": pair[0],
                "species2": pair[1],
                "tree_distance": tree_dist,
                "known_divergence_myr": known_time,
                "ratio": ratio,
                "is_approximately_2x": is_double,
                "interpretation": "2× divergence time" if is_double else f"{ratio:.2f}× divergence time"
            })
            
            log(f"\n{pair[0]} - {pair[1]}:")
            log(f"  Tree distance: {tree_dist:.2f}")
            log(f"  Known divergence: {known_time:.2f} Myr")
            log(f"  Ratio: {ratio:.2f}")
            log(f"  Interpretation: {results[-1]['interpretation']}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_path = os.path.join(RESULTS, "timetree_comparison.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    log(f"\nSaved comparison table to {output_path}")
    
    # Create visualization
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tree_dists = [r["tree_distance"] for r in results]
        known_times_list = [r["known_divergence_myr"] for r in results]
        pair_labels = [f"{r['species1']}\n{r['species2']}" for r in results]
        
        x = np.arange(len(results))
        width = 0.35
        
        ax.bar(x - width/2, tree_dists, width, label="Tree Distance", alpha=0.7)
        ax.bar(x + width/2, known_times_list, width, label="Known Divergence (Myr)", alpha=0.7)
        
        # Add 2× line for reference
        ax.plot([-0.5, len(results)-0.5], [0, max(known_times_list)*2.2], 'r--', 
                label="2× Divergence Time", linewidth=2, alpha=0.5)
        
        ax.set_xlabel("Species Pair", fontsize=12)
        ax.set_ylabel("Distance / Time (Myr)", fontsize=12)
        ax.set_title("TimeTree Distance vs Known Divergence Times", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS, "timetree_verification_plot.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        log(f"Saved verification plot to {plot_path}")
    
    # Generate report
    report_path = os.path.join(RESULTS, "timetree_verification_report.md")
    with open(report_path, "w") as f:
        f.write("# TimeTree Distance Verification Report\n\n")
        f.write("## Summary\n\n")
        
        if len(results) > 0:
            avg_ratio = df["ratio"].mean()
            f.write(f"- Average ratio (Tree Distance / Known Divergence): {avg_ratio:.2f}\n")
            f.write(f"- Range: {df['ratio'].min():.2f} to {df['ratio'].max():.2f}\n\n")
            
            if df["is_approximately_2x"].sum() >= len(results) * 0.8:
                f.write("**CONCLUSION: TimeTree distances appear to be approximately 2× divergence times (double distances).**\n\n")
            elif 0.9 <= avg_ratio <= 1.1:
                f.write("**CONCLUSION: TimeTree distances appear to be approximately 1× divergence times (single distances).**\n\n")
            else:
                f.write(f"**CONCLUSION: TimeTree distances appear to be approximately {avg_ratio:.2f}× divergence times.**\n\n")
        
        f.write("## Detailed Comparisons\n\n")
        f.write("| Species 1 | Species 2 | Tree Distance | Known Divergence (Myr) | Ratio | Interpretation |\n")
        f.write("|-----------|-----------|---------------|------------------------|-------|----------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['species1']} | {row['species2']} | {row['tree_distance']:.2f} | "
                   f"{row['known_divergence_myr']:.2f} | {row['ratio']:.2f} | {row['interpretation']} |\n")
        f.write("\n\n## Notes\n\n")
        f.write("- Divergence times are approximate and should be verified with TimeTree.org\n")
        f.write("- Tree distances are patristic distances from the Newick tree file\n")
        f.write("- If ratio ≈ 2.0, distances are double distances (2× divergence time)\n")
        f.write("- If ratio ≈ 1.0, distances are single distances (1× divergence time)\n")
    
    log(f"Saved verification report to {report_path}")
    log("\n=== Verification Complete ===")


if __name__ == "__main__":
    main()

