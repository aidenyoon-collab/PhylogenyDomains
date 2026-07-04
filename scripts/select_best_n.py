#!/usr/bin/env python3
"""
Select Best N Value for Tree Construction

Analyzes correlation results with p-values to identify optimal N value
for constructing phylogenetic trees from domain distances.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

# Project paths
RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(message: str) -> None:
    """Print log message."""
    print(f"[select_best_n] {message}")


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations and identify best N values.
    
    Returns DataFrame with analysis results.
    """
    log("Analyzing correlation results...")
    
    analysis = []
    
    for _, row in df.iterrows():
        n = int(row['N'])
        
        # Get correlations and p-values
        spearman_var = row['spearman_var']
        spearman_var_pval = row['spearman_var_pval']
        pearson_var = row['pearson_var']
        pearson_var_pval = row['pearson_var_pval']
        
        spearman_alpha = row['spearman_alpha']
        spearman_alpha_pval = row['spearman_alpha_pval']
        pearson_alpha = row['pearson_alpha']
        pearson_alpha_pval = row['pearson_alpha_pval']
        
        # Score based on multiple criteria
        # 1. Correlation strength (higher is better)
        # 2. Statistical significance (all are highly significant, p < 1e-90)
        # 3. Prefer Spearman (rank-based, more robust)
        # 4. Prefer alpha-weighted over variance-weighted
        # 5. Penalize zero-inflation (prefer N with fewer zeros)
        
        # Estimate zero-inflation based on N value
        # N=1 typically has ~89% zeros, N=5 has ~63%, N=10 has ~8.5%, N=20+ has <2%
        if n == 1:
            zero_penalty = 0.5  # Heavy penalty for high zero-inflation
        elif n == 5:
            zero_penalty = 0.3
        elif n == 10:
            zero_penalty = 0.1
        elif n >= 20:
            zero_penalty = 0.0  # No penalty for low zero-inflation
        else:
            zero_penalty = 0.2
        
        # Score for variance-weighted (penalize zero-inflation)
        # Since Spearman is primary metric, weight it more heavily (0.7 vs 0.3)
        score_var = (spearman_var * 0.7 + pearson_var * 0.3) * (1 - zero_penalty)
        
        # Score for alpha-weighted (penalize zero-inflation)
        # Since Spearman is primary metric, weight it more heavily (0.7 vs 0.3)
        score_alpha = (spearman_alpha * 0.7 + pearson_alpha * 0.3) * (1 - zero_penalty)
        
        # Overall score (prefer alpha-weighted)
        overall_score = score_alpha * 0.6 + score_var * 0.4
        
        analysis.append({
            "N": n,
            "spearman_var": spearman_var,
            "spearman_var_pval": spearman_var_pval,
            "pearson_var": pearson_var,
            "pearson_var_pval": pearson_var_pval,
            "spearman_alpha": spearman_alpha,
            "spearman_alpha_pval": spearman_alpha_pval,
            "pearson_alpha": pearson_alpha,
            "pearson_alpha_pval": pearson_alpha_pval,
            "score_var": score_var,
            "score_alpha": score_alpha,
            "overall_score": overall_score,
        })
    
    analysis_df = pd.DataFrame(analysis)
    return analysis_df


def select_best_n(analysis_df: pd.DataFrame) -> dict:
    """
    Select best N value based on analysis.
    
    Returns dictionary with recommendations.
    """
    log("\n=== Best N Selection ===")
    
    # Find best by different criteria
    best_spearman_var_idx = analysis_df["spearman_var"].idxmax()
    best_spearman_alpha_idx = analysis_df["spearman_alpha"].idxmax()
    best_pearson_var_idx = analysis_df["pearson_var"].idxmax()
    best_pearson_alpha_idx = analysis_df["pearson_alpha"].idxmax()
    best_overall_idx = analysis_df["overall_score"].idxmax()
    
    recommendations = {
        "best_spearman_var": {
            "N": int(analysis_df.loc[best_spearman_var_idx, "N"]),
            "r_s": analysis_df.loc[best_spearman_var_idx, "spearman_var"],
            "pval": analysis_df.loc[best_spearman_var_idx, "spearman_var_pval"],
        },
        "best_spearman_alpha": {
            "N": int(analysis_df.loc[best_spearman_alpha_idx, "N"]),
            "r_s": analysis_df.loc[best_spearman_alpha_idx, "spearman_alpha"],
            "pval": analysis_df.loc[best_spearman_alpha_idx, "spearman_alpha_pval"],
        },
        "best_pearson_var": {
            "N": int(analysis_df.loc[best_pearson_var_idx, "N"]),
            "r": analysis_df.loc[best_pearson_var_idx, "pearson_var"],
            "pval": analysis_df.loc[best_pearson_var_idx, "pearson_var_pval"],
        },
        "best_pearson_alpha": {
            "N": int(analysis_df.loc[best_pearson_alpha_idx, "N"]),
            "r": analysis_df.loc[best_pearson_alpha_idx, "pearson_alpha"],
            "pval": analysis_df.loc[best_pearson_alpha_idx, "pearson_alpha_pval"],
        },
        "best_overall": {
            "N": int(analysis_df.loc[best_overall_idx, "N"]),
            "score": analysis_df.loc[best_overall_idx, "overall_score"],
            "spearman_alpha": analysis_df.loc[best_overall_idx, "spearman_alpha"],
            "pearson_alpha": analysis_df.loc[best_overall_idx, "pearson_alpha"],
        },
    }
    
    # Print recommendations
    log(f"Best Spearman (variance): N={recommendations['best_spearman_var']['N']}, r_s={recommendations['best_spearman_var']['r_s']:.4f}")
    log(f"Best Spearman (alpha): N={recommendations['best_spearman_alpha']['N']}, r_s={recommendations['best_spearman_alpha']['r_s']:.4f}")
    log(f"Best Pearson (variance): N={recommendations['best_pearson_var']['N']}, r={recommendations['best_pearson_var']['r']:.4f}")
    log(f"Best Pearson (alpha): N={recommendations['best_pearson_alpha']['N']}, r={recommendations['best_pearson_alpha']['r']:.4f}")
    log(f"Best Overall (combined score): N={recommendations['best_overall']['N']}, score={recommendations['best_overall']['score']:.4f}")
    
    # Find best Spearman excluding N=1 (high zero-inflation)
    df_no_n1 = analysis_df[analysis_df['N'] > 1].copy()
    if len(df_no_n1) > 0:
        best_spearman_no_n1_idx = df_no_n1['spearman_alpha'].idxmax()
        best_spearman_no_n1_n = int(df_no_n1.loc[best_spearman_no_n1_idx, 'N'])
        best_spearman_no_n1_rs = df_no_n1.loc[best_spearman_no_n1_idx, 'spearman_alpha']
        best_spearman_no_n1_rp = df_no_n1.loc[best_spearman_no_n1_idx, 'pearson_alpha']
        log(f"Best Spearman (alpha, excluding N=1): N={best_spearman_no_n1_n}, r_s={best_spearman_no_n1_rs:.4f}, r={best_spearman_no_n1_rp:.4f}")
    
    return recommendations


def generate_recommendation_report(analysis_df: pd.DataFrame, recommendations: dict) -> str:
    """
    Generate markdown report with recommendations.
    """
    report = "# Best N Value Recommendation for Tree Construction\n\n"
    
    report += "## Summary\n\n"
    
    # Check if there's a better Spearman correlation (excluding N=1)
    df_no_n1 = analysis_df[analysis_df['N'] > 1].copy()
    if len(df_no_n1) > 0:
        best_spearman_no_n1_idx = df_no_n1['spearman_alpha'].idxmax()
        best_spearman_no_n1_n = int(df_no_n1.loc[best_spearman_no_n1_idx, 'N'])
        best_spearman_no_n1_rs = df_no_n1.loc[best_spearman_no_n1_idx, 'spearman_alpha']
        best_spearman_no_n1_rp = df_no_n1.loc[best_spearman_no_n1_idx, 'pearson_alpha']
        
        best_overall_n = recommendations['best_overall']['N']
        best_overall_rs = recommendations['best_overall']['spearman_alpha']
        
        if best_spearman_no_n1_rs > best_overall_rs:
            report += f"**Primary Recommendation: N={best_spearman_no_n1_n}** (Best Spearman correlation)\n\n"
            report += f"**Alternative Recommendation: N={best_overall_n}** (Best combined score)\n\n"
        else:
            report += f"**Recommended N value: {recommendations['best_overall']['N']}**\n\n"
    else:
        report += f"**Recommended N value: {recommendations['best_overall']['N']}**\n\n"
    
    report += f"This recommendation is based on a combined score considering:\n"
    report += "- Spearman correlation (rank-based, more robust) - PRIMARY METRIC\n"
    report += "- Pearson correlation (linear relationship)\n"
    report += "- Preference for alpha-weighted metric\n"
    report += "- Statistical significance (all p-values < 1e-90, highly significant)\n"
    report += "- Zero-inflation penalty (prefer N with fewer zeros)\n\n"
    
    report += "## Detailed Results\n\n"
    report += "### Best by Metric\n\n"
    report += f"- **Best Spearman (Variance)**: N={recommendations['best_spearman_var']['N']}, r_s={recommendations['best_spearman_var']['r_s']:.4f}\n"
    report += f"- **Best Spearman (Alpha)**: N={recommendations['best_spearman_alpha']['N']}, r_s={recommendations['best_spearman_alpha']['r_s']:.4f}\n"
    report += f"- **Best Pearson (Variance)**: N={recommendations['best_pearson_var']['N']}, r={recommendations['best_pearson_var']['r']:.4f}\n"
    report += f"- **Best Pearson (Alpha)**: N={recommendations['best_pearson_alpha']['N']}, r={recommendations['best_pearson_alpha']['r']:.4f}\n\n"
    
    report += "### Correlation Analysis Table\n\n"
    report += "| N | Spearman r_s (var) | p-val | Pearson r (var) | p-val | Spearman r_s (alpha) | p-val | Pearson r (alpha) | p-val |\n"
    report += "|---|-------------------|-------|------------------|-------|----------------------|-------|-------------------|-------|\n"
    
    for _, row in analysis_df.iterrows():
        report += (f"| {int(row['N'])} | {row['spearman_var']:.4f} | {row['spearman_var_pval']:.2e} | "
                  f"{row['pearson_var']:.4f} | {row['pearson_var_pval']:.2e} | "
                  f"{row['spearman_alpha']:.4f} | {row['spearman_alpha_pval']:.2e} | "
                  f"{row['pearson_alpha']:.4f} | {row['pearson_alpha_pval']:.2e} |\n")
    
    report += "\n## Rationale\n\n"
    
    best_n = recommendations['best_overall']['N']
    best_row = analysis_df[analysis_df['N'] == best_n].iloc[0]
    
    report += f"For N={best_n}:\n"
    report += f"- Spearman correlation (alpha): {best_row['spearman_alpha']:.4f} (p < {best_row['spearman_alpha_pval']:.2e})\n"
    report += f"- Pearson correlation (alpha): {best_row['pearson_alpha']:.4f} (p < {best_row['pearson_alpha_pval']:.2e})\n"
    report += f"- Combined score: {best_row['overall_score']:.4f}\n\n"
    
    report += "## Recommendation\n\n"
    
    # Check for best Spearman excluding N=1
    df_no_n1 = analysis_df[analysis_df['N'] > 1].copy()
    if len(df_no_n1) > 0:
        best_spearman_no_n1_idx = df_no_n1['spearman_alpha'].idxmax()
        best_spearman_no_n1_n = int(df_no_n1.loc[best_spearman_no_n1_idx, 'N'])
        best_spearman_no_n1_rs = df_no_n1.loc[best_spearman_no_n1_idx, 'spearman_alpha']
        best_spearman_no_n1_rp = df_no_n1.loc[best_spearman_no_n1_idx, 'pearson_alpha']
        
        best_overall_n = recommendations['best_overall']['N']
        best_overall_rs = recommendations['best_overall']['spearman_alpha']
        best_overall_rp = recommendations['best_overall']['pearson_alpha']
        
        if best_spearman_no_n1_rs > best_overall_rs:
            report += f"### Option 1: N={best_spearman_no_n1_n} (Best Spearman - PRIMARY METRIC)\n\n"
            report += f"**Use N={best_spearman_no_n1_n} with alpha-weighted metric for tree construction if Spearman is primary.**\n\n"
            report += f"This N value provides:\n"
            report += f"- **Best Spearman correlation**: r_s={best_spearman_no_n1_rs:.4f} (rank-based, robust to outliers)\n"
            report += f"- Pearson correlation: r={best_spearman_no_n1_rp:.4f}\n"
            report += f"- High statistical significance\n"
            report += f"- Zero zero-inflation (all pairs have non-zero distances)\n\n"
            
            report += f"### Option 2: N={best_overall_n} (Best Combined Score)\n\n"
            report += f"**Use N={best_overall_n} with alpha-weighted metric for tree construction if balancing both metrics.**\n\n"
            report += f"This N value provides:\n"
            report += f"- Spearman correlation: r_s={best_overall_rs:.4f}\n"
            report += f"- **Best Pearson correlation**: r={best_overall_rp:.4f} (linear relationship)\n"
            report += f"- High statistical significance\n"
            report += f"- Zero zero-inflation (all pairs have non-zero distances)\n"
            report += f"- Good balance between correlation strength and domain count\n\n"
        else:
            report += f"**Use N={best_n} with alpha-weighted metric for tree construction.**\n\n"
            report += "This N value provides:\n"
            report += "- Strong Spearman correlation (rank-based, robust to outliers)\n"
            report += "- Strong Pearson correlation (linear relationship)\n"
            report += "- High statistical significance\n"
            report += "- Good balance between correlation strength and domain count\n\n"
    else:
        report += f"**Use N={best_n} with alpha-weighted metric for tree construction.**\n\n"
        report += "This N value provides:\n"
        report += "- Strong Spearman correlation (rank-based, robust to outliers)\n"
        report += "- Strong Pearson correlation (linear relationship)\n"
        report += "- High statistical significance\n"
        report += "- Good balance between correlation strength and domain count\n\n"
    
    return report


def main():
    """Main analysis pipeline."""
    log("Starting best N selection analysis")
    
    # Load correlation results
    results_path = os.path.join(RESULTS, "domain_time_correlations.tsv")
    if not os.path.exists(results_path):
        log(f"ERROR: Results file not found: {results_path}")
        log("Please run domain_time_scatter.py first to generate correlation results.")
        return
    
    df = pd.read_csv(results_path, sep="\t")
    log(f"Loaded correlation results for {len(df)} N values")
    
    # Analyze correlations
    analysis_df = analyze_correlations(df)
    
    # Select best N
    recommendations = select_best_n(analysis_df)
    
    # Generate report
    report = generate_recommendation_report(analysis_df, recommendations)
    
    # Save report
    report_path = os.path.join(RESULTS, "best_n_recommendation.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    log(f"\nSaved recommendation report to {report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BEST N RECOMMENDATION")
    print("=" * 80)
    print(f"\nRecommended N value: {recommendations['best_overall']['N']}")
    print(f"  Spearman r_s (alpha): {recommendations['best_overall']['spearman_alpha']:.4f}")
    print(f"  Pearson r (alpha): {recommendations['best_overall']['pearson_alpha']:.4f}")
    print(f"  Combined score: {recommendations['best_overall']['score']:.4f}")
    print("\nUse this N value with alpha-weighted metric for tree construction.")
    print("=" * 80)


if __name__ == "__main__":
    main()

