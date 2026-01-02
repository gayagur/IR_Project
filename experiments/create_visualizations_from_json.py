# experiments/create_visualizations_from_json.py
"""
Script to create visualizations from existing weight_tuning_results.json file.
This script only generates plots - it doesn't run any evaluations.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def create_visualizations(results: List[Dict], output_dir: Path):
    """Create visualization plots of the results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available, skipping visualizations")
        print("Please install: pip install matplotlib numpy")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    map10_scores = [r['map_at_10'] for r in results]
    harmonic_means = [r['harmonic_mean_p5_f1_30'] for r in results]
    weights_list = [r['weights'] for r in results]
    
    # Sort by MAP@10 for better visualization
    sorted_results = sorted(results, key=lambda x: x['map_at_10'], reverse=True)
    
    print("Creating visualizations...")
    
    # 1. Scatter plot: MAP@10 vs Harmonic Mean
    print("  1. Creating scatter plot: MAP@10 vs Harmonic Mean...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(map10_scores, harmonic_means, alpha=0.6, s=50)
    ax.set_xlabel('MAP@10', fontsize=12)
    ax.set_ylabel('Harmonic Mean (P@5, F1@30)', fontsize=12)
    ax.set_title('Weight Tuning Results: MAP@10 vs Harmonic Mean', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight top 5
    top5 = sorted_results[:5]
    top5_map10 = [r['map_at_10'] for r in top5]
    top5_hm = [r['harmonic_mean_p5_f1_30'] for r in top5]
    ax.scatter(top5_map10, top5_hm, color='red', s=100, alpha=0.8, label='Top 5', zorder=5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'map10_vs_harmonic_mean.png', dpi=150)
    print(f"     ✓ Saved: {output_dir / 'map10_vs_harmonic_mean.png'}")
    plt.close()
    
    # 2. Bar chart: Top 20 configurations by MAP@10
    print("  2. Creating bar chart: Top 20 configurations...")
    fig, ax = plt.subplots(figsize=(14, 8))
    top20 = sorted_results[:20]
    x_pos = np.arange(len(top20))
    map10_values = [r['map_at_10'] for r in top20]
    
    bars = ax.bar(x_pos, map10_values, alpha=0.7)
    ax.set_xlabel('Configuration Rank', fontsize=12)
    ax.set_ylabel('MAP@10', fontsize=12)
    ax.set_title('Top 20 Configurations by MAP@10', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"#{i+1}" for i in range(len(top20))], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if i < 3:
            bar.set_color('gold')
        elif i < 10:
            bar.set_color('silver')
        else:
            bar.set_color('lightblue')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top20_map10.png', dpi=150)
    print(f"     ✓ Saved: {output_dir / 'top20_map10.png'}")
    plt.close()
    
    # 3. Heatmap of weights for top 10 configurations
    print("  3. Creating heatmap: Top 10 configurations weights...")
    fig, ax = plt.subplots(figsize=(12, 8))
    top10 = sorted_results[:10]
    
    weight_names = ['body', 'title', 'anchor', 'lsi', 'pagerank', 'pageview']
    weight_matrix = []
    for r in top10:
        w = r['weights']
        weight_matrix.append([w.get(name, 0.0) for name in weight_names])
    
    im = ax.imshow(weight_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(np.arange(len(weight_names)))
    ax.set_xticklabels(weight_names, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(top10)))
    ax.set_yticklabels([f"#{i+1} (MAP@{top10[i]['map_at_10']:.3f})" for i in range(len(top10))])
    ax.set_title('Weight Values for Top 10 Configurations', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(top10)):
        for j in range(len(weight_names)):
            text = ax.text(j, i, f"{weight_matrix[i][j]:.2f}",
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Weight Value')
    plt.tight_layout()
    plt.savefig(output_dir / 'top10_weights_heatmap.png', dpi=150)
    print(f"     ✓ Saved: {output_dir / 'top10_weights_heatmap.png'}")
    plt.close()
    
    # 4. Comparison: With/Without LSI, With/Without Boosts
    print("  4. Creating comparison: With/Without LSI and Boosts...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: With/Without LSI
    with_lsi = [r for r in results if r['weights'].get('lsi', 0) > 0]
    without_lsi = [r for r in results if r['weights'].get('lsi', 0) == 0]
    
    if with_lsi and without_lsi:
        ax = axes[0]
        ax.boxplot([
            [r['map_at_10'] for r in with_lsi],
            [r['map_at_10'] for r in without_lsi]
        ], labels=['With LSI', 'Without LSI'])
        ax.set_ylabel('MAP@10', fontsize=12)
        ax.set_title('Effect of LSI on Performance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Right: With/Without Boosts
    with_boosts = [r for r in results if r['weights'].get('pagerank', 0) > 0 or r['weights'].get('pageview', 0) > 0]
    without_boosts = [r for r in results if r['weights'].get('pagerank', 0) == 0 and r['weights'].get('pageview', 0) == 0]
    
    if with_boosts and without_boosts:
        ax = axes[1]
        ax.boxplot([
            [r['map_at_10'] for r in with_boosts],
            [r['map_at_10'] for r in without_boosts]
        ], labels=['With Boosts', 'Without Boosts'])
        ax.set_ylabel('MAP@10', fontsize=12)
        ax.set_title('Effect of PageRank/PageView Boosts', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_comparison.png', dpi=150)
    print(f"     ✓ Saved: {output_dir / 'feature_comparison.png'}")
    plt.close()
    
    # 5. Weight sensitivity analysis
    print("  5. Creating sensitivity analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    weight_params = ['title', 'anchor', 'lsi', 'pagerank', 'pageview']
    for idx, param in enumerate(weight_params):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        # Group by weight value for this parameter
        param_values = {}
        for r in results:
            val = r['weights'].get(param, 0.0)
            if val not in param_values:
                param_values[val] = []
            param_values[val].append(r['map_at_10'])
        
        if param_values:
            sorted_vals = sorted(param_values.keys())
            map10_by_weight = [np.mean(param_values[v]) for v in sorted_vals]
            std_by_weight = [np.std(param_values[v]) for v in sorted_vals]
            
            ax.errorbar(sorted_vals, map10_by_weight, yerr=std_by_weight, 
                       marker='o', capsize=5, capthick=2)
            ax.set_xlabel(f'{param} Weight', fontsize=11)
            ax.set_ylabel('Mean MAP@10', fontsize=11)
            ax.set_title(f'Sensitivity: {param} Weight', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    if len(weight_params) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_sensitivity.png', dpi=150)
    print(f"     ✓ Saved: {output_dir / 'weight_sensitivity.png'}")
    plt.close()
    
    print("\n✓ All visualizations created successfully!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations from existing weight_tuning_results.json")
    parser.add_argument("--json-file", default="experiments/weight_tuning_results/weight_tuning_results.json", 
                       help="Path to weight_tuning_results.json file")
    parser.add_argument("--output-dir", default="experiments/weight_tuning_results", 
                       help="Directory to save visualization images")
    args = parser.parse_args()
    
    # Load JSON file
    json_path = Path(args.json_file)
    if not json_path.is_absolute() and not json_path.exists():
        json_path = parent_dir / args.json_file
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print(f"Please provide the path to weight_tuning_results.json")
        return
    
    print(f"Loading results from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'results' not in data:
        print("Error: 'results' key not found in JSON file")
        return
    
    results = data['results']
    print(f"Loaded {len(results)} weight combinations")
    
    if not results:
        print("Error: No results found in JSON file")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations from JSON data...")
    print("=" * 80)
    create_visualizations(results, output_dir)
    
    # Print summary
    sorted_results = sorted(results, key=lambda x: x['map_at_10'], reverse=True)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTop 10 configurations by MAP@10:")
    print(f"{'Rank':<6} {'MAP@10':<10} {'MAP@5':<10} {'Harmonic Mean':<15} {'Weights'}")
    print("-" * 100)
    
    for i, result in enumerate(sorted_results[:10], 1):
        w = result['weights']
        print(f"{i:<6} {result['map_at_10']:>8.4f}   {result['map_at_5']:>8.4f}   "
              f"{result['harmonic_mean_p5_f1_30']:>13.4f}   "
              f"body={w['body']:.2f}, title={w['title']:.2f}, anchor={w['anchor']:.2f}, "
              f"lsi={w['lsi']:.2f}, pr={w['pagerank']:.2f}, pv={w['pageview']:.2f}")
    
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"\nBest configuration:")
    best = sorted_results[0]
    w = best['weights']
    print(f"  MAP@10: {best['map_at_10']:.4f}")
    print(f"  MAP@5: {best['map_at_5']:.4f}")
    print(f"  Harmonic Mean: {best['harmonic_mean_p5_f1_30']:.4f}")
    print(f"  Mean Time: {best['mean_time']:.3f}s")
    print(f"  Weights: {w}")


if __name__ == "__main__":
    main()

