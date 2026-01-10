# scripts/tune_glove_hyperparameters.py
"""
Hyperparameter tuning script for GloVe reranking feature.

Performs grid search over glove_beta and glove_top_k parameters,
evaluates performance using Average Precision@10, Average Precision@5, Precision@5, and F1@30 metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import config
from experiments.evaluate import load_queries_train, average_precision_at_k
from experiments.run_evaluation import (
    harmonic_mean_precision_f1,
    precision_at_k,
    f1_at_k,
)


def query_search_engine(
    query: str,
    server_url: str,
    enable_glove: bool = True,
    glove_beta: float = 0.2,
    glove_top_k: int = 10,
    timeout: int = 120,
) -> Tuple[List[int], float]:
    """
    Query the search engine with specific GloVe parameters.
    
    Args:
        query: Search query
        server_url: Base URL of the search engine server
        enable_glove: Enable GloVe reranking
        glove_beta: GloVe beta weight
        glove_top_k: Number of top documents for query embedding
        timeout: Request timeout in seconds
        
    Returns:
        (doc_ids, elapsed_time) tuple
    """
    url = f"{server_url}/search"
    params = {
        'query': query,
        'enable_glove': str(enable_glove).lower(),
        'glove_beta': str(glove_beta),
        'glove_top_k': str(glove_top_k),
    }
    
    start_time = time.time()
    try:
        response = requests.get(url, params=params, timeout=timeout)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"  ⚠ Warning: Server returned status {response.status_code} for query '{query}'")
            return [], elapsed
        
        results = response.json()
        doc_ids = [int(doc_id) for doc_id, _ in results]
        return doc_ids, elapsed
    except requests.exceptions.Timeout:
        print(f"  ⚠ Warning: Timeout for query '{query}'")
        return [], time.time() - start_time
    except Exception as e:
        print(f"  ⚠ Error querying '{query}': {e}")
        return [], time.time() - start_time


def evaluate_parameters(
    queries: List[str],
    gold: Dict[str, List[int]],
    server_url: str,
    glove_beta: float,
    glove_top_k: int,
) -> Dict:
    """
    Evaluate a specific parameter combination.
    
    Args:
        queries: List of queries
        gold: Gold standard relevance judgments
        server_url: Base URL of the search engine server
        glove_beta: GloVe beta weight
        glove_top_k: Number of top documents for query embedding
        
    Returns:
        Dictionary with evaluation metrics
    """
    all_pred: Dict[str, List[int]] = {}
    times: List[float] = []
    
    # Query all queries with progress bar
    for query in tqdm(queries, desc=f"β={glove_beta}, k={glove_top_k}", leave=False):
        doc_ids, elapsed = query_search_engine(
            query=query,
            server_url=server_url,
            enable_glove=True,
            glove_beta=glove_beta,
            glove_top_k=glove_top_k,
        )
        all_pred[query] = doc_ids
        times.append(elapsed)
    
    # Calculate metrics
    avg_precision_at_10 = average_precision_at_k(all_pred, gold, k=10)
    avg_precision_at_5 = average_precision_at_k(all_pred, gold, k=5)
    
    precisions_5 = []
    f1_scores_30 = []
    harmonic_means = []
    
    for q in queries:
        pred = all_pred.get(q, [])
        gold_list = gold.get(q, [])
        
        p5 = precision_at_k(pred, gold_list, k=5)
        f1_30 = f1_at_k(pred, gold_list, k=30)
        hm = harmonic_mean_precision_f1(pred, gold_list, p_k=5, f1_k=30)
        
        precisions_5.append(p5)
        f1_scores_30.append(f1_30)
        harmonic_means.append(hm)
    
    avg_time = sum(times) / len(times) if times else 0.0
    
    return {
        'glove_beta': glove_beta,
        'glove_top_k': glove_top_k,
        'avg_precision_at_10': avg_precision_at_10,
        'avg_precision_at_5': avg_precision_at_5,
        'precision_at_5': sum(precisions_5) / len(precisions_5) if precisions_5 else 0.0,
        'f1_at_30': sum(f1_scores_30) / len(f1_scores_30) if f1_scores_30 else 0.0,
        'harmonic_mean': sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0.0,
        'avg_time': avg_time,
    }


def create_visualizations(
    all_results: List[Dict],
    beta_values: List[float],
    top_k_values: List[int],
    output_dir: Path,
) -> None:
    """
    Create beautiful visualizations of the hyperparameter tuning results.
    
    Args:
        all_results: List of result dictionaries
        beta_values: List of beta values tested
        top_k_values: List of top_k values tested
        output_dir: Directory to save the plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set style for beautiful plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create colormap
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_k_values)))
    
    # =========================================================================
    # GRAPH 1: Heatmap of Average Precision@10 for each parameter combination
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create matrix for heatmap
    heatmap_data = np.zeros((len(top_k_values), len(beta_values)))
    for r in all_results:
        beta_idx = beta_values.index(r['glove_beta'])
        topk_idx = top_k_values.index(r['glove_top_k'])
        heatmap_data[topk_idx, beta_idx] = r['avg_precision_at_10']
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=heatmap_data.min() - 0.01, vmax=heatmap_data.max() + 0.01)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Average Precision@10', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Set ticks
    ax.set_xticks(np.arange(len(beta_values)))
    ax.set_yticks(np.arange(len(top_k_values)))
    ax.set_xticklabels([f'{b:.1f}' for b in beta_values])
    ax.set_yticklabels([str(k) for k in top_k_values])
    
    # Add value annotations
    for i in range(len(top_k_values)):
        for j in range(len(beta_values)):
            value = heatmap_data[i, j]
            text_color = 'white' if value < (heatmap_data.max() + heatmap_data.min()) / 2 else 'black'
            ax.text(j, i, f'{value:.4f}', ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
    
    # Find best combination and highlight it
    best_result = max(all_results, key=lambda x: x['avg_precision_at_10'])
    best_beta_idx = beta_values.index(best_result['glove_beta'])
    best_topk_idx = top_k_values.index(best_result['glove_top_k'])
    ax.add_patch(plt.Rectangle((best_beta_idx - 0.5, best_topk_idx - 0.5), 1, 1, fill=False, edgecolor='gold', linewidth=3))
    
    ax.set_xlabel('GloVe Beta (β)', fontweight='bold')
    ax.set_ylabel('Top-K Documents', fontweight='bold')
    ax.set_title('Average Precision@10 Heatmap - GloVe Hyperparameter Tuning\n(Gold border = Best combination)', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_map10.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: heatmap_map10.png")
    
    # =========================================================================
    # GRAPH 2: Line plot - Average Precision@10 vs Beta for different Top-K values
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, top_k in enumerate(top_k_values):
        top_k_results = [r for r in all_results if r['glove_top_k'] == top_k]
        top_k_results.sort(key=lambda x: x['glove_beta'])
        
        betas = [r['glove_beta'] for r in top_k_results]
        ap10s = [r['avg_precision_at_10'] for r in top_k_results]
        
        ax.plot(betas, ap10s, marker='o', markersize=10, linewidth=2.5, 
                color=colors[idx], label=f'Top-K = {top_k}', alpha=0.9)
        
        # Add value labels
        for beta, ap10 in zip(betas, ap10s):
            ax.annotate(f'{ap10:.3f}', (beta, ap10), textcoords='offset points', 
                       xytext=(0, 8), ha='center', fontsize=8, alpha=0.7)
    
    # Highlight best point
    ax.scatter([best_result['glove_beta']], [best_result['avg_precision_at_10']], 
               s=300, c='gold', marker='*', edgecolors='black', linewidths=1.5, zorder=5,
               label=f"Best: β={best_result['glove_beta']}, k={best_result['glove_top_k']}")
    
    ax.set_xlabel('GloVe Beta (β)', fontweight='bold')
    ax.set_ylabel('Average Precision@10', fontweight='bold')
    ax.set_title('Average Precision@10 vs GloVe Beta\nfor Different Top-K Values', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Add subtle background gradient
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'line_plot_beta.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: line_plot_beta.png")
    
    # =========================================================================
    # GRAPH 3: Multi-metric comparison bar chart for top 10 combinations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort and get top 10
    sorted_results = sorted(all_results, key=lambda x: x['avg_precision_at_10'], reverse=True)[:10]
    
    # Prepare data
    labels = [f"β={r['glove_beta']}, k={r['glove_top_k']}" for r in sorted_results]
    x = np.arange(len(labels))
    width = 0.2
    
    metrics = {
        'Avg P@10': [r['avg_precision_at_10'] for r in sorted_results],
        'Avg P@5': [r['avg_precision_at_5'] for r in sorted_results],
        'Precision@5': [r['precision_at_5'] for r in sorted_results],
        'Harmonic Mean': [r['harmonic_mean'] for r in sorted_results],
    }
    
    # Color palette
    metric_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    # Create grouped bars
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=metric_colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7, rotation=45)
    
    # Highlight the best combination (first one)
    ax.axvspan(-0.5, 0.5, alpha=0.15, color='gold', label='Best Combination')
    
    ax.set_xlabel('Parameter Combinations', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Top 10 Parameter Combinations - Multi-Metric Comparison\n(Sorted by Average Precision@10)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # Set y-axis limits for better visualization
    all_values = [v for vals in metrics.values() for v in vals]
    ax.set_ylim([min(all_values) * 0.95, max(all_values) * 1.08])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bar_chart_top10.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: bar_chart_top10.png")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    """Main hyperparameter tuning function."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for GloVe reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default server IP from config
  python scripts/tune_glove_hyperparameters.py
  
  # Specify custom server IP
  python scripts/tune_glove_hyperparameters.py --server-ip 104.198.58.119
  
  # Use custom port
  python scripts/tune_glove_hyperparameters.py --server-ip 104.198.58.119 --port 8080
        """
    )
    parser.add_argument(
        '--server-ip',
        type=str,
        default=None,
        help='Server IP address (default: from config.INSTANCE_IP)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Server port (default: 8080)'
    )
    parser.add_argument(
        '--queries-path',
        type=str,
        default=None,
        help='Path to queries_train.json (default: queries_train.json in project root)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/glove_tuning_results',
        help='Output directory for results (default: experiments/glove_tuning_results)'
    )
    args = parser.parse_args()
    
    # Determine server URL
    if args.server_ip:
        server_url = f"http://{args.server_ip}:{args.port}"
    else:
        instance_ip = getattr(config, 'INSTANCE_IP', None)
        if instance_ip:
            server_url = f"http://{instance_ip}:{args.port}"
        else:
            print("Error: Server IP not specified and not found in config.py")
            print("Please use --server-ip or set INSTANCE_IP in config.py")
            return
    
    print("=" * 80)
    print("GloVe Hyperparameter Tuning")
    print("=" * 80)
    print(f"Server URL: {server_url}")
    
    # Test server connection
    print("\nTesting server connection...")
    try:
        test_url = f"{server_url}/search"
        response = requests.get(test_url, params={'query': 'test'}, timeout=10)
        if response.status_code == 200:
            print("✓ Server is reachable and responding")
        else:
            print(f"⚠ Server responded with status {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("  Make sure the server is running and the IP/port are correct")
        return
    
    # Load queries
    if args.queries_path:
        queries_path = Path(args.queries_path)
    else:
        queries_path = config.QUERIES_DIR / "queries_train.json"
    
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        return
    
    print(f"\nLoading queries from {queries_path}...")
    queries, gold = load_queries_train(str(queries_path))
    print(f"✓ Loaded {len(queries)} queries")
    
    # Parameter grid
    beta_values = [2.0, 2.2, 2.5, 2.7, 3.0]
    top_k_values = [8, 10, 12, 15, 17, 20]
    
    print(f"\nParameter grid:")
    print(f"  glove_beta: {beta_values}")
    print(f"  glove_top_k: {top_k_values}")
    print(f"  Total combinations: {len(beta_values) * len(top_k_values)}")
    
    # Run grid search
    print("\n" + "=" * 80)
    print("Running grid search...")
    print("=" * 80)
    
    all_results = []
    total_combinations = len(beta_values) * len(top_k_values)
    
    with tqdm(total=total_combinations, desc="Grid search progress") as pbar:
        for beta in beta_values:
            for top_k in top_k_values:
                result = evaluate_parameters(
                    queries=queries,
                    gold=gold,
                    server_url=server_url,
                    glove_beta=beta,
                    glove_top_k=top_k,
                )
                all_results.append(result)
                pbar.update(1)
    
    # Sort by Average Precision@10 descending
    all_results.sort(key=lambda x: x['avg_precision_at_10'], reverse=True)
    
    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS (sorted by Average Precision@10)")
    print("=" * 80)
    print(f"{'Beta':<8} {'Top-K':<8} {'Avg P@10':<12} {'Avg P@5':<12} {'P@5':<10} {'F1@30':<10} {'Time':<10}")
    print("-" * 80)
    
    for r in all_results:
        print(
            f"{r['glove_beta']:<8.1f} "
            f"{r['glove_top_k']:<8} "
            f"{r['avg_precision_at_10']:<12.4f} "
            f"{r['avg_precision_at_5']:<12.4f} "
            f"{r['precision_at_5']:<10.4f} "
            f"{r['f1_at_30']:<10.4f} "
            f"{r['avg_time']:<10.2f}"
        )
    
    # Best parameters
    best = all_results[0]
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)
    print(f"glove_beta: {best['glove_beta']}")
    print(f"glove_top_k: {best['glove_top_k']}")
    print(f"\nMetrics:")
    print(f"  Average Precision@10:        {best['avg_precision_at_10']:.4f}")
    print(f"  Average Precision@5:         {best['avg_precision_at_5']:.4f}")
    print(f"  Precision@5:   {best['precision_at_5']:.4f}")
    print(f"  F1@30:         {best['f1_at_30']:.4f}")
    print(f"  Harmonic Mean: {best['harmonic_mean']:.4f}")
    print(f"  Avg Time:      {best['avg_time']:.2f}s")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "glove_tuning_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'server_url': server_url,
            'num_queries': len(queries),
            'parameter_grid': {
                'beta_values': beta_values,
                'top_k_values': top_k_values,
            },
            'best_parameters': {
                'glove_beta': best['glove_beta'],
                'glove_top_k': best['glove_top_k'],
            },
            'best_metrics': {
                'avg_precision_at_10': best['avg_precision_at_10'],
                'avg_precision_at_5': best['avg_precision_at_5'],
                'precision_at_5': best['precision_at_5'],
                'f1_at_30': best['f1_at_30'],
                'harmonic_mean': best['harmonic_mean'],
                'avg_time': best['avg_time'],
            },
            'all_results': all_results,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    
    try:
        create_visualizations(all_results, beta_values, top_k_values, output_dir)
    except ImportError as e:
        print(f"⚠ Could not create visualizations: {e}")
        print("  Install matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"⚠ Error creating visualizations: {e}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()