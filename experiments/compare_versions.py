# experiments/version_comparison.py
"""
Compare different "versions" of the search engine by simulating
different configurations using weight parameters.
Generates precision-recall curves and performance comparison graphs.
"""
import json
import requests
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from experiments.evaluate import load_queries_train, mean_ap_at_k
from experiments.run_evaluation import (
    harmonic_mean_precision_f1,
    precision_at_k,
    recall_at_k,
    f1_at_k,
)

BASE_URL = "http://104.198.58.119:8080"

# Define versions - each simulates a stage in development
VERSIONS = [
    {
        'name': 'v1: TF-IDF baseline',
        'short_name': 'v1',
        'description': 'TF-IDF baseline',
        'endpoint': '/search_body',  # TF-IDF cosine
        'weights': None,
    },
    {
        'name': 'v2: BM25 only',
        'short_name': 'v2',
        'description': 'Added BM25',
        'endpoint': '/search_with_weights',
        'weights': {
            'body_weight': 1.0,
            'title_weight': 0.0,
            'anchor_weight': 0.0,
            'lsi_weight': 0.0,
            'pagerank_boost': 0.0,
            'pageview_boost': 0.0,
        },
    },
    {
        'name': 'v3: + Title + Anchor',
        'short_name': 'v3',
        'description': 'Added Title + Anchor',
        'endpoint': '/search_with_weights',
        'weights': {
            'body_weight': 1.0,
            'title_weight': 0.5,
            'anchor_weight': 0.3,
            'lsi_weight': 0.0,
            'pagerank_boost': 0.0,
            'pageview_boost': 0.0,
        },
    },
    {
        'name': 'v4: + PageRank + PageViews',
        'short_name': 'v4',
        'description': 'Added PageRank + PageViews',
        'endpoint': '/search_with_weights',
        'weights': {
            'body_weight': 1.0,
            'title_weight': 0.5,
            'anchor_weight': 0.3,
            'lsi_weight': 0.0,
            'pagerank_boost': 0.15,
            'pageview_boost': 0.10,
        },
    },
    {
        'name': 'v5: BM25 tuned (k1=2.5, b=0)',
        'short_name': 'v5',
        'description': 'BM25 parameter tuning',
        'endpoint': '/search_with_weights',
        'weights': {
            'body_weight': 1.0,
            'title_weight': 0.5,
            'anchor_weight': 0.3,
            'lsi_weight': 0.0,
            'pagerank_boost': 0.15,
            'pageview_boost': 0.10,
        },
    },
    {
        'name': 'v6: + LSI rerank',
        'short_name': 'v6',
        'description': 'Adding LSI rerank',
        'endpoint': '/search_with_weights',
        'weights': {
            'body_weight': 1.0,
            'title_weight': 0.5,
            'anchor_weight': 0.3,
            'lsi_weight': 0.25,
            'pagerank_boost': 0.15,
            'pageview_boost': 0.10,
        },
    },
    {
        'name': 'v7: Weight tuning (final)',
        'short_name': 'v7',
        'description': 'Weight tuning (final)',
        'endpoint': '/search_with_weights',
        'weights': {
            # TODO: Fill with your best weights from tuning
            'body_weight': 0.0,
            'title_weight': 2.0,
            'anchor_weight': 0.75,
            'lsi_weight': 0.0,
            'pagerank_boost': 0.15,
            'pageview_boost': 0.10,
        },
    },
]


def query_version(query: str, version: dict) -> Tuple[List[int], float]:
    """Query a specific version and return results + time."""
    endpoint = version['endpoint']
    weights = version.get('weights')
    
    url = f"{BASE_URL}{endpoint}"
    params = {'query': query}
    
    if weights:
        params.update(weights)
    
    start_time = time.time()
    try:
        response = requests.get(url, params=params, timeout=120)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            return [], elapsed
        
        results = response.json()
        doc_ids = [int(doc_id) for doc_id, _ in results]
        return doc_ids, elapsed
    except Exception as e:
        return [], time.time() - start_time


def calculate_precision_recall_curve(pred: List[int], gold: List[int], k_values: List[int]) -> Tuple[List[float], List[float]]:
    """Calculate precision and recall at different k values."""
    precisions = []
    recalls = []
    gold_set = set(gold)
    
    for k in k_values:
        pred_at_k = pred[:k]
        relevant_at_k = len(set(pred_at_k) & gold_set)
        
        precision = relevant_at_k / k if k > 0 else 0
        recall = relevant_at_k / len(gold_set) if gold_set else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls


def evaluate_version(version: dict, queries: List[str], gold: Dict[str, List[int]]) -> Dict:
    """Evaluate a single version with all metrics."""
    print(f"\nTesting {version['name']}...")
    
    all_pred = {}
    times = []
    k_values = [1, 3, 5, 10, 15, 20, 30, 50, 100]
    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    
    for i, query in enumerate(queries, 1):
        doc_ids, elapsed = query_version(query, version)
        all_pred[query] = doc_ids
        times.append(elapsed)
        
        # Calculate P-R curve points
        gold_list = gold.get(query, [])
        precisions, recalls = calculate_precision_recall_curve(doc_ids, gold_list, k_values)
        for j, k in enumerate(k_values):
            all_precisions[k].append(precisions[j])
            all_recalls[k].append(recalls[j])
        
        print(f"  {i}/{len(queries)}", end='\r')
    
    print()
    
    # Calculate aggregate metrics
    map_at_10 = mean_ap_at_k(all_pred, gold, k=10)
    map_at_5 = mean_ap_at_k(all_pred, gold, k=5)
    
    harmonic_means = []
    precisions_5 = []
    recalls_30 = []
    f1_scores_30 = []
    
    for q in queries:
        pred = all_pred.get(q, [])
        gold_list = gold.get(q, [])
        
        hm = harmonic_mean_precision_f1(pred, gold_list, p_k=5, f1_k=30)
        harmonic_means.append(hm)
        precisions_5.append(precision_at_k(pred, gold_list, k=5))
        recalls_30.append(recall_at_k(pred, gold_list, k=30))
        f1_scores_30.append(f1_at_k(pred, gold_list, k=30))
    
    # Average P-R at each k
    avg_precisions = {k: np.mean(all_precisions[k]) for k in k_values}
    avg_recalls = {k: np.mean(all_recalls[k]) for k in k_values}
    
    result = {
        'name': version['name'],
        'short_name': version['short_name'],
        'description': version['description'],
        'map_at_10': map_at_10,
        'map_at_5': map_at_5,
        'harmonic_mean': np.mean(harmonic_means),
        'precision_at_5': np.mean(precisions_5),
        'recall_at_30': np.mean(recalls_30),
        'f1_at_30': np.mean(f1_scores_30),
        'avg_time': np.mean(times),
        'precision_at_k': avg_precisions,
        'recall_at_k': avg_recalls,
        'k_values': k_values,
    }
    
    print(f"  MAP@10={result['map_at_10']:.4f}, HM={result['harmonic_mean']:.4f}, Time={result['avg_time']:.2f}s")
    
    return result


def create_visualizations(results: List[Dict], output_dir: Path):
    """Create all visualization graphs."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    versions = [r['short_name'] for r in results]
    descriptions = [r['description'] for r in results]
    map10_scores = [r['map_at_10'] for r in results]
    hm_scores = [r['harmonic_mean'] for r in results]
    times = [r['avg_time'] for r in results]
    
    # =========================================================================
    # Graph 1: MAP@10 Across Versions
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(versions))
    bars = ax.bar(x, map10_scores, color=COLORS[:len(versions)], edgecolor='white', linewidth=2, alpha=0.85)
    
    # Trend line
    z = np.polyfit(x, map10_scores, 2)
    p = np.poly1d(z)
    ax.plot(x, p(x), '--', color='#2c3e50', linewidth=2, alpha=0.7, label='Trend')
    
    ax.set_xlabel('Version', fontsize=13, fontweight='bold')
    ax.set_ylabel('MAP@10', fontsize=13, fontweight='bold')
    ax.set_title('Search Engine Performance: MAP@10 Across Versions', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=11)
    ax.set_ylim(0, max(map10_scores) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Value labels
    for i, (bar, val, desc) in enumerate(zip(bars, map10_scores, descriptions)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, 0.01,
               desc, ha='center', va='bottom', fontsize=7, rotation=90, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_map10.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'version_map10.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 2: Retrieval Time Across Versions
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars = ax.bar(x, times, color=COLORS[:len(versions)], edgecolor='white', linewidth=2, alpha=0.85)
    
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2.5, label='Target: 1s (7 pts)')
    ax.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, label='Threshold: 2s (5 pts)')
    
    ax.set_xlabel('Version', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Query Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Search Engine Efficiency: Query Time Across Versions', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}\n{d}' for v, d in zip(versions, descriptions)], fontsize=9)
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_time.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'version_time.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 3: MAP@10 vs Harmonic Mean Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    
    width = 0.35
    bars1 = ax.bar(x - width/2, map10_scores, width, label='MAP@10', color='#3498db', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, hm_scores, width, label='Harmonic Mean (P@5, F1@30)', color='#e74c3c', alpha=0.85, edgecolor='white')
    
    ax.set_xlabel('Version', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('MAP@10 vs Harmonic Mean Across Versions', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}\n{d}' for v, d in zip(versions, descriptions)], fontsize=9)
    ax.set_ylim(0, max(max(map10_scores), max(hm_scores)) * 1.25)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, map10_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#3498db')
    for bar, val in zip(bars2, hm_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_map_vs_hm.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'version_map_vs_hm.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 4: All Metrics Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    
    metrics = ['MAP@10', 'MAP@5', 'P@5', 'R@30', 'F1@30', 'HM']
    x_metrics = np.arange(len(metrics))
    width = 0.11
    
    for i, r in enumerate(results):
        values = [
            r['map_at_10'],
            r['map_at_5'],
            r['precision_at_5'],
            r['recall_at_30'],
            r['f1_at_30'],
            r['harmonic_mean'],
        ]
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar(x_metrics + offset, values, width, label=r['short_name'], 
               color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor='white')
    
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('All Metrics Comparison Across Versions', fontsize=16, fontweight='bold')
    ax.set_xticks(x_metrics)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_all_metrics.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'version_all_metrics.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 5: Precision-Recall Curves (All versions together)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, r in enumerate(results):
        recalls = [r['recall_at_k'][k] for k in r['k_values']]
        precisions = [r['precision_at_k'][k] for k in r['k_values']]
        
        ax.plot(recalls, precisions, marker='o', markersize=8, linewidth=2.5,
               color=COLORS[i % len(COLORS)], label=f"{r['short_name']}: {r['description']}", alpha=0.85)
    
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curves Across Versions', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_pr_curves.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'version_pr_curves.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 6: Individual P-R Curves for each version
    # =========================================================================
    n_versions = len(results)
    cols = 4
    rows = (n_versions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()
    
    for i, r in enumerate(results):
        ax = axes[i]
        recalls = [r['recall_at_k'][k] for k in r['k_values']]
        precisions = [r['precision_at_k'][k] for k in r['k_values']]
        
        ax.plot(recalls, precisions, marker='o', markersize=8, linewidth=2.5,
               color=COLORS[i % len(COLORS)], alpha=0.85)
        ax.fill_between(recalls, precisions, alpha=0.2, color=COLORS[i % len(COLORS)])
        
        # Annotate k values
        for k, rec, prec in zip(r['k_values'], recalls, precisions):
            if k in [1, 5, 10, 30, 100]:
                ax.annotate(f'k={k}', (rec, prec), textcoords="offset points", 
                           xytext=(5, 5), fontsize=7, alpha=0.7)
        
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f"{r['short_name']}: {r['description']}\nMAP@10={r['map_at_10']:.4f}", fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_pr_individual.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'version_pr_individual.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 7: Summary Dashboard
    # =========================================================================
    fig = plt.figure(figsize=(20, 12))
    
    # Subplot 1: MAP@10 bars
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.bar(range(len(versions)), map10_scores, color=COLORS[:len(versions)], alpha=0.85)
    ax1.set_title('MAP@10', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(versions)))
    ax1.set_xticklabels(versions)
    ax1.set_ylim(0, max(map10_scores) * 1.2)
    for i, v in enumerate(map10_scores):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Subplot 2: Harmonic Mean bars
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(range(len(versions)), hm_scores, color=COLORS[:len(versions)], alpha=0.85)
    ax2.set_title('Harmonic Mean (P@5, F1@30)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(versions)))
    ax2.set_xticklabels(versions)
    ax2.set_ylim(0, max(hm_scores) * 1.2)
    for i, v in enumerate(hm_scores):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Subplot 3: Time bars
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(range(len(versions)), times, color=COLORS[:len(versions)], alpha=0.85)
    ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Target: 1s')
    ax3.set_title('Avg Query Time (s)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(versions)))
    ax3.set_xticklabels(versions)
    ax3.legend(loc='upper right', fontsize=8)
    for i, v in enumerate(times):
        ax3.text(i, v + 0.05, f'{v:.2f}s', ha='center', fontsize=9, fontweight='bold')
    
    # Subplot 4: P-R curves combined
    ax4 = fig.add_subplot(2, 3, 4)
    for i, r in enumerate(results):
        recalls = [r['recall_at_k'][k] for k in r['k_values']]
        precisions = [r['precision_at_k'][k] for k in r['k_values']]
        ax4.plot(recalls, precisions, marker='o', markersize=4, linewidth=2,
                color=COLORS[i % len(COLORS)], label=r['short_name'], alpha=0.85)
    ax4.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Improvement chart
    ax5 = fig.add_subplot(2, 3, 5)
    improvements = [0]
    for i in range(1, len(map10_scores)):
        imp = ((map10_scores[i] - map10_scores[i-1]) / map10_scores[i-1]) * 100
        improvements.append(imp)
    
    colors_imp = ['green' if v >= 0 else 'red' for v in improvements]
    ax5.bar(range(len(versions)), improvements, color=colors_imp, alpha=0.85)
    ax5.axhline(y=0, color='black', linewidth=1)
    ax5.set_title('% Improvement from Previous Version', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(versions)))
    ax5.set_xticklabels(versions)
    for i, v in enumerate(improvements):
        ax5.text(i, v + (1 if v >= 0 else -3), f'{v:+.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # Subplot 6: Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    best_map_idx = np.argmax(map10_scores)
    best_hm_idx = np.argmax(hm_scores)
    fastest_idx = np.argmin(times)
    
    summary_text = f"""
    ================================================
              VERSION COMPARISON SUMMARY
    ================================================
    
    Best MAP@10:     {versions[best_map_idx]} ({map10_scores[best_map_idx]:.4f})
    Best HM:         {versions[best_hm_idx]} ({hm_scores[best_hm_idx]:.4f})
    Fastest:         {versions[fastest_idx]} ({times[fastest_idx]:.2f}s)
    
    ------------------------------------------------
    Total Improvement (v1 to v{len(versions)}):
      MAP@10: {((map10_scores[-1]/map10_scores[0])-1)*100:+.1f}%
      HM:     {((hm_scores[-1]/hm_scores[0])-1)*100:+.1f}%
    ================================================
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    fig.suptitle('Search Engine Version Comparison Dashboard', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_dashboard.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'version_dashboard.png'}")
    plt.close()


def main():
    # Load queries - use config for path resolution
    import config
    queries_path = config.QUERIES_DIR / "test_queries.json"
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"Loaded {len(queries)} queries")
    
    # Test server
    print(f"\nBase URL: {BASE_URL}")
    try:
        requests.get(BASE_URL, timeout=10)
        print("Server reachable")
    except:
        print("Cannot connect to server")
        return
    
    # Evaluate each version
    results = []
    for version in VERSIONS:
        result = evaluate_version(version, queries, gold)
        results.append(result)
    
    # Create output directory
    output_dir = Path("experiments/version_comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(output_dir / 'version_results.json', 'w') as f:
        json.dump({'results': convert(results)}, f, indent=2)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print summary table for report
    print("\n" + "="*90)
    print("RESULTS TABLE FOR REPORT:")
    print("="*90)
    print(f"{'Version':<10} {'Description':<30} {'MAP@10':<10} {'HM':<10} {'Avg Time':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['short_name']:<10} {r['description']:<30} {r['map_at_10']:.4f}     {r['harmonic_mean']:.4f}     {r['avg_time']:.2f}s")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()