# experiments/weight_tuning_full.py
"""
Comprehensive weight tuning with broad search including LSI.
Generates visualizations for the report.
"""
import json
import requests
import sys
import time
from pathlib import Path
from typing import Dict, List

script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from experiments.evaluate import load_queries_train, mean_ap_at_k
from experiments.run_evaluation import (
    harmonic_mean_precision_f1,
    precision_at_k,
    f1_at_k,
)

BASE_URL = "http://104.198.58.119:8080"


def generate_weight_combinations():
    """Generate comprehensive weight combinations including LSI."""
    combinations = []
    
    # =========================================================================
    # SECTION 1: Body-focused (traditional BM25 dominant)
    # =========================================================================
    for body_w in [0.5, 0.75, 1.0, 1.5]:
        for title_w in [0.25, 0.5, 0.75, 1.0]:
            for anchor_w in [0.25, 0.5, 0.75]:
                combinations.append({
                    'body_weight': body_w,
                    'title_weight': title_w,
                    'anchor_weight': anchor_w,
                    'lsi_weight': 0.0,
                    'pagerank_boost': 0.15,
                    'pageview_boost': 0.1,
                })
    
    # =========================================================================
    # SECTION 2: Title/Anchor dominant (body low)
    # =========================================================================
    for body_w in [0.0, 0.1, 0.2, 0.3]:
        for title_w in [0.75, 1.0, 1.5, 2.0, 2.5]:
            for anchor_w in [0.5, 0.75, 1.0, 1.5]:
                combinations.append({
                    'body_weight': body_w,
                    'title_weight': title_w,
                    'anchor_weight': anchor_w,
                    'lsi_weight': 0.0,
                    'pagerank_boost': 0.15,
                    'pageview_boost': 0.1,
                })
    
    # =========================================================================
    # SECTION 3: With LSI (various combinations)
    # =========================================================================
    for lsi_w in [0.1, 0.2, 0.3, 0.5]:
        # LSI + body focused
        for body_w in [0.3, 0.5, 0.75, 1.0]:
            combinations.append({
                'body_weight': body_w,
                'title_weight': 0.75,
                'anchor_weight': 0.5,
                'lsi_weight': lsi_w,
                'pagerank_boost': 0.15,
                'pageview_boost': 0.1,
            })
        
        # LSI + title focused
        for title_w in [1.0, 1.5, 2.0]:
            combinations.append({
                'body_weight': 0.1,
                'title_weight': title_w,
                'anchor_weight': 0.75,
                'lsi_weight': lsi_w,
                'pagerank_boost': 0.15,
                'pageview_boost': 0.1,
            })
    
    # =========================================================================
    # SECTION 4: PageRank/PageView variations
    # =========================================================================
    for pr in [0.0, 0.1, 0.2, 0.3, 0.5]:
        for pv in [0.0, 0.1, 0.2, 0.3]:
            # With body
            combinations.append({
                'body_weight': 0.5,
                'title_weight': 1.0,
                'anchor_weight': 0.75,
                'lsi_weight': 0.0,
                'pagerank_boost': pr,
                'pageview_boost': pv,
            })
            # Without body
            combinations.append({
                'body_weight': 0.1,
                'title_weight': 1.5,
                'anchor_weight': 0.75,
                'lsi_weight': 0.0,
                'pagerank_boost': pr,
                'pageview_boost': pv,
            })
    
    # =========================================================================
    # SECTION 5: Extreme/baseline configurations
    # =========================================================================
    # Body only
    combinations.append({
        'body_weight': 1.0, 'title_weight': 0.0, 'anchor_weight': 0.0,
        'lsi_weight': 0.0, 'pagerank_boost': 0.0, 'pageview_boost': 0.0,
    })
    # Title only
    combinations.append({
        'body_weight': 0.0, 'title_weight': 1.0, 'anchor_weight': 0.0,
        'lsi_weight': 0.0, 'pagerank_boost': 0.0, 'pageview_boost': 0.0,
    })
    # Anchor only
    combinations.append({
        'body_weight': 0.0, 'title_weight': 0.0, 'anchor_weight': 1.0,
        'lsi_weight': 0.0, 'pagerank_boost': 0.0, 'pageview_boost': 0.0,
    })
    # LSI only
    combinations.append({
        'body_weight': 0.0, 'title_weight': 0.0, 'anchor_weight': 0.0,
        'lsi_weight': 1.0, 'pagerank_boost': 0.0, 'pageview_boost': 0.0,
    })
    # Balanced
    combinations.append({
        'body_weight': 1.0, 'title_weight': 1.0, 'anchor_weight': 1.0,
        'lsi_weight': 0.0, 'pagerank_boost': 0.15, 'pageview_boost': 0.1,
    })
    # Very high title
    for title_w in [3.0, 4.0, 5.0]:
        combinations.append({
            'body_weight': 0.0, 'title_weight': title_w, 'anchor_weight': 0.5,
            'lsi_weight': 0.0, 'pagerank_boost': 0.15, 'pageview_boost': 0.1,
        })
    
    # =========================================================================
    # SECTION 6: High LSI combinations
    # =========================================================================
    for lsi_w in [0.5, 0.75, 1.0]:
        combinations.append({
            'body_weight': 0.0, 'title_weight': 1.0, 'anchor_weight': 0.5,
            'lsi_weight': lsi_w, 'pagerank_boost': 0.15, 'pageview_boost': 0.1,
        })
        combinations.append({
            'body_weight': 0.3, 'title_weight': 0.5, 'anchor_weight': 0.5,
            'lsi_weight': lsi_w, 'pagerank_boost': 0.15, 'pageview_boost': 0.1,
        })
    
    # Remove duplicates
    seen = set()
    unique = []
    for c in combinations:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    print(f"Generated {len(unique)} unique weight combinations")
    return unique


def test_weights(queries, gold):
    combinations = generate_weight_combinations()
    print(f"\nTesting {len(combinations)} weight combinations\n")
    
    results = []
    start_total = time.time()
    
    for idx, weights in enumerate(combinations, 1):
        all_pred = {}
        query_times = []
        
        elapsed_total = time.time() - start_total
        eta = (elapsed_total / idx) * (len(combinations) - idx) if idx > 1 else 0
        
        print(f"[{idx}/{len(combinations)}] ETA: {eta/60:.1f}min | "
              f"body={weights['body_weight']:.2f}, title={weights['title_weight']:.2f}, "
              f"anchor={weights['anchor_weight']:.2f}, lsi={weights['lsi_weight']:.2f}, "
              f"pr={weights['pagerank_boost']:.2f}, pv={weights['pageview_boost']:.2f}")
        
        for i, q in enumerate(queries, 1):
            params = {**weights, 'query': q}
            start = time.time()
            try:
                resp = requests.get(f"{BASE_URL}/search_with_weights", params=params, timeout=120)
                doc_ids = [int(d) for d, _ in resp.json()] if resp.ok else []
            except Exception as e:
                doc_ids = []
            query_times.append(time.time() - start)
            all_pred[q] = doc_ids
            print(f"  {i}/{len(queries)}", end='\r')
        
        # Calculate metrics
        map10 = mean_ap_at_k(all_pred, gold, k=10)
        map5 = mean_ap_at_k(all_pred, gold, k=5)
        
        harmonic_means = []
        precisions_5 = []
        f1_scores_30 = []
        
        for q in queries:
            pred = all_pred.get(q, [])
            gold_list = gold.get(q, [])
            hm = harmonic_mean_precision_f1(pred, gold_list, p_k=5, f1_k=30)
            harmonic_means.append(hm)
            precisions_5.append(precision_at_k(pred, gold_list, k=5))
            f1_scores_30.append(f1_at_k(pred, gold_list, k=30))
        
        avg_hm = sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0
        avg_p5 = sum(precisions_5) / len(precisions_5) if precisions_5 else 0
        avg_f1_30 = sum(f1_scores_30) / len(f1_scores_30) if f1_scores_30 else 0
        avg_time = sum(query_times) / len(query_times) if query_times else 0
        
        results.append({
            'weights': weights,
            'map_at_10': map10,
            'map_at_5': map5,
            'harmonic_mean': avg_hm,
            'precision_at_5': avg_p5,
            'f1_at_30': avg_f1_30,
            'avg_time': avg_time,
        })
        
        print(f"  MAP@10={map10:.4f}, HM={avg_hm:.4f}, Time={avg_time:.2f}s")
    
    # Sort by MAP@10
    results.sort(key=lambda x: x['map_at_10'], reverse=True)
    
    return results


def create_visualizations(results: List[Dict], output_dir: Path):
    """Create visualizations for the report."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    COLORS = {
        'gold': '#FFD700',
        'silver': '#C0C0C0',
        'bronze': '#CD7F32',
        'blue': '#3498db',
        'green': '#2ecc71',
        'red': '#e74c3c',
        'purple': '#9b59b6',
        'orange': '#f39c12',
        'teal': '#1abc9c',
    }
    
    # =========================================================================
    # Graph 1: Top 20 Configurations Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top20 = results[:20]
    x = np.arange(len(top20))
    map10_vals = [r['map_at_10'] for r in top20]
    
    colors = [COLORS['gold'] if i == 0 else COLORS['silver'] if i == 1 else COLORS['bronze'] if i == 2 else COLORS['blue'] for i in range(len(top20))]
    bars = ax.bar(x, map10_vals, color=colors, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Weight Configurations by MAP@10', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i+1}' for i in range(len(top20))], rotation=45, ha='right')
    ax.set_ylim(min(map10_vals) * 0.9, max(map10_vals) * 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, map10_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_tuning_top20.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'weight_tuning_top20.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 2: MAP@10 vs Harmonic Mean Scatter
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    map10_all = [r['map_at_10'] for r in results]
    hm_all = [r['harmonic_mean'] for r in results]
    
    scatter = ax.scatter(map10_all, hm_all, c=map10_all, cmap='viridis', s=80, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # Highlight top 5
    for i in range(min(5, len(results))):
        ax.scatter([results[i]['map_at_10']], [results[i]['harmonic_mean']], 
                  c='red', s=200, marker='*', zorder=10, edgecolors='white')
    
    ax.set_xlabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_ylabel('Harmonic Mean (P@5, F1@30)', fontsize=12, fontweight='bold')
    ax.set_title('MAP@10 vs Harmonic Mean (Top 5 marked with stars)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='MAP@10')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_tuning_map_vs_hm.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'weight_tuning_map_vs_hm.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 3: Weight Sensitivity Analysis (6 weights including LSI)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Weight Sensitivity Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    weight_params = ['body_weight', 'title_weight', 'anchor_weight', 'lsi_weight', 'pagerank_boost', 'pageview_boost']
    param_labels = ['Body', 'Title', 'Anchor', 'LSI', 'PageRank', 'PageView']
    colors_list = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red'], COLORS['purple'], COLORS['teal']]
    
    for idx, (param, label, color) in enumerate(zip(weight_params, param_labels, colors_list)):
        ax = axes[idx // 3, idx % 3]
        
        # Group by weight value
        param_values = {}
        for r in results:
            val = round(r['weights'].get(param, 0.0), 2)
            if val not in param_values:
                param_values[val] = []
            param_values[val].append(r['map_at_10'])
        
        if param_values:
            sorted_vals = sorted(param_values.keys())
            means = [np.mean(param_values[v]) for v in sorted_vals]
            stds = [np.std(param_values[v]) for v in sorted_vals]
            counts = [len(param_values[v]) for v in sorted_vals]
            
            ax.errorbar(sorted_vals, means, yerr=stds, marker='o', markersize=8,
                       capsize=4, capthick=1.5, color=color, linewidth=2,
                       markerfacecolor='white', markeredgewidth=2)
            
            ax.fill_between(sorted_vals, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)
            
            # Add count annotations
            for x_val, y_val, c in zip(sorted_vals, means, counts):
                ax.annotate(f'n={c}', (x_val, y_val), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=7, alpha=0.7)
        
        ax.set_xlabel(f'{label} Weight', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean MAP@10', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Sensitivity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_sensitivity.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'weight_sensitivity.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 4: All Metrics for Top 10
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top10 = results[:10]
    metrics = ['MAP@10', 'MAP@5', 'P@5', 'F1@30', 'HM']
    x = np.arange(len(top10))
    width = 0.15
    
    metric_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple'], COLORS['red']]
    
    for i, (metric_name, color) in enumerate(zip(metrics, metric_colors)):
        if metric_name == 'MAP@10':
            vals = [r['map_at_10'] for r in top10]
        elif metric_name == 'MAP@5':
            vals = [r['map_at_5'] for r in top10]
        elif metric_name == 'P@5':
            vals = [r['precision_at_5'] for r in top10]
        elif metric_name == 'F1@30':
            vals = [r['f1_at_30'] for r in top10]
        else:  # HM
            vals = [r['harmonic_mean'] for r in top10]
        
        ax.bar(x + i*width, vals, width, label=metric_name, color=color, alpha=0.85)
    
    ax.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('All Metrics for Top 10 Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*2)
    ax.set_xticklabels([f'#{i+1}' for i in range(len(top10))])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_tuning_all_metrics.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'weight_tuning_all_metrics.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 5: LSI Impact Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LSI Weight Impact Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # 5a. With vs Without LSI
    ax = axes[0]
    with_lsi = [r['map_at_10'] for r in results if r['weights'].get('lsi_weight', 0) > 0]
    without_lsi = [r['map_at_10'] for r in results if r['weights'].get('lsi_weight', 0) == 0]
    
    if with_lsi and without_lsi:
        bp = ax.boxplot([without_lsi, with_lsi], labels=['No LSI', 'With LSI'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['blue'])
        bp['boxes'][1].set_facecolor(COLORS['green'])
        for box in bp['boxes']:
            box.set_alpha(0.6)
    
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('LSI vs No LSI Performance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5b. LSI weight vs MAP@10
    ax = axes[1]
    lsi_results = [(r['weights'].get('lsi_weight', 0), r['map_at_10']) for r in results]
    lsi_weights = [x[0] for x in lsi_results]
    lsi_maps = [x[1] for x in lsi_results]
    
    ax.scatter(lsi_weights, lsi_maps, c=COLORS['green'], s=60, alpha=0.5, edgecolors='white')
    
    # Add trend line
    if len(set(lsi_weights)) > 1:
        z = np.polyfit(lsi_weights, lsi_maps, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lsi_weights), max(lsi_weights), 100)
        ax.plot(x_line, p(x_line), '--', color=COLORS['red'], linewidth=2, label='Trend')
        ax.legend()
    
    ax.set_xlabel('LSI Weight', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('LSI Weight vs Performance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lsi_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'lsi_analysis.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 6: Weight Distribution for Top 10
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top10 = results[:10]
    weight_names = ['Body', 'Title', 'Anchor', 'LSI', 'PR', 'PV']
    weight_keys = ['body_weight', 'title_weight', 'anchor_weight', 'lsi_weight', 'pagerank_boost', 'pageview_boost']
    x = np.arange(len(top10))
    width = 0.13
    
    weight_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red'], COLORS['purple'], COLORS['teal']]
    
    for i, (wname, wkey, wcolor) in enumerate(zip(weight_names, weight_keys, weight_colors)):
        vals = [r['weights'].get(wkey, 0) for r in top10]
        ax.bar(x + i*width, vals, width, label=wname, color=wcolor, alpha=0.85)
    
    ax.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Value', fontsize=12, fontweight='bold')
    ax.set_title('Weight Distribution for Top 10 Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*2.5)
    ax.set_xticklabels([f'#{i+1}' for i in range(len(top10))])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_distribution_top10.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'weight_distribution_top10.png'}")
    plt.close()
    
    # =========================================================================
    # Graph 7: Summary Card
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 12))
    
    best = results[0]
    w = best['weights']
    
    summary_text = f"""
    ============================================================
                    WEIGHT TUNING SUMMARY
    ============================================================
    
    Total Configurations Tested: {len(results)}
    
    ------------------------------------------------------------
    BEST CONFIGURATION (#1):
    ------------------------------------------------------------
    
    Weights:
      Body:     {w.get('body_weight', 0):.2f}
      Title:    {w.get('title_weight', 0):.2f}
      Anchor:   {w.get('anchor_weight', 0):.2f}
      LSI:      {w.get('lsi_weight', 0):.2f}
      PageRank: {w.get('pagerank_boost', 0):.2f}
      PageView: {w.get('pageview_boost', 0):.2f}
    
    Performance:
      MAP@10:        {best['map_at_10']:.4f}
      MAP@5:         {best['map_at_5']:.4f}
      Precision@5:   {best['precision_at_5']:.4f}
      F1@30:         {best['f1_at_30']:.4f}
      Harmonic Mean: {best['harmonic_mean']:.4f}
      Avg Time:      {best['avg_time']:.2f}s
    
    ------------------------------------------------------------
    PERFORMANCE RANGE:
    ------------------------------------------------------------
      Best MAP@10:  {max(r['map_at_10'] for r in results):.4f}
      Worst MAP@10: {min(r['map_at_10'] for r in results):.4f}
      Mean MAP@10:  {np.mean([r['map_at_10'] for r in results]):.4f}
      Std MAP@10:   {np.std([r['map_at_10'] for r in results]):.4f}
    
    ------------------------------------------------------------
    TOP 5 CONFIGURATIONS:
    ------------------------------------------------------------
"""
    
    for i, r in enumerate(results[:5], 1):
        rw = r['weights']
        summary_text += f"    #{i}: MAP@10={r['map_at_10']:.4f} | b={rw.get('body_weight',0):.1f} t={rw.get('title_weight',0):.1f} a={rw.get('anchor_weight',0):.2f} lsi={rw.get('lsi_weight',0):.1f}\n"
    
    summary_text += """
    ============================================================
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_tuning_summary.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / 'weight_tuning_summary.png'}")
    plt.close()


def main():
    # Load queries
    queries_path = parent_dir / "queries_train.json"
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"Loaded {len(queries)} queries")
    
    # Test server
    print(f"Base URL: {BASE_URL}")
    try:
        requests.get(BASE_URL, timeout=10)
        print("Server reachable\n")
    except:
        print("Cannot connect to server")
        return
    
    # Run weight tuning
    start_time = time.time()
    results = test_weights(queries, gold)
    total_time = time.time() - start_time
    
    # Create output directory
    output_dir = Path("experiments/weight_tuning_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results JSON
    with open(output_dir / 'weight_tuning_full.json', 'w') as f:
        json.dump({'results': results, 'total_time_minutes': total_time/60}, f, indent=2)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print summary
    print("\n" + "="*120)
    print("TOP 20 RESULTS (sorted by MAP@10):")
    print("="*120)
    print(f"{'Rank':<6} {'MAP@10':<10} {'HM':<10} {'P@5':<10} {'F1@30':<10} {'Time':<8} {'Weights'}")
    print("-"*120)
    
    for i, r in enumerate(results[:20], 1):
        w = r['weights']
        marker = " <--BEST" if i == 1 else ""
        print(f"{i:<6} {r['map_at_10']:.4f}     {r['harmonic_mean']:.4f}     {r['precision_at_5']:.4f}     {r['f1_at_30']:.4f}     {r['avg_time']:.2f}s   "
              f"body={w.get('body_weight',0):.2f} title={w.get('title_weight',0):.2f} anchor={w.get('anchor_weight',0):.2f} "
              f"lsi={w.get('lsi_weight',0):.2f} pr={w.get('pagerank_boost',0):.2f} pv={w.get('pageview_boost',0):.2f}{marker}")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()