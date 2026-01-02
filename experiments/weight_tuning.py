# experiments/weight_tuning.py
"""
Weight Tuning Script - Tests different weight combinations for search engine optimization.
Generates comprehensive evaluation reports with professional visualizations.
"""
import json
import requests
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product
import time

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from experiments.evaluate import load_queries_train, mean_ap_at_k
from experiments.run_evaluation import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    harmonic_mean_precision_f1,
)

# Get BASE_URL from config
try:
    import config
    BASE_URL = getattr(config, 'BASE_URL', 'http://localhost:8080')
except ImportError:
    BASE_URL = 'http://localhost:8080'


def query_with_weights(
    base_url: str,
    query: str,
    weights: Dict[str, float],
    max_retries: int = 3,
) -> Tuple[List[int], float]:
    """Query search engine with custom weights with retry logic."""
    params = {
        'query': query,
        'body_weight': weights.get('body', 1.0),
        'title_weight': weights.get('title', 0.35),
        'anchor_weight': weights.get('anchor', 0.25),
        'lsi_weight': weights.get('lsi', 0.0),
        'pagerank_boost': weights.get('pagerank', 0.15),
        'pageview_boost': weights.get('pageview', 0.10),
    }
    
    url = f"{base_url}/search_with_weights"
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=120)
            elapsed = time.time() - start_time
            
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return [], elapsed
            
            results = response.json()
            doc_ids = [int(doc_id) for doc_id, _ in results]
            return doc_ids, elapsed
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    ⚠ Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                elapsed = time.time() - start_time
                return [], elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            return [], elapsed
    
    return [], time.time() - start_time


def evaluate_with_weights(
    base_url: str,
    queries: List[str],
    gold: Dict[str, List[int]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Evaluate search engine with custom weights."""
    all_pred: Dict[str, List[int]] = {}
    times: List[float] = []
    failed_queries = 0
    
    for i, query in enumerate(queries, 1):
        doc_ids, elapsed = query_with_weights(base_url, query, weights)
        all_pred[query] = doc_ids
        times.append(elapsed)
        if not doc_ids:
            failed_queries += 1
        if i % 5 == 0:
            print(f"    {i}/{len(queries)} queries...", end='\r', flush=True)
    
    print()
    
    # Compute metrics
    map_at_10 = mean_ap_at_k(all_pred, gold, k=10)
    map_at_5 = mean_ap_at_k(all_pred, gold, k=5)
    
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
    
    avg_harmonic_mean = sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0.0
    avg_p5 = sum(precisions_5) / len(precisions_5) if precisions_5 else 0.0
    avg_f1_30 = sum(f1_scores_30) / len(f1_scores_30) if f1_scores_30 else 0.0
    mean_time = sum(times) / len(times) if times else 0.0
    
    return {
        'map_at_10': map_at_10,
        'map_at_5': map_at_5,
        'precision_at_5': avg_p5,
        'f1_at_30': avg_f1_30,
        'harmonic_mean_p5_f1_30': avg_harmonic_mean,
        'mean_time': mean_time,
        'failed_queries': failed_queries,
        'weights': weights.copy(),
    }


def generate_weight_combinations() -> List[Dict[str, float]]:
    """Generate comprehensive weight combinations with emphasis on title/anchor > body."""
    combinations = []
    
    # =========================================================================
    # SECTION 1: Title and Anchor dominant configurations
    # =========================================================================
    
    # Title dominant - body minimal
    for title_w in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for anchor_w in [0.3, 0.5, 0.75, 1.0]:
            for body_w in [0.0, 0.1, 0.2, 0.3]:
                combinations.append({
                    'body': body_w, 'title': title_w, 'anchor': anchor_w,
                    'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
                })
    
    # Anchor dominant - body minimal
    for anchor_w in [1.0, 1.5, 2.0, 2.5]:
        for title_w in [0.5, 0.75, 1.0, 1.5]:
            for body_w in [0.0, 0.1, 0.2, 0.3]:
                combinations.append({
                    'body': body_w, 'title': title_w, 'anchor': anchor_w,
                    'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
                })
    
    # Title + Anchor equal, both high
    for high_w in [1.0, 1.5, 2.0, 2.5]:
        for body_w in [0.0, 0.1, 0.2, 0.3, 0.5]:
            combinations.append({
                'body': body_w, 'title': high_w, 'anchor': high_w,
                'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
            })
    
    # Zero body - pure title/anchor
    for title_w in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for anchor_w in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            combinations.append({
                'body': 0.0, 'title': title_w, 'anchor': anchor_w,
                'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
            })
    
    # =========================================================================
    # SECTION 2: PageRank/PageView variations with title/anchor focus
    # =========================================================================
    
    # High PageRank boost with title/anchor dominant
    for pr_boost in [0.2, 0.3, 0.4, 0.5]:
        for pv_boost in [0.1, 0.2, 0.3]:
            combinations.append({
                'body': 0.1, 'title': 2.0, 'anchor': 1.0,
                'lsi': 0.0, 'pagerank': pr_boost, 'pageview': pv_boost
            })
            combinations.append({
                'body': 0.0, 'title': 1.5, 'anchor': 0.75,
                'lsi': 0.0, 'pagerank': pr_boost, 'pageview': pv_boost
            })
    
    # No boosts - pure signal weights
    for title_w in [1.0, 1.5, 2.0, 2.5]:
        for anchor_w in [0.5, 0.75, 1.0, 1.5]:
            combinations.append({
                'body': 0.0, 'title': title_w, 'anchor': anchor_w,
                'lsi': 0.0, 'pagerank': 0.0, 'pageview': 0.0
            })
    
    # =========================================================================
    # SECTION 3: LSI combinations (with title/anchor focus)
    # =========================================================================
    
    # LSI with title/anchor dominant
    for lsi_w in [0.1, 0.2, 0.3, 0.5]:
        for title_w in [1.0, 1.5, 2.0]:
            for anchor_w in [0.5, 0.75, 1.0]:
                combinations.append({
                    'body': 0.1, 'title': title_w, 'anchor': anchor_w,
                    'lsi': lsi_w, 'pagerank': 0.15, 'pageview': 0.10
                })
    
    # =========================================================================
    # SECTION 4: Balanced configurations (for comparison)
    # =========================================================================
    
    # Balanced body/title/anchor
    for balance in [0.5, 0.75, 1.0]:
        combinations.append({
            'body': balance, 'title': balance, 'anchor': balance,
            'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
        })
    
    # Body-focused (baseline comparison)
    combinations.append({
        'body': 1.0, 'title': 0.35, 'anchor': 0.25,
        'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
    })
    
    # =========================================================================
    # SECTION 5: Extreme configurations
    # =========================================================================
    
    # Title only
    combinations.append({
        'body': 0.0, 'title': 1.0, 'anchor': 0.0,
        'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
    })
    
    # Anchor only
    combinations.append({
        'body': 0.0, 'title': 0.0, 'anchor': 1.0,
        'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
    })
    
    # Title only with high boosts
    combinations.append({
        'body': 0.0, 'title': 1.0, 'anchor': 0.0,
        'lsi': 0.0, 'pagerank': 0.5, 'pageview': 0.3
    })
    
    # Very high title
    for title_w in [3.0, 4.0, 5.0]:
        combinations.append({
            'body': 0.0, 'title': title_w, 'anchor': 0.5,
            'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
        })
    
    # =========================================================================
    # SECTION 6: Fine-grained grid search around promising areas
    # =========================================================================
    
    # Fine grid: title 1.5-2.5, anchor 0.5-1.0, body 0-0.2
    for title_w in [1.5, 1.75, 2.0, 2.25, 2.5]:
        for anchor_w in [0.5, 0.625, 0.75, 0.875, 1.0]:
            for body_w in [0.0, 0.05, 0.1, 0.15, 0.2]:
                combinations.append({
                    'body': body_w, 'title': title_w, 'anchor': anchor_w,
                    'lsi': 0.0, 'pagerank': 0.15, 'pageview': 0.10
                })
    
    # Remove duplicates
    seen = set()
    unique_combos = []
    for combo in combinations:
        key = tuple(sorted((k, round(v, 3)) for k, v in combo.items()))
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)
    
    print(f"Generated {len(unique_combos)} unique weight combinations")
    return unique_combos


def create_visualizations(results: List[Dict], output_dir: Path):
    """Create professional visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        plt.style.use('seaborn-v0_8-whitegrid')
    except ImportError:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.style.use('ggplot')
        except:
            print("⚠ matplotlib not available, skipping visualizations")
            return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Color palette
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
        'dark': '#2c3e50',
    }
    
    sorted_results = sorted(results, key=lambda x: x['map_at_10'], reverse=True)
    
    # =========================================================================
    # 1. Main Performance Overview (2x2 grid)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('🔍 Search Engine Weight Tuning Results', fontsize=20, fontweight='bold', y=1.02)
    
    # 1a. Top 20 configurations bar chart
    ax = axes[0, 0]
    top20 = sorted_results[:20]
    x = np.arange(len(top20))
    map10_vals = [r['map_at_10'] for r in top20]
    
    colors = [COLORS['gold'] if i < 3 else COLORS['silver'] if i < 7 else COLORS['blue'] for i in range(len(top20))]
    bars = ax.bar(x, map10_vals, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('🏆 Top 20 Configurations by MAP@10', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i+1}' for i in range(len(top20))], rotation=45, ha='right')
    ax.set_ylim(0, max(map10_vals) * 1.15)
    
    # Add value labels on bars
    for bar, val in zip(bars, map10_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 1b. MAP@10 vs Harmonic Mean scatter
    ax = axes[0, 1]
    map10_all = [r['map_at_10'] for r in results]
    hm_all = [r['harmonic_mean_p5_f1_30'] for r in results]
    
    scatter = ax.scatter(map10_all, hm_all, c=map10_all, cmap='viridis', alpha=0.6, s=50, edgecolors='white', linewidth=0.3)
    
    # Highlight top 5
    top5_map = [r['map_at_10'] for r in sorted_results[:5]]
    top5_hm = [r['harmonic_mean_p5_f1_30'] for r in sorted_results[:5]]
    ax.scatter(top5_map, top5_hm, c=COLORS['red'], s=200, marker='★', label='Top 5', zorder=10, edgecolors='white')
    
    ax.set_xlabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_ylabel('Harmonic Mean (P@5, F1@30)', fontsize=12, fontweight='bold')
    ax.set_title('📊 Performance Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    plt.colorbar(scatter, ax=ax, label='MAP@10')
    
    # 1c. Weight distribution for top 10
    ax = axes[1, 0]
    top10 = sorted_results[:10]
    weight_names = ['body', 'title', 'anchor', 'pagerank', 'pageview']
    x = np.arange(len(top10))
    width = 0.15
    
    weight_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple'], COLORS['teal']]
    for i, (wname, wcolor) in enumerate(zip(weight_names, weight_colors)):
        vals = [r['weights'].get(wname, 0) for r in top10]
        ax.bar(x + i*width, vals, width, label=wname.capitalize(), color=wcolor, alpha=0.85)
    
    ax.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Value', fontsize=12, fontweight='bold')
    ax.set_title('⚖️ Weight Distribution for Top 10', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*2)
    ax.set_xticklabels([f'#{i+1}' for i in range(len(top10))])
    ax.legend(loc='upper right', fontsize=9)
    
    # 1d. Metrics comparison for top 10
    ax = axes[1, 1]
    metrics = ['map_at_10', 'map_at_5', 'precision_at_5', 'f1_at_30']
    metric_labels = ['MAP@10', 'MAP@5', 'P@5', 'F1@30']
    metric_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]
    x = np.arange(len(top10))
    width = 0.2
    
    for i, (metric, label, mcolor) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [r.get(metric, 0) for r in top10]
        ax.bar(x + i*width, vals, width, label=label, color=mcolor, alpha=0.85)
    
    ax.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('📈 Multiple Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([f'#{i+1}' for i in range(len(top10))])
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overview.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'overview.png'}")
    plt.close()
    
    # =========================================================================
    # 2. Weight Sensitivity Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🎯 Weight Sensitivity Analysis', fontsize=18, fontweight='bold', y=1.02)
    
    weight_params = ['body', 'title', 'anchor', 'pagerank', 'pageview', 'lsi']
    colors_list = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple'], COLORS['teal'], COLORS['red']]
    
    for idx, (param, color) in enumerate(zip(weight_params, colors_list)):
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
            
            ax.errorbar(sorted_vals, means, yerr=stds, marker='o', markersize=10,
                       capsize=6, capthick=2, color=color, linewidth=2.5, elinewidth=2,
                       markerfacecolor='white', markeredgewidth=2)
            
            # Fill between for confidence interval
            ax.fill_between(sorted_vals, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)
            
            # Add count annotations
            for x_val, y_val, c in zip(sorted_vals, means, counts):
                ax.annotate(f'n={c}', (x_val, y_val), textcoords="offset points", 
                           xytext=(0, 12), ha='center', fontsize=8, alpha=0.7)
        
        ax.set_xlabel(f'{param.capitalize()} Weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean MAP@10', fontsize=12, fontweight='bold')
        ax.set_title(f'{param.capitalize()} Sensitivity', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'sensitivity_analysis.png'}")
    plt.close()
    
    # =========================================================================
    # 3. Heatmap: Title vs Anchor (body=0)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Filter results where body is close to 0
    zero_body = [r for r in results if r['weights'].get('body', 0) <= 0.1]
    
    if zero_body:
        # Create grid
        title_vals = sorted(set(round(r['weights'].get('title', 0), 2) for r in zero_body))
        anchor_vals = sorted(set(round(r['weights'].get('anchor', 0), 2) for r in zero_body))
        
        heatmap_data = np.zeros((len(anchor_vals), len(title_vals)))
        heatmap_data[:] = np.nan
        
        for r in zero_body:
            t = round(r['weights'].get('title', 0), 2)
            a = round(r['weights'].get('anchor', 0), 2)
            if t in title_vals and a in anchor_vals:
                ti = title_vals.index(t)
                ai = anchor_vals.index(a)
                if np.isnan(heatmap_data[ai, ti]) or r['map_at_10'] > heatmap_data[ai, ti]:
                    heatmap_data[ai, ti] = r['map_at_10']
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        
        ax.set_xticks(np.arange(len(title_vals)))
        ax.set_yticks(np.arange(len(anchor_vals)))
        ax.set_xticklabels([f'{v:.2f}' for v in title_vals], rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([f'{v:.2f}' for v in anchor_vals], fontsize=9)
        ax.set_xlabel('Title Weight', fontsize=14, fontweight='bold')
        ax.set_ylabel('Anchor Weight', fontsize=14, fontweight='bold')
        ax.set_title('🗺️ MAP@10 Heatmap: Title vs Anchor (Body ≤ 0.1)', fontsize=16, fontweight='bold')
        
        # Add value annotations
        for i in range(len(anchor_vals)):
            for j in range(len(title_vals)):
                if not np.isnan(heatmap_data[i, j]):
                    text_color = 'white' if heatmap_data[i, j] > np.nanmean(heatmap_data) else 'black'
                    ax.text(j, i, f'{heatmap_data[i, j]:.3f}', ha='center', va='center', 
                           fontsize=8, color=text_color, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, label='MAP@10', pad=0.02)
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'title_vs_anchor_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'title_vs_anchor_heatmap.png'}")
    plt.close()
    
    # =========================================================================
    # 4. Feature Comparison Box Plots
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('📦 Feature Impact Analysis', fontsize=18, fontweight='bold', y=1.02)
    
    # 4a. Body weight impact
    ax = axes[0]
    low_body = [r['map_at_10'] for r in results if r['weights'].get('body', 0) <= 0.1]
    med_body = [r['map_at_10'] for r in results if 0.1 < r['weights'].get('body', 0) <= 0.5]
    high_body = [r['map_at_10'] for r in results if r['weights'].get('body', 0) > 0.5]
    
    data_body = [d for d in [low_body, med_body, high_body] if d]
    labels_body = [l for l, d in zip(['Low (≤0.1)', 'Medium (0.1-0.5)', 'High (>0.5)'], 
                                      [low_body, med_body, high_body]) if d]
    
    bp = ax.boxplot(data_body, labels=labels_body, patch_artist=True)
    box_colors = [COLORS['green'], COLORS['blue'], COLORS['red']][:len(data_body)]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('Body Weight Impact', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4b. Title weight impact
    ax = axes[1]
    low_title = [r['map_at_10'] for r in results if r['weights'].get('title', 0) < 1.0]
    med_title = [r['map_at_10'] for r in results if 1.0 <= r['weights'].get('title', 0) <= 2.0]
    high_title = [r['map_at_10'] for r in results if r['weights'].get('title', 0) > 2.0]
    
    data_title = [d for d in [low_title, med_title, high_title] if d]
    labels_title = [l for l, d in zip(['Low (<1.0)', 'Medium (1.0-2.0)', 'High (>2.0)'], 
                                       [low_title, med_title, high_title]) if d]
    
    bp = ax.boxplot(data_title, labels=labels_title, patch_artist=True)
    box_colors = [COLORS['red'], COLORS['blue'], COLORS['green']][:len(data_title)]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('Title Weight Impact', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4c. Anchor weight impact
    ax = axes[2]
    low_anchor = [r['map_at_10'] for r in results if r['weights'].get('anchor', 0) < 0.5]
    med_anchor = [r['map_at_10'] for r in results if 0.5 <= r['weights'].get('anchor', 0) <= 1.0]
    high_anchor = [r['map_at_10'] for r in results if r['weights'].get('anchor', 0) > 1.0]
    
    data_anchor = [d for d in [low_anchor, med_anchor, high_anchor] if d]
    labels_anchor = [l for l, d in zip(['Low (<0.5)', 'Medium (0.5-1.0)', 'High (>1.0)'], 
                                        [low_anchor, med_anchor, high_anchor]) if d]
    
    bp = ax.boxplot(data_anchor, labels=labels_anchor, patch_artist=True)
    box_colors = [COLORS['red'], COLORS['blue'], COLORS['green']][:len(data_anchor)]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('Anchor Weight Impact', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'feature_comparison.png'}")
    plt.close()
    
    # =========================================================================
    # 5. Top 10 Detailed Cards
    # =========================================================================
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    fig.suptitle('🥇 Top 10 Configurations - Detailed View', fontsize=18, fontweight='bold', y=1.02)
    
    medal_colors = [COLORS['gold'], COLORS['silver'], COLORS['bronze']] + [COLORS['dark']] * 7
    
    for idx, r in enumerate(sorted_results[:10]):
        ax = axes[idx // 5, idx % 5]
        w = r['weights']
        
        # Bar chart of weights
        weight_names = ['Body', 'Title', 'Anchor', 'PR', 'PV']
        weight_vals = [w.get('body', 0), w.get('title', 0), w.get('anchor', 0), 
                      w.get('pagerank', 0), w.get('pageview', 0)]
        bar_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple'], COLORS['teal']]
        
        bars = ax.bar(weight_names, weight_vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1)
        
        # Medal border
        for spine in ax.spines.values():
            spine.set_edgecolor(medal_colors[idx])
            spine.set_linewidth(4)
        
        # Medal emoji
        medal = '🥇' if idx == 0 else '🥈' if idx == 1 else '🥉' if idx == 2 else f'#{idx+1}'
        ax.set_title(f"{medal} MAP@10={r['map_at_10']:.4f}", fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(weight_vals) * 1.4 if max(weight_vals) > 0 else 1)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Add value labels
        for bar, val in zip(bars, weight_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top10_detailed.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'top10_detailed.png'}")
    plt.close()
    
    # =========================================================================
    # 6. Summary Card
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    
    best = sorted_results[0]
    
    # Create styled summary
    summary_lines = [
        "═" * 60,
        "🔍 WEIGHT TUNING SUMMARY",
        "═" * 60,
        "",
        f"📊 Total Configurations Tested: {len(results)}",
        "",
        "🏆 BEST CONFIGURATION (#1):",
        "─" * 40,
        f"  • MAP@10:        {best['map_at_10']:.4f}",
        f"  • MAP@5:         {best['map_at_5']:.4f}",
        f"  • Precision@5:   {best.get('precision_at_5', 0):.4f}",
        f"  • F1@30:         {best.get('f1_at_30', 0):.4f}",
        f"  • Harmonic Mean: {best['harmonic_mean_p5_f1_30']:.4f}",
        f"  • Query Time:    {best['mean_time']:.2f}s",
        "",
        "⚖️ OPTIMAL WEIGHTS:",
        "─" * 40,
        f"  • Body:     {best['weights'].get('body', 0):.2f}",
        f"  • Title:    {best['weights'].get('title', 0):.2f}",
        f"  • Anchor:   {best['weights'].get('anchor', 0):.2f}",
        f"  • LSI:      {best['weights'].get('lsi', 0):.2f}",
        f"  • PageRank: {best['weights'].get('pagerank', 0):.2f}",
        f"  • PageView: {best['weights'].get('pageview', 0):.2f}",
        "",
        "📈 PERFORMANCE RANGE:",
        "─" * 40,
        f"  • Best MAP@10:  {max(r['map_at_10'] for r in results):.4f}",
        f"  • Worst MAP@10: {min(r['map_at_10'] for r in results):.4f}",
        f"  • Mean MAP@10:  {np.mean([r['map_at_10'] for r in results]):.4f}",
        f"  • Std MAP@10:   {np.std([r['map_at_10'] for r in results]):.4f}",
        "",
        "═" * 60,
    ]
    
    summary_text = '\n'.join(summary_lines)
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'summary.png'}")
    plt.close()


def main():
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Test different weight combinations")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL of search engine")
    parser.add_argument("--queries", default="test_queries.json", help="Path to queries JSON file")
    parser.add_argument("--output", default="weight_tuning_results.json", help="Output file for results")
    parser.add_argument("--output-dir", default="experiments/weight_tuning_results", help="Directory for output files")
    parser.add_argument("--max-combinations", type=int, default=None, help="Maximum number of combinations to test")
    args = parser.parse_args()
    
    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.is_absolute() and not queries_path.exists():
        queries_path = parent_dir / args.queries
    
    if not queries_path.exists():
        print(f"❌ Error: {queries_path} not found!")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"✅ Loaded {len(queries)} queries from {queries_path}")
    
    # Generate weight combinations
    weight_combos = generate_weight_combinations()
    if args.max_combinations:
        weight_combos = weight_combos[:args.max_combinations]
    
    print(f"\n🧪 Testing {len(weight_combos)} weight combinations...")
    print(f"🌐 Base URL: {args.base_url}")
    print("=" * 80)
    
    # Test server connection
    print("\n🔌 Testing server connection...")
    try:
        test_response = requests.get(args.base_url, timeout=10)
        print(f"✅ Server is reachable (status: {test_response.status_code})")
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to server at {args.base_url}")
        return
    except Exception as e:
        print(f"⚠️ Warning: {e}")
    
    print("=" * 80)
    
    results = []
    start_time = time.time()
    
    for i, weights in enumerate(weight_combos, 1):
        elapsed_total = time.time() - start_time
        eta = (elapsed_total / i) * (len(weight_combos) - i) if i > 0 else 0
        
        print(f"\n[{i}/{len(weight_combos)}] ⏱️ ETA: {eta/60:.1f}min")
        print(f"  Weights: body={weights['body']:.2f}, title={weights['title']:.2f}, "
              f"anchor={weights['anchor']:.2f}, pr={weights['pagerank']:.2f}")
        
        try:
            metrics = evaluate_with_weights(args.base_url, queries, gold, weights)
            results.append(metrics)
            print(f"  ✅ MAP@10: {metrics['map_at_10']:.4f}, HM: {metrics['harmonic_mean_p5_f1_30']:.4f}, "
                  f"Time: {metrics['mean_time']:.2f}s")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Sort by MAP@10
    results.sort(key=lambda x: x['map_at_10'], reverse=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_path = output_dir / args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_combinations_tested': len(results),
            'queries_file': str(queries_path),
            'base_url': args.base_url,
            'results': results,
        }, f, indent=2)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("🎨 Creating visualizations...")
    print("=" * 80)
    create_visualizations(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n🏆 Top 20 configurations by MAP@10:")
    print(f"{'Rank':<5} {'MAP@10':<9} {'MAP@5':<9} {'HM':<9} {'Time':<7} {'Weights'}")
    print("-" * 110)
    
    for i, r in enumerate(results[:20], 1):
        w = r['weights']
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
        print(f"{medal:<5} {r['map_at_10']:.4f}    {r['map_at_5']:.4f}    {r['harmonic_mean_p5_f1_30']:.4f}    "
              f"{r['mean_time']:.2f}s   "
              f"b={w['body']:.2f} t={w['title']:.2f} a={w['anchor']:.2f} "
              f"pr={w['pagerank']:.2f} pv={w['pageview']:.2f}")
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"✅ Visualizations saved to: {output_dir}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()