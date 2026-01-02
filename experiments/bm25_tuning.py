# experiments/bm25_tuning.py
"""
BM25 Parameter Tuning Script.
Tests different k1 and b parameter combinations.
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
from experiments.run_evaluation import harmonic_mean_precision_f1

try:
    import config
    BASE_URL = getattr(config, 'BASE_URL', 'http://localhost:8080')
except ImportError:
    BASE_URL = 'http://localhost:8080'


def query_with_bm25_params(
    base_url: str,
    query: str,
    k1: float,
    b: float,
    max_retries: int = 3,
) -> Tuple[List[int], float]:
    """Query search engine with custom BM25 parameters."""
    params = {
        'query': query,
        'k1': k1,
        'b': b,
    }
    
    url = f"{base_url}/search_body_bm25"
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
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return [], time.time() - start_time
    
    return [], time.time() - start_time


def evaluate_bm25_params(
    base_url: str,
    queries: List[str],
    gold: Dict[str, List[int]],
    k1: float,
    b: float,
) -> Dict[str, float]:
    """Evaluate BM25 with specific k1 and b parameters."""
    all_pred: Dict[str, List[int]] = {}
    times: List[float] = []
    
    for i, query in enumerate(queries, 1):
        doc_ids, elapsed = query_with_bm25_params(base_url, query, k1, b)
        all_pred[query] = doc_ids
        times.append(elapsed)
        if i % 5 == 0:
            print(f"    {i}/{len(queries)}...", end='\r', flush=True)
    
    print()
    
    map_at_10 = mean_ap_at_k(all_pred, gold, k=10)
    map_at_5 = mean_ap_at_k(all_pred, gold, k=5)
    
    harmonic_means = []
    for q in queries:
        pred = all_pred.get(q, [])
        gold_list = gold.get(q, [])
        hm = harmonic_mean_precision_f1(pred, gold_list, p_k=5, f1_k=30)
        harmonic_means.append(hm)
    
    return {
        'k1': k1,
        'b': b,
        'map_at_10': map_at_10,
        'map_at_5': map_at_5,
        'harmonic_mean': sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0,
        'mean_time': sum(times) / len(times) if times else 0,
    }


def generate_bm25_combinations() -> List[Tuple[float, float]]:
    """Generate k1 and b parameter combinations to test."""
    # Coarse grid first
    k1_values = [0.5, 0.75, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]
    b_values = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    
    combinations = []
    for k1 in k1_values:
        for b in b_values:
            combinations.append((k1, b))
    
    print(f"Generated {len(combinations)} BM25 parameter combinations")
    return combinations


def create_visualizations(results: List[Dict], output_dir: Path):
    """Create visualization plots for BM25 tuning."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by MAP@10
    sorted_results = sorted(results, key=lambda x: x['map_at_10'], reverse=True)
    
    # =========================================================================
    # 1. Heatmap: k1 vs b
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    k1_vals = sorted(set(r['k1'] for r in results))
    b_vals = sorted(set(r['b'] for r in results))
    
    heatmap_data = np.zeros((len(b_vals), len(k1_vals)))
    heatmap_data[:] = np.nan
    
    for r in results:
        k1_idx = k1_vals.index(r['k1'])
        b_idx = b_vals.index(r['b'])
        heatmap_data[b_idx, k1_idx] = r['map_at_10']
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(k1_vals)))
    ax.set_yticks(np.arange(len(b_vals)))
    ax.set_xticklabels([f'{v:.2f}' for v in k1_vals], rotation=45, ha='right')
    ax.set_yticklabels([f'{v:.2f}' for v in b_vals])
    ax.set_xlabel('k1 (Term Frequency Saturation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('b (Document Length Normalization)', fontsize=14, fontweight='bold')
    ax.set_title('🎯 BM25 Parameter Tuning: MAP@10 Heatmap', fontsize=16, fontweight='bold')
    
    # Add value annotations
    for i in range(len(b_vals)):
        for j in range(len(k1_vals)):
            if not np.isnan(heatmap_data[i, j]):
                text_color = 'white' if heatmap_data[i, j] > np.nanmean(heatmap_data) else 'black'
                ax.text(j, i, f'{heatmap_data[i, j]:.3f}', ha='center', va='center',
                       fontsize=8, color=text_color, fontweight='bold')
    
    # Mark best
    best = sorted_results[0]
    best_k1_idx = k1_vals.index(best['k1'])
    best_b_idx = b_vals.index(best['b'])
    ax.add_patch(plt.Rectangle((best_k1_idx - 0.5, best_b_idx - 0.5), 1, 1,
                                fill=False, edgecolor='gold', linewidth=4))
    
    plt.colorbar(im, ax=ax, label='MAP@10')
    plt.tight_layout()
    plt.savefig(output_dir / 'bm25_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'bm25_heatmap.png'}")
    plt.close()
    
    # =========================================================================
    # 2. k1 Sensitivity (fixed b at optimal)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # k1 sensitivity
    ax = axes[0]
    best_b = best['b']
    k1_results = [r for r in results if r['b'] == best_b]
    k1_results.sort(key=lambda x: x['k1'])
    
    k1_x = [r['k1'] for r in k1_results]
    k1_y = [r['map_at_10'] for r in k1_results]
    
    ax.plot(k1_x, k1_y, marker='o', markersize=10, linewidth=2.5, color='#3498db')
    ax.fill_between(k1_x, k1_y, alpha=0.2, color='#3498db')
    ax.axvline(best['k1'], color='gold', linestyle='--', linewidth=2, label=f'Best k1={best["k1"]}')
    ax.set_xlabel('k1', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title(f'k1 Sensitivity (b={best_b})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # b sensitivity
    ax = axes[1]
    best_k1 = best['k1']
    b_results = [r for r in results if r['k1'] == best_k1]
    b_results.sort(key=lambda x: x['b'])
    
    b_x = [r['b'] for r in b_results]
    b_y = [r['map_at_10'] for r in b_results]
    
    ax.plot(b_x, b_y, marker='o', markersize=10, linewidth=2.5, color='#2ecc71')
    ax.fill_between(b_x, b_y, alpha=0.2, color='#2ecc71')
    ax.axvline(best['b'], color='gold', linestyle='--', linewidth=2, label=f'Best b={best["b"]}')
    ax.set_xlabel('b', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title(f'b Sensitivity (k1={best_k1})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bm25_sensitivity.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'bm25_sensitivity.png'}")
    plt.close()
    
    # =========================================================================
    # 3. Top 20 configurations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top20 = sorted_results[:20]
    x = np.arange(len(top20))
    map10_vals = [r['map_at_10'] for r in top20]
    labels = [f'k1={r["k1"]}\nb={r["b"]}' for r in top20]
    
    colors = ['#FFD700' if i < 3 else '#C0C0C0' if i < 7 else '#3498db' for i in range(len(top20))]
    bars = ax.bar(x, map10_vals, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAP@10', fontsize=12, fontweight='bold')
    ax.set_title('🏆 Top 20 BM25 Parameter Combinations', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    for bar, val in zip(bars, map10_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bm25_top20.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'bm25_top20.png'}")
    plt.close()
    
    # =========================================================================
    # 4. Summary
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    summary_lines = [
        "═" * 50,
        "🎯 BM25 PARAMETER TUNING SUMMARY",
        "═" * 50,
        "",
        f"📊 Configurations Tested: {len(results)}",
        "",
        "🏆 OPTIMAL PARAMETERS:",
        "─" * 35,
        f"  • k1 = {best['k1']}",
        f"  • b  = {best['b']}",
        "",
        "📈 PERFORMANCE:",
        "─" * 35,
        f"  • MAP@10:        {best['map_at_10']:.4f}",
        f"  • MAP@5:         {best['map_at_5']:.4f}",
        f"  • Harmonic Mean: {best['harmonic_mean']:.4f}",
        f"  • Query Time:    {best['mean_time']:.2f}s",
        "",
        "📉 RANGE:",
        "─" * 35,
        f"  • Best:  {max(r['map_at_10'] for r in results):.4f}",
        f"  • Worst: {min(r['map_at_10'] for r in results):.4f}",
        f"  • Mean:  {np.mean([r['map_at_10'] for r in results]):.4f}",
        "",
        "💡 INTERPRETATION:",
        "─" * 35,
        f"  • k1={best['k1']}: {'High' if best['k1'] > 1.5 else 'Low'} term freq saturation",
        f"  • b={best['b']}: {'Strong' if best['b'] > 0.6 else 'Weak'} length normalization",
        "",
        "═" * 50,
    ]
    
    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bm25_summary.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir / 'bm25_summary.png'}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune BM25 parameters")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL")
    parser.add_argument("--queries", default="queries_train.json", help="Queries file")
    parser.add_argument("--output-dir", default="experiments/bm25_tuning_results", help="Output dir")
    args = parser.parse_args()
    
    # Load queries - use BASE_DIR from config for queries
    import config
    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        # Try relative to queries directory (root by default)
        queries_path = config.QUERIES_DIR / args.queries
        if not queries_path.exists():
            # Fallback to parent_dir for backward compatibility
            queries_path = parent_dir / args.queries
    
    if not queries_path.exists():
        print(f"❌ {queries_path} not found!")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"✅ Loaded {len(queries)} queries")
    
    # Generate combinations
    combinations = generate_bm25_combinations()
    
    print(f"\n🧪 Testing {len(combinations)} BM25 parameter combinations...")
    print(f"🌐 Base URL: {args.base_url}")
    print("=" * 60)
    
    # Test server
    try:
        requests.get(args.base_url, timeout=10)
        print("✅ Server reachable")
    except:
        print("❌ Cannot connect to server")
        return
    
    results = []
    start_time = time.time()
    
    for i, (k1, b) in enumerate(combinations, 1):
        eta = ((time.time() - start_time) / i) * (len(combinations) - i) / 60
        print(f"\n[{i}/{len(combinations)}] k1={k1}, b={b} (ETA: {eta:.1f}min)")
        
        try:
            metrics = evaluate_bm25_params(args.base_url, queries, gold, k1, b)
            results.append(metrics)
            print(f"  ✅ MAP@10: {metrics['map_at_10']:.4f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Sort
    results.sort(key=lambda x: x['map_at_10'], reverse=True)
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'bm25_results.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    # Visualize
    print("\n🎨 Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 TOP 10 BM25 CONFIGURATIONS:")
    print("=" * 60)
    print(f"{'Rank':<5} {'k1':<6} {'b':<6} {'MAP@10':<10}")
    print("-" * 30)
    
    for i, r in enumerate(results[:10], 1):
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
        print(f"{medal:<5} {r['k1']:<6.2f} {r['b']:<6.2f} {r['map_at_10']:.4f}")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"⏱️ Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Recommend updating config
    best = results[0]
    print(f"\n💡 RECOMMENDED: Update config.py with:")
    print(f"   BM25_K1 = {best['k1']}")
    print(f"   BM25_B = {best['b']}")


if __name__ == "__main__":
    main()