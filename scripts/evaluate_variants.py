# scripts/evaluate_variants.py
"""
Evaluate different search engine variants:
1) Baseline (no GloVe)
2) GloVe only

Outputs CSV summary and console markdown report with deltas vs baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

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

# Use BASE_URL from config, or fallback to default
BASE_URL = getattr(config, 'BASE_URL', f"http://{getattr(config, 'INSTANCE_IP', 'localhost')}:{getattr(config, 'INSTANCE_PORT', 8080)}")


def query_search_engine(
    query: str, 
    enable_glove: bool = False,
    glove_beta: float | None = None,
    glove_pool: int | None = None,
    glove_top_k: int | None = None,
) -> Tuple[List[int], float]:
    """
    Query the search engine with a specific variant configuration.
    
    Args:
        query: Search query
        enable_glove: Enable GloVe reranking
        glove_beta: GloVe beta weight (overrides config if provided)
        glove_pool: GloVe candidate pool size (overrides config if provided)
        glove_top_k: GloVe top-k for query embedding (overrides config if provided)
        
    Returns:
        (doc_ids, elapsed_time) tuple
    """
    url = f"{BASE_URL}/search"
    params = {
        'query': query,
        'enable_glove': str(enable_glove).lower(),
    }
    if glove_beta is not None:
        params['glove_beta'] = str(glove_beta)
    if glove_pool is not None:
        params['glove_pool'] = str(glove_pool)
    if glove_top_k is not None:
        params['glove_top_k'] = str(glove_top_k)
    
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
        print(f"Error querying '{query}': {e}")
        return [], time.time() - start_time


def evaluate_variant(
    variant_name: str,
    queries: List[str],
    gold: Dict[str, List[int]],
    enable_glove: bool = False,
    glove_beta: float | None = None,
    glove_pool: int | None = None,
    glove_top_k: int | None = None,
) -> Dict:
    """
    Evaluate a search variant.
    
    Args:
        variant_name: Name of the variant
        queries: List of queries
        gold: Gold standard relevance judgments
        enable_glove: Enable GloVe reranking
        glove_beta: GloVe beta weight (overrides config if provided)
        glove_pool: GloVe candidate pool (overrides config if provided)
        glove_top_k: GloVe top-k for query embedding (overrides config if provided)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {variant_name}...")
    print(f"  GloVe: {enable_glove}")
    if glove_beta is not None:
        print(f"  GloVe beta: {glove_beta} (custom)")
    if glove_pool is not None:
        print(f"  GloVe pool: {glove_pool} (custom)")
    if glove_top_k is not None:
        print(f"  GloVe top_k: {glove_top_k} (custom)")
    
    all_pred: Dict[str, List[int]] = {}
    times: List[float] = []
    
    for i, query in enumerate(queries, 1):
        doc_ids, elapsed = query_search_engine(
            query, 
            enable_glove=enable_glove,
            glove_beta=glove_beta,
            glove_pool=glove_pool,
            glove_top_k=glove_top_k,
        )
        all_pred[query] = doc_ids
        times.append(elapsed)
        if i % 10 == 0:
            print(f"  Processed {i}/{len(queries)} queries...")
    
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
        'variant': variant_name,
        'avg_precision_at_10': avg_precision_at_10,
        'avg_precision_at_5': avg_precision_at_5,
        'precision_at_5': sum(precisions_5) / len(precisions_5) if precisions_5 else 0.0,
        'f1_at_30': sum(f1_scores_30) / len(f1_scores_30) if f1_scores_30 else 0.0,
        'harmonic_mean': sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0.0,
        'avg_time': avg_time,
    }


def main():
    """Main evaluation function."""
    # Load queries
    queries_path = config.QUERIES_DIR / "queries_train.json"
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"Loaded {len(queries)} queries")
    
    # Test server
    print(f"Base URL: {BASE_URL}")
    try:
        # Test by querying the search endpoint with an empty query
        test_url = f"{BASE_URL}/search"
        response = requests.get(test_url, params={'query': 'test'}, timeout=10)
        if response.status_code == 200:
            print("✓ Server is reachable and responding")
        else:
            print(f"⚠ Server responded with status {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("  Make sure the server is running and BASE_URL is correct")
        return
    
    # Note: This script uses query parameters to control GloVe,
    # so you don't need to restart the server for each variant.
    # However, make sure ENABLE_GLOVE=True in config.py
    # so that the GloVe embeddings are loaded.
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate search engine variants")
    parser.add_argument(
        '--variant',
        choices=['baseline', 'glove', 'all'],
        default='all',
        help='Variant to evaluate (default: all)'
    )
    parser.add_argument(
        '--glove-beta',
        type=float,
        default=None,
        help='GloVe beta weight (overrides config, e.g., 0.5, 1.0)'
    )
    parser.add_argument(
        '--glove-pool',
        type=int,
        default=None,
        help='GloVe candidate pool size (overrides config, e.g., 200)'
    )
    parser.add_argument(
        '--glove-top-k',
        type=int,
        default=None,
        help='GloVe top-k for query embedding (overrides config, e.g., 15)'
    )
    args = parser.parse_args()
    
    if args.variant == 'all':
        print("\nEvaluating all variants...")
        print("Note: Make sure ENABLE_GLOVE=True in config.py")
        print("so that the GloVe embeddings are loaded. The script will control usage via API.\n")
        
        variants_to_eval = ['baseline', 'glove']
        all_results = []
        
        for variant in variants_to_eval:
            enable_glove = (variant == 'glove')
            
            results = evaluate_variant(
                variant_name=variant,
                queries=queries,
                gold=gold,
                enable_glove=enable_glove,
                glove_beta=args.glove_beta,
                glove_pool=args.glove_pool,
                glove_top_k=args.glove_top_k,
            )
            all_results.append(results)
            
            # Save results
            output_dir = Path("experiments/variant_evaluation_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / f"{variant}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ {variant} evaluation complete. Results saved to {results_file}")
        
        # Print summary comparison
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)
        baseline = all_results[0]
        print(f"\n{'Variant':<15} {'Avg P@10':<12} {'P@5':<12} {'F1@30':<12} {'Δ Avg P@10':<12}")
        print("-" * 80)
        for r in all_results:
            delta = r['avg_precision_at_10'] - baseline['avg_precision_at_10']
            print(f"{r['variant']:<15} {r['avg_precision_at_10']:<12.4f} {r['precision_at_5']:<12.4f} {r['f1_at_30']:<12.4f} {delta:+.4f}")
        
        return
    
    # Map variant name to flags
    enable_glove = (args.variant == 'glove')
    
    # Check if GloVe is loaded
    current_glove = getattr(config, 'ENABLE_GLOVE', False)
    
    if enable_glove and not current_glove:
        print(f"\n⚠ WARNING: GloVe embeddings may not be loaded!")
        print(f"  Current config: GloVe={current_glove}")
        print(f"  Recommended: Set ENABLE_GLOVE=True in config.py")
        print(f"  (The script will control actual usage via API parameters)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Evaluate variant
    results = evaluate_variant(
        variant_name=args.variant,
        queries=queries,
        gold=gold,
        enable_glove=enable_glove,
        glove_beta=args.glove_beta,
        glove_pool=args.glove_pool,
        glove_top_k=args.glove_top_k,
    )
    
    # Save results
    output_dir = Path("experiments/variant_evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{args.variant}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "=" * 80)
    print(f"RESULTS: {args.variant.upper()}")
    print("=" * 80)
    print(f"Average Precision@10:        {results['avg_precision_at_10']:.4f}")
    print(f"Average Precision@5:         {results['avg_precision_at_5']:.4f}")
    print(f"Precision@5:   {results['precision_at_5']:.4f}")
    print(f"F1@30:         {results['f1_at_30']:.4f}")
    print(f"Harmonic Mean: {results['harmonic_mean']:.4f}")
    print(f"Avg Time:      {results['avg_time']:.2f}s")
    print(f"\nResults saved to: {results_file}")
    
    # If baseline exists, compute deltas
    baseline_file = output_dir / "baseline_results.json"
    if baseline_file.exists() and args.variant != 'baseline':
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        print("\n" + "=" * 80)
        print("DELTAS vs BASELINE")
        print("=" * 80)
        # Handle both old format (map_at_*) and new format (avg_precision_at_*)
        baseline_ap10 = baseline.get('avg_precision_at_10', baseline.get('map_at_10', 0.0))
        baseline_ap5 = baseline.get('avg_precision_at_5', baseline.get('map_at_5', 0.0))
        results_ap10 = results.get('avg_precision_at_10', results.get('map_at_10', 0.0))
        results_ap5 = results.get('avg_precision_at_5', results.get('map_at_5', 0.0))
        
        print(f"Average Precision@10:        {results_ap10 - baseline_ap10:+.4f} ({((results_ap10/baseline_ap10-1)*100):+.2f}%)" if baseline_ap10 > 0 else f"Average Precision@10:        {results_ap10 - baseline_ap10:+.4f}")
        print(f"Average Precision@5:         {results_ap5 - baseline_ap5:+.4f} ({((results_ap5/baseline_ap5-1)*100):+.2f}%)" if baseline_ap5 > 0 else f"Average Precision@5:         {results_ap5 - baseline_ap5:+.4f}")
        print(f"Precision@5:   {results['precision_at_5'] - baseline['precision_at_5']:+.4f} ({((results['precision_at_5']/baseline['precision_at_5']-1)*100):+.2f}%)" if baseline['precision_at_5'] > 0 else f"Precision@5:   {results['precision_at_5'] - baseline['precision_at_5']:+.4f}")
        print(f"F1@30:         {results['f1_at_30'] - baseline['f1_at_30']:+.4f} ({((results['f1_at_30']/baseline['f1_at_30']-1)*100):+.2f}%)" if baseline['f1_at_30'] > 0 else f"F1@30:         {results['f1_at_30'] - baseline['f1_at_30']:+.4f}")
        print(f"Harmonic Mean: {results['harmonic_mean'] - baseline['harmonic_mean']:+.4f} ({((results['harmonic_mean']/baseline['harmonic_mean']-1)*100):+.2f}%)" if baseline['harmonic_mean'] > 0 else f"Harmonic Mean: {results['harmonic_mean'] - baseline['harmonic_mean']:+.4f}")
        print(f"Avg Time:      {results['avg_time'] - baseline['avg_time']:+.2f}s")


if __name__ == "__main__":
    main()
