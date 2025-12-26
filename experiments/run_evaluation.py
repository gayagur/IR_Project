# experiments/run_evaluation.py
"""
Evaluation script for the search engine.
Measures performance metrics and timing for different ranking methods.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from experiments.evaluate import (
    average_precision_at_k,
    load_queries_train,
    mean_ap_at_k,
)


def precision_at_k(pred: List[int], gold: List[int], k: int) -> float:
    """Compute Precision@K."""
    if not gold or k == 0:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / k


def recall_at_k(pred: List[int], gold: List[int], k: int) -> float:
    """Compute Recall@K."""
    if not gold:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / len(gold_set)


def f1_at_k(pred: List[int], gold: List[int], k: int) -> float:
    """Compute F1@K."""
    p = precision_at_k(pred, gold, k)
    r = recall_at_k(pred, gold, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def harmonic_mean_precision_f1(pred: List[int], gold: List[int], p_k: int = 5, f1_k: int = 30) -> float:
    """Compute harmonic mean of Precision@P_K and F1@F1_K."""
    p = precision_at_k(pred, gold, p_k)
    f1 = f1_at_k(pred, gold, f1_k)
    if p + f1 == 0:
        return 0.0
    return 2 * p * f1 / (p + f1)


def query_search_engine(base_url: str, endpoint: str, query: str) -> Tuple[List[int], float]:
    """
    Query the search engine and return (doc_ids, time_taken).
    """
    url = f"{base_url}/{endpoint}"
    params = {"query": query}
    
    start_time = time.time()
    try:
        response = requests.get(url, params=params, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"Error {response.status_code} for query: {query}")
            return [], elapsed
        
        results = response.json()
        # Extract doc_ids from [(doc_id, title), ...] format
        doc_ids = [int(doc_id) for doc_id, _ in results]
        return doc_ids, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Exception for query '{query}': {e}")
        return [], elapsed


def evaluate_endpoint(
    base_url: str,
    endpoint: str,
    queries: List[str],
    gold: Dict[str, List[int]],
    endpoint_name: str = "",
) -> Dict[str, float]:
    """
    Evaluate a search endpoint and return metrics.
    """
    if not endpoint_name:
        endpoint_name = endpoint
    
    print(f"\nEvaluating {endpoint_name}...")
    
    all_pred: Dict[str, List[int]] = {}
    times: List[float] = []
    
    for i, query in enumerate(queries, 1):
        print(f"  Query {i}/{len(queries)}: {query[:50]}...", end=" ", flush=True)
        doc_ids, elapsed = query_search_engine(base_url, endpoint, query)
        all_pred[query] = doc_ids
        times.append(elapsed)
        print(f"{elapsed:.2f}s")
    
    # Compute metrics
    metrics = {
        "mean_time": sum(times) / len(times) if times else 0.0,
        "max_time": max(times) if times else 0.0,
        "min_time": min(times) if times else 0.0,
        "map_at_10": mean_ap_at_k(all_pred, gold, k=10),
        "map_at_5": mean_ap_at_k(all_pred, gold, k=5),
        "harmonic_mean_p5_f1_30": 0.0,
    }
    
    # Compute harmonic mean for each query
    harmonic_means = []
    for q in queries:
        pred = all_pred.get(q, [])
        gold_list = gold.get(q, [])
        hm = harmonic_mean_precision_f1(pred, gold_list, p_k=5, f1_k=30)
        harmonic_means.append(hm)
    
    metrics["harmonic_mean_p5_f1_30"] = sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0.0
    
    return metrics, all_pred


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate search engine performance")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Base URL of search engine")
    parser.add_argument("--queries", default="queries_train.json", help="Path to queries_train.json")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--endpoints", nargs="+", 
                       default=["search", "search_body", "search_title", "search_anchor", "search_pagerank", "search_pageview"],
                       help="Endpoints to evaluate")
    args = parser.parse_args()
    
    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        print("Please provide the queries_train.json file.")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"Loaded {len(queries)} queries from {queries_path}")
    
    # Evaluate each endpoint
    all_results = {}
    all_predictions = {}
    
    for endpoint in args.endpoints:
        metrics, predictions = evaluate_endpoint(
            args.base_url,
            endpoint,
            queries,
            gold,
            endpoint_name=endpoint,
        )
        all_results[endpoint] = metrics
        all_predictions[endpoint] = predictions
        
        print(f"\n{endpoint} Results:")
        print(f"  Mean time: {metrics['mean_time']:.3f}s")
        print(f"  Max time: {metrics['max_time']:.3f}s")
        print(f"  Min time: {metrics['min_time']:.3f}s")
        print(f"  MAP@10: {metrics['map_at_10']:.4f}")
        print(f"  MAP@5: {metrics['map_at_5']:.4f}")
        print(f"  Harmonic Mean (P@5, F1@30): {metrics['harmonic_mean_p5_f1_30']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "predictions": all_predictions,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Endpoint':<20} {'Mean Time':<12} {'MAP@10':<10} {'Harmonic Mean':<15}")
    print("-"*60)
    for endpoint, metrics in all_results.items():
        print(f"{endpoint:<20} {metrics['mean_time']:>8.3f}s   {metrics['map_at_10']:>8.4f}   {metrics['harmonic_mean_p5_f1_30']:>13.4f}")


if __name__ == "__main__":
    main()


