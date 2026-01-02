# experiments/run_evaluation.py
"""
Evaluation script for the search engine.
Measures performance metrics and timing for different ranking methods.
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

from experiments.evaluate import (
    average_precision_at_k,
    load_queries_train,
    mean_ap_at_k,
)

BASE_URL = "http://104.198.58.119:8080"

# ============================================================
# BEST WEIGHTS - EDIT HERE
# ============================================================
WEIGHTS = {
    'body': 0.3,
    'title': 1.0,
    'anchor': 1.0,
    'lsi': 0.0,
    'pagerank': 0.2,
    'pageview': 0.10,
}

NUM_QUERIES = 30  # How many queries to test
# ============================================================


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
    # Build URL - handle trailing slashes
    base_url_clean = base_url.rstrip('/')
    endpoint_clean = endpoint.lstrip('/')
    url = f"{base_url_clean}/{endpoint_clean}"
    params = {"query": query}
    
    start_time = time.time()
    try:
        response = requests.get(url, params=params, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"\nError {response.status_code} for query: {query}")
            print(f"URL: {url}")
            print(f"Response: {response.text[:500]}")
            return [], elapsed
        
        results = response.json()
        
        # Handle different response formats
        if not results:
            return [], elapsed
        
        # Extract doc_ids from [[doc_id, title], ...] or [(doc_id, title), ...] format
        doc_ids = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                doc_id = item[0]
                # Convert to int if needed (handle both string and int doc_ids)
                try:
                    doc_ids.append(int(doc_id))
                except (ValueError, TypeError) as e:
                    print(f"\nWarning: Could not convert doc_id '{doc_id}' to int: {e}")
                    continue
            else:
                print(f"\nWarning: Unexpected result format: {item}")
        
        return doc_ids, elapsed
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        print(f"\nRequest exception for query '{query}': {e}")
        print(f"URL: {url}")
        return [], elapsed
    except json.JSONDecodeError as e:
        elapsed = time.time() - start_time
        print(f"\nJSON decode error for query '{query}': {e}")
        print(f"URL: {url}")
        print(f"Response text: {response.text[:500]}")
        return [], elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nUnexpected exception for query '{query}': {e}")
        print(f"URL: {url}")
        import traceback
        traceback.print_exc()
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


# Additional metric functions (with default values for backward compatibility)
# These are used by other scripts that import from run_evaluation
def precision_at_k(pred: List[int], gold: List[int], k: int = 5) -> float:
    """Compute Precision@K (default k=5 for backward compatibility)."""
    if not gold or k == 0:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / k


def recall_at_k(pred: List[int], gold: List[int], k: int = 30) -> float:
    """Compute Recall@K (default k=30 for backward compatibility)."""
    if not gold:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / len(gold_set)


def f1_at_k(pred: List[int], gold: List[int], k: int = 30) -> float:
    """Compute F1@K (default k=30 for backward compatibility)."""
    p = precision_at_k(pred, gold, k)
    r = recall_at_k(pred, gold, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def query_with_weights(query: str, weights: Dict[str, float]) -> Tuple[List[int], float]:
    """Query the search engine with custom weights."""
    
    params = {
        'query': query,
        'body_weight': weights.get('body', 1.0),
        'title_weight': weights.get('title', 0.35),
        'anchor_weight': weights.get('anchor', 0.25),
        'lsi_weight': weights.get('lsi', 0.0),
        'pagerank_boost': weights.get('pagerank', 0.15),
        'pageview_boost': weights.get('pageview', 0.10),
    }
    
    url = f"{BASE_URL}/search_with_weights"
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


def main():
    # Load queries - use config for path resolution
    import config
    queries_path = config.QUERIES_DIR / "test_queries.json"
    
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"Loaded {len(queries)} queries from {queries_path}")
    
    # Test server
    print(f"Base URL: {BASE_URL}")
    try:
        requests.get(BASE_URL, timeout=10)
        print("Server reachable")
    except:
        print("Cannot connect to server")
        return
    
    # Use configured weights and number of queries
    test_queries = queries[:NUM_QUERIES]
    
    print(f"\nEvaluating on {len(test_queries)} queries:")
    print(f"  body={WEIGHTS['body']}, title={WEIGHTS['title']}, anchor={WEIGHTS['anchor']}")
    print(f"  lsi={WEIGHTS['lsi']}, pagerank={WEIGHTS['pagerank']}, pageview={WEIGHTS['pageview']}")
    print("-" * 60)
    
    all_pred = {}
    times = []
    
    for i, query in enumerate(test_queries, 1):
        doc_ids, elapsed = query_with_weights(query, WEIGHTS)
        all_pred[query] = doc_ids
        times.append(elapsed)
        print(f"  [{i}/{len(test_queries)}] {elapsed:.2f}s - {query[:50]}", end='\r')
    
    print()
    
    # Calculate all metrics
    map_at_10 = mean_ap_at_k(all_pred, gold, k=10)
    map_at_5 = mean_ap_at_k(all_pred, gold, k=5)
    
    precisions_5 = []
    recalls_30 = []
    f1_scores_30 = []
    harmonic_means = []
    
    for q in test_queries:
        pred = all_pred.get(q, [])
        gold_list = gold.get(q, [])
        
        p5 = precision_at_k(pred, gold_list, k=5)
        r30 = recall_at_k(pred, gold_list, k=30)
        f1_30 = f1_at_k(pred, gold_list, k=30)
        hm = harmonic_mean(p5, f1_30)
        
        precisions_5.append(p5)
        recalls_30.append(r30)
        f1_scores_30.append(f1_30)
        harmonic_means.append(hm)
    
    avg_time = sum(times) / len(times) if times else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nWeights:")
    print(f"  body:     {WEIGHTS['body']:.2f}")
    print(f"  title:    {WEIGHTS['title']:.2f}")
    print(f"  anchor:   {WEIGHTS['anchor']:.2f}")
    print(f"  lsi:      {WEIGHTS['lsi']:.2f}")
    print(f"  pagerank: {WEIGHTS['pagerank']:.2f}")
    print(f"  pageview: {WEIGHTS['pageview']:.2f}")
    
    print(f"\nQueries tested: {len(test_queries)}")
    
    print("\nMetrics:")
    print(f"  MAP@10:        {map_at_10:.4f}")
    print(f"  MAP@5:         {map_at_5:.4f}")
    print(f"  Precision@5:   {sum(precisions_5)/len(precisions_5):.4f}")
    print(f"  Recall@30:     {sum(recalls_30)/len(recalls_30):.4f}")
    print(f"  F1@30:         {sum(f1_scores_30)/len(f1_scores_30):.4f}")
    print(f"  Harmonic Mean: {sum(harmonic_means)/len(harmonic_means):.4f}")
    
    print("\nTiming:")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Min:     {min(times):.2f}s")
    print(f"  Max:     {max(times):.2f}s")
    
    print("=" * 60)


if __name__ == "__main__":
    main()