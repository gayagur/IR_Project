# experiments/evaluate.py
from __future__ import annotations

import json
import argparse
import requests
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

BASE_URL = "http://104.198.58.119:8080"


def average_precision_at_k(pred: List[int], gold: List[int], k: int = 10) -> float:
    """
    gold: ranked list of relevant docs (as provided), we treat all as relevant set.
    """
    if not gold:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(pred_k, start=1):
        if doc_id in gold_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(gold_set), k)


def mean_ap_at_k(all_pred: Dict[str, List[int]], all_gold: Dict[str, List[int]], k: int = 10) -> float:
    aps = []
    for q, pred in all_pred.items():
        gold = all_gold.get(q, [])
        aps.append(average_precision_at_k(pred, gold, k=k))
    return sum(aps) / max(1, len(aps))


def precision_at_k(pred: List[int], gold: List[int], k: int = 5) -> float:
    if not gold:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / k


def recall_at_k(pred: List[int], gold: List[int], k: int = 30) -> float:
    if not gold:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / len(gold_set)


def f1_at_k(pred: List[int], gold: List[int], k: int = 30) -> float:
    p = precision_at_k(pred, gold, k)
    r = recall_at_k(pred, gold, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def load_queries_train(path: str) -> Tuple[List[str], Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support two formats:
    # 1. List format: [{"query": "...", "relevant_docs": [..]}, ...]
    # 2. Dict format: {"query": [doc_ids], ...}
    if isinstance(data, list):
        queries = [x["query"] for x in data]
        gold = {x["query"]: x["relevant_docs"] for x in data}
    elif isinstance(data, dict):
        queries = list(data.keys())
        gold = {query: [int(doc_id) for doc_id in doc_ids] for query, doc_ids in data.items()}
    else:
        raise ValueError(f"Unexpected format in {path}: expected list or dict")
    
    return queries, gold


def query_with_weights(query: str, weights: Dict[str, float]) -> Tuple[List[int], float]:
    """Query the search engine with custom weights."""
    import time
    
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


def evaluate_weights(weights: Dict[str, float], queries: List[str], gold: Dict[str, List[int]], num_queries: int = 20) -> Dict:
    """Evaluate weights on a subset of queries and return statistics."""
    
    # Use only first num_queries
    test_queries = queries[:num_queries]
    
    all_pred = {}
    times = []
    
    print(f"\nEvaluating weights on {len(test_queries)} queries:")
    print(f"  body={weights.get('body', 1.0)}, title={weights.get('title', 0.35)}, "
          f"anchor={weights.get('anchor', 0.25)}, lsi={weights.get('lsi', 0.0)}, "
          f"pr={weights.get('pagerank', 0.15)}, pv={weights.get('pageview', 0.1)}")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        doc_ids, elapsed = query_with_weights(query, weights)
        all_pred[query] = doc_ids
        times.append(elapsed)
        print(f"  [{i}/{len(test_queries)}] {elapsed:.2f}s - {query[:40]}...", end='\r')
    
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
    
    results = {
        'weights': weights,
        'num_queries': len(test_queries),
        'map_at_10': map_at_10,
        'map_at_5': map_at_5,
        'precision_at_5': sum(precisions_5) / len(precisions_5),
        'recall_at_30': sum(recalls_30) / len(recalls_30),
        'f1_at_30': sum(f1_scores_30) / len(f1_scores_30),
        'harmonic_mean': sum(harmonic_means) / len(harmonic_means),
        'avg_time': avg_time,
        'min_time': min(times),
        'max_time': max(times),
    }
    
    return results


def print_results(results: Dict):
    """Print results in a nice format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nWeights:")
    w = results['weights']
    print(f"  body:     {w.get('body', 1.0):.2f}")
    print(f"  title:    {w.get('title', 0.35):.2f}")
    print(f"  anchor:   {w.get('anchor', 0.25):.2f}")
    print(f"  lsi:      {w.get('lsi', 0.0):.2f}")
    print(f"  pagerank: {w.get('pagerank', 0.15):.2f}")
    print(f"  pageview: {w.get('pageview', 0.1):.2f}")
    
    print(f"\nQueries tested: {results['num_queries']}")
    
    print("\nMetrics:")
    print(f"  MAP@10:        {results['map_at_10']:.4f}")
    print(f"  MAP@5:         {results['map_at_5']:.4f}")
    print(f"  Precision@5:   {results['precision_at_5']:.4f}")
    print(f"  Recall@30:     {results['recall_at_30']:.4f}")
    print(f"  F1@30:         {results['f1_at_30']:.4f}")
    print(f"  Harmonic Mean: {results['harmonic_mean']:.4f}")
    
    print("\nTiming:")
    print(f"  Average: {results['avg_time']:.2f}s")
    print(f"  Min:     {results['min_time']:.2f}s")
    print(f"  Max:     {results['max_time']:.2f}s")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate search engine with custom weights")
    parser.add_argument("--body", type=float, default=1.0, help="Body weight")
    parser.add_argument("--title", type=float, default=0.35, help="Title weight")
    parser.add_argument("--anchor", type=float, default=0.25, help="Anchor weight")
    parser.add_argument("--lsi", type=float, default=0.0, help="LSI weight")
    parser.add_argument("--pagerank", type=float, default=0.15, help="PageRank boost")
    parser.add_argument("--pageview", type=float, default=0.10, help="PageView boost")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to test")
    parser.add_argument("--queries-file", default="test_queries.json", help="Path to queries file")
    args = parser.parse_args()
    
    # Build weights dict
    weights = {
        'body': args.body,
        'title': args.title,
        'anchor': args.anchor,
        'lsi': args.lsi,
        'pagerank': args.pagerank,
        'pageview': args.pageview,
    }
    
    # Load queries
    queries_path = Path(args.queries_file)
    if not queries_path.is_absolute() and not queries_path.exists():
        queries_path = parent_dir / args.queries_file
    
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
    
    # Evaluate
    results = evaluate_weights(weights, queries, gold, num_queries=args.queries)
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()


    # Test on 30 queries
# python experiments/evaluate.py --body 0.3 --title 1.5 --anchor 0.75 --lsi 0.0 --pagerank 0.15 --pageview 0.1 --queries 30
