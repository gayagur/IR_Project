# experiments/evaluate.py
from __future__ import annotations

import json
from typing import Dict, List, Tuple


def precision_at_k(pred: List[int], gold: List[int], k: int) -> float:
    """
    Precision@K for a single query.
    
    Formula: (number of relevant docs in top K) / K
    """
    if k <= 0:
        return 0.0
    pred_at_k = pred[:k]
    gold_set = set(gold)
    relevant_in_top_k = sum(1 for doc in pred_at_k if doc in gold_set)
    return relevant_in_top_k / k


def average_precision_at_k(
    all_predictions: Dict[str, List[int]], 
    gold: Dict[str, List[int]], 
    k: int
) -> float:
    """
    Average Precision@K across all queries.
    
    For each query: compute Precision@K = (relevant in top K) / K
    Then average across all queries.
    """
    precisions = []
    for query, pred in all_predictions.items():
        gold_list = gold.get(query, [])
        p_at_k = precision_at_k(pred, gold_list, k)
        precisions.append(p_at_k)
    
    return sum(precisions) / len(precisions) if precisions else 0.0


# Backward compatibility alias
def mean_ap_at_k(all_pred: Dict[str, List[int]], all_gold: Dict[str, List[int]], k: int = 10) -> float:
    """
    Alias for average_precision_at_k (for backward compatibility).
    Note: This is NOT Mean Average Precision (position-aware), but Average Precision@K.
    """
    return average_precision_at_k(all_pred, all_gold, k)


def load_queries_train(path: str) -> Tuple[List[str], Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support two formats:
    # 1. List format: [{"query": "...", "relevant_docs": [..]}, ...]
    # 2. Dict format: {"query": [doc_ids], ...}
    if isinstance(data, list):
        # List format
        queries = [x["query"] for x in data]
        gold = {x["query"]: x["relevant_docs"] for x in data}
    elif isinstance(data, dict):
        # Dict format (like test_queries.json)
        queries = list(data.keys())
        gold = {query: [int(doc_id) for doc_id in doc_ids] for query, doc_ids in data.items()}
    else:
        raise ValueError(f"Unexpected format in {path}: expected list or dict")
    
    return queries, gold
