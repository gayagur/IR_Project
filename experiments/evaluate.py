# experiments/evaluate.py
from __future__ import annotations

import json
from typing import Dict, List, Tuple


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


def load_queries_train(path: str) -> Tuple[List[str], Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # expected: list of {"query": "...", "relevant_docs": [..]}
    queries = [x["query"] for x in data]
    gold = {x["query"]: x["relevant_docs"] for x in data}
    return queries, gold
