# ranking/merge.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def merge_rankings(
    ranked_lists: List[Tuple[List[Tuple[int, float]], float]],
    *,
    top_n: int = 100,
) -> List[Tuple[int, float]]:
    """
    ranked_lists: [ ([(doc_id,score),...], weight), ... ]
    Returns merged top_n list by weighted sum.
    """
    scores = defaultdict(float)
    for lst, w in ranked_lists:
        if w == 0 or not lst:
            continue
        for doc_id, s in lst:
            scores[doc_id] += w * s

    res = list(scores.items())
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_n]


def add_feature_boost(
    ranking: List[Tuple[int, float]],
    feature: Dict[int, float],
    *,
    weight: float,
) -> List[Tuple[int, float]]:
    """
    Adds weight*normalized(feature) to scores for docs present in ranking.
    """
    if not ranking or weight == 0:
        return ranking

    vals = [feature.get(d, 0.0) for d, _ in ranking]
    mx = max(vals) if vals else 0.0
    if mx == 0:
        return ranking

    out = []
    for d, s in ranking:
        out.append((d, s + weight * (feature.get(d, 0.0) / mx)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
