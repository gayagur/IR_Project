# ranking/tfidf_cosine.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from inverted_index_gcp import InvertedIndex


def _idf(N: int, df: int) -> float:
    return math.log((N + 1) / (df + 1))


def search_tfidf_cosine(
    query_tokens: List[str],
    index: InvertedIndex,
    index_dir: str,
    doc_norms: Dict[int, float],
    *,
    top_n: int = 100,
    max_terms: int = 50,
    bucket_name: str | None = None,
) -> List[Tuple[int, float]]:
    """
    Returns top_n (doc_id, score) using TF-IDF dot-product / cosine.
    index_dir is the folder where postings are stored (same dir passed to read_a_posting_list).
    """
    if not query_tokens:
        return []

    # keep some cap to avoid insane queries
    q = query_tokens[:max_terms]

    # query tf-idf
    q_tf = defaultdict(int)
    for t in q:
        q_tf[t] += 1

    N = getattr(index, "N", None)
    if N is None:
        # fallback: approximate by sum of df? better: store N in index, but not always available.
        # We'll approximate by max doc_id count if missing; safest: use len(doc_norms).
        N = max(1, len(doc_norms))

    q_weights: Dict[str, float] = {}
    q_sq = 0.0
    for term, tf in q_tf.items():
        df = index.df.get(term)
        if df is None:
            continue
        w = tf * _idf(N, df)
        q_weights[term] = w
        q_sq += w * w
    q_norm = math.sqrt(q_sq) if q_sq > 0 else 1.0

    scores = defaultdict(float)
    for term, wq in q_weights.items():
        pls = index.read_a_posting_list(index_dir, term, bucket_name=bucket_name)  # [(doc_id, tf), ...]
        df = index.df[term]
        idf = _idf(N, df)
        for doc_id, tf in pls:
            wd = tf * idf
            scores[doc_id] += wd * wq

    # cosine normalize
    results = []
    for doc_id, dot in scores.items():
        dn = doc_norms.get(doc_id, 0.0)
        if dn == 0.0:
            continue
        results.append((doc_id, dot / (dn * q_norm)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
