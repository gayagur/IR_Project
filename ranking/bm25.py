# ranking/bm25.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from inverted_index_gcp import InvertedIndex


class BM25FromIndex:
    def __init__(
        self,
        index: InvertedIndex,
        index_dir: str,
        doc_len: Dict[int, int],
        avgdl: float,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        bucket_name: str | None = None,
    ):
        self.index = index
        self.index_dir = index_dir
        self.doc_len = doc_len
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b
        self.bucket_name = bucket_name

        N = getattr(index, "N", None)
        self.N = N if N is not None else max(1, len(doc_len))

    def _idf(self, df: int) -> float:
        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(
        self, 
        query_tokens: List[str], 
        *, 
        top_n: int = 100, 
        max_terms: int = 50,
        k1: float | None = None,
        b: float | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Search using BM25 scoring.
        
        Args:
            query_tokens: List of query terms
            top_n: Number of top results to return
            max_terms: Maximum number of query terms to use
            k1: Term frequency saturation parameter (default: uses instance k1)
            b: Document length normalization parameter (default: uses instance b)
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if not query_tokens:
            return []

        # Use provided k1 and b, or fall back to instance defaults
        k1_val = k1 if k1 is not None else self.k1
        b_val = b if b is not None else self.b

        q = query_tokens[:max_terms]
        q_terms = list(dict.fromkeys(q))  # unique, keeps order

        scores = defaultdict(float)
        for term in q_terms:
            df = self.index.df.get(term)
            if df is None:
                continue
            idf = self._idf(df)
            pls = self.index.read_a_posting_list(self.index_dir, term, bucket_name=self.bucket_name)

            for doc_id, tf in pls:
                dl = self.doc_len.get(doc_id, 0)
                if dl == 0:
                    continue
                # BM25 formula with custom k1 and b
                denom = tf + k1_val * (1 - b_val + b_val * (dl / self.avgdl))
                score = idf * (tf * (k1_val + 1)) / denom
                scores[doc_id] += score

        res = list(scores.items())
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_n]
