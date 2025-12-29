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
        # common BM25 idf
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: List[str], *, top_n: int = 100, max_terms: int = 50) -> List[Tuple[int, float]]:
        if not query_tokens:
            return []

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
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] += score

        res = list(scores.items())
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_n]
