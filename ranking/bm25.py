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
        k1: float = 2.5,
        b: float = 0.0,
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
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if not query_tokens:
            return []

        k1_val = k1 if k1 is not None else self.k1
        b_val = b if b is not None else self.b

        q = query_tokens[:max_terms]
        q_terms = list(dict.fromkeys(q))  # unique, keeps order

        def process_term(term):
            df = self.index.df.get(term)
            if df is None:
                return None
            idf = self._idf(df)
            pls = self.index.read_a_posting_list(self.index_dir, term, bucket_name=self.bucket_name)
            # Limit very large posting lists
            if len(pls) > 200000:
                pls = pls[:200000]
            return idf, pls

        # Read posting lists in parallel
        with ThreadPoolExecutor(max_workers=min(len(q_terms), 4)) as executor:
            results = list(executor.map(process_term, q_terms))

        scores = defaultdict(float)
        for result in results:
            if result is None:
                continue
            idf, pls = result
            for doc_id, tf in pls:
                dl = self.doc_len.get(doc_id, 0)
                if dl == 0:
                    continue
                denom = tf + k1_val * (1 - b_val + b_val * (dl / self.avgdl))
                score = idf * (tf * (k1_val + 1)) / denom
                scores[doc_id] += score

        res = list(scores.items())
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_n]
