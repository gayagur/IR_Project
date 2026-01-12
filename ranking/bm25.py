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
        q_tokens: List[str],
        *,
        top_n: int = 100,
        k1: float = 2.5,
        b: float = 0.0,
    ) -> List[Tuple[int, float]]:
        from concurrent.futures import ThreadPoolExecutor
        
        scores: Dict[int, float] = {}
        unique_terms = [t for t in set(q_tokens) if t in self.index.df]
        
        if not unique_terms:
            return []
        
        def process_term(term):
            df = self.index.df[term]
            idf = self._idf(df)
            posting_list = self.index.read_a_posting_list(
                self.index_dir, term, bucket_name=self.bucket_name
            )
            return idf, posting_list
        
        # Read posting lists in parallel
        with ThreadPoolExecutor(max_workers=min(len(unique_terms), 4)) as executor:
            results = list(executor.map(process_term, unique_terms))
        
        # Score documents
        for idf, posting_list in results:
            for doc_id, tf in posting_list:
                doc_id = int(doc_id)
                dl = self.doc_len.get(doc_id, self.avgdl)
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
                score = idf * tf_component
                scores[doc_id] = scores.get(doc_id, 0.0) + score
        
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_n] if top_n else ranked