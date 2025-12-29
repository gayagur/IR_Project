# search_runtime.py
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
from inverted_index_gcp import InvertedIndex
from text_processing import tokenize

from ranking.bm25 import BM25FromIndex
from ranking.tfidf_cosine import search_tfidf_cosine
from ranking.merge import merge_rankings


def _load_pickle(path: str | Path, default, bucket_name=None):
    """Load pickle file from local filesystem or GCS."""
    p = Path(path)
    
    # Try GCS first if bucket_name is provided
    if bucket_name:
        try:
            from google.cloud import storage
            from inverted_index_gcp import get_bucket
            bucket = get_bucket(bucket_name)
            # Convert local path to GCS path (e.g., aux/titles.pkl -> aux/titles.pkl)
            gcs_path = str(p)
            if p.is_absolute():
                # Try to extract relative path - for aux files
                if 'aux' in str(p):
                    gcs_path = f"aux/{p.name}"
                elif 'indices' in str(p):
                    # This shouldn't happen for pickle files, but handle it
                    parts = p.parts
                    if 'indices' in parts:
                        idx = parts.index('indices')
                        gcs_path = '/'.join(parts[idx:])
            
            blob = bucket.blob(gcs_path)
            if blob.exists():
                with blob.open("rb") as f:
                    return pickle.load(f)
        except Exception as e:
            # Fall back to local if GCS fails
            pass
    
    # Try local filesystem
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    
    return default


def _load_float_text(path: str | Path, default: float, bucket_name=None) -> float:
    """Load text file from local filesystem or GCS."""
    p = Path(path)
    
    # Try GCS first if bucket_name is provided
    if bucket_name:
        try:
            from google.cloud import storage
            from inverted_index_gcp import get_bucket
            bucket = get_bucket(bucket_name)
            # Convert local path to GCS path
            gcs_path = str(p)
            if p.is_absolute():
                if 'aux' in str(p):
                    gcs_path = f"aux/{p.name}"
            
            blob = bucket.blob(gcs_path)
            if blob.exists():
                content = blob.download_as_text(encoding="utf-8").strip()
                return float(content)
        except Exception as e:
            # Fall back to local if GCS fails
            pass
    
    # Try local filesystem
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            return default
    
    return default


@dataclass
class SearchEngine:
    body_index: InvertedIndex
    title_index: InvertedIndex
    anchor_index: InvertedIndex
    titles: Dict[int, str]
    doc_norms: Dict[int, float]
    doc_len: Dict[int, int]
    avgdl: float
    pagerank: Dict[int, float]
    pageviews: Dict[int, int]
    body_bm25: BM25FromIndex
    body_index_dir: str
    title_index_dir: str
    anchor_index_dir: str
    bucket_name: Optional[str] = None

    def tokenize_query(self, query: str) -> List[str]:
        return tokenize(query)

    def search_body_bm25(self, q_tokens: List[str], *, top_n: int = 100) -> List[Tuple[int, float]]:
        return self.body_bm25.search(q_tokens, top_n=top_n)

    def search_body_tfidf_cosine(self, q_tokens: List[str], *, top_n: int = 100) -> List[Tuple[int, float]]:
        return search_tfidf_cosine(
            q_tokens,
            self.body_index,
            self.body_index_dir,
            self.doc_norms,
            top_n=top_n,
            bucket_name=self.bucket_name,
        )

    def _count_index_matches(
        self,
        q_tokens: List[str],
        *,
        index: InvertedIndex,
        index_dir: str,
        top_n: Optional[int],
        bucket_name: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """Rank docs by number of DISTINCT query tokens that appear in the doc (title/anchor)."""
        scores: Dict[int, int] = {}
        seen_terms = set()

        for t in q_tokens:
            if t in seen_terms or t not in index.df:
                continue
            seen_terms.add(t)
            pl = index.read_a_posting_list(index_dir, t, bucket_name=bucket_name)
            for doc_id, _tf in pl:
                doc_id = int(doc_id)
                scores[doc_id] = scores.get(doc_id, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if top_n is None:
            return [(d, float(s)) for d, s in ranked]
        return [(d, float(s)) for d, s in ranked[:top_n]]

    def search_title_count(self, q_tokens: List[str], *, top_n: Optional[int]) -> List[Tuple[int, float]]:
        return self._count_index_matches(q_tokens, index=self.title_index, index_dir=self.title_index_dir, top_n=top_n, bucket_name=self.bucket_name)

    def search_anchor_count(self, q_tokens: List[str], *, top_n: Optional[int]) -> List[Tuple[int, float]]:
        return self._count_index_matches(q_tokens, index=self.anchor_index, index_dir=self.anchor_index_dir, top_n=top_n, bucket_name=self.bucket_name)

    def merge_signals(
        self,
        *,
        body_ranked: List[Tuple[int, float]],
        title_ranked: List[Tuple[int, float]],
        anchor_ranked: List[Tuple[int, float]],
        top_n: int = 100,
    ) -> List[Tuple[int, float]]:
        """Weighted merge + light PageRank/PageViews boosting over the merged candidate set."""
        merged = merge_rankings(
            [
                (body_ranked, 1.0),
                (title_ranked, 0.35),
                (anchor_ranked, 0.25),
            ],
            top_n=max(500, top_n),
        )

        # Add PR/PV boosts for the current candidate set only (avoid normalizing over all docs).
        cand_ids = [doc_id for doc_id, _ in merged]
        if not cand_ids:
            return []

        pr_vals = [float(self.pagerank.get(d, 0.0)) for d in cand_ids]
        pv_vals = [float(self.pageviews.get(d, 0)) for d in cand_ids]

        pr_max = max(pr_vals) if pr_vals else 0.0
        pv_max = max(pv_vals) if pv_vals else 0.0

        rescored: List[Tuple[int, float]] = []
        for (doc_id, base), pr, pv in zip(merged, pr_vals, pv_vals):
            pr_norm = (pr / pr_max) if pr_max > 0 else 0.0
            pv_norm = (pv / pv_max) if pv_max > 0 else 0.0
            score = base + 0.15 * pr_norm + 0.10 * pv_norm
            rescored.append((doc_id, score))

        rescored.sort(key=lambda x: (-x[1], x[0]))
        return rescored[:top_n]


_ENGINE: Optional[SearchEngine] = None


def get_engine() -> SearchEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    # NOTE: We can read from GCS if indices are stored there.
    # Set config.READ_FROM_GCS=True to read all files (indices + aux) from GCS.
    # Otherwise, reads from local filesystem.
    read_from_gcs = getattr(config, 'READ_FROM_GCS', False)
    bucket_name = config.BUCKET_NAME if read_from_gcs else None

    body_dir = str(config.BODY_INDEX_DIR)
    title_dir = str(config.TITLE_INDEX_DIR)
    anchor_dir = str(config.ANCHOR_INDEX_DIR)

    body = InvertedIndex.read_index(body_dir, "body", bucket_name=bucket_name)
    title = InvertedIndex.read_index(title_dir, "title", bucket_name=bucket_name)
    anchor = InvertedIndex.read_index(anchor_dir, "anchor", bucket_name=bucket_name)

    titles = _load_pickle(config.TITLES_PATH, {}, bucket_name=bucket_name)
    doc_norms = _load_pickle(config.DOC_NORMS_PATH, {}, bucket_name=bucket_name)
    doc_len = _load_pickle(config.DOC_LEN_PATH, {}, bucket_name=bucket_name)
    avgdl = _load_float_text(config.AVGDL_PATH, default=0.0, bucket_name=bucket_name)

    pagerank = _load_pickle(config.PAGERANK_PATH, {}, bucket_name=bucket_name)
    pageviews = _load_pickle(config.PAGEVIEWS_PATH, {}, bucket_name=bucket_name)

    body_bm25 = BM25FromIndex(body, body_dir, doc_len, avgdl, bucket_name=bucket_name)

    _ENGINE = SearchEngine(
        body_index=body,
        title_index=title,
        anchor_index=anchor,
        titles=titles,
        doc_norms=doc_norms,
        doc_len=doc_len,
        avgdl=avgdl,
        pagerank=pagerank,
        pageviews=pageviews,
        body_bm25=body_bm25,
        body_index_dir=body_dir,
        title_index_dir=title_dir,
        anchor_index_dir=anchor_dir,
        bucket_name=bucket_name,
    )
    return _ENGINE
