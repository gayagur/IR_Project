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
    body_bm25: Optional[BM25FromIndex]
    body_index_dir: str
    title_index_dir: str
    anchor_index_dir: str
    bucket_name: Optional[str] = None

    def tokenize_query(self, query: str) -> List[str]:
        return tokenize(query)

    def search_body_bm25(self, q_tokens: List[str], *, top_n: int = 100) -> List[Tuple[int, float]]:
        if self.body_bm25 is None:
            print("[WARNING] body_bm25 is None, returning empty results")
            return []
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
        """Rank docs by number of DISTINCT query tokens that appear in the doc (title/anchor).
        
        Uses parallel posting list reads for faster performance.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        scores: Dict[int, int] = {}
        seen_terms = set()
        
        # Collect valid terms
        valid_terms = []
        for t in q_tokens:
            if t in seen_terms or t not in index.df:
                continue
            seen_terms.add(t)
            valid_terms.append(t)
        
        if not valid_terms:
            return []
        
        # Read posting lists in parallel
        def read_posting_list(term):
            """Read a single posting list."""
            try:
                return term, index.read_a_posting_list(index_dir, term, bucket_name=bucket_name)
            except Exception as e:
                print(f"  ⚠ Error reading posting list for '{term}': {e}")
                return term, []
        
        # Read all posting lists in parallel
        with ThreadPoolExecutor(max_workers=min(10, len(valid_terms))) as executor:
            future_to_term = {
                executor.submit(read_posting_list, term): term
                for term in valid_terms
            }
            
            for future in as_completed(future_to_term):
                term, pl = future.result()
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

        # Debug: check if pagerank has any values
        if not self.pagerank:
            print(f"[MERGE_SIGNALS WARNING] pagerank dictionary is empty!")
        else:
            # Check if any candidate IDs exist in pagerank
            found_in_pr = sum(1 for d in cand_ids[:10] if d in self.pagerank)
            if found_in_pr == 0 and len(cand_ids) > 0:
                print(f"[MERGE_SIGNALS WARNING] None of first 10 candidate IDs found in pagerank!")
                print(f"  Sample candidate IDs: {cand_ids[:5]}")
                print(f"  Sample pagerank keys: {list(self.pagerank.keys())[:5] if self.pagerank else []}")
        
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


def _check_if_indices_exist_locally() -> bool:
    """Check if indices exist on local disk."""
    from pathlib import Path
    import config
    
    # Check main index files
    body_pkl = config.INDICES_DIR / "body" / "body.pkl"
    title_pkl = config.INDICES_DIR / "title" / "title.pkl"
    anchor_pkl = config.INDICES_DIR / "anchor" / "anchor.pkl"
    
    if not (body_pkl.exists() and title_pkl.exists() and anchor_pkl.exists()):
        return False
    
    # Check at least one aux file exists
    aux_dir = config.AUX_DIR
    if not aux_dir.exists():
        return False
    
    # Check if at least titles.pkl exists
    if not (aux_dir / "titles.pkl").exists():
        return False
    
    return True


def _download_indices_from_gcs_to_local(bucket_name: str) -> bool:
    """Download all indices and auxiliary files from GCS to local disk.
    
    Uses parallel downloads to speed up the process.
    
    Returns True if download was successful, False otherwise.
    """
    from pathlib import Path
    from google.cloud import storage
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import config
    
    print("=" * 60)
    print("Downloading indices from GCS to local disk (parallel)...")
    print("=" * 60)
    
    try:
        client = storage.Client(project=config.PROJECT_ID)
        bucket = client.bucket(bucket_name)
        
        # Ensure local directories exist
        config.INDICES_DIR.mkdir(parents=True, exist_ok=True)
        config.AUX_DIR.mkdir(parents=True, exist_ok=True)
        
        (config.INDICES_DIR / "body").mkdir(parents=True, exist_ok=True)
        (config.INDICES_DIR / "title").mkdir(parents=True, exist_ok=True)
        (config.INDICES_DIR / "anchor").mkdir(parents=True, exist_ok=True)
        
        def download_file(blob, local_path):
            """Download a single file from GCS."""
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(local_path))
                return True
            except Exception as e:
                print(f"  ✗ Error downloading {blob.name}: {e}")
                return False
        
        # Download indices in parallel
        indices_to_download = [
            ("indices/body", config.INDICES_DIR / "body"),
            ("indices/title", config.INDICES_DIR / "title"),
            ("indices/anchor", config.INDICES_DIR / "anchor"),
        ]
        
        total_files = 0
        # Use ThreadPoolExecutor for parallel downloads
        # Adjust max_workers based on your needs (more = faster but more memory/network)
        max_workers = 10  # Download 10 files in parallel
        
        for gcs_prefix, local_dir in indices_to_download:
            print(f"Downloading {gcs_prefix} (parallel, {max_workers} workers)...")
            
            # Collect all files to download
            files_to_download = []
            for blob in bucket.list_blobs(prefix=gcs_prefix + "/"):
                if blob.name.endswith("/"):
                    continue
                
                relative_path = blob.name[len(gcs_prefix) + 1:]
                local_path = local_dir / relative_path
                files_to_download.append((blob, local_path))
            
            # Download files in parallel
            files_downloaded = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                future_to_file = {
                    executor.submit(download_file, blob, local_path): (blob.name, local_path)
                    for blob, local_path in files_to_download
                }
                
                # Process completed downloads
                for future in as_completed(future_to_file):
                    blob_name, local_path = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            files_downloaded += 1
                            total_files += 1
                            
                            if files_downloaded % 100 == 0:
                                print(f"  Downloaded {files_downloaded}/{len(files_to_download)} files...")
                    except Exception as e:
                        print(f"  ✗ Error downloading {blob_name}: {e}")
            
            print(f"  ✓ {gcs_prefix}: {files_downloaded}/{len(files_to_download)} files")
        
        # Download auxiliary files (small files, can do sequentially or in parallel)
        print("Downloading auxiliary files...")
        aux_files = [
            "titles.pkl",
            "doc_norms.pkl",
            "doc_len.pkl",
            "avgdl.txt",
            "pagerank.pkl",
            "pageviews.pkl",
        ]
        
        aux_downloaded = 0
        for filename in aux_files:
            blob_path = f"aux/{filename}"
            blob = bucket.blob(blob_path)
            if blob.exists():
                local_path = config.AUX_DIR / filename
                blob.download_to_filename(str(local_path))
                aux_downloaded += 1
                print(f"  ✓ {filename}")
            else:
                print(f"  ⚠ {filename} not found in GCS")
        
        print("=" * 60)
        print(f"✓ Download completed: {total_files} index files, {aux_downloaded} aux files")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading from GCS: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_engine() -> SearchEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    print("=" * 60)
    print("Loading search engine...")
    print("=" * 60)

    # Determine if we should read from GCS or local filesystem
    read_from_gcs = getattr(config, 'READ_FROM_GCS', False)
    
    # If READ_FROM_GCS = True, always read from GCS (even if local files exist)
    if read_from_gcs:
        print("READ_FROM_GCS = True - reading directly from GCS")
        bucket_name = config.BUCKET_NAME
    else:
        # READ_FROM_GCS = False - prefer local filesystem if available
        if _check_if_indices_exist_locally():
            print("Indices found locally - using local filesystem (faster)")
            read_from_gcs = False
            bucket_name = None
        else:
            # Indices not found locally - fall back to GCS even if READ_FROM_GCS=False
            print("⚠ Warning: Indices not found locally")
            if config.BUCKET_NAME:
                print("Falling back to reading from GCS (slower)")
                bucket_name = config.BUCKET_NAME
                read_from_gcs = True  # Temporarily enable GCS reading as fallback
            else:
                print("⚠ Error: No local indices and BUCKET_NAME not set")
                bucket_name = None
    
    print(f"READ_FROM_GCS: {read_from_gcs}, BUCKET_NAME: {bucket_name}")

    body_dir = str(config.BODY_INDEX_DIR)
    title_dir = str(config.TITLE_INDEX_DIR)
    anchor_dir = str(config.ANCHOR_INDEX_DIR)

    # Load indices in parallel for faster startup
    print("Loading indices in parallel...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def load_index(index_name, index_dir, name):
        """Load a single index."""
        try:
            index = InvertedIndex.read_index(index_dir, name, bucket_name=bucket_name)
            return index_name, index, None
        except Exception as e:
            return index_name, InvertedIndex(), str(e)
    
    def load_aux_file(file_name, load_func, *args):
        """Load a single auxiliary file."""
        try:
            result = load_func(*args)
            return file_name, result, None
        except Exception as e:
            default = {} if 'pickle' in str(load_func) else 0.0
            return file_name, default, str(e)
    
    # Load all indices in parallel
    indices = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(load_index, "body", body_dir, "body"): "body",
            executor.submit(load_index, "title", title_dir, "title"): "title",
            executor.submit(load_index, "anchor", anchor_dir, "anchor"): "anchor",
        }
        
        for future in as_completed(future_to_name):
            name, index, error = future.result()
            indices[name] = index
            if error:
                print(f"  ✗ Failed to load {name} index: {error}")
            else:
                print(f"  ✓ {name.capitalize()} index loaded: {len(index.df):,} terms")
    
    body = indices.get("body", InvertedIndex())
    title = indices.get("title", InvertedIndex())
    anchor = indices.get("anchor", InvertedIndex())

    # Load auxiliary files in parallel
    print("Loading auxiliary files in parallel...")
    aux_files = {}
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_name = {
            executor.submit(load_aux_file, "titles", _load_pickle, config.TITLES_PATH, {}, bucket_name): "titles",
            executor.submit(load_aux_file, "doc_norms", _load_pickle, config.DOC_NORMS_PATH, {}, bucket_name): "doc_norms",
            executor.submit(load_aux_file, "doc_len", _load_pickle, config.DOC_LEN_PATH, {}, bucket_name): "doc_len",
            executor.submit(load_aux_file, "avgdl", _load_float_text, config.AVGDL_PATH, 0.0, bucket_name): "avgdl",
            executor.submit(load_aux_file, "pagerank", _load_pickle, config.PAGERANK_PATH, {}, bucket_name): "pagerank",
            executor.submit(load_aux_file, "pageviews", _load_pickle, config.PAGEVIEWS_PATH, {}, bucket_name): "pageviews",
        }
        
        for future in as_completed(future_to_name):
            name, result, error = future.result()
            aux_files[name] = result
            if error:
                print(f"    ✗ Failed to load {name}: {error}")
            else:
                if isinstance(result, dict):
                    print(f"    ✓ {name.capitalize()} loaded: {len(result):,} entries")
                    # Special check for pagerank
                    if name == "pagerank" and result:
                        sample_ids = list(result.keys())[:3]
                        sample_prs = [result[id] for id in sample_ids]
                        max_pr = max(result.values()) if result else 0.0
                        min_pr = min(result.values()) if result else 0.0
                        print(f"      Sample: {list(zip(sample_ids, sample_prs))}")
                        print(f"      Range: [{min_pr:.6f}, {max_pr:.6f}]")
                        non_zero = sum(1 for v in result.values() if v > 0)
                        print(f"      Non-zero entries: {non_zero:,}/{len(result):,} ({100*non_zero/len(result):.1f}%)")
                else:
                    print(f"    ✓ {name.capitalize()} loaded: {result}")
    
    titles = aux_files.get("titles", {})
    doc_norms = aux_files.get("doc_norms", {})
    doc_len = aux_files.get("doc_len", {})
    avgdl = aux_files.get("avgdl", 0.0)
    pagerank = aux_files.get("pagerank", {})
    pageviews = aux_files.get("pageviews", {})

    print("Initializing BM25...")
    try:
        if len(body.df) > 0 and len(doc_len) > 0:
            body_bm25 = BM25FromIndex(body, body_dir, doc_len, avgdl, bucket_name=bucket_name)
            print("  ✓ BM25 initialized")
        else:
            print("  ⚠ Body index or doc_len is empty, BM25 will be None")
            body_bm25 = None
    except Exception as e:
        print(f"  ✗ Failed to initialize BM25: {e}")
        import traceback
        traceback.print_exc()
        body_bm25 = None

    print("Creating SearchEngine object...")
    try:
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
        print("=" * 60)
        print("✓ Search engine loaded successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Failed to create SearchEngine: {e}")
        import traceback
        traceback.print_exc()
        raise

    return _ENGINE
