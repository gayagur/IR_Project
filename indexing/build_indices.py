# indexing/build_indices.py
from __future__ import annotations

import bz2
import math
import os
import pickle
import re
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

import config
from inverted_index_gcp import InvertedIndex
from parser_utils import get_wikilinks, page_iter, page_iter_parquet, page_iter_parquet_with_anchors, remove_markdown
from text_processing import tokenize

NUM_BUCKETS = 124

# Temporary local directory for partial anchor indices (always local)
ANCHOR_PARTS_DIR = Path(tempfile.gettempdir()) / "ir_project_anchor_parts"


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _ensure_local_dirs() -> None:
    """Create required local directories when writing to local FS."""
    if config.WRITE_TO_GCS:
        return
    config.DATA_DIR.mkdir(exist_ok=True)
    config.INDICES_DIR.mkdir(parents=True, exist_ok=True)
    config.AUX_DIR.mkdir(parents=True, exist_ok=True)
    _as_path(config.BODY_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    _as_path(config.TITLE_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    _as_path(config.ANCHOR_INDEX_DIR).mkdir(parents=True, exist_ok=True)


def _tokenize(text: str) -> List[str]:
    return tokenize(text)


def _save_pickle(obj, path: str | Path) -> None:
    path_p = _as_path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)
    with open(path_p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _write_index(index: InvertedIndex, base_dir: str | Path, name: str) -> None:
    """Write the index postings and globals to either local FS or GCS."""
    base_dir_str = str(base_dir)
    bucket_name = config.BUCKET_NAME if config.WRITE_TO_GCS else None

    # Group posting lists into deterministic buckets
    bucket_to_w_pl: DefaultDict[int, List[Tuple[str, List[Tuple[int, int]]]]] = defaultdict(list)
    for term, pl in index._posting_list.items():
        pl = sorted(pl, key=lambda x: x[0])
        b = hash(term) % NUM_BUCKETS
        bucket_to_w_pl[b].append((term, pl))

    index.posting_locs = defaultdict(list)
    for b, w_pl in bucket_to_w_pl.items():
        w_pl.sort(key=lambda x: x[0])
        bucket_id = InvertedIndex.write_a_posting_list((b, w_pl), base_dir_str, bucket_name=bucket_name)

        # Merge posting_locs back (writer stores it in a side pickle)
        locs_path = Path(base_dir_str) / f"{bucket_id}_posting_locs.pickle"
        with open(locs_path, "rb") as f:
            posting_locs_bucket = pickle.load(f)
        for term, locs in posting_locs_bucket.items():
            index.posting_locs[term].extend(locs)

    index.write_index(base_dir_str, name, bucket_name=bucket_name)


def build_title_mapping(dump_path: str, *, is_parquet: bool) -> Dict[int, str]:
    titles: Dict[int, str] = {}
    it = page_iter_parquet(dump_path) if is_parquet else page_iter(dump_path)
    for doc_id, title, _ in it:
        titles[int(doc_id)] = title
    _save_pickle(titles, config.TITLES_PATH)
    return titles


def build_title_index(dump_path: str, *, is_parquet: bool) -> InvertedIndex:
    _ensure_local_dirs()
    out_dir = config.TITLE_INDEX_DIR

    index = InvertedIndex()
    it = page_iter_parquet(dump_path) if is_parquet else page_iter(dump_path)
    for doc_id, title, _ in it:
        toks = _tokenize(title)
        if toks:
            index.add_doc(int(doc_id), toks)

    _write_index(index, out_dir, "title")
    bucket_name = config.BUCKET_NAME if config.WRITE_TO_GCS else None
    return InvertedIndex.read_index(str(out_dir), "title", bucket_name=bucket_name)


def build_body_index_and_aux(dump_path: str, *, is_parquet: bool) -> InvertedIndex:
    """Build body index + doc_len + avgdl + doc_norms (for cosine)."""
    _ensure_local_dirs()
    out_dir = config.BODY_INDEX_DIR

    index = InvertedIndex()
    doc_len: Dict[int, int] = {}

    it = page_iter_parquet(dump_path) if is_parquet else page_iter(dump_path)
    for doc_id, _, body in it:
        clean = body if is_parquet else remove_markdown(body)
        toks = _tokenize(clean)
        if not toks:
            continue
        doc_id_i = int(doc_id)
        index.add_doc(doc_id_i, toks)
        doc_len[doc_id_i] = len(toks)

    _write_index(index, out_dir, "body")
    _save_pickle(doc_len, config.DOC_LEN_PATH)

    avgdl = sum(doc_len.values()) / max(1, len(doc_len))
    avgdl_path = _as_path(config.AVGDL_PATH)
    avgdl_path.parent.mkdir(parents=True, exist_ok=True)
    avgdl_path.write_text(str(avgdl), encoding="utf-8")

    # Compute cosine norms by scanning postings once
    bucket_name = config.BUCKET_NAME if config.WRITE_TO_GCS else None
    loaded = InvertedIndex.read_index(str(out_dir), "body", bucket_name=bucket_name)
    N = max(1, len(doc_len))

    sq_sum: DefaultDict[int, float] = defaultdict(float)
    for term, df in loaded.df.items():
        idf = math.log((N + 1) / (df + 1))
        pls = loaded.read_a_posting_list(str(out_dir), term, bucket_name=bucket_name)
        for d, tf in pls:
            w = tf * idf
            sq_sum[int(d)] += w * w

    doc_norms = {d: math.sqrt(v) for d, v in sq_sum.items()}
    _save_pickle(doc_norms, config.DOC_NORMS_PATH)

    return loaded


def build_anchor_index_batched(dump_path: str, *, is_parquet: bool, pages_per_batch: int = 20_000) -> InvertedIndex:
    """Build anchor index using batching + k-way merge of partial indices."""
    _ensure_local_dirs()
    out_dir = _as_path(config.ANCHOR_INDEX_DIR)
    ANCHOR_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    titles_path = _as_path(config.TITLES_PATH)
    if not titles_path.exists():
        build_title_mapping(dump_path, is_parquet=is_parquet)

    with open(titles_path, "rb") as f:
        id2title: Dict[int, str] = pickle.load(f)

    title2id: Dict[str, int] = {}
    for doc_id, title in id2title.items():
        if not title:
            continue
        title2id[title] = int(doc_id)
        title2id[title.replace("_", " ")] = int(doc_id)

    # 1) Build partial indices
    part_dirs: List[Path] = []
    batch_counts: DefaultDict[int, Counter] = defaultdict(Counter)
    batch_i = 0
    in_batch = 0

    if is_parquet:
        # Use anchor_text column directly from parquet
        it = page_iter_parquet_with_anchors(dump_path)
        for doc_id, title, body, anchor_text_list in it:
            in_batch += 1
            
            # Process anchor_text_list: [{'id': 1234, 'text': 'anchor text'}, ...]
            if anchor_text_list and isinstance(anchor_text_list, list):
                for link in anchor_text_list:
                    if not isinstance(link, dict):
                        continue
                    target_id = link.get('id')
                    anchor_text = link.get('text', '')
                    if target_id is None:
                        continue
                    try:
                        target_id = int(target_id)
                    except (ValueError, TypeError):
                        continue
                    toks = _tokenize(anchor_text)
                    if toks:
                        batch_counts[target_id].update(toks)
            
            # Check if we need to flush this batch
            if in_batch >= pages_per_batch:
                part_dir = ANCHOR_PARTS_DIR / f"part_{batch_i:05d}"
                part_dir.mkdir(parents=True, exist_ok=True)
                part_index = InvertedIndex()
                for target_id, cnt in batch_counts.items():
                    part_index.add_doc(int(target_id), cnt)
                _write_index(part_index, part_dir, "anchor_part")
                part_dirs.append(part_dir)
                batch_counts = defaultdict(Counter)
                in_batch = 0
                batch_i += 1
    else:
        # Keep existing logic for XML dumps using get_wikilinks(body)
        it = page_iter(dump_path)
        for _, __, body in it:
            in_batch += 1
            for target_title, anchor_text in get_wikilinks(body):
                target_title_norm = target_title.replace("_", " ").strip()
                target_id = title2id.get(target_title) or title2id.get(target_title_norm)
                if target_id is None:
                    continue
                toks = _tokenize(anchor_text)
                if toks:
                    batch_counts[int(target_id)].update(toks)

        if in_batch >= pages_per_batch:
            part_dir = ANCHOR_PARTS_DIR / f"part_{batch_i:05d}"
            part_dir.mkdir(parents=True, exist_ok=True)

            part_index = InvertedIndex()
            for doc_id, cnt in batch_counts.items():
                part_index.add_doc(int(doc_id), cnt)  # cnt is Counter -> treated as iterable of tokens with multiplicity
            _write_index(part_index, part_dir, "anchor_part")

            part_dirs.append(part_dir)
            batch_counts = defaultdict(Counter)
            in_batch = 0
            batch_i += 1

    if batch_counts:
        part_dir = ANCHOR_PARTS_DIR / f"part_{batch_i:05d}"
        part_dir.mkdir(parents=True, exist_ok=True)

        part_index = InvertedIndex()
        for doc_id, cnt in batch_counts.items():
            part_index.add_doc(int(doc_id), cnt)
        _write_index(part_index, part_dir, "anchor_part")
        part_dirs.append(part_dir)

    # 2) Merge partials (term-wise)
    partials: List[Tuple[InvertedIndex, str]] = []
    for pdir in part_dirs:
        idx = InvertedIndex.read_index(str(pdir), "anchor_part", bucket_name=None)
        partials.append((idx, str(pdir)))

    # Build sorted term lists for each partial
    term_lists = [sorted(idx.df.keys()) for idx, _ in partials]
    pointers = [0] * len(term_lists)

    final = InvertedIndex()
    final._posting_list = defaultdict(list)
    final.df = Counter()
    final.term_total = Counter()

    CHUNK_TERMS = 20_000
    chunk_terms = 0
    chunk_id = 0
    tmp_merge = out_dir / "_tmp_anchor_merge"
    tmp_merge.mkdir(parents=True, exist_ok=True)
    merged_chunks: List[Path] = []

    def flush():
        nonlocal chunk_terms, chunk_id, final
        if not final._posting_list:
            return
        chunk_dir = tmp_merge / f"chunk_{chunk_id:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        _write_index(final, chunk_dir, "anchor_chunk")
        merged_chunks.append(chunk_dir)

        final = InvertedIndex()
        final._posting_list = defaultdict(list)
        final.df = Counter()
        final.term_total = Counter()
        chunk_terms = 0
        chunk_id += 1

    while True:
        current_terms = []
        for i, terms in enumerate(term_lists):
            if pointers[i] < len(terms):
                current_terms.append(terms[pointers[i]])
        if not current_terms:
            break

        term = min(current_terms)
        doc_tf: DefaultDict[int, int] = defaultdict(int)

        for i, terms in enumerate(term_lists):
            if pointers[i] < len(terms) and terms[pointers[i]] == term:
                idx, pdir = partials[i]
                pl = idx.read_a_posting_list(pdir, term)
                for d, tf in pl:
                    doc_tf[int(d)] += int(tf)
                pointers[i] += 1

        merged_pl = sorted(doc_tf.items(), key=lambda x: x[0])
        final._posting_list[term] = merged_pl
        final.df[term] = len(merged_pl)
        final.term_total[term] = sum(tf for _, tf in merged_pl)

        chunk_terms += 1
        if chunk_terms >= CHUNK_TERMS:
            flush()

    flush()

    # If multiple chunks, do one more merge pass (usually small number)
    if not merged_chunks:
        # empty anchor index
        _write_index(final, out_dir, "anchor")
        return InvertedIndex.read_index(str(out_dir), "anchor", bucket_name=None)

    if len(merged_chunks) == 1:
        # Promote single chunk to anchor
        idx = InvertedIndex.read_index(str(merged_chunks[0]), "anchor_chunk", bucket_name=None)
        idx.write_index(str(out_dir), "anchor", bucket_name=None)
        return InvertedIndex.read_index(str(out_dir), "anchor", bucket_name=None)

    # Merge chunks into final anchor
    chunk_partials = [(InvertedIndex.read_index(str(cd), "anchor_chunk", bucket_name=None), str(cd)) for cd in merged_chunks]
    term_lists = [sorted(idx.df.keys()) for idx, _ in chunk_partials]
    pointers = [0] * len(term_lists)

    final2 = InvertedIndex()
    final2._posting_list = defaultdict(list)
    final2.df = Counter()
    final2.term_total = Counter()

    while True:
        current_terms = []
        for i, terms in enumerate(term_lists):
            if pointers[i] < len(terms):
                current_terms.append(terms[pointers[i]])
        if not current_terms:
            break
        term = min(current_terms)

        doc_tf: DefaultDict[int, int] = defaultdict(int)
        for i, terms in enumerate(term_lists):
            if pointers[i] < len(terms) and terms[pointers[i]] == term:
                idx, pdir = chunk_partials[i]
                pl = idx.read_a_posting_list(pdir, term)
                for d, tf in pl:
                    doc_tf[int(d)] += int(tf)
                pointers[i] += 1

        merged_pl = sorted(doc_tf.items(), key=lambda x: x[0])
        final2._posting_list[term] = merged_pl
        final2.df[term] = len(merged_pl)
        final2.term_total[term] = sum(tf for _, tf in merged_pl)

    _write_index(final2, out_dir, "anchor")
    return InvertedIndex.read_index(str(out_dir), "anchor", bucket_name=None)


def build_pageviews(dump_path: str = "dummy") -> Dict[int, int]:
    """Build pageviews dictionary from Wikimedia pageview dump for August 2021.
    
    Downloads and processes pageviews-202108-user.bz2 from Wikimedia.
    Creates a mapping from wiki_id to total pageviews for that month.
    """
    _ensure_local_dirs()
    
    # Using user page views (as opposed to spiders and automated traffic) for the
    # month of August 2021
    pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
    p = Path(pv_path)
    pv_name = p.name
    pv_temp = f'{p.stem}-4dedup.txt'
    pv_clean = _as_path(config.PAGEVIEWS_PATH)
    
    print("Downloading pageviews file (this may take a while)...")
    if not Path(pv_name).exists():
        urllib.request.urlretrieve(pv_path, pv_name)
        print(f"Downloaded {pv_name}")
    else:
        print(f"Using existing {pv_name}")
    
    print("Processing pageviews...")
    # Filter for English pages, and keep just two fields: article ID (3) and monthly
    # total number of page views (5). Then, remove lines with article id or page
    # view values that are not a sequence of digits.
    with bz2.open(pv_name, 'rt', encoding='utf-8', errors='ignore') as f_in:
        with open(pv_temp, 'wt', encoding='utf-8') as f_out:
            for line in f_in:
                if line.startswith('en.wikipedia'):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        article_id = parts[2]
                        pageviews = parts[4]
                        # Check if both are digits
                        if re.match(r'^\d+$', article_id) and re.match(r'^\d+$', pageviews):
                            f_out.write(f"{article_id} {pageviews}\n")
    
    print("Aggregating pageviews...")
    # Create a Counter (dictionary) that sums up the pages views for the same
    # article, resulting in a mapping from article id to total page views.
    wid2pv = Counter()
    with open(pv_temp, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    wid = int(parts[0])
                    pv = int(parts[1])
                    wid2pv[wid] += pv
                except ValueError:
                    continue
    
    # Write out the counter as binary file (pickle it)
    _save_pickle(dict(wid2pv), config.PAGEVIEWS_PATH)
    
    # Clean up temp file
    if os.path.exists(pv_temp):
        os.unlink(pv_temp)
    
    print(f"✓ Saved pageviews to {config.PAGEVIEWS_PATH}")
    print(f"  Total articles with pageviews: {len(wid2pv):,}")
    print(f"  Total pageviews: {sum(wid2pv.values()):,}")
    
    return dict(wid2pv)


def build_pagerank(dump_path: str, *, is_parquet: bool) -> Dict[int, float]:
    """Build PageRank scores from Wikipedia link graph.
    
    Constructs a link graph from wikilinks in articles and computes PageRank.
    """
    _ensure_local_dirs()
    
    # Load titles mapping (needed to resolve link targets)
    titles_path = _as_path(config.TITLES_PATH)
    if not titles_path.exists():
        print("Building titles mapping first...")
        build_title_mapping(dump_path, is_parquet=is_parquet)
    
    with open(titles_path, "rb") as f:
        id2title: Dict[int, str] = pickle.load(f)
    
    title2id: Dict[str, int] = {}
    for doc_id, title in id2title.items():
        if not title:
            continue
        title2id[title] = int(doc_id)
        title2id[title.replace("_", " ")] = int(doc_id)
    
    print("Building link graph from Wikipedia dump...")
    # Build graph: from_id -> set of to_ids
    graph: Dict[int, Set[int]] = defaultdict(set)
    processed = 0
    
    if is_parquet:
        # Use anchor_text column directly from parquet
        it = page_iter_parquet_with_anchors(dump_path)
        for doc_id, title, body, anchor_text_list in it:
            processed += 1
            if processed % 100_000 == 0:
                print(f"Processed {processed:,} pages...")
            
            doc_id_i = int(doc_id)
            
            if anchor_text_list is None or not anchor_text_list:
                continue
            
            # Process anchor_text_list: [{'id': 1234, 'text': 'anchor text'}, ...]
            for link in anchor_text_list:
                if not isinstance(link, dict):
                    continue
                target_id = link.get('id')
                if target_id is None:
                    continue
                try:
                    target_id = int(target_id)
                except (ValueError, TypeError):
                    continue
                if target_id != doc_id_i:
                    graph[doc_id_i].add(target_id)
    else:
        # Keep existing logic for XML dumps using get_wikilinks(body)
        it = page_iter(dump_path)
        for doc_id, _, body in it:
            processed += 1
            if processed % 100_000 == 0:
                print(f"Processed {processed:,} pages...")
            
            doc_id_i = int(doc_id)
            for target_title, _ in get_wikilinks(body):
                target_title_norm = target_title.replace("_", " ").strip()
                target_id = title2id.get(target_title) or title2id.get(target_title_norm)
                if target_id is not None and target_id != doc_id_i:
                    graph[doc_id_i].add(target_id)
    
    print(f"✓ Built graph with {len(graph):,} nodes and {sum(len(neighbors) for neighbors in graph.values()):,} edges")
    
    # Compute PageRank
    print("Computing PageRank (N={:,} nodes, damping=0.85)...".format(len(graph)))
    
    N = len(graph)
    if N == 0:
        print("Warning: Empty graph, returning empty PageRank")
        _save_pickle({}, config.PAGERANK_PATH)
        return {}
    
    damping = 0.85
    max_iterations = 100
    tolerance = 1e-6
    
    # Initialize PageRank scores
    pr: Dict[int, float] = {node: 1.0 / N for node in graph.keys()}
    
    # Find sink nodes (nodes with no outgoing links)
    all_nodes: Set[int] = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    sink_nodes: Set[int] = all_nodes - set(graph.keys())
    
    for iteration in range(max_iterations):
        new_pr: Dict[int, float] = {}
        max_change = 0.0
        
        for node in all_nodes:
            # Contribution from random jump
            new_pr[node] = (1 - damping) / N
            
            # Contribution from incoming links
            incoming_sum = 0.0
            for from_node, neighbors in graph.items():
                if node in neighbors:
                    out_degree = len(neighbors) if neighbors else 1
                    incoming_sum += pr[from_node] / out_degree
            
            # Contribution from sink nodes (distributed evenly)
            sink_contribution = sum(pr[sink] for sink in sink_nodes) / N if N > 0 else 0.0
            
            new_pr[node] += damping * (incoming_sum + sink_contribution)
            
            change = abs(new_pr[node] - pr.get(node, 0.0))
            max_change = max(max_change, change)
        
        pr = new_pr
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: max change = {max_change:.2e}")
        
        if max_change < tolerance:
            print(f"✓ Converged after {iteration + 1} iterations")
            break
    
    # Normalize (optional, but helps with interpretation)
    total = sum(pr.values())
    if total > 0:
        pr = {k: v / total for k, v in pr.items()}
    
    _save_pickle(pr, config.PAGERANK_PATH)
    
    print(f"✓ Saved PageRank to {config.PAGERANK_PATH}")
    print(f"  Nodes with PageRank: {len(pr):,}")
    print(f"  Max PageRank: {max(pr.values()):.6f}")
    print(f"  Min PageRank: {min(pr.values()):.6f}")
    
    return pr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="XML .bz2, parquet file/dir, or gs://... parquet path")
    parser.add_argument("--build", choices=["body", "title", "anchor", "pageviews", "pagerank", "lsi", "all"], default="all")
    parser.add_argument("--parquet", action="store_true", help="Force parquet mode")
    args = parser.parse_args()

    is_parquet = args.parquet or args.dump.endswith(".parquet") or args.dump.endswith(".parq") or args.dump.startswith("gs://")
    
    if args.build in ("title", "all"):
        build_title_mapping(args.dump, is_parquet=is_parquet)
        build_title_index(args.dump, is_parquet=is_parquet)
    if args.build in ("body", "all"):
        build_body_index_and_aux(args.dump, is_parquet=is_parquet)
    if args.build in ("anchor", "all"):
        build_anchor_index_batched(args.dump, is_parquet=is_parquet, pages_per_batch=config.ANCHOR_PAGES_PER_BATCH)
    if args.build == "pageviews":
        build_pageviews(args.dump)
    if args.build == "pagerank":
        build_pagerank(args.dump, is_parquet=is_parquet)
    if args.build == "lsi":
        # LSI requires body index and doc_norms to be built first
        print("\n" + "=" * 60)
        print("Building LSI index...")
        print("=" * 60)
        from ranking.lsi import build_lsi_index
        
        # Load body index and doc_norms
        bucket_name = config.BUCKET_NAME if config.WRITE_TO_GCS else None
        body_index = InvertedIndex.read_index(str(config.BODY_INDEX_DIR), "body", bucket_name=bucket_name)
        
        # Load doc_norms
        doc_norms_path = _as_path(config.DOC_NORMS_PATH)
        if bucket_name:
            from google.cloud import storage
            from inverted_index_gcp import get_bucket
            gcs_bucket = get_bucket(bucket_name)
            blob = gcs_bucket.blob(str(doc_norms_path))
            if not blob.exists():
                print("Error: doc_norms.pkl not found. Please build body index first.")
                sys.exit(1)
            import tempfile
            temp_path = Path(tempfile.mktemp(suffix='.pkl'))
            blob.download_to_filename(str(temp_path))
            with open(temp_path, 'rb') as f:
                doc_norms = pickle.load(f)
            temp_path.unlink()
        else:
            if not doc_norms_path.exists():
                print("Error: doc_norms.pkl not found. Please build body index first.")
                sys.exit(1)
            with open(doc_norms_path, 'rb') as f:
                doc_norms = pickle.load(f)
        
        # Build LSI index
        output_dir = _as_path(config.LSI_DIR)
        build_lsi_index(
            body_index=body_index,
            body_index_dir=str(config.BODY_INDEX_DIR),
            doc_norms=doc_norms,
            output_dir=output_dir,
            n_components=config.LSI_N_COMPONENTS,
            max_terms=config.LSI_MAX_TERMS,
            max_docs=config.LSI_MAX_DOCS,
        )
        print("✓ LSI index built successfully")
    if args.build == "all":
        # Build pageviews and pagerank as part of "all"
        print("\n" + "=" * 60)
        print("Building PageViews...")
        print("=" * 60)
        build_pageviews()
        print("\n" + "=" * 60)
        print("Building PageRank...")
        print("=" * 60)
        build_pagerank(args.dump, is_parquet=is_parquet)

    print("Index building completed.")
