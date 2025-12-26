# indexing/build_indices.py
from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import config
from inverted_index_gcp import InvertedIndex
from parser_utils import get_wikilinks, page_iter, page_iter_parquet, remove_markdown
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

    it = page_iter_parquet(dump_path) if is_parquet else page_iter(dump_path)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="XML .bz2, parquet file/dir, or gs://... parquet path")
    parser.add_argument("--build", choices=["body", "title", "anchor", "all"], default="all")
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

    print("Index building completed.")
