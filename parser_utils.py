# parser_utils.py
from __future__ import annotations

import bz2
import os
import re
import tempfile
from pathlib import Path
from typing import Generator, List, Optional, Tuple
from xml.etree import ElementTree

import mwparserfromhell as mwp

# Optional imports
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq  # type: ignore
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

try:
    from google.cloud import storage  # type: ignore
    HAS_GCS = True
except Exception:
    HAS_GCS = False


PageTuple = Tuple[int, str, str]  # (wiki_id, title, body)


def _get_namespace(tag: str) -> str:
    """Extract XML namespace from tag like '{http://...}mediawiki'."""
    m = re.match(r"\{(.*)\}", tag)
    return m.group(1) if m else ""


def page_iter(wiki_file: str) -> Generator[PageTuple, None, None]:
    """Stream-iterate a Wikipedia XML dump (.bz2) yielding (wiki_id, title, body)
    for main namespace pages (ns == '0') that are not redirects.

    This is meant for OFFLINE indexing from the raw XML dump.
    """
    with bz2.open(wiki_file, "rt", encoding="utf-8", errors="ignore") as f_in:
        context = ElementTree.iterparse(f_in, events=("start", "end"))

        event, root = next(context)  # root with namespace
        ns_uri = _get_namespace(root.tag)
        ns = {"ns": ns_uri} if ns_uri else {}

        for event, elem in context:
            if event != "end":
                continue

            if elem.tag.endswith("page"):
                redirect = elem.find("./ns:redirect", ns) if ns else elem.find("./redirect")
                if redirect is not None:
                    elem.clear()
                    continue

                ns_elem = elem.find("./ns:ns", ns) if ns else elem.find("./ns")
                if ns_elem is None or (ns_elem.text or "").strip() != "0":
                    elem.clear()
                    continue

                id_elem = elem.find("./ns:id", ns) if ns else elem.find("./id")
                title_elem = elem.find("./ns:title", ns) if ns else elem.find("./title")
                text_elem = elem.find("./ns:revision/ns:text", ns) if ns else elem.find("./revision/text")

                wiki_id = int((id_elem.text or "0").strip()) if id_elem is not None else 0
                title = (title_elem.text or "").strip() if title_elem is not None else ""
                body = text_elem.text if text_elem is not None and text_elem.text is not None else ""

                yield wiki_id, title, body

                elem.clear()
                root.clear()


def remove_markdown(text: str) -> str:
    """Strip MediaWiki markup, leaving mostly plain text."""
    if text is None:
        return ""
    return mwp.parse(text).strip_code()


def filter_article_links(title: Optional[str]) -> bool:
    """Return True only for wikilinks that likely point to actual articles."""
    if not title:
        return False

    t = str(title).strip()
    if "#" in t:
        t = t.split("#", 1)[0].strip()
    if t.startswith(":"):
        t = t[1:].strip()
    if not t:
        return False

    lower = t.lower()
    bad_prefixes = (
        "file:", "image:", "media:",
        "category:",
        "help:", "special:", "portal:", "template:",
        "talk:", "user:", "wikipedia:", "draft:",
        "module:", "book:", "education program:", "timedtext:",
    )
    if lower.startswith(bad_prefixes):
        return False

    return True


def get_wikilinks(markdown: str) -> List[Tuple[str, str]]:
    """Extract outgoing links from a wiki page body.

    Returns list of (linked_title, anchor_text). If anchor text is missing, uses the title.
    """
    if markdown is None:
        return []

    wikicode = mwp.parse(markdown)
    links: List[Tuple[str, str]] = []
    for wl in wikicode.ifilter_wikilinks():
        title = str(wl.title).strip()
        if not filter_article_links(title):
            continue

        title = re.sub(r"#.*$", "", title).strip()
        if not title:
            continue

        text = wl.text
        if text is None:
            anchor = title
        else:
            anchor = text.strip_code().strip()
            if not anchor:
                anchor = title

        links.append((title, anchor))

    return links


def _list_gcs_parquet_files(gcs_path: str) -> List[str]:
    """List all parquet files under a GCS prefix (gs://bucket/prefix/)."""
    if not HAS_GCS:
        raise ImportError("google-cloud-storage is required for reading from GCS.")

    if not gcs_path.startswith("gs://"):
        return []

    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    import config
    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(bucket_name)

    parquet_files: List[str] = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".parquet") or blob.name.endswith(".parq"):
            parquet_files.append(f"gs://{bucket_name}/{blob.name}")
    parquet_files.sort()
    return parquet_files


def _iter_parquet_batches(local_parquet_file: str, *, batch_size: int = 10_000, columns: Optional[List[str]] = None):
    """Yield record batches from a local parquet file with minimal RAM use.
    
    Args:
        local_parquet_file: Path to local parquet file
        batch_size: Number of rows per batch
        columns: Optional list of column names to read (more efficient if specified)
    """
    if HAS_PYARROW:
        pf = pq.ParquetFile(local_parquet_file)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield batch
    elif HAS_PANDAS:
        # Fallback: loads whole file into memory (only suitable for small dev tests)
        df = pd.read_parquet(local_parquet_file, engine="pyarrow" if HAS_PYARROW else "auto", columns=columns)
        yield df
    else:
        raise ImportError("Need either pyarrow (preferred) or pandas to read parquet.")


def page_iter_parquet(parquet_path: str) -> Generator[PageTuple, None, None]:
    """Stream-iterate preprocessed parquet file(s) yielding (wiki_id, title, body).

    Supported:
    - Local parquet file: /path/file.parquet
    - Local directory: /path/dir/ (all *.parquet/*.parq)
    - GCS parquet file: gs://bucket/path/file.parquet
    - GCS directory: gs://bucket/path/dir/
    """
    is_gcs = parquet_path.startswith("gs://")

    parquet_files: List[str] = []
    if is_gcs:
        if parquet_path.endswith("/"):
            parquet_files = _list_gcs_parquet_files(parquet_path)
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in GCS directory: {parquet_path}")
        else:
            parquet_files = [parquet_path]
    else:
        p = Path(parquet_path)
        if p.is_dir():
            parquet_files = [str(f) for f in (list(p.glob("*.parquet")) + list(p.glob("*.parq")))]
            parquet_files.sort()
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in directory: {parquet_path}")
        elif p.exists():
            parquet_files = [str(p)]
        else:
            raise FileNotFoundError(f"Parquet file/directory not found: {parquet_path}")

    # For each parquet file, produce rows in a streaming fashion.
    for parquet_file in parquet_files:
        tmp_file_path: Optional[str] = None
        try:
            local_file = parquet_file
            if is_gcs:
                if not HAS_GCS:
                    raise ImportError("google-cloud-storage is required for reading from GCS.")
                import config
                parts = parquet_file[5:].split("/", 1)
                bucket_name = parts[0]
                blob_path = parts[1]

                client = storage.Client(project=config.PROJECT_ID)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                if not blob.exists():
                    continue

                tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
                tmp_file_path = tmp.name
                tmp.close()
                blob.download_to_filename(tmp_file_path)
                local_file = tmp_file_path

            # First batch: detect columns.
            id_col = title_col = body_col = None

            if HAS_PYARROW:
                pf = pq.ParquetFile(local_file)
                # Find columns by common names
                for name in pf.schema.names:
                    nl = name.lower()
                    if id_col is None and nl in ("doc_id", "id", "wiki_id", "document_id"):
                        id_col = name
                    if title_col is None and nl in ("title", "page_title", "name"):
                        title_col = name
                    if body_col is None and nl in ("body", "text", "content", "article_text", "page_text"):
                        body_col = name

                if id_col is None or title_col is None or body_col is None:
                    # Can't process this file
                    continue

                # Use _iter_parquet_batches for efficient batch processing with column selection
                for batch in _iter_parquet_batches(local_file, batch_size=10_000, columns=[id_col, title_col, body_col]):
                    # Extract columns from batch
                    if HAS_PYARROW and hasattr(batch, 'column'):  # PyArrow RecordBatch
                        doc_ids = batch.column(0).to_pylist()
                        titles = batch.column(1).to_pylist()
                        bodies = batch.column(2).to_pylist()
                    else:  # Pandas DataFrame fallback
                        doc_ids = batch[id_col].tolist()
                        titles = batch[title_col].tolist()
                        bodies = batch[body_col].tolist()
                    
                    for doc_id, title, body in zip(doc_ids, titles, bodies):
                        if doc_id is None or title is None or body is None:
                            continue
                        try:
                            wiki_id = int(doc_id)
                        except Exception:
                            continue
                        t = str(title)
                        b = str(body)
                        if wiki_id == 0 or not t or not b:
                            continue
                        yield wiki_id, t, b

            else:
                # Pandas fallback (dev only)
                if not HAS_PANDAS:
                    raise ImportError("Need pyarrow (preferred) or pandas to read parquet.")
                df = pd.read_parquet(local_file, engine="pyarrow" if HAS_PYARROW else "auto")
                cols = list(df.columns)
                for col in cols:
                    cl = col.lower()
                    if id_col is None and cl in ("doc_id", "id", "wiki_id", "document_id"):
                        id_col = col
                    if title_col is None and cl in ("title", "page_title", "name"):
                        title_col = col
                    if body_col is None and cl in ("body", "text", "content", "article_text", "page_text"):
                        body_col = col
                if id_col is None or title_col is None or body_col is None:
                    continue
                for _, row in df.iterrows():
                    doc_id = row[id_col]
                    title = row[title_col]
                    body = row[body_col]
                    if pd.isna(doc_id) or pd.isna(title) or pd.isna(body):
                        continue
                    wiki_id = int(doc_id)
                    t = str(title)
                    b = str(body)
                    if wiki_id == 0 or not t or not b:
                        continue
                    yield wiki_id, t, b

        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass
