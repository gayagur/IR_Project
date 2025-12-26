# config.py
from __future__ import annotations

from pathlib import Path

# ============================================================================
# Project base paths (local filesystem)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDICES_DIR = BASE_DIR / "indices"
AUX_DIR = BASE_DIR / "aux"

# ============================================================================
# GCP configuration (only needed if you read source data from GCS or store outputs on GCS)
# ============================================================================
PROJECT_ID = "ir-project-481821"
BUCKET_NAME = "matiasgaya333"

# ============================================================================
# Input data path
# ============================================================================
# Examples:
#   Local XML dump: "/path/to/enwiki.xml.bz2"
#   Local parquet dir: "/path/to/wikidata20210801_preprocessed/"
#   GCS parquet dir: "gs://<bucket>/raw/wikidata20210801_preprocessed/"
RAW_DATA_PATH = "gs://wikidata20210801_preprocessed/"

# ============================================================================
# Storage mode for indices
# ============================================================================
# Recommended: keep indices on LOCAL disk on your VM/Dataproc master, and only use GCS as the input source.
# If you really need to store indices on GCS, set WRITE_TO_GCS = True AND ensure your InvertedIndex uses bucket_name.
WRITE_TO_GCS = False

# When WRITE_TO_GCS=True, we use bucket-relative paths (NO 'gs://...' prefix) because inverted_index_gcp.py
# opens blobs relative to bucket root.
GCS_INDICES_DIR = "indices"
GCS_AUX_DIR = "aux"

# Index directories
BODY_INDEX_DIR = (GCS_INDICES_DIR + "/body") if WRITE_TO_GCS else (INDICES_DIR / "body")
TITLE_INDEX_DIR = (GCS_INDICES_DIR + "/title") if WRITE_TO_GCS else (INDICES_DIR / "title")
ANCHOR_INDEX_DIR = (GCS_INDICES_DIR + "/anchor") if WRITE_TO_GCS else (INDICES_DIR / "anchor")

# Auxiliary file paths
DOC_NORMS_PATH = (GCS_AUX_DIR + "/doc_norms.pkl") if WRITE_TO_GCS else (AUX_DIR / "doc_norms.pkl")
DOC_LEN_PATH = (GCS_AUX_DIR + "/doc_len.pkl") if WRITE_TO_GCS else (AUX_DIR / "doc_len.pkl")
AVGDL_PATH = (GCS_AUX_DIR + "/avgdl.txt") if WRITE_TO_GCS else (AUX_DIR / "avgdl.txt")
TITLES_PATH = (GCS_AUX_DIR + "/titles.pkl") if WRITE_TO_GCS else (AUX_DIR / "titles.pkl")

PAGERANK_PATH = (GCS_AUX_DIR + "/pagerank.pkl") if WRITE_TO_GCS else (AUX_DIR / "pagerank.pkl")
PAGEVIEWS_PATH = (GCS_AUX_DIR + "/pageviews.pkl") if WRITE_TO_GCS else (AUX_DIR / "pageviews.pkl")

# ============================================================================
# LSI configuration (optional)
# ============================================================================
LSI_DIR = (GCS_AUX_DIR + "/lsi") if WRITE_TO_GCS else (AUX_DIR / "lsi")
LSI_VECTORS_PATH = (LSI_DIR + "/lsi_vectors.pkl") if WRITE_TO_GCS else (AUX_DIR / "lsi" / "lsi_vectors.pkl")
LSI_SVD_COMPONENTS_PATH = (LSI_DIR + "/svd_components.pkl") if WRITE_TO_GCS else (AUX_DIR / "lsi" / "svd_components.pkl")
TERM_TO_IDX_PATH = (LSI_DIR + "/term_to_idx.pkl") if WRITE_TO_GCS else (AUX_DIR / "lsi" / "term_to_idx.pkl")
DOC_TO_IDX_PATH = (LSI_DIR + "/doc_to_idx.pkl") if WRITE_TO_GCS else (AUX_DIR / "lsi" / "doc_to_idx.pkl")

LSI_N_COMPONENTS = 100
LSI_MAX_TERMS = 50000
LSI_MAX_DOCS = None  # None = all documents

# ============================================================================
# Indexing parameters
# ============================================================================
ANCHOR_PAGES_PER_BATCH = 20000
