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
QUERIES_DIR = BASE_DIR  # Queries files are in the root directory

# ============================================================================
# GCP configuration (only needed if you read source data from GCS or store outputs on GCS)
# ============================================================================
PROJECT_ID = "ir-project-481821"
BUCKET_NAME = "matiasgaya333"

# ============================================================================
# Server configuration
# ============================================================================
# IP address and port of the GCP instance running the search engine
INSTANCE_IP = "104.198.58.119"
INSTANCE_PORT = 8080
# Base URL for the search engine server
BASE_URL = f"http://{INSTANCE_IP}:{INSTANCE_PORT}"

# ============================================================================
# Input data path
# ============================================================================
# Examples:
#   Local XML dump: "/path/to/enwiki.xml.bz2"
#   Local parquet dir: "/path/to/wikidata20210801_preprocessed/"
#   GCS parquet dir: "gs://<bucket>/raw/wikidata20210801_preprocessed/"
RAW_DATA_PATH = "gs://matiasgaya333/raw/wikidata20210801_preprocessed"

# ============================================================================
# Storage mode for indices
# ============================================================================
# Recommended: keep indices on LOCAL disk on your VM/Dataproc master, and only use GCS as the input source.
# If you really need to store indices on GCS, set WRITE_TO_GCS = True AND ensure your InvertedIndex uses bucket_name.
WRITE_TO_GCS =  False

# Set READ_FROM_GCS = True to read all indices and auxiliary files from GCS at runtime.
# This is useful when running the server on a local machine but indices are stored in GCS.
READ_FROM_GCS = False

# When WRITE_TO_GCS=True or READ_FROM_GCS=True, we use bucket-relative paths (NO 'gs://...' prefix) 
# because inverted_index_gcp.py opens blobs relative to bucket root.
GCS_INDICES_DIR = "indices"
GCS_AUX_DIR = "aux"

# Determine if we should use GCS paths (either for writing OR reading from GCS)
USE_GCS_PATHS = WRITE_TO_GCS or READ_FROM_GCS

# Index directories
BODY_INDEX_DIR = (GCS_INDICES_DIR + "/body") if USE_GCS_PATHS else (INDICES_DIR / "body")
TITLE_INDEX_DIR = (GCS_INDICES_DIR + "/title") if USE_GCS_PATHS else (INDICES_DIR / "title")
ANCHOR_INDEX_DIR = (GCS_INDICES_DIR + "/anchor") if USE_GCS_PATHS else (INDICES_DIR / "anchor")

# Auxiliary file paths
DOC_NORMS_PATH = (GCS_AUX_DIR + "/doc_norms.pkl") if USE_GCS_PATHS else (AUX_DIR / "doc_norms.pkl")
DOC_LEN_PATH = (GCS_AUX_DIR + "/doc_len.pkl") if USE_GCS_PATHS else (AUX_DIR / "doc_len.pkl")
AVGDL_PATH = (GCS_AUX_DIR + "/avgdl.txt") if USE_GCS_PATHS else (AUX_DIR / "avgdl.txt")
TITLES_PATH = (GCS_AUX_DIR + "/titles.pkl") if USE_GCS_PATHS else (AUX_DIR / "titles.pkl")

PAGERANK_PATH = (GCS_AUX_DIR + "/pagerank.pkl") if USE_GCS_PATHS else (AUX_DIR / "pagerank.pkl")
PAGEVIEWS_PATH = (GCS_AUX_DIR + "/pageviews.pkl") if USE_GCS_PATHS else (AUX_DIR / "pageviews.pkl")


# ============================================================================
# BM25 parameters
# ============================================================================
# BM25 scoring parameters (tuned for best Average Precision@10 performance)
BM25_K1 = 3.0 # Term frequency saturation parameter (default: 2.5)
BM25_B = 0.25  # Document length normalization parameter (default: 0.0, no normalization)

# ============================================================================
# Ranking weights for signal merging
# ============================================================================
# Weights for merging different search signals in /search endpoint
# These can be tuned to optimize performance
BODY_WEIGHT = 0.4      # BM25 body search weight
TITLE_WEIGHT = 0.75    # Title match weight
ANCHOR_WEIGHT = 1.0   # Anchor text weight

# PageRank and PageView boost weights (applied after merging)
PAGERANK_BOOST = 0.15  # PageRank boost weight
PAGEVIEW_BOOST = 0.10  # PageView boost weight

# ============================================================================
# GloVe semantic features configuration
# ============================================================================
ENABLE_GLOVE = True  # Enable GloVe-based semantic reranking (set to False to disable)
GLOVE_BETA = 2.7  # Weight for GloVe score: final = base_score + beta * cosine
GLOVE_CANDIDATE_POOL = 150  # Number of candidates to consider for GloVe reranking
GLOVE_TOP_K = 12  # Number of top documents to use for query embedding

# GloVe paths
GLOVE_EMBEDDINGS_PATH = (GCS_AUX_DIR + "/glove_doc_embeddings.pkl") if USE_GCS_PATHS else (AUX_DIR / "glove_doc_embeddings.pkl")
GLOVE_VECTORS_PATH = None  # Path to pretrained GloVe vectors file (set this to your GloVe file path)
GLOVE_DIM = 300  # GloVe vector dimension (default: 300)

# ============================================================================
# Indexing parameters
# ============================================================================
ANCHOR_PAGES_PER_BATCH = 20000