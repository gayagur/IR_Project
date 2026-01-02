# 🔍 Wikipedia Search Engine
> **Information Retrieval Course Project** | A high-performance full-text search engine for English Wikipedia.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![GCP](https://img.shields.io/badge/Storage-GCP%20Bucket-orange.svg)

---

## 📖 Overview
This project implements a complete search pipeline for the English Wikipedia corpus. It features multi-stage ranking, efficient inverted indexing, LSI (Latent Semantic Indexing) reranking, and a Flask-based REST API for real-time querying.

---

## 🏗️ Project Structure
```text
IR_Project/
├── 🌐 search_frontend.py       # Flask web application & API endpoints
├── ⚙️ search_runtime.py         # Search engine runtime & query processing
├── ⚙️ config.py                 # Configuration & data paths
├── 🗄️ inverted_index_gcp.py    # Inverted index with GCP storage support
├── 🧪 text_processing.py       # Tokenization & preprocessing
├── 📄 parser_utils.py          # Wikipedia XML parsing utilities
├── 📂 indexing/                # Index building scripts
│   └── build_indices.py        # Build body, title, and anchor indices
├── ⚖️ ranking/                 # Ranking algorithms
│   ├── bm25.py                 # BM25 ranking
│   ├── lsi.py                  # LSI (Latent Semantic Indexing)
│   ├── merge.py                # Ranking fusion
│   └── tfidf_cosine.py         # TF-IDF cosine similarity
├── 📊 experiments/             # Evaluation & tuning scripts
│   ├── evaluate.py             # Core evaluation metrics (MAP@K, AP@K)
│   ├── run_evaluation.py       # Main evaluation script
│   ├── bm25_tuning.py          # BM25 parameter tuning
│   ├── weight_tuning.py        # Ranking weight optimization
│   └── compare_versions.py    # Version comparison & visualization
├── 📋 queries_train.json       # Training queries with relevance judgments
└── 📝 requirements.txt         # Python dependencies
```

---

## 🚀 Key Components

### 1. Search Frontend (`search_frontend.py`)

Provides multiple endpoints for different search strategies:

* **`/search`**: 🏆 **Main Engine** - Combines BM25, Title, Anchor, LSI reranking, PageRank, and Pageviews.
* **`/search_with_weights`**: Custom weight configuration for fine-tuning.
* **`/search_body`**: TF-IDF Cosine similarity on article text.
* **`/search_title`**: Binary ranking based on article titles.
* **`/search_anchor`**: Ranking based on incoming link text.
* **`/search_lsi`**: LSI-only search (for testing).

### 2. Search Runtime (`search_runtime.py`)

Core search engine implementation:
* **Multi-signal ranking**: BM25, Title, Anchor, and LSI
* **LSI reranking**: Optimized reranking on top-K results
* **Weighted fusion**: Configurable weights for each signal
* **PageRank & PageView boosting**: Quality signals integration

### 3. Ranking Algorithms

| Algorithm | Module | Description |
| --- | --- | --- |
| **BM25** | `ranking/bm25.py` | Probabilistic ranking function |
| **LSI** | `ranking/lsi.py` | Latent Semantic Indexing with reranking |
| **TF-IDF** | `ranking/tfidf_cosine.py` | Vector space model |
| **Merge** | `ranking/merge.py` | Weighted ranking fusion |

### 4. Indexing Engine (`indexing/build_indices.py`)

Builds three specialized indices for fast retrieval:

| Index Type | Description | Weight in `/search` |
| --- | --- | --- |
| **Body Index** | Full-text content using BM25 | Primary (configurable) |
| **Title Index** | Exact and partial title matching | 0.35 (configurable) |
| **Anchor Index** | Text from incoming Wikipedia links | 0.25 (configurable) |

### 5. LSI Configuration

* **LSI Reranking**: Only reranks top-K results (default: 100) for efficiency
* **Configurable**: `LSI_TOP_K` in `config.py` controls reranking depth
* **Weight control**: Set `LSI_WEIGHT = 0.0` to disable LSI entirely

---

## 🛠️ Usage

### 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 🔨 Building Indices

To process the Wikipedia dump and build the indices locally or on GCP:

```bash
python -m indexing.build_indices --dump path/to/enwiki-latest.xml.bz2 --build all
```

### 💻 Running the Search Engine

```bash
python search_frontend.py
```

The server will start at `http://127.0.0.1:8050` (local) or `http://0.0.0.0:8080` (production).

### 🔍 Example Queries

```bash
# Main search (Weighted Fusion)
curl "http://localhost:8080/search?query=artificial+intelligence"

# Search with custom weights
curl "http://localhost:8080/search_with_weights?query=python&body_weight=1.0&title_weight=0.5&lsi_weight=0.25"

# Title-only search
curl "http://localhost:8080/search_title?query=python+programming"
```

### ⚙️ Configuration

Edit `config.py` to customize:
* **Ranking weights**: `BODY_WEIGHT`, `TITLE_WEIGHT`, `ANCHOR_WEIGHT`, `LSI_WEIGHT`
* **LSI settings**: `LSI_TOP_K` (number of results to rerank)
* **BM25 parameters**: `BM25_K1`, `BM25_B`
* **Boost weights**: `PAGERANK_BOOST`, `PAGEVIEW_BOOST`

---

## 📈 Evaluation

The `experiments/` directory includes comprehensive evaluation tools:

### Metrics
* **MAP@K**: Mean Average Precision at K
* **Precision@K**: Precision at K
* **Recall@K**: Recall at K
* **F1@K**: F1 score at K
* **Harmonic Mean**: Combined metric (Precision@5, F1@30)

### Scripts

```bash
# Run evaluation with default weights
python experiments/run_evaluation.py

# Tune BM25 parameters
python experiments/bm25_tuning.py --queries queries_train.json

# Optimize ranking weights
python experiments/weight_tuning.py

# Compare different configurations
python experiments/compare_versions.py
```

### Results
* Evaluation results are saved in `experiments/*_tuning_results/`
* Visualizations (graphs, heatmaps) are generated automatically
* JSON results files contain detailed metrics

---

## 📂 Data Requirements

Ensure the following structure in your project root:

```text
IR_Project/
├── queries_train.json          # Training queries (required for evaluation)
├── data/                       # Optional: local data files
├── indices/                    # Inverted index files (generated)
└── aux/                        # Auxiliary files (generated)
    ├── doc_norms.pkl           # TF-IDF norms
    ├── doc_len.pkl             # Document lengths
    ├── avgdl.txt               # Average document length
    ├── titles.pkl              # Document titles mapping
    ├── pagerank.pkl            # PageRank scores
    ├── pageviews.pkl           # Page view counts
    └── lsi/                    # LSI index files (optional)
        ├── lsi_vectors.pkl
        ├── svd_components.pkl
        ├── term_to_idx.pkl
        └── doc_to_idx.pkl
```

---

## 📦 Dependencies

Core dependencies (see `requirements.txt`):
* **Flask**: Web framework
* **pandas, numpy**: Data processing
* **scikit-learn**: LSI implementation
* **google-cloud-storage**: GCP integration
* **mwparserfromhell**: Wikipedia parsing

Evaluation dependencies:
* **requests**: HTTP client for testing
* **matplotlib**: Visualization

---

## 🎯 Features

* ✅ **Multi-signal ranking**: BM25, Title, Anchor, LSI
* ✅ **LSI reranking**: Efficient top-K reranking
* ✅ **Configurable weights**: Easy parameter tuning
* ✅ **PageRank & PageView integration**: Quality signals
* ✅ **GCP support**: Cloud storage integration
* ✅ **Comprehensive evaluation**: Multiple metrics and visualization
* ✅ **Optimized performance**: Vectorized operations, lazy loading

---

**Developed as part of the Information Retrieval course project.**
