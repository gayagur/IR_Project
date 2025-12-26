
# 🔍 Wikipedia Search Engine
> **Information Retrieval Course Project** | A high-performance full-text search engine for English Wikipedia.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![GCP](https://img.shields.io/badge/Storage-GCP%20Bucket-orange.svg)

---

## 📖 Overview
This project implements a complete search pipeline for the English Wikipedia corpus. It features multi-stage ranking, efficient inverted indexing, and a Flask-based REST API for real-time querying.

---

## 🏗️ Project Structure
```text
IR_Project/
├── 🌐 search_frontend.py       # Flask web application & API endpoints
├── ⚙️ config.py                # Configuration & Data paths
├── 🗄️ inverted_index_gcp.py    # Inverted index with GCP storage support
├── 🧪 text_processing.py       # Tokenization & Preprocessing
├── 📂 indexing/                # Index building scripts (Body, Title, Anchor)
├── ⚖️ ranking/                 # BM25, TF-IDF, and Ranking Fusion
└── 📊 experiments/             # Evaluation metrics (MAP@K, AP@K)

```

---

## 🚀 Key Components

### 1. Search Frontend (`search_frontend.py`)

Provides multiple endpoints for different search strategies:

* **`/search`**: 🏆 **Main Engine** - Combines BM25, Title, Anchor, PageRank, and Pageviews.
* **`/search_body`**: TF-IDF Cosine similarity on article text.
* **`/search_title`**: Binary ranking based on article titles.
* **`/search_anchor`**: Ranking based on incoming link text.
* **`/search_pagerank`**: Ranking by authority scores.

### 2. Indexing Engine

Builds three specialized indices to ensure fast retrieval:

| Index Type | Description | Weight in `/search` |
| --- | --- | --- |
| **Body Index** | Full-text content using BM25 | Primary |
| **Title Index** | Exact and partial title matching | 0.4 |
| **Anchor Index** | Text from incoming Wikipedia links | 0.2 |

### 3. Ranking & Optimization

* **BM25 & TF-IDF**: Advanced probabilistic and vector space models.
* **Feature Boosting**: Integration of **PageRank** and **Pageviews** as quality signals.
* **Efficiency**:
* Binary posting lists with fixed-size tuples.
* Lazy loading of indices to minimize memory footprint.
* Query term limiting for consistent performance.



---

## 🛠️ Usage

### 📦 Building Indices

To process the Wikipedia dump and build the indices locally or on GCP:

```bash
python -m indexing.build_indices --dump path/to/enwiki-latest.xml.bz2 --build all

```

### 💻 Running the Search Engine

```bash
python search_frontend.py

```

The server will start at `http://0.0.0.0:8080`

### 🔍 Example Queries

```bash
# Main search (Weighted Fusion)
curl "http://localhost:8080/search?query=artificial+intelligence"

# Title-only search
curl "http://localhost:8080/search_title?query=python+programming"

```

---

## 📈 Evaluation

The `experiments/` directory includes tools for measuring:

* **Metrics**: MAP@10, Precision@5, F1@30.
* **Latencies**: Tracking mean and max query processing time.
* **Comparison**: Scripts to visualize performance between different versions.

---

## 📂 Data Requirements

Ensure the following structure in your project root:

```text
data/
├── indices/            # Inverted index files
└── aux/
    ├── doc_norms.pkl   # TF-IDF norms
    ├── doc_len.pkl     # Document lengths
    ├── pagerank.pkl    # PageRank scores
    └── pageviews.pkl   # Page view counts

```

---

## 📦 Dependencies

* **Core**: Flask, mwparserfromhell, google-cloud-storage.
* **Evaluation**: requests, matplotlib, numpy.

---

**Developed as part of the Information Retrieval course project.**

```
