# 🔍 Wikipedia Search Engine

> **Information Retrieval Course Project** | A high-performance full-text search engine for English Wikipedia (6.3M articles)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![GCP](https://img.shields.io/badge/Cloud-Google%20Cloud%20Platform-orange.svg)
![Wikipedia](https://img.shields.io/badge/Corpus-Wikipedia%202021-green.svg)

---

## 📖 Overview

A complete search pipeline for the English Wikipedia corpus featuring:
- **Multi-signal ranking** combining text relevance, link analysis, and popularity metrics
- **BM25 probabilistic ranking** with tuned parameters
- **6.3M documents** indexed across body, title, and anchor text
- **Sub-second query latency** with lazy index loading
- **RESTful API** for easy integration

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask API (search_frontend.py)                │
│                                                                  │
│  /search ─────► Multi-Signal Fusion (BM25 + Title + Anchor)      │
│  /search_body ─► TF-IDF Cosine on article text                   │
│  /search_title ► Title matching                                  │
│  /search_anchor► Anchor text search                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ranking Engine                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Body    │ │  Title   │ │  Anchor  │ │ PageRank │            │
│  │  BM25    │ │  Binary  │ │  Binary  │ │  Boost   │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GCP Storage (Indices)                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ Body Index │ │Title Index │ │Anchor Index│ │ Aux Files    │  │
│  │  28M terms │ │ 1.7M terms │ │ 2.4M terms │ │ PR, PV, Norms│  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure
```
IR_Project/
├── search_frontend.py        # Flask REST API
├── search_runtime.py         # Search engine core logic
├── config.py                 # Configuration & weights
├── inverted_index_gcp.py     # Inverted index with GCP support
├── text_processing.py        # Tokenization & stemming
│
├── indexing/
│   └── build_indices.py      # Index construction pipeline
│
├── ranking/
│   ├── bm25.py               # BM25 implementation
│   ├── tfidf_cosine.py       # TF-IDF cosine similarity
│   └── merge.py              # Score fusion
│
└── experiments/
    ├── evaluate.py           # MAP@K, Precision, Recall metrics
    ├── bm25_tuning.py        # BM25 parameter optimization
    └── weight_tuning.py      # Multi-signal weight optimization
```

---

## 🚀 API Endpoints

### Main Search (Recommended)
```bash
GET /search?query=<query>
```
Multi-signal fusion combining all ranking signals.

**Response:**
```json
[
  [12345, "Article Title"],
  [67890, "Another Article"],
  ...
]
```

### Specialized Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | GET | 🏆 Main engine - BM25 + Title + Anchor + PageRank + PageView |
| `/search_body` | GET | TF-IDF Cosine similarity on article body |
| `/search_title` | GET | Binary title matching |
| `/search_anchor` | GET | Binary anchor text search |
| `/search_with_weights` | GET | Custom weight configuration |
| `/get_pagerank` | POST | Get PageRank scores for doc IDs |
| `/get_pageview` | POST | Get page view counts for doc IDs |

### Custom Weight Search
```bash
GET /search_with_weights?query=<query>&body_weight=1.0&title_weight=2.0&anchor_weight=0.75&pagerank_boost=0.15
```

---

## ⚙️ Ranking Algorithms

### BM25 Scoring (Main Search - Body Component)
```
score(D, Q) = Σ IDF(qi) · (tf(qi, D) · (k1 + 1)) / (tf(qi, D) + k1 · (1 - b + b · |D|/avgdl))
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `k1` | Term frequency saturation | 1.5 |
| `b` | Document length normalization | 0.75 |

### TF-IDF Cosine Similarity (`/search_body` Endpoint)
```
score(D, Q) = (D · Q) / (||D|| · ||Q||)
```
Where D and Q are TF-IDF weighted vectors.

### Binary Scoring (Title & Anchor)
```
score(D, Q) = number of query terms found in document
```

### Multi-Signal Fusion (`/search` Endpoint)
```python
final_score = (
    body_weight * BM25_body(q, d) +
    title_weight * binary_title(q, d) +
    anchor_weight * binary_anchor(q, d) +
    pagerank_boost * log(1 + pagerank(d)) +
    pageview_boost * log(1 + pageviews(d))
)
```

**Default Weights:**
| Signal | Weight | Method |
|--------|--------|--------|
| Body | 1.0 | BM25 |
| Title | 2.0 | Binary |
| Anchor | 0.75 | Binary |
| PageRank | 0.15 | Log boost |
| PageView | 0.10 | Log boost |

---

## 📊 Index Statistics

| Index | Terms | Documents | Size |
|-------|-------|-----------|------|
| Body | 28M | 6.3M | ~15 GB |
| Title | 1.7M | 6.3M | ~500 MB |
| Anchor | 2.4M | 5.8M | ~1.1 GB |
| PageRank | - | 6.3M | ~50 MB |
| PageViews | - | 10.7M | ~100 MB |

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install flask google-cloud-storage nltk numpy
```

### Running Locally
```bash
# Set up configuration
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

# Start server
python search_frontend.py
```

### GCP Deployment
```bash
# SSH to instance
gcloud compute ssh <instance-name> --zone=us-central1-c

# Activate environment
source ~/venv/bin/activate
cd ~/IR_Project

# Run server
nohup python search_frontend.py > ~/frontend.log 2>&1 &
```

---

## 📈 Evaluation & Tuning

### Running Evaluation
```bash
python experiments/run_evaluation.py --base-url http://<SERVER_IP>:8080
```

### BM25 Parameter Tuning
```bash
python experiments/bm25_tuning.py --base-url http://<SERVER_IP>:8080
```
Outputs heatmaps and sensitivity plots to `experiments/bm25_tuning_results/`.

### Weight Tuning
```bash
python experiments/weight_tuning.py --base-url http://<SERVER_IP>:8080
```
Tests hundreds of weight combinations and generates visualization reports.

### Metrics
- **MAP@10** - Mean Average Precision at 10
- **MAP@5** - Mean Average Precision at 5
- **Precision@5** - Precision at rank 5
- **F1@30** - F1 score at rank 30
- **Harmonic Mean** - Combined P@5 and F1@30

---

## 📂 Data Directory Structure
```
data/
├── indices/
│   ├── body/           # Body inverted index
│   │   ├── body.pkl
│   │   └── *.bin       # Posting lists
│   ├── title/          # Title inverted index
│   └── anchor/         # Anchor text inverted index
│
└── aux/
    ├── doc_norms.pkl   # TF-IDF normalization factors
    ├── doc_len.pkl     # Document lengths (for BM25)
    ├── pagerank.pkl    # PageRank scores (6.3M entries)
    ├── pageviews.pkl   # Page view counts
    └── titles.pkl      # doc_id → title mapping
```

---

## 🔧 Configuration

Edit `config.py` to customize:
```python
# Index paths
BODY_INDEX_PATH = "indices/body"
TITLE_INDEX_PATH = "indices/title"
ANCHOR_INDEX_PATH = "indices/anchor"

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Ranking weights
BODY_WEIGHT = 1.0
TITLE_WEIGHT = 2.0
ANCHOR_WEIGHT = 0.75
PAGERANK_BOOST = 0.15
PAGEVIEW_BOOST = 0.10

# Performance
MAX_QUERY_TERMS = 10
RESULTS_LIMIT = 100
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average Query Latency | ~0.5s |
| Index Load Time | ~2 min |
| Memory Usage | ~8 GB |
| Throughput | ~10 queries/sec |

---

## 🧪 Example Queries

> Replace `<SERVER_IP>` with your instance IP (e.g., `104.198.58.119`)
```bash
# Main search (BM25 + all signals)
curl "http://<SERVER_IP>:8080/search?query=machine+learning"

# Body search (TF-IDF Cosine)
curl "http://<SERVER_IP>:8080/search_body?query=artificial+intelligence"

# Title search (Binary)
curl "http://<SERVER_IP>:8080/search_title?query=python+programming"

# Custom weights
curl "http://<SERVER_IP>:8080/search_with_weights?query=deep+learning&title_weight=3.0&body_weight=0.5"
```

### Live Examples (Current Deployment)
```bash
# Main search (BM25 + all signals)
curl "http://104.198.58.119:8080/search?query=machine+learning"

# Body search (TF-IDF Cosine)
curl "http://104.198.58.119:8080/search_body?query=artificial+intelligence"

# Title search (Binary)
curl "http://104.198.58.119:8080/search_title?query=python+programming"

# Anchor search (Binary)
curl "http://104.198.58.119:8080/search_anchor?query=united+states"

# Custom weights
curl "http://104.198.58.119:8080/search_with_weights?query=deep+learning&title_weight=3.0&body_weight=0.5"

# Get PageRank for documents
curl -X POST "http://104.198.58.119:8080/get_pagerank" \
  -H "Content-Type: application/json" \
  -d '[12345, 67890, 11111]'

# Get PageViews for documents
curl -X POST "http://104.198.58.119:8080/get_pageview" \
  -H "Content-Type: application/json" \
  -d '[12345, 67890, 11111]'
```

---

## 📚 References

- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*
- Page, L., et al. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*

---

## 👥 Authors

Developed as part of the **Information Retrieval** course project.

---

## 📄 License

This project is for educational purposes only.
