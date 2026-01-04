# 🔍 Wikipedia Search Engine

> **Information Retrieval Course Project** | A high-performance full-text search engine for English Wikipedia (6.3M articles)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![GCP](https://img.shields.io/badge/Cloud-Google%20Cloud%20Platform-orange.svg)
![Wikipedia](https://img.shields.io/badge/Corpus-Wikipedia%202021-green.svg)

---
## 👥 Authors

<div align="center">

### ✨ Project Creators ✨

  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/%20%20👑%20%20Gaya%20Gur%20%20👑%20%20-800020?style=for-the-badge&labelColor=800020" height="100" alt="Gaya Gur"/>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/%20%20👑%20%20Matias%20Guernik%20%20👑%20%20-800020?style=for-the-badge&labelColor=800020" height="100" alt="Matias Guernik"/>
    </td>
  </tr>


<br/>
<p align="center">
  <img src="assets/team.png" width="260" />
</p>


<img src="https://img.shields.io/badge/📚%20Information%20Retrieval-FF8C00?style=flat&labelColor=FF8C00" height="22"/>
&nbsp;
<img src="https://img.shields.io/badge/🎓%202025%2F2026-FF8C00?style=flat&labelColor=FF8C00" height="22"/>

</div>
---



## 📖 Overview

A complete search pipeline for the English Wikipedia corpus featuring:
- **Multi-signal ranking** combining text relevance, link analysis, and popularity metrics
- **BM25 probabilistic ranking** with tuned parameters
- **LSI reranking** on top-K results for semantic similarity
- **6.3M documents** indexed across body, title, and anchor text
- **Sub-second query latency** with lazy index loading
- **RESTful API** for easy integration

---

## 🏗️ Architecture

<br/>
<p align="center">
  <img src="assets/build.png" width="700" />
</p>
---

## 📁 Project Structure
```
IR_Project/
├── search_frontend.py        # Flask REST API
├── search_runtime.py         # Search engine core logic
├── config.py                 # Configuration & weights
├── inverted_index_gcp.py     # Inverted index with GCP support
├── text_processing.py        # Tokenization & stemming
├── parser_utils.py           # Wikipedia XML parsing utilities
│
├── indexing/
│   └── build_indices.py      # Index construction pipeline
│
├── ranking/
│   ├── bm25.py               # BM25 implementation
│   ├── lsi.py                # LSI (Latent Semantic Indexing)
│   ├── tfidf_cosine.py       # TF-IDF cosine similarity
│   └── merge.py              # Score fusion
│
└── experiments/
    ├── evaluate.py           # MAP@K, Precision, Recall metrics
    ├── run_evaluation.py     # Main evaluation script
    ├── bm25_tuning.py        # BM25 parameter optimization
    ├── weight_tuning.py      # Multi-signal weight optimization
    └── compare_versions.py   # Version comparison & visualization
```

---
<br/>
<p align="center">
  <img src="assets/unnamed.png" width="560" />
</p>


## 🚀 API Endpoints

### Main Search (Recommended)
```bash
GET /search?query=<query>
```
Multi-signal fusion combining all ranking signals with LSI reranking.

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
| `/search` | GET | 🏆 Main engine - BM25 + Title + Anchor + LSI + PageRank + PageView |
| `/search_body` | GET | TF-IDF Cosine similarity on article body |
| `/search_title` | GET | Binary title matching |
| `/search_anchor` | GET | Binary anchor text search |
| `/search_lsi` | GET | LSI-only search (for testing) |
| `/search_with_weights` | GET | Custom weight configuration |
| `/get_pagerank` | POST | Get PageRank scores for doc IDs |
| `/get_pageview` | POST | Get page view counts for doc IDs |

### Custom Weight Search
```bash
GET /search_with_weights?query=<query>&body_weight=1.0&title_weight=0.35&anchor_weight=0.25&lsi_weight=0.25&pagerank_boost=0.15
```

### PageRank & PageView Endpoints

#### Get PageRank Scores
Returns PageRank values for a list of Wikipedia article IDs.

**Endpoint:** `POST /get_pagerank`

**Request:**
```bash
curl -X POST "http://<SERVER_IP>:8080/get_pagerank" \
  -H "Content-Type: application/json" \
  -d '[12345, 67890, 11111]'
```

**Response:**
```json
[0.000123, 0.000456, 0.000789]
```

**Python Example:**
```python
import requests

# Get PageRank for specific documents
doc_ids = [12345, 67890, 11111]
response = requests.post(
    "http://<SERVER_IP>:8080/get_pagerank",
    json=doc_ids
)
pagerank_scores = response.json()
# Returns: [0.000123, 0.000456, 0.000789]
```

#### Get PageView Counts
Returns the number of page views that each Wikipedia article had in August 2021.

**Endpoint:** `POST /get_pageview`

**Request:**
```bash
curl -X POST "http://<SERVER_IP>:8080/get_pageview" \
  -H "Content-Type: application/json" \
  -d '[12345, 67890, 11111]'
```

**Response:**
```json
[15234, 8921, 4567]
```

**Python Example:**
```python
import requests

# Get page views for specific documents
doc_ids = [12345, 67890, 11111]
response = requests.post(
    "http://<SERVER_IP>:8080/get_pageview",
    json=doc_ids
)
pageview_counts = response.json()
# Returns: [15234, 8921, 4567]
```

**Note:** If a document ID is not found, the endpoint returns `0.0` for PageRank or `0` for PageView.

---

## ⚙️ Ranking Algorithms

### BM25 Scoring (Main Search - Body Component)
```
score(D, Q) = Σ IDF(qi) · (tf(qi, D) · (k1 + 1)) / (tf(qi, D) + k1 · (1 - b + b · |D|/avgdl))
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `k1` | Term frequency saturation | 3.0 |
| `b` | Document length normalization | 0.25 |

### LSI (Latent Semantic Indexing)
LSI reranking on top-K results for semantic similarity:
- Projects queries and documents into latent semantic space
- Uses TruncatedSVD with 100 components
- Only reranks top-K results (default: 100) for efficiency
- Can be disabled by setting `LSI_WEIGHT = 0.0`

**Configuration:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `LSI_TOP_K` | Number of results to rerank | 100 |
| `LSI_WEIGHT` | LSI weight in fusion | 0.0 |
| `LSI_N_COMPONENTS` | Latent dimensions | 100 |

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
# Step 1: Initial ranking (without LSI)
initial_score = (
    body_weight * BM25_body(q, d) +
    title_weight * binary_title(q, d) +
    anchor_weight * binary_anchor(q, d)
)

# Step 2: Add PageRank and PageView boosts (normalized)
pr_normalized = pagerank(d) / max_pagerank_in_candidates
pv_normalized = pageviews(d) / max_pageviews_in_candidates
boosted_score = initial_score + pagerank_boost * pr_normalized + pageview_boost * pv_normalized

# Step 3: LSI reranking on top-K
if LSI_WEIGHT > 0:
    top_k_results = get_top_k(boosted_score, k=LSI_TOP_K)
    lsi_score = LSI_similarity(q, d)
    final_score = blend(boosted_score, lsi_score, LSI_WEIGHT)
```


**Default Weights:**
| Signal | Weight | Method |
|--------|--------|--------|
| Body | 0.4 | BM25 |
| Title | 0.75 | Binary |
| Anchor | 1.0 | Binary |
| LSI | 0.0 | Reranking (top-K) |
| PageRank | 0.15 | Log boost |
| PageView | 0.10 | Log boost |

---

## 📊 Index Statistics

| Index | Terms | Documents | Size |
|-------|-------|-----------|------|
| Body | 28M | 6.3M | ~15 GB |
| Title | 1.7M | 6.3M | ~500 MB |
| Anchor | 2.4M | 5.8M | ~1.1 GB |
| LSI | 50K | 6.3M | ~500 MB (optional) |
| PageRank | - | 6.3M | ~50 MB |
| PageViews | - | 10.7M | ~100 MB |

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
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

### Version Comparison
```bash
python experiments/compare_versions.py --base-url http://<SERVER_IP>:8080
```
Compares different search engine configurations and generates comparison visualizations.

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
    ├── avgdl.txt       # Average document length
    ├── pagerank.pkl    # PageRank scores (6.3M entries)
    ├── pageviews.pkl   # Page view counts
    ├── titles.pkl      # doc_id → title mapping
    └── lsi/            # LSI index files (optional)
        ├── lsi_vectors.pkl
        ├── svd_components.pkl
        ├── term_to_idx.pkl
        └── doc_to_idx.pkl
```

---

## 🔧 Configuration

Edit `config.py` to customize:
```python
# Index paths
BODY_INDEX_DIR = "indices/body"
TITLE_INDEX_DIR = "indices/title"
ANCHOR_INDEX_DIR = "indices/anchor"

# BM25 parameters
BM25_K1 = 3.0
BM25_B = 0.25

# LSI configuration
LSI_TOP_K = 100          # Number of results to rerank
LSI_WEIGHT = 0.0        # LSI weight (0.0 to disable)- we decided to not use LSI but you can change it
LSI_N_COMPONENTS = 100   # Latent dimensions

# Ranking weights
BODY_WEIGHT = 0.4
TITLE_WEIGHT = 0.75
ANCHOR_WEIGHT = 1.0
LSI_WEIGHT = 0.0
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
| Average Query Latency | ~2.0s |
| Index Load Time | ~2 min |
| Memory Usage | ~8 GB |
| LSI Reranking Time | ~50ms (on top-100) |

---

## 🧪 Example Queries

> Replace `<SERVER_IP>` with your instance IP (e.g., `104.198.58.119`)
```bash
# Main search (BM25 + all signals + LSI reranking)
curl "http://<SERVER_IP>:8080/search?query=machine+learning"

# Body search (TF-IDF Cosine)
curl "http://<SERVER_IP>:8080/search_body?query=artificial+intelligence"

# Title search (Binary)
curl "http://<SERVER_IP>:8080/search_title?query=python+programming"

# LSI-only search
curl "http://<SERVER_IP>:8080/search_lsi?query=deep+learning"

# Custom weights
curl "http://<SERVER_IP>:8080/search_with_weights?query=deep+learning&title_weight=3.0&body_weight=0.5&lsi_weight=0.3"

# Get PageRank for documents
curl -X POST "http://<SERVER_IP>:8080/get_pagerank" \
  -H "Content-Type: application/json" \
  -d '[12345, 67890, 11111]'

# Get PageViews for documents
curl -X POST "http://<SERVER_IP>:8080/get_pageview" \
  -H "Content-Type: application/json" \
  -d '[12345, 67890, 11111]'
```

### Live Examples (Current Deployment)
```bash
# Main search (BM25 + all signals + LSI reranking)
curl "http://104.198.58.119:8080/search?query=machine+learning"

# Body search (TF-IDF Cosine)
curl "http://104.198.58.119:8080/search_body?query=artificial+intelligence"

# Title search (Binary)
curl "http://104.198.58.119:8080/search_title?query=python+programming"

# Anchor search (Binary)
curl "http://104.198.58.119:8080/search_anchor?query=united+states"

# LSI-only search
curl "http://104.198.58.119:8080/search_lsi?query=deep+learning"

# Custom weights
curl "http://104.198.58.119:8080/search_with_weights?query=deep+learning&title_weight=3.0&body_weight=0.5&lsi_weight=0.3&pagerank_boost=0.2&pageview_boost=0.15"

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
- Deerwester, S., et al. (1990). *Indexing by Latent Semantic Analysis*

---

## 📄 License

This project is for educational purposes only.
