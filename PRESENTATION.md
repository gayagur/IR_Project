# Wikipedia Search Engine - Project Presentation

---

## Slide 1: Project Overview

### Building a Search Engine for English Wikipedia

**Team Members**: [Your Names]  
**Course**: Information Retrieval 2025-2026

**Objectives**:
- Build a functional search engine for the entire English Wikipedia
- Support 5 different ranking methods
- Achieve high efficiency (<1s average query time)
- Optimize for quality (MAP@10 > 0.1, competitive harmonic mean)

**Key Features**:
- Full-text search over 6M+ Wikipedia articles
- Multiple ranking signals (body, title, anchor, PageRank, pageviews)
- Efficient inverted index with GCP storage support
- Flask-based web API

---

## Slide 2: System Architecture

### Core Components

**Indexing Layer**:
- Body index: Full-text inverted index (TF-IDF, BM25)
- Title index: Article title matching
- Anchor index: Link anchor text
- Auxiliary: PageRank, pageviews, document norms

**Ranking Layer**:
- Primary: BM25 on article body
- Secondary: Title matching (0.4 weight)
- Tertiary: Anchor text (0.2 weight)
- Boosting: PageRank (0.1) + Pageviews (0.1)

**Query Processing**:
- Tokenization with stopword removal
- Multi-signal ranking fusion
- Top-100 result retrieval

---

## Slide 3: Ranking Methods

### Five Ranking Methods Implemented

1. **TF-IDF Cosine** (`/search_body`)
   - Standard cosine similarity with TF-IDF weighting
   - Document normalization using L2 norms

2. **Binary Title Ranking** (`/search_title`)
   - Distinct term matching in article titles
   - Simple but effective for exact matches

3. **Binary Anchor Ranking** (`/search_anchor`)
   - Term frequency in incoming link anchor text
   - Captures document popularity/relevance

4. **PageRank Ranking** (`/search_pagerank`)
   - Ranks by PageRank scores
   - Query-filtered using title matching

5. **Pageview Ranking** (`/search_pageview`)
   - Ranks by article view counts
   - Query-filtered using title matching

**Main Search**: Combines all signals using weighted fusion

---

## Slide 4: Experiments & Optimizations

### Key Experiments

**BM25 vs TF-IDF**:
- BM25 chosen for better ad-hoc retrieval performance
- Parameters: k1=1.5, b=0.75

**Ranking Fusion Weights**:
- Tested multiple weight combinations
- Final: body=1.0, title=0.4, anchor=0.2

**Feature Boosting**:
- PageRank and pageviews boost: 0.1 each
- Small boosts improve without over-weighting popularity

**Efficiency Optimizations**:
- Multi-file posting list storage (~2MB blocks)
- Lazy loading of posting lists
- Query term limiting (max 50 terms)
- Binary format for fast I/O

---

## Slide 5: Results & Performance

### Evaluation Results

**Quality Metrics** (on training set):
- MAP@10: [To be filled after evaluation]
- MAP@5: [To be filled after evaluation]
- Harmonic Mean (P@5, F1@30): [To be filled after evaluation]

**Efficiency Metrics**:
- Mean query time: [To be filled after evaluation]
- Target: <1 second (7 points)
- All queries: <35 seconds (requirement)

**Key Achievements**:
- ✅ All 5 ranking methods implemented
- ✅ Efficient index structure
- ✅ Multi-signal ranking fusion
- ✅ GCP deployment ready

**Future Improvements**:
- Query expansion
- Learning-to-rank for weight optimization
- Semantic embeddings

---

## Appendix: Technical Details

### Index Statistics
- Total documents: ~6M articles
- Index size: [To be filled]
- Average document length: [To be filled]

### Implementation
- Language: Python 3
- Framework: Flask
- Storage: Local files + GCP Storage
- Dependencies: mwparserfromhell, google-cloud-storage

### Repository
- GitHub: [Your repo link]
- GCP Bucket: [Your bucket link]


