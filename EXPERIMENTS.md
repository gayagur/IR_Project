# Experiments and Evaluation Report

This document describes the experiments conducted to develop and optimize the Wikipedia search engine.

## Overview

The search engine implements multiple ranking methods and combines them using weighted fusion. This document describes the evolution of the system and the experiments conducted to optimize performance.

## System Architecture

### Base Components

1. **Body Index**: Full-text inverted index of article content
2. **Title Index**: Inverted index of article titles
3. **Anchor Index**: Inverted index of anchor text from incoming links
4. **Auxiliary Data**: PageRank scores, pageview counts, document norms

### Ranking Methods Implemented

1. **TF-IDF Cosine Similarity** (`/search_body`)
   - Standard cosine similarity with TF-IDF weighting
   - Uses document norms for normalization
   - Baseline method for content-based retrieval

2. **BM25** (used in main `/search` endpoint)
   - More effective than TF-IDF for ad-hoc retrieval
   - Parameters: k1=1.5, b=0.75 (standard values)
   - Better handling of term frequency saturation

3. **Binary Title Ranking** (`/search_title`)
   - Binary matching: document gets score 1.0 for each distinct query term in title
   - Simple but effective for exact/partial title matches

4. **Binary Anchor Ranking** (`/search_anchor`)
   - Sum of term frequencies in anchor text
   - Captures how documents are referenced by others

5. **PageRank Ranking** (`/search_pagerank`)
   - Ranks documents by PageRank scores
   - Filters candidates using title matching first

6. **Pageview Ranking** (`/search_pageview`)
   - Ranks documents by pageview counts
   - Filters candidates using title matching first

## Experiments Conducted

### Experiment 1: BM25 vs TF-IDF

**Hypothesis**: BM25 should outperform TF-IDF for ad-hoc retrieval.

**Method**: 
- Implemented both BM25 and TF-IDF cosine
- Compared on training queries

**Results**:
- BM25 showed better performance on average
- BM25 chosen as primary body ranking method in `/search` endpoint

**Conclusion**: BM25 is used as the default body ranking method, with TF-IDF as fallback.

### Experiment 2: Ranking Fusion

**Hypothesis**: Combining multiple signals (body, title, anchor) should improve retrieval quality.

**Method**:
- Tested different weight combinations:
  - Version 1: body=1.0, title=0.3, anchor=0.1
  - Version 2: body=1.0, title=0.4, anchor=0.2
  - Version 3: body=1.0, title=0.5, anchor=0.3

**Results**:
- Version 2 (body=1.0, title=0.4, anchor=0.2) showed best balance
- Title boost helps for exact matches
- Anchor boost adds diversity

**Conclusion**: Final weights: body=1.0, title=0.4, anchor=0.2

### Experiment 3: Feature Boosting

**Hypothesis**: Adding PageRank and pageviews as features should improve ranking of popular/relevant documents.

**Method**:
- Tested different boost weights:
  - PageRank: 0.05, 0.1, 0.15
  - Pageviews: 0.05, 0.1, 0.15

**Results**:
- Small boosts (0.1 each) improved results without over-weighting popularity
- Larger boosts hurt precision for specific queries

**Conclusion**: Final weights: PageRank=0.1, Pageviews=0.1

### Experiment 4: Query Processing

**Hypothesis**: Limiting query terms and applying stopword removal should improve efficiency.

**Method**:
- Tested with/without stopword removal
- Tested different max_terms limits: 30, 50, 100

**Results**:
- Stopword removal essential for efficiency
- Max 50 terms provides good balance between coverage and speed
- Very long queries (>100 terms) are rare and slow

**Conclusion**: 
- Stopword removal: Enabled
- Max query terms: 50

### Experiment 5: Top-N Retrieval

**Hypothesis**: Retrieving more candidates before merging should improve final results.

**Method**:
- Tested different top_n values for individual rankings: 100, 150, 200
- Tested different top_n for merged ranking: 100, 150

**Results**:
- Retrieving top 200 from each method before merging improved recall
- Merging top 150 before final selection improved diversity

**Conclusion**: 
- Individual rankings: top_n=200
- Merged ranking: top_n=150
- Final output: top_n=100

## Performance Optimizations

### Indexing Optimizations

1. **Multi-file Storage**: Posting lists split into ~2MB files to avoid memory issues
2. **Binary Format**: Fixed-size tuples (6 bytes) for doc_id+tf for fast reading
3. **Lazy Loading**: Posting lists read on-demand, not all at once

### Query Processing Optimizations

1. **Term Limiting**: Max 50 query terms to prevent excessive processing
2. **Early Termination**: Stop processing if no results found
3. **Efficient Merging**: Use dictionaries for O(1) lookups during ranking fusion

### Memory Optimizations

1. **Posting List Streaming**: Read posting lists one at a time
2. **Sparse Storage**: Only store non-zero values in auxiliary data structures
3. **Pickle Compression**: Use highest protocol for smaller files

## Evaluation Metrics

### Metrics Used

1. **Mean Average Precision (MAP)**: Primary quality metric
   - MAP@10: Average precision at rank 10
   - MAP@5: Average precision at rank 5

2. **Harmonic Mean of Precision@5 and F1@30**: Competition metric
   - Combines early precision with recall

3. **Query Time**: Efficiency metric
   - Mean, min, max query processing time

### Training Set Evaluation

- **Dataset**: queries_train.json (30 queries with relevance judgments)
- **Evaluation**: Run `experiments/run_evaluation.py` to measure performance

### Expected Performance

Based on experiments:
- **MAP@10**: Target > 0.15 (minimum requirement: > 0.1)
- **Mean Query Time**: Target < 1 second (for full points: < 1s)
- **Harmonic Mean**: Target competitive with other submissions

## Future Improvements (Not Implemented)

The following techniques were considered but not implemented due to time constraints:

1. **Stemming**: Could improve recall for morphological variants
2. **Query Expansion**: Could help with synonym matching
3. **Learning to Rank**: Could learn optimal weights from training data
4. **Embeddings**: Could use word/document embeddings for semantic matching
5. **Caching**: Not allowed per requirements, but would improve speed

## Version History

### Version 1.0 (Initial)
- Basic TF-IDF cosine similarity
- Simple title and anchor matching
- No feature boosting

### Version 2.0 (Current)
- BM25 as primary ranking
- Weighted fusion of body, title, anchor
- PageRank and pageview boosting
- Optimized query processing

## Running Experiments

To reproduce experiments:

```bash
# Evaluate all endpoints
python experiments/run_evaluation.py --base-url http://localhost:8080 --queries queries_train.json

# Evaluate specific endpoints
python experiments/run_evaluation.py --endpoints search search_body
```

## Results Storage

Evaluation results are saved to `evaluation_results.json` with:
- Metrics for each endpoint
- Predictions for each query
- Timing information

Use these results to:
- Compare different versions
- Identify problematic queries
- Optimize weights and parameters


