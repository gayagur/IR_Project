# ranking/lsi.py
from __future__ import annotations

import pickle
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from inverted_index_gcp import InvertedIndex


class LSISearcher:
    """
    LSI (Latent Semantic Indexing) searcher using TruncatedSVD.
    Uses pre-computed LSI vectors, SVD components, and term/document mappings.
    """

    def __init__(
        self,
        lsi_vectors_path: str | Path,
        svd_components_path: str | Path,
        term_to_idx_path: str | Path,
        doc_to_idx_path: str | Path,
        *,
        n_components: int = 100,
    ):
        """
        Initialize LSI searcher from pre-computed files.
        
        Args:
            lsi_vectors_path: Path to LSI vectors (numpy array, shape: n_docs x n_components)
            svd_components_path: Path to SVD components (numpy array, shape: n_components x n_terms)
            term_to_idx_path: Path to term->index mapping
            doc_to_idx_path: Path to doc_id->index mapping
            n_components: Number of LSI components (must match saved vectors)
        """
        self.n_components = n_components
        
        # Load mappings
        with open(term_to_idx_path, "rb") as f:
            self.term_to_idx: Dict[str, int] = pickle.load(f)
        
        with open(doc_to_idx_path, "rb") as f:
            self.doc_to_idx: Dict[int, int] = pickle.load(f)
        
        # Reverse mapping: idx -> doc_id
        self.idx_to_doc: Dict[int, int] = {v: k for k, v in self.doc_to_idx.items()}
        
        # Load LSI vectors (documents in latent space)
        with open(lsi_vectors_path, "rb") as f:
            self.lsi_vectors = pickle.load(f)  # Shape: (n_docs, n_components)
        
        # Load SVD components (for query transformation)
        with open(svd_components_path, "rb") as f:
            self.svd_components = pickle.load(f)  # Shape: (n_components, n_terms)
        
        self.n_terms = len(self.term_to_idx)
        
    def search(
        self,
        query_tokens: List[str],
        *,
        top_n: int = 100,
        max_terms: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        Search using LSI by projecting query into latent space and computing cosine similarity.
        
        Args:
            query_tokens: List of query terms
            top_n: Number of top results to return
            max_terms: Maximum number of query terms to use
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if not query_tokens or not self.term_to_idx or self.lsi_vectors is None:
            return []
        
        # Limit query terms
        query_terms = list(dict.fromkeys(query_tokens[:max_terms]))
        
        # Build query vector in term space (binary: 1 if term in query, 0 otherwise)
        query_term_vector = np.zeros(self.n_terms)
        for term in query_terms:
            if term in self.term_to_idx:
                term_idx = self.term_to_idx[term]
                query_term_vector[term_idx] = 1.0
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_term_vector)
        if query_norm == 0:
            return []
        query_term_vector = query_term_vector / query_norm
        
        # Project query into latent space: query_latent = svd_components @ query_term_vector
        query_latent = self.svd_components @ query_term_vector  # Shape: (n_components,)
        
        # Normalize query in latent space
        query_latent_norm = np.linalg.norm(query_latent)
        if query_latent_norm == 0:
            return []
        query_latent = query_latent / query_latent_norm
        
        # Compute cosine similarity between query and all documents in latent space
        scores = {}
        for doc_idx, doc_vector in enumerate(self.lsi_vectors):
            if doc_idx not in self.idx_to_doc:
                continue
            
            doc_id = self.idx_to_doc[doc_idx]
            
            # Cosine similarity: dot product of normalized vectors
            doc_norm = np.linalg.norm(doc_vector)
            if doc_norm == 0:
                continue
            
            doc_vector_norm = doc_vector / doc_norm
            similarity = float(np.dot(query_latent, doc_vector_norm))
            scores[doc_id] = similarity
        
        # Sort by score
        res = list(scores.items())
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_n]


def build_lsi_index(
    body_index: InvertedIndex,
    body_index_dir: str,
    doc_norms: Dict[int, float],
    output_dir: str | Path,
    *,
    n_components: int = 100,
    max_terms: int = 50000,
    max_docs: int = None,
) -> None:
    """
    Build LSI index from body index.
    
    This function:
    1. Builds a term-document matrix from the body index
    2. Applies TF-IDF weighting
    3. Performs TruncatedSVD to get latent semantic space
    4. Saves LSI vectors and mappings
    
    Args:
        body_index: InvertedIndex object for body text
        body_index_dir: Directory where body index is stored
        doc_norms: Document norms (for TF-IDF)
        output_dir: Directory to save LSI index files
        n_components: Number of LSI components (latent dimensions)
        max_terms: Maximum number of terms to include (top by document frequency)
        max_docs: Maximum number of documents to include (None = all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building LSI index with {n_components} components...")
    
    # Get all documents and terms
    all_doc_ids = sorted(body_index.df.keys())  # Actually, we need doc_ids from the index
    # We need to get doc_ids from posting lists or from doc_norms
    all_doc_ids = sorted(doc_norms.keys())
    
    if max_docs is not None:
        all_doc_ids = all_doc_ids[:max_docs]
    
    print(f"Processing {len(all_doc_ids)} documents...")
    
    # Get top terms by document frequency
    term_df_pairs = sorted(body_index.df.items(), key=lambda x: x[1], reverse=True)
    if max_terms is not None:
        term_df_pairs = term_df_pairs[:max_terms]
    
    selected_terms = [term for term, _ in term_df_pairs]
    term_to_idx = {term: idx for idx, term in enumerate(selected_terms)}
    
    print(f"Selected {len(selected_terms)} terms...")
    
    # Build term-document matrix
    # This is memory-intensive, so we'll build it in chunks or use sparse representation
    from scipy.sparse import csr_matrix
    
    N = len(all_doc_ids)
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(all_doc_ids)}
    
    # Build sparse matrix: rows = documents, cols = terms
    rows = []
    cols = []
    data = []
    
    print("Building term-document matrix...")
    for term_idx, term in enumerate(selected_terms):
        if term not in body_index.df:
            continue
        
        # Get posting list for this term
        pls = body_index.read_a_posting_list(body_index_dir, term, bucket_name=None)
        
        # Compute IDF
        df = body_index.df[term]
        idf = math.log((N + 1) / (df + 1))
        
        # Add to matrix
        for doc_id, tf in pls:
            if doc_id not in doc_to_idx:
                continue
            
            doc_idx = doc_to_idx[doc_id]
            tfidf = tf * idf
            
            rows.append(doc_idx)
            cols.append(term_idx)
            data.append(tfidf)
    
    # Create sparse matrix
    td_matrix = csr_matrix((data, (rows, cols)), shape=(len(all_doc_ids), len(selected_terms)))
    
    print(f"Term-document matrix shape: {td_matrix.shape}")
    print("Applying TruncatedSVD...")
    
    # Apply TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_vectors = svd.fit_transform(td_matrix)  # Shape: (n_docs, n_components)
    
    print(f"LSI vectors shape: {lsi_vectors.shape}")
    
    # Save LSI vectors, SVD components, and mappings
    lsi_vectors_path = output_dir / "lsi_vectors.pkl"
    svd_components_path = output_dir / "svd_components.pkl"
    term_to_idx_path = output_dir / "term_to_idx.pkl"
    doc_to_idx_path = output_dir / "doc_to_idx.pkl"
    
    with open(lsi_vectors_path, "wb") as f:
        pickle.dump(lsi_vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save SVD components (for query transformation)
    svd_components = svd.components_  # Shape: (n_components, n_terms)
    with open(svd_components_path, "wb") as f:
        pickle.dump(svd_components, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(term_to_idx_path, "wb") as f:
        pickle.dump(term_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(doc_to_idx_path, "wb") as f:
        pickle.dump(doc_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"LSI index saved to {output_dir}")

