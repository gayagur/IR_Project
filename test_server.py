"""
Simple script to test the search server.
Run this after starting the server with: python search_frontend.py
"""

import requests
import json
import sys
from typing import Dict, List, Tuple

# Import evaluation functions
try:
    from experiments.evaluate import average_precision_at_k, mean_ap_at_k, load_queries_train
    from pathlib import Path
    # Add parent directory to path
    script_dir = Path(__file__).parent
    if str(script_dir / "experiments") not in sys.path:
        sys.path.insert(0, str(script_dir / "experiments"))
except ImportError:
    # Fallback if experiments module not available
    def average_precision_at_k(pred: List[int], gold: List[int], k: int = 10) -> float:
        if not gold:
            return 0.0
        gold_set = set(gold)
        pred_k = pred[:k]
        hits = 0
        sum_prec = 0.0
        for i, doc_id in enumerate(pred_k, start=1):
            if doc_id in gold_set:
                hits += 1
                sum_prec += hits / i
        return sum_prec / min(len(gold_set), k)
    
    def mean_ap_at_k(all_pred: Dict[str, List[int]], all_gold: Dict[str, List[int]], k: int = 10) -> float:
        aps = []
        for q, pred in all_pred.items():
            gold = all_gold.get(q, [])
            aps.append(average_precision_at_k(pred, gold, k=k))
        return sum(aps) / max(1, len(aps))
    
    def load_queries_train(path: str) -> Tuple[List[str], Dict[str, List[int]]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            queries = [x["query"] for x in data]
            gold = {x["query"]: x["relevant_docs"] for x in data}
        elif isinstance(data, dict):
            queries = list(data.keys())
            gold = {query: [int(doc_id) for doc_id in doc_ids] for query, doc_ids in data.items()}
        else:
            raise ValueError(f"Unexpected format in {path}")
        return queries, gold


def precision_at_k(pred: List[int], gold: List[int], k: int) -> float:
    """Compute Precision@K."""
    if not gold or k == 0:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / k


def recall_at_k(pred: List[int], gold: List[int], k: int) -> float:
    """Compute Recall@K."""
    if not gold:
        return 0.0
    gold_set = set(gold)
    pred_k = pred[:k]
    hits = sum(1 for doc_id in pred_k if doc_id in gold_set)
    return hits / len(gold_set)


def f1_at_k(pred: List[int], gold: List[int], k: int) -> float:
    """Compute F1@K."""
    p = precision_at_k(pred, gold, k)
    r = recall_at_k(pred, gold, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def harmonic_mean_precision_f1(pred: List[int], gold: List[int], p_k: int = 5, f1_k: int = 30) -> float:
    """Compute harmonic mean of Precision@P_K and F1@F1_K."""
    p = precision_at_k(pred, gold, p_k)
    f1 = f1_at_k(pred, gold, f1_k)
    if p + f1 == 0:
        return 0.0
    return 2 * p * f1 / (p + f1)

def test_endpoint(base_url: str, endpoint: str, query: str) -> bool:
    """Test a single endpoint with a query."""
    url = f"{base_url}/{endpoint}"
    params = {"query": query}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            results = response.json()
            print(f"  ✓ {endpoint}: {len(results)} results")
            if results:
                print(f"    First result: {results[0]}")
            return True
        else:
            print(f"  ✗ {endpoint}: Error {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ✗ {endpoint}: Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"  ✗ {endpoint}: {e}")
        return False

def test_post_endpoint(base_url: str, endpoint: str, data: List[int]) -> bool:
    """Test a POST endpoint."""
    url = f"{base_url}/{endpoint}"
    
    try:
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            results = response.json()
            print(f"  ✓ {endpoint}: {len(results)} results")
            print(f"    Results: {results[:5]}...")  # Show first 5
            return True
        else:
            print(f"  ✗ {endpoint}: Error {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ✗ {endpoint}: Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"  ✗ {endpoint}: {e}")
        return False

def main(base_url: str = None):
    # Get default from config if not provided
    if base_url is None:
        try:
            import config
            base_url = getattr(config, 'BASE_URL', 'http://localhost:8080')
        except ImportError:
            base_url = 'http://localhost:8080'
    # Test if server is running
    print("Testing search server...")
    print(f"Base URL: {base_url}\n")
    
    try:
        response = requests.get(base_url, timeout=5)
        print("Server is responding!\n")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server!")
        print("\nPlease start the server first:")
        print("  python search_frontend.py")
        print("\nOr if running on a different port/host:")
        print("  python test_server.py --url http://YOUR_HOST:PORT")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
    
    # Test queries from test_queries.json
    test_queries_file = "test_queries.json"
    try:
        with open(test_queries_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        print("=" * 60)
        print("Testing GET endpoints with sample queries")
        print("=" * 60)
        
        # Test with first few queries
        queries_to_test = list(test_data.keys())[:3]
        
        for query in queries_to_test:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            # Test main search endpoint
            test_endpoint(base_url, "search", query)
            
            # Test other endpoints
            test_endpoint(base_url, "search_body", query)
            test_endpoint(base_url, "search_title", query)
            test_endpoint(base_url, "search_anchor", query)
        
        print("\n" + "=" * 60)
        print("Testing POST endpoints")
        print("=" * 60)
        
        # Test POST endpoints with sample doc IDs
        sample_doc_ids = [42179, 7955, 23295]
        print(f"\nTesting with doc IDs: {sample_doc_ids}")
        print("-" * 60)
        
        test_post_endpoint(base_url, "get_pagerank", sample_doc_ids)
        test_post_endpoint(base_url, "get_pageview", sample_doc_ids)
        
        # Evaluate metrics with test_queries.json
        print("\n" + "=" * 60)
        print("Evaluating Metrics with test_queries.json")
        print("=" * 60)
        
        try:
            queries, gold = load_queries_train(test_queries_file)
            print(f"Loaded {len(queries)} queries from {test_queries_file}")
            
            # Evaluate main search endpoint
            all_pred: Dict[str, List[int]] = {}
            for query in queries:
                try:
                    response = requests.get(f"{base_url}/search", params={"query": query}, timeout=30)
                    if response.status_code == 200:
                        results = response.json()
                        doc_ids = [int(doc_id) for doc_id, _ in results]
                        all_pred[query] = doc_ids
                except Exception as e:
                    print(f"  ⚠ Error for query '{query}': {e}")
                    all_pred[query] = []
            
            if all_pred:
                map_at_10 = mean_ap_at_k(all_pred, gold, k=10)
                
                # Compute harmonic mean for each query
                harmonic_means = []
                for q in queries:
                    pred = all_pred.get(q, [])
                    gold_list = gold.get(q, [])
                    hm = harmonic_mean_precision_f1(pred, gold_list, p_k=5, f1_k=30)
                    harmonic_means.append(hm)
                
                avg_harmonic_mean = sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0.0
                
                print(f"\nMetrics for /search endpoint:")
                print(f"  Average Precision@10 (MAP@10): {map_at_10:.4f}")
                print(f"  Harmonic Mean (P@5, F1@30): {avg_harmonic_mean:.4f}")
            else:
                print("  ⚠ No predictions to evaluate")
        except Exception as e:
            print(f"  ⚠ Could not evaluate metrics: {e}")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"Warning: {test_queries_file} not found. Testing with default query.")
        print("\nTesting with query: 'Mount Everest'")
        test_endpoint(base_url, "search", "Mount Everest")
        test_endpoint(base_url, "search_body", "Mount Everest")
        test_endpoint(base_url, "search_title", "Mount Everest")
        test_endpoint(base_url, "search_anchor", "Mount Everest")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the search server")
    # Get default from config
    try:
        import config
        default_url = getattr(config, 'BASE_URL', 'http://localhost:8080')
    except ImportError:
        default_url = 'http://localhost:8080'
    
    parser.add_argument("--url", default=default_url, 
                       help=f"Base URL of the server (default: {default_url})")
    args = parser.parse_args()
    
    main(args.url)

