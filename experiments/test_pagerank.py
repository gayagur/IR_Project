# test_pagerank.py - Comprehensive PageRank testing
import requests
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import config
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Get BASE_URL from config
try:
    import config
    BASE_URL = getattr(config, 'BASE_URL', 'http://localhost:8080')
except ImportError:
    BASE_URL = 'http://localhost:8080'

# Import evaluation functions
try:
    from evaluate import average_precision_at_k, mean_ap_at_k, load_queries_train
except ImportError:
    # Fallback definitions
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
        import json
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

def test_pagerank_endpoint():
    """Test if /get_pagerank endpoint works."""
    print("=" * 60)
    print("Test 1: Testing /get_pagerank endpoint")
    print("=" * 60)
    
    # Test with some known doc IDs from test_queries.json
    test_ids = [42179, 5208803, 7955, 16289, 23295]
    
    try:
        response = requests.post(f"{BASE_URL}/get_pagerank", json=test_ids, timeout=10)
        if response.status_code == 200:
            pr_values = response.json()
            print(f"✓ Endpoint works!")
            print(f"Requested IDs: {test_ids}")
            print(f"PageRank values: {pr_values}")
            
            # Check if there are non-zero values
            non_zero = [pr for pr in pr_values if pr > 0]
            print(f"\nNon-zero PageRank values: {len(non_zero)}/{len(test_ids)}")
            
            if non_zero:
                print(f"Max PageRank: {max(pr_values):.6f}")
                print(f"Min PageRank (non-zero): {min(non_zero):.6f}")
                return True
            else:
                print("⚠ Warning: All PageRank values are 0!")
                return False
        else:
            print(f"✗ Error {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pagerank_in_search():
    """Test if PageRank affects search results."""
    print("\n" + "=" * 60)
    print("Test 2: Testing if PageRank affects search results")
    print("=" * 60)
    
    query = "Mount Everest climbing expeditions"
    
    try:
        # Get search results
        response = requests.get(f"{BASE_URL}/search", params={"query": query}, timeout=30)
        if response.status_code != 200:
            print(f"✗ Search failed: {response.status_code}")
            return False
        
        results = response.json()
        if not results:
            print("⚠ No results returned")
            return False
        
        print(f"Query: '{query}'")
        print(f"Total results: {len(results)}")
        
        # Get PageRank for top results
        doc_ids = [int(doc_id) for doc_id, _ in results[:20]]
        pr_response = requests.post(f"{BASE_URL}/get_pagerank", json=doc_ids, timeout=10)
        
        if pr_response.status_code == 200:
            pr_values = pr_response.json()
            
            print(f"\nTop 10 results with PageRank:")
            print("-" * 80)
            for i, ((doc_id, title), pr) in enumerate(zip(results[:10], pr_values[:10]), 1):
                print(f"{i:2d}. [{doc_id:8d}] PR: {pr:8.6f} | {title[:60]}")
            
            # Check correlation between position and PageRank
            if len(pr_values) >= 10:
                avg_pr_top5 = sum(pr_values[:5]) / 5
                avg_pr_bottom5 = sum(pr_values[5:10]) / 5
                
                print(f"\nAverage PageRank - Top 5: {avg_pr_top5:.6f}")
                print(f"Average PageRank - Next 5: {avg_pr_bottom5:.6f}")
                
                if avg_pr_top5 > avg_pr_bottom5:
                    print("✓ PageRank seems to be helping (top results have higher PR)")
                    return True
                else:
                    print("⚠ PageRank might not be helping much (top results don't have higher PR)")
                    return False
            else:
                print("⚠ Not enough results to analyze")
                return False
        else:
            print(f"✗ Failed to get PageRank: {pr_response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pagerank_file():
    """Test if pagerank.pkl file exists and is valid."""
    print("\n" + "=" * 60)
    print("Test 3: Checking pagerank.pkl file locally")
    print("=" * 60)
    
    try:
        import sys
        from pathlib import Path
        
        # Add parent directory to path to import config
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        import config
        import pickle
        
        pr_path = Path(config.PAGERANK_PATH)
        print(f"Checking: {pr_path}")
        
        if pr_path.exists():
            with open(pr_path, 'rb') as f:
                pr = pickle.load(f)
            
            print(f"✓ File exists")
            print(f"Number of entries: {len(pr):,}")
            
            if pr:
                sample = list(pr.items())[:5]
                print(f"Sample entries:")
                for doc_id, pr_val in sample:
                    print(f"  [{doc_id:8d}] PR: {pr_val:.6f}")
                
                max_pr = max(pr.values())
                min_pr = min(pr.values())
                print(f"\nMax PageRank: {max_pr:.6f}")
                print(f"Min PageRank: {min_pr:.6f}")
                
                # Count non-zero
                non_zero = sum(1 for v in pr.values() if v > 0)
                print(f"Non-zero entries: {non_zero:,}/{len(pr):,} ({100*non_zero/len(pr):.1f}%)")
                return True
            else:
                print("⚠ File is empty!")
                return False
        else:
            print(f"✗ File not found at {pr_path}")
            print("Note: This is OK if you're reading from GCS")
            return None  # Not an error if reading from GCS
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_with_queries():
    """Test metrics (MAP@10 and Harmonic Mean) with test_queries.json."""
    print("\n" + "=" * 60)
    print("Test 4: Evaluating Metrics with test_queries.json")
    print("=" * 60)
    
    test_queries_file = parent_dir / "test_queries.json"
    
    if not test_queries_file.exists():
        print(f"⚠ {test_queries_file} not found, skipping metrics test")
        return None
    
    try:
        queries, gold = load_queries_train(str(test_queries_file))
        print(f"Loaded {len(queries)} queries from {test_queries_file}")
        
        # Evaluate main search endpoint
        all_pred: Dict[str, List[int]] = {}
        print("\nQuerying search engine...")
        for i, query in enumerate(queries, 1):
            try:
                response = requests.get(f"{BASE_URL}/search", params={"query": query}, timeout=30)
                if response.status_code == 200:
                    results = response.json()
                    doc_ids = [int(doc_id) for doc_id, _ in results]
                    all_pred[query] = doc_ids
                    if i % 5 == 0:
                        print(f"  Processed {i}/{len(queries)} queries...")
                else:
                    print(f"  ⚠ Error {response.status_code} for query '{query}'")
                    all_pred[query] = []
            except Exception as e:
                print(f"  ⚠ Error for query '{query}': {e}")
                all_pred[query] = []
        
        if not all_pred:
            print("✗ No predictions to evaluate")
            return False
        
        # Compute MAP@10
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
        
        return True
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("PageRank Testing Script")
    print("=" * 60)
    
    results = []
    
    # Test 1: Endpoint
    results.append(("Endpoint", test_pagerank_endpoint()))
    
    # Test 2: Impact on search
    results.append(("Search Impact", test_pagerank_in_search()))
    
    # Test 3: Local file
    file_result = test_pagerank_file()
    if file_result is not None:
        results.append(("Local File", file_result))
    
    # Test 4: Metrics
    metrics_result = test_metrics_with_queries()
    if metrics_result is not None:
        results.append(("Metrics Evaluation", metrics_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result for _, result in results if result is not None)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed or inconclusive")
        return 1


if __name__ == "__main__":
    sys.exit(main())
