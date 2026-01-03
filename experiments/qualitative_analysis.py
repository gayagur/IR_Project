# experiments/qualitative_analysis.py
"""
Qualitative analysis - find best and worst performing queries
and analyze the top 10 results for each.
"""
import json
import requests
import sys
from pathlib import Path

script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import config
from experiments.evaluate import load_queries_train, average_precision_at_k

BASE_URL = "http://104.198.58.119:8080"

# Best weights from tuning
WEIGHTS = {
    'body': 0.4,
    'title': 1.0,
    'anchor': 0.75,
    'lsi': 0.0,
    'pagerank': 0.15,
    'pageview': 0.10,
}


def query_search(query: str) -> list:
    """Query the search engine and return results with titles."""
    params = {
        'query': query,
        'body_weight': WEIGHTS['body'],
        'title_weight': WEIGHTS['title'],
        'anchor_weight': WEIGHTS['anchor'],
        'lsi_weight': WEIGHTS['lsi'],
        'pagerank_boost': WEIGHTS['pagerank'],
        'pageview_boost': WEIGHTS['pageview'],
    }
    
    try:
        resp = requests.get(f"{BASE_URL}/search_with_weights", params=params, timeout=60)
        if resp.ok:
            return resp.json()  # Returns [(doc_id, title), ...]
        return []
    except:
        return []


def analyze_all_queries(queries, gold):
    """Run all queries and calculate AP@10 for each."""
    results = []
    
    print(f"Analyzing {len(queries)} queries...\n")
    
    for i, query in enumerate(queries, 1):
        search_results = query_search(query)
        doc_ids = [int(doc_id) for doc_id, _ in search_results]
        titles = {int(doc_id): title for doc_id, title in search_results}
        
        gold_docs = gold.get(query, [])
        ap10 = average_precision_at_k(doc_ids, gold_docs, k=10)
        
        results.append({
            'query': query,
            'ap_at_10': ap10,
            'top_10_results': search_results[:10],
            'gold_docs': gold_docs,
            'doc_ids': doc_ids[:10],
        })
        
        print(f"  [{i}/{len(queries)}] AP@10={ap10:.4f} - {query[:50]}")
    
    # Sort by AP@10
    results.sort(key=lambda x: x['ap_at_10'], reverse=True)
    
    return results


def print_detailed_analysis(result, rank_type="BEST"):
    """Print detailed analysis of a query's results."""
    query = result['query']
    ap10 = result['ap_at_10']
    top_10 = result['top_10_results']
    gold_set = set(result['gold_docs'])
    
    print("\n" + "=" * 80)
    print(f"{rank_type} PERFORMING QUERY")
    print("=" * 80)
    print(f"\nQuery: \"{query}\"")
    print(f"AP@10: {ap10:.4f}")
    print(f"Relevant docs in gold standard: {len(gold_set)}")
    
    print(f"\nTop 10 Results:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Relevant':<10} {'Doc ID':<12} {'Title'}")
    print("-" * 80)
    
    hits_at_10 = 0
    for rank, (doc_id, title) in enumerate(top_10, 1):
        doc_id = int(doc_id)
        is_relevant = doc_id in gold_set
        if is_relevant:
            hits_at_10 += 1
        relevant_marker = "YES" if is_relevant else "no"
        title_display = title[:55] + "..." if len(title) > 55 else title
        print(f"{rank:<6} {relevant_marker:<10} {doc_id:<12} {title_display}")
    
    print("-" * 80)
    print(f"Relevant in top 10: {hits_at_10}/{len(top_10)}")
    print(f"Precision@10: {hits_at_10/10:.2%}")
    
    # Show gold docs not found in top 10
    top_10_ids = set(int(doc_id) for doc_id, _ in top_10)
    missed_gold = gold_set - top_10_ids
    if missed_gold and len(missed_gold) <= 10:
        print(f"\nRelevant docs NOT in top 10: {list(missed_gold)}")
    elif missed_gold:
        print(f"\nRelevant docs NOT in top 10: {len(missed_gold)} docs")
    
    return {
        'query': query,
        'ap_at_10': ap10,
        'hits_at_10': hits_at_10,
        'gold_count': len(gold_set),
        'top_10': [(int(doc_id), title, int(doc_id) in gold_set) for doc_id, title in top_10],
    }


def main():
    # Load queries
    queries_path = config.QUERIES_DIR / "queries_train.json"
    
    if not queries_path.exists():
        # Try alternative path
        queries_path = parent_dir / "test_queries.json"
    
    if not queries_path.exists():
        print(f"Error: Could not find queries file")
        return
    
    queries, gold = load_queries_train(str(queries_path))
    print(f"Loaded {len(queries)} queries from {queries_path}")
    
    # Test server
    print(f"Base URL: {BASE_URL}")
    try:
        requests.get(BASE_URL, timeout=10)
        print("Server reachable\n")
    except:
        print("Cannot connect to server")
        return
    
    # Analyze all queries
    results = analyze_all_queries(queries, gold)
    
    # Find best and worst
    best = results[0]  # Highest AP@10
    worst = results[-1]  # Lowest AP@10
    
    # Print detailed analysis
    best_analysis = print_detailed_analysis(best, "BEST")
    worst_analysis = print_detailed_analysis(worst, "WORST")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    all_ap = [r['ap_at_10'] for r in results]
    print(f"\nAP@10 Distribution:")
    print(f"  Best:   {max(all_ap):.4f}")
    print(f"  Worst:  {min(all_ap):.4f}")
    print(f"  Mean:   {sum(all_ap)/len(all_ap):.4f}")
    print(f"  Median: {sorted(all_ap)[len(all_ap)//2]:.4f}")
    
    # Count by performance level
    perfect = sum(1 for ap in all_ap if ap == 1.0)
    good = sum(1 for ap in all_ap if 0.5 <= ap < 1.0)
    medium = sum(1 for ap in all_ap if 0.2 <= ap < 0.5)
    poor = sum(1 for ap in all_ap if ap < 0.2)
    
    print(f"\nPerformance Distribution:")
    print(f"  Perfect (AP=1.0):  {perfect} queries")
    print(f"  Good (0.5-1.0):    {good} queries")
    print(f"  Medium (0.2-0.5):  {medium} queries")
    print(f"  Poor (<0.2):       {poor} queries")
    
    # Print for report
    print("\n" + "=" * 80)
    print("FOR REPORT - BEST QUERY ANALYSIS:")
    print("=" * 80)
    print(f"""
Query: "{best['query']}"
AP@10: {best['ap_at_10']:.4f}

Top 10 Results:
""")
    for rank, (doc_id, title, is_rel) in enumerate(best_analysis['top_10'], 1):
        rel = "[RELEVANT]" if is_rel else ""
        print(f"  {rank}. {title} {rel}")
    
    print("\n" + "=" * 80)
    print("FOR REPORT - WORST QUERY ANALYSIS:")
    print("=" * 80)
    print(f"""
Query: "{worst['query']}"
AP@10: {worst['ap_at_10']:.4f}

Top 10 Results:
""")
    for rank, (doc_id, title, is_rel) in enumerate(worst_analysis['top_10'], 1):
        rel = "[RELEVANT]" if is_rel else ""
        print(f"  {rank}. {title} {rel}")


if __name__ == "__main__":
    main()