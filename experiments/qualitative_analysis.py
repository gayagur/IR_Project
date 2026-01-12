# experiments/qualitative_analysis.py
"""
Qualitative analysis - find best and worst performing queries
and analyze the top 10 results for each.
Generates professional visualizations for reports.
"""
import json
import requests
import sys
from pathlib import Path
from typing import Dict, List

script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import config
from experiments.evaluate import load_queries_train, precision_at_k

BASE_URL = "http://104.198.58.119:8080"

# Best weights from tuning
WEIGHTS = {
    'body': 0.4,
    'title': 1.0,
    'anchor': 0.75,
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
        ap10 = precision_at_k(doc_ids, gold_docs, k=10)
        
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


def create_visualizations(
    results: List[Dict],
    best_analysis: Dict,
    worst_analysis: Dict,
    output_dir: Path,
) -> None:
    """
    Create professional tables for the report.
    
    Args:
        results: List of all query results
        best_analysis: Analysis of best performing query
        worst_analysis: Analysis of worst performing query
        output_dir: Directory to save tables
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available, skipping tables")
        print("  Install with: pip install matplotlib")
        return
    
    # Set professional style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.dpi'] = 300
    
    all_ap = [r['ap_at_10'] for r in results]
    
    # =========================================================================
    # TABLE 1: All Queries Results (sorted by Precision@10)
    # =========================================================================
    print("  Creating table: All queries results...")
    
    # Calculate number of pages needed (max 30 rows per page)
    rows_per_page = 30
    num_pages = (len(results) + rows_per_page - 1) // rows_per_page
    
    for page in range(num_pages):
        start_idx = page * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(results))
        page_results = results[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Rank', 'Query', 'Precision@10', 'Top 10 Results']
        
        for idx, result in enumerate(page_results, start=start_idx + 1):
            query = result['query']
            if len(query) > 50:
                query = query[:47] + '...'
            
            # Count relevant in top 10
            top_10_ids = [int(doc_id) for doc_id, _ in result['top_10_results'][:10]]
            gold_set = set(result['gold_docs'])
            relevant_count = sum(1 for doc_id in top_10_ids if doc_id in gold_set)
            
            results_text = f"{relevant_count}/10 relevant"
            
            table_data.append([
                str(idx),
                query,
                f"{result['ap_at_10']:.4f}",
                results_text
            ])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='left', loc='center',
                        colWidths=[0.05, 0.55, 0.15, 0.25])
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Color header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white', size=10)
            cell.set_height(0.04)
        
        # Color rows by performance
        for i, result in enumerate(page_results, 1):
            ap = result['ap_at_10']
            if ap >= 0.8:
                color = '#d5f4e6'  # Green - excellent
            elif ap >= 0.5:
                color = '#d6eaf8'  # Blue - good
            elif ap >= 0.2:
                color = '#fdebd0'  # Orange - medium
            else:
                color = '#fadbd8'  # Red - poor
            
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(color)
        
        # Title
        page_suffix = f" (Page {page + 1}/{num_pages})" if num_pages > 1 else ""
        fig.suptitle(f'📊 All Queries Results - Sorted by Precision@10{page_suffix}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        filename = f'all_queries_results.png' if num_pages == 1 else f'all_queries_results_page_{page + 1}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
    
    print(f"  ✓ Saved: all_queries_results.png ({num_pages} page{'s' if num_pages > 1 else ''})")
    
    # =========================================================================
    # GRAPH 1: AP@10 Distribution Histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n, bins, patches = ax.hist(all_ap, bins=20, color='#3498db', edgecolor='white', 
                              linewidth=1.5, alpha=0.85)
    
    # Color bars by performance level
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val >= 0.8:
            patch.set_facecolor('#2ecc71')  # Green - excellent
        elif bin_val >= 0.5:
            patch.set_facecolor('#3498db')  # Blue - good
        elif bin_val >= 0.2:
            patch.set_facecolor('#f39c12')  # Orange - medium
        else:
            patch.set_facecolor('#e74c3c')  # Red - poor
    
    # Add vertical lines for statistics
    mean_ap = np.mean(all_ap)
    median_ap = np.median(all_ap)
    ax.axvline(mean_ap, color='#9b59b6', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_ap:.3f}', alpha=0.8)
    ax.axvline(median_ap, color='#e67e22', linestyle='--', linewidth=2, 
              label=f'Median: {median_ap:.3f}', alpha=0.8)
    
    ax.set_xlabel('Precision@10', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Queries', fontweight='bold', fontsize=12)
    ax.set_title('📊 Precision@10 Distribution Across All Queries', 
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'AP@10_distribution.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved: AP@10_distribution.png")
    
    # =========================================================================
    # TABLE 2: Best Query Results Table
    # =========================================================================
    print("  Creating table: Best query results...")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Rank', 'Relevant', 'Document ID', 'Title']
    
    for rank, (doc_id, title, is_rel) in enumerate(best_analysis['top_10'], 1):
        rel_text = '✓ YES' if is_rel else '✗ no'
        title_short = title[:60] + '...' if len(title) > 60 else title
        table_data.append([str(rank), rel_text, str(doc_id), title_short])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='left', loc='center',
                    colWidths=[0.08, 0.12, 0.15, 0.65])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.08)
    
    # Color relevant rows
    for i, (_, _, is_rel) in enumerate(best_analysis['top_10'], 1):
        if is_rel:
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor('#d5f4e6')
    
    # Title
    query_text = best_analysis['query']
    if len(query_text) > 70:
        query_text = query_text[:67] + '...'
    
    fig.suptitle(f'🏆 Best Performing Query Results\n"{query_text}"\nPrecision@10: {best_analysis["ap_at_10"]:.4f}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_query_results.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved: best_query_results.png")
    
    # =========================================================================
    # TABLE 3: Worst Query Results Table
    # =========================================================================
    print("  Creating table: Worst query results...")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    
    for rank, (doc_id, title, is_rel) in enumerate(worst_analysis['top_10'], 1):
        rel_text = '✓ YES' if is_rel else '✗ no'
        title_short = title[:60] + '...' if len(title) > 60 else title
        table_data.append([str(rank), rel_text, str(doc_id), title_short])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='left', loc='center',
                    colWidths=[0.08, 0.12, 0.15, 0.65])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#c0392b')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.08)
    
    # Color relevant rows
    for i, (_, _, is_rel) in enumerate(worst_analysis['top_10'], 1):
        if is_rel:
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor('#fadbd8')
    
    # Title
    query_text = worst_analysis['query']
    if len(query_text) > 70:
        query_text = query_text[:67] + '...'
    
    fig.suptitle(f'⚠️ Worst Performing Query Results\n"{query_text}"\nPrecision@10: {worst_analysis["ap_at_10"]:.4f}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'worst_query_results.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved: worst_query_results.png")
    
    # =========================================================================
    # TABLE 4: Summary Statistics Table
    # =========================================================================
    print("  Creating table: Summary statistics...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    perfect = sum(1 for ap in all_ap if ap == 1.0)
    good = sum(1 for ap in all_ap if 0.5 <= ap < 1.0)
    medium = sum(1 for ap in all_ap if 0.2 <= ap < 0.5)
    poor = sum(1 for ap in all_ap if ap < 0.2)
    
    table_data = [
        ['Metric', 'Value'],
        ['Best Precision@10', f'{max(all_ap):.4f}'],
        ['Worst Precision@10', f'{min(all_ap):.4f}'],
        ['Mean Precision@10', f'{np.mean(all_ap):.4f}'],
        ['Median Precision@10', f'{np.median(all_ap):.4f}'],
        ['Total Queries', str(len(results))],
        ['Perfect (1.0)', f'{perfect} ({perfect/len(results)*100:.1f}%)'],
        ['Good (0.5-1.0)', f'{good} ({good/len(results)*100:.1f}%)'],
        ['Medium (0.2-0.5)', f'{medium} ({medium/len(results)*100:.1f}%)'],
        ['Poor (<0.2)', f'{poor} ({poor/len(results)*100:.1f}%)'],
    ]
    
    headers = table_data[0]
    data_rows = table_data[1:]
    
    # Create table
    table = ax.table(cellText=data_rows, colLabels=headers, 
                    cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', size=12)
        cell.set_height(0.05)
    
    # Color alternating rows
    for i in range(len(data_rows)):
        color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(color)
    
    fig.suptitle('📊 Summary Statistics', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved: summary_statistics.png")


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
    
    # Create professional visualizations
    print("\n" + "=" * 80)
    print("Creating professional visualizations for report...")
    print("=" * 80)
    
    output_dir = script_dir / "qualitative_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    create_visualizations(
        results=results,
        best_analysis=best_analysis,
        worst_analysis=worst_analysis,
        output_dir=output_dir,
    )
    
    print(f"\n✓ All tables saved to: {output_dir}")
    print(f"  - all_queries_results.png (all queries sorted by Precision@10)")
    print(f"  - best_query_results.png (top 10 results for best query)")
    print(f"  - worst_query_results.png (top 10 results for worst query)")
    print(f"  - summary_statistics.png (summary statistics table)")


if __name__ == "__main__":
    main()