# experiments/compare_versions.py
"""
Compare different versions of the search engine.
Loads evaluation results and creates comparison graphs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib numpy")


def load_results(results_file: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_performance_comparison(results_files: List[str], output_dir: Path):
    """
    Create graphs comparing different versions.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create graphs without matplotlib")
        return
    
    all_data = {}
    for results_file in results_files:
        data = load_results(results_file)
        version_name = Path(results_file).stem
        all_data[version_name] = data["results"]
    
    # Graph 1: MAP@10 for each endpoint
    fig, ax = plt.subplots(figsize=(12, 6))
    
    endpoints = ["search", "search_body", "search_title", "search_anchor", "search_pagerank", "search_pageview"]
    x = np.arange(len(endpoints))
    width = 0.8 / len(all_data)
    
    for i, (version, results) in enumerate(all_data.items()):
        values = [results.get(ep, {}).get("map_at_10", 0.0) for ep in endpoints]
        ax.bar(x + i * width, values, width, label=version)
    
    ax.set_xlabel("Endpoint")
    ax.set_ylabel("MAP@10")
    ax.set_title("MAP@10 Comparison Across Versions")
    ax.set_xticks(x + width * (len(all_data) - 1) / 2)
    ax.set_xticklabels(endpoints, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "map10_comparison.png", dpi=150)
    print(f"Saved: {output_dir / 'map10_comparison.png'}")
    
    # Graph 2: Average retrieval time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (version, results) in enumerate(all_data.items()):
        values = [results.get(ep, {}).get("mean_time", 0.0) for ep in endpoints]
        ax.bar(x + i * width, values, width, label=version)
    
    ax.set_xlabel("Endpoint")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title("Average Retrieval Time Comparison")
    ax.set_xticks(x + width * (len(all_data) - 1) / 2)
    ax.set_xticklabels(endpoints, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="1s target")
    ax.axhline(y=35.0, color="orange", linestyle="--", alpha=0.5, label="35s limit")
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png", dpi=150)
    print(f"Saved: {output_dir / 'time_comparison.png'}")
    
    # Graph 3: Harmonic Mean (P@5, F1@30)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (version, results) in enumerate(all_data.items()):
        values = [results.get(ep, {}).get("harmonic_mean_p5_f1_30", 0.0) for ep in endpoints]
        ax.bar(x + i * width, values, width, label=version)
    
    ax.set_xlabel("Endpoint")
    ax.set_ylabel("Harmonic Mean (P@5, F1@30)")
    ax.set_title("Harmonic Mean Comparison (Competition Metric)")
    ax.set_xticks(x + width * (len(all_data) - 1) / 2)
    ax.set_xticklabels(endpoints, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "harmonic_mean_comparison.png", dpi=150)
    print(f"Saved: {output_dir / 'harmonic_mean_comparison.png'}")


def create_version_summary(results_files: List[str], output_file: Path):
    """Create a text summary comparing versions."""
    all_data = {}
    for results_file in results_files:
        data = load_results(results_file)
        version_name = Path(results_file).stem
        all_data[version_name] = data["results"]
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Version Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        
        endpoints = ["search", "search_body", "search_title", "search_anchor", "search_pagerank", "search_pageview"]
        
        for endpoint in endpoints:
            f.write(f"\n{endpoint}:\n")
            f.write("-" * 40 + "\n")
            for version, results in all_data.items():
                metrics = results.get(endpoint, {})
                f.write(f"  {version}:\n")
                f.write(f"    MAP@10: {metrics.get('map_at_10', 0.0):.4f}\n")
                f.write(f"    Mean Time: {metrics.get('mean_time', 0.0):.3f}s\n")
                f.write(f"    Harmonic Mean: {metrics.get('harmonic_mean_p5_f1_30', 0.0):.4f}\n")
            f.write("\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare different search engine versions")
    parser.add_argument("--results", nargs="+", required=True, help="JSON result files to compare")
    parser.add_argument("--output-dir", default="experiments/graphs", help="Output directory for graphs")
    parser.add_argument("--summary", default="experiments/version_comparison.txt", help="Text summary file")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create graphs
    plot_performance_comparison(args.results, output_dir)
    
    # Create summary
    summary_path = Path(args.summary)
    create_version_summary(args.results, summary_path)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()


