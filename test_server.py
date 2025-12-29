"""
Simple script to test the search server.
Run this after starting the server with: python search_frontend.py
"""

import requests
import json
import sys
from typing import Dict, List

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

def main(base_url: str = "http://localhost:8080"):
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
    parser.add_argument("--url", default="http://localhost:8080", 
                       help="Base URL of the server (default: http://localhost:8080)")
    args = parser.parse_args()
    
    main(args.url)

