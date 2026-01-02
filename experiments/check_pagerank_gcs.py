# check_pagerank_gcs.py - Check pagerank.pkl in GCS
import sys
from pathlib import Path

# Add parent directory to path to import config
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

print("=" * 60)
print("Checking pagerank.pkl")
print("=" * 60)

# Method 1: Try to check via GCS Python API
try:
    from google.cloud import storage
    import config
    import pickle
    
    print("Using Google Cloud Storage API...")
    
    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(config.BUCKET_NAME)
    blob = bucket.blob('aux/pagerank.pkl')
    
    if not blob.exists():
        print("✗ File does not exist in GCS!")
        sys.exit(1)
    
    print(f"✓ File exists in GCS")
    print(f"Size: {blob.size:,} bytes ({blob.size / 1024:.2f} KB)")
    
    if blob.size < 100_000:  # Less than 100KB is suspicious
        print("⚠ Warning: File is very small! This might be empty or incomplete.")
        print("   A valid pagerank.pkl should be at least several MB.")
    
    # Download and check contents
    print("\nDownloading and checking contents...")
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name
        blob.download_to_filename(tmp_path)
    
    with open(tmp_path, 'rb') as f:
        pr = pickle.load(f)
    
    # Clean up
    import os
    os.unlink(tmp_path)
    
    print(f"\nEntries in dictionary: {len(pr):,}")
    
    if len(pr) == 0:
        print("✗ Dictionary is EMPTY!")
        print("\nSolution: You need to rebuild pagerank.pkl")
        sys.exit(1)
    
    # Check sample
    sample_ids = list(pr.keys())[:5]
    sample_prs = [pr[id] for id in sample_ids]
    print(f"\nSample entries:")
    for doc_id, pr_val in zip(sample_ids, sample_prs):
        print(f"  [{doc_id:8d}] PR: {pr_val:.6f}")
    
    # Statistics
    max_pr = max(pr.values())
    min_pr = min(pr.values())
    non_zero = sum(1 for v in pr.values() if v > 0)
    
    print(f"\nStatistics:")
    print(f"  Max PageRank: {max_pr:.6f}")
    print(f"  Min PageRank: {min_pr:.6f}")
    print(f"  Non-zero entries: {non_zero:,}/{len(pr):,} ({100*non_zero/len(pr):.1f}%)")
    
    if non_zero == 0:
        print("\n✗ All PageRank values are 0! File is not valid.")
        print("\nSolution: You need to rebuild pagerank.pkl")
        sys.exit(1)
    
    # Check if doc_ids are integers
    sample_key = list(pr.keys())[0]
    if not isinstance(sample_key, int):
        print(f"\n⚠ Warning: doc_ids are not integers! Type: {type(sample_key)}")
        print(f"  This might cause lookup issues.")
    
    print("\n✓ File appears to be valid!")
    
except ImportError:
    # Method 2: Check locally if file exists
    print("Google Cloud Storage API not available.")
    print("Trying to check local file...")
    
    try:
        import config
        import pickle
        
        pr_path = Path(config.PAGERANK_PATH)
        print(f"\nChecking local file: {pr_path}")
        
        if pr_path.exists():
            print(f"✓ File exists locally")
            size = pr_path.stat().st_size
            print(f"Size: {size:,} bytes ({size / 1024:.2f} KB)")
            
            if size < 100_000:
                print("⚠ Warning: File is very small! This might be empty or incomplete.")
            
            with open(pr_path, 'rb') as f:
                pr = pickle.load(f)
            
            print(f"\nEntries in dictionary: {len(pr):,}")
            
            if len(pr) == 0:
                print("✗ Dictionary is EMPTY!")
                print("\nSolution: You need to rebuild pagerank.pkl")
                sys.exit(1)
            
            # Check sample
            sample_ids = list(pr.keys())[:5]
            sample_prs = [pr[id] for id in sample_ids]
            print(f"\nSample entries:")
            for doc_id, pr_val in zip(sample_ids, sample_prs):
                print(f"  [{doc_id:8d}] PR: {pr_val:.6f}")
            
            # Statistics
            max_pr = max(pr.values())
            min_pr = min(pr.values())
            non_zero = sum(1 for v in pr.values() if v > 0)
            
            print(f"\nStatistics:")
            print(f"  Max PageRank: {max_pr:.6f}")
            print(f"  Min PageRank: {min_pr:.6f}")
            print(f"  Non-zero entries: {non_zero:,}/{len(pr):,} ({100*non_zero/len(pr):.1f}%)")
            
            if non_zero == 0:
                print("\n✗ All PageRank values are 0! File is not valid.")
                print("\nSolution: You need to rebuild pagerank.pkl")
                sys.exit(1)
            
            print("\n✓ File appears to be valid!")
        else:
            print(f"✗ File does not exist locally")
            print(f"\nThe file should be at: {pr_path}")
            print("\nIf you're reading from GCS, check the file size:")
            print("  gsutil ls -lh gs://matiasgaya333/aux/pagerank.pkl")
            print("\nIf the file is small (< 100KB), you need to rebuild it:")
            print("  On your GCP instance:")
            print("    cd ~/IR_Project")
            print("    source venv/bin/activate")
            print("    nohup python -m indexing.build_indices --dump gs://matiasgaya333/raw/wikidata20210801_preprocessed/ --build pagerank --parquet > pagerank_build.log 2>&1 &")
            sys.exit(1)
            
    except ImportError:
        print("✗ Cannot import config module")
        print("\nPlease run this script from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("If the file is small (< 100KB) or has 0 entries, you need to rebuild it.")
print("\nOn your GCP instance, run:")
print("  cd ~/IR_Project")
print("  source venv/bin/activate")
print("  nohup python -m indexing.build_indices \\")
print("    --dump gs://matiasgaya333/raw/wikidata20210801_preprocessed/ \\")
print("    --build pagerank --parquet > pagerank_build.log 2>&1 &")
print("\nThen check progress:")
print("  tail -f pagerank_build.log")
print("\nAfter it finishes, upload to GCS:")
print("  gsutil cp aux/pagerank.pkl gs://matiasgaya333/aux/pagerank.pkl")
