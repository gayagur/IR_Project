from flask import Flask, request, jsonify
import traceback

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors and show the actual error message."""
    return jsonify({
        "error": "Internal Server Error",
        "message": str(error),
        "traceback": traceback.format_exc()
    }), 500

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    try:
        print(f"[SEARCH] Query: '{query}'")
        from search_runtime import get_engine
        engine = get_engine()

        q_tokens = engine.tokenize_query(query)
        print(f"[SEARCH] Tokenized query: {q_tokens}")
        if not q_tokens:
            print("[SEARCH] No tokens after tokenization, returning empty results")
            return jsonify(res)

        # Candidates + scores from multiple signals
        # Run searches in parallel for faster performance (without LSI - LSI will rerank later)
        print("[SEARCH] Searching all indices in parallel...")
        from concurrent.futures import ThreadPoolExecutor
        import config
        
        # Check LSI weight - if 0, skip LSI entirely
        lsi_weight = getattr(config, 'LSI_WEIGHT', 0.25)
        use_lsi = engine.body_lsi is not None and lsi_weight > 0
        
        # Run body/title/anchor searches in parallel (no LSI search)
        max_workers = 3
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_body = executor.submit(engine.search_body_bm25, q_tokens, top_n=150)
            future_title = executor.submit(engine.search_title_count, q_tokens, top_n=200)
            future_anchor = executor.submit(engine.search_anchor_count, q_tokens, top_n=200)
            
            # Wait for all to complete
            body_ranked = future_body.result()
            title_ranked = future_title.result()
            anchor_ranked = future_anchor.result()
        
        print(f"[SEARCH] Body results: {len(body_ranked)}")
        print(f"[SEARCH] Title results: {len(title_ranked)}")
        print(f"[SEARCH] Anchor results: {len(anchor_ranked)}")

        print("[SEARCH] Merging signals (without LSI)...")
        # Merge without LSI first to get initial ranking
        merged = engine.merge_signals(
            body_ranked=body_ranked,
            title_ranked=title_ranked,
            anchor_ranked=anchor_ranked,
            lsi_ranked=None,  # No LSI in initial merge
            top_n=150,  # Get more candidates for reranking
        )
        print(f"[SEARCH] Merged results (before LSI rerank): {len(merged)}")
        
        # Rerank top K with LSI if available and weight > 0
        if use_lsi:
            lsi_top_k = getattr(config, 'LSI_TOP_K', 100)
            print(f"[SEARCH] Reranking top {lsi_top_k} results with LSI...")
            merged = engine.rerank_with_lsi(
                q_tokens,
                merged,
                top_k=lsi_top_k,
                lsi_weight=lsi_weight,  # Use LSI weight from config
            )
            print(f"[SEARCH] Reranked results: {len(merged)}")
        
        # Take final top 100
        merged = merged[:100]

        res = [(doc_id, engine.titles.get(doc_id, "")) for doc_id, _ in merged]
        print(f"[SEARCH] Final results: {len(res)}")
    except Exception as e:
        import traceback
        error_msg = f"Error in search: {str(e)}\n{traceback.format_exc()}"
        print(f"[SEARCH ERROR] {error_msg}")  # Print to server console
        # Return empty results instead of error to avoid crashing
        return jsonify(res)  # Return empty list instead of error
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    try:
        print(f"[SEARCH_BODY] Query: '{query}'")
        from search_runtime import get_engine
        engine = get_engine()

        q_tokens = engine.tokenize_query(query)
        print(f"[SEARCH_BODY] Tokenized query: {q_tokens}")
        if not q_tokens:
            print("[SEARCH_BODY] No tokens, returning empty results")
            return jsonify(res)

        ranked = engine.search_body_tfidf_cosine(q_tokens, top_n=100)
        print(f"[SEARCH_BODY] Results: {len(ranked)}")
        res = [(doc_id, engine.titles.get(doc_id, "")) for doc_id, _ in ranked]
    except Exception as e:
        import traceback
        error_msg = f"Error in search_body: {str(e)}\n{traceback.format_exc()}"
        print(f"[SEARCH_BODY ERROR] {error_msg}")
        return jsonify(res)  # Return empty list instead of error
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    try:
        print(f"[SEARCH_TITLE] Query: '{query}'")
        from search_runtime import get_engine
        engine = get_engine()

        q_tokens = engine.tokenize_query(query)
        print(f"[SEARCH_TITLE] Tokenized query: {q_tokens}")
        if not q_tokens:
            print("[SEARCH_TITLE] No tokens, returning empty results")
            return jsonify(res)

        ranked = engine.search_title_count(q_tokens, top_n=None)  # ALL
        print(f"[SEARCH_TITLE] Results: {len(ranked)}")
        res = [(doc_id, engine.titles.get(doc_id, "")) for doc_id, _ in ranked]
    except Exception as e:
        import traceback
        error_msg = f"Error in search_title: {str(e)}\n{traceback.format_exc()}"
        print(f"[SEARCH_TITLE ERROR] {error_msg}")
        return jsonify(res)  # Return empty list instead of error
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    try:
        print(f"[SEARCH_ANCHOR] Query: '{query}'")
        from search_runtime import get_engine
        engine = get_engine()

        q_tokens = engine.tokenize_query(query)
        print(f"[SEARCH_ANCHOR] Tokenized query: {q_tokens}")
        if not q_tokens:
            print("[SEARCH_ANCHOR] No tokens, returning empty results")
            return jsonify(res)

        ranked = engine.search_anchor_count(q_tokens, top_n=None)  # ALL
        print(f"[SEARCH_ANCHOR] Results: {len(ranked)}")
        res = [(doc_id, engine.titles.get(doc_id, "")) for doc_id, _ in ranked]
    except Exception as e:
        import traceback
        error_msg = f"Error in search_anchor: {str(e)}\n{traceback.format_exc()}"
        print(f"[SEARCH_ANCHOR ERROR] {error_msg}")
        return jsonify(res)  # Return empty list instead of error
    # END SOLUTION
    return jsonify(res)

@app.route("/search_lsi")
def search_lsi():
    ''' Returns up to a 100 search results for the query using LSI (Latent Semantic Indexing).
    
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_lsi?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    try:
        print(f"[SEARCH_LSI] Query: '{query}'")
        from search_runtime import get_engine
        engine = get_engine()
        
        if engine.body_lsi is None:
            print("[SEARCH_LSI] LSI not available, returning empty results")
            return jsonify(res)
        
        q_tokens = engine.tokenize_query(query)
        print(f"[SEARCH_LSI] Tokenized query: {q_tokens}")
        if not q_tokens:
            print("[SEARCH_LSI] No tokens, returning empty results")
            return jsonify(res)
        
        ranked = engine.search_body_lsi(q_tokens, top_n=100)
        print(f"[SEARCH_LSI] Results: {len(ranked)}")
        res = [(doc_id, engine.titles.get(doc_id, "")) for doc_id, _ in ranked]
    except Exception as e:
        import traceback
        error_msg = f"Error in search_lsi: {str(e)}\n{traceback.format_exc()}"
        print(f"[SEARCH_LSI ERROR] {error_msg}")
        return jsonify(res)  # Return empty list instead of error
    # END SOLUTION
    return jsonify(res)

@app.route("/search_with_weights")
def search_with_weights():
    ''' Returns search results with custom weights.
    
        Query parameters:
        - query: search query (required)
        - body_weight: weight for BM25 body search (default: from config)
        - title_weight: weight for title search (default: from config)
        - anchor_weight: weight for anchor search (default: from config)
        - lsi_weight: weight for LSI search (default: from config, 0.0 to disable)
        - pagerank_boost: PageRank boost weight (default: from config)
        - pageview_boost: PageView boost weight (default: from config)
        
        Example:
        http://YOUR_SERVER/search_with_weights?query=hello&body_weight=1.0&title_weight=0.5&lsi_weight=0.0
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    try:
        from search_runtime import get_engine
        import config
        
        engine = get_engine()
        q_tokens = engine.tokenize_query(query)
        if not q_tokens:
            return jsonify(res)
        
        # Get custom weights from query parameters, fallback to config
        body_weight = float(request.args.get('body_weight', getattr(config, 'BODY_WEIGHT', 1.0)))
        title_weight = float(request.args.get('title_weight', getattr(config, 'TITLE_WEIGHT', 0.35)))
        anchor_weight = float(request.args.get('anchor_weight', getattr(config, 'ANCHOR_WEIGHT', 0.25)))
        lsi_weight = float(request.args.get('lsi_weight', getattr(config, 'LSI_WEIGHT', 0.25)))
        pr_boost = float(request.args.get('pagerank_boost', getattr(config, 'PAGERANK_BOOST', 0.15)))
        pv_boost = float(request.args.get('pageview_boost', getattr(config, 'PAGEVIEW_BOOST', 0.10)))
        
        # Run searches in parallel (without LSI - LSI will rerank later)
        from concurrent.futures import ThreadPoolExecutor
        max_workers = 3
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_body = executor.submit(engine.search_body_bm25, q_tokens, top_n=150)
            future_title = executor.submit(engine.search_title_count, q_tokens, top_n=200)
            future_anchor = executor.submit(engine.search_anchor_count, q_tokens, top_n=200)
            
            body_ranked = future_body.result()
            title_ranked = future_title.result()
            anchor_ranked = future_anchor.result()
        
        # Merge without LSI first
        merged = engine.merge_signals(
            body_ranked=body_ranked,
            title_ranked=title_ranked,
            anchor_ranked=anchor_ranked,
            lsi_ranked=None,  # No LSI in initial merge
            top_n=150,  # Get more candidates for reranking
            body_weight=body_weight,
            title_weight=title_weight,
            anchor_weight=anchor_weight,
            lsi_weight=0.0,  # No LSI in merge
            pagerank_boost=pr_boost,
            pageview_boost=pv_boost,
        )
        
        # Rerank top K with LSI if available and weight > 0
        use_lsi = engine.body_lsi is not None and lsi_weight > 0
        if use_lsi:
            lsi_top_k = getattr(config, 'LSI_TOP_K', 100)
            merged = engine.rerank_with_lsi(
                q_tokens,
                merged,
                top_k=lsi_top_k,
                lsi_weight=lsi_weight,  # Use custom LSI weight
            )
        
        # Take final top 100
        merged = merged[:100]
        
        res = [(doc_id, engine.titles.get(doc_id, "")) for doc_id, _ in merged]
    except Exception as e:
        import traceback
        error_msg = f"Error in search_with_weights: {str(e)}\n{traceback.format_exc()}"
        print(f"[SEARCH_WITH_WEIGHTS ERROR] {error_msg}")
        return jsonify(res)
    
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    try:
        print(f"[GET_PAGERANK] Requested IDs: {wiki_ids}")
        from search_runtime import get_engine
        engine = get_engine()
        pr = engine.pagerank
        res = [float(pr.get(int(wid), 0.0)) for wid in wiki_ids]
        print(f"[GET_PAGERANK] Returning {len(res)} values")
    except Exception as e:
        import traceback
        print(f"[GET_PAGERANK ERROR] {e}\n{traceback.format_exc()}")
        res = [0.0] * len(wiki_ids) if wiki_ids else []
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body_bm25")
def search_body_bm25():
    """Search body with custom BM25 parameters."""
    res = []
    query = request.args.get('query', '')
    if not query:
        return jsonify(res)
    
    try:
        from search_runtime import get_engine
        import config
        engine = get_engine()
        
        # Use config defaults if not provided
        default_k1 = getattr(config, 'BM25_K1', 2.5)
        default_b = getattr(config, 'BM25_B', 0.0)
        k1 = float(request.args.get('k1', default_k1))
        b = float(request.args.get('b', default_b))
        
        q_tokens = engine.tokenize_query(query)
        if not q_tokens:
            return jsonify(res)
        
        ranked = engine.search_body_bm25(q_tokens, top_n=100, k1=k1, b=b)
        res = [(doc_id, engine.titles.get(doc_id, "")) for doc_id, _ in ranked]
    except Exception as e:
        print(f"[BM25 ERROR] {e}")
    
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    try:
        print(f"[GET_PAGEVIEW] Requested IDs: {wiki_ids}")
        from search_runtime import get_engine
        engine = get_engine()
        pv = engine.pageviews
        res = [int(pv.get(int(wid), 0)) for wid in wiki_ids]
        print(f"[GET_PAGEVIEW] Returning {len(res)} values")
    except Exception as e:
        import traceback
        print(f"[GET_PAGEVIEW ERROR] {e}\n{traceback.format_exc()}")
        res = [0] * len(wiki_ids) if wiki_ids else []
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    import time
    
    # Step 1: Download indices from GCS if needed (BEFORE starting server)
    # Only download if READ_FROM_GCS = False (if True, will read directly from GCS)
    print("=" * 60)
    print("Checking if indices need to be downloaded from GCS...")
    print("=" * 60)
    
    # Add detailed logging
    try:
        print("[DEBUG] Importing modules...")
        from search_runtime import _check_if_indices_exist_locally, _download_indices_from_gcs_to_local
        import config
        
        print(f"[DEBUG] READ_FROM_GCS = {getattr(config, 'READ_FROM_GCS', False)}")
        print(f"[DEBUG] BUCKET_NAME = {getattr(config, 'BUCKET_NAME', 'NOT SET')}")
        print(f"[DEBUG] INDICES_DIR = {config.INDICES_DIR}")
        print(f"[DEBUG] AUX_DIR = {config.AUX_DIR}")
        
        read_from_gcs = getattr(config, 'READ_FROM_GCS', False)
        
        if read_from_gcs:
            print("READ_FROM_GCS = True - will read directly from GCS (no download)")
            print("Skipping download step...")
        else:
            # READ_FROM_GCS = False - try to download to local disk for faster access
            print("[DEBUG] READ_FROM_GCS = False - checking if indices exist locally...")
            print("[DEBUG] Calling _check_if_indices_exist_locally()...")
            
            try:
                indices_exist = _check_if_indices_exist_locally()
                print(f"[DEBUG] _check_if_indices_exist_locally() returned: {indices_exist}")
            except Exception as check_error:
                print(f"[DEBUG] ERROR in _check_if_indices_exist_locally(): {check_error}")
                import traceback
                traceback.print_exc()
                indices_exist = False  # Assume they don't exist if check fails
            
            if not indices_exist:
                print("=" * 60)
                print("Indices not found locally - downloading from GCS bucket...")
                print("=" * 60)
                
                if not config.BUCKET_NAME:
                    print("⚠ ERROR: BUCKET_NAME is not set in config.py")
                    print("Cannot download indices from GCS")
                else:
                    print(f"[DEBUG] Starting download from bucket: {config.BUCKET_NAME}")
                    print("[DEBUG] Calling _download_indices_from_gcs_to_local()...")
                    
                    t0_download = time.time()
                    try:
                        success = _download_indices_from_gcs_to_local(config.BUCKET_NAME)
                        download_time = time.time() - t0_download
                        
                        if success:
                            print("=" * 60)
                            print(f"✓ All indices downloaded in {download_time:.1f} seconds")
                            print("=" * 60)
                        else:
                            print("=" * 60)
                            print("⚠ ERROR: _download_indices_from_gcs_to_local() returned False")
                            print("Server will attempt to read directly from GCS as fallback")
                            print("=" * 60)
                    except Exception as download_error:
                        print("=" * 60)
                        print(f"⚠ ERROR during download: {download_error}")
                        import traceback
                        traceback.print_exc()
                        print("=" * 60)
            else:
                print("=" * 60)
                print("✓ Indices already exist locally - no download needed")
                print("=" * 60)
                
    except Exception as e:
        import traceback
        print("=" * 60)
        print(f"⚠ ERROR in download check: {e}")
        print(traceback.format_exc())
        print("Will continue with server startup...")
        print("=" * 60)
    
    # Step 2: Pre-load search engine (AFTER indices are downloaded)
    print("=" * 60)
    print("Pre-loading search engine...")
    print("=" * 60)
    t0 = time.time()
    try:
        from search_runtime import get_engine
        engine = get_engine()
        load_time = time.time() - t0
        print("=" * 60)
        print(f"✓ Engine ready in {load_time:.1f} seconds")
        print("=" * 60)
    except Exception as e:
        import traceback
        print(f"✗ Error pre-loading engine: {e}")
        print(traceback.format_exc())
        print("Server will start but queries may be slow or fail")
        print("=" * 60)
    
    # Step 3: Start the server (ONLY after indices are downloaded and engine is loaded)
    print("=" * 60)
    print("Starting Flask server...")
    print("=" * 60)
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
