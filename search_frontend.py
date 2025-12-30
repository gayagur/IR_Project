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
        # Run searches in parallel for faster performance
        print("[SEARCH] Searching all indices in parallel...")
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all search tasks
            future_body = executor.submit(engine.search_body_bm25, q_tokens, top_n=300)
            future_title = executor.submit(engine.search_title_count, q_tokens, top_n=5000)
            future_anchor = executor.submit(engine.search_anchor_count, q_tokens, top_n=5000)
            
            # Wait for all to complete
            body_ranked = future_body.result()
            title_ranked = future_title.result()
            anchor_ranked = future_anchor.result()
        
        print(f"[SEARCH] Body results: {len(body_ranked)}")
        print(f"[SEARCH] Title results: {len(title_ranked)}")
        print(f"[SEARCH] Anchor results: {len(anchor_ranked)}")

        print("[SEARCH] Merging signals...")
        merged = engine.merge_signals(
            body_ranked=body_ranked,
            title_ranked=title_ranked,
            anchor_ranked=anchor_ranked,
            top_n=100,
        )
        print(f"[SEARCH] Merged results: {len(merged)}")

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
    try:
        from search_runtime import _check_if_indices_exist_locally, _download_indices_from_gcs_to_local
        import config
        
        read_from_gcs = getattr(config, 'READ_FROM_GCS', False)
        
        if read_from_gcs:
            print("READ_FROM_GCS = True - will read directly from GCS (no download)")
            print("Skipping download step...")
        else:
            # READ_FROM_GCS = False - try to download to local disk for faster access
            if not _check_if_indices_exist_locally():
                print("Indices not found locally - downloading from GCS bucket...")
                if not config.BUCKET_NAME:
                    print("⚠ Warning: BUCKET_NAME is not set in config.py")
                    print("Cannot download indices from GCS")
                else:
                    t0_download = time.time()
                    if _download_indices_from_gcs_to_local(config.BUCKET_NAME):
                        download_time = time.time() - t0_download
                        print("=" * 60)
                        print(f"✓ All indices downloaded in {download_time:.1f} seconds")
                        print("=" * 60)
                    else:
                        print("⚠ Warning: Failed to download indices from GCS")
                        print("Server will attempt to read directly from GCS as fallback")
            else:
                print("✓ Indices already exist locally - no download needed")
    except Exception as e:
        import traceback
        print(f"⚠ Error checking/downloading indices: {e}")
        print(traceback.format_exc())
        print("Will continue with server startup...")
    
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
