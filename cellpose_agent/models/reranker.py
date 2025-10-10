"""

"""

from llama_index.core.postprocessor import SentenceTransformerRerank

# --- Global Singleton for Reranker Model ---
_reranker_model = None

def get_reranker():
    """
    Initializes and returns the SentenceTransformerRerank model (singleton pattern).
    This model will download on first use.
    """
    
    global _reranker_model
    
    if _reranker_model is None:
        
        print("Initializing Cross-Encoder Reranker model...")
        
        # A popular, lightweight, and effective cross-encoder
        _reranker_model = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
            top_n=3  # The number of documents to return after reranking
        )
        print("✓ Reranker model initialized.")
        
    return _reranker_model
