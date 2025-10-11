"""

"""

import chromadb
from config import settings

# --- Global Singleton for ChromaDB Client ---
_chroma_client = None

def get_client():
    """
    Initializes and returns the ChromaDB persistent client (singleton pattern).
    """
    global _chroma_client
    if _chroma_client is None:
        print("Initializing ChromaDB client...")
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        print(f"✓ ChromaDB client connected to path: {settings.CHROMA_DB_PATH}")
    return _chroma_client
