"""

"""

from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from neo4j import GraphDatabase

from config import settings
from stores import chroma_store

# --- Global Singleton for KG Index ---
_kg_index = None

def get_graph_store():
    """Initializes and returns the Neo4jGraphStore."""
    return Neo4jGraphStore(
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        url=settings.NEO4J_URI,
        database=settings.NEO4J_DATABASE,
    )

def check_graph_status():
    """Checks if the Neo4j graph contains any nodes or relationships."""
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        nodes_result = session.run("MATCH (n) RETURN count(n) as count")
        node_count = nodes_result.single()['count']
        rels_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = rels_result.single()['count']
    driver.close()
    return node_count, rel_count

def initialize_knowledge_graph():
    """Builds the knowledge graph from documents in ChromaDB and stores it in Neo4j."""
    print("\n--- Building Knowledge Graph in Neo4j ---")
    chroma_client = chroma_store.get_client()
    doc_collection = chroma_client.get_collection(name='cellpose_docs')
    doc_data = doc_collection.get()

    documents = [
        Document(text=text, metadata=meta)
        for text, meta in zip(doc_data['documents'], doc_data['metadatas'])
    ]

    storage_context = StorageContext.from_defaults(graph_store=get_graph_store())
    
    KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=3,
        include_embeddings=True,
        show_progress=True
    )
    print("✓ Knowledge Graph built and stored in Neo4j successfully.")

def get_kg_index():
    """Loads the KnowledgeGraphIndex from the existing Neo4j graph store."""
    global _kg_index
    if _kg_index is None:
        print("Loading Knowledge Graph index from Neo4j...")
        storage_context = StorageContext.from_defaults(graph_store=get_graph_store())
        _kg_index = KnowledgeGraphIndex(nodes=[], storage_context=storage_context)
        print("✓ Knowledge Graph index loaded.")
    return _kg_index
