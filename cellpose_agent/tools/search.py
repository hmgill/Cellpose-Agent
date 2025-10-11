"""

"""

# project/tools/search.py
from smolagents import tool
from langfuse import get_client
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from stores import get_chroma_client, get_kg_index
from models.reranker import get_reranker

langfuse = get_client()

@tool
def list_all_collections() -> list[str]:
    """Lists the names of all available collections in the ChromaDB database."""
    # This is fine because it has no arguments.
    print("\n--- TOOL CALLED: list_all_collections ---")
    client = get_chroma_client()
    collections = client.list_collections()
    return [c.name for c in collections]


@tool
def search_documentation_vector(query: str) -> str:
    """
    Searches cellpose documentation using vector search followed by a reranking step.

    Args:
        query (str): The question or search term to look up in the documentation.
    """
    print(f"\n--- TOOL CALLED: search_documentation_vector (with Reranker) for '{query}' ---")
    try:
        client = get_chroma_client()
        collection = client.get_collection(name='cellpose_docs')
        vector_store = ChromaVectorStore(chroma_collection=collection)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=25,
            node_postprocessors=[get_reranker()]
        )
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error searching documentation: {e}"


@tool
def search_knowledge_graph(query: str) -> str:
    """
    Searches using knowledge graph relationships (Neo4j). Best for "how" and "why" questions.

    Args:
        query (str): The question about relationships between concepts (e.g., parameters).
    """
    print(f"\n--- TOOL CALLED: search_knowledge_graph for '{query}' ---")
    try:
        kg_index = get_kg_index()
        query_engine = kg_index.as_query_engine(
            include_text=True, response_mode="tree_summarize"
        )
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error querying knowledge graph: {e}."


@tool
def get_parameter_relationships(parameter_name: str) -> str:
    """
    Gets information about how a parameter relates to others using the knowledge graph.

    Args:
        parameter_name (str): The specific parameter name to investigate (e.g., 'flow_threshold').
    """
    print(f"\n--- TOOL CALLED: get_parameter_relationships for '{parameter_name}' ---")
    query = f"What is {parameter_name} and how does it relate to other parameters?"
    return search_knowledge_graph(query)


@tool
def hybrid_search(query: str) -> str:
    """
    Combines reranked vector search and knowledge graph search for complex questions.

    Args:
        query (str): The complex question that may require both semantic and relational understanding.
    """
    print(f"\n--- TOOL CALLED: hybrid_search (with Reranker) for '{query}' ---")
    try:
        vector_response_str = search_documentation_vector(query)
        kg_response = search_knowledge_graph(query)

        return f"Vector Search Results (Reranked):\n{vector_response_str}\n\nKnowledge Graph Insights:\n{kg_response}"
    
    except Exception as e:
        print(f"--- Hybrid search failed, falling back to vector search: {e} ---")
        return search_documentation_vector(query)
