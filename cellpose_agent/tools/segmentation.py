"""

"""

from smolagents import tool
from langfuse import get_client
from stores import chroma_store
from models.embeddings import get_image_embedding

langfuse = get_client()

@tool
def get_segmentation_parameters(image_path: str) -> str:
    """
    Finds the best cellpose-sam segmentation parameters for an image using vector similarity.
    Args:
        image_path (str): Path to the image file to segment.
    Returns:
        str: Recommended segmentation parameters.
    """
    print(f"\n--- TOOL CALLED: get_segmentation_parameters for '{image_path}' ---")
    try:
        client = chroma_store.get_client()
        collection = client.get_collection(name='cellpose-sam_parameters_by_image_similarity')
        query_embedding = get_image_embedding(image_path)
        
        results = collection.query(query_embeddings=[query_embedding], n_results=1)

        if not (results['metadatas'] and results['metadatas'][0]):
            return "No similar images found in the database."

        best_config = results['metadatas'][0][0].get('parameter_text', 'N/A')
        best_image = results['metadatas'][0][0].get('image_name', 'N/A')
        distance = results['distances'][0][0]

        print(f"Most similar: {best_image} (distance: {distance:.3f})")
        print(f"Recommended: {best_config}")
        
        return f"Based on similarity to '{best_image}' (distance: {distance:.3f}), {best_config}"

    except Exception as e:
        return f"Error: {e}"
