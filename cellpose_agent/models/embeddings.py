"""

"""

from PIL import Image
from sentence_transformers import SentenceTransformer
from config import settings

# --- Global Singleton for Embedding Model ---
_embedding_model = None

def get_embedding_model():
    """
    Initializes and returns the SentenceTransformer model (singleton pattern).
    """
    global _embedding_model
    if _embedding_model is None:
        print("Initializing embedding model...")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
        print(f"✓ Embedding model initialized ({settings.EMBEDDING_MODEL_ID})")
    return _embedding_model

def get_image_embedding(image_path: str) -> list[float]:
    """
    Generates a CLIP embedding for a given image file.

    Args:
        image_path (str): The path to the image file.

    Returns:
        list[float]: The image embedding as a list of floats.
    """
    model = get_embedding_model()
    img = Image.open(image_path).convert("RGB")
    embedding = model.encode(img, convert_to_numpy=True)
    return embedding.tolist()
