import csv
import base64
from pathlib import Path

# Import our project's centralized components
from config import settings
from stores import get_chroma_client
from models.embeddings import get_image_embedding

# --- Configuration ---
# Use the same collection name as your agent's tool
IMAGE_PARAMS_COLLECTION = "cellpose-sam_parameters_by_image_similarity"
IMAGE_DIR = Path("/N/project/retinal_images/ginkgo/scripts/example_images")
MANIFEST_PATH = Path("/N/project/retinal_images/ginkgo/scripts/gdpx3_data_manifest.csv")


def ingest_image_data(manifest_path: Path, image_dir: Path):
    """
    Reads a manifest CSV to ingest image-parameter pairs into ChromaDB,
    using the globally configured embedding model and DB client.
    """
    print("--- Starting Image Data Ingestion ---")

    if not manifest_path.exists():
        print(f"❌ ERROR: Manifest file not found at {manifest_path}")
        return
    if not image_dir.exists():
        print(f"❌ ERROR: Image directory not found at {image_dir}")
        return

    # 1. Reuse the project's ChromaDB client
    client = get_chroma_client()

    # 2. Ensure a clean slate by deleting the old collection
    try:
        client.delete_collection(name=IMAGE_PARAMS_COLLECTION)
        print(f"✓ Deleted old '{IMAGE_PARAMS_COLLECTION}' collection.")
    except Exception:
        print(f"Collection '{IMAGE_PARAMS_COLLECTION}' not found, creating a new one.")

    # 3. Create the collection using consistent settings
    collection = client.get_or_create_collection(
        name=IMAGE_PARAMS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Read the manifest and process each image
    print(f"Reading manifest from {manifest_path}...")
    with open(manifest_path.as_posix(), mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)
            image_path = image_dir / row['image_path']
            config_text = row['parameter_text']
            
            if not image_path.exists():
                print(f"⚠️  WARNING: Skipping. Image not found: {image_path}")
                continue

            # 5. Reuse the project's embedding model and function
            embedding = get_image_embedding(str(image_path))

            # Prepare data for ChromaDB
            doc_id = image_path.stem
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode()

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{
                    "parameter_text": config_text,
                    "image_base64": image_base64,
                    "image_name": image_path.name
                }]
            )
            print(f"  + Added: {doc_id}")
    
    print("\n✅ --- Image Ingestion Complete ---")
    print(f"Processed images from {manifest_path} into the '{IMAGE_PARAMS_COLLECTION}' collection.")


if __name__ == "__main__":
    ingest_image_data(MANIFEST_PATH, IMAGE_DIR)
