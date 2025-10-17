"""

"""

# project/config/settings.py
import os
import torch
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
#from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.prompts import PromptTemplate
from transformers import BitsAndBytesConfig 

from utils.gpu import get_max_memory

# --- Environment & Paths ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://8d0af37b.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "b5zqfnglm_CWHVYpmuXBR8oDyjaOqvT17L8pBUnfUJ0")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
CHROMA_DB_PATH = "./rag/cellpose_db/"


# --- HF TOKEN ---
HF_TOKEN = os.getenv("HF_TOKEN")


# --- Model Identifiers ---
EMBEDDING_MODEL_ID = 'clip-ViT-B-32'
SAFETY_AGENT_MODEL_ID = "/N/project/retinal_images/hub/models--google--shieldgemma-2-4b-it/snapshots/eaf60452b5fc41a911338a022e628b0c15283897/"
CELLPOSE_AGENT_MODEL_ID = "/N/project/retinal_images/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80/"
MANAGER_AGENT_MODEL_ID = "/N/project/retinal_images/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80/"




# --- LlamaIndex Global Settings ---
def configure_llama_index():
    """
    Configures global LlamaIndex settings for the embedding model and the LLM.
    """
    print("✓ Configuring LlamaIndex settings...")

    # This prompt template is REQUIRED for Gemma models to function correctly
    query_wrapper_prompt = PromptTemplate(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"        
    )
    
    llm = HuggingFaceLLM(
        model_name=CELLPOSE_AGENT_MODEL_ID,
        tokenizer_name=CELLPOSE_AGENT_MODEL_ID,
        query_wrapper_prompt=query_wrapper_prompt,
        device_map="auto",
        model_kwargs={
            "dtype" : torch.float32,
            "quantization_config": quantization_config
        }        
    )
    
    Settings.llm = llm

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL_ID}"
    )
    
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    print("✓ LlamaIndex configured to use local Embedding Model and LLM.")
