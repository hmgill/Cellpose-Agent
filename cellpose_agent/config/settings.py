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

MAX_MEMORY = get_max_memory(memory_fraction=0.85, cpu_memory="50GB")

# --- Environment & Paths ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
CHROMA_DB_PATH = "./rag/cellpose_db/"


# --- Model Identifiers ---
EMBEDDING_MODEL_ID = 'clip-ViT-B-32'


AGENT_MODEL_ID = "/N/project/retinal_images/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80/"


# --- HF TOKEN ---
HF_TOKEN = os.getenv("HF_TOKEN")


# --- LlamaIndex Global Settings ---
def configure_llama_index():
    """
    Configures global LlamaIndex settings for the embedding model and the LLM.
    This function ensures that all parts of LlamaIndex use our specified local models
    instead of defaulting to OpenAI.
    """
    print("✓ Configuring LlamaIndex settings...")

    # This prompt template is REQUIRED for Gemma models to function correctly
    query_wrapper_prompt = PromptTemplate(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    
    llm = HuggingFaceLLM(
        model_name=AGENT_MODEL_ID,
        tokenizer_name=AGENT_MODEL_ID,
        query_wrapper_prompt=query_wrapper_prompt,
        device_map="auto",
        model_kwargs={
            "torch_dtype" : torch.float32,
            "quantization_config": quantization_config,
            "max_memory": MAX_MEMORY            
        }        
    )
    
    Settings.llm = llm

    """
    HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        token = HF_TOKEN,
        provider = "auto"
    )
    """

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL_ID}"
    )
    
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    print("✓ LlamaIndex configured to use local Embedding Model and LLM.")
