"""
Safety VLM Agent for content moderation
"""
import torch
from smolagents import ToolCallingAgent, TransformersModel
from transformers import AutoProcessor, ShieldGemma2ForImageClassification, BitsAndBytesConfig
from PIL import Image
import json

from config import settings
from utils.gpu import clear_gpu_cache
from tools import safety_tools


class SafetyAgent:
    """
    A VLM-based safety agent that scans user inputs (text and images)
    for inappropriate content before allowing the main agent to proceed.
    """
    
    def __init__(self):
        # Simplified instructions focused on tool calling
        self.instructions = """
        You are a safety moderation agent for a biological image segmentation system.
        
        Your job is to check if user requests are appropriate for biological image analysis.
        
        ## WORKFLOW
        
        When given a task with text and/or image:
        
        1. **If both text and image are provided:**
           - Use check_combined_safety(text=<text>, image_path=<image_path>)
           - Review the JSON output
           - Make your final determination
        
        2. **If only text is provided:**
           - Use check_text_safety(text=<text>)
        
        3. **If only image is provided:**
           - Use check_image_safety(image_path=<image_path>)
        
        4. After receiving tool results, provide your final answer in this structure:
           
           ### 1. Task outcome (short version):
           APPROVED or REJECTED
           
           ### 2. Task outcome (extremely detailed version):
           [Explain what you found and why you made your decision]
           
           ### 3. Additional context (if relevant):
           [Any additional details about the safety check]
        
        ## IMPORTANT
        - Always call the appropriate safety tool first
        - Review the tool's JSON output carefully
        - Provide clear reasoning for your decision
        - Be permissive for legitimate scientific content
        - Be firm on blocking inappropriate/off-topic content
        """
        
        self.model = self._initialize_model()
        self.agent = self._create_agent()
    
    def _initialize_model(self):
        """Initialize a VLM for safety checks."""
        clear_gpu_cache()
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        return TransformersModel(
            model_id=settings.SAFETY_AGENT_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float32,
            model_kwargs={
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": True
            }
        )
    
    def _create_agent(self):
        """Create the safety agent with safety-specific tools."""
        return ToolCallingAgent(
            model=self.model,
            tools=safety_tools,  
            instructions=self.instructions,
            max_steps=5,  
            name="safety_agent",
            description="Checks user inputs for appropriateness and relevance to biological image analysis."
        )
