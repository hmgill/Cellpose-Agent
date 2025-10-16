"""
Safety VLM Agent for content moderation
"""
import torch
from smolagents import ToolCallingAgent, TransformersModel
from transformers import BitsAndBytesConfig
from PIL import Image
import json

from config import settings
from utils.gpu import clear_gpu_cache
from tools import safety_tools  # Import only safety tools


class SafetyAgent:
    
    """
    A VLM-based safety agent that scans user inputs (text and images)
    for inappropriate content before allowing the main agent to proceed.
    """
    
    def __init__(self):
        self.instructions = """
        You are a safety moderation agent for a biological image segmentation system.
        
        Your job is to analyze user text queries and any provided images to determine if they are:
        1. Related to legitimate biological/scientific image analysis
        2. Free from harmful, inappropriate, or off-topic content
        
        ## WORKFLOW
        
        1. If the user provides text AND an image:
           - Call check_combined_safety(text, image_path) first for automated checks
           - If result is "needs_visual_inspection", examine the image yourself visually
           - Make final determination based on what you see
        
        2. If the user provides only text:
           - Call check_text_safety(text)
           - Review the results
        
        3. If the user provides only an image:
           - Call check_image_safety(image_path)
           - If result is "needs_visual_inspection", examine the image yourself visually
           - Make final determination based on what you see
        
        ## VISUAL INSPECTION GUIDANCE
        
        When you need to visually inspect an image, look for:
        
        APPROVE if you see:
        - Microscopy images of cells or tissues
        - Scientific/medical imaging (even if unclear quality)
        - Biological structures, organisms, or specimens
        - Diagrams or illustrations of biological concepts
        
        REJECT if you see:
        - Human faces or identifiable people (privacy risk)
        - Inappropriate, harmful, or offensive content
        - Personal photos, social media content, memes
        - Non-biological content 
        - Documents with potentially sensitive text
        
        ## RESPONSE FORMAT
        
        You must end your response with ONLY a JSON object in this exact format:
        {
            "safe": true/false,
            "reason": "Brief explanation of your decision",
            "category": "approved" | "inappropriate_content" | "off_topic" | "privacy_concern",
            "visual_description": "Optional: what you saw in the image"
        }
        
        DO NOT wrap the JSON in markdown code blocks or backticks.
        DO NOT include any other text after the JSON object.
        
        Be permissive for legitimate scientific use cases, but firm on blocking clearly 
        inappropriate content.
        
        ## IMPORTANT
        - Use the safety tools provided to help with your analysis
        - Always include the JSON output at the end
        - For edge cases, err on the side of approving legitimate scientific content
        - If unsure, approve but include a note in your reason
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
            name = "safety_agent",
            description = "checks user inputs for relevance"
        )


    
    def check_safety(self, text: str, image_path: str = None) -> dict:
        
        """
        Check if the user input is safe to process.
        
        Args:
            text: User's text query
            image_path: Optional path to image file
            
        Returns:
            dict: Safety check result with keys 'safe', 'reason', 'category'
        """
        
        print(f"\n{'='*60}\nSAFETY CHECK\n{'='*60}")
        print(f"Text: {text[:100]}...")
        if image_path:
            print(f"Image: {image_path}")
        
        # Construct the safety check prompt
        prompt = f"""Perform a safety check on this user request:
        TEXT: {text}
        """
        
        if image_path:
            prompt += f"\nIMAGE PATH: {image_path}"
            prompt += """
            The image will be provided to you for visual inspection. 
            Use the safety tools to help with your analysis, then provide your final assessment.
            """
        
        prompt += "\n\nProvide your safety assessment as a JSON object (no markdown, no backticks)."
        
        try:
            # If there's an image, provide it to the agent
            if image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    # Run agent with image
                    result = self.agent.run(prompt, images=[image])
                except Exception as img_error:
                    print(f"⚠ Warning: Could not load image: {img_error}")
                    # Fallback to text-only check
                    result = self.agent.run(prompt)
            else:
                result = self.agent.run(prompt)
            
            # Parse the JSON response from the end of the output
            result_str = str(result).strip()
            
            # Try to extract JSON from the response
            # Look for the last occurrence of { and parse from there
            json_start = result_str.rfind('{')
            if json_start != -1:
                json_str = result_str[json_start:]
                safety_result = json.loads(json_str)
            else:
                # Couldn't find JSON, try parsing the whole thing
                safety_result = json.loads(result_str)
            
            print(f"\n{'='*60}")
            print(f"Safety Result: {'✓ APPROVED' if safety_result.get('safe') else '✗ REJECTED'}")
            print(f"Reason: {safety_result.get('reason', 'N/A')}")
            print(f"Category: {safety_result.get('category', 'N/A')}")
            if 'visual_description' in safety_result and safety_result['visual_description']:
                print(f"Visual: {safety_result['visual_description']}")
            print(f"{'='*60}\n")
            
            return safety_result
            
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Could not parse safety agent response as JSON: {e}")
            print(f"Raw response: {result}")
            
            # Try to infer safety from the text response
            result_lower = str(result).lower()
            if any(word in result_lower for word in ['reject', 'unsafe', 'inappropriate', 'harmful']):
                return {
                    "safe": False,
                    "reason": "Safety agent indicated concerns but response format was unclear",
                    "category": "inappropriate_content",
                    "raw_response": str(result)[:200]
                }
            else:
                # Default to safe if we can't parse and no obvious rejection
                return {
                    "safe": True,
                    "reason": "Could not parse safety check, but no obvious concerns detected",
                    "category": "approved",
                    "raw_response": str(result)[:200]
                }
        except Exception as e:
            print(f"⚠ Error in safety check: {e}")
            # Default to safe if error (fail open) - you can change this to fail-closed
            return {
                "safe": True,
                "reason": f"Safety check error: {e}",
                "category": "approved"
            }
        finally:
            clear_gpu_cache()
