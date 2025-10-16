"""
CellposeAgent with proper VLM configuration
"""
import torch
from datetime import datetime
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.agents import ActionStep
from langfuse import get_client, observe
from transformers import BitsAndBytesConfig
from PIL import Image
import json

from config import settings
from utils.gpu import clear_gpu_cache
from tools import cellpose_tools


langfuse = get_client()


class CellposeAgent:

    @staticmethod
    def attach_images_callback(step_log: ActionStep, agent: ToolCallingAgent) -> None:
        """
        Callback to attach images to ActionStep after tool execution.
        This allows the VLM to see images without base64 in tool returns.
        
        Processes tool observations to extract image paths and attach the actual images.
        """
        if not isinstance(step_log, ActionStep):
            return
        
        # Check if there are observations to process
        if not step_log.observations:
            return
        
        try:
            # Try to parse observations as JSON
            obs_data = json.loads(step_log.observations)
            
            # Extract image paths from different tool responses
            images_to_attach = []
            
            # Pattern 1: get_segmentation_parameters returns image_path
            if obs_data.get("status") == "success" and "image_path" in obs_data:
                image_path = obs_data["image_path"]
                print(f"[Callback] Attaching image from get_segmentation_parameters: {image_path}")
                images_to_attach.append(image_path)
            
            # Pattern 2: refine_segmentation returns both original and segmented paths
            elif obs_data.get("status") == "ready_for_visual_analysis":
                paths = obs_data.get("image_paths", {})
                original = paths.get("original")
                segmented = paths.get("segmented")
                
                if original and segmented:
                    print(f"[Callback] Attaching images from refine_segmentation:")
                    print(f"  - Original: {original}")
                    print(f"  - Segmented: {segmented}")
                    images_to_attach.extend([original, segmented])
            
            # Attach all collected images
            if images_to_attach:
                pil_images = []
                for img_path in images_to_attach:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        pil_images.append(img.copy())  # CRITICAL: use .copy()
                    except Exception as e:
                        print(f"[Callback] Error loading image {img_path}: {e}")
                
                if pil_images:
                    step_log.observations_images = pil_images
                    print(f"[Callback] ✓ Attached {len(pil_images)} image(s) to step {step_log.step_number}")
                    for i, img in enumerate(pil_images):
                        print(f"  Image {i+1}: {img.size} pixels")
        
        except json.JSONDecodeError:
            # Observations aren't JSON, skip
            pass
        except Exception as e:
            print(f"[Callback] Error in attach_images_callback: {e}")

    
    @staticmethod
    def manage_image_memory(step_log: ActionStep, agent: ToolCallingAgent) -> None:
        """
        Memory management callback for vision agents.
        Removes old images to prevent token explosion while keeping recent ones for context.
        
        Following smolagents best practices: keep only last 2 steps with images.
        """
        current_step = step_log.step_number
        
        # Remove images from steps older than 2 steps ago
        for previous_step in agent.memory.steps:
            if isinstance(previous_step, ActionStep) and \
               previous_step.step_number <= current_step - 2:
                if previous_step.observations_images is not None:
                    print(f"  Clearing images from step {previous_step.step_number} to save memory")
                    previous_step.observations_images = None

                    
    def __init__(self):
        self.instructions = """
        You are an assistant for the cellpose-sam segmentation tool.

        ## PRIMARY WORKFLOW - IMAGE SEGMENTATION 

        When a user provides an image:
        1. use appropriate tools to review which cellpose-sam parameters are available. 
        2. use the tool: `get_segmentation_parameters`
           - **IMPORTANT**: After this tool runs, you will be able to SEE the input image
           - The image is automatically attached to the step for your visual inspection
        3. carefully analyze the image visually:
           - look at cell morphology, density, and boundaries 
           - compare what you see to the matched parameter values 
           - consider if adjustments would likely improve the segmentation
        4. Be conservative: if you make changes, assess if they should differ significantly from the original values
        5. Provide your final parameter recommendations in a clear, structured format 
        6. Use the parameters to run cellpose_sam through the tool: run_cellpose_sam
        7. after run_cellpose_sam, call the tool: refine_cellpose_sam_segmentation
           - **IMPORTANT**: After this tool runs, you will be able to SEE both the original and segmented images side-by-side
           - The images are automatically attached for your comparison
        8. You will visually examine BOTH images:
           - Identify segmentation issues: under-segmentation (masks don't cover full cells) or over-segmentation (masks extend beyond boundaries)
        If refinement is needed:
           - Use knowledge graph and RAG tools to understand parameter effects and value ranges
           - Decide which parameters to adjust and by how much based on your visual analysis
           - Re-run run_cellpose_sam with adjusted parameters   
        **CRITICAL: Call refine_cellpose_sam_segmentation AT MOST 2 TIMES total**
           - First call: Check initial segmentation quality
           - Second call (if needed): Verify refinement improved results
           - NEVER call it a third time - always stop after 2 refinement checks

        ## DOCUMENTATION QUERY WORKFLOW ##

        - "What is X": use `search_documentation_vector`
        - "How does X affect Y": use `search_knowledge_graph`  
        - Complex analysis: use `hybrid_search`
        - Parameter relationships: use `get_parameter_relationships`        

        ## RESPONSE STYLE ##
        - Be concise and actionable
        - Always explain your visual reasoning when adjusting parameters
        - If keeping original matched parameters, briefly confirm why it's appropriate
        - Images are provided to you automatically - do not ask for them

        """
        
        self.model = self._initialize_model()
        self.agent = self._create_agent()
        

    def _initialize_model(self):
        """Initializes the TransformersModel for the agent with VLM support."""
        clear_gpu_cache()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"            
        )        
        
        return TransformersModel(
            model_id=settings.AGENT_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float32,
            model_kwargs={
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": True
            }
        )

    def _create_agent(self):
        """Creates the ToolCallingAgent with all available tools and memory management."""
        return ToolCallingAgent(
            model=self.model,
            tools=all_tools,
            instructions=self.instructions,
            max_steps=10,
            step_callbacks=[
                self.attach_images_callback,
                self.manage_image_memory,
            ]
        )

    @observe()
    def run(self, task: str):
        """Runs the agent on a given task with Langfuse tracing."""
        print(f"\n{'='*60}\nTASK: {task}\n{'='*60}")
        
        langfuse.update_current_trace(
            input={"task": task},
            user_id="user_001",
            tags=["rag", "cellpose", "knowledge-graph", "vision"],
            metadata={"agent_type": "ToolCallingAgent", "model_id": settings.AGENT_MODEL_ID}
        )

        try:
            final_answer = self.agent.run(task)
            print("\n--- Final Answer from Agent ---\n", final_answer)
            langfuse.update_current_trace(output={"final_answer": final_answer})
            return final_answer
        except Exception as e:
            print(f"Agent run failed: {e}")
            langfuse.update_current_trace(output={"error": str(e)})
            raise
        finally:
            clear_gpu_cache()
