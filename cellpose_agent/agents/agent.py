"""

"""
import torch
from datetime import datetime
from smolagents import ToolCallingAgent, TransformersModel
from langfuse import get_client, observe
from transformers import BitsAndBytesConfig

from config import settings
from utils.gpu import clear_gpu_cache
from tools import all_tools

langfuse = get_client()

class CellposeAgent:
    def __init__(self):
        self.instructions = """
        You are an assistant for the cellpose-sam segmentation tool.

        ## PRIMARY WORKFLOW - IMAGE SEGMENTATION 

        When a user provides an image:
        1. use appropriate tools to review which cellpose-sam parameters are available. 
        2. use the tool: `get_segmentation_parameters`
        3. **IMPORTANT**: you will receive (a) matched parameters from an image similarity search and (b) can see the actual image
        4. carefully analyze the image visually. 
        - look at cell morphology, density, and boundaries 
        - compare what you see to the matched parameter values 
        - consider if adjustments would likely improve the segmentation or not 
        5. Be conservative: if you make changes, assess if they should differ significantly from the original values or not 
        6. Provide your final parameter recommendations in a clear, structured format 
        7. Use the parameter segmentations to run cellpose_sam through the tool: run_cellpose_sam
        8. after run_cellpose_sam, call the tool: refine_segmentation. 
        9. you will receive (a) the original image, (b) the segmented image, (c) the cellpose-sam parameters used to create the segmented image.
        You will: 
           - Examine BOTH the original and segmented images carefully
           - Identify segmentation issues: under-segmentation (masks cover too little of the cells) or over-segmentation (masks go over cell boundaries)
        If refinement is needed:
           - Use knowledge graph and RAG tools to understand parameter effects and value ranges
           - Decide which parameters to adjust and by how much based on your knowledge
           - Re-run run_cellpose_sam with adjusted parameters   
        **CRITICAL: Call refine_segmentation AT MOST 2 TIMES total**
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

        """
        
        self.model = self._initialize_model()
        self.agent = self._create_agent()
        

    def _initialize_model(self):
        """Initializes the TransformersModel for the agent."""
        clear_gpu_cache()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )        
        
        return TransformersModel(
            model_id=settings.AGENT_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float32,
            model_kwargs={
                "quantization_config": quantization_config,
                "max_memory": settings.MAX_MEMORY                
            }
        )

    def _create_agent(self):
        """Creates the ToolCallingAgent with all available tools."""
        return ToolCallingAgent(
            model=self.model,
            tools=all_tools,
            instructions=self.instructions,
            max_steps=10,
        )

    @observe()
    def run(self, task: str):
        """Runs the agent on a given task with Langfuse tracing."""
        print(f"\n{'='*60}\nTASK: {task}\n{'='*60}")
        
        langfuse.update_current_trace(
            input={"task": task},
            user_id="user_001",
            tags=["rag", "cellpose", "knowledge-graph"],
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
