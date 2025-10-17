"""
Manager Agent that coordinates safety checks and cellpose segmentation
"""
import torch
from smolagents import ToolCallingAgent, TransformersModel
from transformers import BitsAndBytesConfig
from langfuse import get_client, observe

from config import settings
from utils.gpu import clear_gpu_cache
from agents.safety_agent import SafetyAgent
from agents.cellpose_agent import CellposeAgent


langfuse = get_client()


class ManagerAgent:
    """
    Orchestrates the workflow between safety checking and cellpose segmentation.
    """
    
    def __init__(self):
        self.instructions = """
        You are a manager agent coordinating image segmentation workflows.
        
        ## YOUR ROLE
        
        You coordinate two specialized agents:
        1. **safety_agent**: Checks if user inputs are appropriate and on-topic
        2. **cellpose_agent**: Performs cell segmentation analysis
        
        ## WORKFLOW
        
        For every user request:
        
        1. **ALWAYS check safety first**:
           - Extract the user's text query and any image path they provided
           - Delegate to safety_agent to verify the request is appropriate
           - Wait for safety_agent's response
        
        2. **Process based on safety results**:
           - If safety_agent returns `safe: false`:
             * DO NOT proceed with segmentation
             * Politely inform the user why their request cannot be processed
             * Suggest what types of requests are appropriate (biological/scientific images)
           
           - If safety_agent returns `safe: true`:
             * Delegate the segmentation task to cellpose_agent
             * Let cellpose_agent handle all parameter selection and segmentation
             * Return cellpose_agent's final results to the user
        
        3. **Communication style**:
           - Be professional and helpful
           - For rejected requests, be polite but firm
           - For approved requests, let the cellpose_agent handle the technical details
           - Summarize results clearly for the user
        
        ## IMPORTANT NOTES
        
        - You do NOT perform segmentation yourself - always delegate to cellpose_agent
        - You do NOT skip safety checks - they are mandatory for every request
        - You coordinate the workflow but let specialized agents do their jobs
        - Focus on high-level orchestration and user communication
        """
        
        self.model = self._initialize_model()
        self.safety_agent = SafetyAgent()
        self.cellpose_agent = CellposeAgent()
        self.manager = self._create_manager()
    
    def _initialize_model(self):
        """Initialize the manager's language model."""
        clear_gpu_cache()
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        return TransformersModel(
            model_id=settings.MANAGER_AGENT_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float32,
            model_kwargs={
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": True
            }
        )
    
    def _create_manager(self):
        """Create the manager CodeAgent with managed agents."""
        # Convert sub-agents to ToolCallingAgents for managed_agents
        safety_tool_agent = self.safety_agent.agent
        cellpose_tool_agent = self.cellpose_agent.agent
        
        return ToolCallingAgent(
            tools=[],  
            model=self.model,
            managed_agents=[safety_tool_agent, cellpose_tool_agent],
            max_steps=10,
        )
    
    @observe()
    def run(self, task: str, image_path: str = None):
        """
        Run the manager agent on a task with optional image.
        
        Args:
            task (str): The user's query/request
            image_path (str, optional): Path to image file if provided
        
        Returns:
            The final result from the workflow
        """
        print(f"\n{'='*60}\nMANAGER AGENT TASK\n{'='*60}")
        print(f"Task: {task}")
        if image_path:
            print(f"Image: {image_path}")
        
        # Construct the manager's prompt
        manager_prompt = f"""
        User request: {task}
        """
        
        if image_path:
            manager_prompt += f"\nImage provided: {image_path}"
        
        manager_prompt += """
        
        Execute the workflow:
        1. First, use safety_agent to check if this request is safe and appropriate
        2. Based on the safety check results, either:
           - Politely decline and explain if unsafe
           - Proceed with cellpose_agent for segmentation if safe
        """
        
        langfuse.update_current_trace(
            input={
                "task": task,
                "image_path": image_path,
                "agent_type": "manager"
            },
            user_id="user_001",
            tags=["manager", "multi-agent", "cellpose", "safety"],
            metadata={
                "model_id": settings.MANAGER_AGENT_MODEL_ID,
                "managed_agents": ["safety_agent", "cellpose_agent"]
            }
        )
        
        try:
            final_answer = self.manager.run(manager_prompt)
            print("\n--- Final Answer from Manager ---\n", final_answer)
            
            langfuse.update_current_trace(
                output={"final_answer": final_answer}
            )
            
            return final_answer
            
        except Exception as e:
            print(f"Manager agent failed: {e}")
            langfuse.update_current_trace(
                output={"error": str(e)}
            )
            raise
        finally:
            clear_gpu_cache()
