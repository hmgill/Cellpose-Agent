"""
Segmentation tools for cellpose-sam pipeline, optimized for smol-agents with image caching.
"""
import base64
import json
import re
from typing import Any, Dict, TYPE_CHECKING
import numpy as np
import cv2
import torch
from PIL import Image
from skimage.measure import regionprops
from cellpose import models
from segment_anything import sam_model_registry, SamPredictor

from smolagents import tool
from langfuse import get_client
from stores import chroma_store
from models.embeddings import get_image_embedding
from utils.image_utils import resize_and_encode_image

# The TYPE_CHECKING block is for static type checkers (like mypy).
# It helps them understand the intended type is a smol-agent,
# but it is not executed at runtime, thus avoiding import errors.
if TYPE_CHECKING:
    from smolagents.agent import BaseAgent

langfuse = get_client()

# --- Global State and Caching ---

# In-memory cache to store loaded images (base64 encoded) by their path.
# This avoids reloading the same image in subsequent tool calls within the same agent run.
_image_cache: Dict[str, tuple[str, str]] = {}

# Initialize models once (singleton pattern)
_cellpose_model = None
_sam_predictor = None

def get_cellpose_model():
    """Initialize Cellpose model (singleton)"""
    global _cellpose_model
    if _cellpose_model is None:
        print("Initializing Cellpose model...")
        _cellpose_model = models.CellposeModel(gpu=torch.cuda.is_available())
        print("✓ Cellpose model initialized")
    return _cellpose_model

def get_sam_predictor():
    """Initialize SAM predictor (singleton)"""
    global _sam_predictor
    if _sam_predictor is None:
        print("Initializing SAM predictor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
        print("✓ SAM predictor initialized")
    return _sam_predictor

def _get_cached_image(image_path: str) -> tuple[str, str] | None:
    """Helper to retrieve an image from the cache."""
    if image_path in _image_cache:
        print(f"Found image in cache: '{image_path}'")
        return _image_cache[image_path]
    return None

def _load_and_cache_image(image_path: str) -> tuple[str, str]:
    """Helper to load, encode, and cache an image."""
    print(f"Loading and caching image: '{image_path}'")
    image_base64, media_type = resize_and_encode_image(image_path)
    _image_cache[image_path] = (image_base64, media_type)
    return image_base64, media_type

def parse_parameters_from_text(param_text: str) -> dict:
    """
    Extract parameter values from parameter text string.
    
    Args:
        param_text: String like "diameter=25, flow_threshold=0.6, cellprob_threshold=0, min_size=15"
    
    Returns:
        dict: Parsed parameters with default values for missing ones
    """
    defaults = {
        'diameter': 25,
        'flow_threshold': 0.6,
        'cellprob_threshold': 0,
        'min_size': 15
    }
    
    params = defaults.copy()
    
    # Try to extract parameters using regex
    patterns = {
        'diameter': r'diameter[=:]\s*(\d+)',
        'flow_threshold': r'flow_threshold[=:]\s*([\d.]+)',
        'cellprob_threshold': r'cellprob_threshold[=:]\s*([-\d.]+)',
        'min_size': r'min_size[=:]\s*(\d+)'
    }
    
    for param_name, pattern in patterns.items():
        match = re.search(pattern, param_text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if param_name in ['diameter', 'min_size']:
                params[param_name] = int(value)
            else:
                params[param_name] = float(value)
    
    return params


@tool
def get_segmentation_parameters(image_path: str, agent: Any = None) -> str:
    """
    Finds the best cellpose-sam segmentation parameters for an image using vector similarity.
    Returns both a text recommendation and the resized image for VLM viewing.
    
    Args:
        image_path (str): Path to the image file to segment.
        agent (Any, optional): The agent instance, passed automatically by smol-agents.
    Returns:
        str: JSON string containing recommended parameters, analysis context, and base64 image
    """
    print(f"\n--- TOOL CALLED: get_segmentation_parameters for '{image_path}' ---")

    try:
        # Check cache first, otherwise load and cache the image
        image_base64, media_type = _get_cached_image(image_path) or _load_and_cache_image(image_path)
    except Exception as e:
        print(f"Warning: Could not read/resize image: {e}")
        return json.dumps({"error": f"Could not read image: {e}"})

    try:
        # Get similar parameters from ChromaDB
        client = chroma_store.get_client()
        collection = client.get_collection(name='cellpose-sam_parameters_by_image_similarity')
        query_embedding = get_image_embedding(image_path)
        
        results = collection.query(query_embeddings=[query_embedding], n_results=1)

        if not (results['metadatas'] and results['metadatas'][0]):
            return json.dumps({"error": "No similar images found in the database."})

        matched_parameters = results['metadatas'][0][0].get('parameter_text', 'N/A')
        matched_image = results['metadatas'][0][0].get('image_name', 'N/A')
        distance = results['distances'][0][0]

        print(f"Most similar: {matched_image} (distance: {distance:.3f})")
        print(f"Recommended: {matched_parameters}")

        # Parse parameters into structured format
        params = parse_parameters_from_text(matched_parameters)

        # Analyze image
        image = np.array(Image.open(image_path).convert("RGB"))
        image_shape = image.shape
        stats = {
            'size': (image_shape[0] * image_shape[1]),
            'mean_intensity': float(np.mean(image)),
            'stdev_intensity': float(np.std(image)),
            'min_intensity': int(np.min(image)),
            'max_intensity': int(np.max(image)),
        }

        # Log to Langfuse with image
        try:
            langfuse.update_current_trace(
                input={
                    "image_path": image_path,
                    "query_image": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        }
                    },
                    "image_stats": stats
                },
                metadata={
                    "matched_image": matched_image,
                    "similarity_distance": float(distance),
                    "matched_parameters": matched_parameters,
                    "parsed_parameters": params
                }
            )
        except Exception as log_error:
            print(f"Warning: Could not log to Langfuse: {log_error}")

        # Determine confidence level
        if distance < 0.2:
            confidence = "high"
            confidence_note = "Very similar image found. Parameters should work well as-is."
        elif distance < 0.4:
            confidence = "medium"
            confidence_note = "Similar image found. Parameters are a good starting point but may need minor adjustments."
        else:
            confidence = "low"
            confidence_note = "No very similar images found. Parameters may need significant adjustment based on visual inspection."

        # Return structured JSON response with image embedded
        response = {
            "status": "success",
            "image_path": image_path,
            "image_base64": image_base64,
            "image_media_type": media_type,
            "recommended_parameters": params,
            "matched_image": matched_image,
            "similarity_distance": float(distance),
            "confidence": confidence,
            "image_stats": stats,
            "raw_parameter_text": matched_parameters,
            "recommendation": f"{confidence_note}\n\nRecommended parameters:\n"
                             f"- diameter: {params['diameter']}\n"
                             f"- flow_threshold: {params['flow_threshold']}\n"
                             f"- cellprob_threshold: {params['cellprob_threshold']}\n"
                             f"- min_size: {params['min_size']}\n\n"
                             f"To run segmentation, use: run_cellpose_sam(image_path='{image_path}', "
                             f"diameter={params['diameter']}, flow_threshold={params['flow_threshold']}, "
                             f"cellprob_threshold={params['cellprob_threshold']}, min_size={params['min_size']})"
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_cellpose_sam(
    image_path: str,
    diameter: int = None,
    flow_threshold: float = None,
    cellprob_threshold: float = None,
    min_size: int = None,
    output_path: str = None,
    use_recommended_params: bool = True,
    agent: Any = None
) -> str:
    """
    Runs cellpose-sam segmentation pipeline on an image with specified parameters.
    Returns results with both input and output images for VLM viewing.
    
    Args:
        image_path (str): Path to the image file to segment
        diameter (int): Expected diameter of cells in pixels (default: auto-detect or 25)
        flow_threshold (float): Flow error threshold (default: auto-detect or 0.6, range: 0-1)
        cellprob_threshold (float): Cell probability threshold (default: auto-detect or 0, range: -6 to 6)
        min_size (int): Minimum cell size in pixels (default: auto-detect or 15)
        output_path (str): Optional path to save the overlay image (default: auto-generated)
        use_recommended_params (bool): If True and params not provided, get recommendations (default: True)
        agent (Any, optional): The agent instance, passed automatically by smol-agents.
    
    Returns:
        str: JSON string with segmentation results, input image, and output image
    """
    print(f"\n--- TOOL CALLED: run_cellpose_sam for '{image_path}' ---")
    
    try:
        input_image_base64, input_media_type = _get_cached_image(image_path) or _load_and_cache_image(image_path)
    except Exception as e:
        return json.dumps({"error": f"Could not read input image: {e}"})
    
    # Auto-fetch recommended parameters if none provided
    if use_recommended_params and all(p is None for p in [diameter, flow_threshold, cellprob_threshold, min_size]):
        print("No parameters provided. Fetching recommended parameters...")
        param_response = get_segmentation_parameters(image_path, agent=agent) # Pass agent along
        
        try:
            param_data = json.loads(param_response)
            if param_data.get("status") == "success":
                rec_params = param_data["recommended_parameters"]
                diameter = diameter or rec_params.get('diameter', 25)
                flow_threshold = flow_threshold or rec_params.get('flow_threshold', 0.6)
                cellprob_threshold = cellprob_threshold or rec_params.get('cellprob_threshold', 0)
                min_size = min_size or rec_params.get('min_size', 15)
                print(f"Using recommended parameters: diameter={diameter}, flow_threshold={flow_threshold}, "
                      f"cellprob_threshold={cellprob_threshold}, min_size={min_size}")
            else:
                print(f"Could not get recommendations: {param_data.get('error', 'Unknown error')}")
                print("Using default parameters...")
                diameter = 25
                flow_threshold = 0.6
                cellprob_threshold = 0
                min_size = 15
        except json.JSONDecodeError:
            print("Could not parse parameter recommendations. Using defaults...")
            diameter = 25
            flow_threshold = 0.6
            cellprob_threshold = 0
            min_size = 15
    else:
        # Use provided values or defaults
        diameter = diameter if diameter is not None else 25
        flow_threshold = flow_threshold if flow_threshold is not None else 0.6
        cellprob_threshold = cellprob_threshold if cellprob_threshold is not None else 0
        min_size = min_size if min_size is not None else 15
    
    print(f"Final parameters: diameter={diameter}, flow_threshold={flow_threshold}, "
          f"cellprob_threshold={cellprob_threshold}, min_size={min_size}")
    
    try:
        # Read image once
        img = cv2.imread(image_path)
        if img is None:
            return json.dumps({"error": f"Could not read image at {image_path}"})
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize models
        cellpose_model = get_cellpose_model()
        sam_predictor = get_sam_predictor()
        
        # Run Cellpose to get initial masks
        print("Running Cellpose...")
        masks_cellpose, flows, styles = cellpose_model.eval(
            img_rgb,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size
        )
        
        if masks_cellpose.max() == 0:
            return json.dumps({
                "status": "no_cells_detected",
                "message": "No cells detected. Try adjusting parameters (lower flow_threshold or cellprob_threshold).",
                "input_image_base64": input_image_base64,
                "input_image_media_type": input_media_type,
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "min_size": min_size
                }
            })
        
        print(f"Cellpose detected {masks_cellpose.max()} regions")
        
        # Set image for SAM
        sam_predictor.set_image(img_rgb)
        
        # Extract bounding boxes from Cellpose masks
        props = regionprops(masks_cellpose)
        boxes = np.array([prop.bbox for prop in props])  # (y1,x1,y2,x2) format
        boxes = boxes[:, [1,0,3,2]]  # convert to (x1,y1,x2,y2) for SAM
        
        print(f"Refining {len(boxes)} masks with SAM...")
        
        # Combine all refined masks
        combined_masks = np.zeros(img_rgb.shape[:2], dtype=np.uint16)
        colored_overlay = img_rgb.copy().astype(np.float32)
        
        for i, box in enumerate(boxes):
            masks, scores, _ = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            
            # Use best scoring mask
            best_mask = masks[np.argmax(scores)]
            combined_masks[best_mask] = i + 1  # assign unique ID
            
            # Create colored overlay
            color = np.random.randint(0, 255, 3)
            colored_overlay[best_mask] = colored_overlay[best_mask] * 0.6 + color * 0.4
        
        # Generate output path if not provided
        if output_path is None:
            base_name = image_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_cellpose_sam_overlay.png"
        
        # Save results
        cv2.imwrite(output_path, cv2.cvtColor(colored_overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Load and cache the newly created output image
        output_image_base64, output_media_type = _load_and_cache_image(output_path)
        
        # Log to Langfuse with both images
        try:
            langfuse.update_current_trace(
                input={
                    "image_path": image_path,
                    "input_image": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{input_media_type};base64,{input_image_base64}"
                        }
                    }
                },
                output={
                    "cell_count": int(masks_cellpose.max()),
                    "output_image": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{output_media_type};base64,{output_image_base64}"
                        }
                    },
                    "output_path": output_path
                },
                metadata={
                    "parameters": {
                        "diameter": diameter,
                        "flow_threshold": flow_threshold,
                        "cellprob_threshold": cellprob_threshold,
                        "min_size": min_size
                    }
                }
            )
        except Exception as log_error:
            print(f"Warning: Could not log output to Langfuse: {log_error}")
        
        # Return structured JSON with both images
        result = {
            "status": "success",
            "cell_count": int(masks_cellpose.max()),
            "output_path": output_path,
            "input_image_base64": input_image_base64,
            "input_image_media_type": input_media_type,
            "output_image_base64": output_image_base64,
            "output_image_media_type": output_media_type,
            "parameters": {
                "diameter": diameter,
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "min_size": min_size
            },
            "summary": f"Detected {masks_cellpose.max()} cells. Output saved to: {output_path}"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error during segmentation: {e}"})


@tool
def refine_segmentation(
    original_image_path: str,
    segmentation_output_path: str,
    current_parameters: dict,
    agent: Any = None,
) -> str:
    """
    Provides original and segmented images for visual analysis to determine if refinement is needed.
    
    Use this tool after run_cellpose_sam to check segmentation quality. The tool returns
    both images and current parameters for you to visually assess the results.
    
    Before calling this tool, consider using search_knowledge_graph or hybrid_search to
    refresh your understanding of how cellpose parameters affect segmentation:
    - flow_threshold: controls cell boundary detection (lower = more permissive)
    - cellprob_threshold: controls cell probability cutoff (lower = more cells detected)
    - diameter: expected cell size in pixels
    - min_size: minimum object size to keep
    
    After visual analysis, use your knowledge to decide if parameters should be adjusted.
    Common issues:
    - Under-segmentation (cells merged): decrease flow_threshold or diameter
    - Over-segmentation (cells fragmented): increase flow_threshold or min_size
    - Too few cells: decrease cellprob_threshold or flow_threshold
    - Too many false positives: increase cellprob_threshold or min_size
    
    Args:
        original_image_path: Path to the original input image
        segmentation_output_path: Path to the segmented overlay image
        current_parameters: Dict with current diameter, flow_threshold, cellprob_threshold, min_size
        agent (Any, optional): The agent instance, passed automatically by smol-agents.
    
    Returns:
        str: JSON string with base64-encoded images and current parameters for analysis
    """
    print(f"\n--- TOOL CALLED: refine_segmentation ---")
    print(f"Original image: {original_image_path}")
    print(f"Segmented image: {segmentation_output_path}")
    print(f"Current parameters: {current_parameters}")
    
    try:
        # Get both images from cache or load them if they're not present
        original_b64, original_type = _get_cached_image(original_image_path) or _load_and_cache_image(original_image_path)
        segmented_b64, segmented_type = _get_cached_image(segmentation_output_path) or _load_and_cache_image(segmentation_output_path)
        
        # Return structured data with embedded base64 images
        result = {
            "status": "ready_for_analysis",
            "original_image": {
                "path": original_image_path,
                "base64": original_b64,
                "media_type": original_type
            },
            "segmented_image": {
                "path": segmentation_output_path,
                "base64": segmented_b64,
                "media_type": segmented_type
            },
            "current_parameters": current_parameters,
            "note": "Visually compare the original and segmented images. Check for under-segmentation (masks do not meet cell boundaries) or over-segmentation (masks are excessively outside of cell boundaries). Use your RAG tools to understand how to adjust parameters if refinement is needed.",
            "next_steps": "Analyze both images visually. If refinement is needed, call run_cellpose_sam again with adjusted parameters. If segmentation looks good, report success to the user."
        }
        
        # Log to Langfuse with both images
        try:
            langfuse.update_current_trace(
                input={
                    "tool": "refine_segmentation",
                    "original_image": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{original_type};base64,{original_b64}"
                        }
                    },
                    "segmented_image": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{segmented_type};base64,{segmented_b64}"
                        }
                    },
                    "current_parameters": current_parameters
                },
                metadata={
                    "original_path": original_image_path,
                    "segmented_path": segmentation_output_path
                }
            )
        except Exception as log_error:
            print(f"Warning: Could not log to Langfuse: {log_error}")
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "message": "Could not load images for refinement analysis. Please check that both file paths are valid."
        }
        return json.dumps(error_result, indent=2)
