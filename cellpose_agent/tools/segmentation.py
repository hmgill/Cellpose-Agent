"""
Segmentation tools for cellpose-sam pipeline with proper smolagents VLM integration.
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
from smolagents.agents import ActionStep
from langfuse import get_client
from stores import chroma_store
from models.embeddings import get_image_embedding
from utils.image_utils import resize_and_encode_image


langfuse = get_client()


# --- Global State and Caching ---
_image_cache: Dict[str, tuple[str, str]] = {}
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
    """Extract parameter values from parameter text string."""
    defaults = {
        'diameter': 25,
        'flow_threshold': 0.6,
        'cellprob_threshold': 0,
        'min_size': 15
    }
    
    params = defaults.copy()
    
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
    The image will be visible to the VLM for visual analysis.
    
    Args:
        image_path (str): Path to the image file to segment.
        agent (Any, optional): The agent instance, passed automatically by smol-agents.
    
    Returns:
        str: JSON string containing recommended parameters and analysis context
             (NO base64 to avoid GPU OOM)
    """
    print(f"\n--- TOOL CALLED: get_segmentation_parameters for '{image_path}' ---")

    try:
        # Load and cache image (for internal use)
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

        # Parse parameters
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

        # Log to Langfuse WITH image (for observability)
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

        # Return WITHOUT base64 (image already attached to ActionStep)
        response = {
            "status": "success",
            "image_path": image_path,
            "recommended_parameters": params,
            "matched_image": matched_image,
            "similarity_distance": float(distance),
            "confidence": confidence,
            "image_stats": stats,
            "raw_parameter_text": matched_parameters,
            "visual_guidance": "IMAGE NOW VISIBLE: The input image is now attached to this step. "
                              "Please visually inspect the image to assess cell morphology, density, "
                              "and boundaries before deciding whether to adjust the recommended parameters.",
            "recommendation": f"{confidence_note}\n\nRecommended parameters:\n"
                             f"- diameter: {params['diameter']}\n"
                             f"- flow_threshold: {params['flow_threshold']}\n"
                             f"- cellprob_threshold: {params['cellprob_threshold']}\n"
                             f"- min_size: {params['min_size']}\n\n"
                             f"Image stats: {image_shape[0]}x{image_shape[1]} pixels, "
                             f"mean intensity {stats['mean_intensity']:.1f}\n\n"
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
    Returns results WITHOUT base64 images to prevent GPU memory issues.
    
    Args:
        image_path (str): Path to the image file to segment
        diameter (int): Expected diameter of cells in pixels
        flow_threshold (float): Flow error threshold (range: 0-1)
        cellprob_threshold (float): Cell probability threshold (range: -6 to 6)
        min_size (int): Minimum cell size in pixels
        output_path (str): Optional path to save the overlay image
        use_recommended_params (bool): If True and params not provided, get recommendations
        agent (Any, optional): The agent instance
    
    Returns:
        str: JSON string with segmentation results (paths and stats, NO base64)
    """
    print(f"\n--- TOOL CALLED: run_cellpose_sam for '{image_path}' ---")
    
    try:
        # Load and cache input image
        input_image_base64, input_media_type = _get_cached_image(image_path) or _load_and_cache_image(image_path)
    except Exception as e:
        return json.dumps({"error": f"Could not read input image: {e}"})
    
    # Auto-fetch recommended parameters if needed
    if use_recommended_params and all(p is None for p in [diameter, flow_threshold, cellprob_threshold, min_size]):
        print("No parameters provided. Fetching recommended parameters...")
        param_response = get_segmentation_parameters(image_path, agent=agent)
        
        try:
            param_data = json.loads(param_response)
            if param_data.get("status") == "success":
                rec_params = param_data["recommended_parameters"]
                diameter = diameter or rec_params.get('diameter', 25)
                flow_threshold = flow_threshold or rec_params.get('flow_threshold', 0.6)
                cellprob_threshold = cellprob_threshold or rec_params.get('cellprob_threshold', 0)
                min_size = min_size or rec_params.get('min_size', 15)
            else:
                diameter, flow_threshold, cellprob_threshold, min_size = 25, 0.6, 0, 15
        except json.JSONDecodeError:
            diameter, flow_threshold, cellprob_threshold, min_size = 25, 0.6, 0, 15
    else:
        diameter = diameter if diameter is not None else 25
        flow_threshold = flow_threshold if flow_threshold is not None else 0.6
        cellprob_threshold = cellprob_threshold if cellprob_threshold is not None else 0
        min_size = min_size if min_size is not None else 15
    
    print(f"Final parameters: diameter={diameter}, flow_threshold={flow_threshold}, "
          f"cellprob_threshold={cellprob_threshold}, min_size={min_size}")
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return json.dumps({"error": f"Could not read image at {image_path}"})
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cellpose_model = get_cellpose_model()
        sam_predictor = get_sam_predictor()
        
        # Run Cellpose
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
                "message": "No cells detected. Try adjusting parameters.",
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "min_size": min_size
                }
            })
        
        print(f"Cellpose detected {masks_cellpose.max()} regions")
        
        # SAM refinement
        sam_predictor.set_image(img_rgb)
        props = regionprops(masks_cellpose)
        boxes = np.array([prop.bbox for prop in props])
        boxes = boxes[:, [1,0,3,2]]
        
        print(f"Refining {len(boxes)} masks with SAM...")
        
        combined_masks = np.zeros(img_rgb.shape[:2], dtype=np.uint16)
        colored_overlay = img_rgb.copy().astype(np.float32)
        
        for i, box in enumerate(boxes):
            masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
            best_mask = masks[np.argmax(scores)]
            combined_masks[best_mask] = i + 1
            color = np.random.randint(0, 255, 3)
            colored_overlay[best_mask] = colored_overlay[best_mask] * 0.6 + color * 0.4
        
        # Generate output path
        if output_path is None:
            base_name = image_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_cellpose_sam_overlay.png"
        
        # Save output
        cv2.imwrite(output_path, cv2.cvtColor(colored_overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Load and cache output image
        output_image_base64, output_media_type = _load_and_cache_image(output_path)
        
        # Log to Langfuse WITH both images
        try:
            langfuse.update_current_trace(
                input={
                    "image_path": image_path,
                    "input_image": {
                        "type": "image_url",
                        "image_url": {"url": f"data:{input_media_type};base64,{input_image_base64}"}
                    }
                },
                output={
                    "cell_count": int(masks_cellpose.max()),
                    "output_image": {
                        "type": "image_url",
                        "image_url": {"url": f"data:{output_media_type};base64,{output_image_base64}"}
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
        
        # Return WITHOUT base64
        result = {
            "status": "success",
            "cell_count": int(masks_cellpose.max()),
            "output_path": output_path,
            "input_path": image_path,
            "parameters": {
                "diameter": diameter,
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "min_size": min_size
            },
            "summary": f"Detected {masks_cellpose.max()} cells. Output saved to: {output_path}",
            "next_step": "Call refine_cellpose_sam_segmentation to visually analyze the segmentation quality and decide if parameter adjustments are needed."
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error during segmentation: {e}"})


@tool
def refine_cellpose_sam_segmentation(
    original_image_path: str,
    segmentation_output_path: str,
    current_parameters: dict,
    agent: Any = None,
) -> str:
    """
    Provides both original and segmented images to the VLM for visual quality assessment.
    The VLM will be able to see both images and provide informed analysis.
    
    Use this tool after run_cellpose_sam to check segmentation quality. The tool attaches
    both images to the current step so you can visually compare them.
    
    Before calling, consider using search_knowledge_graph or hybrid_search to refresh
    your understanding of how cellpose parameters affect segmentation.
    
    Common issues and fixes:
    - Under-segmentation (cells merged): decrease flow_threshold or diameter
    - Over-segmentation (cells fragmented): increase flow_threshold or min_size
    - Too few cells: decrease cellprob_threshold or flow_threshold
    - Too many false positives: increase cellprob_threshold or min_size
    
    Args:
        original_image_path: Path to the original input image
        segmentation_output_path: Path to the segmented overlay image
        current_parameters: Dict with current diameter, flow_threshold, cellprob_threshold, min_size
        agent: The agent instance (passed automatically)
    
    Returns:
        str: JSON with guidance for VLM analysis (NO base64 images)
    """
    print(f"\n--- TOOL CALLED: refine_cellpose_sam_segmentation ---")
    print(f"Original image: {original_image_path}")
    print(f"Segmented image: {segmentation_output_path}")
    print(f"Current parameters: {current_parameters}")
    
    try:
        # Load both images (for cache)
        original_b64, original_type = _get_cached_image(original_image_path) or _load_and_cache_image(original_image_path)
        segmented_b64, segmented_type = _get_cached_image(segmentation_output_path) or _load_and_cache_image(segmentation_output_path)
        
        # CRITICAL: Attach BOTH images to ActionStep so VLM can see them
        if agent is not None and hasattr(agent, 'memory') and hasattr(agent.memory, 'steps'):
            current_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
            if current_steps:
                current_step = current_steps[-1]
                
                # Load both as PIL Images
                original_img = Image.open(original_image_path).convert("RGB")
                segmented_img = Image.open(segmentation_output_path).convert("RGB")
                
                # CRITICAL: Use .copy() for both images
                current_step.observations_images = [original_img.copy(), segmented_img.copy()]
                print(f"✓ Attached both images to ActionStep for VLM comparison")
        
        # Get image dimensions for context
        original_img_array = np.array(Image.open(original_image_path).convert("RGB"))
        img_size = original_img_array.shape[0] * original_img_array.shape[1]
        
        # Log to Langfuse WITH both images
        try:
            langfuse.update_current_trace(
                input={
                    "tool": "refine_cellpose_sam_segmentation",
                    "original_image": {
                        "type": "image_url",
                        "image_url": {"url": f"data:{original_type};base64,{original_b64}"}
                    },
                    "segmented_image": {
                        "type": "image_url",
                        "image_url": {"url": f"data:{segmented_type};base64,{segmented_b64}"}
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
        
        # Return analysis guidance WITHOUT base64
        analysis = {
            "status": "ready_for_visual_analysis",
            "images_attached": "BOTH IMAGES NOW VISIBLE: The first image is the original input, "
                              "the second is the segmented overlay. Compare them visually to assess quality.",
            "image_paths": {
                "original": original_image_path,
                "segmented": segmentation_output_path
            },
            "current_parameters": current_parameters,
            "image_info": {
                "dimensions": f"{original_img_array.shape[1]}x{original_img_array.shape[0]}",
                "total_pixels": img_size
            },
            "visual_analysis_checklist": [
                "1. Do the colored masks accurately cover entire cells without extending beyond boundaries?",
                "2. Are neighboring cells properly separated, or are they merged together?",
                "3. Are there many small false positive detections (noise)?",
                "4. Are any large, obvious cells being missed completely?",
                "5. Overall quality assessment: excellent, good, needs_refinement, or poor?"
            ],
            "parameter_adjustment_guide": {
                "under_segmentation": {
                    "symptoms": "Masks don't reach cell edges, cells appear merged",
                    "solution": "Decrease flow_threshold by 0.1-0.2 OR decrease diameter by 10-20%"
                },
                "over_segmentation": {
                    "symptoms": "Masks extend past boundaries, cells fragmented into pieces",
                    "solution": "Increase flow_threshold by 0.1-0.2 OR increase min_size to 2-3x current value"
                },
                "too_few_cells": {
                    "symptoms": "Obvious cells in image are not being detected",
                    "solution": "Decrease cellprob_threshold by 1-2 OR decrease flow_threshold by 0.1-0.2"
                },
                "too_many_false_positives": {
                    "symptoms": "Many tiny spurious detections, background noise detected as cells",
                    "solution": "Increase cellprob_threshold by 1-2 OR increase min_size to 2-3x current value"
                }
            },
            "next_steps": {
                "if_good": "If segmentation looks accurate, inform the user of success and provide the output_path.",
                "if_needs_refinement": "Based on your visual analysis, adjust the appropriate parameters and call run_cellpose_sam again with the new values.",
                "important": "You can only call refine_cellpose_sam_segmentation AT MOST 2 TIMES total. If this is your second call, you must make a final decision."
            }
        }
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "message": "Could not load images for refinement. Check that both file paths are valid."
        }
        return json.dumps(error_result, indent=2)
