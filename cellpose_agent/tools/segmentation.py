"""
Segmentation tools for cellpose-sam pipeline
"""
import base64
import json
import re
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


langfuse = get_client()

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
def get_segmentation_parameters(image_path: str) -> str:
    """
    Finds the best cellpose-sam segmentation parameters for an image using vector similarity.
    Returns both a text recommendation and structured parameter data.
    
    Args:
        image_path (str): Path to the image file to segment.
    Returns:
        str: JSON string containing recommended parameters and analysis context
    """
    print(f"\n--- TOOL CALLED: get_segmentation_parameters for '{image_path}' ---")

    # Read image once and reuse
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine media type
        if image_path.lower().endswith('.png'):
            media_type = "image/png"
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"
            
    except Exception as e:
        print(f"Warning: Could not read image: {e}")
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

        # Return structured JSON response
        response = {
            "status": "success",
            "image_path": image_path,
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
    use_recommended_params: bool = True
) -> str:
    """
    Runs cellpose-sam segmentation pipeline on an image with specified parameters.
    If use_recommended_params is True and no parameters are provided, will automatically
    get recommended parameters for the image.
    
    Args:
        image_path (str): Path to the image file to segment
        diameter (int): Expected diameter of cells in pixels (default: auto-detect or 25)
        flow_threshold (float): Flow error threshold (default: auto-detect or 0.6, range: 0-1)
        cellprob_threshold (float): Cell probability threshold (default: auto-detect or 0, range: -6 to 6)
        min_size (int): Minimum cell size in pixels (default: auto-detect or 15)
        output_path (str): Optional path to save the overlay image (default: auto-generated)
        use_recommended_params (bool): If True and params not provided, get recommendations (default: True)
    
    Returns:
        str: Summary of segmentation results including cell count and output path
    """
    print(f"\n--- TOOL CALLED: run_cellpose_sam for '{image_path}' ---")
    
    # Auto-fetch recommended parameters if none provided
    if use_recommended_params and all(p is None for p in [diameter, flow_threshold, cellprob_threshold, min_size]):
        print("No parameters provided. Fetching recommended parameters...")
        param_response = get_segmentation_parameters(image_path)
        
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
                diameter = diameter or 25
                flow_threshold = flow_threshold or 0.6
                cellprob_threshold = cellprob_threshold or 0
                min_size = min_size or 15
        except json.JSONDecodeError:
            print("Could not parse parameter recommendations. Using defaults...")
            diameter = diameter or 25
            flow_threshold = flow_threshold or 0.6
            cellprob_threshold = cellprob_threshold or 0
            min_size = min_size or 15
    else:
        # Use provided values or defaults
        diameter = diameter or 25
        flow_threshold = flow_threshold or 0.6
        cellprob_threshold = cellprob_threshold or 0
        min_size = min_size or 15
    
    print(f"Final parameters: diameter={diameter}, flow_threshold={flow_threshold}, "
          f"cellprob_threshold={cellprob_threshold}, min_size={min_size}")
    
    try:
        # Read image once
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: Could not read image at {image_path}"
        
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
            return "No cells detected. Try adjusting parameters (lower flow_threshold or cellprob_threshold)."
        
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
        cv2.imwrite(output_path, colored_overlay.astype(np.uint8))
        
        # Log to Langfuse
        try:
            # Encode output image
            with open(output_path, "rb") as f:
                output_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            langfuse.update_current_trace(
                output={
                    "cell_count": int(masks_cellpose.max()),
                    "output_image": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{output_base64}"
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
        
        result = f"""
        Segmentation Complete!
        - Detected cells: {masks_cellpose.max()}
        - Output saved to: {output_path}
        - Parameters used:
        • diameter: {diameter}
        • flow_threshold: {flow_threshold}
        • cellprob_threshold: {cellprob_threshold}
        • min_size: {min_size}
        """
        return result.strip()
        
    except Exception as e:
        return f"Error during segmentation: {e}"
