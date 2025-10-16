"""
Safety checking tools for content moderation
"""
import json
from typing import Any
from PIL import Image
import numpy as np

from smolagents import tool


@tool
def check_text_safety(text: str) -> str:
    """
    Analyzes text content for safety and appropriateness.
    
    Checks for:
    - Scientific/biological relevance
    - Harmful or inappropriate language
    - Privacy concerns
    - Attempts to misuse the system
    
    Args:
        text (str): The user's text query to analyze
        
    Returns:
        str: JSON string with safety assessment
    """
    print(f"\n--- TOOL CALLED: check_text_safety ---")
    print(f"Analyzing text: {text[:100]}...")
    
    # Keywords that indicate legitimate use
    scientific_keywords = [
        'cell', 'cells', 'segmentation', 'microscopy', 'image', 'biology',
        'cellpose', 'parameters', 'diameter', 'threshold', 'mask', 'roi',
        'tissue', 'nucleus', 'nuclei', 'organelle', 'cytoplasm', 'membrane'
    ]
    
    # Keywords that might indicate inappropriate use
    warning_keywords = [
        'hack', 'exploit', 'bypass', 'override', 'jailbreak',
        'ignore instructions', 'forget previous', 'violence', 'weapon'
    ]
    
    # Privacy-related terms
    privacy_keywords = [
        'personal information', 'credit card', 'ssn', 'social security',
        'password', 'private', 'confidential', 'medical record'
    ]
    
    text_lower = text.lower()
    
    # Check for scientific relevance
    has_scientific_content = any(kw in text_lower for kw in scientific_keywords)
    
    # Check for warning signs
    has_warning_signs = any(kw in text_lower for kw in warning_keywords)
    
    # Check for privacy concerns
    has_privacy_concerns = any(kw in text_lower for kw in privacy_keywords)
    
    # Determine safety
    if has_privacy_concerns:
        result = {
            "safe": False,
            "reason": "Text contains potential privacy-sensitive information",
            "category": "privacy_concern",
            "details": {
                "has_scientific_content": has_scientific_content,
                "has_warning_signs": has_warning_signs,
                "has_privacy_concerns": has_privacy_concerns
            }
        }
    elif has_warning_signs and not has_scientific_content:
        result = {
            "safe": False,
            "reason": "Text contains suspicious keywords and lacks scientific context",
            "category": "inappropriate_content",
            "details": {
                "has_scientific_content": has_scientific_content,
                "has_warning_signs": has_warning_signs,
                "has_privacy_concerns": has_privacy_concerns
            }
        }
    elif not has_scientific_content and len(text) > 50:
        # Longer queries without scientific content are suspicious
        result = {
            "safe": False,
            "reason": "Text appears off-topic for biological image analysis",
            "category": "off_topic",
            "details": {
                "has_scientific_content": has_scientific_content,
                "has_warning_signs": has_warning_signs,
                "has_privacy_concerns": has_privacy_concerns
            }
        }
    else:
        result = {
            "safe": True,
            "reason": "Text appears to be legitimate scientific inquiry",
            "category": "approved",
            "details": {
                "has_scientific_content": has_scientific_content,
                "has_warning_signs": has_warning_signs,
                "has_privacy_concerns": has_privacy_concerns
            }
        }
    
    print(f"Text safety result: {result['category']}")
    return json.dumps(result, indent=2)


@tool
def check_image_safety(image_path: str, agent: Any = None) -> str:
    """
    Analyzes an image for safety and appropriateness using visual inspection.
    
    The VLM will visually inspect the image to determine if it appears to be:
    - A legitimate microscopy or biological image
    - Appropriate scientific content
    - Free from inappropriate or harmful content
    
    Args:
        image_path (str): Path to the image file to analyze
        agent (Any, optional): The agent instance (passed automatically)
        
    Returns:
        str: JSON string with safety assessment including visual analysis
    """
    print(f"\n--- TOOL CALLED: check_image_safety for '{image_path}' ---")
    
    try:
        # Load image for basic technical checks
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Basic technical analysis
        width, height = img.size
        total_pixels = width * height
        mean_intensity = float(np.mean(img_array))
        std_intensity = float(np.std(img_array))
        
        # Check if image is too small (might be icon/avatar)
        if total_pixels < 10000:  # Less than 100x100
            result = {
                "safe": False,
                "reason": "Image is too small to be a microscopy image",
                "category": "inappropriate_content",
                "technical_details": {
                    "dimensions": f"{width}x{height}",
                    "total_pixels": total_pixels,
                    "mean_intensity": mean_intensity
                },
                "visual_inspection_needed": False
            }
            return json.dumps(result, indent=2)
        
        # Check if image has very low variance (solid color, might be placeholder)
        if std_intensity < 5:
            result = {
                "safe": False,
                "reason": "Image appears to be solid color or placeholder",
                "category": "off_topic",
                "technical_details": {
                    "dimensions": f"{width}x{height}",
                    "total_pixels": total_pixels,
                    "mean_intensity": mean_intensity,
                    "std_intensity": std_intensity
                },
                "visual_inspection_needed": False
            }
            return json.dumps(result, indent=2)
        
        # Image passes basic technical checks - needs VLM visual inspection
        result = {
            "safe": True,  # Tentatively safe, pending visual inspection
            "reason": "Image passes basic technical checks. Visual inspection recommended.",
            "category": "needs_visual_inspection",
            "technical_details": {
                "dimensions": f"{width}x{height}",
                "total_pixels": total_pixels,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity
            },
            "visual_inspection_needed": True,
            "image_path": image_path,
            "guidance_for_vlm": (
                "Please visually inspect this image. Approve if it appears to be:\n"
                "- A microscopy image (cells, tissues, biological structures)\n"
                "- Scientific/medical imaging\n"
                "- Any legitimate biological research image\n\n"
                "Reject if it contains:\n"
                "- Inappropriate or harmful content\n"
                "- Personal photos (faces, people)\n"
                "- Non-biological content unrelated to scientific analysis\n"
                "- Memes, cartoons, or social media content"
            )
        }
        
        print(f"Image safety result: {result['category']}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        result = {
            "safe": False,
            "reason": f"Could not analyze image: {str(e)}",
            "category": "error",
            "error": str(e)
        }
        return json.dumps(result, indent=2)


@tool
def check_combined_safety(text: str, image_path: str = None) -> str:
    """
    Performs comprehensive safety check on both text and optional image.
    
    This tool combines text analysis and image analysis to provide a complete
    safety assessment of the user's request.
    
    Args:
        text (str): The user's text query
        image_path (str, optional): Path to image file if provided
        
    Returns:
        str: JSON string with comprehensive safety assessment
    """
    print(f"\n--- TOOL CALLED: check_combined_safety ---")
    print(f"Text: {text[:100]}...")
    if image_path:
        print(f"Image: {image_path}")
    
    results = {
        "text_check": None,
        "image_check": None,
        "overall_safe": True,
        "overall_reason": "",
        "overall_category": "approved"
    }
    
    # Check text
    text_result_str = check_text_safety(text)
    text_result = json.loads(text_result_str)
    results["text_check"] = text_result
    
    if not text_result["safe"]:
        results["overall_safe"] = False
        results["overall_reason"] = f"Text check failed: {text_result['reason']}"
        results["overall_category"] = text_result["category"]
        return json.dumps(results, indent=2)
    
    # Check image if provided
    if image_path:
        image_result_str = check_image_safety(image_path)
        image_result = json.loads(image_result_str)
        results["image_check"] = image_result
        
        if image_result["category"] == "needs_visual_inspection":
            # Return early - VLM needs to do visual inspection
            results["overall_safe"] = True  # Tentatively
            results["overall_reason"] = "Text approved. Image needs visual inspection by VLM."
            results["overall_category"] = "needs_visual_inspection"
            return json.dumps(results, indent=2)
        
        if not image_result["safe"]:
            results["overall_safe"] = False
            results["overall_reason"] = f"Image check failed: {image_result['reason']}"
            results["overall_category"] = image_result["category"]
            return json.dumps(results, indent=2)
    
    # Both passed
    results["overall_reason"] = "All safety checks passed"
    results["overall_category"] = "approved"
    
    print(f"Combined safety result: {results['overall_category']}")
    return json.dumps(results, indent=2)


@tool
def perform_visual_inspection(image_path: str, technical_analysis: dict) -> str:
    """
    Guidance tool for VLM to perform visual inspection of an image.
    
    This tool provides the VLM with the image and technical analysis results,
    allowing it to make a final safety determination based on visual content.
    
    Args:
        image_path (str): Path to the image file
        technical_analysis (dict): Results from technical image analysis
        
    Returns:
        str: Guidance for VLM inspection (image will be attached automatically)
    """
    print(f"\n--- TOOL CALLED: perform_visual_inspection for '{image_path}' ---")
    
    guidance = {
        "status": "ready_for_visual_inspection",
        "image_path": image_path,
        "technical_analysis": technical_analysis,
        "inspection_instructions": {
            "task": "Visually inspect this image and determine if it is safe and appropriate",
            "approve_if": [
                "Image shows cells, tissues, or biological structures under microscopy",
                "Image appears to be scientific/medical imaging",
                "Image is biological research content (even if not microscopy)",
                "Image quality is sufficient for analysis (even if blurry/low quality scientific image)"
            ],
            "reject_if": [
                "Image contains inappropriate, harmful, or offensive content",
                "Image shows human faces or identifiable people (privacy concern)",
                "Image is clearly non-biological (landscapes, objects, memes, etc.)",
                "Image appears to be social media content or personal photos",
                "Image contains text/documents that might have private information"
            ],
            "edge_cases": [
                "Diagrams/illustrations of cells: APPROVE (educational content)",
                "Histology slides: APPROVE (legitimate scientific content)",
                "Fluorescence microscopy: APPROVE (specialized imaging)",
                "Low quality cell images: APPROVE if clearly biological",
                "Medical scans (CT/MRI): APPROVE if anonymized"
            ]
        },
        "required_output": (
            "After visual inspection, respond with a JSON object:\n"
            "{\n"
            '  "safe": true/false,\n'
            '  "reason": "Description of what you see and why it\'s approved/rejected",\n'
            '  "category": "approved" | "inappropriate_content" | "off_topic" | "privacy_concern",\n'
            '  "visual_description": "Brief description of image content"\n'
            "}"
        )
    }
    
    return json.dumps(guidance, indent=2)
