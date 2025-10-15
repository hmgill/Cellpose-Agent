"""
Image utilities for encoding and resizing
"""
import base64
from io import BytesIO
from PIL import Image


def resize_and_encode_image(image_path: str, size: tuple = (512, 512)) -> tuple[str, str]:
    """
    Resize an image to specified size and encode as base64.
    
    Args:
        image_path (str): Path to the image file
        size (tuple): Target size as (width, height), default (1024, 1024)
    
    Returns:
        tuple: (base64_string, media_type)
    """
    # Open and convert to RGB
    img = Image.open(image_path).convert("RGB")
    
    # Resize with high-quality resampling
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    
    # Encode to base64
    buffered = BytesIO()
    img_resized.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64, "image/png"
