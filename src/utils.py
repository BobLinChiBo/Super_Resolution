import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Tuple, Optional
import json
from datetime import datetime
import math


def load_image(image_path: str) -> Image.Image:
    """Load an image from file path"""
    return Image.open(image_path).convert('RGB')


def save_image(image: Image.Image, output_path: str, quality: int = 95):
    """Save an image to file path"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, quality=quality)


def get_image_list(input_dir: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> List[str]:
    """Get list of image files from directory"""
    image_files = []
    for file in sorted(os.listdir(input_dir)):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(input_dir, file))
    return image_files


def preprocess_old_document(image: Image.Image) -> Image.Image:
    """Preprocess old document images for better enhancement"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply slight denoising
    img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
    
    # Convert back to PIL
    image = Image.fromarray(img_array)
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)
    
    return image


def postprocess_enhanced_image(image: Image.Image, enhance_text: bool = True) -> Image.Image:
    """Post-process enhanced image for better text clarity"""
    if not enhance_text:
        return image
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Convert back to RGB
    img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Blend with original (keep some color information)
    img_array = cv2.addWeighted(np.array(image), 0.3, img_array, 0.7, 0)
    
    return Image.fromarray(img_array)


def create_progress_tracker(output_dir: str) -> dict:
    """Create or load progress tracker"""
    tracker_path = os.path.join(output_dir, 'progress.json')
    
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as f:
            tracker = json.load(f)
    else:
        tracker = {
            'started_at': datetime.now().isoformat(),
            'completed_files': [],
            'failed_files': [],
            'last_processed': None
        }
    
    return tracker


def save_progress_tracker(tracker: dict, output_dir: str):
    """Save progress tracker to file"""
    tracker_path = os.path.join(output_dir, 'progress.json')
    with open(tracker_path, 'w') as f:
        json.dump(tracker, f, indent=2)


def estimate_memory_usage(image_size: Tuple[int, int], scale: int) -> float:
    """Estimate GPU memory usage in GB"""
    width, height = image_size
    # Rough estimation: input + output + intermediate tensors
    pixels = width * height
    scaled_pixels = (width * scale) * (height * scale)
    
    # Assuming float32 (4 bytes) and RGB (3 channels)
    # Plus some overhead for model and intermediate computations
    memory_gb = (pixels * 3 * 4 + scaled_pixels * 3 * 4 + scaled_pixels * 3 * 4 * 2) / (1024**3)
    
    return memory_gb * 1.5  # Add 50% safety margin


def split_image_if_needed(image: Image.Image, max_size: int = 2048) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """Split large images into tiles if needed"""
    width, height = image.size
    
    if width <= max_size and height <= max_size:
        return [(image, (0, 0, width, height))]
    
    tiles = []
    tile_size = max_size
    overlap = 128  # Overlap between tiles
    
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            tile = image.crop((x, y, x_end, y_end))
            tiles.append((tile, (x, y, x_end, y_end)))
    
    return tiles


def merge_tiles(tiles: List[Tuple[Image.Image, Tuple[int, int, int, int]]], 
                original_size: Tuple[int, int], scale: int) -> Image.Image:
    """Merge enhanced tiles back into a single image"""
    width, height = original_size
    merged = Image.new('RGB', (width * scale, height * scale))
    
    for tile, (x, y, x_end, y_end) in tiles:
        # Scale the coordinates
        x_scaled = x * scale
        y_scaled = y * scale
        
        # Paste the tile
        merged.paste(tile, (x_scaled, y_scaled))
    
    return merged


def clean_background(image: Image.Image, threshold: int = 240) -> Image.Image:
    """Clean the background of old document images"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Create mask for background (light pixels)
    mask = gray > threshold
    
    # Set background pixels to white
    img_array[mask] = [255, 255, 255]
    
    return Image.fromarray(img_array)


def get_image_dpi(image_path: str) -> Tuple[int, int]:
    """Get DPI information from image"""
    image = Image.open(image_path)
    dpi = image.info.get('dpi', (300, 300))  # Default to 300 DPI if not specified
    return dpi


def calculate_target_size_from_dpi(current_size: Tuple[int, int], 
                                 current_dpi: Tuple[int, int], 
                                 target_dpi: int) -> Tuple[int, int]:
    """Calculate target size based on DPI"""
    width, height = current_size
    dpi_x, dpi_y = current_dpi
    
    # Calculate physical size in inches
    width_inches = width / dpi_x
    height_inches = height / dpi_y
    
    # Calculate new pixel dimensions for target DPI
    new_width = int(width_inches * target_dpi)
    new_height = int(height_inches * target_dpi)
    
    return (new_width, new_height)


def two_stage_upscale(image: Image.Image, 
                     model_enhance_func, 
                     initial_scale: float = 2.0,
                     target_size: Optional[Tuple[int, int]] = None,
                     target_dpi: Optional[int] = None,
                     current_dpi: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Two-stage upscaling process:
    1. Upscale with AI model (typically 2x)
    2. Downsample to target size/DPI with high-quality resampling
    """
    # Stage 1: AI upscaling
    upscaled = model_enhance_func(image, outscale=initial_scale)
    
    # Stage 2: Resize to target
    if target_size:
        final_size = target_size
    elif target_dpi and current_dpi:
        # Calculate target size from DPI
        original_size = image.size
        final_size = calculate_target_size_from_dpi(original_size, current_dpi, target_dpi)
    else:
        # No downsampling needed
        return upscaled
    
    # Only downsample if the target is smaller than current
    if final_size[0] < upscaled.width or final_size[1] < upscaled.height:
        # Use Lanczos for high-quality downsampling
        return upscaled.resize(final_size, Image.Resampling.LANCZOS)
    else:
        return upscaled


def auto_detect_content_type(image: Image.Image) -> str:
    """Detect if image is text, line art, or mixed content"""
    # Convert to grayscale
    gray = image.convert('L')
    gray_array = np.array(gray)
    
    # Calculate statistics
    unique_values = len(np.unique(gray_array))
    std_dev = np.std(gray_array)
    
    # Simple heuristics
    if unique_values < 10:  # Very few colors - likely pure line art
        return 'line_art'
    elif std_dev < 50:  # Low variation - likely text document
        return 'text'
    else:
        return 'mixed'


def select_model_for_content(content_type: str) -> str:
    """Select best model based on content type"""
    model_map = {
        'text': 'realesr-general-x4v3',
        'line_art': 'realesrgan-animevideo-xsx2',  # Anime model preserves strokes
        'mixed': 'realesr-general-x4v3'
    }
    return model_map.get(content_type, 'realesr-general-x4v3')