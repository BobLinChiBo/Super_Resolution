#!/usr/bin/env python3
"""
Super Resolution Enhancement for Old Documents
Main script for batch processing images
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model
from src.utils import (
    load_image, save_image, get_image_list,
    preprocess_old_document, postprocess_enhanced_image,
    create_progress_tracker, save_progress_tracker,
    estimate_memory_usage, split_image_if_needed, merge_tiles,
    clean_background, get_image_dpi, two_stage_upscale,
    auto_detect_content_type, select_model_for_content
)


def process_single_image(image_path: str, model, args) -> str:
    """Process a single image with super resolution"""
    try:
        # Load image
        image = load_image(image_path)
        original_size = image.size
        
        # Preprocess if requested
        if args.preprocess:
            image = preprocess_old_document(image)
        
        # Clean background if requested
        if args.clean_background:
            image = clean_background(image, threshold=args.background_threshold)
        
        # Auto-detect content type if requested
        if args.auto_detect:
            content_type = auto_detect_content_type(image)
            model_name = select_model_for_content(content_type)
            print(f"  Detected content type: {content_type}, using model: {model_name}")
        else:
            model_name = args.model_variant
        
        # Get DPI information
        current_dpi = get_image_dpi(image_path) if args.target_dpi else None
        
        # Check if we need to split the image
        memory_usage = estimate_memory_usage(image.size, args.scale)
        if memory_usage > args.max_memory_gb:
            print(f"  Large image detected ({image.size}), splitting into tiles...")
            tiles = split_image_if_needed(image, max_size=args.tile_size)
            
            enhanced_tiles = []
            for i, (tile, coords) in enumerate(tiles):
                print(f"  Processing tile {i+1}/{len(tiles)}...", end='\r')
                
                if args.two_stage:
                    # Two-stage processing
                    enhanced_tile = two_stage_upscale(
                        tile, 
                        lambda img, outscale: model.enhance(img, outscale=outscale, model_name=model_name),
                        initial_scale=args.scale,
                        target_dpi=args.target_dpi,
                        current_dpi=current_dpi
                    )
                else:
                    enhanced_tile = model.enhance(tile, outscale=args.scale, model_name=model_name)
                
                enhanced_tiles.append((enhanced_tile, coords))
            
            # Merge tiles
            enhanced = merge_tiles(enhanced_tiles, original_size, args.scale)
        else:
            # Process whole image
            if args.two_stage:
                # Two-stage processing
                enhanced = two_stage_upscale(
                    image,
                    lambda img, outscale: model.enhance(img, outscale=outscale, model_name=model_name),
                    initial_scale=args.scale,
                    target_dpi=args.target_dpi,
                    current_dpi=current_dpi
                )
            else:
                enhanced = model.enhance(image, outscale=args.scale, model_name=model_name)
        
        # Postprocess if requested
        if args.postprocess:
            enhanced = postprocess_enhanced_image(enhanced, enhance_text=args.enhance_text)
        
        # Generate output path
        input_filename = os.path.basename(image_path)
        output_filename = f"{os.path.splitext(input_filename)[0]}_x{args.scale}_enhanced.jpg"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Save enhanced image
        save_image(enhanced, output_path, quality=args.quality)
        
        return output_path
        
    except Exception as e:
        print(f"  Error processing {image_path}: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Enhance old Chinese document images with super resolution')
    
    # Input/Output arguments
    parser.add_argument('--input-dir', type=str, default='data/input',
                        help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, default='data/output',
                        help='Output directory for enhanced images')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='realesrgan',
                        choices=['realesrgan', 'swinir', 'bsrgan'],
                        help='Super resolution model to use')
    parser.add_argument('--model-variant', type=str, default='realesr-general-x4v3',
                        choices=['realesr-general-x4v3', 'realesr-general-wdn-x4v3', 
                                'RealESRGAN_x4plus', 'RealESRGAN_x2plus',
                                'realesrgan-animevideo-xsx2', 'RealESRGAN_x4plus_anime_6B'],
                        help='Specific model variant to use')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4],
                        help='Upscaling factor (default: 2 for text documents)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Processing arguments
    parser.add_argument('--preprocess', action='store_true',
                        help='Apply preprocessing for old documents')
    parser.add_argument('--postprocess', action='store_true',
                        help='Apply postprocessing for better text clarity')
    parser.add_argument('--enhance-text', action='store_true',
                        help='Apply text-specific enhancements')
    parser.add_argument('--clean-background', action='store_true',
                        help='Clean background of old documents')
    parser.add_argument('--background-threshold', type=int, default=240,
                        help='Threshold for background cleaning (0-255)')
    parser.add_argument('--two-stage', action='store_true',
                        help='Use two-stage processing (upscale then downsample)')
    parser.add_argument('--target-dpi', type=int, default=None,
                        help='Target DPI for output (e.g., 300, 600)')
    parser.add_argument('--auto-detect', action='store_true',
                        help='Auto-detect content type and select best model')
    
    # Performance arguments
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of images to process in parallel')
    parser.add_argument('--tile-size', type=int, default=1024,
                        help='Tile size for large images')
    parser.add_argument('--max-memory-gb', type=float, default=6.0,
                        help='Maximum GPU memory to use (GB)')
    
    # Output arguments
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for output images (1-100)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of images
    print(f"Scanning input directory: {args.input_dir}")
    image_files = get_image_list(args.input_dir)
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No images found in input directory!")
        return
    
    # Load progress tracker
    tracker = create_progress_tracker(args.output_dir)
    
    # Filter out already processed images if resuming
    if args.resume and tracker['completed_files']:
        completed_set = set(tracker['completed_files'])
        image_files = [f for f in image_files if os.path.basename(f) not in completed_set]
        print(f"Resuming: {len(image_files)} images remaining")
    
    # Initialize model
    print(f"\nInitializing {args.model.upper()} model on {args.device}...")
    model = get_model(args.model, device=args.device)
    
    # Process images
    print(f"\nStarting enhancement with scale={args.scale}x")
    print("=" * 60)
    
    start_time = datetime.now()
    
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for i, image_path in enumerate(image_files):
            try:
                # Update progress bar description
                filename = os.path.basename(image_path)
                pbar.set_description(f"Processing {filename}")
                
                # Process image
                output_path = process_single_image(image_path, model, args)
                
                # Update tracker
                tracker['completed_files'].append(filename)
                tracker['last_processed'] = filename
                
                # Save progress every 10 images
                if (i + 1) % 10 == 0:
                    save_progress_tracker(tracker, args.output_dir)
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing {filename}: {str(e)}")
                tracker['failed_files'].append({
                    'filename': filename,
                    'error': str(e)
                })
                pbar.update(1)
                continue
    
    # Save final progress
    save_progress_tracker(tracker, args.output_dir)
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print(f"Total images processed: {len(tracker['completed_files'])}")
    print(f"Failed images: {len(tracker['failed_files'])}")
    print(f"Total time: {duration}")
    print(f"Average time per image: {duration / len(tracker['completed_files']) if tracker['completed_files'] else 0}")
    print(f"\nEnhanced images saved to: {args.output_dir}")
    
    if tracker['failed_files']:
        print(f"\nFailed files saved in progress.json")


if __name__ == "__main__":
    main()