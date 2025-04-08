#!/usr/bin/env python3
"""
Process PDF blueprints for symbol detection.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import subprocess
from tqdm import tqdm

# Import shared functionality from blueprint_utils
from blueprint_utils import (
    convert_pdf_to_images,
    create_tiles,
    is_blank_tile,
    invert_image,
    process_reference_image,
    setup_output_directories
)

def process_blueprint_images(image_paths, reference_path, output_dir="output", 
                            tile_size=640, overlap=6, debug=False, tiling_only=False):
    """
    Process blueprint images to prepare for symbol detection.
    
    Args:
        image_paths: List of paths to blueprint images
        reference_path: Path to the reference symbol image
        output_dir: Directory to save output files
        tile_size: Size of tiles for processing (default: 640)
        overlap: Overlap between tiles (default: 6)
        debug: Whether to enable debug mode
        tiling_only: If True, only create tiles without processing for symbol detection
        
    Returns:
        Dictionary with processing results
    """
    # Create output directories
    directories = setup_output_directories(output_dir, debug)
    
    # Create additional directories for tiles
    tiles_dir = os.path.join(output_dir, "tiles")
    inverted_tiles_dir = os.path.join(output_dir, "inverted_tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(inverted_tiles_dir, exist_ok=True)
    
    # Process reference symbol
    if not tiling_only and reference_path:
        reference = process_reference_image(reference_path)
    
    # Store processing results
    results = {
        "pages": [],
        "total_tiles": 0,
        "blank_tiles": 0,
        "processed_tiles": 0
    }
    
    # Process each page
    for page_idx, image_path in enumerate(image_paths):
        print(f"Processing page {page_idx+1}/{len(image_paths)}")
        
        # Load blueprint image
        blueprint = cv2.imread(image_path)
        if blueprint is None:
            print(f"Error: Cannot load blueprint image from {image_path}")
            continue
        
        # Invert blueprint for better symbol detection
        inverted_blueprint = invert_image(blueprint)
        
        # Save inverted blueprint if debugging
        if debug:
            inverted_path = os.path.join(directories["debug"], f"inverted_page{page_idx+1}.jpg")
            cv2.imwrite(inverted_path, inverted_blueprint)
        
        # Create tiles from both original and inverted blueprints
        tiles = create_tiles(inverted_blueprint, tile_size, overlap)
        original_tiles = create_tiles(blueprint, tile_size, overlap)
        
        # Save page information
        page_info = {
            "page_index": page_idx + 1,
            "image_path": image_path,
            "total_tiles": len(tiles),
            "blank_tiles": 0,
            "tiles": []
        }
        
        results["total_tiles"] += len(tiles)
        
        # Process each tile
        for tile_idx, ((inverted_tile, coords), (original_tile, _)) in enumerate(zip(tiles, original_tiles)):
            # Create base name for files
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            tile_filename = f"{base_name}_tile{tile_idx+1}_x{coords[0]}_y{coords[1]}.jpg"
            
            tile_info = {
                "tile_index": tile_idx + 1,
                "coords": coords,
                "is_blank": False
            }
            
            # Save all tiles if debug is enabled
            if debug:
                all_tile_path = os.path.join(directories["all_tiles"], tile_filename)
                cv2.imwrite(all_tile_path, inverted_tile)
                tile_info["all_tile_path"] = all_tile_path
            
            # Save both original and inverted tiles
            original_tile_path = os.path.join(tiles_dir, tile_filename)
            inverted_tile_path = os.path.join(inverted_tiles_dir, tile_filename)
            
            cv2.imwrite(original_tile_path, original_tile)
            cv2.imwrite(inverted_tile_path, inverted_tile)
            
            tile_info["tile_path"] = original_tile_path
            tile_info["inverted_tile_path"] = inverted_tile_path
            
            # If not in tiling-only mode, check for blank tiles
            if not tiling_only:
                if is_blank_tile(inverted_tile):
                    tile_info["is_blank"] = True
                    page_info["blank_tiles"] += 1
                    results["blank_tiles"] += 1
                    
                    if debug:
                        blank_path = os.path.join(directories["debug"], f"blank_{tile_filename}")
                        cv2.imwrite(blank_path, inverted_tile)
                else:
                    results["processed_tiles"] += 1
            else:
                results["processed_tiles"] += 1
            
            page_info["tiles"].append(tile_info)
        
        results["pages"].append(page_info)
    
    return results

def main():
    """
    Main function to process PDF blueprints.
    Supports both symbol detection and tiling-only modes.
    """
    parser = argparse.ArgumentParser(description="Process PDF blueprints for symbol detection or tiling")
    parser.add_argument("--blueprint", required=True, help="Path to the blueprint PDF")
    parser.add_argument("--reference", help="Path to the reference symbol image")
    parser.add_argument("--output-dir", default="output", help="Directory to save outputs")
    parser.add_argument("--tile-size", type=int, default=640, help="Size of tiles (default: 640)")
    parser.add_argument("--overlap", type=int, default=6, help="Overlap between tiles (default: 6)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF conversion")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--tiling-only", action="store_true", help="Only create tiles without symbol detection")
    
    args = parser.parse_args()
    
    # Check if blueprint exists
    if not os.path.exists(args.blueprint):
        print(f"Error: Blueprint file not found: {args.blueprint}")
        sys.exit(1)
    
    # Check if reference exists when not in tiling-only mode
    if not args.tiling_only and args.reference and not os.path.exists(args.reference):
        print(f"Error: Reference symbol file not found: {args.reference}")
        sys.exit(1)
    
    # Convert PDF to images
    print("Converting PDF to images...")
    image_paths = convert_pdf_to_images(args.blueprint, dpi=args.dpi)
    if not image_paths:
        print("Error: Failed to convert PDF to images")
        sys.exit(1)
    
    print(f"Processing {len(image_paths)} pages...")
    results = process_blueprint_images(
        image_paths=image_paths,
        reference_path=args.reference,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        debug=args.debug,
        tiling_only=args.tiling_only
    )
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Total tiles created: {results['total_tiles']}")
    if not args.tiling_only:
        print(f"Blank tiles: {results['blank_tiles']}")
        print(f"Processed tiles: {results['processed_tiles']}")
    
    print(f"\nOutput files saved to: {args.output_dir}")
    print(f"- Original tiles: {os.path.join(args.output_dir, 'tiles')}")
    print(f"- Inverted tiles: {os.path.join(args.output_dir, 'inverted_tiles')}")
    if args.debug:
        print(f"- Debug files: {os.path.join(args.output_dir, 'debug')}")
        print(f"- All tiles: {os.path.join(args.output_dir, 'all_tiles')}")

if __name__ == "__main__":
    main()