#!/usr/bin/env python3
"""
Analyze a single tile with a reference symbol using Claude Vision.
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from datetime import datetime

from blueprint_utils import (
    process_reference_image,
    create_combined_image,
    claude_vision_api_call,
    create_claude_prompt,
    extract_count_from_response
)

def analyze_tile(tile_path, reference_path, output_dir="output", debug=False):
    """
    Analyze a single tile with a reference symbol.
    
    Args:
        tile_path: Path to the tile image
        reference_path: Path to the reference symbol
        output_dir: Directory to save outputs
        debug: Whether to save debug images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    tile = cv2.imread(str(tile_path))
    if tile is None:
        raise ValueError(f"Cannot load tile image from {tile_path}")
    
    # Process reference symbol
    reference = process_reference_image(reference_path)
    
    # Create combined image for API
    combined = create_combined_image(tile, reference)
    
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "combined.jpg"), combined)
    
    # Get prompt
    prompt = create_claude_prompt()
    
    # Call API
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    response = claude_vision_api_call(combined, prompt, api_key)
    
    # Extract results
    results = extract_count_from_response(response)
    
    # Add metadata
    results.update({
        "tile_path": str(tile_path),
        "reference_path": str(reference_path),
        "timestamp": datetime.now().isoformat()
    })
    
    # Save results
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    tile_name = Path(tile_path).stem
    results_path = os.path.join(results_dir, f"{tile_name}_results.json")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results, results_path

def main():
    parser = argparse.ArgumentParser(description="Analyze a single tile with a reference symbol")
    parser.add_argument("--tile", required=True, help="Path to tile image")
    parser.add_argument("--reference", required=True, help="Path to reference symbol")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    
    args = parser.parse_args()
    
    try:
        results, results_path = analyze_tile(
            args.tile,
            args.reference,
            args.output_dir,
            args.debug
        )
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Symbols found: {results['count']}")
        if results['locations']:
            print(f"Locations: {results['locations']}")
        if results['labels']:
            print(f"Labels: {results['labels']}")
        print(f"\nDetailed results saved to: {results_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())