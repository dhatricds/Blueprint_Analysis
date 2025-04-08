#!/usr/bin/env python3
"""
Invert and process reference symbols for better detection.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def load_image(image_path):
    """Load an image and check if it was loaded correctly."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img

def process_symbol(image, target_size=None, invert=True, threshold=127):
    """Process symbol image for better detection."""
    # Convert to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize if target size specified
    if target_size:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        gray = cv2.resize(gray, target_size)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Invert if needed
    if invert:
        binary = cv2.bitwise_not(binary)
    
    return binary

def create_variations(image, scales=(0.5, 0.75, 1.0, 1.25, 1.5)):
    """Create scaled variations of the symbol."""
    variations = []
    
    for scale in scales:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        scaled = cv2.resize(image, (width, height))
        variations.append((scaled, scale))
    
    return variations

def save_variations(variations, output_dir, base_name):
    """Save symbol variations to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for image, scale in variations:
        path = output_dir / f"{base_name}_scale_{scale:.2f}.png"
        cv2.imwrite(str(path), image)
        saved_paths.append(path)
    
    return saved_paths

def main():
    parser = argparse.ArgumentParser(description="Process reference symbols for detection")
    parser.add_argument("--symbol", required=True, help="Path to symbol image")
    parser.add_argument("--output-dir", default="symbols", help="Output directory")
    parser.add_argument("--size", type=int, help="Target size (square)")
    parser.add_argument("--no-invert", action="store_true", help="Don't invert the image")
    parser.add_argument("--threshold", type=int, default=127, help="Threshold value")
    args = parser.parse_args()
    
    try:
        # Load symbol
        symbol = load_image(args.symbol)
        
        # Process symbol
        processed = process_symbol(
            symbol,
            target_size=args.size,
            invert=not args.no_invert,
            threshold=args.threshold
        )
        
        # Create variations
        variations = create_variations(processed)
        
        # Save results
        base_name = Path(args.symbol).stem
        saved_paths = save_variations(variations, args.output_dir, base_name)
        
        print(f"Processed symbol saved to:")
        for path in saved_paths:
            print(f"  {path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())