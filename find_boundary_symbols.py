#!/usr/bin/env python3
"""
Detect symbols near boundaries in blueprint tiles.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json

def load_image(image_path):
    """Load an image and check if it was loaded correctly."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img

def process_reference_symbol(reference, target_size=128, invert=True):
    """Process reference symbol for template matching."""
    # Convert to grayscale
    if len(reference.shape) > 2:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    # Resize if needed
    if reference.shape[0] != target_size or reference.shape[1] != target_size:
        reference = cv2.resize(reference, (target_size, target_size))
    
    # Threshold
    _, reference = cv2.threshold(reference, 127, 255, cv2.THRESH_BINARY)
    
    # Invert if needed
    if invert:
        reference = cv2.bitwise_not(reference)
    
    return reference

def find_symbols_in_tile(tile, reference, threshold=0.6, scales=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2), boundary_margin=0.1):
    """Find symbols in tile using template matching at multiple scales."""
    if tile is None or reference is None:
        return []
    
    # Convert to grayscale
    if len(tile.shape) > 2:
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    else:
        tile_gray = tile
        
    if len(reference.shape) > 2:
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference
    
    matches = []
    tile_height, tile_width = tile_gray.shape
    boundary_pixels = int(min(tile_width, tile_height) * boundary_margin)
    
    # Try different scales
    for scale in scales:
        # Resize reference symbol
        if scale != 1.0:
            width = int(reference_gray.shape[1] * scale)
            height = int(reference_gray.shape[0] * scale)
            scaled_reference = cv2.resize(reference_gray, (width, height))
        else:
            scaled_reference = reference_gray
        
        # Template matching
        result = cv2.matchTemplate(tile_gray, scaled_reference, cv2.TM_CCOEFF_NORMED)
        
        # Find matches above threshold
        locations = np.where(result >= threshold)
        for pt in zip(*locations[::-1]):
            x, y = pt[0], pt[1]
            w, h = scaled_reference.shape[1], scaled_reference.shape[0]
            
            # Check if match is near boundary
            is_boundary = (
                x <= boundary_pixels or
                y <= boundary_pixels or
                x + w >= tile_width - boundary_pixels or
                y + h >= tile_height - boundary_pixels
            )
            
            matches.append({
                "x": int(x),
                "y": int(y),
                "width": w,
                "height": h,
                "confidence": float(result[y, x]),
                "scale": scale,
                "is_boundary": is_boundary
            })
    
    # Non-maximum suppression
    return non_max_suppression(matches)

def non_max_suppression(matches, overlap_threshold=0.3):
    """Remove overlapping matches using non-maximum suppression."""
    if not matches:
        return []
    
    # Sort by confidence
    matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    
    for match in matches:
        # Check overlap with kept matches
        should_keep = True
        for kept in keep:
            if calculate_overlap(match, kept) > overlap_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(match)
    
    return keep

def calculate_overlap(rect1, rect2):
    """Calculate overlap ratio between two rectangles."""
    x1 = max(rect1["x"], rect2["x"])
    y1 = max(rect1["y"], rect2["y"])
    x2 = min(rect1["x"] + rect1["width"], rect2["x"] + rect2["width"])
    y2 = min(rect1["y"] + rect1["height"], rect2["y"] + rect2["height"])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = rect1["width"] * rect1["height"]
    area2 = rect2["width"] * rect2["height"]
    
    return intersection / min(area1, area2)

def create_visualization(tile, reference, matches, all_results, output_path):
    """Create visualization of template matching process."""
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original tile
    ax1.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Tile")
    ax1.axis("off")
    
    # Reference symbol
    ax2.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    ax2.set_title("Reference Symbol")
    ax2.axis("off")
    
    # Matches
    vis = tile.copy()
    for match in matches:
        color = (0, 255, 0) if match["is_boundary"] else (0, 0, 255)
        cv2.rectangle(
            vis,
            (match["x"], match["y"]),
            (match["x"] + match["width"], match["y"] + match["height"]),
            color,
            2
        )
        
        # Add confidence score
        text = f"{match['confidence']:.2f}"
        cv2.putText(
            vis,
            text,
            (match["x"], match["y"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    ax3.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax3.set_title("Detected Symbols")
    ax3.axis("off")
    
    # Template matching results
    ax4.imshow(all_results, cmap="jet")
    ax4.set_title("Template Matching Results")
    ax4.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Find symbols near boundaries in blueprint tiles")
    parser.add_argument("--tile", required=True, help="Path to tile image")
    parser.add_argument("--symbol", required=True, help="Path to reference symbol image")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="Matching threshold")
    parser.add_argument("--boundary-margin", type=float, default=0.1, help="Boundary margin as fraction of tile size")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load images
        tile = load_image(args.tile)
        reference = load_image(args.symbol)
        
        # Process reference symbol
        processed_reference = process_reference_symbol(reference)
        
        # Find symbols
        matches = find_symbols_in_tile(
            tile,
            processed_reference,
            threshold=args.threshold,
            boundary_margin=args.boundary_margin
        )
        
        # Save results
        results = {
            "tile_path": str(args.tile),
            "symbol_path": str(args.symbol),
            "matches": matches,
            "total_matches": len(matches),
            "boundary_matches": sum(1 for m in matches if m["is_boundary"])
        }
        
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create visualization if matches found
        if matches:
            vis_path = output_dir / "visualization.jpg"
            create_visualization(
                tile,
                reference,
                matches,
                all_results,
                vis_path
            )
            
            # Try to open visualization
            if sys.platform == "darwin":
                subprocess.run(["open", str(vis_path)])
            elif sys.platform == "win32":
                os.startfile(str(vis_path))
            else:
                subprocess.run(["xdg-open", str(vis_path)])
        
        print(f"Found {len(matches)} symbols ({results['boundary_matches']} near boundaries)")
        print(f"Results saved to {results_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()