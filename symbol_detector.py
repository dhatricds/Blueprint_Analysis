"""
Core functionality for detecting symbols in blueprint tiles.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import os

def find_symbols_in_tile(tile, reference, threshold=0.6, scales=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)):
    """
    Find instances of a reference symbol in a tile using template matching.
    
    Args:
        tile: Input tile image
        reference: Reference symbol image
        threshold: Matching threshold (default: 0.6)
        scales: Tuple of scales to try (default: 0.5-1.2)
        
    Returns:
        List of matches with coordinates and confidence scores
    """
    if tile is None or reference is None:
        return []
    
    # Convert to grayscale if needed
    if len(tile.shape) > 2:
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    else:
        tile_gray = tile
        
    if len(reference.shape) > 2:
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference
    
    matches = []
    
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
            matches.append({
                "x": int(pt[0]),
                "y": int(pt[1]),
                "width": scaled_reference.shape[1],
                "height": scaled_reference.shape[0],
                "confidence": float(result[pt[1], pt[0]]),
                "scale": scale
            })
    
    # Non-maximum suppression
    return non_max_suppression(matches)

def non_max_suppression(matches, overlap_threshold=0.3):
    """
    Apply non-maximum suppression to remove overlapping matches.
    
    Args:
        matches: List of match dictionaries
        overlap_threshold: Maximum allowed overlap ratio (default: 0.3)
        
    Returns:
        Filtered list of matches
    """
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
    """
    Calculate overlap ratio between two rectangles.
    
    Args:
        rect1, rect2: Dictionaries with x, y, width, height
        
    Returns:
        Overlap ratio (0-1)
    """
    # Calculate intersection
    x1 = max(rect1["x"], rect2["x"])
    y1 = max(rect1["y"], rect2["y"])
    x2 = min(rect1["x"] + rect1["width"], rect2["x"] + rect2["width"])
    y2 = min(rect1["y"] + rect1["height"], rect2["y"] + rect2["height"])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = rect1["width"] * rect1["height"]
    area2 = rect2["width"] * rect2["height"]
    
    # Return overlap ratio
    return intersection / min(area1, area2)

def create_visualization(tile, reference, matches, output_path):
    """
    Create visualization of detected symbols.
    
    Args:
        tile: Input tile image
        reference: Reference symbol image
        matches: List of match dictionaries
        output_path: Path to save visualization
    """
    # Create copy for drawing
    vis = tile.copy() if len(tile.shape) == 3 else cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    for match in matches:
        x, y = match["x"], match["y"]
        w, h = match["width"], match["height"]
        conf = match["confidence"]
        
        # Draw rectangle
        color = (0, int(255 * conf), 0)  # Green, intensity based on confidence
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        # Draw confidence score
        text = f"{conf:.2f}"
        cv2.putText(vis, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save visualization
    cv2.imwrite(output_path, vis)

def save_results(matches, tile_info, output_path):
    """
    Save detection results to JSON file.
    
    Args:
        matches: List of match dictionaries
        tile_info: Dictionary with tile metadata
        output_path: Path to save JSON file
    """
    results = {
        "tile_info": tile_info,
        "matches": matches,
        "total_matches": len(matches)
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

def process_tile(tile_path, reference_path, output_dir, threshold=0.6, debug=False):
    """
    Process a single tile for symbol detection.
    
    Args:
        tile_path: Path to tile image
        reference_path: Path to reference symbol
        output_dir: Output directory for results
        threshold: Matching threshold (default: 0.6)
        debug: Whether to save debug visualizations
        
    Returns:
        Dictionary with detection results
    """
    # Load images
    tile = cv2.imread(str(tile_path))
    reference = cv2.imread(str(reference_path))
    
    if tile is None or reference is None:
        return None
    
    # Get tile info
    tile_name = Path(tile_path).stem
    tile_info = {
        "name": tile_name,
        "path": str(tile_path),
        "size": tile.shape[:2]
    }
    
    # Find symbols
    matches = find_symbols_in_tile(tile, reference, threshold)
    
    # Save results
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / f"{tile_name}_results.json"
    save_results(matches, tile_info, results_path)
    
    # Create debug visualization if requested
    if debug and matches:
        debug_dir = Path(output_dir) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        vis_path = debug_dir / f"{tile_name}_debug.jpg"
        create_visualization(tile, reference, matches, vis_path)
    
    return {
        "tile_info": tile_info,
        "matches": matches,
        "results_path": str(results_path)
    }