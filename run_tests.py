#!/usr/bin/env python3
"""
Run tests on blueprint symbol detection.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from symbol_detector import find_symbols_in_tile, process_reference_symbol

def create_test_image(size=500, symbol_size=50, num_symbols=5, noise_level=0.1):
    """Create a test image with known symbol locations."""
    # Create blank image
    image = np.ones((size, size), dtype=np.uint8) * 255
    
    # Create simple symbol (circle)
    symbol = np.ones((symbol_size, symbol_size), dtype=np.uint8) * 255
    cv2.circle(
        symbol,
        (symbol_size//2, symbol_size//2),
        symbol_size//3,
        0,
        -1
    )
    
    # Add symbols at random locations
    true_locations = []
    for _ in range(num_symbols):
        x = np.random.randint(0, size - symbol_size)
        y = np.random.randint(0, size - symbol_size)
        
        # Add symbol
        image[y:y+symbol_size, x:x+symbol_size] = symbol
        
        true_locations.append({
            "x": x,
            "y": y,
            "width": symbol_size,
            "height": symbol_size
        })
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image, symbol, true_locations

def evaluate_detection(detected, true_locations, iou_threshold=0.5):
    """Evaluate detection results against ground truth."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Mark true locations as unmatched initially
    unmatched = true_locations.copy()
    
    for det in detected:
        # Find best matching true location
        best_iou = 0
        best_idx = -1
        
        for i, true in enumerate(unmatched):
            iou = calculate_iou(det, true)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        # Check if match is good enough
        if best_iou >= iou_threshold:
            true_positives += 1
            unmatched.pop(best_idx)
        else:
            false_positives += 1
    
    # Remaining unmatched true locations are false negatives
    false_negatives = len(unmatched)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    # Calculate intersection coordinates
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def run_tests(num_tests=10, **kwargs):
    """Run multiple detection tests."""
    results = []
    
    for i in tqdm(range(num_tests), desc="Running tests"):
        # Create test image
        image, symbol, true_locations = create_test_image(**kwargs)
        
        # Process reference symbol
        processed_symbol = process_reference_symbol(symbol)
        
        # Run detection
        detected = find_symbols_in_tile(image, processed_symbol)
        
        # Evaluate results
        metrics = evaluate_detection(detected, true_locations)
        results.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in results[0].keys():
        avg_metrics[key] = sum(r[key] for r in results) / len(results)
    
    return results, avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Run symbol detection tests")
    parser.add_argument("--num-tests", type=int, default=10, help="Number of test images")
    parser.add_argument("--image-size", type=int, default=500, help="Test image size")
    parser.add_argument("--symbol-size", type=int, default=50, help="Symbol size")
    parser.add_argument("--num-symbols", type=int, default=5, help="Symbols per image")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level")
    parser.add_argument("--output-dir", default="test_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run tests
        results, avg_metrics = run_tests(
            num_tests=args.num_tests,
            size=args.image_size,
            symbol_size=args.symbol_size,
            num_symbols=args.num_symbols,
            noise_level=args.noise_level
        )
        
        # Save results
        output = {
            "test_parameters": vars(args),
            "individual_results": results,
            "average_metrics": avg_metrics
        }
        
        output_path = output_dir / "test_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        # Print summary
        print("\nTest Results:")
        print(f"Number of tests: {args.num_tests}")
        print("\nAverage Metrics:")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.3f}")
        
        print(f"\nDetailed results saved to {output_path}")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())