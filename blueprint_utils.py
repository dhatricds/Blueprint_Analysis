"""
Utility functions for blueprint processing and symbol detection.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile

def convert_pdf_to_images(pdf_path, dpi=200):
    """
    Convert a PDF file to a list of images using pdftoppm.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for conversion (default: 200)
        
    Returns:
        List of paths to the generated images
    """
    # Create temporary directory for images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to images using pdftoppm
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_prefix = os.path.join(temp_dir, base_name)
        
        try:
            subprocess.run([
                "pdftoppm",
                "-jpeg",
                "-r", str(dpi),
                pdf_path,
                output_prefix
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting PDF: {e}")
            return []
        except FileNotFoundError:
            print("Error: pdftoppm not found. Please install poppler-utils.")
            return []
        
        # Get list of generated images
        image_paths = sorted(Path(temp_dir).glob("*.jpg"))
        if not image_paths:
            print("No images generated from PDF")
            return []
        
        # Copy images to a more permanent location
        output_dir = os.path.join("output", "pages")
        os.makedirs(output_dir, exist_ok=True)
        
        final_paths = []
        for i, img_path in enumerate(image_paths):
            output_path = os.path.join(output_dir, f"{base_name}_page{i+1}.jpg")
            cv2.imwrite(output_path, cv2.imread(str(img_path)))
            final_paths.append(output_path)
        
        return final_paths

def create_tiles(image, tile_size=640, overlap=6):
    """
    Create overlapping tiles from an image.
    
    Args:
        image: Input image as numpy array
        tile_size: Size of tiles (default: 640)
        overlap: Overlap between tiles in pixels (default: 6)
        
    Returns:
        List of tuples (tile, coordinates)
    """
    if image is None:
        return []
    
    height, width = image.shape[:2]
    tiles = []
    
    # Calculate step size (tile size minus overlap)
    step = tile_size - overlap
    
    # Calculate number of tiles in each dimension
    num_tiles_h = max(1, int(np.ceil((height - overlap) / step)))
    num_tiles_w = max(1, int(np.ceil((width - overlap) / step)))
    
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            # Calculate tile coordinates
            x = j * step
            y = i * step
            
            # Adjust coordinates for edge tiles
            if x + tile_size > width:
                x = max(0, width - tile_size)
            if y + tile_size > height:
                y = max(0, height - tile_size)
            
            # Extract tile
            tile = image[y:y+tile_size, x:x+tile_size]
            
            # Skip if tile is too small
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                continue
            
            tiles.append((tile, (x, y)))
    
    return tiles

def is_blank_tile(tile, threshold=0.98, black_percentage=0.95):
    """
    Check if a tile is mostly blank (white or black).
    
    Args:
        tile: Input tile image
        threshold: Threshold for considering a pixel blank (default: 0.98)
        black_percentage: Percentage of black pixels for black tile (default: 0.95)
        
    Returns:
        Boolean indicating if tile is blank
    """
    if tile is None:
        return True
    
    # Convert to grayscale if needed
    if len(tile.shape) > 2:
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    else:
        gray = tile
    
    # Check for mostly white
    white_pixels = np.sum(gray >= 255 * threshold)
    total_pixels = gray.size
    white_ratio = white_pixels / total_pixels
    
    # Check for mostly black
    black_pixels = np.sum(gray <= 255 * (1 - threshold))
    black_ratio = black_pixels / total_pixels
    
    return white_ratio >= threshold or black_ratio >= black_percentage

def invert_image(image):
    """
    Invert an image for better symbol detection.
    
    Args:
        image: Input image
        
    Returns:
        Inverted image
    """
    return cv2.bitwise_not(image)

def process_reference_image(image_path, target_size=None):
    """
    Load and process a reference symbol image.
    
    Args:
        image_path: Path to reference image
        target_size: Optional size to resize to
        
    Returns:
        Processed reference image
    """
    # Load image
    reference = cv2.imread(image_path)
    if reference is None:
        raise ValueError(f"Cannot load reference image: {image_path}")
    
    # Convert to grayscale
    if len(reference.shape) > 2:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    # Resize if target size specified
    if target_size:
        reference = cv2.resize(reference, (target_size, target_size))
    
    # Threshold to binary
    _, reference = cv2.threshold(reference, 127, 255, cv2.THRESH_BINARY)
    
    return reference

def setup_output_directories(base_dir="output", debug=False):
    """
    Create necessary output directories.
    
    Args:
        base_dir: Base output directory (default: "output")
        debug: Whether to create debug directories
        
    Returns:
        Dictionary of directory paths
    """
    directories = {
        "base": base_dir,
        "pages": os.path.join(base_dir, "pages"),
        "tiles": os.path.join(base_dir, "tiles"),
        "inverted_tiles": os.path.join(base_dir, "inverted_tiles"),
        "results": os.path.join(base_dir, "results")
    }
    
    if debug:
        directories.update({
            "debug": os.path.join(base_dir, "debug"),
            "all_tiles": os.path.join(base_dir, "all_tiles")
        })
    
    # Create all directories
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    
    return directories