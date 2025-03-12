#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
from pathlib import Path
import math
import argparse
from sklearn.cluster import DBSCAN

# Directory containing the JSON files with bounding box data
BOUNDS_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/bounds/initial-bounds"
# Directory to save the visualized images
OUTPUT_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/visualized"
# Directory to save the final processed bounds
FINAL_BOUNDS_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/bounds/final-bounds"
# Configuration variables
MAX_BUBBLES = 2
CLUSTERING_EPS = 225
VERTICAL_WEIGHT = 1.5
HORIZONTAL_PENALTY = 0.75

# ANSI color codes for better logging
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def ensure_directory_exists(directory):
    """Ensure the specified directory exists."""
    os.makedirs(directory, exist_ok=True)

def get_json_files(directory):
    """Get all JSON files from the specified directory."""
    json_files = []
    
    for file in os.listdir(directory):
        if file.endswith("_bounds.json"):
            json_files.append(os.path.join(directory, file))
    
    return json_files

def group_text_regions_by_distance(regions, eps=225, vertical_weight=1.5, horizontal_penalty=0.5):
    """
    Group text regions into separate bubbles based on distance.
    
    Args:
        regions: List of text region dictionaries
        eps: Maximum distance between two points to be considered in the same cluster
        vertical_weight: Weight factor for vertical distances (higher values make vertical
                         distances more significant than horizontal distances)
        horizontal_penalty: Penalty factor for horizontal distances (lower values make
                           horizontal distances less significant)
        
    Returns:
        List of lists, where each inner list contains text regions for one bubble
    """
    if not regions:
        return []
    
    # Extract center points of each text region
    points = []
    for region in regions:
        x = region['x'] + region['width'] // 2
        y = region['y'] + region['height'] // 2
        # Scale y-coordinates by the vertical weight to make vertical distances more significant
        # Scale x-coordinates by the horizontal penalty to make horizontal distances less significant
        points.append([x * horizontal_penalty, y * vertical_weight])
    
    # Use DBSCAN clustering to group nearby text regions
    clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
    labels = clustering.labels_
    
    # Group regions by cluster label
    grouped_regions = {}
    for i, label in enumerate(labels):
        if label not in grouped_regions:
            grouped_regions[label] = []
        grouped_regions[label].append(regions[i])
    
    return list(grouped_regions.values())

def merge_overlapping_bubbles(bubble_groups):
    """
    Merge bubble groups that have overlapping bounding boxes.
    
    Args:
        bubble_groups: List of lists, where each inner list contains text regions for one bubble
        
    Returns:
        List of lists with overlapping bubbles merged
    """
    if not bubble_groups or len(bubble_groups) <= 1:
        return bubble_groups
    
    # Calculate bounding boxes for each bubble group
    bubble_boxes = []
    for bubble in bubble_groups:
        min_x = min(region['x'] for region in bubble)
        min_y = min(region['y'] for region in bubble)
        max_x = max(region['x'] + region['width'] for region in bubble)
        max_y = max(region['y'] + region['height'] for region in bubble)
        bubble_boxes.append((min_x, min_y, max_x, max_y))
    
    # Check for overlaps and merge until no more overlaps exist
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(bubble_groups):
            j = i + 1
            while j < len(bubble_groups):
                # Check if bubbles i and j overlap
                box_i = bubble_boxes[i]
                box_j = bubble_boxes[j]
                
                # Check for overlap
                if (box_i[0] < box_j[2] and box_i[2] > box_j[0] and
                    box_i[1] < box_j[3] and box_i[3] > box_j[1]):
                    # Merge bubble j into bubble i
                    bubble_groups[i].extend(bubble_groups[j])
                    
                    # Update bounding box for the merged bubble
                    min_x = min(box_i[0], box_j[0])
                    min_y = min(box_i[1], box_j[1])
                    max_x = max(box_i[2], box_j[2])
                    max_y = max(box_i[3], box_j[3])
                    bubble_boxes[i] = (min_x, min_y, max_x, max_y)
                    
                    # Remove bubble j
                    bubble_groups.pop(j)
                    bubble_boxes.pop(j)
                    
                    merged = True
                else:
                    j += 1
            
            if merged:
                break
            i += 1
    
    return bubble_groups

def filter_and_prioritize_bubbles(bubble_groups, max_bubbles=MAX_BUBBLES):
    """
    Filter bubble groups to keep only the largest ones, up to max_bubbles.
    
    Args:
        bubble_groups: List of lists, where each inner list contains text regions for one bubble
        max_bubbles: Maximum number of bubbles to keep
        
    Returns:
        List of the largest bubble groups, up to max_bubbles
    """
    if not bubble_groups or len(bubble_groups) <= max_bubbles:
        return bubble_groups
    
    # Calculate area for each bubble group
    bubble_areas = []
    for bubble in bubble_groups:
        min_x = min(region['x'] for region in bubble)
        min_y = min(region['y'] for region in bubble)
        max_x = max(region['x'] + region['width'] for region in bubble)
        max_y = max(region['y'] + region['height'] for region in bubble)
        area = (max_x - min_x) * (max_y - min_y)
        bubble_areas.append((area, bubble))
    
    # Sort bubbles by area in descending order and keep only the largest ones
    bubble_areas.sort(reverse=True)
    return [bubble for _, bubble in bubble_areas[:max_bubbles]]

def save_final_bounds(data, bubble_groups, json_path, overwrite=False):
    """
    Save the final processed bounds as a new JSON file.
    
    Args:
        data: Original JSON data
        bubble_groups: Processed bubble groups
        json_path: Path to the original JSON file
        overwrite: Whether to overwrite existing files
    
    Returns:
        Path to the saved final bounds JSON file or None if file exists and overwrite is False
    """
    # Create a new data structure for the final bounds
    final_data = {
        'image': data['image'],
        'path': data['path'],
        'folder': data['folder'],
        'bubble_number': data['bubble_number'],
        'bubbles': []
    }
    
    # Add each bubble group
    for i, bubble in enumerate(bubble_groups):
        # Find the bounding box that contains all text regions in this bubble
        min_x = min(region['x'] for region in bubble)
        min_y = min(region['y'] for region in bubble)
        max_x = max(region['x'] + region['width'] for region in bubble)
        max_y = max(region['y'] + region['height'] for region in bubble)
        
        # Combine all text in this bubble
        full_text = " ".join(region['text'] for region in bubble)
        
        # Add bubble data
        final_data['bubbles'].append({
            'bubble_number': i + 1,
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'text': full_text,
            'text_regions': bubble
        })
    
    # Create output filename
    base_filename = os.path.basename(json_path)
    output_path = os.path.join(FINAL_BOUNDS_DIR, base_filename.replace('_bounds.json', '_final.json'))
    
    # Check if file exists and we're not overwriting
    if os.path.exists(output_path) and not overwrite:
        print(f"{YELLOW}Skipping: Final bounds file already exists at {output_path}{RESET}")
        print(f"{YELLOW}Use -o/--overwrite to overwrite existing files{RESET}")
        return None
    
    # Save the final bounds
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    
    print(f"{GREEN}Saved final bounds to {output_path}{RESET}")
    return output_path

def visualize_speech_bubbles(json_path, overwrite=False):
    """
    Visualize speech bubbles on the image and save the result.
    
    Args:
        json_path: Path to the JSON file with bounding box data
        overwrite: Whether to overwrite existing files
    
    Returns:
        Path to the saved visualized image or None if processing failed
    """
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the image path from the JSON data
    image_path = data['path']
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"{RED}Error: Image {image_path} does not exist{RESET}")
        return None
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"{RED}Error: Could not read image {image_path}{RESET}")
        return None
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Get all text regions
    regions = data['text_regions']
    
    if not regions:
        print(f"{YELLOW}No text regions found in {json_path}{RESET}")
        return None
    
    # Group text regions into separate bubbles
    bubble_groups = group_text_regions_by_distance(regions, eps=CLUSTERING_EPS, 
                                                 vertical_weight=VERTICAL_WEIGHT, 
                                                 horizontal_penalty=HORIZONTAL_PENALTY)
    
    # Merge overlapping bubbles
    bubble_groups = merge_overlapping_bubbles(bubble_groups)
    
    # Filter and prioritize bubbles (keep only the 2 largest)
    bubble_groups = filter_and_prioritize_bubbles(bubble_groups)
    
    # Save the final processed bounds
    final_bounds_path = save_final_bounds(data, bubble_groups, json_path, overwrite)
    
    # If we're not overwriting and the file exists, skip visualization too
    if final_bounds_path is None:
        return None
    
    # Colors for different bubbles (BGR format)
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
    ]
    
    # Draw bounding boxes for each bubble group
    for i, bubble in enumerate(bubble_groups):
        color = colors[i % len(colors)]
        
        # Find the bounding box that contains all text regions in this bubble
        min_x = min(region['x'] for region in bubble)
        min_y = min(region['y'] for region in bubble)
        max_x = max(region['x'] + region['width'] for region in bubble)
        max_y = max(region['y'] + region['height'] for region in bubble)
        
        # Draw a rectangle around the bubble
        cv2.rectangle(vis_image, (min_x, min_y), (max_x, max_y), color, 2)
        
        # Combine all text in this bubble
        full_text = " ".join(region['text'] for region in bubble)
        
        # Add the full text above the rectangle with bubble number
        label = f"Bubble {i+1}: {full_text}"
        cv2.putText(vis_image, label, (min_x, min_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Create output filename
    folder = data['folder']
    bubble_number = data['bubble_number']
    output_filename = f"vis_bubbles_spread-{folder}-{bubble_number}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Check if visualization file exists and we're not overwriting
    if os.path.exists(output_path) and not overwrite:
        print(f"{YELLOW}Skipping: Visualization file already exists at {output_path}{RESET}")
        return None
    
    # Save the visualized image
    cv2.imwrite(output_path, vis_image)
    print(f"{GREEN}Saved visualized image to {output_path}{RESET}")
    
    return output_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize speech bubbles and generate final bounds")
    parser.add_argument("-o", "--overwrite", action="store_true", 
                        help="Overwrite existing final bounds files")
    args = parser.parse_args()
    
    # Ensure output directories exist
    ensure_directory_exists(OUTPUT_DIR)
    ensure_directory_exists(FINAL_BOUNDS_DIR)
    
    # Get all JSON files
    json_files = get_json_files(BOUNDS_DIR)
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for json_path in json_files:
        print(f"Processing {json_path}...")
        result = visualize_speech_bubbles(json_path, args.overwrite)
        
        if result:
            success_count += 1
        elif result is None and os.path.exists(os.path.join(
                FINAL_BOUNDS_DIR, 
                os.path.basename(json_path).replace('_bounds.json', '_final.json'))):
            skip_count += 1
        else:
            error_count += 1
    
    # Print summary
    print(f"\nVisualization complete:")
    print(f"  - {GREEN}Files processed successfully: {success_count}{RESET}")
    print(f"  - {YELLOW}Files skipped (already exist): {skip_count}{RESET}")
    print(f"  - {RED}Errors encountered: {error_count}{RESET}")
    
    if skip_count > 0 and not args.overwrite:
        print(f"\n{YELLOW}Note: Use -o or --overwrite to overwrite existing files{RESET}")

if __name__ == "__main__":
    main()