import cv2
import numpy as np
import argparse
import os
import json
from PIL import Image
from scipy import ndimage
from pathlib import Path

# Configuration variables
TEXT_PADDING_X = 10  # Horizontal padding around text regions in pixels
TEXT_PADDING_Y = 5  # Vertical padding around text regions in pixels


def get_text_regions_from_json(spread_num, bubble_num):
    """
    Get text regions from the JSON file in the final-final-final directory.
    
    Args:
        spread_num (int): Spread number
        bubble_num (int): Bubble number
        
    Returns:
        list: List of text region dictionaries or None if not found
    """
    # Construct the filename
    filename = f"spread-{spread_num}-{bubble_num}_final.json"
    
    # Path to the final bounds file
    bounds_file = os.path.join(
        "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/final-bounds", 
        filename
    )
    
    # Check if the file exists
    if os.path.exists(bounds_file):
        with open(bounds_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Extract text regions from all bubbles
            all_regions = []
            if 'bubbles' in data:
                for bubble in data['bubbles']:
                    if 'text_regions' in bubble:
                        all_regions.extend(bubble['text_regions'])
                    # If no text_regions in bubble, use the bubble bounds itself
                    elif 'x' in bubble and 'y' in bubble and 'width' in bubble and 'height' in bubble:
                        all_regions.append({
                            'x': bubble['x'],
                            'y': bubble['y'],
                            'width': bubble['width'],
                            'height': bubble['height']
                        })
            return all_regions
    
    return None


def remove_text_from_speech_bubble(
    image_path, output_path=None, border_thickness=5, softness=1
):
    """
    Removes text from a speech bubble image while preserving the bubble structure
    with a specified border thickness and softness.

    Args:
        image_path (str): Path to the input speech bubble image (.webp)
        output_path (str, optional): Path to save the processed image. If None,
                                    will use input_name_blank.webp
        border_thickness (int): Thickness of the border in pixels
        softness (int): Amount of blur to apply to the border (0 for sharp)

    Returns:
        str: Path to the saved blank speech bubble image
    """
    if output_path is None:
        # Create default output filename
        name_parts = image_path.rsplit(".", 1)
        output_path = f"{name_parts[0]}_blank.webp"

    # Open the image with PIL to preserve transparency
    pil_img = Image.open(image_path)
    
    # Convert to numpy array for processing
    img = np.array(pil_img)
    
    # Check if image has an alpha channel
    if img.shape[2] == 4:
        # Try to extract spread and bubble numbers from the filename
        filename = os.path.basename(image_path)
        match = None
        
        # Try different filename patterns
        if filename.startswith("spch-"):
            spread_num = os.path.basename(os.path.dirname(image_path))
            # Extract just the number part between "spch-" and either "_" or "."
            bubble_part = filename.split("-")[1]
            if "_" in bubble_part:
                bubble_num = bubble_part.split("_")[0]
            else:
                # If no underscore, remove any file extension
                bubble_num = bubble_part.split(".")[0]
            
            match = (spread_num, bubble_num)
        
        if match:
            spread_num, bubble_num = match
            try:
                spread_num = int(spread_num)
                bubble_num = int(bubble_num)
                
                # Get text regions from JSON
                text_regions = get_text_regions_from_json(spread_num, bubble_num)
                
                if text_regions:
                    # Create a copy of the image
                    result = img.copy()
                    
                    # For each text region, fill with white
                    for region in text_regions:
                        x, y = region['x'], region['y']
                        width, height = region['width'], region['height']
                        
                        # Add some padding around the text region
                        padding_x = TEXT_PADDING_X
                        padding_y = TEXT_PADDING_Y
                        x = max(0, x - padding_x)
                        y = max(0, y - padding_y)
                        width = min(img.shape[1] - x, width + 2 * padding_x)
                        height = min(img.shape[0] - y, height + 2 * padding_y)
                        
                        # Fill the region with white
                        result[y:y+height, x:x+width, :3] = [255, 255, 255]  # White RGB
                        
                        # We'll remove the border drawing code for text regions
                        # as it's creating unwanted black borders
                    
                    # Convert back to PIL and save
                    result_pil = Image.fromarray(result)
                    result_pil.save(output_path)
                    
                    print(f"Processed image saved to {output_path} (using text regions)")
                    return output_path
            except (ValueError, TypeError) as e:
                print(f"Could not parse spread/bubble numbers from path: {e}")
                # Fall back to the original method
        
        # If we couldn't use text regions, fall back to the original method
        # Split the image into color channels and alpha
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
        
        # Convert the alpha channel to binary (transparent vs non-transparent)
        _, binary = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
        
        # Find large contours (the speech bubble outline)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create a mask for the entire bubble interior
        bubble_mask = np.zeros_like(binary)
        
        # Find the largest contour (assumes the speech bubble is the largest object)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(
                bubble_mask, [largest_contour], 0, 255, -1
            )  # Fill the contour
        
        # Create a mask for the border by finding the difference between the filled contour
        # and an eroded version of it with the specified border thickness
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(bubble_mask, kernel, iterations=border_thickness)
        border_mask = bubble_mask - eroded
        
        # Apply gaussian blur to soften the border if requested
        if softness > 0:
            border_mask_float = border_mask.astype(float) / 255.0
            border_mask_float = ndimage.gaussian_filter(
                border_mask_float, sigma=softness
            )
            border_mask = (border_mask_float * 255).astype(np.uint8)
        
        # Create the result image
        result = np.zeros_like(img)
        
        # First fill the entire bubble with white
        result[bubble_mask > 0, :3] = [255, 255, 255]  # White RGB
        result[bubble_mask > 0, 3] = 255  # Fully opaque
        
        # Then blend in the border based on border mask intensity
        # This creates a softer transition for the border
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if border_mask[y, x] > 0:
                    # Calculate opacity based on border mask intensity
                    opacity = border_mask[y, x] / 255.0
                    # Blend between white and black based on opacity
                    result[y, x, 0:3] = [int(255 * (1 - opacity))] * 3  # Blend to black
                    result[y, x, 3] = 255  # Keep fully opaque
        
        # Convert back to PIL and save
        result_pil = Image.fromarray(result)
        result_pil.save(output_path)
        
        print(f"Processed image saved to {output_path} (using fallback method)")
        return output_path
    else:
        print(f"Error: Input image {image_path} doesn't have an alpha channel")
        return None


def process_directory(input_dir, output_dir=None, border_thickness=4, softness=0.6, overwrite=False):
    """
    Process all spch-*.webp files in a directory to remove text from speech bubbles.

    Args:
        input_dir (str): Directory containing the speech bubble images
        output_dir (str, optional): Directory to save the processed images. If None,
                                    will use the default blanks directory.
        border_thickness (int): Thickness of the border in pixels
        softness (float): Amount of blur to apply to the border
        overwrite (bool): Whether to overwrite existing blank bubble files
    """
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/blanks"
    
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Counters for statistics
    files_processed = 0
    files_skipped = 0
    errors = 0

    # Recursively walk through the directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            # Process only webp files that match the pattern spch-*
            if filename.endswith(".webp") and filename.startswith("spch-") and not filename.endswith("_blank.webp"):
                input_file = os.path.join(root, filename)

                # Get the spread number from the directory name
                spread_dir = os.path.basename(root)
                
                # Create corresponding output directory
                spread_output_dir = os.path.join(output_dir, spread_dir)
                if not os.path.exists(spread_output_dir):
                    os.makedirs(spread_output_dir)
                
                # Determine output path
                output_file = os.path.join(
                    spread_output_dir, filename.replace(".webp", "_blank.webp")
                )

                # Skip if the output file already exists and overwrite is False
                if os.path.exists(output_file) and not overwrite:
                    print(f"Skipping {input_file} (output already exists at {output_file})")
                    files_skipped += 1
                    continue

                # Process the image
                try:
                    result = remove_text_from_speech_bubble(
                        input_file, output_file, border_thickness, softness
                    )
                    if result:
                        files_processed += 1
                    else:
                        errors += 1
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
                    errors += 1

    # Print summary
    print(f"\nProcessing complete:")
    print(f"  - Files processed successfully: {files_processed}")
    print(f"  - Files skipped (already exist): {files_skipped}")
    print(f"  - Errors encountered: {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove text from speech bubble images in a directory."
    )
    parser.add_argument(
        "input_path",
        type=str,
        nargs='?',
        default="/Users/robertsbrinkis/Documents/work/gamebook/Spreads/GIMP",
        help="Path to a directory containing speech bubble images or a single image file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/blanks",
        help="Output directory or file path for the blank speech bubbles",
    )
    parser.add_argument(
        "--thickness",
        "-t",
        type=int,
        default=4,
        help="Border thickness in pixels (default: 4)",
    )
    parser.add_argument(
        "--softness",
        "-s",
        type=float,
        default=0.6,
        help="Border softness, higher values create softer borders (default: 0.6, 0 for sharp)",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="Overwrite existing blank bubble files",
    )

    args = parser.parse_args()

    # Check if the input path is a directory or a file
    if os.path.isdir(args.input_path):
        # Process the entire directory
        process_directory(args.input_path, args.output, args.thickness, args.softness, args.overwrite)
    else:
        # Process a single file
        remove_text_from_speech_bubble(
            args.input_path, args.output, args.thickness, args.softness
        )


if __name__ == "__main__":
    main()

