#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import pytesseract
from pathlib import Path
import re

# Directory containing the speech bubble images
SOURCE_DIR = "/Users/robertsbrinkis/Documents/work/gamebook/Spreads/GIMP"
# Output directory for JSON files
OUTPUT_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/bounds/initial-bounds"
# Regex pattern for valid Spanish text (letters, numbers, punctuation, and spaces)
# Modified to properly handle Spanish words with punctuation
SPANISH_TEXT_PATTERN = r'^[a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ¿¡.,;:()\-\'\"\s?!]+$'

# ANSI color codes for better logging
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Log separator
LOG_SEPARATOR = f"{YELLOW}{'—' * 50}{RESET}"

def ensure_directory_exists(directory):
    """Ensure the specified directory exists."""
    os.makedirs(directory, exist_ok=True)
    print(f"{BLUE}Directory ensured: {directory}{RESET}")

def get_speech_bubble_files(directory):
    """Get all speech bubble files from the specified directory structure."""
    speech_bubble_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Only process files that match the pattern spch-*.webp
            if file.startswith("spch-") and file.endswith(".webp"):
                speech_bubble_files.append(os.path.join(root, file))
    
    return speech_bubble_files

def detect_text_regions(image_path):
    """Detect text regions in the image using Tesseract OCR."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"{RED}Error: Could not read image {image_path}{RESET}")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing to improve text detection
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Try multiple configurations for better detection
    bounding_boxes = []
    
    # Try different PSM modes
    psm_modes = [11, 6, 3]
    for psm in psm_modes:
        custom_config = f'--oem 3 --psm {psm} -l spa'
        data = pytesseract.image_to_data(enhanced, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Extract bounding boxes for text regions
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            # Lower confidence threshold to 40 to catch more text
            if int(data['conf'][i]) > 40 and data['text'][i].strip() != '':
                text = data['text'][i].strip()
                
                # Filter out non-Spanish text using regex
                if re.match(SPANISH_TEXT_PATTERN, text):
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bounding_boxes.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'text': text,
                        'confidence': int(data['conf'][i])
                    })
                    print(f"{GREEN}Found text: '{text}' (confidence: {int(data['conf'][i])}%){RESET}")
                else:
                    print(f"{YELLOW}Filtered out non-Spanish text: '{text}'{RESET}")
        
        # If we found text, no need to try other PSM modes
        if bounding_boxes:
            break
    
    # If still no text detected, try direct OCR on the whole image
    if not bounding_boxes:
        # Try direct OCR with different config
        custom_config = '--oem 3 --psm 4 -l spa'
        text = pytesseract.image_to_string(enhanced, config=custom_config).strip()
        
        if text and re.match(SPANISH_TEXT_PATTERN, text):
            print(f"{GREEN}Found text using whole image OCR: '{text}'{RESET}")
            # Create a bounding box for the entire image with some margin
            h, w = enhanced.shape
            margin = int(min(w, h) * 0.1)
            bounding_boxes.append({
                'x': margin,
                'y': margin,
                'width': w - 2*margin,
                'height': h - 2*margin,
                'text': text,
                'confidence': 70  # Assign a reasonable confidence
            })
    
    return bounding_boxes

def save_as_json(data, image_path, output_dir):
    """Save the bounding box data as JSON."""
    # Get the folder number (e.g., '0', '1', '2') from the path
    folder_name = os.path.basename(os.path.dirname(image_path))
    
    # Get the base filename without extension (e.g., 'spch-0')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create a structured filename for the JSON: spread-FOLDER-NUMBER_bounds.json
    # Extract the bubble number from the filename (e.g., '0' from 'spch-0')
    bubble_number = base_name.split('-')[1]
    json_filename = f"spread-{folder_name}-{bubble_number}_bounds.json"
    
    # Create the full path for the JSON file
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"{CYAN}Saved bounding box data to {json_path}{RESET}")
    return json_path

def main():
    print(f"{MAGENTA}Starting speech bubble detection...{RESET}")
    print(LOG_SEPARATOR)
    
    # Ensure output directory exists
    ensure_directory_exists(OUTPUT_DIR)
    
    # Get all speech bubble files
    speech_bubble_files = get_speech_bubble_files(SOURCE_DIR)
    print(f"{BLUE}Found {len(speech_bubble_files)} speech bubble files{RESET}")
    print(LOG_SEPARATOR)
    
    # Process each image
    for i, image_path in enumerate(speech_bubble_files):
        print(f"{MAGENTA}Processing [{i+1}/{len(speech_bubble_files)}] {image_path}...{RESET}")
        bounding_boxes = detect_text_regions(image_path)
        
        if bounding_boxes:
            # Get folder name (e.g., '0', '1', '2')
            folder_name = os.path.basename(os.path.dirname(image_path))
            
            # Get bubble number from filename (e.g., '0' from 'spch-0.webp')
            bubble_number = os.path.basename(image_path).split('-')[1].split('.')[0]
            
            # Create a data structure for the JSON
            data = {
                'image': os.path.basename(image_path),
                'path': image_path,
                'folder': folder_name,
                'bubble_number': bubble_number,
                'text_regions': bounding_boxes
            }
            
            # Save the data as JSON
            save_as_json(data, image_path, OUTPUT_DIR)
            print(f"{GREEN}✓ Successfully processed with {len(bounding_boxes)} text regions{RESET}")
        else:
            print(f"{RED}✗ No text regions detected in {image_path}{RESET}")
        
        print(LOG_SEPARATOR)
    
    print(f"{MAGENTA}Speech bubble detection completed!{RESET}")

if __name__ == "__main__":
    main()