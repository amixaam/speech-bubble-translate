#!/usr/bin/env python3
import os
import json
import argparse
import textwrap
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
from pathlib import Path

# Font settings
FONT_PATH = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/McLaren-Regular.ttf"
FONT_SIZE = 16
TEXT_COLOR = (0, 0, 0)  # Black

## Directories
FINAL_BOUNDS_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/bounds/final-bounds"
BLANK_BUBBLES_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/blanks"
TRANSLATED_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/translated"
BASE_DIR = "/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate"

# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# Log separator
LOG_SEPARATOR = f"{YELLOW}{'—' * 50}{RESET}"

# Counters for statistics
SUCCESS_COUNT = 0
FAILURE_COUNT = 0
WARNING_COUNT = 0

def get_relative_path(path):
    """Convert absolute path to relative path from BASE_DIR"""
    try:
        return os.path.relpath(path, BASE_DIR)
    except:
        return path


def ensure_directory_exists(directory):
    """Ensure the specified directory exists."""
    os.makedirs(directory, exist_ok=True)


def find_speech_bubble_bounds(image_path):
    """
    Detect the boundaries of the speech bubble in the image.
    
    Args:
        image_path (str): Path to the speech bubble image
        
    Returns:
        tuple: (x, y, width, height) of the speech bubble bounds
    """
    # Open image with PIL to handle transparency
    pil_img = Image.open(image_path)
    
    # Convert to numpy array
    img = np.array(pil_img)
    
    # If the image has an alpha channel, use it to find the bubble
    if img.shape[2] == 4:
        # Extract alpha channel
        alpha = img[:, :, 3]
        
        # Threshold the alpha channel to get a binary mask of the bubble
        _, binary = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # If no contours found, return the full image bounds
        if not contours:
            return (0, 0, img.shape[1], img.shape[0])
        
        # Find the largest contour (assuming it's the speech bubble)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding (10% of width/height)
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        # Ensure bounds don't exceed image dimensions
        x = max(0, x + padding_x)
        y = max(0, y + padding_y)
        w = min(img.shape[1] - x, w - 2 * padding_x)
        h = min(img.shape[0] - y, h - 2 * padding_y)
        
        return (x, y, w, h)
    else:
        # If no alpha channel, use the entire image
        return (0, 0, img.shape[1], img.shape[0])


def get_processed_bounds(spread_num, bubble_num):
    """
    Get the processed bounds from the final-bounds directory.
    
    Args:
        spread_num (int): Spread number
        bubble_num (int): Bubble number
        
    Returns:
        list: List of bubble bounds or None if not found
    """
    # Construct the filename
    filename = f"spread-{spread_num}-{bubble_num}_final.json"
    
    # Path to the final bounds file
    bounds_file = os.path.join(FINAL_BOUNDS_DIR, filename)
    
    # Check if the file exists
    if os.path.exists(bounds_file):
        with open(bounds_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('bubbles', [])
    
    # If the file doesn't exist, try the original bounds file
    filename = f"spread-{spread_num}-{bubble_num}_bounds.json"
    bounds_file = os.path.join(FINAL_BOUNDS_DIR, filename)
    
    if os.path.exists(bounds_file):
        print(f"{YELLOW}Warning: Using legacy bounds file for spread-{spread_num}-{bubble_num}{RESET}")
        return None
    
    return None

def fit_text_to_bubble(draw, text, font_path, bubble_width, bubble_height, font_size=FONT_SIZE):
    """
    Find the optimal way to fit the text within the bubble using the specified font size.
    
    Args:
        draw (ImageDraw): ImageDraw object
        text (str): Text to fit
        font_path (str): Path to the font file
        bubble_width (int): Width of the speech bubble
        bubble_height (int): Height of the speech bubble
        font_size (int): Font size to use
        
    Returns:
        tuple: (Font, list of text lines, line height)
    """
    # Start with a reasonable font size
    min_font_size = 12
    max_font_size = 64  # Upper limit to prevent extremely large text
    
    # Try different font sizes to find the best fit with word wrapping
    for test_size in range(max_font_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, test_size)
        line_height = font.getbbox("Tg")[3] + 4  # Add space between lines
        
        # First try to wrap without breaking words
        lines = textwrap.wrap(text, width=int(bubble_width * 0.9 / font.getbbox("m")[2]))
        
        # Calculate total height needed
        total_height = len(lines) * line_height
        
        # Check if all lines fit within the bubble width and the total height fits
        all_lines_fit = True
        for line in lines:
            if font.getbbox(line)[2] > bubble_width * 0.95:
                all_lines_fit = False
                break
        
        # If text fits both width and height constraints, use this font size
        if all_lines_fit and total_height <= bubble_height * 0.95:
            return font, lines, line_height
        
        # If it doesn't fit, try with hyphenation
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            # Skip empty words
            if not word:
                continue
                
            # Try adding the word to the current line
            test_line = current_line + " " + word if current_line else word
            
            # Check if it fits
            if font.getbbox(test_line)[2] <= bubble_width * 0.95:
                current_line = test_line
            else:
                # If the word is too long, hyphenate it
                if font.getbbox(word)[2] > bubble_width * 0.95:
                    # Add the current line if it's not empty
                    if current_line:
                        lines.append(current_line)
                        current_line = ""
                    
                    # Hyphenate the long word
                    remaining_word = word
                    while remaining_word:
                        found_fit = False
                        for i in range(len(remaining_word), 0, -1):
                            test_segment = remaining_word[:i]
                            if i < len(remaining_word):
                                test_segment += "-"
                            
                            if font.getbbox(test_segment)[2] <= bubble_width * 0.95:
                                lines.append(test_segment)
                                remaining_word = remaining_word[i:]
                                found_fit = True
                                break
                        
                        # If we couldn't find any fit, force break the first character
                        # Only try to access remaining_word[0] if remaining_word is not empty
                        if not found_fit:
                            if remaining_word:  # Check if remaining_word is not empty
                                lines.append(remaining_word[0])
                                remaining_word = remaining_word[1:] if len(remaining_word) > 1 else ""
                            else:
                                # If we somehow got here with an empty string, break the loop
                                break
                else:
                    # Add the current line and start a new one with this word
                    if current_line:
                        lines.append(current_line)
                    current_line = word
        
        # Add the last line if there's anything left
        if current_line:
            lines.append(current_line)
        
        # Check if the hyphenated version fits
        total_height = len(lines) * line_height
        if total_height <= bubble_height * 0.95:
            return font, lines, line_height
    
    # If we get here, use the smallest font size with hyphenation
    font = ImageFont.truetype(font_path, min_font_size)
    line_height = font.getbbox("Tg")[3] + 4
    
    # Apply the same hyphenation logic with the smallest font
    lines = []
    words = text.split()
    current_line = ""
    
    for word in words:
        # Skip empty words
        if not word:
            continue
            
        test_line = current_line + " " + word if current_line else word
        
        if font.getbbox(test_line)[2] <= bubble_width * 0.95:
            current_line = test_line
        else:
            if font.getbbox(word)[2] > bubble_width * 0.95:
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                
                remaining_word = word
                while remaining_word:
                    found_fit = False
                    for i in range(len(remaining_word), 0, -1):
                        test_segment = remaining_word[:i]
                        if i < len(remaining_word):
                            test_segment += "-"
                        
                        if font.getbbox(test_segment)[2] <= bubble_width * 0.95:
                            lines.append(test_segment)
                            remaining_word = remaining_word[i:]
                            found_fit = True
                            break
                    
                    # If we couldn't find any fit, force break the first character
                    if not found_fit:
                        if remaining_word:  # Check if remaining_word is not empty
                            lines.append(remaining_word[0])
                            remaining_word = remaining_word[1:] if len(remaining_word) > 1 else ""
                        else:
                            # If we somehow got here with an empty string, break the loop
                            break
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return font, lines, line_height


def add_text_to_bubble(blank_bubble_path, text, output_path, spread_num, bubble_num, font_path=FONT_PATH, font_size=FONT_SIZE, bubble_index=0):
    """
    Add text to a speech bubble image using processed bounds.
    
    Args:
        blank_bubble_path (str): Path to the blank speech bubble image
        text (str): Text to add to the bubble
        output_path (str): Path to save the resulting image
        spread_num (int): Spread number
        bubble_num (int): Bubble number
        font_path (str): Path to the font file
        font_size (int): Font size to use (used as a maximum size)
        bubble_index (int): Index of the bubble within the image (0-based)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the image
        img = Image.open(blank_bubble_path)
        
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        
        # Get the processed bounds
        bubbles = get_processed_bounds(spread_num, bubble_num)
        
        if not bubbles:
            print(f"Warning: No processed bounds found for spread-{spread_num}-{bubble_num}")
            return False
        
        # Use the specified bubble based on bubble_index
        if bubble_index < len(bubbles):
            bubble = bubbles[bubble_index]
        else:
            print(f"Warning: Bubble index {bubble_index} out of range for spread-{spread_num}-{bubble_num}, using first bubble")
            bubble = bubbles[0]
            
        x, y = bubble['x'], bubble['y']
        width, height = bubble['width'], bubble['height']
        
        # Clean up text (remove excess whitespace)
        text = text.strip()
        
        # Fit text to bubble with word wrapping
        max_size = min(font_size, 32)  # Cap at 32 or the specified font_size, whichever is smaller
        font, lines, line_height = fit_text_to_bubble(
            draw, text, font_path, width, height, max_size
        )
        
        # Get the actual font size used
        actual_font_size = font.size
        
        # Calculate vertical positioning (center the text block in the bubble)
        total_text_height = len(lines) * line_height
        y_offset = y + (height - total_text_height) // 2
        
        # Draw each line of text
        for i, line in enumerate(lines):
            # Calculate width of this line
            line_width = font.getbbox(line)[2]
            
            # Center the line horizontally within the bubble
            x_position = x + (width - line_width) // 2
            y_position = y_offset + i * line_height
            
            # Draw the text
            draw.text((x_position, y_position), line, fill=TEXT_COLOR, font=font)
        
        # Remove the bounding box visualization code
        # overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        # overlay_draw = ImageDraw.Draw(overlay)
        # overlay_draw.rectangle([x, y, x + width, y + height], outline=(255, 0, 0, 128), width=2)
        # img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        # Save the image
        img.save(output_path)
        print(f"Created: {output_path} (font size: {actual_font_size}, lines: {len(lines)}, bubble: {bubble_index+1}/{len(bubbles)})")
        return True
        
    except Exception as e:
        print(f"Error processing {blank_bubble_path}: {str(e)}")
        return False


def process_bubbles(language, font_size=FONT_SIZE):
    """
    Process all blank speech bubbles for a specific language.
    
    Args:
        language (str): Language code
        font_size (int): Font size to use
    """
    # Create output directory
    output_dir = f"/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/final/{language}"
    ensure_directory_exists(output_dir)
    
    # Path to the translated text directory for this language
    translated_lang_dir = os.path.join(TRANSLATED_DIR, language)
    
    # Check if translated directory exists
    if not os.path.exists(translated_lang_dir):
        print(f"{RED}Error: Translated directory for language '{language}' does not exist.{RESET}")
        return
    
    print(LOG_SEPARATOR)
    print(f"Processing bubbles for language: {language}")
    print(LOG_SEPARATOR)
    
    # Walk through the blank bubbles directory
    for root, dirs, files in os.walk(BLANK_BUBBLES_DIR):
        # Get the spread number from the directory name
        spread_dir = os.path.basename(root)
        
        try:
            spread_num = int(spread_dir)
            
            # Create corresponding output directory
            spread_output_dir = os.path.join(output_dir, spread_dir)
            ensure_directory_exists(spread_output_dir)
            
            # Process each blank bubble file
            for file in files:
                if file.endswith("_blank.webp") and file.startswith("spch-"):
                    # Extract bubble number
                    match = re.match(r"spch-(\d+)_blank\.webp", file)
                    if match:
                        bubble_num = int(match.group(1))
                        
                        # Get the processed bounds to determine how many bubbles are in this image
                        bubbles = get_processed_bounds(spread_num, bubble_num)
                        
                        if not bubbles:
                            print(f"{YELLOW}Warning: No processed bounds found for spread-{spread_num}-{bubble_num}{RESET}")
                            print(LOG_SEPARATOR)
                            continue
                        
                        # Input and output paths
                        blank_bubble_path = os.path.join(root, file)
                        output_filename = f"spch-{bubble_num}-{language}.webp"
                        output_path = os.path.join(spread_output_dir, output_filename)
                        
                        print(f"Processing: {get_relative_path(blank_bubble_path)}")
                        
                        # Process all bubbles in the image at once
                        process_all_bubbles_in_image(
                            blank_bubble_path,
                            output_path,
                            spread_num,
                            bubble_num,
                            bubbles,
                            language,
                            font_path=FONT_PATH,
                            font_size=font_size
                        )
        except ValueError:
            # Skip directories that are not numbers
            continue


def process_all_bubbles_in_image(blank_bubble_path, output_path, spread_num, bubble_num, bubbles, language, font_path=FONT_PATH, font_size=FONT_SIZE):
    """
    Process all bubbles in a single image and draw text for all of them.
    
    Args:
        blank_bubble_path (str): Path to the blank speech bubble image
        output_path (str): Path to save the resulting image
        spread_num (int): Spread number
        bubble_num (int): Bubble number
        bubbles (list): List of bubble data
        language (str): Language code
        font_path (str): Path to the font file
        font_size (int): Font size to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    global SUCCESS_COUNT, FAILURE_COUNT, WARNING_COUNT
    
    try:
        # Open the image
        img = Image.open(blank_bubble_path)
        
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        
        # Remove the overlay creation for bounding boxes
        # overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        # overlay_draw = ImageDraw.Draw(overlay)
        
        # Track if any bubbles were successfully processed
        bubbles_processed = 0
        
        # Process each bubble
        for bubble_index, bubble in enumerate(bubbles):
            print(f"Processing bubble {bubble_index+1}/{len(bubbles)} in {get_relative_path(blank_bubble_path)}")
            
            # Get the translated text for this specific bubble
            translated_text = get_translated_text(spread_num, bubble_num, bubble_index, language, TRANSLATED_DIR)
            
            if not translated_text:
                print(f"{YELLOW}Warning: No translated text found for spread-{spread_num}-{bubble_num}, bubble {bubble_index+1}{RESET}")
                WARNING_COUNT += 1
                continue
                
            # Validate bubble dimensions
            x, y = bubble.get('x', 0), bubble.get('y', 0)
            width, height = bubble.get('width', 0), bubble.get('height', 0)
            print(f"Bubble dimensions: x={x}, y={y}, width={width}, height={height}")
            
            # Skip bubbles with invalid dimensions
            if width <= 0 or height <= 0:
                print(f"{YELLOW}Warning: Invalid bubble dimensions (width={width}, height={height}) for spread-{spread_num}-{bubble_num}, bubble {bubble_index+1}{RESET}")
                WARNING_COUNT += 1
                continue
            
            # Clean up text (remove excess whitespace)
            text = translated_text.strip()
            print(f"Text to insert: '{text}' (length: {len(text)})")
            
            # Skip empty text
            if not text:
                print(f"{YELLOW}Warning: Empty text for spread-{spread_num}-{bubble_num}, bubble {bubble_index+1}{RESET}")
                WARNING_COUNT += 1
                continue
            
            try:
                print(f"Attempting to fit text to bubble (max font size: {min(font_size, 32)})")
                # Fit text to bubble with word wrapping
                max_size = min(font_size, 32)  # Cap at 32 or the specified font_size, whichever is smaller
                font, lines, line_height = fit_text_to_bubble(
                    draw, text, font_path, width, height, max_size
                )
                
                # Get the actual font size used
                actual_font_size = font.size
                print(f"Text fitted with font size {actual_font_size} and {len(lines)} lines")
                
                # Calculate vertical positioning (center the text block in the bubble)
                total_text_height = len(lines) * line_height
                y_offset = y + (height - total_text_height) // 2
                
                # Draw each line of text
                for i, line in enumerate(lines):
                    # Calculate width of this line
                    line_width = font.getbbox(line)[2]
                    
                    # Center the line horizontally within the bubble
                    x_position = x + (width - line_width) // 2
                    y_position = y_offset + i * line_height
                    
                    print(f"Drawing line {i+1}: '{line}' at position ({x_position}, {y_position})")
                    # Draw the text
                    draw.text((x_position, y_position), line, fill=TEXT_COLOR, font=font)
                
                # Remove the bounding box visualization
                # overlay_draw.rectangle([x, y, x + width, y + height], outline=(255, 0, 0, 128), width=2)
                
                print(f"Added text for bubble {bubble_index+1}/{len(bubbles)} (font size: {actual_font_size}, lines: {len(lines)})")
                bubbles_processed += 1
            except Exception as e:
                import traceback
                print(f"{RED}Error processing bubble {bubble_index+1} in {get_relative_path(blank_bubble_path)}: {str(e)}{RESET}")
                print(f"{RED}Error details: {traceback.format_exc()}{RESET}")
                WARNING_COUNT += 1
                continue
        
        # Remove the overlay compositing
        # img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        # Save the image
        img.save(output_path)
        print(f"{GREEN}Created: {get_relative_path(output_path)} with {bubbles_processed}/{len(bubbles)} bubbles{RESET}")
        print(LOG_SEPARATOR)
        
        if bubbles_processed > 0:
            SUCCESS_COUNT += 1
        else:
            FAILURE_COUNT += 1
            
        return bubbles_processed > 0
        
    except Exception as e:
        import traceback
        print(f"{RED}Error processing {get_relative_path(blank_bubble_path)}: {str(e)}{RESET}")
        print(f"{RED}Error details: {traceback.format_exc()}{RESET}")
        print(LOG_SEPARATOR)
        FAILURE_COUNT += 1
        return False


def main():
    global SUCCESS_COUNT, FAILURE_COUNT, WARNING_COUNT
    
    parser = argparse.ArgumentParser(description="Insert translated text into blank speech bubbles")
    parser.add_argument("language", help="Language code (e.g., EN-US, LV)")
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help="Font size to use")
    parser.add_argument("-c", "--check", action="store_true", help="Check if each blank bubble has its final bounds file without inserting text")
    
    args = parser.parse_args()
    
    # Reset counters
    SUCCESS_COUNT = 0
    FAILURE_COUNT = 0
    WARNING_COUNT = 0
    
    if args.check:
        # Only check if final bounds files exist for each blank bubble
        check_final_bounds_files()
    else:
        # Process bubbles for the specified language with the specified font size
        process_bubbles(args.language, args.font_size)
    
    # Print summary statistics
    print(LOG_SEPARATOR)
    print(f"{GREEN}✓ Successful: {SUCCESS_COUNT} images{RESET}")
    print(f"{RED}✗ Failed: {FAILURE_COUNT} images{RESET}")
    print(f"{YELLOW}⚠ Warnings: {WARNING_COUNT} issues{RESET}")
    print(LOG_SEPARATOR)
    
    if args.check:
        print(f"Check complete for final bounds files")
    else:
        print(f"Processing complete for language: {args.language}")


def check_final_bounds_files():
    """
    Check if each blank bubble has its corresponding final bounds file.
    """
    global SUCCESS_COUNT, FAILURE_COUNT, WARNING_COUNT
    
    print(f"{MAGENTA}Checking for final bounds files...{RESET}")
    print(LOG_SEPARATOR)
    
    # Walk through the blank bubbles directory
    for root, dirs, files in os.walk(BLANK_BUBBLES_DIR):
        # Get the spread number from the directory name
        spread_dir = os.path.basename(root)
        
        try:
            spread_num = int(spread_dir)
            
            # Process each blank bubble file
            for file in files:
                if file.endswith("_blank.webp") and file.startswith("spch-"):
                    # Extract bubble number
                    match = re.match(r"spch-(\d+)_blank\.webp", file)
                    if match:
                        bubble_num = int(match.group(1))
                        
                        # Construct the filename for final bounds
                        final_bounds_filename = f"spread-{spread_num}-{bubble_num}_final.json"
                        final_bounds_path = os.path.join(FINAL_BOUNDS_DIR, final_bounds_filename)
                        
                        blank_bubble_path = os.path.join(root, file)
                        print(f"Checking: {get_relative_path(blank_bubble_path)}")
                        
                        # Check if the final bounds file exists
                        if os.path.exists(final_bounds_path):
                            # Check if the file has valid bubble data
                            try:
                                with open(final_bounds_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    bubbles = data.get('bubbles', [])
                                    
                                    if bubbles:
                                        print(f"{GREEN}✓ Found final bounds file with {len(bubbles)} bubbles: {get_relative_path(final_bounds_path)}{RESET}")
                                        SUCCESS_COUNT += 1
                                    else:
                                        print(f"{YELLOW}⚠ Final bounds file exists but contains no bubbles: {get_relative_path(final_bounds_path)}{RESET}")
                                        WARNING_COUNT += 1
                            except Exception as e:
                                print(f"{RED}✗ Error reading final bounds file: {get_relative_path(final_bounds_path)}: {str(e)}{RESET}")
                                FAILURE_COUNT += 1
                        else:
                            print(f"{RED}✗ Missing final bounds file: {final_bounds_filename}{RESET}")
                            FAILURE_COUNT += 1
                        
                        print(LOG_SEPARATOR)
        except ValueError:
            # Skip directories that are not numbers
            continue


def get_translated_text(spread_num, bubble_num, bubble_index, language, translated_dir):
    """
    Get the translated text for a specific speech bubble.
    
    Args:
        spread_num (int): Spread number
        bubble_num (int): Bubble number
        bubble_index (int): Index of the bubble within the JSON file (0-based)
        language (str): Language code
        translated_dir (str): Directory containing translated JSON files
        
    Returns:
        str: Translated text or None if not found
    """
    # Construct the filename
    filename = f"spread-{spread_num}-{bubble_num}_final.json"
    
    # Path to the translated file
    translated_file = os.path.join(translated_dir, language, filename)
    
    # Check if the file exists
    if os.path.exists(translated_file):
        try:
            with open(translated_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Get the bubbles from the JSON
                bubbles = data.get('bubbles', [])
                
                # If bubble_index is within range, use that specific bubble
                if 0 <= bubble_index < len(bubbles):
                    bubble_text = bubbles[bubble_index].get('text', '')
                    if bubble_text:
                        return bubble_text
                    else:
                        print(f"{RED}Error: No 'text' property found for spread-{spread_num}-{bubble_num}, bubble {bubble_index+1}{RESET}")
                        return None
                
                # If no matching bubble found but there are bubbles, use the first one
                if bubbles and bubble_index >= len(bubbles):
                    print(f"{YELLOW}Warning: Bubble index {bubble_index} out of range for spread-{spread_num}-{bubble_num}, using first bubble{RESET}")
                    bubble_text = bubbles[0].get('text', '')
                    if bubble_text:
                        return bubble_text
                    else:
                        print(f"{RED}Error: No 'text' property found in first bubble for spread-{spread_num}-{bubble_num}{RESET}")
                        return None
        except Exception as e:
            print(f"{RED}Error reading translated file {get_relative_path(translated_file)}: {str(e)}{RESET}")
    
    # Fall back to the old text file method
    old_filename = f"spread-{spread_num}-{bubble_num}.txt"
    old_translated_file = os.path.join(translated_dir, language, old_filename)
    
    if os.path.exists(old_translated_file):
        with open(old_translated_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    return None


if __name__ == "__main__":
    main()