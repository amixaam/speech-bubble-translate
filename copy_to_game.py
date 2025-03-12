#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path

# ANSI color codes for better logging
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# Log separator
LOG_SEPARATOR = f"{YELLOW}{'—' * 50}{RESET}"

def copy_translated_bubbles(language, source_dir=None, target_dir=None, overwrite=False):
    """
    Copy translated speech bubbles from the translation project to the game assets directory.
    
    Args:
        language (str): Language code (e.g., 'En-US', 'Es-ES')
        source_dir (str): Source directory containing translated speech bubbles
        target_dir (str): Target directory in the game assets
        overwrite (bool): Whether to overwrite existing files
    """
    # Set default directories if not specified
    if source_dir is None:
        source_dir = f"/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/final/{language}"
    
    if target_dir is None:
        target_dir = "/Users/robertsbrinkis/Documents/GitHub/gamebook-1/assets/pirate"
    
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        print(f"{RED}Error: Source directory '{source_dir}' does not exist.{RESET}")
        return False
    
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        print(f"{RED}Error: Target directory '{target_dir}' does not exist.{RESET}")
        return False
    
    print(f"{MAGENTA}Copying translated speech bubbles for language: {language}{RESET}")
    print(f"{BLUE}Source: {source_dir}{RESET}")
    print(f"{BLUE}Target: {target_dir}{RESET}")
    print(LOG_SEPARATOR)
    
    # Counters for statistics
    files_copied = 0
    files_skipped = 0
    errors = 0
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Get the relative path from the source directory
        rel_path = os.path.relpath(root, source_dir)
        
        # Skip the root directory itself
        if rel_path == '.':
            continue
        
        # Create the corresponding target directory
        target_subdir = os.path.join(target_dir, rel_path)
        os.makedirs(target_subdir, exist_ok=True)
        
        # Process each file in the current directory
        for file in files:
            # Only process webp files
            if file.endswith('.webp'):
                source_file = os.path.join(root, file)
                
                # Extract the base name (e.g., 'spch-0-En-US.webp' -> 'spch-0')
                base_name = file.split('-')[0] + '-' + file.split('-')[1].split('.')[0]
                
                # Create the target filename with language suffix
                target_file = os.path.join(target_subdir, f"{base_name}-{language}.webp")
                
                # Check if target file already exists
                if os.path.exists(target_file) and not overwrite:
                    print(f"{YELLOW}Skipping: {target_file} (already exists){RESET}")
                    files_skipped += 1
                    continue
                
                try:
                    # Copy the file
                    shutil.copy2(source_file, target_file)
                    print(f"{GREEN}Copied: {source_file} -> {target_file}{RESET}")
                    files_copied += 1
                except Exception as e:
                    print(f"{RED}Error copying {source_file}: {str(e)}{RESET}")
                    errors += 1
    
    print(LOG_SEPARATOR)
    print(f"{MAGENTA}Copy operation complete:{RESET}")
    print(f"{GREEN}✓ Files copied: {files_copied}{RESET}")
    print(f"{YELLOW}⚠ Files skipped: {files_skipped}{RESET}")
    print(f"{RED}✗ Errors: {errors}{RESET}")
    
    return files_copied > 0

def main():
    parser = argparse.ArgumentParser(description="Copy translated speech bubbles to the game assets directory")
    parser.add_argument("language", help="Language code (e.g., 'En-US', 'Es-ES')")
    parser.add_argument("--source", "-s", help="Source directory containing translated speech bubbles")
    parser.add_argument("--target", "-t", help="Target directory in the game assets")
    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    # If source is not specified, use the default based on language
    source_dir = args.source
    if source_dir is None:
        source_dir = f"/Users/robertsbrinkis/Documents/GitHub/speech-bubble-translate/media/final/{args.language}"
    
    # Copy the translated bubbles
    copy_translated_bubbles(args.language, source_dir, args.target, args.overwrite)

if __name__ == "__main__":
    main()