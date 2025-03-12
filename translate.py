import os
import json
import deepl
from pathlib import Path

# DeepL API key (replace with your own key)
DEEPL_API_KEY = "API KEY"

# Directory containing the JSON files to translate
INPUT_DIR = "./media/bounds/final-bounds"
# Base directory for translated output
OUTPUT_BASE_DIR = "./media/translated"

# Language to translate to (e.g., "DE" for German, "FR" for French)
TARGET_LANG = "LV" # En-US, LV

# Initialize the DeepL translator
translator = deepl.Translator(DEEPL_API_KEY)


def ensure_directory_exists(directory):
    """Ensure the specified directory exists."""
    os.makedirs(directory, exist_ok=True)


def main():
    # Create output directory for the target language
    output_dir = os.path.join(OUTPUT_BASE_DIR, TARGET_LANG)
    ensure_directory_exists(output_dir)
    
    # Ensure the input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' does not exist.")
        return

    # Collect all JSON files and their bubble texts
    files_to_translate = []
    texts_to_translate = []
    bubble_indices = []  # To keep track of which bubble each text belongs to

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith("_final.json"):
            input_file = os.path.join(INPUT_DIR, filename)
            output_file = os.path.join(output_dir, filename)

            # Skip if the output file already exists
            if os.path.exists(output_file):
                print(f"Skipping {input_file} (already translated)")
                continue

            try:
                # Read the JSON file
                with open(input_file, "r", encoding="utf-8") as file:
                    data = json.load(file)
                
                # Extract text from each bubble
                if 'bubbles' in data and data['bubbles']:
                    for i, bubble in enumerate(data['bubbles']):
                        if 'text' in bubble and bubble['text']:
                            # Add the file, bubble index, and its text to the lists
                            files_to_translate.append((input_file, output_file, data))
                            texts_to_translate.append(bubble['text'])
                            bubble_indices.append(i)
            except Exception as e:
                print(f"Error reading {input_file}: {e}")

    # If no texts to translate, exit
    if not texts_to_translate:
        print("No texts to translate.")
        return

    # Batch translate all texts
    try:
        print(f"Translating {len(texts_to_translate)} bubble texts...")
        translated_results = translator.translate_text(
            texts_to_translate, source_lang="ES", target_lang=TARGET_LANG
        )

        # Group translations by file
        file_translations = {}
        for (input_file, output_file, data), translated_text, bubble_idx in zip(
            files_to_translate, translated_results, bubble_indices
        ):
            if output_file not in file_translations:
                file_translations[output_file] = (data, [])
            
            # Store the translation with its bubble index
            file_translations[output_file][1].append((bubble_idx, translated_text.text))
        
        # Write the translated results to the output files
        for output_file, (data, translations) in file_translations.items():
            # Create a deep copy of the data to avoid modifying the original
            translated_data = json.loads(json.dumps(data))
            
            # Update each bubble with its translation
            for bubble_idx, translated_text in translations:
                translated_data['bubbles'][bubble_idx]['text'] = translated_text
            
            # Write the updated JSON to the output file
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(translated_data, file, indent=4, ensure_ascii=False)
            
            print(f"Translated and saved to {output_file}")

    except Exception as e:
        print(f"Error during translation: {e}")


if __name__ == "__main__":
    main()
