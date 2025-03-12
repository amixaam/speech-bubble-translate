# Speech Bubble Translate

These scripts takes in a regular .webp speech bubble and:

1. Using tesseract OCR, copies the text from the initial image,
2. Copies the text and creates bounding boxes around words from the image,
3. Creates a new, blank speech bubble without the original text,
4. Refines bounding boxes to support multiple speech bubbles in a single image,
5. Using DeepL API, translates the text into desired language,
6. Using the refined bounds and translated text, inserts the text into the blank speech bubble,
7. And finally, returns the new speech bubble.

order how to use scripts:
1. `detect_speech_bubbles.py` (get text & bounds)
2. `visualize_bounds.py` (refine and visualize bounds)
3. `translate.py` (get translations)
4. `main.py` (get empty bubble)
5. `insert_translater_text.py` (insert translated text into empty bubble)

the translated speech bubble is saved in `./media/final/`

This project uses [uv](https://docs.astral.sh/uv/).

This script is made by [@amixaam](https://github.com/amixaam)
