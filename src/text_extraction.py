"""
Text Extraction Module
===============================================================================================================
Extracts text from PDF pages using direct extraction (PyMuPDF) with OCR fallback (Tesseract) for scanned pages.
"""

import fitz  
from PIL import Image
import pytesseract
import io
import re


def extract_text_direct(page):
    """
    Extract text directly from a PDF page using PyMuPDF.

    Args:
        page: A PyMuPDF page object.

    Returns:
        str: The extracted text from the page.
    """
    return page.get_text()


def extract_text_ocr(page, zoom=5):
    """
    Extract text from a PDF page using Tesseract OCR.

    Converts the page to a high-resolution image, then applies OCR
    to recognise and extract text from the image.

    Args:
        page: A PyMuPDF page object.
        zoom (int): Zoom factor for image resolution. Higher values
                    improve OCR accuracy but increase processing time.
                    Default is 2 (approximately 144 DPI).

    Returns:
        str: The OCR-extracted text from the page.
    """
    # Create a transformation matrix to scale the page image
    matrix = fitz.Matrix(zoom, zoom)

    # Render the page as a pixmap (image) at the specified resolution
    pixmap = page.get_pixmap(matrix=matrix)

    # Convert the pixmap to PNG image bytes
    image_bytes = pixmap.tobytes("png")

    # Open the image bytes as a PIL Image object
    image = Image.open(io.BytesIO(image_bytes))

    # Run Tesseract OCR on the image and return the extracted text
    text = pytesseract.image_to_string(image)

    return text


def is_text_quality_acceptable(text, min_chars=50, min_words=10):
    """
    Assess whether extracted text meets minimum quality thresholds.

    This determines if direct text extraction produced meaningful
    results or if OCR fallback is needed.

    Args:
        text (str): The extracted text to evaluate.
        min_chars (int): Minimum number of characters required.
        min_words (int): Minimum number of words required.

    Returns:
        bool: True if text quality is acceptable, False otherwise.
    """
    # Check if text meets minimum character count
    if len(text.strip()) < min_chars:
        return False

    # Count the number of words (sequences of alphanumeric characters)
    words = re.findall(r'[a-zA-Z]{2,}', text)

    # Check if text meets minimum word count
    if len(words) < min_words:
        return False

    return True



def clean_text(text):
    """
    Clean and normalise extracted text.

    Removes excessive whitespace, stray non-printable characters,
    and normalises line breaks for consistent downstream processing.

    Args:
        text (str): The raw extracted text.

    Returns:
        str: The cleaned and normalised text.
    """
    # Remove non-printable characters except common whitespace
    text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)

    # Replace three or more consecutive newlines with two newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading and trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Strip leading and trailing whitespace from the entire text
    text = text.strip()

    return text



def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF document.

    For each page, attempts direct text extraction first. If the
    quality is insufficient, falls back to OCR. Returns a dictionary
    mapping page numbers to their extracted text and metadata.

    Args:
        pdf_path (str): File path to the PDF document.

    Returns:
        dict: A dictionary where each key is a page number (1-indexed)
              and each value is a dictionary containing:
              - 'text': the cleaned extracted text
              - 'method': extraction method used ('direct' or 'ocr')
              - 'char_count': number of characters in the cleaned text
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"PDF loaded: {total_pages} pages found.")

    # Dictionary to store results for each page
    results = {}

    # Process each page
    for page_num in range(total_pages):
        # PyMuPDF uses 0-based indexing, but we display 1-based for readability
        display_num = page_num + 1
        page = doc[page_num]

        print(f"\nProcessing page {display_num} of {total_pages}...")

        # Step 1: Attempt direct text extraction
        text = extract_text_direct(page)

        # Step 2: Check if the extracted text is good enough
        if is_text_quality_acceptable(text):
            method = "direct"
            print(f"  Page {display_num}: Direct extraction successful.")
        else:
            # Step 3: Fall back to OCR
            print(f"  Page {display_num}: Direct extraction insufficient. Applying OCR...")
            text = extract_text_ocr(page)
            method = "ocr"
            print(f"  Page {display_num}: OCR extraction complete.")

        # Step 4: Clean the extracted text
        cleaned_text = clean_text(text)

        # Store the results
        results[display_num] = {
            'text': cleaned_text,
            'method': method,
            'char_count': len(cleaned_text)
        }

        print(f"  Page {display_num}: {len(cleaned_text)} characters extracted via {method}.")

    # Close the PDF document
    doc.close()
    print(f"\nExtraction complete. {total_pages} pages processed.")

    return results



if __name__ == "__main__":
    # Define the path to the PDF file
    pdf_path = "data/anonymised-1.pdf"

    # Run the extraction pipeline
    results = extract_text_from_pdf(pdf_path)

    # Print a summary of results
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    for page_num, data in results.items():
        print(f"\nPage {page_num}:")
        print(f"  Method: {data['method']}")
        print(f"  Characters: {data['char_count']}")
        print(f"  Preview: {data['text'][:200]}...")
        print("-" * 40)