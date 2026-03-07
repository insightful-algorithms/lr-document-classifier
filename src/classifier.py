"""
Page Classification Module
====================================================================================================================================
Classifies PDF pages into document categories using zero-shot classification with a pre-trained transformer model from Hugging Face.
"""

import sys
import os

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import pipeline


def load_classifier():
    """
    Load the zero-shot classification pipeline.

    Uses the facebook/bart-large-mnli model, which is a widely used
    and reliable pre-trained model for zero-shot text classification.

    Returns:
        transformers.Pipeline: A zero-shot classification pipeline.
    """
    print("Loading zero-shot classification model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    print("Model loaded successfully.")
    return classifier


def get_category_labels():
    """
    Define the page category labels for classification.

    Categories are based on EDA findings from the planning document
    pages. Each label is a natural language description that the
    zero-shot model can match against page content.

    Returns:
        list: A list of category label strings.
    """
    labels = [
        "Planning Charges Register",
        "Decision Notice for Planning Permission",
        "Decision Notice for Conditional Planning Permission",
        "Decision Notice for Approval of Details",
    ]
    return labels


def classify_page(classifier, text, labels):
    """
    Classify a single page's text against the defined categories.

    If the text is too short or empty, returns 'Unclassifiable' rather
    than forcing a potentially unreliable prediction.

    Args:
        classifier: The zero-shot classification pipeline.
        text (str): The extracted text from a single page.
        labels (list): The list of category labels.

    Returns:
        dict: A dictionary containing:
              - 'label': the predicted category
              - 'score': the confidence score (0 to 1)
              - 'all_scores': scores for all categories
    """
    # Handle pages with very little text
    if len(text.strip()) < 30:
        return {
            'label': 'Unclassifiable',
            'score': 0.0,
            'all_scores': {label: 0.0 for label in labels}
        }

    # Truncate very long text to avoid model input limits
    # BART has a maximum input length of 1024 tokens
    max_chars = 3000
    input_text = text[:max_chars] if len(text) > max_chars else text

    # Run zero-shot classification
    result = classifier(input_text, labels)

    # Build a dictionary of all scores for transparency
    all_scores = {}
    for label, score in zip(result['labels'], result['scores']):
        all_scores[label] = round(score, 4)

    return {
        'label': result['labels'][0],
        'score': round(result['scores'][0], 4),
        'all_scores': all_scores
    }


def classify_all_pages(results):
    """
    Classify all pages from the extraction results.

    Loads the model once and applies it to each page, returning
    classification results alongside the original extraction data.

    Args:
        results (dict): The extraction results from extract_text_from_pdf.

    Returns:
        dict: The results dictionary with classification data added
              to each page entry.
    """
    # Load the model once
    classifier = load_classifier()

    # Get the category labels
    labels = get_category_labels()

    print(f"\nClassifying {len(results)} pages against {len(labels)} categories...")
    print(f"Categories: {labels}\n")

    # Classify each page
    for page_num, data in results.items():
        print(f"Classifying page {page_num}...")

        classification = classify_page(classifier, data['text'], labels)

        # Add classification results to the page data
        data['classification'] = classification['label']
        data['confidence'] = classification['score']
        data['all_scores'] = classification['all_scores']

        print(f"  Page {page_num}: {classification['label']} "
              f"(confidence: {classification['score']:.4f})")

    print("\nClassification complete.")
    return results


if __name__ == "__main__":
    from src.text_extraction import extract_text_from_pdf

    # Define the path to the PDF file
    pdf_path = "data/anonymised 1.pdf"

    # Run text extraction
    print("Running text extraction...")
    results = extract_text_from_pdf(pdf_path)

    # Run classification
    results = classify_all_pages(results)

    # Print detailed results
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    for page_num, data in results.items():
        print(f"\nPage {page_num}:")
        print(f"  Classification: {data['classification']}")
        print(f"  Confidence: {data['confidence']:.4f}")
        print(f"  All scores:")
        for label, score in data['all_scores'].items():
            # Mark the winning category
            marker = " <-- SELECTED" if label == data['classification'] else ""
            print(f"    {label}: {score:.4f}{marker}")


