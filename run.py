"""
Planning Document Classifier Pipeline
=======================================
Main entry point for the document classification and entity
extraction pipeline. Processes a PDF of planning decision notices,
classifies each page by document type, and extracts application
numbers and applicant names.

Usage:
    python run.py

Input:
    Place the PDF file in the data/ folder.

Output:
    Results are saved to the outputs/ folder as CSV.
"""

import sys
import os
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.text_extraction import extract_text_from_pdf
from src.classifier import classify_all_pages
from src.entity_extraction import extract_entities_all_pages



def save_extracted_text(results, output_dir="outputs"):
    """
    Save the extracted text for each page to a text file.

    Creates a single file containing all pages' extracted text,
    providing transparency into what the pipeline worked with.

    Args:
        results (dict): The pipeline results dictionary.
        output_dir (str): Directory to save the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "extracted_text.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("EXTRACTED TEXT FROM PDF PAGES\n")
        f.write("=" * 60 + "\n\n")

        for page_num, data in results.items():
            f.write(f"PAGE {page_num}\n")
            f.write(f"Extraction method: {data['method']}\n")
            f.write(f"Characters: {data['char_count']}\n")
            f.write("-" * 40 + "\n")
            f.write(data['text'])
            f.write("\n\n" + "=" * 60 + "\n\n")

    print(f"Extracted text saved to: {filepath}")



def save_results_csv(results, output_dir="outputs"):
    """
    Save the pipeline results as a structured CSV file.

    Creates a table with one row per page showing classification,
    confidence score, application numbers, and applicant names.

    Args:
        results (dict): The pipeline results dictionary.
        output_dir (str): Directory to save the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "classification_results.csv")

    rows = []

    for page_num, data in results.items():
        # Format application numbers as a semicolon-separated string
        app_numbers = "; ".join(data.get('application_numbers', []))

        # Format likely applicant names
        likely_applicants = data.get('likely_applicants', [])
        applicant_names = "; ".join(
            [e['name'] for e in likely_applicants]
        )

        # If no likely applicants, check other names
        if not applicant_names:
            other_names = data.get('other_names', [])
            if other_names:
                applicant_names = "; ".join(
                    [e['name'] + " (unconfirmed)" for e in other_names]
                )

        # If still no names found
        if not applicant_names:
            applicant_names = "None identified"

        rows.append({
            'Page': page_num,
            'Classification': data.get('classification', 'Not classified'),
            'Confidence': data.get('confidence', 0.0),
            'Application Numbers': app_numbers if app_numbers else "None found",
            'Applicant Names': applicant_names
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

    print(f"Results CSV saved to: {filepath}")

    return df    



def print_summary(results, df):
    """
    Print a clear, formatted summary of the pipeline results
    to the terminal.

    Args:
        results (dict): The pipeline results dictionary.
        df (pd.DataFrame): The results DataFrame.
    """
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 60)

    for page_num, data in results.items():
        print(f"\nPage {page_num}:")
        print(f"  Classification:      {data.get('classification', 'N/A')}")
        print(f"  Confidence:          {data.get('confidence', 0.0):.4f}")

        app_nums = data.get('application_numbers', [])
        if app_nums:
            print(f"  Application Numbers: {', '.join(app_nums)}")
        else:
            print(f"  Application Numbers: None found")

        likely = data.get('likely_applicants', [])
        others = data.get('other_names', [])

        if likely:
            names = [e['name'] for e in likely]
            print(f"  Applicant Names:     {', '.join(names)}")
        elif others:
            names = [e['name'] + " (unconfirmed)" for e in others]
            print(f"  Applicant Names:     {', '.join(names)}")
        else:
            print(f"  Applicant Names:     None identified")

        print("-" * 40)

    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(df.to_string(index=False))



def main():
    """
    Run the full document classification and entity extraction pipeline.

    Sequence:
        1. Extract text from PDF pages
        2. Classify pages by document type
        3. Extract application numbers and applicant names
        4. Save results and print summary
    """
    print("=" * 60)
    print("PLANNING DOCUMENT CLASSIFIER PIPELINE")
    print("=" * 60)

    # Define the input PDF path
    pdf_path = "data/anonymised-1.pdf"

    # Check that the input file exists
    if not os.path.exists(pdf_path):
        print(f"\nError: PDF file not found at '{pdf_path}'")
        print("Please place your PDF file in the 'data/' folder.")
        sys.exit(1)

    # Step 1: Text Extraction
    print("\n[Step 1/3] Extracting text from PDF pages...")
    print("-" * 40)
    results = extract_text_from_pdf(pdf_path)

    # Step 2: Page Classification
    print("\n[Step 2/3] Classifying pages...")
    print("-" * 40)
    results = classify_all_pages(results)

    # Step 3: Entity Extraction
    print("\n[Step 3/3] Extracting entities...")
    print("-" * 40)
    results = extract_entities_all_pages(results)

    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    save_extracted_text(results)
    df = save_results_csv(results)

    # Print summary
    print_summary(results, df)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: outputs/classification_results.csv")
    print(f"Extracted text saved to: outputs/extracted_text.txt")
    print("=" * 60)



if __name__ == "__main__":
    main()