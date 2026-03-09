"""
Exploratory Data Analysis Module
==================================================================================================================================
Examines extracted text from PDF pages to understand page types, identify patterns for classification, and discover entity formats 
for application numbers and applicant names.
"""

import re
import sys
import os

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_extraction import extract_text_from_pdf


def analyse_text_statistics(results):
    """
    Analyse basic text statistics for each page.

    Examines character count, word count, line count, and average
    word length to understand the volume and structure of extracted text.

    Args:
        results (dict): The extraction results from extract_text_from_pdf.
    """
    print("\n" + "=" * 60)
    print("1. TEXT STATISTICS")
    print("=" * 60)

    for page_num, data in results.items():
        text = data['text']
        words = text.split()
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        print(f"\nPage {page_num}:")
        print(f"  Extraction method: {data['method']}")
        print(f"  Total characters: {len(text)}")
        print(f"  Total words: {len(words)}")
        print(f"  Total lines: {len(lines)}")
        print(f"  Non-empty lines: {len(non_empty_lines)}")

        if len(words) > 0:
            avg_word_length = sum(len(w) for w in words) / len(words)
            print(f"  Average word length: {avg_word_length:.1f} characters")


def analyse_keywords(results):
    """
    Search for key phrases that indicate page type and content.

    Looks for planning-specific terminology that helps distinguish
    between different document types (decision notices, registers,
    approval of details, etc.).

    Args:
        results (dict): The extraction results from extract_text_from_pdf.
    """
    print("\n" + "=" * 60)
    print("2. KEYWORD ANALYSIS")
    print("=" * 60)

    # Keywords that help identify page types
    classification_keywords = [
        "planning permission",
        "notice of approval",
        "approval of details",
        "conditional planning permission",
        "grant of conditional",
        "planning charges",
        "conditions imposed",
        "part i",
        "part ii",
        "particulars of application",
        "particulars of decision",
        "appeal",
        "building regulations",
        "schedule",
        "permission has been granted",
        "approval has been granted",
    ]

    # Keywords that help locate entities
    entity_keywords = [
        "applicant",
        "application number",
        "application no",
        "approval granted to",
        "agent",
        "date of application",
        "reference number",
        "proposal",
        "location",
    ]

    for page_num, data in results.items():
        text_lower = data['text'].lower()

        print(f"\nPage {page_num}:")

        print("  Classification keywords found:")
        found_any = False
        for keyword in classification_keywords:
            if keyword in text_lower:
                print(f"    - '{keyword}'")
                found_any = True
        if not found_any:
            print("    - None found")

        print("  Entity keywords found:")
        found_any = False
        for keyword in entity_keywords:
            if keyword in text_lower:
                print(f"    - '{keyword}'")
                found_any = True
        if not found_any:
            print("    - None found")


def analyse_application_numbers(results):
    """
    Search for application number patterns across all pages.

    Tests multiple regex patterns to identify the formats used
    for planning application references in this document.

    Args:
        results (dict): The extraction results from extract_text_from_pdf.
    """
    print("\n" + "=" * 60)
    print("3. APPLICATION NUMBER PATTERNS")
    print("=" * 60)

    # Patterns that match common planning application number formats
    patterns = {
        "Letter/YY/NNNN (e.g. P/00/0759)": r'[A-Z]/\d{2}/\d{3,5}',
        "NN/YY/NNNN (e.g. 02/80/1609)": r'\d{2}/\d{2}/\d{3,5}',
        "Letters/YYYY/Letters/N (e.g. JK/2000/FS/1)": r'[A-Z]{2}/\d{4}/[A-Z]{1,3}/\d+',
    }

    for page_num, data in results.items():
        text = data['text']

        print(f"\nPage {page_num}:")
        found_any = False

        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                found_any = True
                for match in matches:
                    print(f"  Pattern '{pattern_name}': {match}")

        if not found_any:
            print("  No application numbers found.")


def analyse_name_contexts(results):
    """
    Identify text surrounding applicant-related keywords.

    Extracts the context around keywords like 'Applicant' and
    'approval granted to' to understand how names appear in
    the text and what patterns can be used to extract them.

    Args:
        results (dict): The extraction results from extract_text_from_pdf.
    """
    print("\n" + "=" * 60)
    print("4. NAME CONTEXT ANALYSIS")
    print("=" * 60)

    # Keywords that typically appear near applicant names
    name_indicators = [
        "applicant",
        "approval granted to",
        "approved to",
        "granted to",
        "application by",
    ]

    for page_num, data in results.items():
        text = data['text']
        text_lower = text.lower()

        print(f"\nPage {page_num}:")
        found_any = False

        for indicator in name_indicators:
            # Find the position of the keyword in lowercase text
            pos = text_lower.find(indicator)

            if pos != -1:
                found_any = True
                # Extract surrounding context (100 characters after the keyword)
                start = max(0, pos)
                end = min(len(text), pos + len(indicator) + 150)
                context = text[start:end]

                # Clean up the context for display
                context = context.replace('\n', ' ')
                print(f"  Keyword: '{indicator}'")
                print(f"  Context: \"{context}\"")
                print()

        if not found_any:
            print("  No name-related keywords found on this page.")


def summarise_page_types(results):
    """
    Provide a human-readable summary of likely page types
    based on the extracted text content.

    Args:
        results (dict): The extraction results from extract_text_from_pdf.
    """
    print("\n" + "=" * 60)
    print("5. PAGE TYPE SUMMARY")
    print("=" * 60)

    for page_num, data in results.items():
        text_lower = data['text'].lower()

        print(f"\nPage {page_num}:")
        print(f"  Characters: {data['char_count']}")

        # Determine likely page type based on keyword presence
        if "planning charges" in text_lower:
            print("  Likely type: PLANNING CHARGES REGISTER")
            print("  Reasoning: Contains 'planning charges'")

        elif "notice of approval" in text_lower and "of details" in text_lower:
            print("  Likely type: DECISION NOTICE - APPROVAL OF DETAILS")
            print("  Reasoning: Contains 'notice of approval' and 'of details'")

        elif "grant of conditional" in text_lower or "conditional planning permission" in text_lower:
            print("  Likely type: DECISION NOTICE - CONDITIONAL PERMISSION")
            print("  Reasoning: Contains 'conditional planning permission' language")

        elif "planning permission" in text_lower and ("notice of approval" in text_lower or "permission has been granted" in text_lower):
            print("  Likely type: DECISION NOTICE - PLANNING PERMISSION")
            print("  Reasoning: Contains 'planning permission' with approval language")


        else:
            print("  Likely type: UNKNOWN / REQUIRES FURTHER ANALYSIS")
            print("  Reasoning: No strong keyword matches found")


if __name__ == "__main__":
    # Define the path to the PDF file
    pdf_path = "data/anonymised-1.pdf"

    # Run text extraction
    print("Running text extraction...")
    results = extract_text_from_pdf(pdf_path)

    # Run all EDA analyses
    analyse_text_statistics(results)
    analyse_keywords(results)
    analyse_application_numbers(results)
    analyse_name_contexts(results)
    summarise_page_types(results)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)                                        