"""
Entity Extraction Module
=========================
Extracts application numbers and applicant names from PDF page text.
Uses regex pattern matching for application numbers and spaCy NER
for applicant name recognition, with contextual filtering to
distinguish applicants from other named individuals.
"""

import re
import spacy


def load_ner_model():
    """
    Load the spaCy English language model for Named Entity Recognition.

    Uses en_core_web_trf (transformer-based) for best accuracy.
    Falls back to en_core_web_sm if the transformer model is not
    available.

    Returns:
        spacy.Language: A loaded spaCy language model.
    """
    print("Loading NER model...")

    try:
        nlp = spacy.load("en_core_web_trf")
        print("Loaded transformer-based model (en_core_web_trf).")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("Loaded small model (en_core_web_sm) as fallback.")
        except OSError:
            raise RuntimeError(
                "No spaCy English model found. "
                "Install one with: python -m spacy download en_core_web_trf"
            )

    return nlp



def extract_application_numbers(text):
    """
    Extract planning application numbers from text using regex patterns.

    Identifies common application reference formats found in planning
    documents while filtering out dates that match similar patterns.

    Args:
        text (str): The extracted text from a single page.

    Returns:
        list: A list of unique application numbers found.
    """
    # Define patterns for known application number formats
    patterns = [
        # Format: Letter/YY/NNNN (e.g. P/00/0759, P/98/0964)
        r'[A-Z]/\d{2}/\d{3,5}',

        # Format: NN/YY/NNNN (e.g. 02/80/1609, 02/81/1237)
        r'\d{2}/\d{2}/\d{3,5}',

        # Format: Letters/YYYY/Letters/N (e.g. JK/2000/FS/1)
        r'[A-Z]{2}/\d{4}/[A-Z]{1,3}/\d+',
    ]

    all_matches = []

    for pattern in patterns:
        matches = re.findall(pattern, text)
        all_matches.extend(matches)

    # Filter out dates (DD/MM/YYYY or DD/MM/YY format)
    filtered = []
    for match in all_matches:
        if is_likely_date(match):
            continue
        filtered.append(match)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for number in filtered:
        if number not in seen:
            seen.add(number)
            unique.append(number)

    return unique



def is_likely_date(text):
    """
    Check if a string matching an application number pattern is
    actually a date.

    Dates in DD/MM/YYYY or DD/MM/YY format can match the NN/YY/NNNN
    application number pattern. This function identifies and filters
    them out.

    Args:
        text (str): A potential application number string.

    Returns:
        bool: True if the string is likely a date, False otherwise.
    """
    # Pattern: DD/MM/YYYY where DD is 01-31 and MM is 01-12
    date_pattern_long = r'^(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}$'

    # Pattern: DD/MM/YY where DD is 01-31 and MM is 01-12
    date_pattern_short = r'^(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{2}$'

    if re.match(date_pattern_long, text) or re.match(date_pattern_short, text):
        return True

    return False



def extract_applicant_names(text, nlp):
    """
    Extract applicant names from text using spaCy NER with
    contextual filtering.

    Runs Named Entity Recognition to find PERSON and ORG entities,
    then applies contextual rules to identify which names are likely
    applicants rather than officers, signatories, or other individuals.

    Args:
        text (str): The extracted text from a single page.
        nlp: A loaded spaCy language model.

    Returns:
        list: A list of dictionaries, each containing:
              - 'name': the extracted name string
              - 'type': the entity type (PERSON or ORG)
              - 'context': how the name was identified
    """
    # Run NER on the text
    doc = nlp(text)

    # Collect all PERSON and ORG entities
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            entities.append({
                'name': ent.text.strip(),
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

    # Define keywords that indicate an applicant name nearby
    applicant_keywords = [
        "applicant",
        "approval granted to",
        "approved to",
        "granted to",
        "application by",
        "submitted by",
    ]

    # Define keywords that indicate a non-applicant (officer, signatory)
    non_applicant_keywords = [
        "director of planning",
        "signed",
        "signature",
        "registrar",
        "planning officer",
        "borough of",
        "district council",
        "council of",
    ]

    # Score each entity based on context
    scored_entities = []

    for entity in entities:
        name = entity['name']
        entity_pos = entity['start']

        # Skip very short names (likely OCR noise)
        if len(name) < 3:
            continue

        # Check surrounding text for context (300 chars before the entity)
        context_start = max(0, entity_pos - 300)
        surrounding_text = text[context_start:entity_pos].lower()

        # Check if near an applicant keyword
        near_applicant_keyword = False
        for keyword in applicant_keywords:
            if keyword in surrounding_text:
                near_applicant_keyword = True
                break

        # Check if near a non-applicant keyword
        near_non_applicant = False
        for keyword in non_applicant_keywords:
            if keyword in surrounding_text:
                near_non_applicant = True
                break

        # Also check text immediately after the entity for non-applicant signals
        context_after = text[entity['end']:entity['end'] + 200].lower()
        for keyword in non_applicant_keywords:
            if keyword in context_after:
                near_non_applicant = True
                break

        # Determine context label
        if near_applicant_keyword and not near_non_applicant:
            context = "Near applicant keyword"
        elif near_non_applicant:
            context = "Likely non-applicant (officer/signatory)"
        else:
            context = "No strong contextual signal"

        scored_entities.append({
            'name': name,
            'type': entity['type'],
            'context': context
        })

    return scored_entities



def clean_extracted_names(entities):
    """
    Post-process extracted name entities to remove obvious
    false positives caused by OCR noise or NER errors.

    Filters out names that are clearly organisations, locations,
    or government bodies rather than applicant names.

    Args:
        entities (list): List of entity dictionaries from
                         extract_applicant_names.

    Returns:
        list: Filtered list of entity dictionaries.
    """
    # Terms that indicate a name is not an applicant
    exclusion_terms = [
        "council",
        "borough",
        "planning",
        "authority",
        "committee",
        "government",
        "parliament",
        "office",
        "offices",
        "annexe",
        "department",
        "state",
        "stat",
        "connell",
        "the local",
        "the north",
    ]

    cleaned = []

    for entity in entities:
        name_lower = entity['name'].lower().strip()

        # Skip names that are too short
        if len(name_lower) < 3:
            continue

        # Skip names that are just OCR noise (no real letters)
        real_letters = re.findall(r'[a-zA-Z]{2,}', entity['name'])
        if len(real_letters) == 0:
            continue

        # Skip names that match exclusion terms
        is_excluded = False
        for term in exclusion_terms:
            if name_lower == term or name_lower.startswith(term + " "):
                is_excluded = True
                break
        if is_excluded:
            continue

        # Skip entities that are clearly place names used as locations
        # (single-word names that are common place indicators)
        place_terms = ["thornton", "annexe", "centre", "house", "sphtey"]
        if name_lower in place_terms:
            continue
        

        # Skip entities containing ellipsis or OCR fragment markers
        if "..." in entity['name'] or ".." in entity['name']:
            continue


        cleaned.append(entity)

    return cleaned



def extract_entities_all_pages(results):
    """
    Extract application numbers and applicant names from all pages.

    Loads the NER model once and processes each page, adding entity
    extraction results to the existing results dictionary.

    Args:
        results (dict): The results dictionary from previous pipeline
                        stages (text extraction and classification).

    Returns:
        dict: The results dictionary with entity data added to
              each page entry.
    """
    # Load the NER model once
    nlp = load_ner_model()

    print(f"\nExtracting entities from {len(results)} pages...\n")

    for page_num, data in results.items():
        text = data['text']

        print(f"Processing page {page_num}...")

        # Extract application numbers
        app_numbers = extract_application_numbers(text)
        data['application_numbers'] = app_numbers

        if app_numbers:
            print(f"  Application numbers: {app_numbers}")
        else:
            print(f"  Application numbers: None found")

        # Extract applicant names
        name_entities = extract_applicant_names(text, nlp)
        name_entities = clean_extracted_names(name_entities)
        data['name_entities'] = name_entities

        # Separate likely applicants from other entities
        likely_applicants = [
            e for e in name_entities
            if e['context'] == "Near applicant keyword"
        ]
        other_names = [
            e for e in name_entities
            if e['context'] == "No strong contextual signal"
        ]
        non_applicants = [
            e for e in name_entities
            if e['context'] == "Likely non-applicant (officer/signatory)"
        ]

        data['likely_applicants'] = likely_applicants
        data['other_names'] = other_names
        data['non_applicants'] = non_applicants

        if likely_applicants:
            names = [e['name'] for e in likely_applicants]
            print(f"  Likely applicants: {names}")
        elif other_names:
            names = [e['name'] for e in other_names]
            print(f"  Possible applicants (no keyword context): {names}")
        else:
            print(f"  Applicant names: None found")

        if non_applicants:
            names = [e['name'] for e in non_applicants]
            print(f"  Non-applicant names filtered: {names}")

    print("\nEntity extraction complete.")
    return results



if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.text_extraction import extract_text_from_pdf

    # Define the path to the PDF file
    pdf_path = "data/anonymised 1.pdf"

    # Run text extraction
    print("Running text extraction...")
    results = extract_text_from_pdf(pdf_path)

    # Run entity extraction
    results = extract_entities_all_pages(results)

    # Print detailed results
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION RESULTS")
    print("=" * 60)

    for page_num, data in results.items():
        print(f"\nPage {page_num}:")
        print(f"  Application numbers: {data['application_numbers']}")

        if data['likely_applicants']:
            for entity in data['likely_applicants']:
                print(f"  Likely applicant: {entity['name']} "
                      f"({entity['type']}) - {entity['context']}")

        if data['other_names']:
            for entity in data['other_names']:
                print(f"  Other name: {entity['name']} "
                      f"({entity['type']}) - {entity['context']}")

        if data['non_applicants']:
            for entity in data['non_applicants']:
                print(f"  Non-applicant: {entity['name']} "
                      f"({entity['type']}) - {entity['context']}")

        if not data['likely_applicants'] and not data['other_names']:
            print(f"  Applicant names: None identified")

        print("-" * 40)