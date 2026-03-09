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
    patterns = [
        r'[A-Z]/\d{2}/\d{3,5}',
        r'\d{2}/\d{2}/\d{3,5}',
        r'[A-Z]{2}/\d{4}/[A-Z]{1,3}/\d+',
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        all_matches.extend(matches)

    filtered = []
    for match in all_matches:
        if is_likely_date(match):
            continue
        filtered.append(match)

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

    Args:
        text (str): A potential application number string.

    Returns:
        bool: True if the string is likely a date, False otherwise.
    """
    date_pattern_long = r'^(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}$'
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
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            entities.append({
                'name': ent.text.strip(),
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

    applicant_keywords = [
        "applicant",
        "approval granted to",
        "approved to",
        "granted to",
        "application by",
        "submitted by",
    ]

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

    scored_entities = []

    for entity in entities:
        name = entity['name']
        entity_pos = entity['start']

        if len(name) < 3:
            continue

        context_start = max(0, entity_pos - 300)
        surrounding_text = text[context_start:entity_pos].lower()

        near_applicant_keyword = False
        for keyword in applicant_keywords:
            if keyword in surrounding_text:
                near_applicant_keyword = True
                break

        near_non_applicant = False
        for keyword in non_applicant_keywords:
            if keyword in surrounding_text:
                near_non_applicant = True
                break

        context_after = text[entity['end']:entity['end'] + 200].lower()
        for keyword in non_applicant_keywords:
            if keyword in context_after:
                near_non_applicant = True
                break

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

    Args:
        entities (list): List of entity dictionaries from
                         extract_applicant_names.

    Returns:
        list: Filtered list of entity dictionaries.
    """
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
        "loval",
        "local",
    ]

    cleaned = []

    for entity in entities:
        name_lower = entity['name'].lower().strip()

        if len(name_lower) < 3:
            continue

        real_letters = re.findall(r'[a-zA-Z]{2,}', entity['name'])
        if len(real_letters) == 0:
            continue

        name_normalised = name_lower.replace('\n', ' ')
        is_excluded = False
        for term in exclusion_terms:
            if (name_normalised == term or
                name_normalised.startswith(term + " ") or
                term in name_normalised):
                is_excluded = True
                break
        if is_excluded:
            continue

        place_terms = ["thornton", "annexe", "centre", "house"]
        if name_lower in place_terms:
            continue

        if "..." in entity['name'] or ".." in entity['name']:
            continue

        cleaned.append(entity)

    return cleaned


def deduplicate_names(entities):
    """
    Remove duplicate and near-duplicate name entities.

    Handles cases where the same applicant appears multiple times
    due to repeated mentions in the document or slight OCR
    variations of the same name.

    Args:
        entities (list): List of entity dictionaries.

    Returns:
        list: Deduplicated list of entity dictionaries.
    """
    if not entities:
        return entities

    # Sort entities so that longer (more complete) names come first
    sorted_entities = sorted(entities, key=lambda e: len(e['name']), reverse=True)

    unique = []
    seen_normalised = []

    for entity in sorted_entities:
        name = entity['name'].lower().strip()
        name = re.sub(r'[^a-z0-9\s&]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()

        if len(name) < 3:
            continue

        # Extract the key name words (ignoring titles like mr, mrs, mra)
        titles = {'mr', 'mrs', 'mra', 'ms', 'dr', 'miss', 'sir', 'virs'}
        name_words = [w for w in name.split() if w not in titles and w != '&']

        is_duplicate = False
        for seen_name, seen_words in seen_normalised:
            # Exact match
            if name == seen_name:
                is_duplicate = True
                break

            # Check if the key name words overlap significantly
            if name_words and seen_words:
                shared = set(name_words) & set(seen_words)
                if len(shared) >= len(min(name_words, seen_words, key=len)):
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique.append(entity)
            seen_normalised.append((name, name_words))

    return unique

def extract_names_by_pattern(text):
    """
    Extract applicant names using regex patterns that target
    known text structures in planning documents.

    This supplements NER by catching names that appear in
    predictable positions relative to keywords, even when
    OCR noise prevents NER from recognising them.

    Args:
        text (str): The extracted text from a single page.

    Returns:
        list: A list of dictionaries containing extracted names.
    """
    names = []

    # Strategy 1: Look for lines containing "Applicant" keyword
    # then capture the name from the following line(s).
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        is_applicant_line = False
        applicant_fragments = [
            "applicant",
            "applivcarnt",
            "npl icant",
            "ppl icant",
        ]

        for fragment in applicant_fragments:
            if fragment in line_lower:
                is_applicant_line = True
                break

        if "application" in line_lower:
            is_applicant_line = False

        if is_applicant_line:
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()

                if not candidate:
                    continue

                skip_terms = [
                    "agent", "part ", "date of", "particulars",
                    "application number", "proposal", "location",
                    "town and country", "planning act",
                    "order", "procedure", "as named",
                    "subject to", "compliance", "condition",
                    "approval", "granted", "permission",
                    "pursuance", "referred", "development",
                    "signature", "registrar", "received",
                    "council", "borough", "reverse",
                ]
                should_skip = False
                for term in skip_terms:
                    if term in candidate.lower():
                        should_skip = True
                        break
                if should_skip:
                    break

                real_words = re.findall(r'[a-zA-Z]{2,}', candidate)
                if len(real_words) < 1:
                    continue

                if len(candidate) > 80:
                    continue

                cleaned_name = candidate.rstrip(',').strip()
                cleaned_name = re.sub(r'^Virs\b', 'Mrs', cleaned_name)
                cleaned_name = re.sub(r'^Mra\.?\b', 'Mrs', cleaned_name)

                names.append({
                    'name': cleaned_name,
                    'type': 'PERSON/ORG',
                    'context': 'Regex pattern match after Applicant label'
                })
                break

    # Strategy 2: Capture names after "approval granted to"
    # This is outside the Strategy 1 loop
    granted_pattern = r'approval granted to\s+(.+)'
    for match in re.finditer(granted_pattern, text, re.IGNORECASE):
        captured = match.group(1).strip()

        stop_words = ['dated', 'under the', 'pursuant', '\n',
                      'council', 'office']
        for stop_word in stop_words:
            pos = captured.lower().find(stop_word)
            if pos != -1:
                captured = captured[:pos].strip()

        captured = captured.rstrip('.,;"\' ')
        captured = re.sub(r'\s*"".*$', '', captured)
        captured = re.sub(r'\s*".*$', '', captured)

        if len(captured) < 4 or len(captured) > 60:
            continue

        skip_terms = ["provision", "condition", "planning", "council",
                      "borough", "section", "act ", "regulation",
                      "signature", "registrar", "date"]
        should_skip = False
        for term in skip_terms:
            if term in captured.lower():
                should_skip = True
                break
        if should_skip:
            continue

        names.append({
            'name': captured,
            'type': 'PERSON/ORG',
            'context': 'Regex pattern match after granted to'
        })

    # Remove duplicates
    seen = set()
    unique = []
    for name in names:
        if name['name'] not in seen:
            seen.add(name['name'])
            unique.append(name)

    return unique


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
    nlp = load_ner_model()

    print(f"\nExtracting entities from {len(results)} pages...\n")

    for page_num, data in results.items():
        text = data['text']

        print(f"Processing page {page_num}...")

        app_numbers = extract_application_numbers(text)
        data['application_numbers'] = app_numbers

        if app_numbers:
            print(f"  Application numbers: {app_numbers}")
        else:
            print(f"  Application numbers: None found")

        # Extract applicant names using NER
        name_entities = extract_applicant_names(text, nlp)
        name_entities = clean_extracted_names(name_entities)

        # Supplement with regex-based pattern extraction
        pattern_names = extract_names_by_pattern(text)

        # Merge pattern results with NER results, avoiding duplicates
        existing_names = {e['name'].lower() for e in name_entities}
        for pname in pattern_names:
            if pname['name'].lower() not in existing_names:
                name_entities.append(pname)
                existing_names.add(pname['name'].lower())

        # Deduplicate near-duplicate names
        name_entities = deduplicate_names(name_entities)

        data['name_entities'] = name_entities

        # Separate likely applicants from other entities
        likely_applicants = [
            e for e in name_entities
            if e['context'] in [
                "Near applicant keyword",
                "Regex pattern match after Applicant label",
                "Regex pattern match after granted to",
            ]
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

    pdf_path = "data/anonymised-1.pdf"

    print("Running text extraction...")
    results = extract_text_from_pdf(pdf_path)

    results = extract_entities_all_pages(results)

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