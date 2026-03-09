# Land Registry Document Classifier

An NLP pipeline that classifies pages from planning permission decision notice PDFs and extracts application numbers and applicant names using zero-shot transformer classification and Named Entity Recognition.

Built as part of the HMLR Data Science Challenge.

## Approach

This pipeline processes scanned PDF documents of historical planning decision notices through three stages:

1. **Text Extraction** — Extracts text from each PDF page using PyMuPDF for direct extraction, with Tesseract OCR as an automatic fallback for scanned pages. Includes automated text quality assessment to determine the appropriate extraction method per page.

2. **Page Classification** — Classifies each page into one of four document categories using zero-shot classification with the `facebook/bart-large-mnli` transformer model from Hugging Face. Categories were defined through exploratory data analysis of the document content. No training data is required.

3. **Entity Extraction** — Extracts application numbers using regex pattern matching with date filtering, and applicant names using spaCy's transformer-based NER model (`en_core_web_trf`) with contextual filtering to distinguish applicants from officers and signatories. A regex-based backup strategy supplements NER for cases where OCR noise degrades keyword recognition.

An NLP-only approach was chosen because all three tasks (classification, number extraction, name extraction) are fundamentally text-based. This provides a single coherent pipeline that addresses the essential Python and NLP criteria as well as the desirable NER criterion.

## Project Structure

```
lr-document-classifier/
│
├── data/                      # Input PDF files
│   └── anonymised 1.pdf       # Planning decision notices PDF
│
├── src/                       # Source code modules
│   ├── text_extraction.py     # PDF text extraction with OCR fallback
│   ├── eda.py                 # Exploratory data analysis
│   ├── classifier.py          # Zero-shot page classification
│   └── entity_extraction.py   # Application number and name extraction
│
├── outputs/                   # Pipeline output files
│   ├── classification_results.csv  # Results table
│   └── extracted_text.txt          # Raw extracted text per page
│
├── reports/                   # Analysis report
│   └── analysis-report.pdf    # One-page EDA, limitations, and alternatives
│
├── run.py                     # Main pipeline entry point
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Requirements

- Python 3.10 or higher
- Tesseract OCR installed on your system (see installation below)

## Installation

### 1. Install Tesseract OCR

Tesseract must be installed separately as a system dependency.

- **Windows:** Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Run the installer and add the installation path (default: `C:\Program Files\Tesseract-OCR`) to your system PATH environment variable.

Verify the installation by running:

```
tesseract --version
```

### 2. Set up the Python environment

Create and activate a virtual environment:

```
python -m venv venv
```

Windows:

```
venv\Scripts\activate
```

### 3. Install Python dependencies

```
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```
python -m spacy download en_core_web_trf
```

## Usage

Place your PDF file in the `data/` folder, then run:

```
python run.py
```

The pipeline will:

1. Extract text from each page using OCR where needed
2. Classify each page by document type
3. Extract application numbers and applicant names
4. Save results to the `outputs/` folder
5. Print a summary to the terminal

**Note:** The first run will download the Hugging Face classification model (~1.6 GB), which may take several minutes. Subsequent runs use the cached model.

## Output

The pipeline produces two output files in the `outputs/` folder:

- **classification_results.csv** — A structured table with one row per page showing the page number, classification category, confidence score, application numbers found, and applicant names found.

- **extracted_text.txt** — The full extracted text for each page, including metadata about the extraction method used. This provides transparency into the OCR output that the classification and entity extraction stages worked with.

## Limitations

- **OCR quality** is the primary constraint on extraction accuracy. Scanned documents of varying age and condition produce inconsistent text, which can degrade both NER performance and keyword-based pattern matching. OCR garbling of keywords like "Applicant" and partial corruption of names directly impacts entity extraction.
- **Zero-shot classification** produces moderate confidence scores (0.35–0.57) compared to fine-tuned models, reflecting the inherent uncertainty of inference without task-specific training data. Despite this, all page classifications were correct.
- **Small sample size** (4 pages) limits statistical validation. Additional page types may exist in larger document sets that are not represented in the current category definitions.
- **Name-location ambiguity** in OCR text can cause trailing location words to be captured alongside applicant names when no clear delimiter is present.

For a detailed discussion of EDA findings, limitations, and alternative methods, see the analysis report in `reports/analysis-report.pdf`.
