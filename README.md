# Document Intelligence

An OCR and document processing pipeline — image preprocessing, text extraction,
and structured output from scanned documents and images.

## Status

🚧 Project scaffolding. Preprocessing and extraction modules are stubbed out
and will be implemented in upcoming commits.

## Planned Pipeline

raw image → preprocess → OCR → clean text → structured output
text


1. **Preprocess** — grayscale, denoise, threshold, deskew
2. **Extract** — OCR via Tesseract, output text and bounding boxes
3. **Clean** — fix OCR artifacts, normalize spacing and line breaks
4. **Structure** — parse into key-value fields, tables, or plain text

## Project Structure

Document-Intelligence/
├── src/
│ ├── init.py
│ ├── preprocess.py # image cleaning for OCR
│ └── extract.py # OCR text extraction
├── data/
│ ├── raw/ # input images (not committed)
│ └── processed/ # cleaned images (not committed)
├── notebooks/ # exploration and evaluation
├── models/ # saved model weights (not committed)
├── tests/ # unit tests
├── docs/ # design notes and references
├── requirements.txt
└── README.md
text


## Tech Stack

- Python 3.10+
- OpenCV — image preprocessing
- Tesseract / pytesseract — OCR
- NumPy, Pillow — image handling
- pandas — structured output

## Installation

```bash
git clone https://github.com/A-creator-collab/Document-Intelligence.git
cd Document-Intelligence
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

Note: You also need Tesseract installed on your system:

    Ubuntu/Debian: sudo apt install tesseract-ocr

    macOS: brew install tesseract

    Windows: installer here

Usage

Not implemented yet. The pipeline will be runnable as:
bash

python -m src.preprocess data/raw/sample.png
python -m src.extract data/processed/sample.png

Roadmap

    □

    Implement preprocessing chain
    □

    Wire up Tesseract extraction
    □

    Text cleaning and normalization
    □

    Handle multi-page PDFs
    □

    Structured output (key-value extraction)
    □

    Evaluation notebook with sample documents

Author

Atharv Singh
B.Tech CSE (AI) — NIET Greater Noida
GitHub: @A-creator-collab
