# 📄 PDF Form Digitiser

## Overview

PDF Form Digitiser is a machine learning-powered tool designed to automatically extract and digitise information from PDF forms. The project uses OCR and ML techniques to parse complex form structures, including tables, checkboxes, and dropdown fields.

## Key Features

This project provides the following features:
- Automatic PDF form field extraction
- Machine learning-powered text recognition
- Metadata and text extraction
- Support for various form elements (tables, checkboxes, dropdowns)

## Technical Approach

### Key Technologies
- Python 3.12.4
- TensorFlow 2.16.1
- Keras
- PyTesseract
- PyPDF2
- pdf2image

## Steps to run it

### Prerequisites
- Python 3.12.4
- pip
- Tesseract OCR

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-form-digitizer.git
cd pdf-form-digitizer
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### Processing PDFs

1. Place PDF files in the `input/` directory

2. Run the main processing script:
```bash
python -m demo
```

3. Processed files will be saved in the `output/` directory

## Dependencies

- PyPDF2==3.0.1
- pandas==2.1.4
- pytesseract==0.3.10
- pdf2image==1.16.3
- tensorflow==2.16.1
- tf-keras==2.16.0
- scikit-learn==1.3.2
- transformers==4.36.2
- torch==2.1.2

## Methodology

1. **PDF Input**: Convert PDFs to images
2. **OCR Extraction**: Apply Tesseract for text recognition
3. **ML Processing**:
   - Train models on extracted data
   - Classify form sections
   - Generate confidence scores
4. **Output Generation**:
   - Create structured JSON and CSV
   - Include extracted text and metadata

## Limitations

- Requires diverse training data
- Performance varies across form layouts
- Computational resource intensive

## Future Improvements

- Enhance error handling
- Support more form elements
- Integrate with other tools
- Streamline workflow
