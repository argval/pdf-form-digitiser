import os
import json
import csv
import logging
import re
import email_validator
import phonenumbers
from datetime import datetime
from typing import Dict, Any, List, Optional

import PyPDF2
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import tensorflow as tf
from tqdm import tqdm
from .ml_components import (
    PDFFormMLExtractor, 
    TransferLearningFeatureExtractor, 
    enhance_extraction_with_ml
)

class PDFFormDigitizer:
    def __init__(self, 
                 input_dir: str = '../input', 
                 output_dir: str = 'output', 
                 log_level: int = logging.INFO,
                 enable_ml: bool = True):
        """
        Initialize PDF Form Digitizer with optional ML components
        
        Args:
            input_dir (str): Directory containing input PDFs
            output_dir (str): Directory to save processed outputs
            log_level (int): Logging level
            enable_ml (bool): Enable machine learning enhancements
        """
        # Resolve absolute path for input and output directories
        self.input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), input_dir))
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), output_dir))
        
        # Create directories first
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'pdf_digitizer.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # ML Components
        self.ml_enabled = enable_ml
        if self.ml_enabled:
            self.ml_extractor = PDFFormMLExtractor()
            self.feature_extractor = TransferLearningFeatureExtractor()
            
            try:
                # Train models on existing extracted data
                ml_training_results = self.ml_extractor.train_models(self.output_dir)
                self.logger.info("ML Models trained successfully")
                self.logger.info(f"Field Extraction Model Performance: {ml_training_results['field_extraction']}")
                self.logger.info(f"Text Classifier Performance: {ml_training_results['text_classifier']}")
            except Exception as ml_train_error:
                self.logger.warning(f"ML Model training failed: {ml_train_error}")
                self.ml_enabled = False

    def _advanced_extraction(self, ocr_text):
        # Enhanced text preprocessing
        def preprocess_text(text):
            # Remove extra whitespaces, normalize line breaks
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', '\n', text)
            # Remove non-printable characters
            text = ''.join(char for char in text if char.isprintable())
            return text.strip()

        # Intelligent text cleaning and filtering
        def clean_and_filter_lines(lines, section_name=None):
            # Global exclusion lists
            global_exclusions = {
                'common': [
                    'topics', 'visit', 'additional', 'comments', 
                    'in-person', 'logistics', 'schedule', 
                    'proposed', 'activities', 'to be presented',
                    'grade level', 'class size', 'date of visit',
                    'brief description', 'equipment', 'resources'
                ],
                'requestor_info': [
                    'contact', 'person', 'name', 'phone', 'email', 
                    'school', 'facility', 'address'
                ],
                'visit_info': [
                    'date', 'visit', 'virtual', 'in-person', 
                    'class', 'size', 'grade', 'level'
                ],
                'topics_activities': [
                    'stage', 'program', 'games', 'high tea', 
                    'fun', 'activities', 'resources'
                ],
                'visit_logistics': [
                    'water', 'cooler', 'computer', 'projector', 
                    'network', 'wifi', 'parking', 'access', 'notes'
                ]
            }

            # Advanced filtering
            filtered_lines = []
            for line in lines:
                # Normalize and clean line
                line = preprocess_text(line)
                
                # Skip empty or very short lines
                if len(line) < 3:
                    continue
                
                # Skip lines with global exclusions
                if any(exclusion in line.lower() for exclusion in global_exclusions['common']):
                    continue
                
                # Section-specific filtering
                if section_name:
                    if any(exclusion in line.lower() for exclusion in global_exclusions.get(section_name, [])):
                        continue
                
                # Remove generic or repetitive phrases
                if line.lower() in ['yes', 'no', 'n/a', 'free']:
                    continue
                
                filtered_lines.append(line)
            
            # Remove duplicates while preserving order
            unique_lines = []
            for line in filtered_lines:
                if line not in unique_lines:
                    unique_lines.append(line)
            
            return unique_lines

        # Advanced section extraction configurations
        section_configs = {
            'requestor_info': {
                'patterns': [
                    # Name extraction
                    {
                        'pattern': r'(?:Name|Contact\s*Person)\s*[:]*\s*([^\n:]+)',
                        'type': 'text',
                        'priority': 1,
                        'post_process': lambda x: x.split('|')[0].strip()
                    },
                    # Phone number extraction
                    {
                        'pattern': r'(?:Phone\s*(?:Number)?)\s*[:]*\s*(\+?[\d\s\-().]{10,})',
                        'type': 'phone_number',
                        'priority': 2,
                        'post_process': lambda x: re.sub(r'\D', '', x)
                    },
                    # Email extraction
                    {
                        'pattern': r'(?:Email\s*(?:Address)?)\s*[:]*\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                        'type': 'email',
                        'priority': 3
                    }
                ]
            },
            'visit_info': {
                'patterns': [
                    # Date extraction (multiple formats)
                    {
                        'pattern': r'(?:Requested|Alternate)\s*(?:Date\s*(?:of\s*)?)?Visit\s*[:]*\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
                        'type': 'date',
                        'priority': 1,
                        'post_process': lambda x: x.replace(' ', '')
                    },
                    # Visit type extraction
                    {
                        'pattern': r'(?:Virtual|In-Person)\s*[:]*\s*([^\n:]+)',
                        'type': 'text',
                        'priority': 2
                    },
                    # Class size extraction
                    {
                        'pattern': r'(?:Class\s*Size)\s*[:]*\s*(\d+)',
                        'type': 'integer',
                        'priority': 3
                    }
                ]
            },
            'schedule': {
                'patterns': [
                    {
                        'pattern': r'(?:Schedule|Proposed\s*Schedule)[:]*\n((?:.*\n)*?)(?=\n\n|\n(?:Topics|Visit|Additional))',
                        'type': 'multi_line_text',
                        'filter': lambda lines: clean_and_filter_lines(lines, 'schedule')
                    }
                ]
            },
            'topics_activities': {
                'patterns': [
                    {
                        'pattern': r'(?:Topics|Activities)(?:\s*(?:to\s*be\s*Presented)?)\s*[:]*\n((?:.*\n)*?)(?=\n\n|\n(?:In-Person|Additional|Visit))',
                        'type': 'multi_line_text',
                        'filter': lambda lines: clean_and_filter_lines(lines, 'topics_activities')
                    }
                ]
            },
            'visit_logistics': {
                'patterns': [
                    {
                        'pattern': r'(?:In-Person\s*Visit\s*Logistics|Logistics\s*Details)[:]*\n((?:.*\n)*?)(?=\n\n|\n(?:Additional|Comments))',
                        'type': 'multi_line_text',
                        'filter': lambda lines: clean_and_filter_lines(lines, 'visit_logistics')
                    }
                ]
            }
        }

        extracted_data = {}

        # Iterate through sections and extract data
        for section_name, section_config in section_configs.items():
            # Try each extractor in order of priority
            for extractor in section_config.get('patterns', []):
                matches = re.findall(extractor['pattern'], ocr_text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
                
                if matches:
                    # Handle multi-line text
                    if extractor['type'] == 'multi_line_text':
                        # Flatten matches if nested
                        if matches and isinstance(matches[0], tuple):
                            matches = [match[0] for match in matches]
                        
                        # Split and filter lines
                        lines = []
                        for match in matches:
                            match_lines = [line.strip() for line in match.split('\n') if line.strip()]
                            
                            # Apply custom filtering if specified
                            if 'filter' in extractor:
                                match_lines = extractor['filter'](match_lines)
                            
                            lines.extend(match_lines)
                        
                        if lines:
                            extracted_data[section_name] = {
                                'value': lines,
                                'type': 'multi_line_text',
                                'metadata': {
                                    'line_count': len(lines)
                                }
                            }
                            break
                    
                    # Handle single line text
                    else:
                        # Take the first match and clean it
                        match = matches[0]
                        if isinstance(match, tuple):
                            match = match[0]
                        match = preprocess_text(match)
                        
                        # Apply post-processing if specified
                        if 'post_process' in extractor:
                            match = extractor['post_process'](match)
                        
                        if match:
                            # Determine field type and metadata
                            field_type = extractor['type']
                            extracted_data[section_name] = {
                                'value': match,
                                'type': field_type,
                                'metadata': {
                                    'is_numeric': field_type in ['integer', 'float', 'phone_number', 'date'],
                                    'is_personal_info': field_type in ['email', 'phone_number']
                                }
                            }
                            break

        return extracted_data

    def process_pdf(self, pdf_path):
        """
        Enhanced PDF processing with optional ML components
        
        Args:
            pdf_path (str): Path to PDF file
        """
        # OCR processing
        ocr_text = self.ocr_pdf(pdf_path)
        
        # Advanced extraction
        extracted_data = self._advanced_extraction(ocr_text)
        
        # Optional ML Enhancements
        if self.ml_enabled:
            try:
                # Extract advanced features
                advanced_features = self.feature_extractor.extract_features([ocr_text])
                
                # Enhance extraction with ML insights
                extracted_data = enhance_extraction_with_ml(
                    extracted_data, 
                    self.ml_extractor
                )
                
                # Add transfer learning features
                extracted_data['ml_features'] = advanced_features.tolist() if advanced_features is not None else None
            
            except Exception as ml_error:
                self.logger.warning(f"ML enhancement failed: {ml_error}")
        
        # Output processing
        output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '_extracted.json'
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Write extracted data to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Successfully processed {os.path.basename(pdf_path)}")
        
        return extracted_data

    def ocr_pdf(self, pdf_path: str, output_format: str = 'txt') -> str:
        """
        Perform OCR on PDF to extract text
        
        Args:
            pdf_path (str): Path to PDF file
            output_format (str): Output format (txt or list)
        
        Returns:
            Extracted text from PDF
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            # Extract text from each page
            extracted_texts = []
            for image in images:
                text = pytesseract.image_to_string(image)
                extracted_texts.append(text)
            
            # Return full text
            return '\n'.join(extracted_texts)
        
        except Exception as e:
            self.logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return ''

    def process_pdf_batch(self, pdf_files: Optional[List[str]] = None):
        """
        Process multiple PDF files in batch
        
        Args:
            pdf_files (Optional[List[str]]): List of PDF file paths. 
                                             If None, processes all PDFs in input_dir
        """
        if pdf_files is None:
            pdf_files = [
                os.path.join(self.input_dir, f) 
                for f in os.listdir(self.input_dir) 
                if f.lower().endswith('.pdf')
            ]
        
        # Use tqdm for progress tracking
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                self.process_pdf(pdf_path)
            
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_path}: {e}")

def main():
    """
    Main function to process PDFs in the input directory
    """
    digitizer = PDFFormDigitizer()
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(digitizer.input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        digitizer.logger.warning("No PDF files found in the input directory.")
        return
    
    # Process PDFs
    digitizer.process_pdf_batch()

if __name__ == "__main__":
    main()
