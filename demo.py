import os
import logging
import json
import numpy as np
import pandas as pd
from src.pdf_processor import PDFFormDigitizer
from src.ml_components import PDFFormMLExtractor, TransferLearningFeatureExtractor

# Custom JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def analyze_extracted_data(output_dir):
    """
    Advanced data analysis with ML insights
    
    Args:
        output_dir (str): Directory containing extracted JSON files
    """
    print("\n🔬 Advanced ML-Powered Data Analysis Report 📊")
    
    # Initialize ML components
    ml_extractor = PDFFormMLExtractor()
    feature_extractor = TransferLearningFeatureExtractor()
    
    # Find all JSON files
    json_files = [f for f in os.listdir(output_dir) if f.endswith('_extracted.json')]
    
    if not json_files:
        print("No extracted JSON files found.")
        return
    
    # Aggregate data across all processed forms
    all_forms_data = {}
    ml_insights = {
        'section_predictions': {},
        'feature_clusters': [],
        'confidence_scores': {}
    }
    
    for json_file in json_files:
        file_path = os.path.join(output_dir, json_file)
        with open(file_path, 'r') as f:
            form_data = json.load(f)
        
        print(f"\n📄 ML Analysis for {json_file}:")
        print("----------------------------")
        
        # ML Section Analysis
        section_predictions = {}
        confidence_scores = {}
        
        for section, details in form_data.items():
            # Skip ML metadata sections
            if section in ['ml_features', 'ml_metadata']:
                continue
            
            text = str(details.get('value', ''))
            
            try:
                # Predict section using ML model
                predicted_section = ml_extractor.predict_section(text)
                section_predictions[section] = predicted_section
                
                # Check ML metadata if available
                ml_metadata = details.get('ml_metadata', {})
                confidence = ml_metadata.get('confidence', 0)
                confidence_scores[section] = confidence
            except Exception as ml_error:
                print(f"ML prediction error for {section}: {ml_error}")
        
        # Transfer Learning Feature Extraction
        try:
            texts = [str(details.get('value', '')) for details in form_data.values() 
                     if section not in ['ml_features', 'ml_metadata']]
            ml_features = feature_extractor.extract_features(texts)
            ml_insights['feature_clusters'].append(ml_features)
        except Exception as feature_error:
            print(f"Feature extraction error: {feature_error}")
        
        # Store ML insights
        ml_insights['section_predictions'][json_file] = section_predictions
        ml_insights['confidence_scores'][json_file] = confidence_scores
        
        # Store form data
        all_forms_data[json_file] = form_data
    
    # Advanced Reporting
    print("\n🤖 Machine Learning Insights:")
    print("----------------------------")
    
    # Section Prediction Analysis
    print("\n📋 Section Prediction Distribution:")
    for filename, predictions in ml_insights['section_predictions'].items():
        print(f"\n{filename}:")
        for original_section, predicted_section in predictions.items():
            print(f"  - {original_section}: Predicted as {predicted_section}")
    
    # Confidence Scores
    print("\n📊 Confidence Scores:")
    for filename, scores in ml_insights['confidence_scores'].items():
        print(f"\n{filename}:")
        for section, confidence in scores.items():
            print(f"  - {section}: {confidence * 100:.2f}%")
    
    return all_forms_data, ml_insights

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('/Users/anuragvallur/Developer/pdf_form_digitizer/output/processing_log.txt'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize the PDF Form Digitizer with ML enabled
    digitizer = PDFFormDigitizer(
        input_dir='/Users/anuragvallur/Developer/pdf_form_digitizer/input',
        output_dir='/Users/anuragvallur/Developer/pdf_form_digitizer/output',
        enable_ml=True
    )

    try:
        # 1. Batch Processing with ML
        print("🔍 Starting PDF Form Processing with ML 🔍")
        digitizer.process_pdf_batch()

        # 2. Advanced ML-Powered Data Analysis
        analysis_results, ml_insights = analyze_extracted_data(
            '/Users/anuragvallur/Developer/pdf_form_digitizer/output'
        )

        # 3. Save ML Insights
        ml_insights_path = os.path.join(
            digitizer.output_dir, 
            'ml_insights.json'
        )
        with open(ml_insights_path, 'w') as f:
            json.dump(ml_insights, f, indent=4, cls=NumpyEncoder)

    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()
