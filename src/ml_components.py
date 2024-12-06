import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, 
    GlobalMaxPooling1D, Dropout, 
    Concatenate
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import re

class PDFFormMLExtractor:
    def __init__(self, max_words=10000, max_len=100):
        """
        Sets up the machine learning components for PDF form extraction.
        
        Args:
            max_words (int): How many unique words we'll consider 
            max_len (int): Maximum length of text we'll process
        """
        self.max_words = max_words
        self.max_len = max_len
        
        # Text preprocessing components
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        
        # ML Models
        self.field_extraction_model = None
        self.text_classifier = None
    
    def preprocess_text(self, text):
        """
        Cleans up the text before feeding it to our ML models
        
        Args:
            text (str): The raw text we want to clean up
        
        Returns:
            Processed text, ready for analysis
        """
        # Remove special characters and extra whitespaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def prepare_training_data(self, json_dir):
        """
        Turns our JSON files into training data for the ML model.
        
        Args:
            json_dir (str): Where our extracted JSON files live
        
        Returns:
            Processed data ready for model training
        """
        all_texts = []
        all_labels = []
        
        # Collect texts and labels from JSON files
        for filename in os.listdir(json_dir):
            if filename.endswith('_extracted.json'):
                filepath = os.path.join(json_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    for section, details in data.items():
                        # Safely extract text, defaulting to empty string
                        text = str(details.get('value', '')) if isinstance(details, dict) else ''
                        
                        # Only add non-empty texts
                        if text.strip():
                            all_texts.append(self.preprocess_text(text))
                            all_labels.append(section)
                
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {filename}: {e}")
        
        # Handle case with no valid training data
        if not all_texts:
            print("Warning: No valid training data found!")
            # Return dummy data to prevent training failure
            all_texts = ['dummy text']
            all_labels = ['dummy_section']
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(all_labels)
        
        # Tokenize texts
        self.tokenizer.fit_on_texts(all_texts)
        text_sequences = self.tokenizer.texts_to_sequences(all_texts)
        text_padded = pad_sequences(text_sequences, maxlen=self.max_len)
        
        return text_padded, labels_encoded
    
    def build_field_extraction_model(self):
        """
        Build a neural network for form field extraction
        
        Returns:
            tf.keras.Model: Compiled ML model for field extraction
        """
        model = Sequential([
            Embedding(self.max_words, 64, input_length=self.max_len),
            LSTM(64, return_sequences=True),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(np.unique(self.label_encoder.classes_)), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model
    
    def build_text_classifier(self):
        """
        Build a text classification model with multiple inputs
        
        Returns:
            tf.keras.Model: Compiled text classification model
        """
        # Text input
        text_input = Input(shape=(self.max_len,), name='text_input')
        text_embedding = Embedding(self.max_words, 64)(text_input)
        text_lstm = LSTM(64)(text_embedding)
        
        # Metadata input
        metadata_input = Input(shape=(10,), name='metadata_input')
        
        # Combine text and metadata
        combined = Concatenate()([text_lstm, metadata_input])
        
        # Classification layers
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.5)(x)
        output = Dense(len(np.unique(self.label_encoder.classes_)), 
                      activation='softmax', 
                      name='section_output')(x)
        
        model = Model(inputs=[text_input, metadata_input], outputs=output)
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, json_dir):
        """
        Train ML models on extracted form data
        
        Args:
            json_dir (str): Directory containing extracted JSON files
        """
        # Prepare training data
        X, y = self.prepare_training_data(json_dir)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Field Extraction Model
        self.field_extraction_model = self.build_field_extraction_model()
        history_extraction = self.field_extraction_model.fit(
            X_train, y_train, 
            epochs=10, 
            validation_split=0.2, 
            batch_size=32
        )
        
        # Text Classifier
        metadata = np.random.rand(len(X_train), 10)  # Placeholder metadata
        self.text_classifier = self.build_text_classifier()
        history_classifier = self.text_classifier.fit(
            [X_train, metadata], y_train,
            epochs=10, 
            validation_split=0.2, 
            batch_size=32
        )
        
        return {
            'field_extraction': history_extraction.history,
            'text_classifier': history_classifier.history
        }
    
    def predict_section(self, text):
        """
        Predict the section of a given text
        
        Args:
            text (str): Input text to classify
        
        Returns:
            str: Predicted section
        """
        preprocessed_text = self.preprocess_text(text)
        sequence = self.tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        
        prediction = self.field_extraction_model.predict(padded_sequence)
        predicted_label_index = np.argmax(prediction)
        
        # Convert back to original label
        return self.label_encoder.inverse_transform([predicted_label_index])[0]

class TransferLearningFeatureExtractor:
    def __init__(self, pretrained_model='simple'):
        """
        Sets up a feature extractor that can learn from existing knowledge.
        
        Args:
            pretrained_model (str): Which pre-trained model to start with
        """
        self.pretrained_model = pretrained_model
        
        # Simple embedding-based feature extractor
        if pretrained_model == 'simple':
            self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
            self.embedding_layer = Embedding(10000, 64)
    
    def extract_features(self, texts):
        """
        For extracting features from text
        
        Args:
            texts (list): Texts we want to dig into
        
        Returns:
            The most meaningful features from the texts
        """
        # Tokenize texts
        self.tokenizer.fit_on_texts(texts)
        text_sequences = self.tokenizer.texts_to_sequences(texts)
        text_padded = pad_sequences(text_sequences, maxlen=100)
        
        # Extract embeddings
        features = self.embedding_layer(text_padded).numpy()
        
        # Average pooling
        return np.mean(features, axis=1)

# Utility functions for ML-enhanced extraction
def enhance_extraction_with_ml(extraction_results, ml_extractor):
    """
    Enhance extraction results with ML insights
    
    Args:
        extraction_results (dict): The original data we extracted
        ml_extractor (PDFFormMLExtractor): Our ML toolkit
    
    Returns:
        Enhanced results with additional insights and confidence
    """
    enhanced_results = extraction_results.copy()
    
    for section, details in extraction_results.items():
        try:
            # Check if value exists and is not None
            text_value = str(details.get('value', '')) if details.get('value') is not None else ''
            
            if text_value:
                # Predict section using ML model
                predicted_section = ml_extractor.predict_section(text_value)
                
                # Add ML insights to the result
                enhanced_results[section]['ml_predicted_section'] = predicted_section
                enhanced_results[section]['ml_confidence'] = float(np.max(
                    ml_extractor.field_extraction_model.predict(
                        pad_sequences(
                            ml_extractor.tokenizer.texts_to_sequences([text_value]), 
                            maxlen=ml_extractor.max_len
                        )
                    )[0]
                ))
        except Exception as e:
            # Log the error without stopping the entire process
            print(f"ML enhancement error for section {section}: {e}")
    
    return enhanced_results