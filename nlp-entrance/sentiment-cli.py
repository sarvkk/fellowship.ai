#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import sys
import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def process(review):
    """Clean and preprocess review text"""
    # Remove HTML tags
    review = BeautifulSoup(review, 'html.parser').get_text()
    # Remove non-alphabetic characters
    review = re.sub("[^a-zA-Z]", ' ', review)
    # Convert to lowercase
    review = review.lower()
    # Split into words
    review = review.split()
    # Remove stopwords
    swords = set(stopwords.words('english'))
    review = [w for w in review if w not in swords]
    # Join words back together
    return " ".join(review)

def load_model(vectorizer_path='tfidf_vectorizer.pkl', model_path='logistic_regression_model.pkl'):
    """Load the saved model and vectorizer"""
    try:
        # Check if files exist
        if not os.path.exists(vectorizer_path):
            print(f"Error: Vectorizer file '{vectorizer_path}' not found.")
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            sys.exit(1)
        
        # Load files
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def analyze_sentiment(text, vectorizer, model):
    """Analyze the sentiment of a given text"""
    # Preprocess the text
    cleaned_text = process(text)
    
    # Vectorize the text
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get prediction probability if the model supports it
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        confidence = proba[prediction]
    else:
        confidence = None
    
    # Map prediction to sentiment label
    sentiment = 'positive' if prediction == 1 else 'negative'
    
    return {
        'sentiment': sentiment,
        'prediction': prediction,
        'confidence': confidence,
        'processed_text': cleaned_text
    }

def main():
    """Main function for the sentiment analysis CLI tool"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Sentiment Analysis CLI Tool')
    
    # Create a group for input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', '-t', type=str, help='Text to analyze')
    input_group.add_argument('--file', '-f', type=str, help='File containing text to analyze')
    input_group.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    # Other options
    parser.add_argument('--show-processed', '-p', action='store_true', 
                      help='Show processed text')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure NLTK resources are downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('stopwords')
    
    # Load the model and vectorizer
    print("Loading sentiment analysis model...")
    vectorizer, model = load_model()
    print("Model loaded successfully!")
    
    # Process based on input option
    if args.text:
        result = analyze_sentiment(args.text, vectorizer, model)
        print(f"\nAnalyzing text: \"{args.text[:100]}{'...' if len(args.text) > 100 else ''}\"")
        print(f"\nSentiment: {result['sentiment'].upper()}")
        
        if result['confidence'] is not None:
            confidence_percentage = result['confidence'] * 100
            print(f"Confidence: {confidence_percentage:.2f}%")
        
        if args.show_processed:
            print(f"\nProcessed Text:")
            print(f"{result['processed_text']}")
    
    elif args.file:
        try:
            print(f"\nAnalyzing file: {args.file}")
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = analyze_sentiment(text, vectorizer, model)
            print(f"\nSentiment: {result['sentiment'].upper()}")
            
            if result['confidence'] is not None:
                confidence_percentage = result['confidence'] * 100
                print(f"Confidence: {confidence_percentage:.2f}%")
            
            if args.show_processed:
                print(f"\nProcessed Text:")
                print(f"{result['processed_text']}")
        except Exception as e:
            print(f"Error analyzing file {args.file}: {e}")
    
    elif args.interactive:
        print("\nInteractive Sentiment Analysis Mode")
        print("Type or paste text and press Enter to analyze. Type 'quit' or 'exit' to end.")
        
        while True:
            print("\nEnter text to analyze:")
            text = input("> ")
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break
            
            if not text.strip():
                print("Please enter some text to analyze.")
                continue
            
            result = analyze_sentiment(text, vectorizer, model)
            print(f"\nSentiment: {result['sentiment'].upper()}")
            
            if result['confidence'] is not None:
                confidence_percentage = result['confidence'] * 100
                print(f"Confidence: {confidence_percentage:.2f}%")
            
            if args.show_processed:
                print(f"\nProcessed Text:")
                print(f"{result['processed_text']}")

if __name__ == '__main__':
    main()