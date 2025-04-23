#!/usr/bin/env python
# coding: utf-8

# ## Import Dependencies

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pickle

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ## Import Dataset 

print("Loading dataset...")
df = pd.read_csv('dataset/IMDB Dataset.csv')

# Display sample data
print("\nSample data:")
print(df.sample(3))

# Map sentiment labels to binary values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("\nSentiment value counts (1: positive, 0: negative):")
print(df['sentiment'].value_counts())

# ## Processing the text of the reviews

print("\nPreprocessing text data...")
swords = set(stopwords.words('english'))

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
    review = [w for w in review if w not in swords]
    # Join words back together
    return " ".join(review)

# Apply preprocessing to all reviews
df['processed_review'] = df.review.apply(process)

# Display a sample of original and processed review
sample_idx = 100
print(f"\nOriginal review [{sample_idx}]:")
print(df['review'][sample_idx][:300] + "...")
print(f"\nProcessed review [{sample_idx}]:")
print(df['processed_review'][sample_idx][:300] + "...")

# ## Feature Engineering: TF-IDF Vectorization

X = df['processed_review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_valid)} samples")

# Create and fit the TF-IDF vectorizer
print("\nFitting TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_valid_tfidf = tfidf_vectorizer.transform(X_valid)

print(f"Shape of TF-IDF matrix (Train): {X_train_tfidf.shape}")
print(f"Shape of TF-IDF matrix (Test): {X_valid_tfidf.shape}")

# ## Cross-Validation and Model Comparison

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train, evaluate, and return results for a model"""
    start_time = time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    train_time = time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'], output_dict=True)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'training_time': train_time,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

# Define models to evaluate
models = [
    (LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='liblinear'), "Logistic Regression"),
    (MultinomialNB(), "Naive Bayes"),
    (LinearSVC(C=1.0, max_iter=1000, random_state=42), "Linear SVM"),
    (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
]

# Evaluate all models
print("\nEvaluating multiple models with cross-validation...")
results = []

for model, name in models:
    print(f"\nTraining and evaluating {name}...")
    model_results = evaluate_model(model, X_train_tfidf, y_train, X_valid_tfidf, y_valid, name)
    results.append(model_results)
    
    print(f"Test accuracy: {model_results['accuracy']:.4f}")
    print(f"Cross-validation (5-fold): {model_results['cv_mean']:.4f} (±{model_results['cv_std']:.4f})")
    print(f"Training time: {model_results['training_time']:.2f} seconds")

# Find the best model based on CV score
best_model_idx = np.argmax([result['cv_mean'] for result in results])
best_model = results[best_model_idx]

print(f"\nBest model: {best_model['name']} with CV accuracy: {best_model['cv_mean']:.4f}")

# ## Detailed evaluation of the best model

print("\nDetailed evaluation of the best model:")
print(f"Confusion Matrix for {best_model['name']}:")
print(best_model['confusion_matrix'])

print(f"\nClassification Report for {best_model['name']}:")
for label, metrics in best_model['classification_report'].items():
    if label in ['Negative (0)', 'Positive (1)']:
        print(f"{label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")

# ## Optional: Hyperparameter tuning for the best model
if best_model['name'] == "Logistic Regression":
    print("\nPerforming hyperparameter tuning for Logistic Regression...")
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), 
                              param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_tfidf, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Update the best model
    best_model['model'] = grid_search.best_estimator_
    best_model['tuned'] = True
    
    # Test the tuned model
    y_pred = best_model['model'].predict(X_valid_tfidf)
    tuned_accuracy = accuracy_score(y_valid, y_pred)
    print(f"Tuned model test accuracy: {tuned_accuracy:.4f}")

# ## Save the best model and vectorizer
print("\nSaving the best model and vectorizer...")
model_filename = f"{best_model['name'].lower().replace(' ', '_')}_model.pkl"
joblib.dump(best_model['model'], model_filename)
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print(f"Model saved as {model_filename}")
print("Vectorizer saved as tfidf_vectorizer.pkl")

# Create a simple sentiment analysis pipeline
sentiment_pipeline = {
    'vectorizer': tfidf_vectorizer,
    'model': best_model['model'],
    'preprocess_func': process
}

# Save the pipeline
with open('sentiment_pipeline.pkl', 'wb') as f:
    pickle.dump(sentiment_pipeline, f)
print("Complete pipeline saved as sentiment_pipeline.pkl")

# Test on example reviews
new_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout.",
    "What a waste of time. The plot was predictable and the characters were incredibly boring. I would not recommend this film.",
    "It was an okay movie, not great but not terrible either. Some good moments but overall quite average."
]

print("\n--- Testing on New Reviews ---")

# Preprocess the new reviews
cleaned_new_reviews = [process(review) for review in new_reviews]

# Transform using the fitted TF-IDF vectorizer
new_reviews_tfidf = tfidf_vectorizer.transform(cleaned_new_reviews)

# Predict sentiment
new_predictions = best_model['model'].predict(new_reviews_tfidf)
sentiment_labels = {1: 'Positive', 0: 'Negative'}

# Print results
for review, prediction in zip(new_reviews, new_predictions):
    print(f"\nReview: \"{review[:100]}...\"")
    print(f"Predicted Sentiment: {sentiment_labels[prediction]} ({prediction})")

print("\n--- Enhanced Sentiment Analysis Project Complete ---")