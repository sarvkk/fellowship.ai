
## Project Overview
This project implements a machine learning pipeline to perform sentiment analysis on movie reviews from the IMDB dataset. The model classifies reviews as either positive or negative based on the text content of the review.

## Dataset
The project uses the IMDB Dataset of 50,000 movie reviews, with an even split between positive and negative sentiment labels. This is a widely used benchmark dataset for binary sentiment classification tasks.
### Data Source
The dataset is available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and consists of 50,000 movie reviews from IMDB, labeled with sentiment (positive/negative).

#### Dataset Details:
- **Size**: 50,000 reviews
- **Format**: CSV file with 'review' and 'sentiment' columns
- **Classes**: Balanced dataset (25,000 positive, 25,000 negative)
- **Text Length**: Reviews vary in length from short paragraphs to longer analyses

## Features
- Text preprocessing pipeline (HTML removal, stopword removal, etc.)
- Feature extraction using TF-IDF vectorization
- Model comparison (Logistic Regression, Naive Bayes, SVM, Random Forest)
- Cross-validation and hyperparameter tuning
- Web interface for real-time sentiment prediction
- Serialized model and vectorizer for easy deployment

## Project Structure
```
fellowship.ai/nlp-entrance/
├── dataset/                     # Dataset directory
│   └── IMDB Dataset.csv         # IMDB movie reviews dataset
├── templates/                   # Web application templates
│   └── index.html               # UI for sentiment analysis
├── imdb-sentiment-analysis.py   # Main script for model training
├── imdb-sentiment-analysis.ipynb # Jupyter notebook version
├── logistic_regression_model.pkl # Trained model
├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
└── sentiment_pipeline.pkl       # Complete sentiment analysis pipeline
```

## Installation

### Requirements
- Python 3.6+
- pandas
- numpy
- scikit-learn
- nltk
- BeautifulSoup4
- matplotlib
- seaborn

### Setup
```bash
# Clone the repository
git clone https://github.com/sarvkk/fellowship.ai.git
cd fellowship.ai/nlp-entrance

# Download NLTK stopwords (if not already downloaded)
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### Training the Model
```bash
python imdb-sentiment-analysis.py
```
This script will:
1. Load and preprocess the IMDB dataset
2. Train multiple classification models
3. Evaluate and compare model performance
4. Save the best performing model and vectorizer

### Using the Web Interface
```bash
# Start the web server (assuming Flask is installed)
python app.py
```
Then open your browser to http://localhost:5000 to access the sentiment analysis interface.

### Using the Model Programmatically
```python
import pickle
import joblib

# Load the pipeline
with open('sentiment_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Process a new review
def predict_sentiment(review_text):
    # Preprocess the text
    processed_text = pipeline['preprocess_func'](review_text)
    
    # Vectorize the processed text
    vectorized_text = pipeline['vectorizer'].transform([processed_text])
    
    # Predict sentiment
    prediction = pipeline['model'].predict(vectorized_text)[0]
    
    return "Positive" if prediction == 1 else "Negative"

# Example usage
review = "This movie was absolutely fantastic! The acting was superb."
sentiment = predict_sentiment(review)
print(f"Sentiment: {sentiment}")
```

## Model Performance
The best performing model achieves approximately 89% accuracy on the test set. Detailed performance metrics include:

- **Accuracy**: ~89%
- **Precision**: ~89% for both positive and negative classes
- **Recall**: ~90% for positive class, ~88% for negative class
- **F1-Score**: ~89% overall

## Example Predictions
```
Review: "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout."
Predicted Sentiment: Positive

Review: "What a waste of time. The plot was predictable and the characters were incredibly boring."
Predicted Sentiment: Negative

Review: "It was an okay movie, not great but not terrible either. Some good moments but overall quite average."
Predicted Sentiment: Negative
```

## Future Improvements
- Experiment with more advanced NLP techniques (word embeddings, transformers)
- Implement multi-class sentiment analysis (very negative, negative, neutral, positive, very positive)
- Add aspect-based sentiment analysis to identify sentiment about specific movie elements (acting, plot, visuals)
- Deploy the model as a public API

## Acknowledgements
- [Lakshmipathi N](https://www.kaggle.com/lakshmi25npathi) for the IMDB Dataset used for this project
- Fellowship.ai for the project idea