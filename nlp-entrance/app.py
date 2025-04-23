#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, jsonify
import pickle
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import os

# Define the process function that was used when creating the model
def process(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove non-letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    # Join words back into text
    return " ".join(words)

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize variables
sentiment_pipeline = None
vectorizer = None
model = None
preprocess_func = None
_is_initialized = False

# Function to load model
def load_model():
    global sentiment_pipeline, vectorizer, model, preprocess_func
    with open('sentiment_pipeline.pkl', 'rb') as f:
        sentiment_pipeline = pickle.load(f)
    
    # Extract components
    vectorizer = sentiment_pipeline['vectorizer']
    model = sentiment_pipeline['model']
    preprocess_func = sentiment_pipeline['preprocess_func']

# Create templates
def create_templates():
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Movie Review Sentiment Analysis</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        textarea {
            width: 100%;
            padding: 8px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            height: 150px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .positive {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .negative {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
        .neutral {
            background-color: #f5f5f5;
            border: 1px solid #e3e3e3;
            color: #333;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 2s linear infinite;
            display: none;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .examples {
            margin-top: 30px;
        }
        .example-review {
            cursor: pointer;
            color: #2196F3;
            text-decoration: underline;
            margin-bottom: 5px;
            display: block;
        }
        .advanced {
            margin-top: 20px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Movie Review Sentiment Analysis</h1>
    
    <div class="form-group">
        <label for="review">Enter a movie review:</label>
        <textarea id="review" placeholder="Type or paste a movie review here..."></textarea>
    </div>
    
    <button id="analyze-btn">Analyze Sentiment</button>
    <div class="loader" id="loader"></div>
    
    <div id="result">
        <h3>Analysis Result:</h3>
        <p><strong>Sentiment:</strong> <span id="sentiment"></span></p>
        <p><strong>Confidence:</strong> <span id="confidence"></span></p>
        <div class="advanced">
            <p><strong>Processed Text:</strong><br>
            <span id="processed-text" style="font-size: 0.9em; color: #666;"></span></p>
        </div>
    </div>
    
    <div class="examples">
        <h3>Example Reviews:</h3>
        <span class="example-review" onclick="useExample(0)">This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout.</span>
        <span class="example-review" onclick="useExample(1)">What a waste of time. The plot was predictable and the characters were incredibly boring. I would not recommend this film.</span>
        <span class="example-review" onclick="useExample(2)">It was an okay movie, not great but not terrible either. Some good moments but overall quite average.</span>
    </div>

    <script>
        // Example reviews
        const exampleReviews = [
            "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout.",
            "What a waste of time. The plot was predictable and the characters were incredibly boring. I would not recommend this film.",
            "It was an okay movie, not great but not terrible either. Some good moments but overall quite average."
        ];
        
        // Use example review
        function useExample(index) {
            document.getElementById('review').value = exampleReviews[index];
        }
        
        // Analyze sentiment
        document.getElementById('analyze-btn').addEventListener('click', function() {
            const review = document.getElementById('review').value.trim();
            
            if (review === '') {
                alert('Please enter a review first.');
                return;
            }
            
            // Show loader
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // Send request to server
            const formData = new FormData();
            formData.append('review', review);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                // Update result
                document.getElementById('sentiment').textContent = data.sentiment;
                document.getElementById('confidence').textContent = data.confidence;
                document.getElementById('processed-text').textContent = data.processed_text;
                
                // Show result with appropriate styling
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = '';
                resultDiv.classList.add(data.sentiment.toLowerCase());
            })
            .catch(error => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            });
        });
    </script>
</body>
</html>
        ''')

# Initialize on first request
@app.before_request
def initialize_on_first_request():
    global _is_initialized
    if not _is_initialized:
        load_model()
        create_templates()
        _is_initialized = True

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the request
    review = request.form['review']
    
    # Preprocess the review
    cleaned_review = preprocess_func(review)
    
    # Vectorize the review
    review_tfidf = vectorizer.transform([cleaned_review])
    
    # Make prediction
    prediction = model.predict(review_tfidf)[0]
    
    # Get prediction probability if the model supports it
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(review_tfidf)[0]
        confidence = proba[prediction]
    else:
        # For models without probability output like SVM
        confidence = None
    
    # Map prediction to sentiment label
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    # Prepare the response
    response = {
        'sentiment': sentiment,
        'confidence': f"{confidence*100:.2f}%" if confidence is not None else "Not available",
        'processed_text': cleaned_review[:200] + "..." if len(cleaned_review) > 200 else cleaned_review
    }
    
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)