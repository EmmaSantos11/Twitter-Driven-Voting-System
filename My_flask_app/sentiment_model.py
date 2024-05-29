import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model, vectorizer = pickle.load(f)
    except FileNotFoundError:
        # Load the data and train the model if pickle file is not found
        trump_data = pd.read_csv('My_flask_app/data/Trumpall2.csv')
        biden_data = pd.read_csv('My_flask_app/data/Bidenall2.csv')
        data = pd.concat([trump_data, biden_data])

        # Initialize the sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        # Perform sentiment analysis and calculate compound polarity scores
        data["Polarity"] = data["text"].apply(lambda x: sid.polarity_scores(x)['compound'])

        # Assign sentiment labels based on compound polarity scores
        data["Sentiment"] = data["Polarity"].apply(lambda x: 1 if x > 0 else 0)

        # Extract features and labels
        X = data["text"]
        y = data["Sentiment"]

        # Convert the text data to TF-IDF features
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)

        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(X_tfidf, y)

        # Save the trained model and vectorizer to a pickle file
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump((model, vectorizer), f)

    return model, vectorizer
