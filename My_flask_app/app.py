import matplotlib
matplotlib.use('Agg')
import nltk
import re
import ssl
import certifi
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from wordcloud import WordCloud, STOPWORDS

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# SSL context for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.data.path.append(certifi.where())
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load and preprocess data
try:
    trump_data = pd.read_csv('My_flask_app/data/Trumpall2.csv')
    biden_data = pd.read_csv('My_flask_app/data/Bidenall2.csv')
except FileNotFoundError as e:
    raise SystemExit("Please ensure the CSV files are in the correct directory.")

sid = SentimentIntensityAnalyzer()

def analyze_sentiments(data):
    data["Polarity"] = data["text"].apply(lambda x: sid.polarity_scores(x)['compound'])
    data["Sentiment"] = np.where(data["Polarity"] > 0, "Positive", np.where(data["Polarity"] < 0, "Negative", "Neutral"))
    return data

trump_data = analyze_sentiments(trump_data)
biden_data = analyze_sentiments(biden_data)

data = pd.concat([trump_data, biden_data])
data = data[data["Sentiment"] != "Neutral"]
data["Sentiment"] = data["Sentiment"].map({"Positive": 1, "Negative": 0})

X = data["text"]
y = data["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

total_positive_trump = (trump_data["Sentiment"] == "Positive").sum()
total_negative_trump = (trump_data["Sentiment"] == "Negative").sum()
total_positive_biden = (biden_data["Sentiment"] == "Positive").sum()
total_negative_biden = (biden_data["Sentiment"] == "Negative").sum()

positive_counts = [total_positive_trump, total_positive_biden]
negative_counts = [total_negative_trump, total_negative_biden]

def predict_sentiment(tweet):
    tweet_tfidf = vectorizer.transform([tweet])
    prediction = model.predict(tweet_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

def update_sentiment_counts(new_tweet, candidate):
    sentiment = predict_sentiment(new_tweet)
    if candidate == "Trump":
        if sentiment == "Positive":
            positive_counts[0] += 1
        else:
            negative_counts[0] += 1
    elif candidate == "Biden":
        if sentiment == "Positive":
            positive_counts[1] += 1
        else:
            negative_counts[1] += 1

    winner = "Trump" if positive_counts[0] - negative_counts[0] > positive_counts[1] - negative_counts[1] else "Biden"
    return sentiment, winner

def plot_sentiments():
    names = ["Trump", "Biden"]
    positive_values = [total_positive_trump, total_positive_biden]
    negative_values = [total_negative_trump, total_negative_biden]

    plt.figure(figsize=(10, 6))
    plt.bar(names, positive_values, color='green', label='Positive')
    plt.bar(names, negative_values, color='red', label='Negative', bottom=positive_values)
    plt.xlabel('Candidates')
    plt.ylabel('Number of Tweets')
    plt.title('Sentiment Analysis of Trump and Biden Tweets')
    plt.legend()

    graph_path = os.path.join('static', 'images', 'updated_sentiment_plot.png')
    if not os.path.exists(os.path.dirname(graph_path)):
        os.makedirs(os.path.dirname(graph_path))
    plt.savefig(graph_path)
    plt.close()
    return graph_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        results, graph_path, total_positive_trump, total_positive_biden, trump_comments, biden_comments = perform_sentiment_analysis()
        winner = "Trump" if total_positive_trump > total_positive_biden else "Biden"
        return render_template('sentiment_analysis.html', 
                               graph_path=graph_path, 
                               results=results, 
                               trump_comments=trump_comments,
                               biden_comments=biden_comments,
                               winner=winner)
    results, graph_path, total_positive_trump, total_positive_biden, trump_comments, biden_comments = perform_sentiment_analysis()
    winner = "Trump" if total_positive_trump > total_positive_biden else "Biden"
    return render_template('sentiment_analysis.html', 
                           graph_path=graph_path, 
                           results=results, 
                           trump_comments=trump_comments,
                           biden_comments=biden_comments,
                           winner=winner)

@app.route('/predictive_analysis', methods=['GET', 'POST'])
def predictive_analysis():
    if request.method == 'POST':
        new_tweet = request.form.get('new_tweet')
        candidate = request.form.get('candidate')
        sentiment, winner = update_sentiment_counts(new_tweet, candidate)
        graph_path = plot_sentiments()
        return render_template('predictive_analysis.html', 
                               accuracy=accuracy, 
                               report=report, 
                               graph_path=graph_path,
                               new_comment_prediction=sentiment,
                               winner=winner,
                               positive_counts=positive_counts,
                               negative_counts=negative_counts)
    graph_path = plot_sentiments()
    return render_template('predictive_analysis.html', 
                           accuracy=accuracy, 
                           report=report, 
                           graph_path=graph_path,
                           new_comment_prediction=None,
                           winner=None,
                           positive_counts=positive_counts,
                           negative_counts=negative_counts)

def perform_sentiment_analysis():
    total_positive_trump = (trump_data["Sentiment"] == "Positive").sum()
    total_negative_trump = (trump_data["Sentiment"] == "Negative").sum()
    total_positive_biden = (biden_data["Sentiment"] == "Positive").sum()
    total_negative_biden = (biden_data["Sentiment"] == "Negative").sum()

    trump_comments = trump_data[["text", "Sentiment"]].head(10).values.tolist()
    biden_comments = biden_data[["text", "Sentiment"]].head(10).values.tolist()

    names = ["Trump", "Biden"]
    positive_values = [total_positive_trump, total_positive_biden]
    negative_values = [total_negative_trump, total_negative_biden]

    results = {
        "names": names,
        "positive_values": positive_values,
        "negative_values": negative_values
    }

    graph_path = plot_sentiments()

    return results, graph_path, total_positive_trump, total_positive_biden, trump_comments, biden_comments

if __name__ == '__main__':
    app.run(debug=True)
