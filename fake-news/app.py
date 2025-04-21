# -*- coding: utf-8 -*-
from pickle import load
from flask import Flask, request, render_template
from joblib import dump
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    true = pd.read_csv('True.csv')
    fake = pd.read_csv('Fake.csv')
    true['label'] = 1
    fake['label'] = 0
    news = pd.concat([true, fake], axis=0)
    news = news.drop(['title', 'subject', 'date'], axis=1)
    news = news.reset_index(drop=True)
    return news

# Text preprocessing function
def wordopt(content):
    content = content.lower()
    content = re.sub(r'https?://\S+|www\.\S+', '', content)
    content = re.sub(r'<.*?>', '', content)
    content = re.sub(r'[^\w\s]', '', content)
    content = re.sub(r'\d', '', content)
    content = re.sub(r'\n', '', content)
    return content

# Train models and save them, or load if they exist
def train_or_load_models(news):
    # Define file paths for saved models
    vectorizer_path = 'vectorizer.joblib'
    lr_path = 'lr_model.joblib'
    gbc_path = 'gbc_model.joblib'
    rfc_path = 'rfc_model.joblib'

    # Check if all model files exist
    if all(os.path.exists(path) for path in [vectorizer_path, lr_path, gbc_path, rfc_path]):
        print("Loading pre-trained models from disk...")
        vectorization = load(vectorizer_path)
        LR = load(lr_path)
        gbc = load(gbc_path)
        rfc = load(rfc_path)
    else:
        print("Training models and saving to disk...")
        # Prepare data
        news['content'] = news['text'].apply(wordopt)
        X = news['text']
        y = news['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Vectorize text data
        vectorization = TfidfVectorizer(max_features=10000)
        Xv_train = vectorization.fit_transform(X_train)
        Xv_test = vectorization.transform(X_test)

        # Train models with regularization
        LR = LogisticRegression(C=0.1)
        LR.fit(Xv_train, y_train)

        gbc = GradientBoostingClassifier(learning_rate=0.05, n_estimators=50)
        gbc.fit(Xv_train, y_train)

        rfc = RandomForestClassifier(max_depth=10, n_estimators=50)
        rfc.fit(Xv_train, y_train)

        # Evaluate models
        print("Logistic Regression Score:", LR.score(Xv_test, y_test))
        print("Gradient Boosting Score:", gbc.score(Xv_test, y_test))
        print("Random Forest Score:", rfc.score(Xv_test, y_test))

        # Save models and vectorizer
        dump(vectorization, vectorizer_path)
        dump(LR, lr_path)
        dump(gbc, gbc_path)
        dump(rfc, rfc_path)
        print("Models and vectorizer saved to disk.")

    return vectorization, LR, gbc, rfc

# Define label output function
def output_label(n):
    if n == 0:
        return "It is fake news"
    elif n == 1:
        return "It is true news"

# Manual testing function for predictions
def manual_testing(news_article, vectorization, LR, gbc, rfc):
    testing_news = {"text": [news_article]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_lr = LR.predict_proba(new_xv_test)[0]
    pred_gbc = gbc.predict_proba(new_xv_test)[0]
    pred_rfc = rfc.predict_proba(new_xv_test)[0]

    return (
        f"Logistic Regression Prediction: {output_label(np.argmax(pred_lr))} (Confidence: {max(pred_lr):.2f})\n"
        f"Gradient Boosting Prediction: {output_label(np.argmax(pred_gbc))} (Confidence: {max(pred_gbc):.2f})\n"
        f"Random Forest Prediction: {output_label(np.argmax(pred_rfc))} (Confidence: {max(pred_rfc):.2f})"
    )

# Load data and initialize models globally
news = load_and_preprocess_data()
vectorization, LR, gbc, rfc = train_or_load_models(news)

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        news_article = request.form['news_article']
        if news_article:
            prediction = manual_testing(news_article, vectorization, LR, gbc, rfc)
    return render_template('index.html', prediction=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)  # Disable debug mode for production-like behavior