# Spam_Classifier_model
Building a spam comments classifier model for integrating it with my team's project.
This project is a machine learning model built to classify comments as either Spam or Not Spam. It uses Natural Language Processing (NLP) to filter out unwanted promotions, advertisements, and scam links.

The model is deployed as an interactive web app using Streamlit.

## Live Demo
You can test the live model here:  https://spamclassifiermodel-by-anshulkumarchauhan.streamlit.app/

1. Project Overview
The goal of this project is to build an effective and fast classifier for spam comments. The model is built as a complete scikit-learn pipeline that handles text preprocessing, feature extraction, and classification all in one object.

Features
NLP Pipeline: Uses TfidfVectorizer to convert raw text into meaningful numerical features.

Classifier: Employs a MultinomialNB (Multinomial Naive Bayes) algorithm, which is a classic, high-performance model for text classification.

Web App: Deployed with Streamlit for easy, interactive testing.

Model: The trained scikit-learn pipeline is serialized using pickle for inference in the app.

2. How It Works: The ML Pipeline
The entire process, from raw text to a final prediction, is handled by a scikit-learn Pipeline. This makes the model robust and easy to deploy.

Here are the two main stages:

1. **Text Vectorization** (TF-IDF)
Computers don't understand words, so we must convert the text comments into numbers.

The pipeline first cleans the text (removes punctuation, makes it lowercase, and removes common "stop words" like 'the', 'is', 'and').

It then uses a TfidfVectorizer. This stands for Term Frequency-Inverse Document Frequency. It's a "smart" way of counting words.

It gives a high score to words that are very frequent in one comment but rare across all other comments (e.g. "subscribe," "free"). This makes it excellent at finding spammy keywords.

2. **Classification** (Multinomial Naive Bayes)
This numerical data from the TF-IDF vectorizer is fed into the classifier.

We use a MultinomialNB (Naive Bayes) model.

This is a fast and highly effective (probability-based) classifier. It calculates the probability that a comment is Spam given the set of words it contains.

It's the industry-standard "baseline" model for spam filtering because it's fast to train and works exceptionally well.

3. Tech Stack
Python 3

Pandas: For loading and combining the datasets.

Scikit-learn: For building the complete ML pipeline (TfidfVectorizer, MultinomialNB).

Pickle: For saving (serializing) the trained model object.

Streamlit: For creating and deploying the interactive web app.

4. Dataset
This model was trained on the YouTube Spam Collection Dataset from Kaggle. This dataset is a collection of about ~2,000 hand-labeled comments from 5 different popular YouTube videos.

Kaggle: YouTube Spam Collection Dataset

Preprocessing: The 5 separate CSV files were combined into a single dataset for training.

Labels: 1 (Spam) or 0 (Not Spam).
