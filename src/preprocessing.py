# src/preprocessing.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    # 소문자 변환, 구두점 제거 등
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    return text

def vectorize_text(data, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data)
    return X, vectorizer
