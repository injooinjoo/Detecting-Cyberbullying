# src/preprocessing.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# NLTK 리소스 다운로드
nltk.download('stopwords')
nltk.download('wordnet')

# 텍스트 정제 함수
def clean_text(text):
    # 소문자 변환
    text = text.lower()
    # 구두점 및 특수 문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# TF-IDF 벡터화 함수
def vectorize_text(data, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data)
    return X, vectorizer
