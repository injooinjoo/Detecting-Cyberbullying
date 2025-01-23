# src/run_pipeline.py

import pandas as pd
from preprocessing import clean_text, vectorize_text
from modeling import train_random_forest, evaluate_model, plot_confusion_matrix
from utils import save_model, save_metrics

# 데이터 로드
data = pd.read_csv('../data/raw/twitter_cyberbullying.csv')

# 텍스트 전처리
data['tweet_text'] = data['tweet_text'].apply(clean_text)

# TF-IDF 벡터화
X, vectorizer = vectorize_text(data['tweet_text'])
y = data['cyberbullying_type']

# 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = train_random_forest(X_train, y_train)

# 모델 평가
accuracy, f1, auc = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# 혼동 행렬 시각화
plot_confusion_matrix(model, X_test, y_test, save_path='../results/plots/confusion_matrix.png')

# 모델 및 평가 지표 저장
save_model(model, '../results/models/RandomForest_model.pkl')
metrics_df = pd.DataFrame({
    'Model': ['Random Forest'],
    'Accuracy': [accuracy],
    'F1 Score': [f1],
    'AUC-ROC': [auc]
})
save_metrics(metrics_df, '../results/metrics/model_performance.csv')
