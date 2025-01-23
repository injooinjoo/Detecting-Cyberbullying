# src/utils.py

import joblib

# 모델 저장 함수
def save_model(model, path):
    joblib.dump(model, path)

# 모델 불러오기 함수
def load_model(path):
    return joblib.load(path)

# 평가 지표 저장 함수
def save_metrics(metrics, path):
    metrics.to_csv(path, index=False)
