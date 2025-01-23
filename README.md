# Detecting-Cyberbullying

# Detecting Cyberbullying in Social Media Text Data

## Project Overview
This project aims to detect and classify cyberbullying in social media text data using machine learning. The dataset consists of 50,000 labeled tweets categorized into 6 types of cyberbullying (Age, Ethnicity, Religion, Gender, Other Cyberbullying, and Not Cyberbullying).

## Key Features
- **Data Preprocessing**: Text cleaning, vectorization (TF-IDF, Word2Vec).
- **Exploratory Data Analysis**: Word clouds, sentiment analysis.
- **Model Development**: Decision Tree, Random Forest, Naive Bayes, Neural Networks.
- **Model Evaluation**: Accuracy, AUC, F1 Score.

## Repository Structure
Detecting-Cyberbullying/
├── data/ # 데이터 파일 저장
│ ├── raw/ # 원본 데이터
│ ├── processed/ # 전처리된 데이터
├── notebooks/ # Jupyter Notebooks
│ ├── 01_EDA.ipynb # 탐색적 데이터 분석
│ ├── 02_Preprocessing.ipynb # 데이터 전처리
│ ├── 03_Modeling.ipynb # 모델 개발 및 평가
├── src/ # Python 스크립트
│ ├── preprocessing.py # 데이터 전처리 스크립트
│ ├── modeling.py # 모델 개발 스크립트
│ ├── utils.py # 유틸리티 함수
├── results/ # 결과 파일
│ ├── models/ # 학습된 모델 파일
│ ├── plots/ # 시각화 결과 (워드 클라우드, 그래프 등)
│ ├── metrics/ # 모델 평가 결과 (CSV 파일 등)
├── README.md # 프로젝트 개요
├── requirements.txt # 필요한 Python 패키지 목록
├── .gitignore # Git에서 무시할 파일 목록

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/injooinjoo/Detecting-Cyberbullying.git
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebooks in the notebooks/ folder to reproduce the analysis:

01_EDA.ipynb: Exploratory Data Analysis.

02_Preprocessing.ipynb: Data preprocessing.

03_Modeling.ipynb: Model development and evaluation.
