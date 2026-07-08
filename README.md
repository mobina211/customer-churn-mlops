# Customer Churn Prediction MLOps Project

## Project Overview

This project implements an end-to-end Machine Learning pipeline for Customer Churn Prediction using the Telco Customer Churn dataset.

The pipeline includes:

- Data Loading
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Selection
- Model Persistence
- Experiment Tracking with MLflow
- Containerization with Docker

---

## Dataset

Dataset used:

Telco Customer Churn Dataset

Dataset versions:

- v1 → Raw Dataset
- v2 → Preprocessed Dataset
- v3 → Feature Engineered Dataset

---

## Project Structure

text
customer-churn-mlops/
│
├── data/
│   ├── v1/
│   ├── v2/
│   └── v3/
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── save_model.py
│   └── mlflow_utils.py
│
├── run_pipeline.py
├── requirements.txt
├── Dockerfile
└── README.md


---

## Machine Learning Pipeline

### 1. Data Loading

Loads the raw customer churn dataset.

### 2. Data Preprocessing

- Remove CustomerID
- Remove Churn Reason
- Handle missing values
- Convert Total Charges to numeric

Output:

text
data/v2/


### 3. Feature Engineering

- Remove unnecessary columns
- One-Hot Encoding
- Convert boolean values to numeric

Output:

text
data/v3/


### 4. Model Training

Models used:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

Cross Validation:

text
Stratified K-Fold (3 folds)


### 5. Model Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### 6. Best Model Selection

Best model is selected using:

text
F1 Score


### 7. Model Saving

The best model is stored as:

text
models/best_model.pkl


---

## MLflow Tracking

MLflow is used for:

- Parameter Tracking
- Metric Tracking
- Model Logging
- Experiment Management

Experiment Name:

text
Customer Churn Prediction


---

## Docker

Build image:

bash
docker build -t customer-churn-mlops .


Run container:

bash
docker run --rm customer-churn-mlops


---

## Results

Best Model:

text
Logistic Regression


Best F1 Score:

text
0.6236


---

## Technologies

- Python
- Pandas
- Scikit-Learn
- XGBoost
- CatBoost
- MLflow
- Docker

---

## Author

Mobina Kazemzadeh