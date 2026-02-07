# 🚀 Credit Risk Model - Training Pipeline & Deployment (FastAPI + LightGBM + Docker)

This project is a complete **end-to-end Credit Risk Prediction System** that predicts whether a borrower is likely to **default on a loan** (serious delinquency within the next 2 years).

It includes:
- A full **training pipeline**
- A saved ML model + preprocessing artifacts
- A **FastAPI production API**
- Docker support
- Railway deployment support

---

## 📌 What is Credit Risk Prediction?

Credit risk prediction is used by banks and NBFCs to estimate whether a customer will:
- repay the loan successfully (**No Default**)
- or fail to repay (**Default**)

This helps financial institutions:
- reduce bad loans
- improve approval decisions
- set better interest rates

---

## 🎯 Project Goal

Build a machine learning model that classifies loan applicants into:

- `0` → No Default
- `1` → Default

and expose it using a production-ready API.

---

## 📊 Dataset Information

- **Rows:** 16,714 credit applications
- **Features:** 10 input variables
- **Target distribution:** balanced (0 and 1 are equal)

Input Features:
- `rev_util` → revolving credit utilization ratio  
- `age` → age of borrower  
- `late_30_59` → 30–59 days late count  
- `late_60_89` → 60–89 days late count  
- `late_90` → 90+ days late count  
- `debt_ratio` → debt-to-income ratio  
- `monthly_inc` → monthly income  
- `open_credit` → open credit lines  
- `real_estate` → real estate loans  
- `dependents` → number of dependents  

---

## 🏗️ Project Structure

Credit_Risk_Project/
│
├── api/
│ ├── main.py # FastAPI application
│ ├── schemas.py # Input/output schema
│ ├── middleware.py # Logging + validation middleware
│
├── src/
│ ├── data/
│ │ ├── data_loader.py # Data loading logic
│ │ ├── eda_report.py # EDA report generation
│ │ └── preprocessor.py # Data preprocessing module
│ │
│ ├── features/
│ │ └── feature_engineering.py # Feature engineering logic
│ │
│ ├── models/
│ │ ├── train.py # Model training logic
│ │ ├── evaluate.py # Model evaluation utilities
│ │ └── predict.py # Predictor class (loads artifacts)
│ │
│ └── utils/
│ └── helpers.py # Logging + config utilities
│
├── models/
│ └── saved_models/
│ ├── best_model.pkl
│ ├── preprocessor.pkl
│ ├── feature_engineer.pkl
│
├── config/
│ └── config.yaml
│
├── docker/
│ └── docker-compose.yml
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md


---

# ⚙️ Training Pipeline (How the Model Works)

This project contains a full training pipeline that performs:

### ✅ Step 1: Data Loading
- Loads dataset
- validates schema

### ✅ Step 2: Train-Test Split
- Train data: 80%
- Test data: 20%

### ✅ Step 3: Preprocessing
- Missing value imputation
- Outlier handling
- Scaling using RobustScaler

### ✅ Step 4: Feature Engineering
- Generates 17 new engineered features  
- Total features: **10 → 27**

### ✅ Step 5: Data Quality Checks
- NaN check
- Infinite value check

### ✅ Step 6: Class Balancing
- Uses **SMOTE** to balance the dataset

### ✅ Step 7: Model Training
Trains multiple ML models:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

---

# 📌 Training Output (Actual Run Results)

Below is the real training pipeline output:

CREDIT RISK MODEL - TRAINING PIPELINE
Loading data...
Loaded: (16714, 11)
Target: {0: 8357, 1: 8357}

Splitting data...
Train: (13371, 10), Test: (3343, 10)

Preprocessing (missing values, outliers, scaling)...
✓ Preprocessed

Feature engineering (creating 17 new features)...
Original: 10 features -> Engineered: 27 features

Data quality check...
X_train NaN: 0
X_train Inf: 0
X_test NaN: 0
X_test Inf: 0
✓ Data is clean and ready

Applying SMOTE for class imbalance...
Before SMOTE: {1: 6686, 0: 6685}
After SMOTE: {1: 6686, 0: 6686}

# 🏆 Model Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7777 | 0.7843 | 0.7660 | 0.7751 | 0.8573 |
| Random Forest | 0.7762 | 0.7778 | 0.7732 | 0.7755 | 0.8510 |
| XGBoost | 0.7637 | 0.7751 | 0.7427 | 0.7586 | 0.8477 |
| **LightGBM (Best)** | **0.7816** | **0.7888** | **0.7690** | **0.7788** | **0.8583** |

✅ **Best Model Selected: LightGBM**  
📌 Best ROC-AUC: **0.8583**
## Why LightGBM?
- This project uses **LightGBM (Light Gradient Boosting Machine)** as the final selected model because it performed best among all trained models.

### 🔍 Key Reasons:

### ✅ 1. Best Performance
LightGBM gave the highest overall score in evaluation:

- **ROC-AUC: 0.8583 (Best)**
- **F1 Score: 0.7788 (Best)**

This means it was able to separate defaulters vs non-defaulters better than other models.

---

### ✅ 2. Handles Non-Linear Relationships
Credit risk data is not purely linear.  
Factors like:

- late payments
- debt ratio
- credit utilization
- income

combine in complex ways.

LightGBM handles these **non-linear feature interactions** better than Logistic Regression.

---

### ✅ 3. Works Very Well with Tabular Financial Data
LightGBM is one of the best algorithms for structured/tabular datasets like:

- banking datasets
- loan datasets
- insurance datasets

That’s why it is widely used in real-world finance companies.

---

### ✅ 4. Fast and Efficient Training
Compared to XGBoost, LightGBM is:

- faster
- memory efficient
- scalable for large datasets

Even if dataset grows in future, the model can still train efficiently.

---


---

# 📦 Saved Artifacts

After training, the following artifacts are saved:

models/saved_models/
├── best_model.pkl
├── preprocessor.pkl
└── feature_engineer.pkl


These artifacts are loaded automatically when the API starts.

---

# 🌐 FastAPI API Endpoints

Once deployed, the API supports:

### ✅ Root
**GET /**
Returns API info.

### ✅ Health Check
**GET /health**
Checks if API and model are loaded.

### ✅ Single Prediction
**POST /predict**
Predict default risk for one applicant.

### ✅ Batch Prediction
**POST /batch_predict**
Predict default risk for multiple applicants.

---

## 🧪 Example Request (Single Prediction)

```json
{
  "rev_util": 0.45,
  "age": 35,
  "late_30_59": 0,
  "debt_ratio": 0.35,
  "monthly_inc": 5000,
  "open_credit": 8,
  "late_90": 0,
  "real_estate": 1,
  "late_60_89": 0,
  "dependents": 2
}
Output:
{
  "prediction": 0,
  "prediction_label": "No Default",
  "probability": 0.1234,
  "probability_no_default": 0.8766,
  "probability_default": 0.1234,
  "risk_level": "Low",
  "reason_codes": [
    "No history of severe delinquency",
    "Low credit utilization (45.0%)"
  ],
  "timestamp": "2024-02-01T10:30:00.123456"
}
```
▶️ Run API Locally
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Swagger UI:

http://127.0.0.1:8000/docs
🐳 Docker Deployment
Build Docker Image
docker build -t credit-risk-api .
Run Container
docker run -p 8000:8000 credit-risk-api
Docs:

http://localhost:8000/docs
🐳 Docker Compose Deployment
docker compose -f docker/docker-compose.yml up --build
Stop:

docker compose -f docker/docker-compose.yml down
🚀 Railway Deployment
This project is deployed on Railway.



API URL: https://<your-service>.up.railway.app

Swagger Docs: https://<your-service>.up.railway.app/docs
