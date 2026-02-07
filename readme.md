# Credit Risk Prediction - Production ML System

A complete end-to-end machine learning system for predicting credit default risk. This project demonstrates best practices in data science, MLOps, and production deployment.

## Project Overview

This system predicts whether a borrower will experience serious delinquency within two years using a comprehensive ML pipeline with:
- **Multiple ML algorithms** (Logistic Regression, Random Forest, XGBoost, LightGBM)
- **Advanced feature engineering** with domain knowledge
- **Model interpretability** using SHAP values
- **Production-ready API** with FastAPI
- **Docker containerization** for easy deployment
- **Complete MLOps practices**

## Dataset

- **Total Records**: ~17,000 credit applications
- **Features**: 10 predictors (financial metrics and personal attributes)
- **Target**: Binary classification (0 = No delinquency, 1 = Delinquency)
- **Key Features**:
  - `rev_util`: Revolving credit utilization ratio
  - `age`: Age of borrower
  - `debt_ratio`: Debt-to-income ratio
  - `monthly_inc`: Monthly income
  - Late payment indicators (30-59, 60-89, 90+ days)

##  Project Structure

```
credit-risk-model/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       └── helpers.py
├── api/
│   ├── main.py (FastAPI)
│   ├── schemas.py
│   └── middleware.py
├── models/
│   └── saved_models/
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/
│   ├── model_monitor.py
│   └── drift_detection.py
├── config/
│   └── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/swayamjaiswal7/credit-risk-model.git
cd credit-risk-model
```

2. **Create virtual environment**
```
python -m venv venv
 # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your data**
```bash
# Place your CSV file in data/raw/
cp your_credit_risk_data.csv data/raw/credit_risk_data.csv
```

## Usage

### 1. Exploratory Data Analysis

Generate comprehensive EDA report:

```bash
python 01_eda_report.py
```

**Outputs:**
- `target_distribution.png` - Class balance visualization
- `feature_distributions.png` - Feature histograms
- `correlation_matrix.png` - Feature correlations
- `outlier_boxplots.png` - Outlier detection
- `bivariate_distributions.png` - Feature vs target analysis

### 2. Run Complete Pipeline

Execute end-to-end training pipeline:

```bash
python main_pipeline.py
```

**Pipeline Steps:**
1. ✅ Data loading and validation
2. ✅ Preprocessing (missing values, outliers, scaling)
3. ✅ Feature engineering (17 new features)
4. ✅ Model training (4 algorithms with hyperparameter tuning)
5. ✅ Model evaluation and comparison
6. ✅ SHAP interpretability analysis
7. ✅ Model persistence

**Artifacts Generated:**
- `models/best_model.pkl` - Best performing model
- `models/preprocessor.pkl` - Fitted preprocessor
- `models/feature_engineer.pkl` - Feature transformer
- `plots/model_comparison.png` - Performance comparison
- `plots/roc_curves.png` - ROC curves
- `reports/interpretability/` - SHAP analysis

### 3. Deploy API with Docker

#### Option A: Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

#### Option B: Docker Only

```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models credit-risk-api
```

#### Option C: Local Development

```bash
# Run FastAPI directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test API

**Access Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Example API Call (Python):**

```python
import requests

# Single prediction
data = {
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

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

**Example Response:**

```json
{
  "prediction": 0,
  "probability": 0.1234,
  "risk_level": "Low",
  "reason_codes": [
    "No history of severe delinquency",
    "Low credit utilization (45.0%)",
    "Based on overall credit profile analysis"
  ],
  "timestamp": "2024-02-01T10:30:00"
}
```

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

##  Model Details

### Preprocessing Pipeline

1. **Missing Value Imputation**
   - Monthly income: Median imputation
   - Dependents: Median imputation

2. **Outlier Treatment**
   - Percentile capping (1st-99th percentile)
   - Prevents extreme value influence

3. **Feature Scaling**
   - RobustScaler for numerical features
   - Handles outliers better than StandardScaler

### Feature Engineering

**Aggregate Features (7):**
- Total late payments
- Severe delinquency indicator
- Total credit lines
- Real estate ratio
- Average late payment severity
- Dependents per income

**Interaction Features (6):**
- Financial stress (utilization × debt ratio)
- Risky borrower score
- Debt-delinquency risk
- Age-debt interaction
- Credit management score
- Income stability

**Binned Features (4):**
- Age groups
- Utilization buckets
- Debt categories
- Income brackets

### Models Trained

| Model | Description | Strengths |
|-------|-------------|-----------|
| **Logistic Regression** | Interpretable baseline | Fast, explainable, good for linear relationships |
| **Random Forest** | Ensemble of decision trees | Handles non-linearity, feature importance |
| **XGBoost** | Gradient boosting | High performance, handles imbalance well |
| **LightGBM** | Efficient gradient boosting | Fast training, memory efficient |

### Evaluation Metrics

Due to class imbalance, we focus on:
- **ROC-AUC**: Overall discriminative ability
- **Precision-Recall AUC**: Better for imbalanced data
- **F1-Score**: Balance of precision and recall
- **Confusion Matrix**: Detailed error analysis

### Model Interpretability

**SHAP (SHapley Additive exPlanations):**
- Global feature importance
- Local prediction explanations
- Feature interaction detection
- Reason code generation for compliance

## API Endpoints

### Health Check
```
GET /health
```
Returns API health status and model loading state.

### Single Prediction
```
POST /predict
```
Predict credit risk for a single application.

**Request Body:**
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
```

### Batch Prediction
```
POST /batch_predict
```
Predict multiple applications at once.

**Request Body:**
```json
{
  "instances": [
    {...application1...},
    {...application2...}
  ]
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific module
pytest tests/test_preprocessing.py -v
```

## Model Performance

Expected performance on test set:

| Metric | Logistic Regression | Random Forest | XGBoost | LightGBM |
|--------|-------------------|---------------|---------|----------|
| ROC-AUC | ~0.85 | ~0.87 | ~0.88 | ~0.88 |
| F1-Score | ~0.45 | ~0.50 | ~0.52 | ~0.52 |
| Precision | ~0.50 | ~0.55 | ~0.58 | ~0.58 |
| Recall | ~0.42 | ~0.47 | ~0.48 | ~0.48 |


## Security & Compliance

### Regulatory Compliance
- **Explainability**: SHAP values for all predictions
- **Reason Codes**: Human-readable explanations
- **Audit Trail**: Complete logging of predictions
- **Model Cards**: Documentation of model behavior

### Security Features
- Non-root Docker user
- Input validation with Pydantic
- Rate limiting ready
- Secure API design
- No hardcoded credentials

## Deployment

### Production Checklist

- [ ] Environment variables configured
- [ ] Model artifacts uploaded
- [ ] API authentication implemented
- [ ] Rate limiting enabled
- [ ] Monitoring dashboards configured
- [ ] Alerting rules set up
- [ ] Backup strategy defined
- [ ] CI/CD pipeline established

### Monitoring (Optional)

Enable monitoring with:

```bash
docker-compose --profile monitoring up
```

**Services:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Cloud Deployment

**AWS Example:**
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag credit-risk-api:latest <account>.dkr.ecr.<region>.amazonaws.com/credit-risk-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/credit-risk-api:latest

# Deploy to ECS
# Use provided task definition
```

## Documentation

- **API Documentation**: Auto-generated at `/docs` and `/redoc`
- **Model Card**: See `reports/model_card.md`
- **Architecture**: See `docs/architecture.md`
- **Contributing**: See `CONTRIBUTING.md`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file.

## Acknowledgments

- Dataset: Credit Risk Benchmark Dataset
- Libraries: scikit-learn, XGBoost, LightGBM, SHAP, FastAPI
- Inspiration: Production ML best practices

## Contact

For questions or issues:
- GitHub Issues: [Project Issues](https://github.com/swayamjaiswal7/credit-risk-model/issues)
- Email: jswayam341@gmail.com

---

**Built with best practices for production ML systems**