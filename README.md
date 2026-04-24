# 🏦 LoanIQ — Loan Approval Prediction System

> An end-to-end Machine Learning system that predicts whether a loan application will be approved or rejected, based on applicant financial and personal details.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)

🔗 Live Demo: https://loan-approval-ml-kaph.onrender.com/app

---

## 📋 Project Overview

LoanIQ is a production-ready ML system built with an industry-standard structure. It trains multiple classifiers on the Kaggle Loan Prediction dataset, selects the best performing model, and exposes it through a FastAPI REST endpoint. A clean HTML/CSS/JS frontend lets users input their details and receive an instant approval prediction.

---

## ✨ Features

- **Multi-model training** — Logistic Regression, Random Forest, XGBoost
- **Automated model selection** by ROC-AUC
- **Sklearn Pipeline** for leak-free preprocessing (imputation + encoding + scaling)
- **FastAPI backend** with Pydantic validation and auto-generated docs (`/docs`)
- **Beautiful frontend** — no framework, pure HTML/CSS/JS
- **Dockerised** — single command to run everything
- **GitHub-ready** structure with `.gitignore`, `README.md`, and clean code

---

## 🗂️ Project Structure

```
loan-approval-ml/
│
├── data/                        # Raw & processed CSVs (not committed)
│   └── train.csv                ← place Kaggle dataset here
│
├── notebooks/
│   └── eda.py                   # Exploratory data analysis script
│
├── src/
│   ├── data_preprocessing.py    # Load, clean, encode, split, save preprocessor
│   ├── train.py                 # Train models, evaluate, save best
│   ├── predict.py               # Reusable prediction pipeline
│   └── utils.py                 # Shared helpers (eval, save, load)
│
├── models/                      # Saved artifacts (auto-created on training)
│   ├── preprocessor.pkl
│   ├── best_model.pkl
│   ├── model_meta.json
│   └── metrics.json
│
├── api/
│   └── main.py                  # FastAPI application
│
├── frontend/
│   └── index.html               # UI (no framework, pure HTML/CSS/JS)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/AkhilNyathani/loan-approval-ml.git
cd loan-approval-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Linux / Mac
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Go to [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) and download `train.csv`. Place it inside the `data/` folder:

```
loan-approval-ml/
└── data/
    └── train.csv   ← here
```

---

## 🚀 Running the Project

### Step 1 — (Optional) Explore the data

```bash
python notebooks/eda.py
```

### Step 2 — Train the models

```bash
python src/train.py
```

This will:
- Preprocess the data and save `models/preprocessor.pkl`
- Train Logistic Regression, Random Forest, and XGBoost
- Save the best model to `models/best_model.pkl`
- Print evaluation metrics for all models

### Step 3 — Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4 — Open the Frontend

Open `frontend/index.html` directly in your browser, **or** visit:

```
http://localhost:8000/app
```

The API docs are available at: `http://localhost:8000/docs`

---

## 🐳 Docker

### Build & Run

```bash
# Build the image
docker build -t loan-approval-ml .

# Run the container (mount pre-trained models)
# Linux / Mac
docker run -p 8000:8000 -v $(pwd)/models:/app/models loan-approval-ml

# Windows (PowerShell)
docker run -p 8000:8000 -v ${PWD}/models:/app/models loan-approval-ml
```

> **Note:** Train the model locally first, then mount the `models/` directory into the container. Alternatively, exec into the container and run `python src/train.py` (requires mounting `data/` too).

---

## 📡 API Reference

### `GET /`  — Health Check

```bash
curl http://localhost:8000/
```

**Response:**
```json
{"status": "ok", "service": "Loan Approval Prediction API", "version": "1.0.0"}
```

---

### `POST /predict`  — Predict Loan Approval

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Approved",
  "probability": 0.8721,
  "message": "Congratulations! Based on your profile, your loan is likely to be approved."
}
```

---

## 📊 Model Performance

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | ~80%     | ~0.83   |
| Random Forest       | ~82%     | ~0.87   |
| XGBoost             | **~83%** | **~0.88** |

*Exact numbers depend on your train/test split. Run `python src/train.py` to see live results.*

---

## 🔮 Future Improvements

- [ ] SHAP explainability for each prediction
- [ ] Hyperparameter tuning with Optuna
- [ ] MLflow experiment tracking
- [ ] User authentication & prediction history
- [ ] Deploy to Render / Railway / HuggingFace Spaces
- [ ] CI/CD with GitHub Actions

---

## 🛠️ Tech Stack

| Layer        | Technology                    |
|--------------|-------------------------------|
| Language     | Python 3.10+                  |
| ML           | Scikit-learn, XGBoost         |
| Data         | Pandas, NumPy                 |
| API          | FastAPI + Uvicorn             |
| Validation   | Pydantic v2                   |
| Frontend     | HTML5, CSS3, Vanilla JS       |
| Deployment   | Docker                        |

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built with ❤️ as a production-grade ML portfolio project.*
