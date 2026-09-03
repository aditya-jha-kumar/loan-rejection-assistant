# Explainable Loan Rejection Assistant

Predicts loan approval or rejection with XGBoost, then explains rejections using SHAP, DiCE counterfactuals, and an optional Gemini letter.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the [Kaggle credit risk dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) at `data/loan_dataset.csv` (32,581 rows; columns such as `person_age`, `person_income`, `loan_status`).

Train and save the model:

```bash
cd src
python model.py
```

Optional: copy `.env.example` to `.env` and set `GEMINI_API_KEY` for applicant-facing explanations.

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, enter an application, and review the decision, risk drivers, and suggested changes.
