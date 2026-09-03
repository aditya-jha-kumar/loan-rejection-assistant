"""Unit tests for preprocessing, fairness, and faithfulness metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from evaluation.faithfulness import features_mentioned, score_faithfulness
from fairness import audit_age_fairness
from pipeline import preprocess_input


FEATURE_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
]


def test_preprocess_one_hot_and_grade():
    df = preprocess_input(
        {
            "person_age": 30,
            "person_income": 50000,
            "person_emp_length": 3,
            "loan_grade": "B",
            "loan_amnt": 8000,
            "loan_int_rate": 11.0,
            "loan_percent_income": 0.16,
            "cb_person_default_on_file": 0,
            "cb_person_cred_hist_length": 4,
            "person_home_ownership": "RENT",
            "loan_intent": "EDUCATION",
        },
        FEATURE_COLS,
    )
    assert list(df.columns) == FEATURE_COLS
    assert df.iloc[0]["loan_grade"] == 5
    assert df.iloc[0]["person_home_ownership_RENT"] == 1
    assert df.iloc[0]["loan_intent_EDUCATION"] == 1
    assert df.iloc[0]["loan_intent_PERSONAL"] == 0


def test_local_explanation_mentions_risks():
    from evaluation.template_explainer import applicant_local_explanation

    shap_df = pd.DataFrame({
        "Feature": ["person_income", "loan_amnt", "person_age"],
        "Value": [20000, 15000, 22],
        "SHAP": [0.4, 0.2, -0.1],
    })
    text = applicant_local_explanation(shap_df, "  - Reduce loan request")
    assert "income" in text.lower()
    assert "Reduce loan request" in text


def test_faithfulness_precision_and_coverage():
    shap_df = pd.DataFrame({
        "Feature": ["person_income", "loan_amnt", "person_age"],
        "Value": [20000, 15000, 22],
        "SHAP": [0.4, 0.2, -0.1],
    })
    text = (
        "Your annual income and loan amount were the main concerns. "
        "We suggest reducing the amount requested."
    )
    scores = score_faithfulness(text, shap_df, cf_features=["loan_amnt"], top_k=2)
    assert scores["coverage"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["hallucination_rate"] == 0.0


def test_faithfulness_flags_ungrounded_mention():
    shap_df = pd.DataFrame({
        "Feature": ["person_income", "loan_amnt"],
        "Value": [20000, 15000],
        "SHAP": [0.4, 0.1],
    })
    text = "Your medical purpose and credit history were problematic."
    scores = score_faithfulness(text, shap_df, top_k=2)
    mentioned = features_mentioned(text)
    assert "loan_intent_MEDICAL" in mentioned or "cb_person_cred_hist_length" in mentioned
    assert scores["precision"] < 1.0


def test_age_fairness_disparity_math():
    X = pd.DataFrame({"person_age": [25] * 50 + [40] * 50 + [60] * 50})
    y = pd.Series([0] * 150)
    # Approve all young/middle, reject all seniors -> large disparity
    preds = [0] * 100 + [1] * 50
    results, disparity = audit_age_fairness(X, y, preds)
    assert disparity == pytest.approx(1.0, abs=1e-6)
    assert len(results) == 3


def test_clean_data_imputes_missing_employment():
    from data import clean_data

    df = pd.DataFrame({
        "person_age": [25, 30, 144],
        "person_emp_length": [3.0, None, 2.0],
        "loan_int_rate": [10.0, 11.0, None],
        "loan_status": [0, 1, 0],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned["person_emp_length"].isna().sum() == 0
    assert cleaned["loan_int_rate"].isna().sum() == 0


def test_shap_explainer_on_trained_model():
    from data import DEFAULT_DATA_PATH
    from explainer import build_explainer, get_shap_values
    from model import DEFAULT_MODEL_PATH, load_model
    from data import run_pipeline

    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_DATA_PATH.exists():
        pytest.skip("Trained model or dataset not available")

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    model = load_model(DEFAULT_MODEL_PATH)
    explainer = build_explainer(model, X_train, background_size=20)
    shap_vals = get_shap_values(explainer, X_test.iloc[:2])
    assert shap_vals.shape == (2, X_test.shape[1])
    assert np.isfinite(shap_vals).all()
