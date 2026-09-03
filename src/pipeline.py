"""
pipeline.py
-----------
End-to-end pipeline for the Explainable Loan Rejection Assistant.

Flow:
    1. Receive raw applicant input (dict)
    2. Preprocess into model-ready format
    3. Predict approval/rejection
    4. If rejected: SHAP + DiCE + grounded Gemini explanation
    5. Attach live fairness audit summary
    6. Optionally score explanation faithfulness
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config_loader import load_config
from counterfactuals import (
    build_dice_explainer,
    format_counterfactuals,
    generate_counterfactuals,
    get_feature_ranges,
)
from data import DEFAULT_DATA_PATH, run_pipeline as load_data
from evaluation.faithfulness import score_faithfulness
from evaluation.template_explainer import applicant_local_explanation
from explainer import (
    build_explanation_prompt,
    build_explainer,
    build_grounded_explanation_prompt,
    get_shap_values,
)
from fairness import run_full_audit
from llm import generate_explanation
from logging_utils import get_logger
from model import DEFAULT_MODEL_PATH, load_model

logger = get_logger("pipeline")

GRADE_REVERSE = {6: "A", 5: "B", 4: "C", 3: "D", 2: "E", 1: "F", 0: "G"}

INTENT_COLS = [
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
]

OWNERSHIP_COLS = [
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
]


def load_artifacts(data_path=DEFAULT_DATA_PATH, model_path=DEFAULT_MODEL_PATH):
    """Load model, explainers, and run fairness audit once at startup."""
    logger.info("Loading artifacts...")
    cfg = load_config()

    X_train, X_test, y_train, y_test, X, y = load_data(data_path)
    model = load_model(model_path)
    explainer = build_explainer(
        model, X_train, background_size=cfg.get("shap", {}).get("background_size", 100)
    )
    dice_exp, _ = build_dice_explainer(model, X_train, y_train)

    predictions = model.predict(X_test)
    fairness_audit = run_full_audit(X_test, y_test, predictions)

    logger.info("Artifacts ready (age disparity=%.1f%%)", 100 * fairness_audit["age_disparity"])

    return {
        "model": model,
        "explainer": explainer,
        "dice_exp": dice_exp,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": X_train.columns.tolist(),
        "fairness_audit": fairness_audit,
        "config": cfg,
    }


def preprocess_input(input_dict, feature_cols):
    """Convert raw form input into a model-ready DataFrame row."""
    grade_map = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}
    row = {col: 0 for col in feature_cols}

    row["person_age"] = input_dict.get("person_age", 30)
    row["person_income"] = input_dict.get("person_income", 50000)
    row["person_emp_length"] = input_dict.get("person_emp_length", 5)
    row["loan_amnt"] = input_dict.get("loan_amnt", 10000)
    row["loan_int_rate"] = input_dict.get("loan_int_rate", 10.0)
    row["loan_percent_income"] = input_dict.get("loan_percent_income", 0.2)
    row["cb_person_default_on_file"] = input_dict.get("cb_person_default_on_file", 0)
    row["cb_person_cred_hist_length"] = input_dict.get("cb_person_cred_hist_length", 2)

    grade_str = input_dict.get("loan_grade", "C")
    row["loan_grade"] = grade_map.get(str(grade_str).upper(), 4)

    ownership = str(input_dict.get("person_home_ownership", "RENT")).upper()
    ownership_col = f"person_home_ownership_{ownership}"
    if ownership_col in row:
        row[ownership_col] = 1

    intent = str(input_dict.get("loan_intent", "PERSONAL")).upper()
    intent_col = f"loan_intent_{intent}"
    if intent_col in row:
        row[intent_col] = 1

    return pd.DataFrame([row])[feature_cols].astype(float)


def run_application(input_dict, artifacts, llm_mode: str | None = None):
    """
    Process one loan application end-to-end.

    llm_mode: "grounded" (default, paper method) | "free" | "off"
    """
    model = artifacts["model"]
    explainer = artifacts["explainer"]
    dice_exp = artifacts["dice_exp"]
    X_train = artifacts["X_train"]
    feature_cols = artifacts["feature_cols"]
    cfg = artifacts.get("config") or load_config()
    mode = llm_mode or cfg.get("llm", {}).get("mode", "grounded")

    applicant_df = preprocess_input(input_dict, feature_cols)
    logger.info("Processing application")

    prediction = model.predict(applicant_df)[0]
    probability = model.predict_proba(applicant_df)[0]
    approval_prob = float(probability[0])
    rejection_prob = float(probability[1])
    decision = "APPROVED" if prediction == 0 else "REJECTED"

    fairness = artifacts.get("fairness_audit") or {}
    result = {
        "decision": decision,
        "approval_prob": approval_prob,
        "rejection_prob": rejection_prob,
        "shap_explanation": None,
        "counterfactuals": None,
        "llm_prompt": None,
        "explanation": None,
        "llm_error": None,
        "explanation_source": None,
        "llm_mode": mode,
        "faithfulness": None,
        "fairness": {
            "note": fairness.get("note"),
            "gender_note": fairness.get("gender_note"),
            "age_disparity": fairness.get("age_disparity"),
            "passed": fairness.get("passed"),
            "threshold": fairness.get("threshold", 0.10),
            "groups": fairness.get("groups", []),
        },
    }

    if prediction == 0:
        result["message"] = (
            "Congratulations! Your loan application has been approved. "
            f"Approval confidence: {approval_prob:.1%}"
        )
        return result

    shap_vals = get_shap_values(explainer, applicant_df)[0]
    shap_df = pd.DataFrame({
        "Feature": feature_cols,
        "Value": applicant_df.iloc[0].values,
        "SHAP": shap_vals,
        "Direction": ["+ Risk" if v > 0 else "- Risk" for v in shap_vals],
    }).sort_values("SHAP", ascending=False)
    result["shap_explanation"] = shap_df

    n_cf = cfg.get("dice", {}).get("n_counterfactuals", 3)
    feature_ranges = get_feature_ranges(X_train, applicant_df.iloc[0])
    cf_result = generate_counterfactuals(
        dice_exp, applicant_df, feature_ranges, n=n_cf
    )
    _, suggestion_text = format_counterfactuals(cf_result, applicant_df.iloc[0])
    result["counterfactuals"] = suggestion_text

    cf_features = list(feature_ranges.keys()) if suggestion_text else []

    if mode == "grounded":
        prompt = build_grounded_explanation_prompt(
            applicant_df.iloc[0], shap_df, suggestion_text
        )
    else:
        prompt = build_explanation_prompt(applicant_df.iloc[0], shap_df)
        if suggestion_text:
            prompt = (
                f"{prompt}\n\nACTIONABLE CHANGES SUGGESTED BY THE MODEL:\n"
                f"{suggestion_text}\n"
                f"Incorporate these into your suggestions where realistic."
            )
    result["llm_prompt"] = prompt

    if mode != "off":
        try:
            result["explanation"] = generate_explanation(
                prompt,
                model=cfg.get("llm", {}).get("model", "gemini-3.6-flash"),
                retries=2,
            )
            result["explanation_source"] = "gemini"
            result["faithfulness"] = score_faithfulness(
                result["explanation"], shap_df, cf_features=cf_features
            )
        except Exception as e:
            logger.warning("Gemini explanation skipped: %s", e)
            result["llm_error"] = str(e)
            result["explanation"] = applicant_local_explanation(
                shap_df, suggestion_text
            )
            result["explanation_source"] = "template"
            result["faithfulness"] = score_faithfulness(
                result["explanation"], shap_df, cf_features=cf_features
            )

    return result


if __name__ == "__main__":
    artifacts = load_artifacts()
    model = artifacts["model"]
    X_test = artifacts["X_test"]
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    rejected = np.where(preds == 1)[0]
    idx = int(rejected[np.argmin(probs[rejected])])
    row = X_test.iloc[idx]

    ownership = "MORTGAGE"
    for col in OWNERSHIP_COLS:
        if row.get(col, 0) == 1:
            ownership = col.replace("person_home_ownership_", "")
            break

    intent = "DEBTCONSOLIDATION"
    for col in INTENT_COLS:
        if row.get(col, 0) == 1:
            intent = col.replace("loan_intent_", "")
            break

    test_input = {
        "person_age": float(row["person_age"]),
        "person_income": float(row["person_income"]),
        "person_emp_length": float(row["person_emp_length"]),
        "loan_grade": GRADE_REVERSE[int(row["loan_grade"])],
        "loan_amnt": float(row["loan_amnt"]),
        "loan_int_rate": float(row["loan_int_rate"]),
        "loan_percent_income": float(row["loan_percent_income"]),
        "cb_person_default_on_file": int(row["cb_person_default_on_file"]),
        "cb_person_cred_hist_length": float(row["cb_person_cred_hist_length"]),
        "person_home_ownership": ownership,
        "loan_intent": intent,
    }
    result = run_application(test_input, artifacts)
    print(f"Decision: {result['decision']}")
    print(f"Fairness: {result['fairness']}")
    if result.get("faithfulness"):
        print(f"Faithfulness: {result['faithfulness']}")
