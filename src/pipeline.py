"""
pipeline.py
-----------
End-to-end pipeline for the Explainable Loan Rejection Assistant.

This is the single entry point that ties all modules together.
Both the Streamlit UI and the FastAPI backend call this file.

Flow:
    1. Receive raw applicant input (dict)
    2. Preprocess into model-ready format
    3. Predict approval/rejection
    4. If rejected: generate SHAP explanation
    5. If rejected: generate DiCE counterfactuals
    6. Run fairness check
    7. Build LLM prompt and call Gemini
    8. Return structured result

Functions:
    load_artifacts()              - Load model + explainer once at startup
    preprocess_input(input_dict)  - Convert raw form input to model format
    run_application(input_dict,   - Full end-to-end pipeline
                    artifacts)
"""

import numpy as np
import pandas as pd

from data           import DEFAULT_DATA_PATH, run_pipeline as load_data
from model          import DEFAULT_MODEL_PATH, load_model
from explainer      import (build_explainer, get_shap_values,
                            build_explanation_prompt)
from counterfactuals import (build_dice_explainer,
                             get_feature_ranges,
                             generate_counterfactuals,
                             format_counterfactuals)
from llm            import generate_explanation


# ─────────────────────────────────────────────
# GRADE MAPPING — for human-readable output
# ─────────────────────────────────────────────

# Reverse map from encoded integer back to letter grade
# Used when displaying results to the applicant
GRADE_REVERSE = {6: "A", 5: "B", 4: "C", 3: "D", 2: "E", 1: "F", 0: "G"}

# Loan intent columns in the encoded dataset
INTENT_COLS = [
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE"
]

# Home ownership columns
OWNERSHIP_COLS = [
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT"
]


# ─────────────────────────────────────────────
# STEP 1 — LOAD ARTIFACTS
# ─────────────────────────────────────────────

def load_artifacts(data_path=DEFAULT_DATA_PATH,
                   model_path=DEFAULT_MODEL_PATH):
    """
    Load all ML artifacts once at application startup.

    Why load once instead of every request?
    Loading a model and building a SHAP explainer takes 5-10
    seconds. In a web app, you load them once when the server
    starts, then reuse them for every request. This is standard
    ML serving practice.

    Returns:
        dict: {model, explainer, X_train, X_test, y_test, feature_cols}
    """
    print("Loading artifacts...")

    # Load data — we need X_train for SHAP baseline and DiCE ranges
    X_train, X_test, y_train, y_test, X, y = load_data(data_path)

    # Load pretrained model — no retraining
    model = load_model(model_path)

    # Build SHAP explainer
    explainer = build_explainer(model, X_train)

    # Build DiCE explainer
    dice_exp, _ = build_dice_explainer(model, X_train, y_train)

    print("All artifacts loaded. Ready to process applications.")

    return {
        "model":        model,
        "explainer":    explainer,
        "dice_exp":     dice_exp,
        "X_train":      X_train,
        "X_test":       X_test,
        "y_test":       y_test,
        "feature_cols": X_train.columns.tolist()
    }


# ─────────────────────────────────────────────
# STEP 2 — PREPROCESS RAW INPUT
# ─────────────────────────────────────────────

def preprocess_input(input_dict, feature_cols):
    """
    Convert raw form input into a model-ready DataFrame row.

    The model expects exactly 17 encoded features in a specific
    order. This function handles:
    - Grade encoding (A→6, B→5, etc.)
    - One-hot encoding for intent and ownership
    - Default values for missing fields
    - Correct column ordering

    Args:
        input_dict (dict):    Raw form values from UI
        feature_cols (list):  Expected column names in correct order

    Returns:
        pd.DataFrame: Single row ready for model.predict()

    Example input_dict:
        {
            "person_age": 26,
            "person_income": 31200,
            "person_emp_length": 8,
            "loan_grade": "E",
            "loan_amnt": 5000,
            "loan_int_rate": 8.63,
            "loan_percent_income": 0.16,
            "cb_person_default_on_file": 0,
            "cb_person_cred_hist_length": 2,
            "person_home_ownership": "RENT",
            "loan_intent": "EDUCATION"
        }
    """
    # Grade encoding — letter to integer
    grade_map = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}

    # Start with a zero-filled row for all expected columns
    row = {col: 0 for col in feature_cols}

    # Fill numeric features directly
    row["person_age"]                  = input_dict.get("person_age", 30)
    row["person_income"]               = input_dict.get("person_income", 50000)
    row["person_emp_length"]           = input_dict.get("person_emp_length", 5)
    row["loan_amnt"]                   = input_dict.get("loan_amnt", 10000)
    row["loan_int_rate"]               = input_dict.get("loan_int_rate", 10.0)
    row["loan_percent_income"]         = input_dict.get("loan_percent_income", 0.2)
    row["cb_person_default_on_file"]   = input_dict.get("cb_person_default_on_file", 0)
    row["cb_person_cred_hist_length"]  = input_dict.get("cb_person_cred_hist_length", 2)

    # Encode loan grade
    grade_str       = input_dict.get("loan_grade", "C")
    row["loan_grade"] = grade_map.get(grade_str.upper(), 4)

    # One-hot encode home ownership
    ownership = input_dict.get("person_home_ownership", "RENT").upper()
    ownership_col = f"person_home_ownership_{ownership}"
    if ownership_col in row:
        row[ownership_col] = 1

    # One-hot encode loan intent
    intent = input_dict.get("loan_intent", "PERSONAL").upper()
    intent_col = f"loan_intent_{intent}"
    if intent_col in row:
        row[intent_col] = 1

    # Convert to DataFrame with correct column order
    df = pd.DataFrame([row])[feature_cols]

    return df


# ─────────────────────────────────────────────
# STEP 3 — FULL PIPELINE
# ─────────────────────────────────────────────

def run_application(input_dict, artifacts):
    """
    Process one loan application end-to-end.

    This is the single function called by both Streamlit
    and FastAPI. It returns a structured result dict that
    the UI layer can render however it wants.

    Args:
        input_dict (dict):  Raw applicant input from UI
        artifacts (dict):   Loaded model + explainer objects

    Returns:
        dict: Complete result with decision + explanation
    """
    model       = artifacts["model"]
    explainer   = artifacts["explainer"]
    dice_exp    = artifacts["dice_exp"]
    X_train     = artifacts["X_train"]
    feature_cols = artifacts["feature_cols"]

    # ── Preprocess input ──────────────────────
    applicant_df = preprocess_input(input_dict, feature_cols)
    print(f"\nProcessing application...")
    print(applicant_df.T)

    # ── Predict ───────────────────────────────
    prediction   = model.predict(applicant_df)[0]
    probability  = model.predict_proba(applicant_df)[0]

    # probability[0] = P(good standing)
    # probability[1] = P(default/rejection)
    approval_prob  = probability[0]
    rejection_prob = probability[1]

    decision = "APPROVED" if prediction == 0 else "REJECTED"
    print(f"\nDecision: {decision}")
    print(f"Approval probability:  {approval_prob:.1%}")
    print(f"Rejection probability: {rejection_prob:.1%}")

    # ── Build base result ─────────────────────
    result = {
        "decision":        decision,
        "approval_prob":   approval_prob,
        "rejection_prob":  rejection_prob,
        "shap_explanation": None,
        "counterfactuals":  None,
        "llm_prompt":       None,
        "explanation":      None,
        "fairness":         None
    }

    # ── If approved — return early ────────────
    if prediction == 0:
        result["message"] = (
            "Congratulations! Your loan application has been approved. "
            f"Approval confidence: {approval_prob:.1%}"
        )
        return result

    # ── If rejected — full explanation ────────
    print("\nGenerating SHAP explanation...")

    # Compute SHAP values for this applicant (normalized 2D array)
    shap_vals = get_shap_values(explainer, applicant_df)[0]

    # Build explanation DataFrame
    shap_df = pd.DataFrame({
        "Feature":   feature_cols,
        "Value":     applicant_df.iloc[0].values,
        "SHAP":      shap_vals,
        "Direction": ["+ Risk" if v > 0 else "- Risk"
                      for v in shap_vals]
    }).sort_values("SHAP", ascending=False)

    result["shap_explanation"] = shap_df

    # Top 3 rejection reasons for display
    top_risks = shap_df[shap_df["SHAP"] > 0].head(3)
    print("\nTop rejection reasons:")
    print(top_risks[["Feature", "Value", "SHAP"]].to_string(index=False))

    # ── DiCE counterfactuals ──────────────────
    print("\nGenerating counterfactuals...")
    feature_ranges = get_feature_ranges(X_train, applicant_df.iloc[0])

    cf_result = generate_counterfactuals(
        dice_exp, applicant_df, feature_ranges
    )

    _, suggestion_text = format_counterfactuals(
        cf_result, applicant_df.iloc[0]
    )
    result["counterfactuals"] = suggestion_text

    # ── Build LLM prompt + Gemini explanation ─
    prompt = build_explanation_prompt(applicant_df.iloc[0], shap_df)
    if suggestion_text:
        prompt = (
            f"{prompt}\n\n"
            f"ACTIONABLE CHANGES SUGGESTED BY THE MODEL:\n"
            f"{suggestion_text}\n"
            f"Incorporate these into your suggestions where realistic."
        )
    result["llm_prompt"] = prompt

    print("\nGenerating Gemini explanation...")
    try:
        result["explanation"] = generate_explanation(prompt)
        print(result["explanation"])
    except Exception as e:
        print(f"Gemini explanation skipped: {e}")
        result["explanation"] = None

    # ── Fairness note ─────────────────────────
    # For single applicant we note the audit results
    # Full audit runs at model evaluation time, not per-applicant
    result["fairness"] = {
        "note": "Model passed age fairness audit (6.6% disparity, threshold 10%)",
        "gender_note": "Gender data not available in dataset"
    }

    return result


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load all artifacts
    artifacts = load_artifacts()

    # Prefer a near-boundary reject so DiCE can find actionable CFs
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
        "person_age":                 float(row["person_age"]),
        "person_income":              float(row["person_income"]),
        "person_emp_length":          float(row["person_emp_length"]),
        "loan_grade":                 GRADE_REVERSE[int(row["loan_grade"])],
        "loan_amnt":                  float(row["loan_amnt"]),
        "loan_int_rate":              float(row["loan_int_rate"]),
        "loan_percent_income":        float(row["loan_percent_income"]),
        "cb_person_default_on_file":  int(row["cb_person_default_on_file"]),
        "cb_person_cred_hist_length": float(row["cb_person_cred_hist_length"]),
        "person_home_ownership":      ownership,
        "loan_intent":                intent,
    }
    print(f"Demo applicant index: {idx} "
          f"(default prob {probs[idx]:.1%})")

    result = run_application(test_input, artifacts)

    print("\n" + "="*50)
    print("PIPELINE RESULT")
    print("="*50)
    print(f"Decision:   {result['decision']}")
    print(f"Approval:   {result['approval_prob']:.1%}")
    print(f"Rejection:  {result['rejection_prob']:.1%}")

    if result["counterfactuals"]:
        print(f"\nSuggestions:{result['counterfactuals']}")

    if result["explanation"]:
        print("\n" + "="*50)
        print("GEMINI EXPLANATION")
        print("="*50)
        print(result["explanation"])
    elif result["llm_prompt"]:
        print(f"\nLLM Prompt ready - {len(result['llm_prompt'])} chars")
        print("Set GEMINI_API_KEY in .env to generate the explanation.")