"""
explainer.py
------------
Generates SHAP-based explanations for individual loan decisions
and global feature importance for the Explainable Loan Rejection
Assistant.

Key concepts:
    Local explanation  - Why was THIS applicant rejected?
    Global explanation - Which features matter most overall?
    SHAP value         - How much each feature pushed the prediction
                         positive = toward default (rejection)
                         negative = toward good standing (approval)

Functions:
    build_explainer(model, X_train)     - Create SHAP TreeExplainer
    get_shap_values(explainer, X)       - Compute SHAP values
    explain_applicant(explainer,        - Local explanation for one
                      shap_values,        applicant
                      X_test, idx)
    plot_waterfall(explainer,           - Waterfall plot for paper
                   shap_values,
                   X_test, idx)
    plot_global_importance(shap_values, - Global importance bar chart
                           X_test)
    build_explanation_prompt(           - Build LLM prompt from SHAP
                      applicant_data,
                      shap_df)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATERFALL_PATH = ROOT / "models" / "waterfall.png"
DEFAULT_IMPORTANCE_PATH = ROOT / "models" / "global_importance.png"


# BUILD EXPLAINER

def _as_numeric_frame(X):
    """Coerce a feature matrix to a float DataFrame (stable for XGBoost/SHAP)."""
    if isinstance(X, pd.DataFrame):
        return X.astype(float)
    return pd.DataFrame(X).astype(float)


def build_explainer(model, X_train, background_size=100):
    """
    Create a SHAP TreeExplainer for the trained XGBoost model.

    Why TreeExplainer specifically?
    - Optimized for tree-based models (XGBoost, RandomForest)
    - Exact SHAP values, not approximations
    - 100x faster than model-agnostic explainers (KernelExplainer)

    XGBoost 2.1+/3.x stores categorical metadata that SHAP's interventional
    TreeSHAP (C extension) rejects. Prefer interventional when it works;
    otherwise fall back to tree_path_dependent (XGBoost native pred_contribs).

    Args:
        model:               Trained XGBoost model
        X_train (DataFrame): Training features for baseline calculation
        background_size:     Rows sampled for background (keeps SHAP fast)

    Returns:
        shap.TreeExplainer
    """
    n = min(background_size, len(X_train))
    background = _as_numeric_frame(shap.sample(X_train, n, random_state=42))

    explainer = None
    mode = "interventional"
    try:
        candidate = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
        )
        # Constructor can succeed while shap_values() still raises
        _to_array(candidate.shap_values(background.iloc[:1]))
        explainer = candidate
    except (NotImplementedError, ValueError, TypeError):
        mode = "tree_path_dependent"
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )

    expected = explainer.expected_value
    if expected is None:
        expected_str = "set on first explanation"
    else:
        if isinstance(expected, (list, np.ndarray)):
            expected = float(np.asarray(expected).ravel()[-1])
        else:
            expected = float(expected)
        expected_str = f"{expected:.4f}"

    print("Explainer created")
    print(f"SHAP mode: {mode}")
    print(f"Baseline (expected value): {expected_str}")
    print("Interpretation: without seeing any features, the model's")
    print(f"default prediction score is {expected_str}")
    return explainer


def _baseline(explainer):
    """Scalar baseline for binary classifiers (handles array expected_value)."""
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        return float(np.asarray(expected).ravel()[-1])
    return float(expected)


def _to_array(shap_values):
    """Normalize SHAP output to a 2D ndarray (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    if hasattr(shap_values, "values"):
        return np.asarray(shap_values.values)
    return np.asarray(shap_values)


# COMPUTE SHAP VALUES

def get_shap_values(explainer, X):
    """
    Compute SHAP values for a set of applicants.

    Output shape: (n_applicants, n_features)
    Each cell = how much that feature pushed that prediction
    Positive = pushed toward default (bad)
    Negative = pushed toward good standing (good)

    Args:
        explainer:     SHAP TreeExplainer
        X (DataFrame): Feature matrix to explain

    Returns:
        np.ndarray: SHAP values matrix
    """
    print(f"Computing SHAP values for {len(X)} applicants...")
    X_num = _as_numeric_frame(X)
    try:
        raw = explainer.shap_values(X_num)
    except NotImplementedError:
        # Last-resort: callable API / path-dependent explainer
        raw = explainer(X_num)
    shap_values = _to_array(raw)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    print(f"SHAP values shape: {shap_values.shape}")
    return shap_values


# LOCAL EXPLANATION

def explain_applicant(explainer, shap_values, X_test, idx):
    """
    Generate a local explanation for one applicant.

    Returns a sorted DataFrame showing which features pushed
    the prediction toward default (positive SHAP) or good
    standing (negative SHAP), ordered by impact magnitude.

    This is the core output that feeds into the LLM prompt.

    Args:
        explainer:          SHAP TreeExplainer
        shap_values:        Precomputed SHAP values matrix
        X_test (DataFrame): Test features
        idx (int):          Row index of the applicant to explain

    Returns:
        pd.DataFrame: Features sorted by SHAP impact
    """
    row_shap = np.asarray(shap_values[idx]).ravel()

    shap_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Value": X_test.iloc[idx].values,
        "SHAP": row_shap,
        "Direction": ["↑ Risk" if v > 0 else "↓ Risk" for v in row_shap],
    }).sort_values("SHAP", ascending=False)

    risk_factors = shap_df[shap_df["SHAP"] > 0].head(3)
    # Most protective = most negative SHAP (end of descending sort)
    protective_factors = shap_df[shap_df["SHAP"] < 0].tail(3).sort_values("SHAP")

    print(f"\n{'=' * 50}")
    print(f"EXPLANATION FOR APPLICANT #{idx}")
    print(f"{'=' * 50}")
    print("\nTop 3 risk factors (pushed toward default):")
    print(risk_factors[["Feature", "Value", "SHAP"]].to_string(index=False))
    print("\nTop 3 protective factors (pushed toward approval):")
    print(protective_factors[["Feature", "Value", "SHAP"]].to_string(index=False))

    return shap_df


# VISUALIZATIONS

def plot_waterfall(explainer, shap_values, X_test, idx,
                   save_path=DEFAULT_WATERFALL_PATH):
    """
    Generate a waterfall plot for one applicant's explanation.

    The waterfall plot shows:
    - Starting point: baseline (expected value)
    - Each feature's contribution (red = increases risk, blue = decreases)
    - Final prediction score

    Args:
        explainer:          SHAP TreeExplainer
        shap_values:        Precomputed SHAP values
        X_test (DataFrame): Test features
        idx (int):          Applicant index
        save_path (str):    Where to save the figure
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    shap.waterfall_plot(
        shap.Explanation(
            values=np.asarray(shap_values[idx]).ravel(),
            base_values=_baseline(explainer),
            data=X_test.iloc[idx].values,
            feature_names=X_test.columns.tolist(),
        ),
        show=False,
    )
    plt.title(f"Loan Decision Waterfall — Applicant #{idx}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Waterfall plot saved to {save_path}")


def plot_global_importance(shap_values, X_test,
                           save_path=DEFAULT_IMPORTANCE_PATH):
    """
    Generate a global feature importance bar chart.

    Shows mean absolute SHAP value per feature across all
    test applicants — which features drive decisions overall.

    Args:
        shap_values:        Precomputed SHAP values matrix
        X_test (DataFrame): Test features
        save_path (str):    Where to save the figure
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
    )
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Global importance plot saved to {save_path}")


# BUILD LLM PROMPT

def build_explanation_prompt(applicant_data, shap_df):
    """
    Convert SHAP output into a structured LLM prompt.

    The prompt injects three things:
    1. Role    — who the LLM is acting as
    2. Context — applicant details + SHAP risk factors
    3. Task    — what to generate and how

    Args:
        applicant_data (Series): One row from X_test
        shap_df (DataFrame):     SHAP explanation DataFrame

    Returns:
        str: Formatted prompt ready for LLM
    """
    top_risks = shap_df[shap_df["SHAP"] > 0].head(3)
    top_protection = shap_df[shap_df["SHAP"] < 0].tail(3).sort_values("SHAP")

    risk_text = "\n".join([
        f"  - {row['Feature']}: {row['Value']} "
        f"(risk contribution: +{row['SHAP']:.3f})"
        for _, row in top_risks.iterrows()
    ])

    protection_text = "\n".join([
        f"  - {row['Feature']}: {row['Value']} "
        f"(protective contribution: {row['SHAP']:.3f})"
        for _, row in top_protection.iterrows()
    ])

    income = applicant_data.get("person_income", "N/A")
    loan_amnt = applicant_data.get("loan_amnt", "N/A")
    pct_income = applicant_data.get("loan_percent_income", "N/A")

    income_str = f"${income:,.0f}" if isinstance(income, (int, float, np.integer, np.floating)) else str(income)
    loan_str = f"${loan_amnt:,.0f}" if isinstance(loan_amnt, (int, float, np.integer, np.floating)) else str(loan_amnt)
    pct_str = f"{pct_income:.1%}" if isinstance(pct_income, (int, float, np.integer, np.floating)) else str(pct_income)

    prompt = f"""You are a loan officer assistant at a bank. Your job is to
explain loan rejection decisions to applicants in clear, empathetic language.

APPLICANT PROFILE:
- Age: {applicant_data.get('person_age', 'N/A')}
- Annual Income: {income_str}
- Employment Length: {applicant_data.get('person_emp_length', 'N/A')} years
- Loan Amount Requested: {loan_str}
- Loan as % of Income: {pct_str}
- Loan Grade: {applicant_data.get('loan_grade', 'N/A')}
- Previous Default on File: {'Yes' if applicant_data.get('cb_person_default_on_file') == 1 else 'No'}

PRIMARY RISK FACTORS (reasons for rejection):
{risk_text}

POSITIVE FACTORS (working in applicant's favor):
{protection_text}

TASK:
Write a 3-paragraph response to this applicant:
1. Acknowledge the rejection with empathy
2. Explain the specific reasons in plain language (no technical jargon,
   no mention of SHAP, models, or algorithms)
3. Give 2-3 concrete, actionable suggestions for improving their
   application in the future

Keep the tone professional but human. The applicant is a real person
who needs clear guidance, not a data point.
"""
    return prompt


def build_grounded_explanation_prompt(applicant_data, shap_df,
                                      suggestion_text: str | None = None,
                                      top_k: int = 5):
    """
    Grounded prompting (paper method): the LLM may only cite features
    that appear in top SHAP risks or DiCE suggestions.

    This is designed to reduce unsupported feature mentions
    (hallucinations) while preserving coverage of true drivers.
    """
    base = build_explanation_prompt(applicant_data, shap_df)
    top_risks = (
        shap_df[shap_df["SHAP"] > 0]
        .sort_values("SHAP", ascending=False)
        .head(top_k)["Feature"]
        .tolist()
    )
    allowed = ", ".join(top_risks) if top_risks else "(none)"

    grounded_rules = f"""
GROUNDING CONSTRAINTS (mandatory):
- You may ONLY discuss these risk factors: {allowed}
- If actionable changes are provided below, you may also discuss those changes
- Do NOT invent other reasons (credit score agency reports, debt-to-income
  from external bureaus, marital status, gender, race, or unlisted features)
- If a factor is not listed above, do not mention it
"""
    if suggestion_text:
        grounded_rules += (
            f"\nACTIONABLE CHANGES SUGGESTED BY THE MODEL:\n"
            f"{suggestion_text}\n"
            f"Incorporate these where realistic.\n"
        )
    return base + "\n" + grounded_rules


# QUICK TEST

if __name__ == "__main__":
    from data import DEFAULT_DATA_PATH, run_pipeline
    from model import DEFAULT_MODEL_PATH, load_model

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    model = load_model(DEFAULT_MODEL_PATH)

    # SHAP on a sample of the test set keeps the script responsive
    X_explain = X_test.head(500)

    explainer = build_explainer(model, X_train)
    shap_values = get_shap_values(explainer, X_explain)

    predictions = model.predict(X_explain)
    rejected = np.where(predictions == 1)[0]
    print(f"\nRejected applicants in explanation sample: {len(rejected)}")

    idx = int(rejected[0])
    shap_df = explain_applicant(explainer, shap_values, X_explain, idx)

    plot_waterfall(explainer, shap_values, X_explain, idx)
    plot_global_importance(shap_values, X_explain)

    prompt = build_explanation_prompt(X_explain.iloc[idx], shap_df)
    print("\n" + "=" * 50)
    print("LLM PROMPT:")
    print("=" * 50)
    print(prompt)
