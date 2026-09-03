"""
counterfactuals.py
------------------
Generates DiCE counterfactual explanations for rejected loan
applicants — answering "what would need to change to get approved?"

Key concepts:
    Counterfactual  - A modified version of the applicant's profile
                      that flips the decision from rejected to approved
    Actionable      - Only features the applicant can realistically change
    Diverse         - Multiple different paths to approval, not just one
    Feasibility     - Changes must be realistic given the applicant's context

Functions:
    build_dice_explainer(model, X_train)  - Create DiCE explainer
    generate_counterfactuals(             - Generate actionable CFs
        dice_exp, applicant_data,
        feature_ranges)
    format_counterfactuals(cf_result,     - Convert to readable format
                           original)
    get_feature_ranges(X_train)           - Compute realistic ranges
                                            from training data
"""

import dice_ml
import numpy as np
import pandas as pd


# WRAPPER — fixes XGBoost dtype incompatibility

class XGBWrapper:
    """
    Wraps XGBoost model to force float64 dtype and stable feature names.

    Why this is needed:
    DiCE internally modifies DataFrames during counterfactual
    generation, sometimes producing object dtype columns or numpy
    arrays without names. XGBoost 3.x is strict about both dtypes
    and training feature names.
    """

    def __init__(self, model):
        self.model = model
        names = getattr(model, "feature_names_in_", None)
        self.feature_names_ = list(names) if names is not None else []

    def _frame(self, X):
        df = pd.DataFrame(X)
        if self.feature_names_ and df.shape[1] == len(self.feature_names_):
            if list(df.columns) != self.feature_names_:
                df.columns = self.feature_names_
        return df.astype(float)

    def predict(self, X):
        return self.model.predict(self._frame(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._frame(X))


# BUILD DiCE EXPLAINER

def build_dice_explainer(model, X_train, y_train):
    """
    Create a DiCE explainer wrapping the XGBoost model.

    DiCE needs three things:
    1. The training data (to understand feature distributions)
    2. The model (to evaluate whether a CF gets approved)
    3. Which features are continuous vs categorical

    Continuous features: DiCE can suggest any value in a range
    Categorical features: DiCE can only suggest valid categories

    Args:
        model:               Trained XGBoost model
        X_train (DataFrame): Training features
        y_train (Series):    Training labels

    Returns:
        tuple: (dice_explainer, dice_data_object)
    """
    train_df = X_train.copy()
    train_df["loan_status"] = y_train.values

    continuous_features = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]

    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name="loan_status",
    )

    wrapped_model = XGBWrapper(model)
    m = dice_ml.Model(model=wrapped_model, backend="sklearn")

    # random is faster than genetic or kdtree for large datasets
    dice_exp = dice_ml.Dice(d, m, method="random")

    print("DiCE explainer created successfully")
    return dice_exp, d


# DEFINE REALISTIC FEATURE RANGES

def get_feature_ranges(X_train, applicant_data):
    """
    Compute realistic feature ranges for counterfactual generation.

    Why custom ranges instead of dataset min/max?
    Dataset min/max includes outliers and edge cases.
    We want ranges that are:
    1. Realistic — within the 5th-95th percentile of training data
    2. Actionable — applicant can actually achieve these values
    3. Directional — income should only go up, loan should go down

    Args:
        X_train (DataFrame):     Training data for percentile calculation
        applicant_data (Series): The specific applicant being explained

    Returns:
        dict: Feature name → [min, max] realistic range
    """
    current_income = float(applicant_data.get("person_income", 30000))
    current_loan = float(applicant_data.get("loan_amnt", 10000))
    current_emp = float(applicant_data.get("person_emp_length", 1))
    current_pct = float(applicant_data.get("loan_percent_income", 0.5))
    current_rate = float(applicant_data.get("loan_int_rate", 12.0))

    income_hi = float(X_train["person_income"].quantile(0.95))
    emp_hi = float(X_train["person_emp_length"].quantile(0.95))
    rate_lo = float(X_train["loan_int_rate"].quantile(0.05))

    # Ensure min <= max for DiCE (applicant may already be near extremes)
    ranges = {
        "person_income": [
            current_income,
            max(current_income, income_hi),
        ],
        "loan_amnt": [
            min(current_loan * 0.1, current_loan),
            current_loan,
        ],
        "person_emp_length": [
            current_emp,
            max(current_emp, emp_hi),
        ],
        "loan_percent_income": [
            0.05,
            max(0.05, current_pct),
        ],
        # Interest rate can decrease (better product / refinance terms)
        "loan_int_rate": [
            min(rate_lo, current_rate),
            current_rate,
        ],
    }

    return ranges


# GENERATE COUNTERFACTUALS

def generate_counterfactuals(dice_exp, applicant_data,
                             feature_ranges, n=3):
    """
    Generate diverse counterfactual explanations for one applicant.

    Counterfactuals show the minimum changes needed to flip the
    decision from rejected (1) to approved (0).

    desired_class="opposite" means: whatever the current prediction
    is, find inputs that produce the opposite prediction.

    features_to_vary controls actionability — we only allow DiCE
    to change features the applicant can realistically modify.

    Args:
        dice_exp:                   DiCE explainer object
        applicant_data (DataFrame): Single row from X_test
        feature_ranges (dict):      Realistic min/max per feature
        n (int):                    Number of counterfactuals to generate

    Returns:
        DiCE counterfactual result object
    """
    try:
        query = applicant_data.copy()
        if isinstance(query, pd.DataFrame):
            query = query.astype(float)
        # Desired class 0 = good standing / approval for rejected applicants
        cf = dice_exp.generate_counterfactuals(
            query_instances=query,
            total_CFs=n,
            desired_class=0,
            permitted_range=feature_ranges,
            features_to_vary=list(feature_ranges.keys()),
            sample_size=2000,
            random_seed=42,
        )
        return cf
    except Exception as e:
        print(f"Counterfactual generation failed: {e}")
        return None


# FORMAT OUTPUT

def format_counterfactuals(cf_result, original_data):
    """
    Convert DiCE output into a clean, human-readable format.

    DiCE returns raw DataFrames with all features. This function:
    1. Extracts only what changed
    2. Formats changes as plain English suggestions
    3. Returns both the raw change list and readable text

    Args:
        cf_result:              DiCE counterfactual result
        original_data (Series): Original applicant features

    Returns:
        tuple: (changes_df, suggestion_text)
    """
    if cf_result is None:
        return None, "Could not generate counterfactuals for this applicant."

    cf_examples = cf_result.cf_examples_list
    if not cf_examples or cf_examples[0].final_cfs_df is None:
        return None, "Could not generate counterfactuals for this applicant."

    cf_df = cf_examples[0].final_cfs_df

    changes = []
    suggestions = []

    feature_labels = {
        "person_income": "Annual Income",
        "loan_amnt": "Loan Amount Requested",
        "person_emp_length": "Employment Length (years)",
        "loan_percent_income": "Loan as % of Income",
        "loan_int_rate": "Interest Rate",
    }

    for _, row in cf_df.iterrows():
        cf_changes = {}
        cf_text = []

        for feature, label in feature_labels.items():
            if feature not in original_data.index:
                continue
            if feature not in row.index:
                continue

            original_val = float(original_data[feature])
            cf_val = float(row[feature])

            # Only report meaningful changes (>1% difference)
            if abs(cf_val - original_val) / (abs(original_val) + 1e-9) > 0.01:
                cf_changes[feature] = {
                    "from": original_val,
                    "to": cf_val,
                }

                if feature == "person_income":
                    cf_text.append(
                        f"Increase annual income from "
                        f"${original_val:,.0f} to ${cf_val:,.0f}"
                    )
                elif feature == "loan_amnt":
                    cf_text.append(
                        f"Reduce loan request from "
                        f"${original_val:,.0f} to ${cf_val:,.0f}"
                    )
                elif feature == "person_emp_length":
                    cf_text.append(
                        f"Build employment history from "
                        f"{original_val:.0f} to {cf_val:.0f} years"
                    )
                elif feature == "loan_percent_income":
                    cf_text.append(
                        f"Reduce loan-to-income ratio from "
                        f"{original_val:.1%} to {cf_val:.1%}"
                    )
                elif feature == "loan_int_rate":
                    cf_text.append(
                        f"Lower interest rate from "
                        f"{original_val:.2f}% to {cf_val:.2f}%"
                    )

        changes.append(cf_changes)
        suggestions.append(cf_text)

    suggestion_text = ""
    for i, cf_text in enumerate(suggestions):
        if cf_text:
            suggestion_text += f"\nOption {i + 1}:\n"
            suggestion_text += "\n".join(f"  - {s}" for s in cf_text)
            suggestion_text += "\n"

    print("\nCounterfactual Suggestions:")
    print(suggestion_text if suggestion_text else "\n(No actionable changes found.)\n")

    return changes, suggestion_text


# QUICK TEST

if __name__ == "__main__":
    from data import DEFAULT_DATA_PATH, run_pipeline
    from model import DEFAULT_MODEL_PATH, load_model

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    model = load_model(DEFAULT_MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    rejected = np.where(predictions == 1)[0]
    # Prefer a near-boundary reject so actionable CFs exist
    idx = int(rejected[np.argmin(probabilities[rejected])])

    print(f"Explaining applicant #{idx}")
    print(f"Default probability: {probabilities[idx]:.4f}")
    print(f"Features:\n{X_test.iloc[idx]}")

    dice_exp, d = build_dice_explainer(model, X_train, y_train)

    applicant_data = X_test.iloc[[idx]]
    feature_ranges = get_feature_ranges(X_train, X_test.iloc[idx])

    print("\nFeature ranges for counterfactuals:")
    for feature, range_ in feature_ranges.items():
        print(f"  {feature}: {range_[0]:.1f} -> {range_[1]:.1f}")

    cf_result = generate_counterfactuals(
        dice_exp, applicant_data, feature_ranges
    )

    changes, suggestion_text = format_counterfactuals(
        cf_result, X_test.iloc[idx]
    )
