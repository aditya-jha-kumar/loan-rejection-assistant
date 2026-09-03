"""
fairness.py
-----------
Fairness audit for the Explainable Loan Rejection Assistant.

Measures whether the model discriminates against protected groups
as defined by ECOA (Equal Credit Opportunity Act):
- Age
- Gender (not in this dataset — noted as limitation)

Metrics computed:
    Demographic Parity   - Equal approval rates across groups
    Equal Opportunity    - Equal recall (TPR) across groups
    Predictive Parity    - Equal precision across groups

Threshold for concern: >10% disparity on any metric

Functions:
    audit_age_fairness(X_test, y_test, predictions)
    audit_feature_fairness(X_test, y_test,
                           predictions, feature, groups)
    run_full_audit(X_test, y_test, predictions)
    plot_fairness_summary(audit_results)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score


# ─────────────────────────────────────────────
# CORE AUDIT FUNCTION
# ─────────────────────────────────────────────

def audit_group_fairness(X_test, y_test, predictions,
                         feature, groups):
    """
    Measure fairness metrics across groups of a single feature.

    Three metrics per group:
    1. Approval Rate (Demographic Parity)
       Are approval rates equal across groups?
       Formula: count(predicted=0) / count(total in group)

    2. Recall / True Positive Rate (Equal Opportunity)
       Among applicants who are truly good (y=0), does the
       model catch them equally across groups?
       Formula: TP / (TP + FN) per group

    3. Precision (Predictive Parity)
       When the model approves someone, is it equally accurate
       across groups?
       Formula: TP / (TP + FP) per group

    Args:
        X_test (DataFrame):   Test features
        y_test (Series):      True labels
        predictions (array):  Model predictions
        feature (str):        Column name to group by
        groups (dict):        {label: value} e.g. {"Young": 0, "Senior": 1}

    Returns:
        pd.DataFrame: Fairness metrics per group
    """
    results = pd.DataFrame()

    base_df = pd.DataFrame({
        "feature":     X_test[feature].values,
        "actual":      y_test.values,
        "predicted":   predictions
    })

    rows = []
    for label, value in groups.items():
        group = base_df[base_df["feature"] == value]

        if len(group) < 10:
            print(f"  WARNING: {label} group has only {len(group)} "
                  f"samples — metrics unreliable")
            continue

        # Approval rate — predicted good standing (0)
        approval_rate = (group["predicted"] == 0).mean()

        # Recall on good standing — among truly good applicants,
        # how many does the model correctly approve?
        good_applicants = group[group["actual"] == 0]
        if len(good_applicants) > 0:
            recall = (
                (good_applicants["predicted"] == 0).sum() /
                len(good_applicants)
            )
        else:
            recall = None

        # Precision on approvals — among model approvals,
        # what fraction are truly good applicants?
        approved_by_model = group[group["predicted"] == 0]
        if len(approved_by_model) > 0:
            precision = (
                (approved_by_model["actual"] == 0).sum() /
                len(approved_by_model)
            )
        else:
            precision = None

        rows.append({
            "Group":         label,
            "Count":         len(group),
            "Approval Rate": approval_rate,
            "Recall":        recall,
            "Precision":     precision
        })

    results = pd.DataFrame(rows)
    return results


# ─────────────────────────────────────────────
# AGE FAIRNESS — ECOA PROTECTED CHARACTERISTIC
# ─────────────────────────────────────────────

def audit_age_fairness(X_test, y_test, predictions):
    """
    Audit fairness across age groups.

    ECOA prohibits discrimination based on age in credit decisions.
    We split into three groups:
    - Young:   20-30 (early career, less credit history)
    - Middle:  31-50 (established, peak earning years)
    - Senior:  51+   (pre-retirement, different risk profile)

    Why these cutoffs?
    They align with typical career and financial lifecycle stages,
    making the fairness comparison meaningful.

    Args:
        X_test (DataFrame):  Test features
        y_test (Series):     True labels
        predictions (array): Model predictions

    Returns:
        pd.DataFrame: Age group fairness metrics
    """
    # Create age group column
    X_test_copy = X_test.copy()
    X_test_copy["age_group"] = pd.cut(
        X_test_copy["person_age"],
        bins=[0, 30, 50, 100],
        labels=["Young (20-30)", "Middle (31-50)", "Senior (51+)"]
    )

    print("\n=== AGE FAIRNESS AUDIT ===")
    print("ECOA requires equal treatment regardless of age\n")

    base_df = pd.DataFrame({
        "age_group": X_test_copy["age_group"].values,
        "actual":    y_test.values,
        "predicted": predictions
    })

    rows = []
    numeric_rates = []
    for label in ["Young (20-30)", "Middle (31-50)", "Senior (51+)"]:
        group = base_df[base_df["age_group"] == label]

        if len(group) < 10:
            continue

        approval_rate = (group["predicted"] == 0).mean()
        good          = group[group["actual"] == 0]
        recall        = (good["predicted"] == 0).mean() if len(good) > 0 else None
        approved      = group[group["predicted"] == 0]
        precision     = (approved["actual"] == 0).mean() if len(approved) > 0 else None
        numeric_rates.append(float(approval_rate))

        rows.append({
            "Age Group":     label,
            "Count":         len(group),
            "Approval Rate": f"{approval_rate:.1%}",
            "Approval Rate Raw": float(approval_rate),
            "Recall":        f"{recall:.1%}" if recall is not None else "N/A",
            "Recall Raw":    float(recall) if recall is not None else None,
            "Precision":     f"{precision:.1%}" if precision is not None else "N/A",
            "Precision Raw": float(precision) if precision is not None else None,
        })

    results = pd.DataFrame(rows)
    print(results.drop(
        columns=[c for c in results.columns if c.endswith("Raw")],
        errors="ignore",
    ).to_string(index=False))

    # Calculate and flag disparity
    disparity = (max(numeric_rates) - min(numeric_rates)) if numeric_rates else 0.0
    flag      = "[CONCERN]" if disparity > 0.10 else "[FAIR]"

    print(f"\nApproval rate disparity: {disparity:.1%} {flag}")
    print("(Threshold: >10% disparity indicates potential bias)")

    return results, disparity


# ─────────────────────────────────────────────
# LOAN INTENT FAIRNESS
# ─────────────────────────────────────────────

def audit_intent_fairness(X_test, y_test, predictions):
    """
    Audit whether loan purpose affects approval rates unfairly.

    While loan intent is a legitimate risk factor, extreme
    disparities might indicate the model is over-penalizing
    certain purposes (e.g., medical loans vs education loans).

    Args:
        X_test (DataFrame):  Test features
        y_test (Series):     True labels
        predictions (array): Model predictions

    Returns:
        pd.DataFrame: Intent group fairness metrics
    """
    print("\n=== LOAN INTENT FAIRNESS AUDIT ===")

    intent_cols = [c for c in X_test.columns
                   if c.startswith("loan_intent_")]

    base_df = pd.DataFrame({
        "actual":    y_test.values,
        "predicted": predictions
    })

    rows = []
    for col in intent_cols:
        intent_name = col.replace("loan_intent_", "").title()
        mask        = X_test[col].values == 1
        group       = base_df[mask]

        if len(group) < 10:
            continue

        approval_rate = (group["predicted"] == 0).mean()
        good          = group[group["actual"] == 0]
        recall        = (good["predicted"] == 0).mean() if len(good) > 0 else None

        rows.append({
            "Loan Intent":   intent_name,
            "Count":         len(group),
            "Approval Rate": f"{approval_rate:.1%}",
            "Recall":        f"{recall:.1%}" if recall is not None else "N/A"
        })

    results = pd.DataFrame(rows).sort_values(
        "Approval Rate", ascending=False
    )
    print(results.to_string(index=False))

    return results


# ─────────────────────────────────────────────
# FULL AUDIT
# ─────────────────────────────────────────────

def run_full_audit(X_test, y_test, predictions):
    """
    Run the complete fairness audit.

    Covers:
    1. Age fairness      (ECOA protected characteristic)
    2. Loan intent fairness (detect undue penalization)

    Note on gender:
    This dataset does not contain a gender column. This is
    noted as a limitation in the paper — a production system
    should audit gender fairness explicitly.

    Args:
        X_test (DataFrame):  Test features
        y_test (Series):     True labels
        predictions (array): Model predictions

    Returns:
        dict: All audit results
    """
    print("=" * 55)
    print("FAIRNESS AUDIT REPORT")
    print("Regulatory framework: ECOA (Equal Credit Opportunity Act)")
    print("=" * 55)

    age_results, age_disparity = audit_age_fairness(
        X_test, y_test, predictions
    )

    intent_results = audit_intent_fairness(
        X_test, y_test, predictions
    )

    print("\n=== AUDIT SUMMARY ===")
    print(f"Age disparity:    {age_disparity:.1%} "
          f"{'[PASS]' if age_disparity < 0.10 else '[REVIEW]'}")
    print(f"Gender audit:     [N/A] Not available - "
          f"dataset lacks gender column (paper limitation)")
    print("\nNote: Disparity threshold of 10% follows the")
    print("80% rule used in US employment discrimination law,")
    print("adapted for credit decision contexts.")

    passed = age_disparity < 0.10
    return {
        "age_results":    age_results,
        "age_disparity":  float(age_disparity),
        "intent_results": intent_results,
        "passed":         passed,
        "threshold":      0.10,
        "note": (
            f"Age approval-rate disparity {age_disparity:.1%} "
            f"({'PASS' if passed else 'REVIEW'} vs 10% threshold)"
        ),
        "gender_note": (
            "Gender data not available in this dataset — "
            "noted as a limitation for ECOA-complete auditing."
        ),
        "groups": (
            age_results[["Age Group", "Count", "Approval Rate Raw", "Recall Raw"]]
            .rename(columns={
                "Approval Rate Raw": "approval_rate",
                "Recall Raw": "recall",
            })
            .to_dict(orient="records")
            if len(age_results) and "Approval Rate Raw" in age_results.columns
            else []
        ),
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from data import DEFAULT_DATA_PATH, run_pipeline
    from model import DEFAULT_MODEL_PATH, load_model

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    model       = load_model(DEFAULT_MODEL_PATH)
    predictions = model.predict(X_test)

    audit_results = run_full_audit(X_test, y_test, predictions)