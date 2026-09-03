"""
recourse_metrics.py
-------------------
Quantitative metrics for counterfactual recourse quality.

These metrics support the paper's evaluation of actionable explanations:
    validity, sparsity, proximity, and actionability delay.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# How quickly an applicant can act on a feature change (higher = harder / slower)
ACTIONABILITY_DELAY = {
    "loan_amnt": 0.0,              # can request less immediately
    "loan_percent_income": 0.1,    # follows from amount/income
    "loan_int_rate": 0.4,          # product shopping / negotiation
    "person_income": 0.7,          # months to years
    "person_emp_length": 1.0,      # purely time-gated
}


def sparse_count(original: pd.Series, cf_row: pd.Series, features: list[str],
                 rel_tol: float = 0.01) -> int:
    n = 0
    for f in features:
        if f not in original.index or f not in cf_row.index:
            continue
        o, c = float(original[f]), float(cf_row[f])
        if abs(c - o) / (abs(o) + 1e-9) > rel_tol:
            n += 1
    return n


def proximity(original: pd.Series, cf_row: pd.Series, features: list[str]) -> float:
    """
    Mean symmetric relative change across features (lower is closer).

    Uses |c-o| / (|o|+|c|+eps) so zero-valued features (e.g. emp_length=0)
    do not explode the score.
    """
    diffs = []
    for f in features:
        if f not in original.index or f not in cf_row.index:
            continue
        o, c = float(original[f]), float(cf_row[f])
        diffs.append(abs(c - o) / (abs(o) + abs(c) + 1e-6))
    return float(np.mean(diffs)) if diffs else 0.0


def actionability_delay_score(original: pd.Series, cf_row: pd.Series,
                              features: list[str], rel_tol: float = 0.01) -> float:
    """
    Weighted delay of changed features in [0, 1].
    0 = all immediate (e.g. reduce loan amount); 1 = only time-gated changes.
    """
    weights, total = [], 0.0
    for f in features:
        if f not in original.index or f not in cf_row.index:
            continue
        o, c = float(original[f]), float(cf_row[f])
        if abs(c - o) / (abs(o) + 1e-9) <= rel_tol:
            continue
        w = ACTIONABILITY_DELAY.get(f, 0.5)
        weights.append(w)
        total += w
    if not weights:
        return 0.0
    return float(total / len(weights))


def score_counterfactual_set(
    model,
    original_df: pd.DataFrame,
    cf_df: pd.DataFrame | None,
    features: list[str],
) -> dict[str, Any]:
    """
    Score a DiCE CF set for one applicant.

    validity: fraction of CFs that flip prediction to approval (class 0)
    """
    if cf_df is None or len(cf_df) == 0:
        return {
            "n_cfs": 0,
            "validity": 0.0,
            "mean_sparsity": None,
            "mean_proximity": None,
            "mean_actionability_delay": None,
        }

    original = original_df.iloc[0]
    # Align columns for prediction
    pred_cols = original_df.columns.tolist()
    valid = 0
    sparsities, proximities, delays = [], [], []

    for _, row in cf_df.iterrows():
        row_full = original.copy()
        for c in cf_df.columns:
            if c in row_full.index and c != "loan_status":
                row_full[c] = row[c]
        X = pd.DataFrame([row_full])[pred_cols].astype(float)
        pred = int(model.predict(X)[0])
        if pred == 0:
            valid += 1

        vary = [f for f in features if f in row.index]
        sparsities.append(sparse_count(original, row, vary))
        proximities.append(proximity(original, row, vary))
        delays.append(actionability_delay_score(original, row, vary))

    n = len(cf_df)
    return {
        "n_cfs": n,
        "validity": valid / n,
        "mean_sparsity": float(np.mean(sparsities)),
        "mean_proximity": float(np.mean(proximities)),
        "mean_actionability_delay": float(np.mean(delays)),
    }
