"""
faithfulness.py
---------------
Automatic faithfulness metrics for LLM loan-rejection explanations.

Research contribution:
    Measure whether natural-language explanations stay grounded in
    SHAP risk factors and DiCE recourse suggestions, and whether
    grounded prompting reduces hallucination of unsupported features.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# Surface forms applicants / LLM might use for each feature
FEATURE_ALIASES: dict[str, list[str]] = {
    "person_age": ["age", "years old"],
    "person_income": ["income", "salary", "earnings", "annual income"],
    "person_emp_length": [
        "employment", "employment length", "job tenure", "years of work",
        "work history", "employed",
    ],
    "loan_grade": ["loan grade", "credit grade", "grade"],
    "loan_amnt": ["loan amount", "loan request", "amount requested", "borrow"],
    "loan_int_rate": ["interest rate", "apr", "rate"],
    "loan_percent_income": [
        "loan as percent", "loan-to-income", "loan to income",
        "percent of income", "percentage of income", "debt-to-income",
        "dti",
    ],
    "cb_person_default_on_file": [
        "previous default", "prior default", "default on file", "past default",
    ],
    "cb_person_cred_hist_length": [
        "credit history", "credit history length", "length of credit",
    ],
    "person_home_ownership_RENT": ["rent", "renting"],
    "person_home_ownership_OWN": ["own home", "homeowner", "own their home"],
    "person_home_ownership_OTHER": ["other home", "home ownership other"],
    "loan_intent_EDUCATION": ["education"],
    "loan_intent_HOMEIMPROVEMENT": ["home improvement"],
    "loan_intent_MEDICAL": ["medical"],
    "loan_intent_PERSONAL": ["personal loan", "personal purpose"],
    "loan_intent_VENTURE": ["venture", "business"],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def features_mentioned(text: str, candidates: Iterable[str] | None = None) -> set[str]:
    """Return feature keys whose aliases appear in text."""
    norm = _normalize(text)
    keys = candidates if candidates is not None else FEATURE_ALIASES.keys()
    found = set()
    for feat in keys:
        aliases = FEATURE_ALIASES.get(feat, [feat.replace("_", " ")])
        for alias in aliases:
            if alias.lower() in norm:
                found.add(feat)
                break
    return found


def grounded_feature_set(shap_df: pd.DataFrame, top_k: int = 5,
                         cf_features: list[str] | None = None) -> set[str]:
    """Allowed features for a grounded explanation."""
    risks = (
        shap_df[shap_df["SHAP"] > 0]
        .sort_values("SHAP", ascending=False)
        .head(top_k)["Feature"]
        .tolist()
    )
    allowed = set(risks)
    if cf_features:
        allowed.update(cf_features)
    return allowed


def score_faithfulness(
    explanation: str,
    shap_df: pd.DataFrame,
    cf_features: list[str] | None = None,
    top_k: int = 5,
) -> dict:
    """
    Compute faithfulness scores for one explanation.

    coverage: fraction of top SHAP risk features mentioned
    precision: fraction of mentioned known features that are grounded
    hallucination_rate: 1 - precision (among recognized feature mentions)
    empty_explanation: True if explanation missing
    """
    if not explanation or not str(explanation).strip():
        return {
            "coverage": 0.0,
            "precision": 0.0,
            "hallucination_rate": 1.0,
            "n_mentioned": 0,
            "n_grounded_available": 0,
            "empty_explanation": True,
        }

    top_risks = (
        shap_df[shap_df["SHAP"] > 0]
        .sort_values("SHAP", ascending=False)
        .head(top_k)["Feature"]
        .tolist()
    )
    allowed = grounded_feature_set(shap_df, top_k=top_k, cf_features=cf_features)
    mentioned = features_mentioned(explanation)

    if top_risks:
        coverage = len(set(top_risks) & mentioned) / len(top_risks)
    else:
        coverage = 0.0

    if mentioned:
        precision = len(mentioned & allowed) / len(mentioned)
    else:
        precision = 0.0

    return {
        "coverage": float(coverage),
        "precision": float(precision),
        "hallucination_rate": float(1.0 - precision) if mentioned else 1.0,
        "n_mentioned": len(mentioned),
        "n_grounded_available": len(allowed),
        "empty_explanation": False,
        "mentioned": sorted(mentioned),
        "allowed": sorted(allowed),
    }


def compare_prompt_modes(scores_grounded: dict, scores_free: dict) -> dict:
    """Delta summary for grounded vs unconstrained prompting."""
    return {
        "coverage_delta": scores_grounded["coverage"] - scores_free["coverage"],
        "precision_delta": scores_grounded["precision"] - scores_free["precision"],
        "hallucination_delta": (
            scores_grounded["hallucination_rate"] - scores_free["hallucination_rate"]
        ),
    }
