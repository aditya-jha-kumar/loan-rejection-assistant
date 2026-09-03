"""Deterministic template explanations for offline ablations (no LLM API)."""

from __future__ import annotations

import pandas as pd

from evaluation.faithfulness import FEATURE_ALIASES


def _label(feature: str) -> str:
    aliases = FEATURE_ALIASES.get(feature)
    if aliases:
        return aliases[0]
    return feature.replace("_", " ")


def template_explanation(shap_df: pd.DataFrame, suggestion_text: str | None = None,
                         top_k: int = 3, grounded: bool = True) -> str:
    """
    Non-LLM baseline used in experiments.

    grounded=True: only top SHAP risks (+ suggestions)
    grounded=False: also injects an ungrounded distractor phrase (simulates drift)
    """
    risks = (
        shap_df[shap_df["SHAP"] > 0]
        .sort_values("SHAP", ascending=False)
        .head(top_k)
    )
    parts = [
        "We regret that we could not approve this application at this time.",
        "The main factors were: "
        + ", ".join(
            f"{_label(r.Feature)} (value {r.Value})"
            for r in risks.itertuples()
        )
        + ".",
    ]
    if not grounded:
        parts.append(
            "In addition, external bureau medical debt signals and marital "
            "status patterns were considered."
        )
    if suggestion_text:
        parts.append("Possible next steps:" + suggestion_text)
    else:
        parts.append(
            "Consider reducing the requested amount or improving income stability."
        )
    return " ".join(parts)


def applicant_local_explanation(shap_df: pd.DataFrame,
                                suggestion_text: str | None = None,
                                top_k: int = 3) -> str:
    """Applicant-facing fallback when Gemini is unavailable."""
    risks = (
        shap_df[shap_df["SHAP"] > 0]
        .sort_values("SHAP", ascending=False)
        .head(top_k)
    )
    protective = (
        shap_df[shap_df["SHAP"] < 0]
        .sort_values("SHAP", ascending=True)
        .head(2)
    )
    lines = [
        "We were not able to approve this application.",
        "",
        "The main concerns were:",
    ]
    for row in risks.itertuples():
        lines.append(f"- {_label(row.Feature)}")
    if len(protective):
        lines.append("")
        lines.append("Factors that helped the application:")
        for row in protective.itertuples():
            lines.append(f"- {_label(row.Feature)}")
    lines.append("")
    if suggestion_text and suggestion_text.strip() and "Could not generate" not in suggestion_text:
        lines.append("Changes that could flip this decision:")
        lines.append(suggestion_text.strip())
    else:
        lines.append(
            "Consider requesting a smaller amount or strengthening income "
            "and employment history before reapplying."
        )
    return "\n".join(lines)
