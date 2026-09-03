"""
api.py
------
FastAPI backend for the Explainable Loan Rejection Assistant.

Run from project root:
    uvicorn api:app --reload
or:
    python api.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Do not import pipeline/SHAP/DiCE here. Vercel crashes if those
# native libs load during cold start of GET /.
_artifacts: dict[str, Any] | None = None
_load_error: str | None = None


class ApplicationRequest(BaseModel):
    person_age: float = Field(..., ge=18, le=100)
    person_income: float = Field(..., ge=0)
    person_emp_length: float = Field(..., ge=0)
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = "C"
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0)
    loan_percent_income: Optional[float] = None
    cb_person_default_on_file: int = Field(0, ge=0, le=1)
    cb_person_cred_hist_length: float = Field(..., ge=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = "RENT"
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ] = "PERSONAL"
    llm_mode: Literal["grounded", "free", "off"] = "grounded"


def get_artifacts() -> dict[str, Any]:
    """Lazy-load so / and /health never crash the Vercel function."""
    global _artifacts, _load_error
    if _artifacts is not None:
        return _artifacts
    try:
        from pipeline import load_artifacts

        _artifacts = load_artifacts()
        _load_error = None
        return _artifacts
    except Exception as e:
        _load_error = str(e)
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts unavailable: {e}",
        ) from e


app = FastAPI(
    title="Explainable Loan Rejection Assistant",
    description=(
        "Predict loan outcomes with SHAP drivers, DiCE recourse, "
        "grounded Gemini explanations, and ECOA-oriented fairness audit."
    ),
    version="1.1.0",
)


@app.get("/")
def root():
    return {
        "name": "Explainable Loan Rejection Assistant",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "fairness": "/fairness",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts_loaded": _artifacts is not None,
        "last_error": _load_error,
    }


@app.get("/fairness")
def fairness():
    audit = get_artifacts().get("fairness_audit", {})
    return {
        "age_disparity": audit.get("age_disparity"),
        "passed": audit.get("passed"),
        "threshold": audit.get("threshold"),
        "note": audit.get("note"),
        "gender_note": audit.get("gender_note"),
        "groups": audit.get("groups", []),
    }


@app.post("/predict")
def predict(req: ApplicationRequest):
    artifacts = get_artifacts()

    payload = req.model_dump()
    llm_mode = payload.pop("llm_mode")
    if payload.get("loan_percent_income") is None:
        income = payload["person_income"]
        payload["loan_percent_income"] = (
            payload["loan_amnt"] / income if income > 0 else 0.0
        )

    from pipeline import run_application

    result = run_application(payload, artifacts, llm_mode=llm_mode)

    shap_df = result.get("shap_explanation")
    if shap_df is not None:
        result["shap_explanation"] = shap_df.to_dict(orient="records")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
