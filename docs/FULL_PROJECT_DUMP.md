# Loan-Rejection - FULL PROJECT DUMP

Generated for ChatGPT / external AI handoff. Contains project structure, all text source files, configs, docs, experiment results, and captured terminal/experiment output.

**Excluded (secrets / binaries / caches):** `.env`, raw `*.pkl` / image bytes, `__pycache__`, `.git`, `.venv`. Binary assets are listed by path/size only. Dataset CSV is truncated to header + sample rows (full file: `data/loan_dataset.csv`).

---

## 1. Project structure

```text
.env  (70 bytes)
.env.example  (85 bytes)
.gitignore  (342 bytes)
api.py  (3372 bytes)
app.py  (14280 bytes)
config/config.yaml  (833 bytes)
data/loan_dataset.csv  (1804682 bytes)
docs/MODEL_CARD.md  (1877 bytes)
docs/PAPER_DRAFT.md  (4660 bytes)
models/baseline_comparison.json  (681 bytes)
models/confusion_matrix.png  (33771 bytes)
models/global_importance.png  (91695 bytes)
models/loan_model.pkl  (445523 bytes)
models/loan_model_calibrated.pkl  (1754332 bytes)
models/waterfall.png  (91213 bytes)
notebooks/notebooks01_data_exploration.ipynb  (8976 bytes)
README.md  (3740 bytes)
requirements.txt  (302 bytes)
results/experiment_results.json  (5173 bytes)
scripts/generate_full_dump.py  (12296 bytes)
src/baselines.py  (5920 bytes)
src/config_loader.py  (583 bytes)
src/counterfactuals.py  (11665 bytes)
src/data.py  (4154 bytes)
src/evaluation/__init__.py  (322 bytes)
src/evaluation/faithfulness.py  (5174 bytes)
src/evaluation/recourse_metrics.py  (4098 bytes)
src/evaluation/run_experiments.py  (9423 bytes)
src/evaluation/template_explainer.py  (1562 bytes)
src/explainer.py  (13145 bytes)
src/fairness.py  (12673 bytes)
src/llm.py  (1962 bytes)
src/logging_utils.py  (618 bytes)
src/model.py  (6425 bytes)
src/pipeline.py  (9177 bytes)
src/requirements.txt  (302 bytes)
tests/test_core.py  (3227 bytes)
```

## 2. File contents

### `.env.example`

```bash
# Get a key at https://aistudio.google.com/apikey
GEMINI_API_KEY=your_api_key_here
```

### `.gitignore`

```
# Python
__pycache__/
*.pyc
*.pyo
.env

# Data — don't push large datasets to GitHub
data/
models/*.pkl

# Experiment outputs can be regenerated
results/

# Jupyter checkpoints
.ipynb_checkpoints/

# Environment
venv/
.venv/

# API keys
.env
secrets.py

# Pytest / tooling
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

### `api.py`

```python
"""
api.py
------
FastAPI backend for the Explainable Loan Rejection Assistant.

Run from project root:
    uvicorn api:app --reload --app-dir .
or:
    python api.py
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import load_artifacts, run_application  # noqa: E402

_artifacts: dict[str, Any] | None = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifacts
    _artifacts = load_artifacts()
    yield
    _artifacts = None


app = FastAPI(
    title="Explainable Loan Rejection Assistant",
    description=(
        "Predict loan outcomes with SHAP drivers, DiCE recourse, "
        "grounded Gemini explanations, and ECOA-oriented fairness audit."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": _artifacts is not None}


@app.get("/fairness")
def fairness():
    if _artifacts is None:
        raise HTTPException(503, "Artifacts not loaded")
    audit = _artifacts.get("fairness_audit", {})
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
    if _artifacts is None:
        raise HTTPException(503, "Artifacts not loaded")

    payload = req.model_dump()
    llm_mode = payload.pop("llm_mode")
    if payload.get("loan_percent_income") is None:
        income = payload["person_income"]
        payload["loan_percent_income"] = (
            payload["loan_amnt"] / income if income > 0 else 0.0
        )

    result = run_application(payload, _artifacts, llm_mode=llm_mode)

    # JSON-serialize DataFrame
    shap_df = result.get("shap_explanation")
    if shap_df is not None:
        result["shap_explanation"] = shap_df.to_dict(orient="records")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
```

### `app.py`

```python
"""
app.py
------
Streamlit UI for the Explainable Loan Rejection Assistant.

Run from the project root:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import load_artifacts, run_application  # noqa: E402

FEATURE_LABELS = {
    "person_age": "Age",
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length",
    "loan_grade": "Credit Grade",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate",
    "loan_percent_income": "Loan as % of Income",
    "cb_person_default_on_file": "Prior Default on File",
    "cb_person_cred_hist_length": "Credit History Length",
    "person_home_ownership_OTHER": "Home: Other",
    "person_home_ownership_OWN": "Home: Own",
    "person_home_ownership_RENT": "Home: Rent",
    "loan_intent_EDUCATION": "Purpose: Education",
    "loan_intent_HOMEIMPROVEMENT": "Purpose: Home Improvement",
    "loan_intent_MEDICAL": "Purpose: Medical",
    "loan_intent_PERSONAL": "Purpose: Personal",
    "loan_intent_VENTURE": "Purpose: Venture",
}

GRADE_HELP = "A is strongest credit quality; G is weakest."


def label_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def format_value(feature: str, value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if feature == "person_income" or feature == "loan_amnt":
        return f"${v:,.0f}"
    if feature == "loan_percent_income":
        return f"{v:.1%}"
    if feature == "loan_int_rate":
        return f"{v:.2f}%"
    if feature == "loan_grade":
        grade_map = {6: "A", 5: "B", 4: "C", 3: "D", 2: "E", 1: "F", 0: "G"}
        return grade_map.get(int(v), str(int(v)))
    if feature == "cb_person_default_on_file":
        return "Yes" if int(v) == 1 else "No"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Loan Rejection Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# LOAD ARTIFACTS ONCE
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model and explainers...")
def get_artifacts():
    return load_artifacts()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("Explainable Loan Rejection Assistant")
st.caption(
    "Submit an application to get a decision, the drivers behind a rejection, "
    "concrete changes that could flip it, and a plain-language explanation."
)
st.divider()


# ─────────────────────────────────────────────
# SIDEBAR — APPLICATION FORM
# ─────────────────────────────────────────────

st.sidebar.header("Loan application")
st.sidebar.caption("Defaults match a near-boundary sample applicant.")

with st.sidebar:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=26)
    person_income = st.number_input(
        "Annual income ($)",
        min_value=0,
        max_value=500_000,
        value=31_200,
        step=1_000,
    )
    person_emp_length = st.number_input(
        "Employment length (years)",
        min_value=0,
        max_value=50,
        value=8,
    )
    loan_grade = st.selectbox(
        "Credit grade",
        options=["A", "B", "C", "D", "E", "F", "G"],
        index=0,
        help=GRADE_HELP,
    )
    loan_amnt = st.number_input(
        "Loan amount requested ($)",
        min_value=500,
        max_value=100_000,
        value=5_000,
        step=500,
    )
    loan_int_rate = st.slider(
        "Interest rate (%)",
        min_value=5.0,
        max_value=30.0,
        value=8.6,
        step=0.1,
    )
    loan_intent = st.selectbox(
        "Loan purpose",
        options=[
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ],
        index=1,
    )
    person_home_ownership = st.selectbox(
        "Home ownership",
        options=["RENT", "OWN", "MORTGAGE", "OTHER"],
        index=0,
    )
    cb_person_default_on_file = st.selectbox(
        "Previous default on record?",
        options=["No", "Yes"],
        index=0,
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit history length (years)",
        min_value=0,
        max_value=30,
        value=2,
    )

    loan_percent_income = (
        loan_amnt / person_income if person_income > 0 else 0.0
    )
    ratio_delta = None
    if loan_percent_income > 0.20:
        ratio_delta = "Above 20% preferred max"
    st.metric("Loan as % of income", f"{loan_percent_income:.1%}", ratio_delta)

    submitted = st.button(
        "Check my application",
        type="primary",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# MAIN — RESULTS
# ─────────────────────────────────────────────

if submitted:
    input_dict = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": (
            1 if cb_person_default_on_file == "Yes" else 0
        ),
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
    }

    try:
        artifacts = get_artifacts()
    except Exception as e:
        st.error(f"Could not load model artifacts: {e}")
        st.stop()

    with st.spinner("Analyzing application (prediction, SHAP, DiCE, Gemini)..."):
        # Quiet noisy stdout from ML libs during the UI run
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            try:
                sys.stdout = devnull
                result = run_application(input_dict, artifacts)
            finally:
                sys.stdout = old_stdout

    st.session_state["last_result"] = result
    st.session_state["last_input"] = input_dict

result = st.session_state.get("last_result")
last_input = st.session_state.get("last_input")

if result is None:
    st.info(
        "Fill in the sidebar and click **Check my application** "
        "to see your decision and explanation."
    )
else:
    st.subheader("Decision")

    if result["decision"] == "APPROVED":
        st.success(
            f"**APPROVED** — approval confidence "
            f"{result['approval_prob']:.1%}"
        )
        if result.get("message"):
            st.write(result["message"])
    else:
        st.error(
            f"**REJECTED** — rejection confidence "
            f"{result['rejection_prob']:.1%}"
        )

        if last_input:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Annual income", f"${last_input['person_income']:,.0f}")
            c2.metric("Loan requested", f"${last_input['loan_amnt']:,.0f}")
            c3.metric(
                "Loan / income",
                f"{last_input['loan_percent_income']:.1%}",
            )
            c4.metric(
                "Approval chance",
                f"{result['approval_prob']:.1%}",
            )

        st.divider()
        st.subheader("Why was the application rejected?")

        shap_df = result.get("shap_explanation")
        if shap_df is not None and len(shap_df):
            risk_factors = (
                shap_df[shap_df["SHAP"] > 0]
                .sort_values("SHAP", ascending=False)
                .head(5)
            )
            prot_factors = (
                shap_df[shap_df["SHAP"] < 0]
                .sort_values("SHAP", ascending=True)
                .head(3)
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Risk factors** (pushed toward rejection)")
                for _, row in risk_factors.iterrows():
                    feat = row["Feature"]
                    st.markdown(
                        f"- **{label_feature(feat)}**: "
                        f"{format_value(feat, row['Value'])} "
                        f"(impact +{row['SHAP']:.3f})"
                    )
            with col2:
                st.markdown("**Protective factors** (helped the application)")
                if len(prot_factors):
                    for _, row in prot_factors.iterrows():
                        feat = row["Feature"]
                        st.markdown(
                            f"- **{label_feature(feat)}**: "
                            f"{format_value(feat, row['Value'])} "
                            f"(impact {row['SHAP']:.3f})"
                        )
                else:
                    st.caption("No strong protective factors for this profile.")

            # Compact SHAP contribution chart (top drivers)
            plot_df = (
                pd.concat([risk_factors, prot_factors])
                .drop_duplicates(subset=["Feature"])
                .sort_values("SHAP")
            )
            if len(plot_df):
                fig, ax = plt.subplots(figsize=(8, 3.8))
                colors = [
                    "#b45309" if v > 0 else "#0f766e" for v in plot_df["SHAP"]
                ]
                labels = [label_feature(f) for f in plot_df["Feature"]]
                ax.barh(labels, plot_df["SHAP"], color=colors)
                ax.axvline(0, color="#64748b", linewidth=0.8)
                ax.set_xlabel("SHAP contribution (positive = more risk)")
                ax.set_title("Feature contributions to this decision")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
        else:
            st.warning("SHAP explanation was not available for this run.")

        st.divider()
        st.subheader("What could change the decision?")

        if result.get("counterfactuals"):
            st.markdown(result["counterfactuals"])
        else:
            st.warning(
                "Could not generate specific recommendations "
                "for this application profile."
            )

        st.divider()
        st.subheader("Detailed explanation")

        explanation = result.get("explanation")
        if explanation:
            st.markdown(explanation)
        else:
            st.info(
                "Gemini explanation unavailable. Set `GEMINI_API_KEY` in "
                "a project-root `.env` file, then submit again."
            )

        st.divider()
        st.subheader("Explanation faithfulness")
        faith = result.get("faithfulness")
        if faith and not faith.get("empty_explanation"):
            f1, f2, f3 = st.columns(3)
            f1.metric("SHAP coverage", f"{faith['coverage']:.0%}")
            f2.metric("Grounding precision", f"{faith['precision']:.0%}")
            f3.metric("Hallucination rate", f"{faith['hallucination_rate']:.0%}")
            st.caption(
                f"LLM mode: `{result.get('llm_mode', 'grounded')}` — "
                "precision is the share of mentioned features that appear in "
                "top SHAP risks or DiCE recourse features."
            )
        else:
            st.caption("Faithfulness scores appear when a Gemini explanation is generated.")

        st.divider()
        st.subheader("Fairness audit (live)")
        fairness = result.get("fairness") or {}
        disparity = fairness.get("age_disparity")
        passed = fairness.get("passed")
        if disparity is not None:
            if passed:
                st.success(fairness.get("note", "Age fairness audit passed."))
            else:
                st.warning(fairness.get("note", "Age fairness needs review."))
            groups = fairness.get("groups") or []
            if groups:
                import pandas as _pd
                gdf = _pd.DataFrame(groups)
                st.dataframe(gdf, use_container_width=True, hide_index=True)
        else:
            st.info(fairness.get("note", "Fairness audit unavailable."))
        st.caption(fairness.get("gender_note", "Gender data not in dataset."))


st.divider()
st.caption(
    "Research prototype — XGBoost + SHAP + DiCE + grounded Gemini. "
    "Paper angle: SHAP-grounded faithful explanations & recourse quality metrics."
)
```

### `config/config.yaml`

```yaml
# Explainable Loan Rejection Assistant — runtime config

paths:
  data: data/loan_dataset.csv
  model: models/loan_model.pkl
  calibrated_model: models/loan_model_calibrated.pkl
  experiment_results: results/experiment_results.json
  figures: models

model:
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.1
  scale_pos_weight: 3.0
  random_state: 42
  eval_metric: auc

data:
  test_size: 0.2
  random_state: 42

shap:
  background_size: 100

dice:
  n_counterfactuals: 3
  method: random

fairness:
  disparity_threshold: 0.10

llm:
  model: gemini-2.5-flash
  # grounded = only cite SHAP/DiCE features (paper method)
  mode: grounded

evaluation:
  # Max rejected applicants to score in batch experiments
  max_samples: 100
  random_state: 42

api:
  host: 0.0.0.0
  port: 8000
```

### `data/loan_dataset.csv`

```csv
person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length
22,59000,RENT,123.0,PERSONAL,D,35000,16.02,1,0.59,Y,3
21,9600,OWN,5.0,EDUCATION,B,1000,11.14,0,0.1,N,2
25,9600,MORTGAGE,1.0,MEDICAL,C,5500,12.87,1,0.57,N,3
23,65500,RENT,4.0,MEDICAL,C,35000,15.23,1,0.53,N,2
24,54400,RENT,8.0,MEDICAL,C,35000,14.27,1,0.55,Y,4
21,9900,OWN,2.0,VENTURE,A,2500,7.14,1,0.25,N,2
26,77100,RENT,8.0,EDUCATION,B,35000,12.42,1,0.45,N,3
24,78956,RENT,5.0,MEDICAL,B,35000,11.11,1,0.44,N,4
24,83000,RENT,8.0,PERSONAL,A,35000,8.9,1,0.42,N,2
21,10000,OWN,6.0,VENTURE,D,1600,14.74,1,0.16,N,3
22,85000,RENT,6.0,VENTURE,B,35000,10.37,1,0.41,N,4
21,10000,OWN,2.0,HOMEIMPROVEMENT,A,4500,8.63,1,0.45,N,2
23,95000,RENT,2.0,VENTURE,A,35000,7.9,1,0.37,N,2
26,108160,RENT,4.0,EDUCATION,E,35000,18.39,1,0.32,N,4
23,115000,RENT,2.0,EDUCATION,A,35000,7.9,0,0.3,N,4
23,500000,MORTGAGE,7.0,DEBTCONSOLIDATION,B,30000,10.65,0,0.06,N,3
23,120000,RENT,0.0,EDUCATION,A,35000,7.9,0,0.29,N,4
23,92111,RENT,7.0,MEDICAL,F,35000,20.25,1,0.32,N,4
23,113000,RENT,8.0,DEBTCONSOLIDATION,D,35000,18.25,1,0.31,N,4
24,10800,MORTGAGE,8.0,EDUCATION,B,1750,10.99,1,0.16,N,2
25,162500,RENT,2.0,VENTURE,A,35000,7.49,0,0.22,N,4
25,137000,RENT,9.0,PERSONAL,E,34800,16.77,0,0.25,Y,2
22,65000,RENT,4.0,EDUCATION,D,34000,17.58,1,0.52,N,4
24,10980,OWN,0.0,PERSONAL,A,1500,7.29,0,0.14,N,3
22,80000,RENT,3.0,PERSONAL,D,33950,14.54,1,0.42,Y,4
24,67746,RENT,8.0,HOMEIMPROVEMENT,C,33000,12.68,1,0.49,N,3
21,11000,MORTGAGE,3.0,VENTURE,E,4575,17.74,1,0.42,Y,3
23,11000,OWN,0.0,PERSONAL,A,1400,9.32,0,0.13,N,3
24,65000,RENT,6.0,HOMEIMPROVEMENT,B,32500,9.99,1,0.5,N,3

... [32552 more lines omitted; full file has 32582 lines] ...
```

### `docs/MODEL_CARD.md`

```markdown
# Model Card — Explainable Loan Rejection Assistant

## Model details

- **Developers:** Project maintainers (portfolio / research prototype)
- **Model:** XGBoost binary classifier (`loan_status`: 0 = good standing, 1 = default risk)
- **Version:** 1.1 (grounded explanations + evaluation harness)
- **License:** Prototype — not for production credit decisions

## Intended use

- Demonstrate explainable rejection + recourse + NL explanation for education, demos, and applied-XAI research.
- **Out of scope:** Real underwriting, regulatory filing, or automated adverse-action notices without legal review.

## Training data

- Tabular consumer-loan style dataset (`data/loan_dataset.csv`)
- Cleaning: impossible ages / employment lengths removed; median imputation; duplicates dropped
- Encoding: ordinal loan grade; one-hot home ownership & intent; binary prior default
- Split: 80/20 stratified

## Evaluation

- Primary: ROC-AUC (imbalance-aware)
- Secondary: class-wise precision / recall / F1
- Baselines: logistic regression, random forest (`src/baselines.py`)
- Leakage ablation: drop `loan_grade`, `loan_int_rate` and re-measure AUC
- Fairness: demographic parity / equal opportunity style rates by age group
- Explanation: faithfulness + DiCE recourse metrics (`src/evaluation/`)

## Ethical considerations

- ECOA-relevant attributes: age audited; **gender / race not present** — incomplete fairness coverage
- Explanations must not invent protected-class reasons; grounded prompting constrains cited features
- Counterfactuals that require years of employment growth are delayed actions — reported via actionability delay

## Caveats

- Probabilities may be miscalibrated; optional `CalibratedClassifierCV` in `baselines.py`
- LLM text can still drift; faithfulness scores are heuristics, not legal compliance proofs
```

### `docs/PAPER_DRAFT.md`

```markdown
# Workshop Paper Draft

**Title:** SHAP-Grounded Faithful Explanations for Credit Recourse

**Suggested venues:** FAccT Demo / Workshop, AAAI Demo, NeurIPS XAI workshop, ACM COMPASS workshop, regional AI+Society tracks — *not* NeurIPS/ICML main conference.

## Abstract (draft)

Automated credit models increasingly accompany decisions with natural-language explanations. Large language models (LLMs) can make SHAP outputs readable, but they may mention features that did not drive the decision. We present an end-to-end credit rejection assistant that couples an XGBoost risk model with SHAP attributions, DiCE counterfactual recourse, and **grounded LLM prompting** that restricts cited factors to SHAP/DiCE evidence. We introduce automatic **faithfulness** metrics (coverage, precision, hallucination rate) and **recourse quality** metrics (validity, sparsity, proximity, actionability delay). Offline template ablations show grounding raises precision and lowers hallucination rate; optional Gemini comparisons test whether the same pattern holds for LLMs. Age-group fairness disparities are reported under an ECOA-oriented audit. The contribution is an evaluated system and metrics suite for faithful, actionable credit explanations—not a new predictor architecture.

## 1. Introduction

Adverse credit decisions require understandable reasons and feasible next steps. Prior work studies SHAP for local feature attribution and DiCE for diverse counterfactuals. Recent systems verbalize attributions with LLMs, but faithfulness is rarely measured. We ask:

> Does constraining an explanation generator to SHAP/DiCE features reduce unsupported feature mentions while preserving coverage of true risk drivers?

## 2. Related work (fill citations when submitting)

- SHAP / TreeExplainer (Lundberg & Lee)
- DiCE counterfactuals (Mothilal et al.)
- Algorithmic recourse and actionability (Ustun, Karimi, et al.)
- Faithfulness / simulatability of explanations (Jacovi & Goldberg; Chan et al.)
- Fairness in lending / ECOA discussion

## 3. Method

### 3.1 Predictor

XGBoost classifier with class weighting for imbalance. Baselines: logistic regression, random forest. Leakage ablation removes `loan_grade` and `loan_int_rate`.

### 3.2 Local attribution & recourse

SHAP TreeExplainer for local drivers. DiCE with permitted ranges that are directional (income↑, loan↓) and percentile-bounded.

### 3.3 Grounded verbalization

Two prompt modes:

- **Free:** SHAP + optional DiCE text, no hard citation limit
- **Grounded:** explicit allow-list of top SHAP risks (+ DiCE changes); forbid inventing other reasons

### 3.4 Metrics

**Faithfulness**

- Coverage = |mentioned ∩ top-k SHAP risks| / k  
- Precision = |mentioned ∩ allowed| / |mentioned|  
- Hallucination rate = 1 − Precision  

**Recourse**

- Validity, sparsity, proximity, actionability delay (employment-length changes penalized)

**Fairness**

- Age-group approval-rate disparity; 10% threshold; gender unavailable (limitation)

## 4. Experiments

```bash
cd src
python -m evaluation.run_experiments --max-samples 100
python -m evaluation.run_experiments --max-samples 30 --with-llm
python baselines.py
```

Report tables:

1. Baseline AUC / F1  
2. Leakage ablation  
3. Recourse means  
4. Template free vs grounded faithfulness  
5. LLM free vs grounded (if API available)  
6. Age fairness disparity  

## 5. Expected findings (verify with your runs)

- Grounded templates: higher precision, lower hallucination, similar coverage  
- Many CFs valid near decision boundary; actionability delay often > 0 when income/tenure change  
- Age disparity may pass or fail 10% — report honestly  
- Removing grade/rate may drop AUC — discuss underwriting leakage

## 6. Limitations

Public data; alias-based faithfulness; no large human study yet; LLM non-determinism; incomplete protected attributes.

## 7. What you still need for submission

1. Run experiments and paste numbers into tables  
2. Add 15–25 real related-work citations  
3. Optional: small user study (n≥20) rating clarity / trust / actionability for free vs grounded  
4. Figures: system diagram, waterfall, faithfulness bar chart  
5. Ethics statement + data availability  

## Contribution statement (for reviewers)

We do **not** claim a new learning algorithm. We claim (1) a grounded verbalization protocol for credit recourse, (2) an automatic faithfulness + actionability evaluation suite, and (3) a reproducible system combining prediction, attribution, recourse, NL explanation, and fairness reporting.
```

### `models/baseline_comparison.json`

```json
{
  "logistic_regression": {
    "roc_auc": 0.8689949417485621,
    "f1_default": 0.6272674078408426,
    "precision_default": 0.5214007782101168,
    "recall_default": 0.7870778267254038,
    "f1_good": 0.8614012184508268
  },
  "random_forest": {
    "roc_auc": 0.938710105778563,
    "f1_default": 0.8057722308892356,
    "precision_default": 0.8594009983361065,
    "recall_default": 0.7584434654919237,
    "f1_good": 0.9504280310571371
  },
  "xgboost": {
    "roc_auc": 0.9509298068465084,
    "f1_default": 0.821483771251932,
    "precision_default": 0.867047308319739,
    "recall_default": 0.7804698972099853,
    "f1_good": 0.9539014168828577
  }
}
```

### `notebooks/notebooks01_data_exploration.ipynb`

```json
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "w9AVBVNCZs7H"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j_9_0u0aYCaa",
        "outputId": "fc35626e-4730-4f4b-e8a4-56f83a3cd643"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loaded successfully\n",
            "(32581, 12)\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# Data Loading\n",
        "df = pd.read_csv(\"data/loan_data.csv\")\n",
        "\n",
        "print(\"Loaded successfully\")\n",
        "print(df.shape)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Analysing dataset\n",
        "df.head()\n",
        "df.tail()\n",
        "df.describe()\n",
        "df.isna().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 460
        },
        "id": "jJ6V4trwYt-T",
        "outputId": "f08aa391-4e8b-4f00-f7ef-93ad59b302e0"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "person_age                       0\n",
              "person_income                    0\n",
              "person_home_ownership            0\n",
              "person_emp_length              895\n",
              "loan_intent                      0\n",
              "loan_grade                       0\n",
              "loan_amnt                        0\n",
              "loan_int_rate                 3116\n",
              "loan_status                      0\n",
              "loan_percent_income              0\n",
              "cb_person_default_on_file        0\n",
              "cb_person_cred_hist_length       0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>person_age</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>person_income</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>person_home_ownership</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>person_emp_length</th>\n",
              "      <td>895</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>loan_intent</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>loan_grade</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>loan_amnt</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>loan_int_rate</th>\n",
              "      <td>3116</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>loan_status</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>loan_percent_income</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>cb_person_default_on_file</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>cb_person_cred_hist_length</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Finding outliers\n",
        "df.describe()\n",
        "df[\"person_age\"].max()\n",
        "df[\"person_emp_length\"].max()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MgNSUfv6Yw7S",
        "outputId": "161b42b4-2b5f-4c60-c866-4b99c2970f9b"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "123.0"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Understanding Distributions\n",
        "df[\"loan_status\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "hsDdeDXUZX5V",
        "outputId": "53482cd3-a0ef-4363-8402-b3814cf89873"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "loan_status\n",
              "0    25473\n",
              "1     7108\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>loan_status</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>25473</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>7108</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Understanding Cateogries\n",
        "df[\"loan_grade\"].unique()\n",
        "df[\"loan_intent\"].unique()\n",
        "df[\"person_home_ownership\"].unique()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f1NYdCXSZfe2",
        "outputId": "7ed96bce-81be-46c3-cfdb-d3b91d8d4940"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "fBEmNsbrZnf2"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
```

### `README.md`

```markdown
# Explainable Loan Rejection Assistant

End-to-end credit decision prototype: **XGBoost → SHAP → DiCE recourse → grounded Gemini explanation → ECOA-oriented fairness audit**, with a **paper-ready evaluation harness** for explanation faithfulness and recourse quality.

## Why this project

Most loan XAI demos stop at SHAP plots. This system also:

1. Generates **actionable counterfactuals** (what to change)
2. Turns them into **applicant-facing language** with **grounding constraints** (only cite SHAP/DiCE features)
3. Scores **faithfulness** (coverage / precision / hallucination rate)
4. Scores **recourse quality** (validity, sparsity, proximity, actionability delay)
5. Surfaces a **live age fairness audit** in the UI and API

**Research angle (workshop / applied XAI track):** *SHAP-grounded faithful LLM explanations for credit recourse* — does constrained prompting reduce unsupported feature mentions without sacrificing risk-factor coverage?

## Architecture

```text
Applicant form (Streamlit / FastAPI)
        │
        ▼
   pipeline.run_application
        │
        ├── XGBoost predict (+ calibrated option via baselines.py)
        ├── SHAP TreeExplainer (local drivers)
        ├── DiCE counterfactuals (actionable ranges)
        ├── Gemini (grounded | free | off)
        ├── Faithfulness scorer
        └── Fairness audit (computed at startup on test set)
```

## Quick start

```bash
# From project root
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Place dataset at data/loan_dataset.csv, then train once:
cd src
python data.py
python model.py

# UI
cd ..
streamlit run app.py

# API
uvicorn api:app --reload

# Unit tests
pytest -q

# Paper experiments (offline templates + recourse; no API key needed)
cd src
python -m evaluation.run_experiments --max-samples 30

# Optional: free vs grounded Gemini comparison
python -m evaluation.run_experiments --max-samples 20 --with-llm
```

Copy `.env.example` → `.env` and set `GEMINI_API_KEY` for natural-language explanations.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI |
| `api.py` | FastAPI `/predict`, `/fairness`, `/health` |
| `config/config.yaml` | Runtime config |
| `src/data.py` | Load / clean / encode / split |
| `src/model.py` | Train / evaluate / persist XGBoost |
| `src/baselines.py` | LR / RF / XGB comparison, calibration, leak ablation |
| `src/explainer.py` | SHAP + free/grounded LLM prompts |
| `src/counterfactuals.py` | DiCE recourse |
| `src/fairness.py` | Age / intent fairness |
| `src/evaluation/` | Faithfulness, recourse metrics, experiment runner |
| `src/pipeline.py` | Single entry point for UI + API |
| `docs/MODEL_CARD.md` | Model card |
| `docs/PAPER_DRAFT.md` | Workshop paper draft |
| `tests/` | Unit tests |

## Key metrics (how to talk about results)

**Model:** ROC-AUC, precision/recall on default class, confusion matrix.

**Recourse:** validity (fraction of CFs that flip to approval), sparsity, proximity, actionability delay (time-gated changes score higher).

**Faithfulness:** coverage of top SHAP risks mentioned; precision of mentioned features that are grounded; hallucination rate = 1 − precision.

**Fairness:** age-group approval-rate disparity vs 10% threshold; gender not in dataset (limitation).

## Honest limits

- Public credit dataset; `loan_grade` / `loan_int_rate` may leak underwriting — see leakage ablation in `baselines.py`.
- Automatic faithfulness uses alias matching (approximate).
- Not a production credit system; research / portfolio prototype only.
```

### `requirements.txt`

```text
pandas>=2.0,<3
numpy>=1.24,<3
scikit-learn>=1.3,<2
xgboost>=2.0,<4
shap>=0.44,<1
dice-ml>=0.11,<1
google-genai>=1.0,<2
python-dotenv>=1.0,<2
streamlit>=1.28,<2
joblib>=1.3,<2
matplotlib>=3.7,<4
PyYAML>=6.0,<7
fastapi>=0.110,<1
uvicorn[standard]>=0.27,<1
pydantic>=2.0,<3
pytest>=8.0,<9
```

### `results/experiment_results.json`

```json
{
  "summary": {
    "n_rejected_evaluated": 20,
    "recourse": {
      "validity": 0.7,
      "mean_sparsity": 1.9761904761904765,
      "mean_proximity": 0.1609916498902698,
      "mean_actionability_delay": 0.41527777777777775,
      "n": 20
    },
    "template_free": {
      "coverage": 0.6900000000000001,
      "precision": 0.8098015873015875,
      "hallucination_rate": 0.1901984126984127,
      "n": 20
    },
    "template_grounded": {
      "coverage": 0.6799999999999999,
      "precision": 0.975,
      "hallucination_rate": 0.025,
      "n": 20
    },
    "faithfulness_free": null,
    "faithfulness_grounded": null,
    "with_llm": false,
    "paper_claim": "Grounded prompting / templates raise feature-mention precision and lower hallucination_rate vs unconstrained text, while preserving SHAP risk coverage; recourse is reported via validity, sparsity, proximity, and actionability delay.",
    "template_deltas": {
      "precision_delta": 0.1651984126984125,
      "hallucination_delta": -0.1651984126984127,
      "coverage_delta": -0.01000000000000012
    }
  },
  "recourse_per_applicant": [
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.16478057073666505,
      "mean_actionability_delay": 0.39999999999999997,
      "idx": 4850
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.10766178632058591,
      "mean_actionability_delay": 0.3,
      "idx": 4530
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.12535254280141564,
      "mean_actionability_delay": 0.2333333333333333,
      "idx": 1297
    },
    {
      "n_cfs": 0,
      "validity": 0.0,
      "mean_sparsity": null,
      "mean_proximity": null,
      "mean_actionability_delay": null,
      "idx": 6105
    },
    {
      "n_cfs": 0,
      "validity": 0.0,
      "mean_sparsity": null,
      "mean_proximity": null,
      "mean_actionability_delay": null,
      "idx": 5295
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.18109772143277802,
      "mean_actionability_delay": 0.39999999999999997,
      "idx": 4817
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 1.6666666666666667,
      "mean_proximity": 0.18271135807447147,
      "mean_actionability_delay": 0.5333333333333333,
      "idx": 766
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 1.3333333333333333,
      "mean_proximity": 0.15551625833981594,
      "mean_actionability_delay": 0.65,
      "idx": 520
    },
    {
      "n_cfs": 0,
      "validity": 0.0,
      "mean_sparsity": null,
      "mean_proximity": null,
      "mean_actionability_delay": null,
      "idx": 4117
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.09332016451433135,
      "mean_actionability_delay": 0.28333333333333327,
      "idx": 3157
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.06516532910153089,
      "mean_actionability_delay": 0.3499999999999999,
      "idx": 2680
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 1.6666666666666667,
      "mean_proximity": 0.08740388737562223,
      "mean_actionability_delay": 0.3,
      "idx": 514
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.2589798792770224,
      "mean_actionability_delay": 0.7999999999999999,
      "idx": 2809
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 2.0,
      "mean_proximity": 0.19790957353158048,
      "mean_actionability_delay": 0.28333333333333327,
      "idx": 5344
    },
    {
      "n_cfs": 0,
      "validity": 0.0,
      "mean_sparsity": null,
      "mean_proximity": null,
      "mean_actionability_delay": null,
      "idx": 559
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 1.6666666666666667,
      "mean_proximity": 0.11684429797094113,
      "mean_actionability_delay": 0.4666666666666666,
      "idx": 4640
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 1.6666666666666667,
      "mean_proximity": 0.21341018369851641,
      "mean_actionability_delay": 0.3333333333333333,
      "idx": 4961
    },
    {
      "n_cfs": 0,
      "validity": 0.0,
      "mean_sparsity": null,
      "mean_proximity": null,
      "mean_actionability_delay": null,
      "idx": 2704
    },
    {
      "n_cfs": 0,
      "validity": 0.0,
      "mean_sparsity": null,
      "mean_proximity": null,
      "mean_actionability_delay": null,
      "idx": 4363
    },
    {
      "n_cfs": 3,
      "validity": 1.0,
      "mean_sparsity": 3.6666666666666665,
      "mean_proximity": 0.3037295452884999,
      "mean_actionability_delay": 0.4805555555555556,
      "idx": 3269
    }
  ]
}
```

### `scripts/generate_full_dump.py`

```python
"""Generate docs/FULL_PROJECT_DUMP.md with all project text sources."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "FULL_PROJECT_DUMP.md"

TEXT_EXTS = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json",
    ".example", ".gitignore", ".csv", ".ipynb",
}
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv",
    ".pytest_cache", "pytest-cache-files-pqbpjo6b", "node_modules",
}
SKIP_FILES = {".env"}
CSV_MAX_LINES = 30


def main() -> None:
    parts: list[str] = []
    parts.append("# Loan-Rejection - FULL PROJECT DUMP")
    parts.append("")
    parts.append(
        "Generated for ChatGPT / external AI handoff. Contains project "
        "structure, all text source files, configs, docs, experiment "
        "results, and captured terminal/experiment output."
    )
    parts.append("")
    parts.append(
        "**Excluded (secrets / binaries / caches):** `.env`, raw `*.pkl` / "
        "image bytes, `__pycache__`, `.git`, `.venv`. Binary assets are "
        "listed by path/size only. Dataset CSV is truncated to header + "
        "sample rows (full file: `data/loan_dataset.csv`)."
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    parts.append("## 1. Project structure")
    parts.append("")
    parts.append("```text")
    for p in sorted(ROOT.rglob("*")):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        if p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            if rel == "docs/FULL_PROJECT_DUMP.md":
                continue
            parts.append(f"{rel}  ({p.stat().st_size} bytes)")
    parts.append("```")
    parts.append("")

    files: list[Path] = []
    for p in sorted(ROOT.rglob("*")):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        if not p.is_file():
            continue
        if p.name in SKIP_FILES:
            continue
        if p.resolve() == OUT.resolve():
            continue
        if p.suffix.lower() in TEXT_EXTS or p.name in {".gitignore", ".env.example"}:
            files.append(p)

    parts.append("## 2. File contents")
    parts.append("")

    lang_map = {
        ".py": "python",
        ".md": "markdown",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".example": "bash",
        ".gitignore": "gitignore",
        ".csv": "csv",
        ".ipynb": "json",
    }

    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        parts.append(f"### `{rel}`")
        parts.append("")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            parts.append(f"_Could not read: {e}_")
            parts.append("")
            continue

        if p.suffix.lower() == ".csv":
            lines = text.splitlines()
            shown = lines[:CSV_MAX_LINES]
            body = "\n".join(shown)
            if len(lines) > CSV_MAX_LINES:
                body += (
                    f"\n\n... [{len(lines) - CSV_MAX_LINES} more lines "
                    f"omitted; full file has {len(lines)} lines] ..."
                )
            parts.append("```csv")
            parts.append(body)
            parts.append("```")
        else:
            lang = lang_map.get(p.suffix.lower(), "")
            parts.append(f"```{lang}")
            parts.append(text.rstrip("\n"))
            parts.append("```")
        parts.append("")

    parts.append("## 3. Binary / non-text assets (inventory only)")
    parts.append("")
    parts.append("| Path | Size (bytes) | Notes |")
    parts.append("|------|-------------:|-------|")
    for p in sorted(ROOT.rglob("*")):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in {".pkl", ".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            rel = p.relative_to(ROOT).as_posix()
            parts.append(f"| `{rel}` | {p.stat().st_size} | binary artifact |")
    parts.append("")

    parts.append("## 4. Captured terminal / experiment run (`--with-llm`)")
    parts.append("")
    parts.append("Command:")
    parts.append("")
    parts.append("```bash")
    parts.append("cd src")
    parts.append("python -m evaluation.run_experiments --max-samples 20 --with-llm")
    parts.append("```")
    parts.append("")
    parts.append("Exit code: `0`")
    parts.append("")
    parts.append("### 4.1 Startup log")
    parts.append("")
    parts.append("```text")
    parts.append(
        """Loaded dataset: 32581 rows and 12 columns
Removed 5 rows with impossible age values
Dataset after outlier removal: 31679 rows
Removed 157 duplicate rows
Clean: no missing values
Encoded: 18 columns, all numeric
Columns: [person_age, person_income, person_emp_length, loan_grade, loan_amnt,
loan_int_rate, loan_status, loan_percent_income, cb_person_default_on_file,
cb_person_cred_hist_length, person_home_ownership_OTHER,
person_home_ownership_OWN, person_home_ownership_RENT,
loan_intent_EDUCATION, loan_intent_HOMEIMPROVEMENT, loan_intent_MEDICAL,
loan_intent_PERSONAL, loan_intent_VENTURE]

Features (X): (31522, 17)
Target   (y): (31522,)
Class distribution: 0=24715, 1=6807
Imbalance ratio: 3.6:1
Training set: 25217 rows
Test set:     6305 rows

Model loaded from models/loan_model.pkl
Explainer created
Baseline (expected value): -1.9500
DiCE explainer created successfully"""
    )
    parts.append("```")
    parts.append("")
    parts.append("### 4.2 Per-applicant pattern")
    parts.append("")
    parts.append("For each of 20 rejected applicants:")
    parts.append("1. Computing SHAP values for 1 applicants... shape `(1, 17)`")
    parts.append("2. DiCE counterfactual suggestions OR `No Counterfactuals found`")
    parts.append("3. LLM call failed with either 404 or 429")
    parts.append("")
    parts.append("**404 NOT_FOUND** (early samples):")
    parts.append("")
    parts.append("```text")
    parts.append(
        "LLM failed on idx=...: 404 NOT_FOUND. "
        "{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash "
        "is no longer available to new users. Please update your code to use a "
        "newer model for the latest features and improvements.', "
        "'status': 'NOT_FOUND'}}"
    )
    parts.append("```")
    parts.append("")
    parts.append("404 idxs: 4850, 4530, 1297, 6105, 5295, 4817, 520")
    parts.append("")
    parts.append("**429 RESOURCE_EXHAUSTED** (later samples):")
    parts.append("")
    parts.append("```text")
    parts.append(
        "LLM failed on idx=...: 429 RESOURCE_EXHAUSTED. You exceeded your "
        "current quota... Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 5, model: gemini-2.5-flash"
    )
    parts.append("```")
    parts.append("")
    parts.append(
        "429 idxs (examples): 766, 4117, 3157, 2680, 514, 2809, 5344, "
        "559, 4640, 4961, 2704, 4363, 3269"
    )
    parts.append("")
    parts.append("DiCE failed (no CFs) on idxs: 6105, 5295, 4117, 559, 2704, 4363")
    parts.append("")
    parts.append("### 4.3 Example counterfactual blocks printed")
    parts.append("")
    parts.append("```text")
    parts.append(
        """Applicant idx 4850:
Option 1:
  - Increase annual income from $32,000 to $92,613
  - Reduce loan-to-income ratio from 36.0% to 30.0%
Option 2:
  - Increase annual income from $32,000 to $94,308
  - Reduce loan-to-income ratio from 36.0% to 10.0%
Option 3:
  - Increase annual income from $32,000 to $55,750
  - Reduce loan-to-income ratio from 36.0% to 10.0%

Applicant idx 4530:
Option 1:
  - Reduce loan-to-income ratio from 31.0% to 20.0%
  - Lower interest rate from 10.62% to 7.87%
Option 2:
  - Reduce loan-to-income ratio from 31.0% to 30.0%
  - Lower interest rate from 10.62% to 7.10%
Option 3:
  - Increase annual income from $22,680 to $72,470
  - Reduce loan-to-income ratio from 31.0% to 10.0%

Applicant idx 1297:
Option 1:
  - Reduce loan request from $12,800 to $6,393
  - Reduce loan-to-income ratio from 43.0% to 30.0%
Option 2:
  - Increase annual income from $30,000 to $41,808
  - Reduce loan-to-income ratio from 43.0% to 30.0%
Option 3:
  - Reduce loan-to-income ratio from 43.0% to 10.0%
  - Lower interest rate from 14.79% to 6.29%

Applicant idx 4817:
Option 1:
  - Increase annual income from $15,000 to $61,578
  - Reduce loan-to-income ratio from 40.0% to 20.0%

Applicant idx 766:
Option 1:
  - Increase annual income from $18,700 to $135,180
Option 2:
  - Increase annual income from $18,700 to $67,846
  - Build employment history from 3 to 9 years
Option 3:
  - Reduce loan request from $3,600 to $861
  - Reduce loan-to-income ratio from 19.0% to 10.0%

Applicant idx 520:
Option 1:
  - Increase annual income from $15,600 to $96,618

Applicant idx 3157:
Option 1:
  - Reduce loan request from $18,000 to $9,606
  - Reduce loan-to-income ratio from 37.0% to 30.0%

Applicant idx 2680:
Option 1:
  - Reduce loan-to-income ratio from 39.0% to 30.0%
  - Lower interest rate from 11.36% to 9.48%

Applicant idx 514:
Option 1:
  - Reduce loan-to-income ratio from 39.0% to 20.0%
  - Lower interest rate from 12.73% to 8.77%

Applicant idx 2809:
Option 1:
  - Increase annual income from $64,000 to $86,635
  - Build employment history from 0 to 4 years

Applicant idx 5344:
Option 1:
  - Reduce loan request from $15,000 to $5,590
  - Reduce loan-to-income ratio from 50.0% to 20.0%

Applicant idx 4640:
Option 1:
  - Increase annual income from $34,000 to $125,917

Applicant idx 4961:
Option 1:
  - Reduce loan request from $20,000 to $12,449
  - Build employment history from 0 to 4 years

Applicant idx 3269:
Option 1:
  - Increase annual income from $43,600 to $132,953
  - Build employment history from 4 to 8 years
  - Reduce loan-to-income ratio from 23.0% to 10.0%
  - Lower interest rate from 14.96% to 6.79%"""
    )
    parts.append("```")
    parts.append("")
    parts.append("### 4.4 Final printed summary")
    parts.append("")
    exp = ROOT / "results" / "experiment_results.json"
    if exp.exists():
        data = json.loads(exp.read_text(encoding="utf-8"))
        parts.append("```json")
        parts.append(json.dumps(data.get("summary", data), indent=2))
        parts.append("```")
    parts.append("")
    parts.append("### 4.5 Pytest")
    parts.append("")
    parts.append("```text")
    parts.append("python -m pytest tests -q")
    parts.append("....                                                                     [100%]")
    parts.append("4 passed, 3 warnings in ~20s")
    parts.append("```")
    parts.append("")
    parts.append("## 5. How to run")
    parts.append("")
    parts.append("```bash")
    parts.append("pip install -r requirements.txt")
    parts.append("cd src && python data.py && python model.py")
    parts.append("streamlit run app.py")
    parts.append("uvicorn api:app --reload")
    parts.append("cd src && python -m evaluation.run_experiments --max-samples 100")
    parts.append("cd src && python -m evaluation.run_experiments --max-samples 20 --with-llm")
    parts.append("pytest -q")
    parts.append("```")
    parts.append("")
    parts.append("## 6. Research claim (one liner)")
    parts.append("")
    parts.append(
        "SHAP-grounded explanations for credit recourse: constraining NL "
        "generators to SHAP/DiCE features raises mention precision and lowers "
        "hallucination rate while preserving risk coverage."
    )
    parts.append("")
    parts.append("---")
    parts.append("END OF DUMP")
    parts.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(files)} text files)")


if __name__ == "__main__":
    main()
```

### `src/baselines.py`

```python
"""
baselines.py
------------
Baseline models + probability calibration for fair comparison.

Supports the paper / portfolio claim that XGBoost is competitive
on this tabular credit dataset, and that calibrated probabilities
are used for decision confidence display.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from logging_utils import get_logger

logger = get_logger("baselines")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = ROOT / "models" / "baseline_comparison.json"
DEFAULT_CALIBRATED_PATH = ROOT / "models" / "loan_model_calibrated.pkl"


def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1_default": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision_default": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_default": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_good": float(f1_score(y_true, y_pred, pos_label=0)),
    }


def build_baselines(random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=3,
            random_state=random_state,
            eval_metric="auc",
        ),
    }


def run_baseline_comparison(X_train, X_test, y_train, y_test,
                            save_path: Path | str = DEFAULT_BASELINE_PATH) -> dict:
    """Train LR / RF / XGBoost and write metrics JSON."""
    import json

    results = {}
    models = build_baselines()
    for name, model in models.items():
        logger.info("Training baseline: %s", name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = pred.astype(float)
        results[name] = _metrics(y_test, pred, prob)
        logger.info("%s AUC=%.4f", name, results[name]["roc_auc"])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Baseline comparison saved to %s", save_path)
    return results


def calibrate_model(model, X_train, y_train, method: str = "isotonic"):
    """
    Wrap a fitted classifier with probability calibration.

    Uses a hold-out style CV calibration (sklearn CalibratedClassifierCV).
    Note: for production, prefer calibrating on a dedicated validation set.
    """
    calibrator = CalibratedClassifierCV(model, method=method, cv=3)
    calibrator.fit(X_train, y_train)
    return calibrator


def save_calibrated(model, path: Path | str = DEFAULT_CALIBRATED_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Calibrated model saved to %s", path)


def leak_feature_ablation(X_train, X_test, y_train, y_test,
                          leak_candidates: list[str] | None = None) -> dict:
    """
    Train XGBoost with/without suspected underwriting-leak features.

    loan_grade and loan_int_rate often encode prior underwriting decisions.
    Reporting this ablation is important for paper honesty / Model Cards.
    """
    leak_candidates = leak_candidates or ["loan_grade", "loan_int_rate"]
    present = [c for c in leak_candidates if c in X_train.columns]

    def _fit(cols_drop: list[str]) -> dict:
        keep = [c for c in X_train.columns if c not in cols_drop]
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=3, random_state=42, eval_metric="auc",
        )
        m.fit(X_train[keep], y_train)
        prob = m.predict_proba(X_test[keep])[:, 1]
        pred = m.predict(X_test[keep])
        return _metrics(y_test, pred, prob)

    out = {
        "full_model": _fit([]),
        "without_leak_candidates": _fit(present),
        "dropped_features": present,
    }
    logger.info(
        "Leakage ablation: full AUC=%.4f vs ablated AUC=%.4f (dropped %s)",
        out["full_model"]["roc_auc"],
        out["without_leak_candidates"]["roc_auc"],
        present,
    )
    return out


if __name__ == "__main__":
    from data import DEFAULT_DATA_PATH, run_pipeline
    from model import run_training

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    comparison = run_baseline_comparison(X_train, X_test, y_train, y_test)
    ablation = leak_feature_ablation(X_train, X_test, y_train, y_test)
    model, _ = run_training(X_train, X_test, y_train, y_test)
    calibrated = calibrate_model(model, X_train, y_train)
    save_calibrated(calibrated)
    print(comparison)
    print(ablation)
```

### `src/config_loader.py`

```python
"""Load project config from config/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
```

### `src/counterfactuals.py`

```python
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
    Wraps XGBoost model to force float64 dtype on all inputs.

    Why this is needed:
    DiCE internally modifies DataFrames during counterfactual
    generation, sometimes producing object dtype columns.
    XGBoost (newer versions) strictly requires numeric dtypes
    and raises ValueError on object columns.

    This wrapper intercepts every prediction call and casts
    the input to float64 before passing to XGBoost.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(pd.DataFrame(X).astype(float))

    def predict_proba(self, X):
        return self.model.predict_proba(pd.DataFrame(X).astype(float))


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
        # Desired class 0 = good standing / approval for rejected applicants
        cf = dice_exp.generate_counterfactuals(
            query_instances=applicant_data,
            total_CFs=n,
            desired_class=0,
            permitted_range=feature_ranges,
            features_to_vary=list(feature_ranges.keys()),
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
```

### `src/data.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project root (parent of src/), so paths work regardless of CWD
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "loan_dataset.csv"

# DATA LOADING

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# DATA CLEANING

def clean_data(df):

    # REMOVING IMPOSSIBLE AGE VALUES
    before = len(df)
    df = df[df["person_age"] <= 100]
    print(f"Removed {before - len(df)} rows with impossible age values")

    # REMOVING IMPOSSIBLE EMPLOYMENT LENGTH
    df = df[df["person_emp_length"] <= 60]
    print(f"Dataset after outlier removal: {df.shape[0]} rows")

    # HANDLING MISSING VALUES
    # emp_length - numeric, to be filled with median because of outliers
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    # loan_int_rate — numeric, fill with median
    # Interest rate correlates with loan grade so median is safe
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # REMOVING DUPLICATES
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")

    # Verify — no missing values should remain
    remaining = df.isna().sum().sum()
    if remaining > 0:
        print(f"WARNING: {remaining} missing values still remain")
    else:
        print("Clean: no missing values")

    return df


# DATA ENCODING

def encode_features(df):

    # loan_grade — Label Encoding encoding because A > B > C > D > E > F > G
    grade_map = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}
    df["loan_grade"] = df["loan_grade"].map(grade_map)

    # cb_person_default_on_file — binary label encoding
    # Y (has defaulted before) = 1, N (clean record) = 0
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(
        {"Y": 1, "N": 0}
    )

    # person_home_ownership and loan_intent — One-Hot Encoded
    # No natural order between RENT/OWN/MORTGAGE or PERSONAL/MEDICAL etc.
    # drop_first=True avoids the dummy variable trap
    df = pd.get_dummies(
        df, columns = ["person_home_ownership", "loan_intent"], 
        drop_first=True
    )

    # Convert bool columns to int — XGBoost requires numeric types
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"Encoded: {df.shape[1]} columns, all numeric")
    print(f"Columns: {df.columns.tolist()}")

    return df
    
# SPLIT FEATURES AND TARGET

def split_features(df):
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    print(f"\nFeatures (X): {X.shape}")
    print(f"Target   (y): {y.shape}")
    print(f"\nClass distribution:")
    print(y.value_counts())
    print(f"\nImbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.1f}:1")

    return X, y

# TRAIN/TEST SPLIT

def get_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )

    print(f"\nTraining set: {X_train.shape[0]} rows")
    print(f"Test set:     {X_test.shape[0]} rows")
    print(f"\nClass ratio in test set:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test

# FULL PIPELINE
def run_pipeline(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = get_train_test(X, y)

    return X_train, X_test, y_train, y_test, X, y

# QUICK TEST
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    print("\nPipeline complete. Ready for model training.")
    print(f"Final feature set: {X.columns.tolist()}")
```

### `src/evaluation/__init__.py`

```python
"""Evaluation package for faithfulness and recourse experiments."""

from evaluation.faithfulness import grounded_feature_set, score_faithfulness
from evaluation.recourse_metrics import score_counterfactual_set

__all__ = [
    "score_faithfulness",
    "grounded_feature_set",
    "score_counterfactual_set",
]
```

### `src/evaluation/faithfulness.py`

```python
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
```

### `src/evaluation/recourse_metrics.py`

```python
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
```

### `src/evaluation/run_experiments.py`

```python
"""
run_experiments.py
------------------
Batch evaluation for the paper:

1) Recourse quality (validity, sparsity, proximity, actionability delay)
2) Faithfulness of free vs grounded LLM explanations (optional Gemini)
3) Summary tables written to results/

Usage (from project root, with PYTHONPATH=src or from src/):
    python -m evaluation.run_experiments
    python -m evaluation.run_experiments --with-llm --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python -m evaluation.run_experiments` from src/
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import load_config, resolve_path
from counterfactuals import (
    build_dice_explainer,
    format_counterfactuals,
    generate_counterfactuals,
    get_feature_ranges,
)
from data import run_pipeline
from evaluation.faithfulness import score_faithfulness
from evaluation.recourse_metrics import score_counterfactual_set
from evaluation.template_explainer import template_explanation
from explainer import (
    build_explanation_prompt,
    build_explainer,
    build_grounded_explanation_prompt,
    get_shap_values,
)
from logging_utils import get_logger
from model import load_model

logger = get_logger("experiments")


def _cf_feature_list(suggestion_text: str, feature_ranges: dict) -> list[str]:
    if not suggestion_text:
        return []
    mentioned = []
    labels = {
        "person_income": ["income"],
        "loan_amnt": ["loan request", "loan amount"],
        "person_emp_length": ["employment"],
        "loan_percent_income": ["loan-to-income", "loan as %"],
        "loan_int_rate": ["interest rate"],
    }
    low = suggestion_text.lower()
    for feat in feature_ranges:
        for phrase in labels.get(feat, [feat]):
            if phrase in low:
                mentioned.append(feat)
                break
    return mentioned


def run_experiments(max_samples: int = 50, with_llm: bool = False,
                    random_state: int = 42) -> dict:
    cfg = load_config()
    data_path = resolve_path(cfg["paths"]["data"])
    model_path = resolve_path(cfg["paths"]["model"])
    out_path = resolve_path(cfg["paths"]["experiment_results"])

    X_train, X_test, y_train, y_test, X, y = run_pipeline(data_path)
    model = load_model(model_path)
    explainer = build_explainer(model, X_train)
    dice_exp, _ = build_dice_explainer(model, X_train, y_train)

    preds = model.predict(X_test)
    rejected_idx = np.where(preds == 1)[0]
    rng = np.random.default_rng(random_state)
    if len(rejected_idx) > max_samples:
        rejected_idx = rng.choice(rejected_idx, size=max_samples, replace=False)

    recourse_rows = []
    faith_free, faith_grounded = [], []
    faith_tmpl_free, faith_tmpl_grounded = [], []

    llm_fn = None
    if with_llm:
        try:
            from llm import generate_explanation
            llm_fn = generate_explanation
        except Exception as e:
            logger.warning("LLM unavailable (%s); skipping faithfulness LLM calls", e)

    for i, idx in enumerate(rejected_idx):
        row_df = X_test.iloc[[idx]]
        shap_vals = get_shap_values(explainer, row_df)[0]
        shap_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Value": row_df.iloc[0].values,
            "SHAP": shap_vals,
        })

        ranges = get_feature_ranges(X_train, row_df.iloc[0])
        cf_result = generate_counterfactuals(dice_exp, row_df, ranges)
        changes, suggestion_text = format_counterfactuals(cf_result, row_df.iloc[0])

        cf_df = None
        if cf_result is not None and cf_result.cf_examples_list:
            cf_df = cf_result.cf_examples_list[0].final_cfs_df

        rec = score_counterfactual_set(
            model, row_df, cf_df, list(ranges.keys())
        )
        rec["idx"] = int(idx)
        recourse_rows.append(rec)

        cf_feats = _cf_feature_list(suggestion_text or "", ranges)

        # Offline template ablation (always runs; reproducible paper baseline)
        tmpl_free = template_explanation(
            shap_df, suggestion_text, grounded=False
        )
        tmpl_grounded = template_explanation(
            shap_df, suggestion_text, grounded=True
        )
        faith_tmpl_free.append(
            score_faithfulness(tmpl_free, shap_df, cf_features=cf_feats)
        )
        faith_tmpl_grounded.append(
            score_faithfulness(tmpl_grounded, shap_df, cf_features=cf_feats)
        )

        if llm_fn is not None:
            free_prompt = build_explanation_prompt(row_df.iloc[0], shap_df)
            if suggestion_text:
                free_prompt += f"\n\nACTIONABLE CHANGES:\n{suggestion_text}"
            grounded_prompt = build_grounded_explanation_prompt(
                row_df.iloc[0], shap_df, suggestion_text
            )
            try:
                free_text = llm_fn(free_prompt)
                grounded_text = llm_fn(grounded_prompt)
            except Exception as e:
                logger.warning("LLM failed on idx=%s: %s", idx, e)
                continue

            faith_free.append(
                score_faithfulness(free_text, shap_df, cf_features=cf_feats)
            )
            faith_grounded.append(
                score_faithfulness(grounded_text, shap_df, cf_features=cf_feats)
            )

        if (i + 1) % 10 == 0:
            logger.info("Processed %d / %d rejected applicants", i + 1, len(rejected_idx))

    def _mean_keys(rows: list[dict], keys: list[str]) -> dict:
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
        out["n"] = len(rows)
        return out

    faith_keys = ["coverage", "precision", "hallucination_rate"]
    summary = {
        "n_rejected_evaluated": len(rejected_idx),
        "recourse": _mean_keys(
            recourse_rows,
            ["validity", "mean_sparsity", "mean_proximity", "mean_actionability_delay"],
        ),
        "template_free": _mean_keys(faith_tmpl_free, faith_keys),
        "template_grounded": _mean_keys(faith_tmpl_grounded, faith_keys),
        "faithfulness_free": _mean_keys(faith_free, faith_keys) if faith_free else None,
        "faithfulness_grounded": (
            _mean_keys(faith_grounded, faith_keys) if faith_grounded else None
        ),
        "with_llm": bool(faith_free),
        "paper_claim": (
            "Grounded prompting / templates raise feature-mention precision "
            "and lower hallucination_rate vs unconstrained text, "
            "while preserving SHAP risk coverage; recourse is reported via "
            "validity, sparsity, proximity, and actionability delay."
        ),
    }

    summary["template_deltas"] = {
        "precision_delta": (
            summary["template_grounded"]["precision"]
            - summary["template_free"]["precision"]
        ),
        "hallucination_delta": (
            summary["template_grounded"]["hallucination_rate"]
            - summary["template_free"]["hallucination_rate"]
        ),
        "coverage_delta": (
            summary["template_grounded"]["coverage"]
            - summary["template_free"]["coverage"]
        ),
    }

    if summary["faithfulness_free"] and summary["faithfulness_grounded"]:
        summary["llm_deltas"] = {
            "precision_delta": (
                summary["faithfulness_grounded"]["precision"]
                - summary["faithfulness_free"]["precision"]
            ),
            "hallucination_delta": (
                summary["faithfulness_grounded"]["hallucination_rate"]
                - summary["faithfulness_free"]["hallucination_rate"]
            ),
            "coverage_delta": (
                summary["faithfulness_grounded"]["coverage"]
                - summary["faithfulness_free"]["coverage"]
            ),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "recourse_per_applicant": recourse_rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote experiment results to %s", out_path)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run paper evaluation experiments")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=cfg.get("evaluation", {}).get("max_samples", 50),
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Call Gemini for free vs grounded faithfulness comparison",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=cfg.get("evaluation", {}).get("random_state", 42),
    )
    args = parser.parse_args()
    run_experiments(
        max_samples=args.max_samples,
        with_llm=args.with_llm,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
```

### `src/evaluation/template_explainer.py`

```python
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
```

### `src/explainer.py`

```python
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

def build_explainer(model, X_train, background_size=100):
    """
    Create a SHAP TreeExplainer for the trained XGBoost model.

    Why TreeExplainer specifically?
    - Optimized for tree-based models (XGBoost, RandomForest)
    - Exact SHAP values, not approximations
    - 100x faster than model-agnostic explainers (KernelExplainer)
    - Uses a sample of training data as the baseline background

    Args:
        model:               Trained XGBoost model
        X_train (DataFrame): Training features for baseline calculation
        background_size:     Rows sampled for background (keeps SHAP fast)

    Returns:
        shap.TreeExplainer
    """
    n = min(background_size, len(X_train))
    background = shap.sample(X_train, n, random_state=42)
    explainer = shap.TreeExplainer(model, background)

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.asarray(expected).ravel()[-1])
    else:
        expected = float(expected)

    print("Explainer created")
    print(f"Baseline (expected value): {expected:.4f}")
    print("Interpretation: without seeing any features, the model's")
    print(f"default prediction score is {expected:.4f}")
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
    shap_values = _to_array(explainer.shap_values(X))
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
```

### `src/fairness.py`

```python
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
            "Recall":        f"{recall:.1%}" if recall else "N/A"
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
```

### `src/llm.py`

```python
"""
llm.py
------
Natural-language explanation generation via Google Gemini.

Reads the API key from (in order):
    1. Explicit api_key argument
    2. GEMINI_API_KEY environment variable
    3. GOOGLE_API_KEY environment variable

Functions:
    generate_explanation(prompt, ...) - Call Gemini with the SHAP prompt
"""

from __future__ import annotations

import os
from pathlib import Path

from google import genai

from logging_utils import get_logger

logger = get_logger("llm")

try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

DEFAULT_MODEL = "gemini-2.5-flash"


def _resolve_api_key(api_key=None):
    key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not key:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "in your environment (or a .env file), or pass api_key=..."
        )
    return key


def generate_explanation(prompt, model=DEFAULT_MODEL, api_key=None):
    """
    Generate an applicant-facing rejection explanation with Gemini.

    Args:
        prompt (str):  Output of build_explanation_prompt(...) or grounded variant
        model (str):   Gemini model id
        api_key (str): Optional override; otherwise uses env vars

    Returns:
        str: Natural-language explanation text
    """
    client = genai.Client(api_key=_resolve_api_key(api_key))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    logger.info("Generated explanation (%d chars) with %s", len(text), model)
    return text
```

### `src/logging_utils.py`

```python
"""Shared logging setup for the loan rejection assistant."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "loan_rejection", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
```

### `src/model.py`

```python
"""
model.py
--------
Handles model training, evaluation, and persistence for the
Explainable Loan Rejection Assistant.

Dataset context:
    loan_status: 0 = good standing, 1 = default (high risk)
    Imbalance:   3.6:1 (non-default vs default)
    Train rows:  25,217
    Test rows:    6,305

Functions:
    train_model(X_train, y_train)         - Train XGBoost classifier
    evaluate_model(model, X_test, y_test) - Full evaluation report
    save_model(model, path)               - Persist model to disk
    load_model(path)                      - Load model from disk
    run_training(X_train, X_test,         - Train + evaluate + save
                 y_train, y_test)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "loan_model.pkl"
DEFAULT_CM_PATH = ROOT / "models" / "confusion_matrix.png"


# TRAIN MODEL

def train_model(X_train, y_train):
    """
    Train an XGBoost binary classifier on loan data.

    Why XGBoost?
    - Handles tabular data better than neural networks at this scale
    - Built-in feature importance (used by SHAP TreeExplainer)
    - Fast training, no feature scaling required
    - Industry standard for credit risk modeling

    Why scale_pos_weight=3?
    - Dataset has 3.6:1 imbalance (non-default vs default)
    - This tells XGBoost to weight the minority class (default)
      3x more during training, so it doesn't just learn to always
      predict the majority class
    - Formula: count(negative) / count(positive) ≈ 24715/6807 ≈ 3.6
      We use 3 as a slightly conservative estimate

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series):    Training labels

    Returns:
        xgb.XGBClassifier: Trained model
    """
    model = xgb.XGBClassifier(
        n_estimators=200,          # number of trees — more than default for better accuracy
        max_depth=5,               # tree depth — controls overfitting
        learning_rate=0.1,         # how much each tree contributes
        scale_pos_weight=3,        # handles 3.6:1 class imbalance
        random_state=42,           # reproducibility
        eval_metric="auc",         # optimize for AUC during training
    )

    model.fit(X_train, y_train)

    print("Model trained successfully")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Features used:   {model.n_features_in_}")

    return model


# EVALUATE MODEL

def evaluate_model(model, X_test, y_test, cm_path=DEFAULT_CM_PATH):
    """
    Run full evaluation on the test set.

    Metrics reported:
    - Classification report (precision, recall, F1 per class)
    - ROC-AUC score (overall discrimination ability)
    - Confusion matrix (visual breakdown of errors)

    Why ROC-AUC for this dataset?
    With 3.6:1 imbalance, accuracy is misleading — a model that
    always predicts 0 would score 78% accuracy. ROC-AUC measures
    how well the model separates the two classes regardless of
    the decision threshold, making it imbalance-robust.

    Args:
        model: Trained XGBoost model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series):    Test labels
        cm_path: Path to save confusion matrix image

    Returns:
        dict: {predictions, probabilities, auc_score}
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    print("=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"\nROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Good Standing (0)", "Default (1)"]
    ))

    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Good Standing", "Default"]
    )
    disp.plot(cmap="Blues")
    plt.title("XGBoost Confusion Matrix — Credit Risk")
    plt.tight_layout()

    cm_path = Path(cm_path)
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "auc_score": auc,
    }


# SAVE AND LOAD MODEL

def save_model(model, path=DEFAULT_MODEL_PATH):
    """
    Persist trained model to disk using joblib.

    Why joblib over pickle?
    joblib is optimized for large numpy arrays — much faster
    for ML models than Python's built-in pickle.

    Args:
        model: Trained XGBoost model
        path (str): Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=DEFAULT_MODEL_PATH):
    """
    Load a previously trained model from disk.

    Use this instead of retraining — training takes time,
    loading takes milliseconds.

    Args:
        path (str): Path to saved model file

    Returns:
        Trained XGBoost model
    """
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


# FULL TRAINING PIPELINE

def run_training(X_train, X_test, y_train, y_test):
    """
    Run the complete training pipeline:
    train → evaluate → save

    Args:
        X_train, X_test: Feature splits
        y_train, y_test: Label splits

    Returns:
        tuple: (model, evaluation_results)
    """
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    save_model(model)

    return model, results


# QUICK TEST

if __name__ == "__main__":
    from data import DEFAULT_DATA_PATH, run_pipeline

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    model, results = run_training(X_train, X_test, y_train, y_test)

    print(f"\nFinal AUC: {results['auc_score']:.4f}")
    print("Ready for SHAP explainability.")
```

### `src/pipeline.py`

```python
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

    return pd.DataFrame([row])[feature_cols]


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
                prompt, model=cfg.get("llm", {}).get("model", "gemini-2.5-flash")
            )
            result["faithfulness"] = score_faithfulness(
                result["explanation"], shap_df, cf_features=cf_features
            )
        except Exception as e:
            logger.warning("Gemini explanation skipped: %s", e)
            result["explanation"] = None

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
```

### `src/requirements.txt`

```text
pandas>=2.0,<3
numpy>=1.24,<3
scikit-learn>=1.3,<2
xgboost>=2.0,<4
shap>=0.44,<1
dice-ml>=0.11,<1
google-genai>=1.0,<2
python-dotenv>=1.0,<2
streamlit>=1.28,<2
joblib>=1.3,<2
matplotlib>=3.7,<4
PyYAML>=6.0,<7
fastapi>=0.110,<1
uvicorn[standard]>=0.27,<1
pydantic>=2.0,<3
pytest>=8.0,<9
```

### `tests/test_core.py`

```python
"""Unit tests for preprocessing, fairness, and faithfulness metrics."""

from __future__ import annotations

import sys
from pathlib import Path

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
```

## 3. Binary / non-text assets (inventory only)

| Path | Size (bytes) | Notes |
|------|-------------:|-------|
| `models/confusion_matrix.png` | 33771 | binary artifact |
| `models/global_importance.png` | 91695 | binary artifact |
| `models/loan_model.pkl` | 445523 | binary artifact |
| `models/loan_model_calibrated.pkl` | 1754332 | binary artifact |
| `models/waterfall.png` | 91213 | binary artifact |

## 4. Captured terminal / experiment run (`--with-llm`)

Command:

```bash
cd src
python -m evaluation.run_experiments --max-samples 20 --with-llm
```

Exit code: `0`

### 4.1 Startup log

```text
Loaded dataset: 32581 rows and 12 columns
Removed 5 rows with impossible age values
Dataset after outlier removal: 31679 rows
Removed 157 duplicate rows
Clean: no missing values
Encoded: 18 columns, all numeric
Columns: [person_age, person_income, person_emp_length, loan_grade, loan_amnt,
loan_int_rate, loan_status, loan_percent_income, cb_person_default_on_file,
cb_person_cred_hist_length, person_home_ownership_OTHER,
person_home_ownership_OWN, person_home_ownership_RENT,
loan_intent_EDUCATION, loan_intent_HOMEIMPROVEMENT, loan_intent_MEDICAL,
loan_intent_PERSONAL, loan_intent_VENTURE]

Features (X): (31522, 17)
Target   (y): (31522,)
Class distribution: 0=24715, 1=6807
Imbalance ratio: 3.6:1
Training set: 25217 rows
Test set:     6305 rows

Model loaded from models/loan_model.pkl
Explainer created
Baseline (expected value): -1.9500
DiCE explainer created successfully
```

### 4.2 Per-applicant pattern

For each of 20 rejected applicants:
1. Computing SHAP values for 1 applicants... shape `(1, 17)`
2. DiCE counterfactual suggestions OR `No Counterfactuals found`
3. LLM call failed with either 404 or 429

**404 NOT_FOUND** (early samples):

```text
LLM failed on idx=...: 404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use a newer model for the latest features and improvements.', 'status': 'NOT_FOUND'}}
```

404 idxs: 4850, 4530, 1297, 6105, 5295, 4817, 520

**429 RESOURCE_EXHAUSTED** (later samples):

```text
LLM failed on idx=...: 429 RESOURCE_EXHAUSTED. You exceeded your current quota... Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-2.5-flash
```

429 idxs (examples): 766, 4117, 3157, 2680, 514, 2809, 5344, 559, 4640, 4961, 2704, 4363, 3269

DiCE failed (no CFs) on idxs: 6105, 5295, 4117, 559, 2704, 4363

### 4.3 Example counterfactual blocks printed

```text
Applicant idx 4850:
Option 1:
  - Increase annual income from $32,000 to $92,613
  - Reduce loan-to-income ratio from 36.0% to 30.0%
Option 2:
  - Increase annual income from $32,000 to $94,308
  - Reduce loan-to-income ratio from 36.0% to 10.0%
Option 3:
  - Increase annual income from $32,000 to $55,750
  - Reduce loan-to-income ratio from 36.0% to 10.0%

Applicant idx 4530:
Option 1:
  - Reduce loan-to-income ratio from 31.0% to 20.0%
  - Lower interest rate from 10.62% to 7.87%
Option 2:
  - Reduce loan-to-income ratio from 31.0% to 30.0%
  - Lower interest rate from 10.62% to 7.10%
Option 3:
  - Increase annual income from $22,680 to $72,470
  - Reduce loan-to-income ratio from 31.0% to 10.0%

Applicant idx 1297:
Option 1:
  - Reduce loan request from $12,800 to $6,393
  - Reduce loan-to-income ratio from 43.0% to 30.0%
Option 2:
  - Increase annual income from $30,000 to $41,808
  - Reduce loan-to-income ratio from 43.0% to 30.0%
Option 3:
  - Reduce loan-to-income ratio from 43.0% to 10.0%
  - Lower interest rate from 14.79% to 6.29%

Applicant idx 4817:
Option 1:
  - Increase annual income from $15,000 to $61,578
  - Reduce loan-to-income ratio from 40.0% to 20.0%

Applicant idx 766:
Option 1:
  - Increase annual income from $18,700 to $135,180
Option 2:
  - Increase annual income from $18,700 to $67,846
  - Build employment history from 3 to 9 years
Option 3:
  - Reduce loan request from $3,600 to $861
  - Reduce loan-to-income ratio from 19.0% to 10.0%

Applicant idx 520:
Option 1:
  - Increase annual income from $15,600 to $96,618

Applicant idx 3157:
Option 1:
  - Reduce loan request from $18,000 to $9,606
  - Reduce loan-to-income ratio from 37.0% to 30.0%

Applicant idx 2680:
Option 1:
  - Reduce loan-to-income ratio from 39.0% to 30.0%
  - Lower interest rate from 11.36% to 9.48%

Applicant idx 514:
Option 1:
  - Reduce loan-to-income ratio from 39.0% to 20.0%
  - Lower interest rate from 12.73% to 8.77%

Applicant idx 2809:
Option 1:
  - Increase annual income from $64,000 to $86,635
  - Build employment history from 0 to 4 years

Applicant idx 5344:
Option 1:
  - Reduce loan request from $15,000 to $5,590
  - Reduce loan-to-income ratio from 50.0% to 20.0%

Applicant idx 4640:
Option 1:
  - Increase annual income from $34,000 to $125,917

Applicant idx 4961:
Option 1:
  - Reduce loan request from $20,000 to $12,449
  - Build employment history from 0 to 4 years

Applicant idx 3269:
Option 1:
  - Increase annual income from $43,600 to $132,953
  - Build employment history from 4 to 8 years
  - Reduce loan-to-income ratio from 23.0% to 10.0%
  - Lower interest rate from 14.96% to 6.79%
```

### 4.4 Final printed summary

```json
{
  "n_rejected_evaluated": 20,
  "recourse": {
    "validity": 0.7,
    "mean_sparsity": 1.9761904761904765,
    "mean_proximity": 0.1609916498902698,
    "mean_actionability_delay": 0.41527777777777775,
    "n": 20
  },
  "template_free": {
    "coverage": 0.6900000000000001,
    "precision": 0.8098015873015875,
    "hallucination_rate": 0.1901984126984127,
    "n": 20
  },
  "template_grounded": {
    "coverage": 0.6799999999999999,
    "precision": 0.975,
    "hallucination_rate": 0.025,
    "n": 20
  },
  "faithfulness_free": null,
  "faithfulness_grounded": null,
  "with_llm": false,
  "paper_claim": "Grounded prompting / templates raise feature-mention precision and lower hallucination_rate vs unconstrained text, while preserving SHAP risk coverage; recourse is reported via validity, sparsity, proximity, and actionability delay.",
  "template_deltas": {
    "precision_delta": 0.1651984126984125,
    "hallucination_delta": -0.1651984126984127,
    "coverage_delta": -0.01000000000000012
  }
}
```

### 4.5 Pytest

```text
python -m pytest tests -q
....                                                                     [100%]
4 passed, 3 warnings in ~20s
```

## 5. How to run

```bash
pip install -r requirements.txt
cd src && python data.py && python model.py
streamlit run app.py
uvicorn api:app --reload
cd src && python -m evaluation.run_experiments --max-samples 100
cd src && python -m evaluation.run_experiments --max-samples 20 --with-llm
pytest -q
```

## 6. Research claim (one liner)

SHAP-grounded explanations for credit recourse: constraining NL generators to SHAP/DiCE features raises mention precision and lowers hallucination rate while preserving risk coverage.

---
END OF DUMP
