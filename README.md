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

## Vercel (FastAPI)

Vercel looks for FastAPI in `app.py` by default; this repo’s UI is Streamlit. The API entrypoint is set in `pyproject.toml`:

```toml
[tool.vercel]
entrypoint = "api:app"
```

Add `GEMINI_API_KEY` in the Vercel project environment if you want Gemini letters. The API lazy-loads the model on `/predict`; `/` and `/health` stay up even before that.

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
