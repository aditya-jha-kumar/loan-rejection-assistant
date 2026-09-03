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
