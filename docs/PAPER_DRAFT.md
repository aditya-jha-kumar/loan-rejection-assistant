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
