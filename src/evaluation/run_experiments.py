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
