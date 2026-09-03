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
