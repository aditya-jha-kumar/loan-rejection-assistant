"""
app.py
------
Streamlit UI for the Explainable Loan Rejection Assistant.

streamlit run app.py

The FastAPI app lives in api.py. Vercel uses api:app via pyproject.toml.
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
    person_age = st.number_input("Age", min_value=18, max_value=100, value=23)
    person_income = st.number_input(
        "Annual income ($)",
        min_value=0,
        max_value=500_000,
        value=30_000,
        step=1_000,
    )
    person_emp_length = st.number_input(
        "Employment length (years)",
        min_value=0,
        max_value=50,
        value=0,
    )
    loan_grade = st.selectbox(
        "Credit grade",
        options=["A", "B", "C", "D", "E", "F", "G"],
        index=2,
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
        value=11.6,
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
        index=0,
    )
    person_home_ownership = st.selectbox(
        "Home ownership",
        options=["RENT", "OWN", "MORTGAGE", "OTHER"],
        index=0,
    )
    cb_person_default_on_file = st.selectbox(
        "Previous default on record?",
        options=["No", "Yes"],
        index=1,
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit history length (years)",
        min_value=0,
        max_value=30,
        value=4,
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
            except Exception as e:
                sys.stdout = old_stdout
                st.error(f"Could not analyze this application: {e}")
                st.stop()
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
            if result.get("explanation_source") == "template":
                st.caption(
                    "Gemini was temporarily busy (high demand). "
                    "This local explanation uses the same SHAP risk factors "
                    "and DiCE suggestions."
                )
        else:
            err = result.get("llm_error")
            if err:
                st.info(f"Gemini explanation unavailable: {err}")
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

