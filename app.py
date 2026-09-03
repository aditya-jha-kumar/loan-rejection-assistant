"""
Streamlit UI for the Explainable Loan Rejection Assistant.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import load_artifacts, run_application  # noqa: E402


st.set_page_config(
    page_title="Loan Rejection Assistant",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model and explainers...")
def get_artifacts():
    return load_artifacts()


def main():
    st.title("Explainable Loan Rejection Assistant")
    st.caption(
        "Submit an application to see the model decision, SHAP risk drivers, "
        "and actionable changes if the loan is rejected."
    )

    artifacts = get_artifacts()

    with st.form("application"):
        col1, col2, col3 = st.columns(3)

        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=26)
            person_income = st.number_input(
                "Annual income ($)", min_value=0, max_value=10_000_000, value=31200, step=1000
            )
            person_emp_length = st.number_input(
                "Employment length (years)", min_value=0.0, max_value=60.0, value=8.0, step=0.5
            )
            cb_person_cred_hist_length = st.number_input(
                "Credit history length (years)", min_value=0, max_value=50, value=2
            )

        with col2:
            loan_amnt = st.number_input(
                "Loan amount ($)", min_value=500, max_value=100_000, value=5000, step=500
            )
            loan_int_rate = st.number_input(
                "Interest rate (%)", min_value=0.0, max_value=40.0, value=8.63, step=0.01
            )
            loan_grade = st.selectbox("Loan grade", list("ABCDEFG"), index=4)
            previous_default = st.selectbox("Previous default on file", ["No", "Yes"])

        with col3:
            home_ownership = st.selectbox(
                "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=0
            )
            loan_intent = st.selectbox(
                "Loan purpose",
                [
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "PERSONAL",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION",
                ],
                index=0,
            )
            loan_percent_income = round(loan_amnt / person_income, 4) if person_income else 0.0
            st.metric("Loan as % of income", f"{loan_percent_income:.1%}")

        submitted = st.form_submit_button("Evaluate application", type="primary")

    if not submitted:
        return

    input_dict = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": 1 if previous_default == "Yes" else 0,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": home_ownership,
        "loan_intent": loan_intent,
    }

    with st.spinner("Scoring application and generating explanation..."):
        result = run_application(input_dict, artifacts)

    decision = result["decision"]
    approved = decision == "APPROVED"

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Decision", decision)
    m2.metric("Approval probability", f"{result['approval_prob']:.1%}")
    m3.metric("Rejection probability", f"{result['rejection_prob']:.1%}")

    if approved:
        st.success(result.get("message", "Application approved."))
        return

    st.error("Application rejected. See the explanation below.")

    shap_df = result.get("shap_explanation")
    if isinstance(shap_df, pd.DataFrame) and not shap_df.empty:
        st.subheader("Risk drivers")
        display_df = shap_df.copy()
        display_df["SHAP"] = display_df["SHAP"].map(lambda v: round(float(v), 4))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        top_risks = shap_df[shap_df["SHAP"] > 0].head(3)
        if not top_risks.empty:
            st.bar_chart(top_risks.set_index("Feature")["SHAP"])

    if result.get("counterfactuals"):
        st.subheader("What would need to change")
        st.markdown(result["counterfactuals"])

    if result.get("explanation"):
        st.subheader("Applicant-facing explanation")
        st.write(result["explanation"])
    else:
        st.info(
            "Gemini explanation is unavailable. Add GEMINI_API_KEY to a `.env` "
            "file in the project root to generate a natural-language letter."
        )

    if result.get("fairness"):
        st.caption(result["fairness"].get("note", ""))


if __name__ == "__main__":
    main()
