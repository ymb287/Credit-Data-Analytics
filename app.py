import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
import gdown

st.set_page_config(page_title="Credit Risk Contact Prioritizer", layout="wide")

# ============================================================
# Helpers
# ============================================================

def format_money(x):
    return f"€{x:,.2f}"


def load_model(model_file):
    return joblib.load(model_file)


def load_training_columns(training_cols_file):
    return joblib.load(training_cols_file)


def load_raw_credit_file(uploaded_file):
    """
    Reads the original UCI / course credit card file.
    Works for xls, xlsx, and csv.
    The original xls often has the real headers in the second row.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    try:
        df = pd.read_excel(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, header=1)

    expected_markers = {
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE",
        "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"
    }
    cols = set(str(c) for c in df.columns)

    if len(expected_markers.intersection(cols)) >= 3:
        return df

    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, header=1)


def standardize_credit_columns(df):
    """
    Standardize original file structure for display/output use.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    rename_map = {
        "default payment next month": "default",
        "default payment next month ": "default",
        "PAY_0": "PAY_1",
    }
    out = out.rename(columns=rename_map)

    unnamed_cols = [c for c in out.columns if c.lower().startswith("unnamed")]
    if unnamed_cols and "ID" not in out.columns:
        out = out.rename(columns={unnamed_cols[0]: "ID"})

    return out


def prepare_raw_credit_data(df):
    """
    Rebuild the SAME feature logic used in training.
    This returns the model-input dataframe.
    """
    out = standardize_credit_columns(df)

    # remove broken rows if needed
    if "ID" in out.columns:
        out = out[out["ID"].notna()].copy()

    # convert numeric where possible
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="ignore")

    # same mapping as notebook
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["EDUCATION"] = out["EDUCATION"].replace({
        1: "graduate_school",
        2: "university",
        3: "high_school",
        4: "others"
    })

    out.loc[out["MARRIAGE"] == 0, "MARRIAGE"] = 3
    out["MARRIAGE"] = out["MARRIAGE"].replace({
        1: "married",
        2: "single",
        3: "others"
    })

    out["SEX"] = out["SEX"].replace({
        1: "male",
        2: "female"
    })

    # feature engineering from notebook
    out["SEX_MARRIAGE"] = out["SEX"] + "_" + out["MARRIAGE"]

    for i in range(1, 7):
        out[f"Closeness_{i}"] = out[f"BILL_AMT{i}"] / out["LIMIT_BAL"]
        out[f"delay_flag_{i}"] = (out[f"PAY_{i}"] > 0).astype(int)

    # same feature selection as training
    features = [col for col in out.columns if col not in ["default", "ID"]]
    out = out[features].copy()

    # same dummy creation as training
    out = pd.get_dummies(out, drop_first=True)

    return out.reset_index(drop=True)


def score_file(model, prepared_df, expected_cols):
    """
    Align uploaded prepared data to exact training columns,
    then score probabilities.
    """
    model_input = prepared_df.reindex(columns=expected_cols, fill_value=0)
    p_default = model.predict_proba(model_input)[:, 1]
    return p_default, model_input


def calculate_contact_strategy(
    df,
    id_col,
    exposure_col,
    budget,
    contact_cost,
    incremental_recovery,
    fp_cost_rate,
    min_prob_threshold,
    max_contacts=None,
):
    """
    Expected value of contacting a customer:

    EV(contact) = p(default) * exposure * incremental_recovery
                  - (1 - p(default)) * exposure * fp_cost_rate
                  - contact_cost
    """

    out = df.copy()

    out["expected_benefit_if_default"] = (
        out["p_default"] * out[exposure_col] * incremental_recovery
    )
    out["expected_false_positive_cost"] = (
        (1 - out["p_default"]) * out[exposure_col] * fp_cost_rate
    )
    out["expected_profit_contact"] = (
        out["expected_benefit_if_default"]
        - out["expected_false_positive_cost"]
        - contact_cost
    )

    out["recommend_contact"] = (
        (out["expected_profit_contact"] > 0)
        & (out["p_default"] >= min_prob_threshold)
    )

    ranked = out.loc[out["recommend_contact"]].copy()
    ranked = ranked.sort_values(
        ["expected_profit_contact", "p_default"],
        ascending=[False, False]
    )

    affordable_n = int(budget // contact_cost) if contact_cost > 0 else len(ranked)

    if max_contacts is None:
        final_n = affordable_n
    else:
        final_n = min(affordable_n, int(max_contacts))

    contact_list = ranked.head(final_n).copy()
    non_contact_list = out.loc[~out.index.isin(contact_list.index)].copy()

    summary = {
        "uploaded_rows": int(len(out)),
        "eligible_positive_ev": int(len(ranked)),
        "recommended_contacts": int(len(contact_list)),
        "remaining_not_contacted": int(len(non_contact_list)),
        "budget": float(budget),
        "contact_cost": float(contact_cost),
        "budget_used": float(len(contact_list) * contact_cost),
        "budget_left": float(budget - len(contact_list) * contact_cost),
        "expected_total_profit": float(contact_list["expected_profit_contact"].sum()),
        "avg_expected_profit_per_contact": float(contact_list["expected_profit_contact"].mean()) if len(contact_list) > 0 else 0.0,
        "avg_pd_contacted": float(contact_list["p_default"].mean()) if len(contact_list) > 0 else 0.0,
        "avg_exposure_contacted": float(contact_list[exposure_col].mean()) if len(contact_list) > 0 else 0.0,
    }

    return contact_list, non_contact_list, out, summary


def make_download_excel(contact_list, scored_df, summary_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        contact_list.to_excel(writer, sheet_name="contact_list", index=False)
        scored_df.to_excel(writer, sheet_name="all_scored_customers", index=False)
        pd.DataFrame([summary_dict]).to_excel(writer, sheet_name="summary", index=False)
    output.seek(0)
    return output


# ============================================================
# App UI
# ============================================================

st.title("Credit Risk Contact Prioritizer")
st.caption(
    "Upload the original raw customer file, prepare it inside the app, "
    "score default risk, and generate a ranked contact list."
)

DEFAULT_MODEL_PATH = "random_forest.pkl"
DEFAULT_COLUMNS_PATH = "training_columns.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1dnac94GppuJGHIIgQoUp1J2pYB-g5Fkf"

@st.cache_resource
def load_default_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        gdown.download(MODEL_URL, DEFAULT_MODEL_PATH, quiet=False)
    return joblib.load(DEFAULT_MODEL_PATH)

with st.sidebar:
    st.header("Inputs")

    model_file = st.file_uploader(
        "Upload trained model (.joblib / .pkl)",
        type=["joblib", "pkl"]
    )

    training_cols_file = st.file_uploader(
        "Upload training columns (.joblib / .pkl)",
        type=["joblib", "pkl"]
    )

    data_file = st.file_uploader(
        "Upload original raw file (.xls, .xlsx or .csv)",
        type=["xls", "xlsx", "csv"]
    )


    st.subheader("Business assumptions")
    budget = st.number_input("Budget", min_value=0.0, value=25000.0, step=1000.0)
    contact_cost = st.number_input("Contact cost per customer", min_value=0.0, value=25.0, step=1.0)
    incremental_recovery = st.slider("Incremental recovery rate", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
    fp_cost_rate = st.slider("False-positive cost rate", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
    min_prob_threshold = st.slider("Minimum default probability threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

    st.subheader("Operational constraints")
    use_max_contacts = st.checkbox("Set a maximum number of contacts", value=False)
    max_contacts = st.number_input("Maximum contacts", min_value=1, value=500, step=10) if use_max_contacts else None

st.markdown("---")

if data_file is not None:
    raw_df = load_raw_credit_file(data_file)
    output_df = standardize_credit_columns(raw_df).copy().reset_index(drop=True)
    prepared_df = prepare_raw_credit_data(raw_df)

    st.subheader("Preview of raw uploaded file")
    st.dataframe(output_df.head(), use_container_width=True)

    st.subheader("Preview of model input after in-app preparation")
    st.dataframe(prepared_df.head(), use_container_width=True)

    prep_info = st.expander("Show preparation steps used in the app")
    with prep_info:
        st.write("- standardize raw column names")
        st.write("- rename `default payment next month` to `default` and `PAY_0` to `PAY_1`")
        st.write("- map `SEX`, `EDUCATION`, `MARRIAGE` exactly as in training")
        st.write("- create `SEX_MARRIAGE`")
        st.write("- create `Closeness_1` to `Closeness_6`")
        st.write("- create `delay_flag_1` to `delay_flag_6`")
        st.write("- remove `default` and `ID` from model input")
        st.write("- apply `pd.get_dummies(..., drop_first=True)`")
        st.write("- reindex to exact training columns before scoring")

    col1, col2 = st.columns(2)
    with col1:
        id_default = output_df.columns.tolist().index("ID") if "ID" in output_df.columns else 0
        id_col = st.selectbox("ID column", options=output_df.columns.tolist(), index=id_default)
    with col2:
        exposure_default = output_df.columns.tolist().index("BILL_AMT1") if "BILL_AMT1" in output_df.columns else 0
        exposure_col = st.selectbox(
            "Exposure column",
            options=output_df.columns.tolist(),
            index=exposure_default
        )
else:
    raw_df = None
    output_df = None
    prepared_df = None
    id_col = None
    exposure_col = None

run_btn = st.button("Calculate", type="primary", use_container_width=True)

if run_btn:
    if model_file is not None:
        model = joblib.load(model_file)
        st.info("Using uploaded custom model")
    else:
        model = load_default_model()
        st.info("Using default model")

    if training_cols_file is not None:
        expected_cols = joblib.load(training_cols_file)
        st.info("Using uploaded custom training columns")
    else:
        expected_cols = joblib.load(DEFAULT_COLUMNS_PATH)
        st.info("Using default training columns")

    if raw_df is None:
        st.error("Please upload a raw scoring dataset first.")
        st.stop()

    try:
        p_default, model_input = score_file(model, prepared_df, expected_cols)

        scored_df = output_df.copy().reset_index(drop=True)
        scored_df["p_default"] = p_default

        contact_list, non_contact_list, full_df, summary = calculate_contact_strategy(
            scored_df,
            id_col=id_col,
            exposure_col=exposure_col,
            budget=budget,
            contact_cost=contact_cost,
            incremental_recovery=incremental_recovery,
            fp_cost_rate=fp_cost_rate,
            min_prob_threshold=min_prob_threshold,
            max_contacts=max_contacts,
        )

        st.success("Calculation finished.")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Recommended contacts", summary["recommended_contacts"])
        k2.metric("Expected total profit", format_money(summary["expected_total_profit"]))
        k3.metric("Budget used", format_money(summary["budget_used"]))
        k4.metric("Budget left", format_money(summary["budget_left"]))

        k5, k6, k7 = st.columns(3)
        k5.metric("Eligible positive EV", summary["eligible_positive_ev"])
        k6.metric("Avg PD contacted", f"{summary['avg_pd_contacted']:.1%}")
        k7.metric("Avg exposure contacted", format_money(summary["avg_exposure_contacted"]))

        st.markdown("### Ranked contact list")
        show_cols = [
            id_col,
            exposure_col,
            "p_default",
            "expected_profit_contact",
            "expected_benefit_if_default",
            "expected_false_positive_cost",
        ]

        available_show_cols = [c for c in show_cols if c in contact_list.columns]

        st.dataframe(
            contact_list[available_show_cols].reset_index(drop=True),
            use_container_width=True,
            height=450,
        )

        st.markdown("### Portfolio statistics")
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.hist(full_df["p_default"], bins=30)
            ax1.set_title("Distribution of predicted default probability")
            ax1.set_xlabel("Predicted probability of default")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            top_plot = contact_list.sort_values("expected_profit_contact", ascending=False).head(20)
            ax2.bar(range(len(top_plot)), top_plot["expected_profit_contact"])
            ax2.set_title("Top 20 expected profits")
            ax2.set_xlabel("Rank")
            ax2.set_ylabel("Expected profit")
            st.pyplot(fig2)

        st.markdown("### Summary table")
        st.dataframe(pd.DataFrame([summary]), use_container_width=True)

        with st.expander("Show model input diagnostics"):
            st.write("Prepared model input shape:", prepared_df.shape)
            st.write("Aligned model input shape:", model_input.shape)
            st.write("First 20 aligned columns:", model_input.columns[:20].tolist())

        download_file = make_download_excel(contact_list, full_df, summary)
        st.download_button(
            label="Download results as Excel",
            data=download_file,
            file_name="credit_risk_contact_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)

st.markdown("---")
st.markdown("### Important note")
st.info(
    "This app assumes the uploaded training-columns file was saved from the exact "
    "training feature matrix after `pd.get_dummies(..., drop_first=True)`. "
    "In your notebook, save it with: `joblib.dump(X.columns.tolist(), 'training_columns.pkl')`."
)

# ============================================================
# Notes for deployment
# ============================================================
# pip install streamlit pandas numpy matplotlib openpyxl scikit-learn joblib xlrd
# streamlit run app.py
