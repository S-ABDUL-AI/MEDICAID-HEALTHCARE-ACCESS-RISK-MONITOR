"""
Healthcare Access Risk Dashboard — Streamlit entry point.

Run from repository root:
    streamlit run healthcare_access_risk_dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from typing import Literal

from config import DATASET_URL, FEATURE_COLUMNS
from data import fetch_public_dataset, generate_synthetic_state_data
from modeling import predict_for_dataframe, train_risk_classifier
from policy import (
    attach_panel_medians,
    policy_brief,
    policy_insight_row,
    recommendation_for_tier,
)


@st.cache_data(show_spinner=False)
def load_dataset_cached(use_real: bool) -> tuple[pd.DataFrame, Literal["real", "simulated"], str | None]:
    """
    Cached data load for the dashboard.

    Returns (dataframe, source_tag, error_message). When the user requests real data
    but the URL fails, we return simulated rows and a non-null error_message.
    """
    if not use_real:
        return generate_synthetic_state_data(), "simulated", None

    df, tag = fetch_public_dataset()
    if tag == "simulated":
        return df, "simulated", "Primary dataset URL could not be loaded; using simulated data."
    return df, "real", None


def _inject_styles() -> None:
    """Lightweight dashboard styling (Streamlit-native first, small CSS overlay)."""
    st.markdown(
        """
        <style>
        div.block-container { padding-top: 1.25rem; }
        h1 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Healthcare Access Risk Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.title("Healthcare Access Risk Dashboard")
    st.caption(
        "Identify populations at higher modeled risk of losing healthcare access using "
        "transparent indicators and a Random Forest classifier."
    )

    with st.sidebar:
        st.header("Controls")
        data_mode = st.radio(
            "Data source",
            options=["Use Real Data", "Use Sample Data"],
            index=0,
            help=(
                "Real data loads the public Plotly CSV and derives state indicators. "
                "Sample data uses a reproducible simulated panel."
            ),
        )
        use_real = data_mode == "Use Real Data"

        df, source_tag, load_error = load_dataset_cached(use_real)

        states_sorted = sorted(df["state"].astype(str).unique().tolist())
        selected_state = st.selectbox(
            "Focus state (for detail cards)",
            options=states_sorted,
            index=0,
        )

        st.divider()
        st.markdown("**Dataset URL (real path)**")
        st.code(DATASET_URL, language="text")

    # User-facing data notices
    if not use_real or source_tag == "simulated":
        st.info("Using simulated data")
    if load_error:
        st.warning(load_error)

    st.markdown(
        f"**Active panel:** {len(df)} rows · "
        f"**Source:** {'Public CSV with derived indicators' if source_tag == 'real' and use_real else 'Simulated panel'}"
    )

    # Train model and predictions
    result = train_risk_classifier(df)
    preds = predict_for_dataframe(result.model, df)
    pred_series = pd.Series(preds, index=df.index, name="predicted_risk_tier")

    display_df = df.copy()
    display_df["risk_score"] = result.risk_score.values
    display_df["true_label_from_score"] = result.labels.values
    display_df["predicted_risk_tier"] = pred_series.values
    display_df["recommendation"] = display_df["predicted_risk_tier"].map(recommendation_for_tier)

    panel_df = attach_panel_medians(display_df)
    display_df["policy_insight"] = panel_df.apply(policy_insight_row, axis=1)

    # --- Key metrics ---
    high_ct = int((pred_series == "High").sum())
    mean_risk = float(result.risk_score.mean())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mean modeled risk score", f"{mean_risk:.1f}", help="0–100 composite from uninsured, cost, income, rural share.")
    with c2:
        st.metric("High-risk jurisdictions", high_ct)
    with c3:
        st.metric("Holdout accuracy", f"{result.accuracy:.1%}", help="RandomForest on tertile labels from the same composite score.")
    with c4:
        st.metric("Data mode", "Real CSV" if (use_real and source_tag == "real") else "Simulated")

    st.subheader("Policy brief")
    st.write(policy_brief(display_df, pred_series))

    st.subheader("Dataset preview")
    preview_cols = ["state", *FEATURE_COLUMNS]
    st.dataframe(display_df[preview_cols].head(15), use_container_width=True, hide_index=True)

    st.subheader("Model diagnostics")
    st.write(
        f"**Holdout accuracy (RandomForestClassifier):** {result.accuracy:.3f}. "
        "Labels are tertiles of the composite risk score, so accuracy reflects how well "
        "the forest recovers thresholded nonlinear structure in the four drivers."
    )

    st.subheader("Predictions table")
    table_cols = [
        "state",
        "median_income",
        "uninsured_rate",
        "healthcare_cost_index",
        "rural_population",
        "risk_score",
        "predicted_risk_tier",
        "recommendation",
        "policy_insight",
    ]
    st.dataframe(display_df[table_cols], use_container_width=True, height=420)

    # --- Charts ---
    chart_df = display_df.copy()
    chart_df["state"] = chart_df["state"].astype(str)

    left, right = st.columns(2)
    with left:
        st.markdown("**Modeled healthcare risk by state**")
        fig_bar = px.bar(
            chart_df.sort_values("risk_score", ascending=False),
            x="state",
            y="risk_score",
            color="predicted_risk_tier",
            category_orders={"predicted_risk_tier": ["Low", "Medium", "High"]},
            color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"},
            labels={"risk_score": "Composite risk score", "state": "State"},
        )
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            height=480,
            margin=dict(b=120),
            legend_title_text="Predicted tier",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown("**Feature importance (Random Forest)**")
        fi = result.feature_importances.reset_index()
        fi.columns = ["feature", "importance"]
        fig_fi = px.bar(
            fi,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig_fi.update_layout(height=480, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_fi, use_container_width=True)

    # --- Focus state detail ---
    st.subheader("Focus state — policy layer")
    row = display_df.loc[display_df["state"].astype(str) == selected_state].iloc[0]
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"**{selected_state}** — predicted **{row['predicted_risk_tier']}** risk")
        st.metric("Composite risk score", f"{float(row['risk_score']):.1f}")
        st.write("**Recommendation:** " + str(row["recommendation"]))
    with fc2:
        st.markdown("**Why this risk level (plain English)**")
        st.write(row["policy_insight"])


if __name__ == "__main__":
    main()
