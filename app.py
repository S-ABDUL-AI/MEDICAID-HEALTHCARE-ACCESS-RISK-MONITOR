"""
Healthcare Access Risk Dashboard — Streamlit entry point.

Run from repository root:
    streamlit run healthcare_access_risk_dashboard/app.py
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DATASET_URL, FEATURE_COLUMNS
from data import (
    ensure_usable_panel,
    fetch_public_dataset,
    generate_synthetic_state_data,
    is_valid_panel,
    minimal_fallback_panel,
)
from modeling import predict_for_dataframe, train_risk_classifier
from policy import (
    attach_panel_medians,
    policy_brief,
    policy_insight_row,
    recommendation_for_tier,
)

# Plain-language names for charts only (does not change model columns).
CHART_FEATURE_LABELS = {
    "median_income": "Typical income in the area",
    "uninsured_rate": "Share of people without insurance",
    "healthcare_cost_index": "Relative cost of care",
    "rural_population": "Share of people in rural communities",
}


@st.cache_data(show_spinner=False)
def load_dataset_cached(use_real: bool) -> tuple[pd.DataFrame, Literal["real", "simulated"], str | None]:
    """
    Load data with try/except so the dashboard always gets a usable table.

    Returns (dataframe, source_tag, warning_message). warning_message is set when
    the user asked for real data but we had to fall back to sample data.
    """
    warning_message: str | None = None
    fallback_warn = (
        "The public data file could not be loaded. "
        "The dashboard is using sample data instead."
    )

    try:
        if not use_real:
            try:
                df = generate_synthetic_state_data()
            except Exception:
                df = minimal_fallback_panel()
            df = ensure_usable_panel(df)
            return df, "simulated", None

        try:
            df, tag = fetch_public_dataset()
        except Exception:
            df = ensure_usable_panel(None)
            tag = "simulated"
            warning_message = fallback_warn
        else:
            if tag == "simulated":
                warning_message = fallback_warn

        try:
            df = ensure_usable_panel(df)
        except Exception:
            df = minimal_fallback_panel()
            tag = "simulated"
            warning_message = warning_message or fallback_warn

        if use_real and tag == "simulated" and warning_message is None:
            warning_message = fallback_warn

        if not is_valid_panel(df):
            df = minimal_fallback_panel()
            tag = "simulated"
            if use_real:
                warning_message = warning_message or fallback_warn

        return df, tag, warning_message
    except Exception:
        try:
            df = ensure_usable_panel(None)
        except Exception:
            df = minimal_fallback_panel()
        return df, "simulated", fallback_warn


def _inject_styles() -> None:
    """Spacing and typography for a readable policy layout."""
    st.markdown(
        """
        <style>
        div.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1 { letter-spacing: -0.02em; font-weight: 600; }
        .policy-purpose { font-size: 1.1rem; line-height: 1.55; color: inherit; margin-bottom: 0.5rem; }
        .data-note { font-size: 0.9rem; opacity: 0.9; margin-top: 0.25rem; }
        .designer-attribution { font-size: 0.85rem; opacity: 0.85; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,0.35); }
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

    # --- Header: title, purpose, public-data note ---
    st.title("Healthcare Access Risk Dashboard")
    st.markdown(
        '<p class="policy-purpose">'
        "This dashboard helps leaders see which states may face stronger pressure on "
        "healthcare access—so you can prioritize outreach, coverage, and rural services. "
        "Scores combine income, insurance gaps, cost pressure, and rural share into one "
        "easy-to-read picture for each state."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="data-note">This tool uses publicly available U.S. data for policy analysis.</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    with st.sidebar:
        st.markdown("### How to use")
        st.caption("Pick a data source, then choose a state to read the short policy notes.")

        data_mode = st.radio(
            "Data source",
            options=["Use Real Data", "Use Sample Data"],
            index=0,
            help=(
                "Live data pulls a public U.S. data file from the internet and builds state indicators. "
                "Sample data is built in for training sessions or when you are offline."
            ),
        )
        use_real = data_mode == "Use Real Data"

        try:
            df, source_tag, load_warning = load_dataset_cached(use_real)
        except Exception:
            df = minimal_fallback_panel()
            source_tag = "simulated"
            load_warning = (
                "The public data file could not be loaded. "
                "The dashboard is using sample data instead."
            )

        try:
            states_sorted = sorted(df["state"].astype(str).unique().tolist())
        except Exception:
            states_sorted = ["Alabama", "California", "Illinois", "New York", "Texas"]

        selected_state = st.selectbox(
            "State to highlight",
            options=states_sorted,
            index=0,
            key="focus_state",
        )

        st.divider()
        st.markdown("**Public data link (live mode)**")
        st.caption("Used when “Use Real Data” is on and the connection succeeds.")
        st.code(DATASET_URL, language="text")

    # Real-data failure → sample data + warning (not shown when user chose sample data on purpose)
    if use_real and load_warning:
        st.warning(load_warning)

    if not use_real:
        st.caption("You are viewing **sample data** for illustration.")

    try:
        row_count = len(df)
    except Exception:
        row_count = 0

    src_label = (
        "Live public file (with state indicators)"
        if (use_real and source_tag == "real")
        else "Sample data"
    )
    st.caption(f"**States in this view:** {row_count} · **Source:** {src_label}")

    st.divider()

    # --- Model run (guarded so bad rows never crash the app) ---
    try:
        result = train_risk_classifier(df)
        preds = predict_for_dataframe(result.model, df)
    except Exception:
        st.warning("The scoring step hit a problem with the current table. Loading a fresh sample panel.")
        df = ensure_usable_panel(None)
        try:
            result = train_risk_classifier(df)
            preds = predict_for_dataframe(result.model, df)
        except Exception:
            df = minimal_fallback_panel()
            result = train_risk_classifier(df)
            preds = predict_for_dataframe(result.model, df)

    pred_series = pd.Series(preds, index=df.index, name="predicted_risk_tier")

    display_df = df.copy()
    display_df["risk_score"] = result.risk_score.values
    display_df["true_label_from_score"] = result.labels.values
    display_df["predicted_risk_tier"] = pred_series.values
    display_df["recommendation"] = display_df["predicted_risk_tier"].map(recommendation_for_tier)

    try:
        panel_df = attach_panel_medians(display_df)
        display_df["policy_insight"] = panel_df.apply(policy_insight_row, axis=1)
    except Exception:
        display_df["policy_insight"] = "A short explanation could not be built for this row."

    # --- Key indicators ---
    st.markdown("### At a glance")
    st.markdown("")  # spacing
    try:
        high_ct = int((pred_series == "High").sum())
        mean_risk = float(result.risk_score.mean())
        med_income = float(df["median_income"].median())
    except Exception:
        high_ct, mean_risk, med_income = 0, 0.0, 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Average access risk score",
            value=f"{mean_risk:.1f}",
            help="From 0 (lower pressure) to 100 (higher pressure) within this group of states.",
        )
    with c2:
        st.metric(
            label="States flagged high priority",
            value=high_ct,
            help="Count of states in the “high” band on this run.",
        )
    with c3:
        st.metric(
            label="How often the model matches the score bands",
            value=f"{result.accuracy:.0%}",
            help="Share of held-back states where the model agrees with the high / medium / low bands from the score.",
        )
    with c4:
        st.metric(
            label="Data in use",
            value="Live public file" if (use_real and source_tag == "real") else "Sample",
        )

    st.markdown("")
    st.markdown("### Policy brief")
    try:
        st.write(policy_brief(display_df, pred_series))
    except Exception:
        st.write("A full brief could not be generated for this panel.")

    st.divider()

    left_intro, right_intro = st.columns(2)
    with left_intro:
        st.markdown("### Data preview")
        st.caption("First rows of the indicators used in this view.")
        preview_cols = ["state", *FEATURE_COLUMNS]
        try:
            st.dataframe(display_df[preview_cols].head(15), use_container_width=True, hide_index=True)
        except Exception:
            st.write("Table preview is not available.")

    with right_intro:
        st.markdown("### About the numbers")
        st.markdown(
            "The **access risk score** pulls together insurance gaps, cost pressure, income, "
            "and rural share. **High / medium / low** bands split states into three groups so "
            "you can spot who may need attention first. The **match rate** tells you how closely "
            "the automated checker lines up with those bands on a slice of states held aside for testing."
        )

    st.divider()

    st.markdown("### State-by-state results")
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
    try:
        st.dataframe(display_df[table_cols], use_container_width=True, height=420)
    except Exception:
        st.warning("The full results table could not be displayed.")

    st.divider()

    st.markdown("### Charts")
    chart_df = display_df.copy()
    chart_df["state"] = chart_df["state"].astype(str)

    left, right = st.columns(2)
    with left:
        st.markdown("**Access risk score by state**")
        try:
            fig_bar = px.bar(
                chart_df.sort_values("risk_score", ascending=False),
                x="state",
                y="risk_score",
                color="predicted_risk_tier",
                category_orders={"predicted_risk_tier": ["Low", "Medium", "High"]},
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"},
                labels={"risk_score": "Access risk score", "state": "State"},
            )
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                height=480,
                margin=dict(b=120),
                legend_title_text="Priority level",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception:
            st.warning("The bar chart could not be drawn for this data.")

    with right:
        st.markdown("**What most influences the scores**")
        try:
            fi = result.feature_importances.reset_index()
            fi.columns = ["feature", "importance"]
            fi["feature_label"] = fi["feature"].map(lambda x: CHART_FEATURE_LABELS.get(str(x), str(x)))
            fig_fi = px.bar(
                fi,
                x="importance",
                y="feature_label",
                orientation="h",
                labels={"importance": "Strength of influence", "feature_label": "Indicator"},
            )
            fig_fi.update_layout(height=480, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            st.warning("The influence chart could not be drawn for this data.")

    st.divider()

    st.markdown("### Focus state")
    st.caption("Read the recommendation and plain-language “why” for one state.")

    try:
        matches = display_df.loc[display_df["state"].astype(str) == str(selected_state)]
        if len(matches) == 0:
            row = display_df.iloc[0]
            focus_name = str(row["state"])
        else:
            row = matches.iloc[0]
            focus_name = str(selected_state)
    except Exception:
        row = display_df.iloc[0]
        focus_name = str(row.get("state", "Unknown"))

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"**{focus_name}** — **{row['predicted_risk_tier']}** priority")
        st.metric("Access risk score", f"{float(row['risk_score']):.1f}")
        st.markdown("**Suggested direction**")
        st.write(str(row["recommendation"]))
    with fc2:
        st.markdown("**Why this priority level**")
        st.write(str(row.get("policy_insight", "")))

    st.markdown(
        '<p class="designer-attribution">Designed by: Sherriff Abdul-Hamid</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
