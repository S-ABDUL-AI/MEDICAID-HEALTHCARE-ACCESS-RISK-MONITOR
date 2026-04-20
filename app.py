"""
Healthcare Access Risk Dashboard — Streamlit entry point.

Run from repository root:
    streamlit run healthcare_access_risk_dashboard/app.py
"""

from __future__ import annotations

import html
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
    focus_recommendation_row,
    policy_brief,
    policy_insight_row,
    recommendation_for_tier,
)


def _focus_insight_for_row(display_df: pd.DataFrame, row: pd.Series) -> str:
    """Build the ‘why’ text from the exact highlighted row (always matches sidebar state)."""
    try:
        panel_full = attach_panel_medians(display_df)
        return policy_insight_row(panel_full.loc[row.name])
    except Exception:
        return str(row.get("policy_insight", "A short explanation could not be built for this row."))


def _focus_recommendation_for_row(display_df: pd.DataFrame, row: pd.Series) -> str:
    """Build a state-specific recommendation from the exact highlighted row."""
    try:
        panel_full = attach_panel_medians(display_df)
        return focus_recommendation_row(panel_full.loc[row.name])
    except Exception:
        return str(row.get("recommendation", "Recommendation is not available for this row."))

# Plain-language names for charts only (does not change model columns).
CHART_FEATURE_LABELS = {
    "median_income": "Typical income in the area",
    "uninsured_rate": "Share of people without insurance",
    "healthcare_cost_index": "Relative cost of care",
    "rural_population": "Share of people in rural communities",
}

TIER_COLORS = {
    "Low": {
        "accent": "#059669",
        "pill_bg": "#ecfdf5",
        "box_bg": "#047857",
        "box_text": "#ffffff",
    },
    "Medium": {
        "accent": "#b45309",
        "pill_bg": "#fffbeb",
        "box_bg": "#b45309",
        "box_text": "#ffffff",
    },
    "High": {
        "accent": "#dc2626",
        "pill_bg": "#fef2f2",
        "box_bg": "#b91c1c",
        "box_text": "#ffffff",
    },
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
        div.block-container { padding-top: 0.35rem; padding-bottom: 1.5rem; }
        h1 { font-size: 1.9rem; letter-spacing: -0.005em; font-weight: 600; margin-bottom: 0.15rem; }
        .policy-purpose { font-size: 0.92rem; line-height: 1.35; color: inherit; margin-bottom: 0.15rem; }
        .designer-attribution { font-size: 0.85rem; opacity: 0.85; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,0.35); }
        .focus-hero { max-width: 100%; }
        .focus-score-wrap { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 1.25rem 2rem; margin: 0.35rem 0 0.75rem 0; }
        .focus-score-label { display: block; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.72; margin-bottom: 0.2rem; }
        .focus-score-num { font-size: 2.65rem; font-weight: 700; color: #0d9488; line-height: 1; letter-spacing: -0.03em; }
        .focus-score-avg { font-size: 1.05rem; line-height: 1.35; color: #6b7280; padding-bottom: 0.2rem; }
        .focus-score-avg strong { color: #374151; font-weight: 600; }
        .focus-insight { font-size: 0.95rem; line-height: 1.5; opacity: 0.95; margin-top: 0.5rem; }
        .priority-textbox { margin-top: 0.35rem; padding: 0.8rem 0.95rem; border-radius: 0.6rem; font-size: 0.96rem; line-height: 1.5; }
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

    # --- Header: title + challenge statement (keeps focus state above the fold) ---
    st.title("Healthcare Access Risk Dashboard")
    st.markdown(
        '<p class="policy-purpose"><strong>Challenge / problem statement:</strong> '
        "Spot states under pressure on healthcare access, then act on coverage, subsidies, and rural services. "
        "Each score blends income, insurance gaps, cost signals, and rural share."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### How to use")
        st.caption("Pick data source and state—the highlighted state is the main headline on the left.")

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

        with st.expander("Public data link (live mode)"):
            st.caption("Used when “Use Real Data” is on and the connection succeeds.")
            st.code(DATASET_URL, language="text")

        st.markdown(
            '<p class="designer-attribution">Designed by: Sherriff Abdul-Hamid</p>',
            unsafe_allow_html=True,
        )

    # Real-data failure → sample data + warning (not shown when user chose sample data on purpose)
    if use_real and load_warning:
        st.warning(load_warning)

    if not use_real:
        st.caption("You are viewing **sample data** for illustration.")

    try:
        df = ensure_usable_panel(df)
    except Exception:
        df = minimal_fallback_panel()

    try:
        row_count = len(df)
    except Exception:
        row_count = 0

    src_label = (
        "Live public file (with state indicators)"
        if (use_real and source_tag == "real")
        else "Sample data"
    )
    st.caption(
        f"This tool uses publicly available U.S. data for policy analysis · "
        f"**{row_count}** states · **{src_label}**"
    )

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
    display_df["recommendation"] = display_df["recommendation"].fillna("Maintain and monitor access.")
    display_df["priority_area"] = display_df["predicted_risk_tier"].map(
        {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
    )
    display_df["priority_area"] = display_df["priority_area"].fillna("⚪ Unknown")

    try:
        panel_df = attach_panel_medians(display_df)
        display_df["policy_insight"] = panel_df.apply(policy_insight_row, axis=1)
    except Exception:
        display_df["policy_insight"] = "A short explanation could not be built for this row."

    try:
        high_ct = int((pred_series == "High").sum())
        mean_risk = float(result.risk_score.mean())
    except Exception:
        high_ct, mean_risk = 0, 0.0

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

    state_score = float(row["risk_score"])
    tier = str(row["predicted_risk_tier"])
    tier_style = TIER_COLORS.get(
        tier,
        {"accent": "#0d9488", "pill_bg": "#f0fdfa", "box_bg": "#0d9488", "box_text": "#ffffff"},
    )
    focus_recommendation = _focus_recommendation_for_row(display_df, row)

    # --- Hero: focus state (left, key message) + compact snapshot (right) ---
    hero_left, hero_right = st.columns([1.45, 0.92], gap="large")

    with hero_left:
        st.markdown("### Focus state")
        _fn = html.escape(focus_name)
        _tier = html.escape(tier)
        st.markdown(
            f'<p class="focus-hero"><span style="font-size:1.35rem;font-weight:600;">{_fn}</span>'
            f' &nbsp;·&nbsp; <span style="font-weight:700;color:{tier_style["accent"]};background:{tier_style["pill_bg"]};'
            f'padding:0.2rem 0.5rem;border-radius:999px;">{_tier}</span> priority</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="focus-score-wrap">'
            f'<div><span class="focus-score-label">Access risk score</span>'
            f'<span class="focus-score-num" style="color:{tier_style["accent"]};">{state_score:.1f}</span></div>'
            f'<div class="focus-score-avg">Average for all states in this view: '
            f"<strong>{mean_risk:.1f}</strong></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Why this priority level**")
        st.caption(
            "State-specific: how this state compares to others in the data on insurance, costs, income, and rural share."
        )
        _insight = _focus_insight_for_row(display_df, row)
        st.markdown(
            f'<div class="priority-textbox" style="background:{tier_style["box_bg"]};'
            f'color:{tier_style["box_text"]};">{html.escape(_insight)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Suggested direction**")
        st.caption(
            "General actions tied to this state’s **priority band** (high, medium, or low). "
            "The same band can share the same direction even when details differ."
        )
        st.markdown(
            f'<div class="priority-textbox" style="background:{tier_style["box_bg"]};'
            f'color:{tier_style["box_text"]};">{html.escape(focus_recommendation)}</div>',
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown("### Panel snapshot")
        st.caption("Rest of the country at a glance—details below.")
        r1, r2 = st.columns(2)
        with r1:
            st.metric(
                label="High-priority states",
                value=high_ct,
                help="Number of states in the high band on this run.",
            )
        with r2:
            st.metric(
                label="Model match rate",
                value=f"{result.accuracy:.0%}",
                help="How often the checker agrees with the high / medium / low bands on held-aside states.",
            )
        r3, r4 = st.columns(2)
        with r3:
            st.metric(label="States in view", value=int(row_count))
        with r4:
            st.metric(
                label="Data",
                value="Live file" if (use_real and source_tag == "real") else "Sample",
            )

    with st.expander("Policy brief (full narrative)", expanded=False):
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
        "priority_area",
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

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error("The dashboard hit an unexpected error but did not crash the session.")
        st.exception(exc)
