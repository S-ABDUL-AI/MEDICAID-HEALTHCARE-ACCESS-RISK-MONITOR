"""
Policy recommendations, plain-language explanations, and auto-generated briefs.
"""

from __future__ import annotations

import pandas as pd

from modeling import compute_risk_score


def recommendation_for_tier(tier: str) -> str:
    """One-line actionable recommendation by risk tier."""
    t = tier.strip().title()
    if t == "High":
        return "Expand Medicaid, increase subsidies, invest in rural clinics."
    if t == "Medium":
        return "Improve insurance coverage and affordability."
    return "Maintain and monitor access."


def policy_insight_row(row: pd.Series) -> str:
    """
    Short plain-English explanation of *why* modeled risk is elevated or subdued
    for a single state, based on feature values vs panel medians.
    """
    parts: list[str] = []
    med_income = float(row["median_income"])
    unins = float(row["uninsured_rate"])
    cost = float(row["healthcare_cost_index"])
    rural = float(row["rural_population"])

    if unins >= float(row.get("_panel_median_uninsured", unins)):
        parts.append("the share without insurance is higher than for the typical state in this view")
    else:
        parts.append("the share without insurance is lower than for the typical state in this view")

    if cost >= float(row.get("_panel_median_cost", cost)):
        parts.append("relative care costs look stronger than for the typical state in this view")
    else:
        parts.append("relative care costs look moderate compared with the typical state in this view")

    if med_income <= float(row.get("_panel_median_income", med_income)):
        parts.append("typical income in this state sits below the middle of this group")
    else:
        parts.append("typical income in this state sits above the middle of this group")

    if rural >= float(row.get("_panel_median_rural", rural)):
        parts.append("the rural share is on the high side, which can make clinics and transport harder")
    else:
        parts.append("the rural share is not the strongest factor compared with other states here")

    return (
        "Taken together for this state: "
        + "; ".join(parts[:3])
        + ". These pieces feed into the overall access risk score you see above."
    )


def attach_panel_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Add median columns for insight comparisons (prefixed to avoid collisions)."""
    out = df.copy()
    out["_panel_median_income"] = out["median_income"].median()
    out["_panel_median_uninsured"] = out["uninsured_rate"].median()
    out["_panel_median_cost"] = out["healthcare_cost_index"].median()
    out["_panel_median_rural"] = out["rural_population"].median()
    return out


def policy_brief(df: pd.DataFrame, predictions: pd.Series) -> str:
    """
    Auto-generate a 3–4 sentence policy brief from aggregate panel statistics.
    """
    n = len(df)
    high_share = (predictions == "High").mean()
    med_share = (predictions == "Medium").mean()
    low_share = (predictions == "Low").mean()
    risk = compute_risk_score(df)
    top = df.loc[risk.idxmax(), "state"] if len(df) else "N/A"
    mean_risk = float(risk.mean())

    s1 = (
        f"In this view, {n} states are scored for access pressure. "
        f"About {high_share:.0%} land in the high band, {med_share:.0%} in medium, and {low_share:.0%} in low."
    )
    s2 = (
        f"The average score is {mean_risk:.1f} on a 0–100 scale (higher means more pressure). "
        f"{top} shows the highest score in this run."
    )
    s3 = (
        "States in the high band are good candidates for stronger coverage help, subsidies people can actually use, "
        "and extra support for rural clinics and transport."
    )
    s4 = (
        "Treat these results as a starting view—pair them with local enrollment, hospital, and budget facts before large commitments."
    )
    return " ".join([s1, s2, s3, s4])
