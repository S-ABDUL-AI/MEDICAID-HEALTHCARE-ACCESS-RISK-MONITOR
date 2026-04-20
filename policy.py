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
        parts.append("uninsured-rate pressure is at or above the national sample median")
    else:
        parts.append("uninsured-rate pressure is below the national sample median")

    if cost >= float(row.get("_panel_median_cost", cost)):
        parts.append("cost-index signals are relatively strong")
    else:
        parts.append("cost-index signals are relatively moderate")

    if med_income <= float(row.get("_panel_median_income", med_income)):
        parts.append("income capacity in this row is on the lower side of the panel")
    else:
        parts.append("income capacity in this row is on the higher side of the panel")

    if rural >= float(row.get("_panel_median_rural", rural)):
        parts.append("rural share is elevated in this panel, which can strain facility access")
    else:
        parts.append("rural share is not the dominant driver relative to peers")

    return (
        "In this modeled snapshot, "
        + "; ".join(parts[:3])
        + ", which together shape the composite access-risk score."
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
        f"This dashboard summarizes modeled healthcare access risk for {n} jurisdictions, "
        f"with about {high_share:.0%} classified as high risk, {med_share:.0%} medium, and {low_share:.0%} low under the current scoring rules."
    )
    s2 = (
        f"The mean composite risk score is {mean_risk:.1f} on a 0–100 scale, with the highest modeled burden appearing in {top} in this run."
    )
    s3 = (
        "Policymakers should treat high-risk clusters as candidates for coverage expansions, "
        "subsidy calibration, and targeted rural-capacity investments, while maintaining surveillance where risk is already low."
    )
    s4 = (
        "Figures are indicators derived for exploration; confirm with state-specific enrollment, "
        "provider, and financing data before committing major funding decisions."
    )
    return " ".join([s1, s2, s3, s4])
