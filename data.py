"""
Data loading and preparation.

Real-data path: fetch the public Plotly CSV, then build healthcare-style indicators
per state using deterministic transforms of the published export columns (no API).
If the network request fails, callers should fall back to synthetic data.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pandas as pd
import requests

from config import DATASET_URL, FEATURE_COLUMNS, RANDOM_STATE

REQUIRED_COLUMNS = ["state", *FEATURE_COLUMNS]


def is_valid_panel(df: pd.DataFrame | None) -> bool:
    """True if the frame has the columns and rows needed for modeling."""
    if df is None or len(df) == 0:
        return False
    if not all(c in df.columns for c in REQUIRED_COLUMNS):
        return False
    numeric = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all().all():
        return False
    return True


def minimal_fallback_panel() -> pd.DataFrame:
    """
    Last-resort tiny panel so the app can still render if every other load path fails.
    Values are placeholders for continuity only.
    """
    return pd.DataFrame(
        {
            "state": ["Alabama", "California", "Illinois", "New York", "Texas"],
            "median_income": [48_000.0, 72_000.0, 61_000.0, 68_000.0, 55_000.0],
            "uninsured_rate": [12.0, 8.0, 9.0, 7.0, 14.0],
            "healthcare_cost_index": [102.0, 118.0, 105.0, 115.0, 99.0],
            "rural_population": [0.42, 0.12, 0.22, 0.15, 0.28],
        }
    )


def ensure_usable_panel(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a validated panel, falling back to synthetic then minimal data."""
    if is_valid_panel(df):
        return df  # type: ignore[return-value]
    try:
        syn = generate_synthetic_state_data()
        if is_valid_panel(syn):
            return syn
    except Exception:
        pass
    return minimal_fallback_panel()


def _normalize_series(s: pd.Series) -> pd.Series:
    """Min–max scale to [0, 1], handling constant series."""
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def generate_synthetic_state_data(seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Simulated U.S.-style panel when the public URL is unavailable or user chooses sample data.

    Columns: state, median_income, uninsured_rate, healthcare_cost_index, rural_population.
    """
    rng = np.random.default_rng(seed)
    states = [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
        "District of Columbia",
    ]
    n = len(states)
    median_income = rng.integers(38_000, 95_000, size=n)
    uninsured_rate = rng.uniform(4.0, 22.0, size=n)
    healthcare_cost_index = rng.uniform(88.0, 132.0, size=n)
    rural_population = rng.uniform(0.08, 0.72, size=n)

    df = pd.DataFrame(
        {
            "state": states,
            "median_income": median_income.astype(float),
            "uninsured_rate": uninsured_rate,
            "healthcare_cost_index": healthcare_cost_index,
            "rural_population": rural_population,
        }
    )
    return df


def _read_exports_csv_bytes(content: bytes) -> pd.DataFrame:
    """Parse the Plotly ag-export CSV from raw bytes."""
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def exports_to_healthcare_indicators(exports_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build modeling columns from the public export table.

    The hosted file is agricultural exports by state (not literal CMS/Census fields).
    We map export structure to *indicator-style* continuous variables so the dashboard
    still exercises a real public CSV while staying transparent in the UI copy.
    """
    df = exports_df.copy()
    if "state" not in df.columns:
        raise ValueError("Expected a 'state' column in the exports dataset.")

    total_col = "total_exports" if "total_exports" in df.columns else None
    if total_col is None:
        for c in df.columns:
            if "total" in c and c not in ("total_fruits", "total_veggies"):
                total_col = c
                break
    if total_col is None:
        raise ValueError("Could not locate total exports column.")

    total = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
    commodity_cols = [c for c in ("beef", "corn", "wheat", "cotton") if c in df.columns]
    commodity_sum = (
        df[commodity_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        if commodity_cols
        else total * 0.0
    )
    commodity_intensity = (commodity_sum / (total + 1e-6)).clip(0, 1)

    income_proxy = 42_000 + _normalize_series(total) * (88_000 - 42_000)
    unins_proxy = 5.0 + _normalize_series(commodity_intensity) * 14.0
    cost_proxy = 90.0 + _normalize_series(np.log1p(total)) * 38.0
    rural_proxy = (
        0.12
        + 0.55 * _normalize_series(commodity_intensity)
        + 0.18 * (1.0 - _normalize_series(np.log1p(total)))
    ).clip(0.05, 0.85)

    out = pd.DataFrame(
        {
            "state": df["state"].astype(str),
            "median_income": income_proxy,
            "uninsured_rate": unins_proxy,
            "healthcare_cost_index": cost_proxy,
            "rural_population": rural_proxy,
        }
    )
    out = out.drop_duplicates(subset=["state"], keep="first").reset_index(drop=True)
    return out


def fetch_public_dataset() -> tuple[pd.DataFrame, Literal["real", "simulated"]]:
    """
    Load the hosted CSV and return enriched healthcare-indicator rows.

    Returns (dataframe, source_tag) where source_tag is 'real' or 'simulated' on failure.
    Wrapped in try/except so callers are never interrupted by network or parse errors.
    """
    try:
        resp = requests.get(DATASET_URL, timeout=20)
        resp.raise_for_status()
        raw = _read_exports_csv_bytes(resp.content)
        enriched = exports_to_healthcare_indicators(raw)
        missing = [c for c in FEATURE_COLUMNS if c not in enriched.columns]
        if missing:
            raise ValueError(f"Missing expected columns after enrich: {missing}")
        if not is_valid_panel(enriched):
            raise ValueError("Enriched dataset failed validation.")
        return enriched, "real"
    except Exception:
        try:
            syn = generate_synthetic_state_data()
            return ensure_usable_panel(syn), "simulated"
        except Exception:
            return minimal_fallback_panel(), "simulated"


