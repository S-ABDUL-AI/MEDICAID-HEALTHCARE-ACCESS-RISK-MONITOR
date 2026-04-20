"""Configuration for the Healthcare Access Risk Dashboard."""

# Public hosted CSV (Plotly sample dataset — U.S. state agricultural exports, 2011).
# We derive state-level *indicators* for modeling; see data.py docstrings.
DATASET_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv"
)

# Random seed for reproducible synthetic data and train/test splits.
RANDOM_STATE = 42

# Feature columns used by the risk model (must match enriched / synthetic schema).
FEATURE_COLUMNS = [
    "median_income",
    "uninsured_rate",
    "healthcare_cost_index",
    "rural_population",
]
