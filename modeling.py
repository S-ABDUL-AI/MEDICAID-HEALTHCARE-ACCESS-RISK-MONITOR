"""
Risk scoring, label construction, and RandomForest training utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import FEATURE_COLUMNS, RANDOM_STATE


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite healthcare access risk score (0–100).

    Higher uninsured rate, higher cost index, lower median income, and higher rural
    share all increase risk.
    """
    work = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # Min–max within the current panel so scores are comparable across rows.
    def mm(col: str, invert: bool = False) -> pd.Series:
        x = work[col].astype(float)
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-9:
            z = pd.Series(0.5, index=x.index)
        else:
            z = (x - lo) / (hi - lo)
        return 1.0 - z if invert else z

    unins = mm("uninsured_rate", invert=False)
    cost = mm("healthcare_cost_index", invert=False)
    income = mm("median_income", invert=True)
    rural = mm("rural_population", invert=False)

    score = 100.0 * (0.30 * unins + 0.30 * cost + 0.25 * income + 0.15 * rural)
    return score.clip(0, 100)


def risk_scores_to_labels(scores: pd.Series) -> pd.Series:
    """Map continuous scores to Low / Medium / High by tertiles within the panel."""
    q1, q2 = scores.quantile([1 / 3, 2 / 3])
    labels = np.where(scores <= q1, "Low", np.where(scores <= q2, "Medium", "High"))
    return pd.Series(labels, index=scores.index, dtype="category")


@dataclass
class TrainedModelResult:
    """Bundle of model outputs for the Streamlit UI."""

    model: RandomForestClassifier
    accuracy: float
    y_true_holdout: np.ndarray
    y_pred_holdout: np.ndarray
    feature_importances: pd.Series
    labels: pd.Series
    risk_score: pd.Series


def train_risk_classifier(df: pd.DataFrame) -> TrainedModelResult:
    """
    Train a RandomForest to predict Low/Medium/High labels derived from risk_score.

    Labels are defined from the same composite score so the forest learns non-linear
    interactions among the four drivers while staying aligned with policy logic.
    """
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    risk = compute_risk_score(df)
    y = risk_scores_to_labels(risk)

    y_str = y.astype(str)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_str,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_str,
        )
    except ValueError:
        # Too few rows per class for stratified split — fall back to unstratified.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_str,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(
        ascending=False
    )

    return TrainedModelResult(
        model=clf,
        accuracy=acc,
        y_true_holdout=y_test.to_numpy(),
        y_pred_holdout=y_pred,
        feature_importances=importances,
        labels=y.astype(str),
        risk_score=risk,
    )


def predict_for_dataframe(model: RandomForestClassifier, df: pd.DataFrame) -> np.ndarray:
    """Run inference on the full panel."""
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return model.predict(X)
