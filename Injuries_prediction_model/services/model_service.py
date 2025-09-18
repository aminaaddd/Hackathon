from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path
import streamlit as st

from Injuries_prediction_model.clean_data import clean_data, prepare_features
from Injuries_prediction_model.model import train_model_with_weights
from Injuries_prediction_model.evaluate_model import evaluate_model

@st.cache_data
def run_clean_and_prepare(df: pd.DataFrame):
    cleaned = clean_data(df.copy())
    X, y = prepare_features(cleaned)
    return cleaned, X, y

@st.cache_resource
def load_or_train_model(df: pd.DataFrame, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "injuries_prediction.pkl"
    cols_path = model_dir / "train_columns.pkl"

    if model_path.exists() and cols_path.exists():
        clf = joblib.load(model_path)
        train_cols = joblib.load(cols_path)
        return clf, list(train_cols)

    cleaned, X, y = run_clean_and_prepare(df)
    y_pred_w, y_test_w, clf_w, X_test_w, X_train_w = train_model_with_weights(X, y)

    try:
        evaluate_model(y_test_w, y_pred_w, clf_w, X_test_w)
    except Exception:
        pass

    joblib.dump(clf_w, model_path)
    joblib.dump(X_train_w.columns, cols_path)
    return clf_w, list(X_train_w.columns)

def prepare_single_row_features(row: pd.Series, raw_df: pd.DataFrame) -> pd.DataFrame:
    # thin wrapper to keep imports centralized
    from Injuries_prediction_model.helpers import prepare_single_row_features as _prep
    return _prep(row, raw_df)

def predict_row(clf, X_single: pd.DataFrame):
    y = clf.predict(X_single)[0]
    prob = None
    if hasattr(clf, "predict_proba"):
        try:
            prob = float(clf.predict_proba(X_single)[0, 1])
        except Exception:
            prob = None
    return y, prob
