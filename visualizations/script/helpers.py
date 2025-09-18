import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

RNG = 42
DATA_PATH = Path("data.csv")

pd.set_option("display.max_columns", 200)


def prepare_xy(
    df,
    target_col: str = "Injury_Next_Season",
    drop_cols: tuple[str, ...] = ("BMI",),
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return X, y, and lists of numeric/categorical feature names."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")

    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, num_cols, cat_cols


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, rng: int = RNG
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=rng)


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))]), cat_cols),
        ]
    )

def make_model(C: float = 0.2, max_iter: int = 2000, rng: int = RNG) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=max_iter, random_state=rng)


def make_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = make_preprocessor(num_cols, cat_cols)
    clf = make_model()
    return Pipeline([("pre", pre), ("clf", clf)])

# ---------- Training & CV ----------
def crossval(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    return {
        "cv_auroc": float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()),
        "cv_pr_auc": float(cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1).mean()),
    }, cv


def fit(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe.fit(X_train, y_train)
    return pipe


def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]

# ---------- Threshold & Metrics ----------
def best_f1_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec_m, rec_m, thr_m = prec[:-1], rec[:-1], thr
    f1 = 2 * prec_m * rec_m / (prec_m + rec_m + 1e-9)
    return float(thr_m[np.nanargmax(f1)]) if thr_m.size else 0.5


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    y_hat = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "report": classification_report(y_true, y_hat, output_dict=False),
        "confusion_matrix": confusion_matrix(y_true, y_hat).tolist(),
        "holdout_auroc": float(roc_auc_score(y_true, proba)),
        "holdout_pr_auc": float(average_precision_score(y_true, proba)),
    }
