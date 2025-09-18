# app_injury_logregression.py
# Helper functions: load assets (pickle+meta+features), robust proba, unit detection, etc.

import numpy as np
import pandas as pd
import pickle, json, joblib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

_CANDIDATE_DIRS = [Path("models"), Path("next_season_injury_prediction/models")]

def _pick_path(filename: str) -> Path:
    for base in _CANDIDATE_DIRS:
        p = base / filename
        if p.exists():
            return p
    # default to first dir if nothing exists yet
    return _CANDIDATE_DIRS[0] / filename

MODEL_PKL  = _pick_path("injury_model.pkl")
MODEL_JBL  = _pick_path("injury_model.joblib")
META_PATH  = _pick_path("injury_meta.json")
FEAT_PATH  = _pick_path("features.json")

# -----------------------------------------
# Load model + meta + feature order
# -----------------------------------------
def load_assets():
    """
    Returns:
        model: sklearn Pipeline with predict_proba
        meta: dict (threshold, model_name, classes_, has_predict_proba, ...)
        feature_order: list of columns as used in training (exact order)
    """
    model = None
    # Prefer pickle; fallback to joblib if needed
    if MODEL_PKL.exists():
        with MODEL_PKL.open("rb") as f:
            model = pickle.load(f)
    elif MODEL_JBL.exists():
        model = joblib.load(MODEL_JBL)
    else:
        # last-ditch: try legacy path names
        if Path("models/injury_model.pkl").exists():
            with open("models/injury_model.pkl", "rb") as f:
                model = pickle.load(f)
        elif Path("models/injury_model.joblib").exists():
            model = joblib.load("models/injury_model.joblib")
        else:
            raise FileNotFoundError("No model file found: injury_model.pkl or injury_model.joblib")

    # Unwrap if someone saved (model, threshold)
    if isinstance(model, (tuple, list)) and len(model) >= 1 and hasattr(model[0], "predict_proba"):
        model = model[0]

    # Meta (threshold, etc.)
    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        # safe defaults
        meta = {"threshold": 0.5, "model_name": "pipe2", "has_predict_proba": hasattr(model, "predict_proba")}

    # Feature order from training
    if FEAT_PATH.exists():
        with FEAT_PATH.open("r", encoding="utf-8") as f:
            features = json.load(f)
    else:
        # fallback list (kept to avoid crash)
        features = [
            "Height_cm",
            "Previous_Injury_Count",
            "Knee_Strength_Score",
            "Hamstring_Flexibility",
            "Reaction_Time_ms",
            "Balance_Test_Score",
            "Sprint_Speed_10m_s",
            "Sleep_Hours_Per_Night",
            "Stress_Level_Score",
            "Nutrition_Quality_Score",
            "Warmup_Routine_Adherence",
        ]

    return model, meta, features

# -----------------------------------------
# Introspection helpers
# -----------------------------------------
def _get_classes(model) -> Optional[List[Any]]:
    """Return classes_ from pipeline or its final estimator."""
    cls = getattr(model, "classes_", None)
    if cls is not None:
        return list(cls)
    if hasattr(model, "steps"):
        final_est = model.steps[-1][1]
        cls = getattr(final_est, "classes_", None)
        if cls is not None:
            return list(cls)
    return None

def _positive_index(classes: Optional[List[Any]]) -> int:
    """Find the positive class index robustly."""
    if not classes or len(classes) < 2:
        return 0
    if 1 in classes:
        return list(classes).index(1)
    s = [str(c) for c in classes]
    if "1" in s:
        return s.index("1")
    # common positive names (rarely needed if labels are 0/1)
    POS = {"injury", "injured", "yes", "true", "positive", "y"}
    lc = [str(c).strip().lower() for c in classes]
    for i, name in enumerate(lc):
        if any(p == name or p in name for p in POS):
            return i
    return 1

def _get_num_stats(model):
    """
    Returns (num_cols, imputer.statistics_, scaler.mean_, scaler.scale_) if available.
    """
    try:
        pre = model.named_steps["pre"]
        for name, trans, cols in pre.transformers_:
            if name == "num":
                imputer = trans.named_steps.get("impute", None)
                scaler  = trans.named_steps.get("scale",  None)
                num_cols = list(cols)
                stats = getattr(imputer, "statistics_", None) if imputer else None
                mean  = getattr(scaler, "mean_", None) if scaler else None
                scale = getattr(scaler, "scale_", None) if scaler else None
                return num_cols, stats, mean, scale
    except Exception:
        pass
    return None, None, None, None

def _detect_units(model) -> Dict[str, str]:
    """
    Detect training units for 'Sprint_Speed_10m_s' and 'Reaction_Time_ms' from scaler.mean_.
    Returns dict: {'Sprint_Speed_10m_s': 'speed'|'time', 'Reaction_Time_ms': 'ms'|'s'}
    """
    units = {"Sprint_Speed_10m_s": "speed", "Reaction_Time_ms": "ms"}
    num_cols, _, mean, _ = _get_num_stats(model)
    if mean is None or num_cols is None:
        return units
    try:
        if "Sprint_Speed_10m_s" in num_cols:
            i = num_cols.index("Sprint_Speed_10m_s")
            sprint_mean = float(mean[i])
            units["Sprint_Speed_10m_s"] = "time" if sprint_mean < 3.0 else "speed"
        if "Reaction_Time_ms" in num_cols:
            j = num_cols.index("Reaction_Time_ms")
            rt_mean = float(mean[j])
            units["Reaction_Time_ms"] = "s" if rt_mean < 5.0 else "ms"
    except Exception:
        pass
    return units

# -----------------------------------------
# Probability & prediction
# -----------------------------------------
def proba_positive(model, X_df: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        vec = np.asarray(model.predict_proba(X_df)[0], dtype=float)
        pos_idx = _positive_index(_get_classes(model))
        if pos_idx < 0 or pos_idx >= vec.shape[0]:
            pos_idx = int(np.argmax(vec))
        return float(vec[pos_idx])
    if hasattr(model, "decision_function"):
        import math
        s = float(model.decision_function(X_df)[0])
        return 1.0 / (1.0 + math.exp(-s))
    return float(model.predict(X_df)[0])

def predict_with_threshold(model, threshold: float, X_df: pd.DataFrame):
    proba = proba_positive(model, X_df)
    pred = int(proba >= threshold)
    return pred, proba
