# injury_predict_app.py — robust probability selection + exact X2 columns
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

META_PATH = Path("next_season_injury_prediction/models/injury_meta.json")
MODEL_PATH = Path("next_season_injury_prediction/models/injury_model.joblib")
FEAT_PATH  = Path("next_season_injury_prediction/models/features.json")

# ---------------------------
# Load model + meta
# ---------------------------
def load_assets():
    model = joblib.load(MODEL_PATH)
    # Unwrap if someone saved (model, threshold)
    if isinstance(model, (tuple, list)) and len(model) >= 1 and hasattr(model[0], "predict_proba"):
        model = model[0]

    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"threshold": 0.5, "model_name": "pipe2", "has_predict_proba": hasattr(model, "predict_proba")}

    # Load expected features from training
    if FEAT_PATH.exists():
        with FEAT_PATH.open("r", encoding="utf-8") as f:
            features = json.load(f)
    else:
        # Fallback to the hardcoded list (not recommended)
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

def _get_classes(model):
    """Return classes_ from pipeline or its final estimator."""
    cls = getattr(model, "classes_", None)
    if cls is not None:
        return list(cls)
    # Fall back to final step of a pipeline
    est = getattr(model, "steps", None)
    if est:
        final_est = est[-1][1]
        cls = getattr(final_est, "classes_", None)
        if cls is not None:
            return list(cls)
    return None  # unknown

def _positive_index(classes):
    """
    Figure out which class is 'positive'. We try common labels first,
    otherwise prefer label 1 if present, else pick a likely positive by name.
    """
    if classes is None or len(classes) < 2:
        return 0
    # 1) exact 1
    if 1 in classes:
        return list(classes).index(1)
    # 2) common positive strings
    POS_NAMES = {"injury", "yes", "positive", "true", "y", "1", "yes injury", "injured"}
    lc = [str(c).strip().lower() for c in classes]
    for i, name in enumerate(lc):
        if any(p in name for p in POS_NAMES):
            return i
    # 3) default to the second column (sklearn returns in classes_ order)
    return 1

# ---------------------------
# Probability of positive class
# ---------------------------
def proba_positive(model, X_df: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba_vec = np.asarray(model.predict_proba(X_df)[0], dtype=float)
        classes = _get_classes(model)
        pos_idx = _positive_index(classes)
        # Safety: keep in bounds
        if pos_idx < 0 or pos_idx >= proba_vec.shape[0]:
            pos_idx = int(np.argmax(proba_vec))
        return float(proba_vec[pos_idx])

    # Fallbacks (rare)
    if hasattr(model, "decision_function"):
        import math
        s = float(model.decision_function(X_df)[0])
        return 1.0 / (1.0 + math.exp(-s))
    return float(model.predict(X_df)[0])

def predict_with_threshold(model, threshold: float, X_df: pd.DataFrame):
    proba = proba_positive(model, X_df)
    pred = int(proba >= threshold)
    return pred, proba

