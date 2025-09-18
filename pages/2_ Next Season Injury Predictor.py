import streamlit as st
import pandas as pd
import numpy as np
from next_season_injury_prediction.app_injury_logregression import (
    load_assets, predict_with_threshold, _get_classes   
)

st.title("⚽ Injury Prediction")

model, meta, expected_cols = load_assets()
threshold = float(meta.get("threshold", 0.5))
has_proba = bool(meta.get("has_predict_proba", hasattr(model, "predict_proba")))

st.caption(
    f"Loaded model: **{meta.get('model_name','pipe2')}** · "
    f"Threshold: **{threshold:.3f}** · "
    f"predict_proba: **{has_proba}**"
)

st.subheader("Player Features")

# Sliders for known fields (keep your current UI)
height_cm = st.slider("Height (cm)", 150, 210, 178)
prev_inj  = st.slider("Previous Injury Count", 0, 10, 0)
knee      = st.slider("Knee Strength Score", 0, 10, 5)
ham       = st.slider("Hamstring Flexibility", 0, 10, 5)
reaction  = st.slider("Reaction Time (ms)", 100, 500, 250)
balance   = st.slider("Balance Test Score", 0, 10, 5)
sprint    = st.slider("Sprint Speed (10m) [m/s]", 3.0, 9.0, 6.0, step=0.1)
sleep     = st.slider("Sleep Hours Per Night", 0, 12, 7)
stress    = st.slider("Stress Level Score", 0, 10, 5)
nutrition = st.slider("Nutrition Quality Score", 0, 10, 5)
warmup    = st.slider("Warmup Routine Adherence", 0, 10, 5)

# Build a full row aligned to expected_cols; any missing features become NaN (imputed by pipeline)
input_map = {
    "Height_cm": float(height_cm),
    "Previous_Injury_Count": float(prev_inj),
    "Knee_Strength_Score": float(knee),
    "Hamstring_Flexibility": float(ham),
    "Reaction_Time_ms": float(reaction),
    "Balance_Test_Score": float(balance),
    "Sprint_Speed_10m_s": float(sprint),
    "Sleep_Hours_Per_Night": float(sleep),
    "Stress_Level_Score": float(stress),
    "Nutrition_Quality_Score": float(nutrition),
    "Warmup_Routine_Adherence": float(warmup),
}
row = {c: input_map.get(c, np.nan) for c in expected_cols}
X_df = pd.DataFrame([row], columns=expected_cols)

# Warn if the UI doesn’t cover all expected features
missing = [c for c in expected_cols if c not in input_map]
if missing:
    st.info(f"Using imputation for missing features: {missing}")

if st.button("Predict Injury Risk"):
    pred, proba = predict_with_threshold(model, threshold, X_df)
    st.subheader("Prediction Result")
    st.write(
        f"Probability of injury **{proba:.6f}** | "
        f"Threshold **{threshold:.3f}** → Predicted label: **{pred}**"
    )

    with st.expander("Debug details"):
        try:
            classes = _get_classes(model)
            st.write("classes_:", classes)
            if hasattr(model, "predict_proba"):
                vec = model.predict_proba(X_df)[0]
                st.write("predict_proba vector:", [float(v) for v in vec])
                st.write("vector sum:", float(np.sum(vec)))
        except Exception as e:
            st.write("debug error:", repr(e))
        st.write("Input shape:", X_df.shape)
        st.write("Columns:", list(X_df.columns))