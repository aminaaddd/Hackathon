# 2_Next Season Injury Predictor.py
# Streamlit UI page using the helper functions (same folder import).

import streamlit as st
import numpy as np
import pandas as pd

from next_season_injury_prediction.app_injury_logregression import (
    load_assets,
    predict_with_threshold,
    _get_classes,
    _get_num_stats,
    _detect_units,
)

# ---------------------------
# Small UI helpers
# ---------------------------
def _risk_label(p: float):
    if p >= 0.75:
        return "High", "⚠️", "error"
    if p >= 0.5:
        return "Moderate", "🟠", "warning"
    return "Low", "✅", "success"

def _badge(text, kind="neutral"):
    colors = {
        "success": "#16a34a",
        "warning": "#d97706",
        "error":   "#dc2626",
        "neutral": "#475569",
    }
    color = colors.get(kind, "#475569")
    return f"""
    <span style="display:inline-block;padding:4px 8px;border-radius:999px;
                 background:{color}20;color:{color};font-weight:600;font-size:0.9rem;">
        {text}
    </span>
    """

# ---------------------------
# Page UI
# ---------------------------
st.set_page_config(page_title="Next Season Injury Predictor", page_icon="⚽", layout="wide")
st.title("Next Season Injury Predictor")

model, meta, feature_order = load_assets()
threshold = float(meta.get("threshold", 0.5))
has_proba = bool(meta.get("has_predict_proba", hasattr(model, "predict_proba")))
classes = _get_classes(model)
units = _detect_units(model)

st.caption(
    f"Model: **{meta.get('model_name','pipe2')}** · "
    f"Threshold: **{threshold:.3f}** · `predict_proba`: **{has_proba}**"
)

left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.subheader("Player Features")

    g1a, g1b = st.columns(2)
    with g1a:
        height_cm = st.slider("Height (cm)", 150, 210, 178)
        prev_inj  = st.slider("Previous Injury Count", 0, 10, 0)
        sleep     = st.slider("Sleep Hours / Night", 0, 12, 8)
        stress    = st.slider("Stress Level (0–100)", 0, 100, 50)
        warmup_ok = st.toggle("Warmup Routine Adherence", value=True, help="On = 1, Off = 0")
    with g1b:
        knee      = st.slider("Knee Strength (0–100)", 0, 100, 75)
        ham       = st.slider("Hamstring Flexibility (0–100)", 0, 100, 75)
        balance   = st.slider("Balance Test (0–100)", 0, 100, 85)
        nutrition = st.slider("Nutrition Quality (0–100)", 0, 100, 80)
        reaction_ms = st.slider("Reaction Time (ms)", 150, 400, 250)

    sprint_mps = st.slider("Sprint Speed over 10m (m/s)", 4.0, 8.0, 6.0, step=0.05)

    # Convert to training units
    sprint_feature   = (10.0 / float(sprint_mps)) if units.get("Sprint_Speed_10m_s") == "time" else float(sprint_mps)
    reaction_feature = (float(reaction_ms) / 1000.0) if units.get("Reaction_Time_ms") == "s" else float(reaction_ms)
    warmup = int(warmup_ok)

    row = {
        "Height_cm": float(height_cm),
        "Previous_Injury_Count": float(prev_inj),
        "Knee_Strength_Score": float(knee),
        "Hamstring_Flexibility": float(ham),
        "Reaction_Time_ms": reaction_feature,
        "Balance_Test_Score": float(balance),
        "Sprint_Speed_10m_s": float(sprint_feature),
        "Sleep_Hours_Per_Night": float(sleep),
        "Stress_Level_Score": float(stress),
        "Nutrition_Quality_Score": float(nutrition),
        "Warmup_Routine_Adherence": float(warmup),
    }

    cols = feature_order or list(row.keys())
    X_df = pd.DataFrame([[row[c] for c in cols]], columns=cols)

    missing = [c for c in cols if c not in row]
    if missing:
        st.info(f"Using imputation for missing features: {missing}")

    predict_clicked = st.button("🔮 Predict Injury Risk", type="primary", use_container_width=True)

with right:
    st.subheader("Result")

    if predict_clicked:
        pred, proba = predict_with_threshold(model, threshold, X_df)
        risk, icon, kind = _risk_label(proba)

        # Clear statement
        if pred == 1:
            st.error(f"{icon} High risk of **injury**", icon=None)
        else:
            st.success(f"{icon} Low risk of injury", icon=None)

        st.markdown(_badge(f"Risk: {risk}", kind), unsafe_allow_html=True)
        st.metric("Probability of Injury", f"{proba*100:.1f}%", delta=None)
        st.progress(min(100, int(round(proba * 100))), text="Injury risk")
        st.caption("Decision is based on the saved model threshold; details below.")

        with st.expander("Advanced details"):
            st.write("**Model metadata**")
            colA, colB, colC = st.columns(3)
            with colA:
                st.write("Threshold:", f"{threshold:.3f}")
            with colB:
                st.write("Classes:", classes)
            with colC:
                st.write("Units:",
                         {"Sprint_Speed_10m_s": units.get("Sprint_Speed_10m_s"),
                          "Reaction_Time_ms": units.get("Reaction_Time_ms")})

            try:
                if hasattr(model, "predict_proba"):
                    vec = model.predict_proba(X_df)[0]
                    st.write("predict_proba vector:", [float(v) for v in vec])
                    st.write("vector sum:", float(np.sum(vec)))
            except Exception as e:
                st.write("predict_proba error:", repr(e))

            try:
                if hasattr(model, "decision_function"):
                    st.write("decision_function:", float(model.decision_function(X_df)[0]))
            except Exception as e:
                st.write("decision_function error:", repr(e))

            # z-scores to spot scale mismatches instantly
            num_cols, _, mean, scale = _get_num_stats(model)
            try:
                if num_cols is not None and mean is not None and scale is not None:
                    rows = []
                    for c, m, s in zip(num_cols, mean, scale):
                        if c in X_df.columns and s != 0:
                            val = float(X_df[c].iloc[0])
                            z = (val - m) / s
                            rows.append({
                                "feature": c,
                                "value": round(val, 4),
                                "mean_train": round(float(m), 4),
                                "scale_train": round(float(s), 4),
                                "z_score": round(float(z), 3),
                                "abs_z": abs(float(z)),
                            })
                    if rows:
                        zdf = pd.DataFrame(rows).sort_values("abs_z", ascending=False).drop(columns=["abs_z"])
                        st.dataframe(zdf, use_container_width=True, hide_index=True)
            except Exception as e:
                st.write("z-score calc error:", repr(e))

            st.write("**Input frame**")
            st.write("shape:", X_df.shape)
            st.write("columns:", list(X_df.columns))
            st.write("row values:", [float(X_df.iloc[0, i]) for i in range(X_df.shape[1])])

    else:
        st.info("Set the features on the left, then click **Predict Injury Risk**.")
        st.write("Waiting for prediction…")
