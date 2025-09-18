import numpy as np
import pandas as pd
import plotly.express as px

def plot_1d_curves(cal, X_train, candidates=None):
    if candidates is None:
        candidates = ["Stress_Level_Score", "Sleep_Hours_Per_Night", "Reaction_Time_ms"]
    candidates = [c for c in candidates if c in X_train.columns]

    figs = []
    if candidates:
        num_defaults = X_train.select_dtypes(include=np.number).median()
        cat_df = X_train.select_dtypes(exclude=np.number)
        cat_defaults = cat_df.mode().iloc[0] if not cat_df.empty else pd.Series(dtype=object)

        for feat in candidates[:3]:
            vals = np.linspace(*np.percentile(X_train[feat].dropna(), [5, 95]), 30)
            rows = []
            for v in vals:
                row = {}
                for c in X_train.columns:
                    if c == feat: row[c] = v
                    elif c in num_defaults.index: row[c] = num_defaults[c]
                    elif c in cat_defaults.index: row[c] = cat_defaults[c]
                    else: row[c] = None
                rows.append(row)

            G = pd.DataFrame(rows, columns=X_train.columns)
            risk = cal.predict_proba(G)[:, 1]

            risk_df = pd.DataFrame({feat: vals, "Predicted risk": risk})
            fig = px.line(risk_df, x=feat, y="Predicted risk", title=f"Risk vs {feat}")
            fig.update_layout(template="plotly_white")
            figs.append(fig)
    return figs
