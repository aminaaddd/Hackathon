import numpy as np
import pandas as pd
import plotly.express as px

def plot_risk_heatmap(cal, X_train, sleep_col="Sleep_Hours_Per_Night", train_col="Training_Hours_Per_Week"):
    if sleep_col not in X_train.columns or train_col not in X_train.columns:
        return None

    s_min, s_max = np.percentile(X_train[sleep_col].dropna(), [5, 95])
    t_min, t_max = np.percentile(X_train[train_col].dropna(), [5, 95])
    s_vals = np.linspace(s_min, s_max, 25)
    t_vals = np.linspace(t_min, t_max, 25)

    grid = []
    num_defaults = X_train.select_dtypes(include=np.number).median()
    cat_df = X_train.select_dtypes(exclude=np.number)
    cat_defaults = cat_df.mode().iloc[0] if not cat_df.empty else pd.Series(dtype=object)

    for sv in s_vals:
        for tv in t_vals:
            row = {}
            for c in X_train.columns:
                if c == sleep_col: row[c] = sv
                elif c == train_col: row[c] = tv
                elif c in num_defaults.index: row[c] = num_defaults[c]
                elif c in cat_defaults.index: row[c] = cat_defaults[c]
                else: row[c] = None
            grid.append(row)

    G = pd.DataFrame(grid, columns=X_train.columns)
    risk = cal.predict_proba(G)[:, 1].reshape(len(s_vals), len(t_vals))
    risk_df = pd.DataFrame(risk, index=s_vals, columns=t_vals)

    fig = px.imshow(
        risk_df,
        x=risk_df.columns,
        y=risk_df.index,
        labels={"x": train_col, "y": sleep_col, "color": "Predicted risk"},
        title="Risk heatmap – Sleep vs Training",
        color_continuous_scale="RdBu_r"
    )
    return fig
