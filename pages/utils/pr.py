import pandas as pd
import plotly.express as px
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr(y_true, proba):
    P, R, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    pr_df = pd.DataFrame({"Recall": R, "Precision": P})

    fig = px.line(
        pr_df,
        x="Recall",
        y="Precision",
        title=f"Precision–Recall Curve (holdout) – AP={ap:.3f}"
    )
    fig.update_layout(template="plotly_white")
    return fig
