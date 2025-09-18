import pandas as pd
import plotly.express as px
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (holdout) – AUROC={auc(fpr, tpr):.3f}"
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(template="plotly_white")
    return fig
