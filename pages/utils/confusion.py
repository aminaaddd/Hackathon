import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    fig.update_layout(xaxis_title="Predicted label", yaxis_title="True label", template="plotly_white")
    return fig
