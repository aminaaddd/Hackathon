import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance

def plot_feature_importance(pipe, X_test, y_test, rng=42, top_n=15):
    perm = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=5, random_state=rng,
        scoring="roc_auc", n_jobs=1
    )

    fi = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    fi_top = fi.head(top_n).sort_values(ascending=True)

    fi_df = fi_top.reset_index()
    fi_df.columns = ["Feature", "Mean decrease in AUROC"]

    fig = px.bar(
        fi_df,
        x="Mean decrease in AUROC",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Features by Permutation Importance"
    )
    fig.update_layout(template="plotly_white")
    return fig, fi
