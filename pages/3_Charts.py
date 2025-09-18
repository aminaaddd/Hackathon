import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import streamlit as st
from pages.utils.roc import plot_roc
from pages.utils.pr import plot_pr
from pages.utils.confusion import plot_confusion
from pages.utils.feature_importance import plot_feature_importance
from pages.utils.heatmap import plot_risk_heatmap
from pages.utils.curves1d import plot_1d_curves
import plotly.express as px
from visualizations.script.helpers import prepare_xy, split_train_test, make_pipeline, fit, predict_proba, crossval, best_f1_threshold, compute_metrics

RNG = 42
DATA_PATH = Path("data.csv")

# === ROC curve data ===
df = pd.read_csv(DATA_PATH)
X, y, num_cols, cat_cols = prepare_xy(df)
X_train, X_test, y_train, y_test = split_train_test(X, y, rng=RNG)
pipe = make_pipeline(num_cols, cat_cols)
    
pipe = fit(pipe, X_train, y_train)
proba_test = predict_proba(pipe, X_test)

t_best_f1 = best_f1_threshold(y_test, proba_test)
metrics = compute_metrics(y_test, proba_test, t_best_f1)

# CV scores
cv_scores, cv = crossval(pipe, X, y)

st.plotly_chart(plot_roc(y_test, proba_test), key="roc")

with st.expander("Cross-validation results"):
    st.write(f"CV AUROC: **{cv_scores['cv_auroc']:.3f}**")
    st.write(f"CV PR AUC: **{cv_scores['cv_pr_auc']:.3f}**")


# Best threshold & metrics
thr = best_f1_threshold(y_test, proba_test)
metrics = compute_metrics(y_test, proba_test, thr)
y_hat = (proba_test >= t_best_f1).astype(int)

cal = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv=3)
cal.fit(X_train, y_train)
proba_cal = cal.predict_proba(X_test)[:, 1]

with st.expander("Best threshold and metrics"):
    st.write(f"Best-F1 Threshold: **{metrics['threshold']:.3f}**")
    st.write("Confusion matrix:")
    st.write(metrics["confusion_matrix"])
    st.write(f"Holdout AUROC: **{metrics['holdout_auroc']:.3f}**")
    st.write(f"Holdout PR AUC: **{metrics['holdout_pr_auc']:.3f}**")
    
with st.expander("Classification report"):
    st.text(metrics["report"])


st.plotly_chart(plot_pr(y_test, proba_test), key="pr")
st.plotly_chart(plot_confusion(y_test, y_hat), key="confusion")

fig_fi, fi = plot_feature_importance(pipe, X_test, y_test)
st.plotly_chart(fig_fi, use_container_width=True, key="fi")

heat = plot_risk_heatmap(cal, X_train)
if heat:
    st.plotly_chart(heat, use_container_width=True, key="heatmap")

for i, fig in enumerate(plot_1d_curves(cal, X_train)):
    st.plotly_chart(fig, use_container_width=True, key=f"curve_{i}")
