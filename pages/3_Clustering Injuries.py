import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from clustering_injury.cluster_core import (
    find_dataset, light_clean, encode_and_scale,
    kmeans_sweep, best_k, pca2
)

def render(dataset_path: str = "Injuries_prediction_model\player_injuries_impact.csv"):
    st.title("Clustering Notebook")
    st.caption("Auto-loads your local CSV. Computes under the hood and shows only the charts.")

    st.sidebar.header("Clustering")
    k_min = st.sidebar.number_input("k min", 2, 15, 2, 1)
    k_max = st.sidebar.number_input("k max", 3, 20, max(7, k_min+1), 1)
    random_state = st.sidebar.number_input("Random state", 0, 10000, 42, 1)
    k_values = list(range(int(k_min), int(k_max) + 1))

    # New: Option to choose number of clusters directly
    use_manual_k = st.sidebar.checkbox("Manually select number of clusters", value=False)
    manual_k = None
    if use_manual_k:
        manual_k = st.sidebar.number_input("Number of clusters (k)", int(k_min), int(k_max), int(k_min), 1)

    path = find_dataset(dataset_path)
    if path is None:
        st.error("Dataset not found. Place your CSV at 'injuries_prediction_model/player_injuries_impact.csv' (or set DATASET_PATH env var).")
        st.stop()

    df_raw = pd.read_csv(path)
    df = light_clean(df_raw)
    X, feature_names = encode_and_scale(df)
    if X.empty:
        st.error("No usable features after encoding. Please ensure numeric or low-cardinality categorical columns exist.")
        st.stop()

    inertias, silhouettes = kmeans_sweep(X, k_values, random_state=random_state)
    chosen_k = best_k(k_values, inertias, silhouettes)
    if chosen_k is None:
        st.error("Could not determine best k (silhouette unavailable). Try a different k range.")
        st.stop()

    # Use manual k if selected, else use chosen_k
    final_k = int(manual_k) if use_manual_k and manual_k is not None else int(chosen_k)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=final_k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    df_clusters = df.copy()
    df_clusters["cluster"] = labels

    c1, c2 = st.columns(2, vertical_alignment="top")
    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=k_values, y=inertias, mode="lines+markers", name="Inertia"))
        fig_elbow.update_layout(title="Elbow (Inertia vs k)", xaxis_title="k", yaxis_title="Inertia", height=360)
        st.plotly_chart(fig_elbow, use_container_width=True)
    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=k_values, y=silhouettes, mode="lines+markers", name="Silhouette"))
        fig_sil.add_vline(x=chosen_k, line_dash="dash", annotation_text=f"best k = {chosen_k}", annotation_position="top")
        st.plotly_chart(fig_sil, use_container_width=True)
        fig_sil.update_layout(title="Silhouette vs k", xaxis_title="k", yaxis_title="Silhouette score", height=360, yaxis_range=[-1,1])

    sizes = df_clusters["cluster"].value_counts().sort_index()
    fig_sizes = px.bar(sizes, title=f"Cluster Sizes (k = {final_k})", labels={"index":"cluster", "value":"count"})
    st.plotly_chart(fig_sizes, use_container_width=True)

    coords = pca2(X)
    pca_df = pd.DataFrame({"PCA1": coords[:,0], "PCA2": coords[:,1], "cluster": labels.astype(int)})
    fig_pca = px.scatter(pca_df, x="PCA1", y="PCA2", color=pca_df["cluster"].astype(str), title="PCA 2D Scatter by Cluster")
    st.plotly_chart(fig_pca, use_container_width=True)

    age_cols = [c for c in df.columns if c.lower() in ("age", "player_age")]
    rating_cols = [c for c in df.columns if ("fifa" in c.lower() and "rating" in c.lower()) or c.lower() in ("rating", "overall_rating")]
    if age_cols and rating_cols:
        a, r = age_cols[0], rating_cols[0]
        tmp = df_clusters.dropna(subset=[a, r]).copy()
        if not tmp.empty:
            fig_ar = px.scatter(tmp, x=a, y=r, color=tmp["cluster"].astype(str), title=f"{a} vs {r} by Cluster", hover_data=tmp.columns)
            st.plotly_chart(fig_ar, use_container_width=True)

    num_cols_present = df_clusters.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_present = [c for c in num_cols_present if c != "cluster"]
    if num_cols_present:
        prof = df_clusters.groupby("cluster")[num_cols_present].mean()
        if prof.shape[1] > 20:
            prof = prof.iloc[:, :20]
        fig_heat = px.imshow(prof.round(2), aspect="auto", title="Cluster Profiles (Numeric Means)")
        st.plotly_chart(fig_heat, use_container_width=True)

if __name__ == "__main__":
    render()