import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_CANDIDATES = [
    "injuries_prediction_model/player_injuries_impact.csv",
    "./injuries_prediction_model/player_injuries_impact.csv",
    "./data/player_injuries_impact.csv",
    "./dataset.csv",
    "./data/dataset.csv",
]

def safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def light_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    cols = df.columns.tolist()
    inj_cols = [c for c in cols if ("injur" in c.lower()) and ("date" in c.lower())]
    ret_cols = [c for c in cols if ("return" in c.lower()) and ("date" in c.lower())]
    if inj_cols and ret_cols:
        inj_dt = safe_to_datetime(df[inj_cols[0]])
        ret_dt = safe_to_datetime(df[ret_cols[0]])
        df["injury_duration_days"] = (ret_dt - inj_dt).dt.days
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def encode_and_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= 12]

    X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    X_cat = pd.get_dummies(df[low_card], drop_first=False, dtype=np.uint8) if low_card else pd.DataFrame(index=df.index)

    if not X_num.empty:
        X_num = X_num.fillna(X_num.mean(numeric_only=True))
    if not X_cat.empty:
        X_cat = X_cat.fillna(0)

    X = pd.concat([X_num, X_cat], axis=1)
    if X.empty:
        return X, []

    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X.values), index=X.index, columns=X.columns)
    return Xs, list(X.columns)

def kmeans_sweep(X: pd.DataFrame, k_values: List[int], random_state: int = 42):
    inertias, silhouettes = [], []
    for k in k_values:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            inertias.append(float(km.inertia_))
            if len(set(labels)) > 1 and X.shape[0] > k:
                silhouettes.append(float(silhouette_score(X, labels)))
            else:
                silhouettes.append(np.nan)
        except Exception:
            inertias.append(np.nan)
            silhouettes.append(np.nan)
    return inertias, silhouettes

def best_k(k_values: List[int], inertias: List[float], silhouettes: List[float]) -> Optional[int]:
    valid = [(k, sil, inh) for k, sil, inh in zip(k_values, silhouettes, inertias) if not np.isnan(sil)]
    if not valid:
        return None
    valid.sort(key=lambda t: (-t[1], t[2]))
    return valid[0][0]

def pca2(X: pd.DataFrame):
    if X.shape[1] < 2:
        pad = np.zeros((X.shape[0], 2 - X.shape[1]))
        Xp = np.hstack([X.values, pad])
        pca = PCA(n_components=2, random_state=42).fit(Xp)
        coords = pca.transform(Xp)
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.values)
    return coords

def find_dataset(dataset_path: Optional[str]) -> Optional[Path]:
    # 1) explicit path
    if dataset_path:
        p = Path(dataset_path)
        if p.exists():
            return p

    # 2) env var
    env_p = os.getenv("DATASET_PATH")
    if env_p and Path(env_p).exists():
        return Path(env_p)

    # 3) common candidates
    for c in DEFAULT_CANDIDATES:
        p = Path(c)
        if p.exists():
            return p

    return None