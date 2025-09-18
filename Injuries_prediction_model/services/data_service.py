from __future__ import annotations
import pandas as pd
from pathlib import Path
import streamlit as st

@st.cache_data
def load_dataset(uploaded, default_path: Path) -> pd.DataFrame:
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv(default_path)
