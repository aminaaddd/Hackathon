import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Home page", layout="wide")
st.title("ML App")

st.markdown("""
Welcome!:
""")

ROOT = Path(__file__).parent
st.caption(f"Project root: `{ROOT}`")
