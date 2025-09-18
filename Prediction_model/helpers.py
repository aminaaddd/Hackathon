from __future__ import annotations
import pandas as pd
import difflib, re, unicodedata
from typing import Optional, Tuple

from Prediction_model.clean_data import clean_data, prepare_features

NAME_COLUMNS_TRY = ["player", "Player", "PLAYER", "name", "Name", "NAME"]
DATE_COLUMNS_TRY = ["date", "Date", "season", "Season", "match_date", "MATCH_DATE"]
NEGATIVE_QUERIES = {"no", "non", "nah", "nope", "cancel", "annuler", "nein"}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c))

def find_name_column(df: pd.DataFrame) -> Optional[str]:
    for c in NAME_COLUMNS_TRY:
        if c in df.columns:
            return c
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        for c in obj_cols:
            try:
                if df[c].nunique(dropna=True) > 0.2 * len(df):
                    return c
            except Exception:
                pass
        return obj_cols[0]
    return None

def get_player_names(df: pd.DataFrame) -> tuple[list[str], Optional[str]]:
    name_col = find_name_column(df)
    if not name_col:
        return [], None
    s = df[name_col].astype(str).dropna().str.strip().replace("", pd.NA).dropna()
    names = sorted(set(s.unique().tolist()))
    return names, name_col

def fuzzy_suggest(query: str, names: list[str], k: int = 8) -> list[str]:
    return difflib.get_close_matches(query, names, n=k, cutoff=0.6)

def _word_or_prefix_match(name: str, q: str) -> bool:
    n = _norm(name)
    q = _norm(q)
    if len(q) < 3 or q in NEGATIVE_QUERIES:
        return False
    if re.search(rf"\b{re.escape(q)}\b", n):  # whole word
        return True
    return any(tok.startswith(q) for tok in n.split())  # token prefix

def _pick_latest_row(df: pd.DataFrame) -> pd.Series:
    for d in DATE_COLUMNS_TRY:
        if d in df.columns:
            try:
                tmp = df.copy()
                tmp[d] = pd.to_datetime(tmp[d], errors="coerce")
                tmp = tmp.sort_values(d, ascending=False)
                return tmp.iloc[0]
            except Exception:
                continue
    return df.iloc[-1]

def lookup_player_row(raw_df: pd.DataFrame, name_query: str) -> Tuple[Optional[pd.Series], Optional[str], Optional[str]]:
    """
    Strict lookup:
    - ignore queries <3 chars or negatives
    - match whole word or token prefix
    - return a row only if exactly one unique player matches
    """
    name_col = find_name_column(raw_df)
    if not name_col:
        return None, None, None

    q = str(name_query).strip()
    qn = _norm(q)
    if len(qn) < 3 or qn in NEGATIVE_QUERIES:
        return None, name_col, None

    mask = raw_df[name_col].astype(str).apply(lambda v: _word_or_prefix_match(v, q))
    hits = raw_df[mask]
    if hits.empty:
        return None, name_col, None

    unique_players = hits[name_col].astype(str).unique().tolist()
    # prefer exact (normalized)
    exact = [orig for orig in unique_players if _norm(orig) == qn]
    if len(exact) == 1:
        player_name = exact[0]
    elif len(unique_players) == 1:
        player_name = unique_players[0]
    else:
        return None, name_col, None  # ask user to be specific

    row = _pick_latest_row(hits[hits[name_col].astype(str) == player_name])

    date_text = None
    for d in DATE_COLUMNS_TRY:
        if d in raw_df.columns:
            try:
                dt = pd.to_datetime(row[d], errors="coerce")
                if pd.notna(dt):
                    date_text = f"{d}={dt.date()}"
            except Exception:
                pass
            break
    return row, name_col, date_text

def prepare_single_row_features(row: pd.Series, full_df_for_context: pd.DataFrame) -> pd.DataFrame:
    cleaned_full = clean_data(full_df_for_context.copy())
    X_full, _ = prepare_features(cleaned_full)
    single_df = pd.DataFrame([row.to_dict()])
    cleaned_single = clean_data(single_df)
    X_single, _ = prepare_features(cleaned_single)
    return X_single.reindex(columns=X_full.columns, fill_value=0)

def build_template_row(raw_df: pd.DataFrame) -> pd.DataFrame:
    tpl = {}
    for col in raw_df.columns:
        s = raw_df[col]
        if pd.api.types.is_numeric_dtype(s):
            tpl[col] = s.median(skipna=True)
        elif pd.api.types.is_datetime64_any_dtype(s):
            v = pd.to_datetime(s, errors="coerce")
            tpl[col] = v.max() if v.notna().any() else pd.Timestamp.today().normalize()
        else:
            try:
                mode_vals = s.dropna().mode()
                tpl[col] = mode_vals.iloc[0] if not mode_vals.empty else ""
            except Exception:
                tpl[col] = ""
    name_col = find_name_column(raw_df)
    if name_col:
        tpl[name_col] = "New Player"
    return pd.DataFrame([tpl])
