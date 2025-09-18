import streamlit as st
import pandas as pd
from pathlib import Path

from Injuries_prediction_model.services.data_service import load_dataset
from Injuries_prediction_model.services.model_service import (
    load_or_train_model, prepare_single_row_features, predict_row
)
from Injuries_prediction_model.helpers import (
    get_player_names, fuzzy_suggest, lookup_player_row,
    find_name_column, build_template_row
)

st.set_page_config(page_title="Injury Risk Chatbot", layout="wide")
st.title("Injury Risk Chatbot")

ROOT = Path(__file__).parent.parent
PKG_DIR = ROOT / "Injuries_prediction_model"
DATA_PATH = PKG_DIR / "player_injuries_impact.csv"
MODEL_DIR = PKG_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


with st.sidebar:
    st.header("Data source")
    uploaded_csv = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    st.caption("If not provided, I use Injuries_prediction_model/player_injuries_impact.csv")

# ---------- Data / Model ----------
raw_df = load_dataset(uploaded_csv, DATA_PATH)
clf, train_columns = load_or_train_model(raw_df, MODEL_DIR)

player_names, detected_name_col = get_player_names(raw_df)

with st.sidebar:
    st.header("Find a player")
    if detected_name_col:
        sel = st.selectbox(
            f"Search in **{detected_name_col}**",
            options=["(type a name below)"] + player_names,
            index=0,
        )
        if sel != "(type a name below)":
            st.session_state["pending_lookup"] = sel
            st.success(f"Selected: {sel}")
        st.divider()
        st.caption(f"{len(player_names)} players in dataset")
        if st.button("Show first 50 players"):
            st.write(pd.DataFrame({detected_name_col: player_names[:50]}))
        st.download_button(
            "Download all players (CSV)",
            data=pd.DataFrame({detected_name_col: player_names}).to_csv(index=False).encode("utf-8"),
            file_name="players.csv",
            mime="text/csv",
            disabled=not player_names,
        )
    else:
        st.info("No name column detected.")

with st.expander("Dataset & Model Info", expanded=False):
    st.write(f"- Rows in dataset: **{len(raw_df):,}**")
    st.write(f"- Train columns: **{len(train_columns)}**")
    st.write(f"- Model type: **{type(clf).__name__}**")

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about a player and I will predict injury risk based on the latest data."}
    ]
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prefilled = st.session_state.pop("pending_lookup", None) if "pending_lookup" in st.session_state else None
user_input = st.chat_input("Type a player name")
if prefilled and not user_input:
    user_input = prefilled

# ---- Chat handling ----
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            text = user_input.strip()

            # Greetings
            if text.lower() in {"hi", "hello", "hey"}:
                msg = "Hi! Tell me a player's name and I'll predict their injury risk. You can also pick a name from the sidebar."
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.stop()

            # Quick intents
            if any(kw in text.lower() for kw in ["list players", "players", "who is in dataset", "show players", "names"]):
                if player_names:
                    sample = ", ".join(player_names[:20])
                    msg = (
                        f"There are **{len(player_names)}** players. "
                        f"Here are the first 20:\n\n{sample}\n\n"
                        "Use the sidebar to search or download the full list."
                    )
                else:
                    msg = "I couldn't detect a player name column. Try uploading a CSV with a clear player/name field."
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.stop()

            # returns None unless exactly one player matches
            row, name_col, date_text = lookup_player_row(raw_df, text)
            if row is None:
                suggestions = fuzzy_suggest(text, player_names) if player_names else []
                if suggestions:
                    msg = "I couldn't find that player, did you mean:\n\n" + ", ".join(f"**{s}**" for s in suggestions)
                else:
                    msg = (
                        "I couldn't find that player in the current dataset.\n\n"
                        "Option 1: Try another spelling, pick from the sidebar, or upload a CSV that includes this player.\n"
                        "Option 2: Use a 1-row template below to predict for any new player."
                    )
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

                # Template & single-row upload
                with st.expander("Predict for a new player (not in dataset)"):
                    template_df = build_template_row(raw_df)
                    nm_col = name_col or find_name_column(raw_df)
                    if nm_col:
                        template_df.loc[0, nm_col] = text
                    st.download_button(
                        "Download 1-row template (CSV)",
                        data=template_df.to_csv(index=False).encode("utf-8"),
                        file_name="player_template.csv",
                        mime="text/csv",
                    )
                    st.caption("Edit the CSV (fill what you know), then upload it below.")

                    one_row_file = st.file_uploader("Upload filled template (1 row CSV)", type=["csv"], key="one_row_upload")
                    if one_row_file is not None:
                        try:
                            one_df = pd.read_csv(one_row_file)
                            if len(one_df) != 1:
                                st.error("Please upload exactly one row.")
                            else:
                                X_single = prepare_single_row_features(one_df.iloc[0], raw_df)
                                X_single = X_single.reindex(columns=train_columns, fill_value=0)
                                y_pred, prob = predict_row(clf, X_single)

                                disp_name_col = find_name_column(one_df) or nm_col
                                player_display = str(one_df.iloc[0].get(disp_name_col, "the player")) if disp_name_col else "the player"

                                label = "injured" if int(y_pred) == 1 else "not injured"
                                prob_text = f" (probability ~ **{prob*100:.1f}%**)" if prob is not None else ""
                                answer_new = f"**{player_display}** is predicted to be **{label}**{prob_text}."
                                st.success(answer_new)


                                st.session_state.messages.append({"role": "assistant", "content": answer_new})
                        except Exception as e:
                            st.error(f"Could not predict from the uploaded row: {e}")
                st.stop()

            # Exactly one player matched 
            X_single = prepare_single_row_features(row, raw_df)
            X_single = X_single.reindex(columns=train_columns, fill_value=0)
            y_pred, prob = predict_row(clf, X_single)

            player_name = str(row[name_col]) if name_col else "the player"
            label = "injured" if int(y_pred) == 1 else "not injured"
            prob_text = f" (probability ~ **{prob*100:.1f}%**)" if prob is not None else ""
            date_note = f" — latest record {date_text}" if date_text else ""
            answer = f"**{player_name}** is predicted to be **{label}**{prob_text}{date_note}."
            st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            err = f"Sorry, something went wrong while predicting: `{e}`"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
