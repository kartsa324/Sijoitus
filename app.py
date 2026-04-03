import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Sijoitustyökalu v6", layout="wide")

st.title("Sijoitustyökalu v6")
st.caption("Watchlist + tallennus")

DEFAULT_OWNED = "NDA-FI.HE\nKNEBV.HE\nSAMPO.HE"
DEFAULT_WATCH = "NOKIA.HE\nUPM.HE\nSPY\nAAPL\nV3AA.DE"

# ------------------------
# LOAD / SAVE SETTINGS
# ------------------------

st.sidebar.header("💾 Tallennus")

settings = {
    "owned": DEFAULT_OWNED,
    "watch": DEFAULT_WATCH
}

uploaded_file = st.sidebar.file_uploader("Lataa tallennetut listat", type="json")

if uploaded_file:
    loaded = json.load(uploaded_file)
    settings = loaded
    st.sidebar.success("Listat ladattu")

# ------------------------
# INPUTS
# ------------------------

owned_text = st.sidebar.text_area("Omat kohteet", settings["owned"], height=120)
watch_text = st.sidebar.text_area("Harkinnassa", settings["watch"], height=140)

if st.sidebar.button("💾 Tallenna listat"):
    save_data = {
        "owned": owned_text,
        "watch": watch_text
    }
    st.sidebar.download_button(
        "⬇️ Lataa tiedosto",
        data=json.dumps(save_data, indent=2),
        file_name="sijoituslistat.json",
        mime="application/json"
    )

# ------------------------
# HELPERS
# ------------------------

def parse(text):
    return [x.strip().upper() for x in text.replace(",", "\n").split() if x.strip()]

def get_data(ticker):
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty:
        return None
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df

def analyze(ticker):
    df = get_data(ticker)
    if df is None:
        return ticker, "Ei dataa", 0

    last = df.iloc[-1]

    score = 0
    if last["Close"] > last["MA50"]: score += 1
    if last["MA50"] > last["MA200"]: score += 1

    if score == 2:
        signal = "Osta"
    elif score == 1:
        signal = "Varo"
    else:
        signal = "Myy"

    return ticker, signal, score

# ------------------------
# WATCHLIST
# ------------------------

owned = parse(owned_text)
watch = parse(watch_text)

rows = []

for t in owned:
    ticker, signal, score = analyze(t)
    rows.append({"Ticker": t, "Signaali": signal, "Pisteet": score, "Ryhmä": "Omat"})

for t in watch:
    if t not in owned:
        ticker, signal, score = analyze(t)
        rows.append({"Ticker": t, "Signaali": signal, "Pisteet": score, "Ryhmä": "Harkinnassa"})

df = pd.DataFrame(rows)

if not df.empty:
    df = df.sort_values(by=["Pisteet"], ascending=False)

    def color(val):
        if val == "Osta": return "background-color: #d1fae5"
        if val == "Varo": return "background-color: #fef3c7"
        if val == "Myy": return "background-color: #fee2e2"
        return ""

    st.subheader("Watchlist")

    st.dataframe(
        df.style.map(color, subset=["Signaali"]),
        use_container_width=True
    )
