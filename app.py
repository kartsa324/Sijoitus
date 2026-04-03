import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Sijoitustyökalu v6", layout="wide")

st.title("Sijoitustyökalu v6")
st.caption("Watchlist + tallennus")

DEFAULT_OWNED = "NDA-FI.HE\nKNEBV.HE\nSAMPO.HE"
DEFAULT_WATCH = "NOKIA.HE\nUPM.HE\nSPY\nAAPL\nV3AA.DE"


def parse_tickers(text: str) -> list[str]:
    if not text:
        return []
    raw = text.replace(",", "\n").splitlines()
    out = []
    seen = set()
    for item in raw:
        t = item.strip().upper()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


@st.cache_data(ttl=3600)
def get_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="1y",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(part) for part in col if str(part) != ""]).strip("_")
            for col in df.columns.to_list()
        ]

    close_candidates = [c for c in df.columns if "Close" in str(c)]
    if not close_candidates:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["Close"] = pd.to_numeric(df[close_candidates[0]], errors="coerce")
    out = out.dropna(subset=["Close"]).copy()

    if out.empty:
        return pd.DataFrame()

    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    return out


def analyze(ticker: str):
    df = get_data(ticker)
    if df.empty:
        return {
            "Ticker": ticker,
            "Signaali": "Ei dataa",
            "Pisteet": -1,
            "Ryhmä": "",
        }

    last = df.iloc[-1]

    close = float(last["Close"]) if pd.notna(last["Close"]) else np.nan
    ma50 = float(last["MA50"]) if pd.notna(last["MA50"]) else np.nan
    ma200 = float(last["MA200"]) if pd.notna(last["MA200"]) else np.nan

    score = 0
    if pd.notna(close) and pd.notna(ma50) and close > ma50:
        score += 1
    if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200:
        score += 1

    if score == 2:
        signal = "Osta"
    elif score == 1:
        signal = "Varo"
    else:
        signal = "Myy"

    return {
        "Ticker": ticker,
        "Signaali": signal,
        "Pisteet": score,
        "Ryhmä": "",
    }


st.sidebar.header("💾 Tallennus")

settings = {
    "owned": DEFAULT_OWNED,
    "watch": DEFAULT_WATCH,
}

uploaded_file = st.sidebar.file_uploader("Lataa tallennetut listat", type="json")

if uploaded_file is not None:
    try:
        loaded = json.load(uploaded_file)
        settings["owned"] = loaded.get("owned", DEFAULT_OWNED)
        settings["watch"] = loaded.get("watch", DEFAULT_WATCH)
        st.sidebar.success("Listat ladattu")
    except Exception:
        st.sidebar.error("Tiedoston lukeminen ei onnistunut.")

owned_text = st.sidebar.text_area("Omat kohteet", settings["owned"], height=120)
watch_text = st.sidebar.text_area("Harkinnassa", settings["watch"], height=140)

save_data = {
    "owned": owned_text,
    "watch": watch_text,
}

st.sidebar.download_button(
    "💾 Lataa listatiedosto",
    data=json.dumps(save_data, ensure_ascii=False, indent=2),
    file_name="sijoituslistat.json",
    mime="application/json",
)

owned = parse_tickers(owned_text)
watch = parse_tickers(watch_text)

rows = []

for t in owned:
    row = analyze(t)
    row["Ryhmä"] = "Omat"
    rows.append(row)

for t in watch:
    if t not in owned:
        row = analyze(t)
        row["Ryhmä"] = "Harkinnassa"
        rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    st.info("Lisää tickereitä sivupalkkiin.")
else:
    signal_rank = {"Osta": 3, "Varo": 2, "Myy": 1, "Ei dataa": 0}
    df["Rank"] = df["Signaali"].map(signal_rank)
    df = df.sort_values(by=["Rank", "Pisteet", "Ticker"], ascending=[False, False, True]).drop(columns=["Rank"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Omia kohteita", int((df["Ryhmä"] == "Omat").sum()))
    c2.metric("Harkinnassa", int((df["Ryhmä"] == "Harkinnassa").sum()))
    c3.metric("Osta-signaaleja", int((df["Signaali"] == "Osta").sum()))
    c4.metric("Myy-signaaleja", int((df["Signaali"] == "Myy").sum()))

    def color_signal(val):
        if val == "Osta":
            return "background-color: #d1fae5; color: #065f46;"
        if val == "Varo":
            return "background-color: #fef3c7; color: #92400e;"
        if val == "Myy":
            return "background-color: #fee2e2; color: #991b1b;"
        if val == "Ei dataa":
            return "background-color: #f3f4f6; color: #6b7280;"
        return ""

    st.subheader("Watchlist")

    tab1, tab2, tab3 = st.tabs(["Kaikki", "Omat", "Harkinnassa"])

    with tab1:
        st.dataframe(
            df.style.map(color_signal, subset=["Signaali"]),
            use_container_width=True,
        )

    with tab2:
        own_df = df[df["Ryhmä"] == "Omat"]
        st.dataframe(
            own_df.style.map(color_signal, subset=["Signaali"]),
            use_container_width=True,
        )

    with tab3:
        watch_df = df[df["Ryhmä"] == "Harkinnassa"]
        st.dataframe(
            watch_df.style.map(color_signal, subset=["Signaali"]),
            use_container_width=True,
        )
