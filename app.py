
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Sijoitustyökalu v8", layout="wide")

st.title("Sijoitustyökalu v8")
st.caption("Watchlist + tallennus + muutokset viime tallennukseen + kurssi- ja tuottotiedot")

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
def get_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
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
    volume_candidates = [c for c in df.columns if "Volume" in str(c)]

    if not close_candidates:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["Close"] = pd.to_numeric(df[close_candidates[0]], errors="coerce")

    if volume_candidates:
        out["Volume"] = pd.to_numeric(df[volume_candidates[0]], errors="coerce")
    else:
        out["Volume"] = np.nan

    out = out.dropna(subset=["Close"]).copy()
    if out.empty:
        return pd.DataFrame()

    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    out["Return_1M"] = out["Close"].pct_change(21)
    out["Return_3M"] = out["Close"].pct_change(63)
    out["Return_12M"] = out["Close"].pct_change(252)
    out["Daily_Return"] = out["Close"].pct_change()
    out["Vol_30D"] = out["Daily_Return"].rolling(30).std() * np.sqrt(252)

    return out


def analyze(ticker: str) -> dict:
    df = get_data(ticker)
    if df.empty:
        return {
            "Ticker": ticker,
            "Kurssi": np.nan,
            "1 kk %": np.nan,
            "3 kk %": np.nan,
            "12 kk %": np.nan,
            "30 pv vol %": np.nan,
            "Signaali": "Ei dataa",
            "Pisteet": -1,
            "Ryhmä": "",
        }

    last = df.iloc[-1]

    close = float(last["Close"]) if pd.notna(last["Close"]) else np.nan
    ma50 = float(last["MA50"]) if pd.notna(last["MA50"]) else np.nan
    ma200 = float(last["MA200"]) if pd.notna(last["MA200"]) else np.nan
    ret_1m = float(last["Return_1M"]) * 100 if pd.notna(last["Return_1M"]) else np.nan
    ret_3m = float(last["Return_3M"]) * 100 if pd.notna(last["Return_3M"]) else np.nan
    ret_12m = float(last["Return_12M"]) * 100 if pd.notna(last["Return_12M"]) else np.nan
    vol_30d = float(last["Vol_30D"]) * 100 if pd.notna(last["Vol_30D"]) else np.nan

    score = 0
    if pd.notna(close) and pd.notna(ma50) and close > ma50:
        score += 1
    if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200:
        score += 1
    if pd.notna(ret_1m) and ret_1m > 0:
        score += 1
    if pd.notna(ret_3m) and ret_3m > 0:
        score += 1
    if pd.notna(ret_12m) and ret_12m > 0:
        score += 1

    if score >= 4:
        signal = "Osta"
    elif score >= 2:
        signal = "Varo"
    else:
        signal = "Myy"

    return {
        "Ticker": ticker,
        "Kurssi": close,
        "1 kk %": ret_1m,
        "3 kk %": ret_3m,
        "12 kk %": ret_12m,
        "30 pv vol %": vol_30d,
        "Signaali": signal,
        "Pisteet": score,
        "Ryhmä": "",
    }


def build_snapshot(rows: list[dict], owned_text: str, watch_text: str) -> dict:
    return {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "owned": owned_text,
        "watch": watch_text,
        "signals": {row["Ticker"]: row["Signaali"] for row in rows if row["Signaali"] != "Ei dataa"},
        "scores": {row["Ticker"]: row["Pisteet"] for row in rows if row["Signaali"] != "Ei dataa"},
    }


def detect_important_changes(previous_signals: dict, current_signals: dict):
    new_buys = []
    new_sells = []

    for ticker, new_signal in current_signals.items():
        old_signal = previous_signals.get(ticker)
        if not old_signal or old_signal == new_signal:
            continue

        if new_signal == "Osta" and old_signal in {"Varo", "Myy"}:
            new_buys.append((ticker, old_signal, new_signal))
        elif new_signal == "Myy" and old_signal in {"Osta", "Varo"}:
            new_sells.append((ticker, old_signal, new_signal))

    return new_buys, new_sells


st.sidebar.header("💾 Tallennus")

settings = {
    "owned": DEFAULT_OWNED,
    "watch": DEFAULT_WATCH,
}
loaded_snapshot = None

uploaded_file = st.sidebar.file_uploader("Tuo tallennetut listat", type="json")

if uploaded_file is not None:
    try:
        loaded = json.load(uploaded_file)
        settings["owned"] = loaded.get("owned", DEFAULT_OWNED)
        settings["watch"] = loaded.get("watch", DEFAULT_WATCH)
        loaded_snapshot = loaded
        st.sidebar.success("Listat ladattu")
    except Exception:
        st.sidebar.error("Tiedoston lukeminen ei onnistunut.")

owned_text = st.sidebar.text_area("Omat kohteet", settings["owned"], height=120)
watch_text = st.sidebar.text_area("Harkinnassa", settings["watch"], height=140)

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

snapshot = build_snapshot(rows, owned_text, watch_text)

st.sidebar.download_button(
    "💾 Tallenna listat koneelle",
    data=json.dumps(snapshot, ensure_ascii=False, indent=2),
    file_name="sijoituslistat.json",
    mime="application/json",
)

st.sidebar.caption("Tallenna tiedosto koneelle aina kun päivität listoja.")

if df.empty:
    st.info("Lisää tickereitä sivupalkkiin.")
else:
    previous_signals = loaded_snapshot.get("signals", {}) if loaded_snapshot else {}
    current_signals = snapshot["signals"]
    new_buys, new_sells = detect_important_changes(previous_signals, current_signals)

    if loaded_snapshot:
        st.subheader("🔔 Muutokset viime tallennukseen")
        saved_at = loaded_snapshot.get("saved_at")
        if saved_at:
            st.caption(f"Vertailu tallennukseen: {saved_at}")

        if not new_buys and not new_sells:
            st.info("Ei tärkeitä muutoksia verrattuna viime tallennettuun tiedostoon.")
        else:
            if new_buys:
                st.success(
                    "🟢 Uudet ostot\n\n" + "\n".join(
                        f"- {ticker}: {old} → {new}" for ticker, old, new in new_buys
                    )
                )
            if new_sells:
                st.error(
                    "🔴 Uudet myynnit\n\n" + "\n".join(
                        f"- {ticker}: {old} → {new}" for ticker, old, new in new_sells
                    )
                )

    signal_rank = {"Osta": 3, "Varo": 2, "Myy": 1, "Ei dataa": 0}
    df["Rank"] = df["Signaali"].map(signal_rank)
    df = df.sort_values(
        by=["Rank", "Pisteet", "12 kk %", "3 kk %", "Ticker"],
        ascending=[False, False, False, False, True]
    ).drop(columns=["Rank"])

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

    format_map = {
        "Kurssi": "{:.2f}",
        "1 kk %": "{:.1f}",
        "3 kk %": "{:.1f}",
        "12 kk %": "{:.1f}",
        "30 pv vol %": "{:.1f}",
    }

    st.subheader("Watchlist")

    tab1, tab2, tab3 = st.tabs(["Kaikki", "Omat", "Harkinnassa"])

    with tab1:
        st.dataframe(
            df.style.map(color_signal, subset=["Signaali"]).format(format_map, na_rep="-"),
            use_container_width=True,
        )

    with tab2:
        own_df = df[df["Ryhmä"] == "Omat"]
        st.dataframe(
            own_df.style.map(color_signal, subset=["Signaali"]).format(format_map, na_rep="-"),
            use_container_width=True,
        )

    with tab3:
        watch_df = df[df["Ryhmä"] == "Harkinnassa"]
        st.dataframe(
            watch_df.style.map(color_signal, subset=["Signaali"]).format(format_map, na_rep="-"),
            use_container_width=True,
        )
