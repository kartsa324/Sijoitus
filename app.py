
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Sijoitustyökalu v5", layout="wide")

STATE_FILE = "signal_state.json"

st.title("Sijoitustyökalu v5")
st.caption("Watchlist + omat/harkinnassa + tärkeät muutokset")

DEFAULT_OWNED = "NDA-FI.HE\nKNEBV.HE\nSAMPO.HE"
DEFAULT_WATCH = "NOKIA.HE\nUPM.HE\nSPY\nAAPL\nV3AA.DE"

PERIOD_OPTIONS = {
    "6 kuukautta": "6mo",
    "1 vuosi": "1y",
    "2 vuotta": "2y",
    "5 vuotta": "5y",
}

SIGNAL_ORDER = {"Myy": 0, "Varo": 1, "Pidä": 2, "Osta": 3}


def parse_tickers(text: str) -> list[str]:
    if not text:
        return []
    raw = text.replace(",", "\n").splitlines()
    tickers = []
    seen = set()
    for item in raw:
        t = item.strip().upper()
        if t and t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers


@st.cache_data(ttl=3600)
def load_data(symbol: str, selected_period: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=selected_period,
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
    out["Volume"] = (
        pd.to_numeric(df[volume_candidates[0]], errors="coerce")
        if volume_candidates else np.nan
    )

    out = out.dropna(subset=["Close"]).copy()
    if out.empty:
        return pd.DataFrame()

    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    out["Return_1M"] = out["Close"].pct_change(21)
    out["Return_3M"] = out["Close"].pct_change(63)
    out["Return_6M"] = out["Close"].pct_change(126)
    out["Return_12M"] = out["Close"].pct_change(252)
    out["Daily_Return"] = out["Close"].pct_change()
    out["Vol_30D"] = out["Daily_Return"].rolling(30).std() * np.sqrt(252)

    return out


def analyze_symbol(ticker: str, period: str) -> dict:
    df = load_data(ticker, period)
    if df.empty:
        return {
            "Ticker": ticker,
            "Kurssi": np.nan,
            "1 kk %": np.nan,
            "3 kk %": np.nan,
            "6 kk %": np.nan,
            "12 kk %": np.nan,
            "Vol 30 pv %": np.nan,
            "Pisteet": -1,
            "Trendi": "Ei dataa",
            "Signaali": "Ei dataa",
            "Selite": "Dataa ei löytynyt.",
            "df": pd.DataFrame(),
        }

    latest = df.iloc[-1]

    close = float(latest["Close"]) if pd.notna(latest["Close"]) else np.nan
    ma20 = float(latest["MA20"]) if pd.notna(latest["MA20"]) else np.nan
    ma50 = float(latest["MA50"]) if pd.notna(latest["MA50"]) else np.nan
    ma200 = float(latest["MA200"]) if pd.notna(latest["MA200"]) else np.nan

    ret_1m = float(latest["Return_1M"]) if pd.notna(latest["Return_1M"]) else np.nan
    ret_3m = float(latest["Return_3M"]) if pd.notna(latest["Return_3M"]) else np.nan
    ret_6m = float(latest["Return_6M"]) if pd.notna(latest["Return_6M"]) else np.nan
    ret_12m = float(latest["Return_12M"]) if pd.notna(latest["Return_12M"]) else np.nan
    vol_30d = float(latest["Vol_30D"]) if pd.notna(latest["Vol_30D"]) else np.nan

    score = 0
    if pd.notna(close) and pd.notna(ma20) and close > ma20:
        score += 1
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

    if score >= 5:
        action = "Osta"
        trend_text = "Vahva nousutrendi"
        explanation = "Kurssi on tärkeiden tasojen yläpuolella ja momentum on positiivinen."
    elif score >= 3:
        action = "Pidä"
        trend_text = "Kohtalainen trendi"
        explanation = "Rakenne on vielä kohtuullinen, mutta ei aivan vahvin."
    elif score >= 2:
        action = "Varo"
        trend_text = "Heikkenevä trendi"
        explanation = "Lyhyempi kehitys on pehmennyt tai rakenne on osittain rikkoutunut."
    else:
        action = "Myy"
        trend_text = "Laskutrendi"
        explanation = "Kurssi- ja momentumrakenne on heikko."

    return {
        "Ticker": ticker,
        "Kurssi": close,
        "1 kk %": ret_1m * 100 if pd.notna(ret_1m) else np.nan,
        "3 kk %": ret_3m * 100 if pd.notna(ret_3m) else np.nan,
        "6 kk %": ret_6m * 100 if pd.notna(ret_6m) else np.nan,
        "12 kk %": ret_12m * 100 if pd.notna(ret_12m) else np.nan,
        "Vol 30 pv %": vol_30d * 100 if pd.notna(vol_30d) else np.nan,
        "Pisteet": score,
        "Trendi": trend_text,
        "Signaali": action,
        "Selite": explanation,
        "df": df,
    }


def signal_to_rank(signal: str) -> int:
    return SIGNAL_ORDER.get(signal, -1)


def load_previous_state() -> dict:
    path = Path(STATE_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    Path(STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def format_change(old: str, new: str) -> bool:
    important_buy = new == "Osta" and old in {"Pidä", "Varo"}
    important_sell = new == "Myy" and old in {"Osta", "Pidä", "Varo"}
    return important_buy or important_sell


st.sidebar.header("Yksittäinen analyysi")
single_ticker = st.sidebar.text_input("Ticker", "NDA-FI.HE").strip().upper()
period_label = st.sidebar.selectbox("Aikaväli", list(PERIOD_OPTIONS.keys()), index=2)
period = PERIOD_OPTIONS[period_label]

st.sidebar.header("Watchlist")
owned_text = st.sidebar.text_area("Omat kohteet", DEFAULT_OWNED, height=120)
watch_text = st.sidebar.text_area("Harkinnassa", DEFAULT_WATCH, height=140)

owned = parse_tickers(owned_text)
watch = parse_tickers(watch_text)

single = analyze_symbol(single_ticker, period)

if single["Signaali"] == "Ei dataa":
    st.error(f"Dataa ei löytynyt tickerille: {single_ticker}")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Viimeisin kurssi", f'{single["Kurssi"]:.2f}')
    c2.metric("Trendipisteet", f'{single["Pisteet"]}/6')
    c3.metric("Trendi", single["Trendi"])
    c4.metric("Toimintatulkinta", single["Signaali"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("1 kk tuotto", f'{single["1 kk %"]:.1f} %' if pd.notna(single["1 kk %"]) else "-")
    c6.metric("3 kk tuotto", f'{single["3 kk %"]:.1f} %' if pd.notna(single["3 kk %"]) else "-")
    c7.metric("6 kk tuotto", f'{single["6 kk %"]:.1f} %' if pd.notna(single["6 kk %"]) else "-")
    c8.metric("12 kk tuotto", f'{single["12 kk %"]:.1f} %' if pd.notna(single["12 kk %"]) else "-")

    st.metric("30 pv volatiliteetti", f'{single["Vol 30 pv %"]:.1f} %' if pd.notna(single["Vol 30 pv %"]) else "-")

    if single["Signaali"] == "Osta":
        st.success(single["Selite"])
    elif single["Signaali"] == "Pidä":
        st.info(single["Selite"])
    elif single["Signaali"] == "Varo":
        st.warning(single["Selite"])
    else:
        st.error(single["Selite"])

    fig = go.Figure()
    df_single = single["df"]
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single["Close"], mode="lines", name="Kurssi"))
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df_single.index, y=df_single["MA200"], mode="lines", name="MA200"))
    fig.update_layout(
        title=f"{single_ticker} – kurssi ja liukuvat keskiarvot",
        xaxis_title="Päivä",
        yaxis_title="Hinta",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

combined = [(t, "Omat") for t in owned] + [(t, "Harkinnassa") for t in watch if t not in owned]

rows = []
for ticker, group in combined:
    result = analyze_symbol(ticker, period)
    result["Ryhmä"] = group
    rows.append(result)

watch_df = pd.DataFrame([{k: v for k, v in row.items() if k != "df"} for row in rows])

if not watch_df.empty:
    previous_state = load_previous_state()
    current_state = {
        row["Ticker"]: row["Signaali"]
        for _, row in watch_df.iterrows()
        if row["Signaali"] != "Ei dataa"
    }

    important_changes = []
    new_buys = []
    new_sells = []

    for ticker, new_signal in current_state.items():
        old_signal = previous_state.get(ticker)
        if old_signal and format_change(old_signal, new_signal):
            important_changes.append(f"{ticker}: {old_signal} → {new_signal}")
            if new_signal == "Osta":
                new_buys.append(f"{ticker}: {old_signal} → {new_signal}")
            elif new_signal == "Myy":
                new_sells.append(f"{ticker}: {old_signal} → {new_signal}")

    if important_changes:
        st.subheader("🔔 Tärkeät muutokset")
        if new_buys:
            st.success("🟢 Uudet ostot\n\n" + "\n".join(f"- {x}" for x in new_buys))
        if new_sells:
            st.error("🔴 Uudet myynnit\n\n" + "\n".join(f"- {x}" for x in new_sells))
    else:
        st.caption("Ei uusia tärkeitä signaalimuutoksia edelliseen tallennettuun tilaan verrattuna.")

    save_state(current_state)

    show_df = watch_df.copy()
    show_df["Rank"] = show_df["Signaali"].map(signal_to_rank)
    show_df = show_df.sort_values(by=["Rank", "Pisteet", "12 kk %"], ascending=[False, False, False]).drop(columns=["Rank", "Selite"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Omia kohteita", int((show_df["Ryhmä"] == "Omat").sum()))
    c2.metric("Harkinnassa", int((show_df["Ryhmä"] == "Harkinnassa").sum()))
    c3.metric("Osta-signaaleja", int((show_df["Signaali"] == "Osta").sum()))
    c4.metric("Myy-signaaleja", int((show_df["Signaali"] == "Myy").sum()))

    def color_signal(val):
        if val == "Osta":
            return "background-color: #d1fae5; color: #065f46;"
        if val == "Pidä":
            return "background-color: #dbeafe; color: #1e3a8a;"
        if val == "Varo":
            return "background-color: #fef3c7; color: #92400e;"
        if val == "Myy":
            return "background-color: #fee2e2; color: #991b1b;"
        return ""

    st.subheader("Watchlist")
    tab1, tab2, tab3 = st.tabs(["Kaikki", "Omat", "Harkinnassa"])

    with tab1:
        st.dataframe(
            show_df.style.map(color_signal, subset=["Signaali"]).format({
                "Kurssi": "{:.2f}",
                "1 kk %": "{:.1f}",
                "3 kk %": "{:.1f}",
                "6 kk %": "{:.1f}",
                "12 kk %": "{:.1f}",
                "Vol 30 pv %": "{:.1f}",
            }),
            use_container_width=True
        )
    with tab2:
        own_df = show_df[show_df["Ryhmä"] == "Omat"]
        st.dataframe(
            own_df.style.map(color_signal, subset=["Signaali"]).format({
                "Kurssi": "{:.2f}",
                "1 kk %": "{:.1f}",
                "3 kk %": "{:.1f}",
                "6 kk %": "{:.1f}",
                "12 kk %": "{:.1f}",
                "Vol 30 pv %": "{:.1f}",
            }),
            use_container_width=True
        )
    with tab3:
        watch_only_df = show_df[show_df["Ryhmä"] == "Harkinnassa"]
        st.dataframe(
            watch_only_df.style.map(color_signal, subset=["Signaali"]).format({
                "Kurssi": "{:.2f}",
                "1 kk %": "{:.1f}",
                "3 kk %": "{:.1f}",
                "6 kk %": "{:.1f}",
                "12 kk %": "{:.1f}",
                "Vol 30 pv %": "{:.1f}",
            }),
            use_container_width=True
        )
