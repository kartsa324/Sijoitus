import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Sijoitustyökalu v1", layout="wide")

st.title("Sijoitustyökalu v1")
st.caption("Yksinkertainen trendi- ja signaalityökalu osakkeille ja ETF:ille")

DEFAULT_TICKERS = ["NDA-FI.HE", "SXR8.DE", "VWCE.DE", "SPY", "AAPL"]
PERIOD_OPTIONS = {
    "6 kuukautta": "6mo",
    "1 vuosi": "1y",
    "2 vuotta": "2y",
    "3 vuotta": "3y",
    "5 vuotta": "5y",
    "10 vuotta": "10y",
}


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    rename_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_map)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[needed].copy()
    df = df.dropna(subset=["Close"])
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    out["DailyReturn"] = out["Close"].pct_change()
    out["Ret_1m"] = out["Close"].pct_change(21)
    out["Ret_3m"] = out["Close"].pct_change(63)
    out["Ret_6m"] = out["Close"].pct_change(126)
    out["Ret_12m"] = out["Close"].pct_change(252)
    out["Vol_30d_annual"] = out["DailyReturn"].rolling(30).std() * np.sqrt(252)
    out["RollingHigh_60"] = out["Close"].rolling(60).max()
    out["DrawdownFrom60High"] = out["Close"] / out["RollingHigh_60"] - 1
    return out


def pct_str(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "–"
    return f"{value * 100:.1f} %"


def eur_str(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "–"
    return f"{value:,.2f}".replace(",", " ")


def classify_signal(row: pd.Series) -> tuple[str, str]:
    close = row.get("Close", np.nan)
    ma20 = row.get("MA20", np.nan)
    ma50 = row.get("MA50", np.nan)
    ma200 = row.get("MA200", np.nan)
    ret_1m = row.get("Ret_1m", np.nan)
    ret_3m = row.get("Ret_3m", np.nan)
    dd60 = row.get("DrawdownFrom60High", np.nan)

    if any(pd.isna(x) for x in [close, ma50, ma200]):
        return "Ei vielä tarpeeksi dataa", "Tarvitaan lisää historiaa, jotta signaali olisi mielekäs."

    bullish_trend = close > ma50 and ma50 > ma200
    weak_trend = close > ma200 and close < ma50
    negative_momentum = (not pd.isna(ret_1m) and ret_1m < 0) and (not pd.isna(ret_3m) and ret_3m < 0)
    strong_momentum = (not pd.isna(ret_1m) and ret_1m > 0) and (not pd.isna(ret_3m) and ret_3m > 0)
    pullback_ok = not pd.isna(dd60) and dd60 <= -0.03 and dd60 >= -0.10

    if bullish_trend and strong_momentum and close >= ma20:
        return "Pidä / trendi kunnossa", "Kurssi on nousurakenteessa ja momentum on positiivinen."

    if bullish_trend and strong_momentum and pullback_ok and close >= ma50:
        return "Ostaehdokas", "Nousutrendi on edelleen voimassa ja kurssi on vetäytynyt hallitusti."

    if weak_trend or (close > ma200 and negative_momentum):
        return "Varo", "Pitkä rakenne ei ole vielä täysin rikki, mutta lyhyempi kehitys on heikentynyt."

    if close < ma50 and negative_momentum:
        return "Myyntisignaali", "Kurssi on heikentynyt keskeisten trenditasojen suhteen ja momentum on negatiivinen."

    return "Neutraali", "Tilanne ei anna selvää osto- tai myyntietua tällä yksinkertaisella mallilla."


def trend_text(row: pd.Series) -> str:
    close = row.get("Close", np.nan)
    ma50 = row.get("MA50", np.nan)
    ma200 = row.get("MA200", np.nan)

    if any(pd.isna(x) for x in [close, ma50, ma200]):
        return "Ei riittävästi dataa"
    if close > ma50 > ma200:
        return "Nousutrendi"
    if close > ma200 and close < ma50:
        return "Heikkenevä trendi"
    if close < ma50 < ma200:
        return "Laskutrendi"
    return "Sivuttais-/epäselvä trendi"


with st.sidebar:
    st.header("Asetukset")
    ticker = st.text_input("Ticker", value="NDA-FI.HE").strip().upper()
    period_label = st.selectbox("Aikaväli", list(PERIOD_OPTIONS.keys()), index=2)
    show_volume = st.checkbox("Näytä volyymi", value=False)
    st.markdown("**Esimerkkejä**")
    st.write(", ".join(DEFAULT_TICKERS))

period = PERIOD_OPTIONS[period_label]

with st.spinner("Haetaan markkinadataa..."):
    raw = load_data(ticker, period)

if raw.empty:
    st.error("Dataa ei löytynyt tällä tickerillä. Tarkista tunnus ja yritä uudelleen.")
    st.stop()


df = add_indicators(raw)
latest = df.iloc[-1]
signal, explanation = classify_signal(latest)
trend = trend_text(latest)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Viimeisin kurssi", eur_str(float(latest["Close"])))
c2.metric("Trendi", trend)
c3.metric("Signaali", signal)
c4.metric("30 pv volatiliteetti", pct_str(float(latest["Vol_30d_annual"])) if pd.notna(latest["Vol_30d_annual"]) else None)

r1, r2, r3, r4 = st.columns(4)
r1.metric("1 kk tuotto", pct_str(float(latest["Ret_1m"])) if pd.notna(latest["Ret_1m"]) else None)
r2.metric("3 kk tuotto", pct_str(float(latest["Ret_3m"])) if pd.notna(latest["Ret_3m"]) else None)
r3.metric("6 kk tuotto", pct_str(float(latest["Ret_6m"])) if pd.notna(latest["Ret_6m"]) else None)
r4.metric("12 kk tuotto", pct_str(float(latest["Ret_12m"])) if pd.notna(latest["Ret_12m"]) else None)

st.info(explanation)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Kurssi"))
if df["MA20"].notna().any():
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
if df["MA50"].notna().any():
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
if df["MA200"].notna().any():
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))
fig.update_layout(height=520, title=f"{ticker} – kurssi ja liukuvat keskiarvot", xaxis_title="Päivä", yaxis_title="Hinta")
st.plotly_chart(fig, use_container_width=True)

if show_volume and "Volume" in df.columns:
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volyymi"))
    vol_fig.update_layout(height=250, title="Volyymi", xaxis_title="Päivä", yaxis_title="Volyymi")
    st.plotly_chart(vol_fig, use_container_width=True)

st.subheader("Yksinkertainen tulkinta")

summary = pd.DataFrame(
    {
        "Mittari": [
            "Kurssi vs MA20",
            "Kurssi vs MA50",
            "MA50 vs MA200",
            "1 kk tuotto",
            "3 kk tuotto",
            "Pudotus 60 pv huipusta",
        ],
        "Arvo": [
            "Yli" if pd.notna(latest["MA20"]) and latest["Close"] > latest["MA20"] else "Alle",
            "Yli" if pd.notna(latest["MA50"]) and latest["Close"] > latest["MA50"] else "Alle",
            "Yli" if pd.notna(latest["MA50"]) and pd.notna(latest["MA200"]) and latest["MA50"] > latest["MA200"] else "Alle",
            pct_str(float(latest["Ret_1m"])) if pd.notna(latest["Ret_1m"]) else "–",
            pct_str(float(latest["Ret_3m"])) if pd.notna(latest["Ret_3m"]) else "–",
            pct_str(float(latest["DrawdownFrom60High"])) if pd.notna(latest["DrawdownFrom60High"]) else "–",
        ],
    }
)
st.dataframe(summary, use_container_width=True, hide_index=True)

with st.expander("Mitä tämä versio tekee ja mitä ei"):
    st.markdown(
        """
        **Tämä versio tekee:**
        - hakee historiallista päivätason hintadataa
        - näyttää kurssin ja 20/50/200 päivän liukuvat keskiarvot
        - laskee perustuotot ja volatiliteetin
        - antaa yksinkertaisen trendi- ja signaalitulkinnan

        **Tämä versio ei vielä tee:**
        - varsinaista ennustemallia
        - backtestiä
        - automaattisia hälytyksiä
        - verojen tai kaupankäyntikulujen huomiointia
        """
    )

st.caption(f"Viimeisin datapiste: {df.index[-1].date() if hasattr(df.index[-1], 'date') else date.today()}")
