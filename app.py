import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Sijoitustyökalu v2", layout="wide")

st.title("Sijoitustyökalu v2")
st.caption("Parempi trendi- ja signaalityökalu osakkeille ja ETF:ille")

DEFAULT_TICKER = "NDA-FI.HE"
PERIOD_OPTIONS = {
    "6 kuukautta": "6mo",
    "1 vuosi": "1y",
    "2 vuotta": "2y",
    "5 vuotta": "5y",
}

ticker = st.sidebar.text_input("Ticker", DEFAULT_TICKER).strip().upper()
period_label = st.sidebar.selectbox("Aikaväli", list(PERIOD_OPTIONS.keys()), index=2)
period = PERIOD_OPTIONS[period_label]


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

    # Jos yfinance palauttaa MultiIndex-sarakkeet, litistetään ne
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(part) for part in col if str(part) != ""]).strip("_")
            for col in df.columns.to_list()
        ]

    # Etsitään päätöskurssisarakkeen nimi turvallisesti
    close_candidates = [c for c in df.columns if "Close" in str(c)]
    volume_candidates = [c for c in df.columns if "Volume" in str(c)]

    if not close_candidates:
        return pd.DataFrame()

    close_col = close_candidates[0]
    volume_col = volume_candidates[0] if volume_candidates else None

    out = pd.DataFrame(index=df.index)
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    if volume_col:
        out["Volume"] = pd.to_numeric(df[volume_col], errors="coerce")
    else:
        out["Volume"] = np.nan

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


df = load_data(ticker, period)

if df.empty:
    st.error(f"Dataa ei löytynyt tickerille: {ticker}")
    st.info("Kokeile esimerkiksi: NOKIA.HE, KNEBV.HE, SAMPO.HE, UPM.HE, SPY, AAPL")
    st.stop()

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
    box_type = "success"
    explanation = "Kurssi on tärkeiden keskiarvojen yläpuolella ja momentum on positiivinen."
elif score >= 3:
    action = "Pidä"
    trend_text = "Kohtalainen trendi"
    box_type = "info"
    explanation = "Rakenne on vielä kohtuullinen, mutta ei aivan vahvin mahdollinen."
elif score >= 2:
    action = "Varo"
    trend_text = "Heikkenevä trendi"
    box_type = "warning"
    explanation = "Lyhyempi kehitys on pehmennyt tai rakenne on osittain rikkoutunut."
else:
    action = "Myy"
    trend_text = "Laskutrendi"
    box_type = "error"
    explanation = "Kurssi- ja momentumrakenne on heikko suhteessa keskiarvoihin."

c1, c2, c3, c4 = st.columns(4)
c1.metric("Viimeisin kurssi", f"{close:.2f}" if pd.notna(close) else "-")
c2.metric("Trendipisteet", f"{score}/6")
c3.metric("Trendi", trend_text)
c4.metric("Toimintatulkinta", action)

c5, c6, c7, c8 = st.columns(4)
c5.metric("1 kk tuotto", f"{ret_1m * 100:.1f} %" if pd.notna(ret_1m) else "-")
c6.metric("3 kk tuotto", f"{ret_3m * 100:.1f} %" if pd.notna(ret_3m) else "-")
c7.metric("6 kk tuotto", f"{ret_6m * 100:.1f} %" if pd.notna(ret_6m) else "-")
c8.metric("12 kk tuotto", f"{ret_12m * 100:.1f} %" if pd.notna(ret_12m) else "-")

st.metric("30 pv volatiliteetti", f"{vol_30d * 100:.1f} %" if pd.notna(vol_30d) else "-")

if box_type == "success":
    st.success(explanation)
elif box_type == "info":
    st.info(explanation)
elif box_type == "warning":
    st.warning(explanation)
else:
    st.error(explanation)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Kurssi"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))

fig.update_layout(
    title=f"{ticker} – kurssi ja liukuvat keskiarvot",
    xaxis_title="Päivä",
    yaxis_title="Hinta",
    legend_title="Sarjat",
    height=600,
)

st.plotly_chart(fig, use_container_width=True)
