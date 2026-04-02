import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Sijoitustyökalu v2", layout="wide")

st.title("Sijoitustyökalu v2")
st.caption("Parempi trendi- ja signaalityökalu")

ticker = st.sidebar.text_input("Ticker", "NDA-FI.HE")
period = st.sidebar.selectbox("Aikaväli", ["6mo", "1y", "2y", "5y"])

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

if df is None or df.empty:
    st.error(f"Dataa ei löytynyt tickerille: {ticker}")
    st.stop()
    return df

df = load_data(ticker, period)

if df.empty:
    st.error("Dataa ei löytynyt")
    st.stop()

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

latest = df.iloc[-1]

score = 0

if latest["Close"] > latest["MA50"]:
    score += 1
if latest["MA50"] > latest["MA200"]:
    score += 1
if latest["Close"] > latest["MA20"]:
    score += 1

if score >= 3:
    signal = "OSTA"
    color = "green"
elif score == 2:
    signal = "PIDÄ"
    color = "blue"
elif score == 1:
    signal = "VARO"
    color = "orange"
else:
    signal = "MYY"
    color = "red"

st.subheader(f"Signaali: {signal}")
st.write(f"Trendipisteet: {score}/3")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Kurssi"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200"))

st.plotly_chart(fig, use_container_width=True)
