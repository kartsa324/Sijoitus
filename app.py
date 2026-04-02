import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Sijoitustyökalu v4", layout="wide")

st.title("Sijoitustyökalu v4")
st.caption("Yksittäinen analyysi + erilliset listat omille ja harkinnassa oleville kohteille")

DEFAULT_TICKER = "NDA-FI.HE"
PERIOD_OPTIONS = {
    "6 kuukautta": "6mo",
    "1 vuosi": "1y",
    "2 vuotta": "2y",
    "5 vuotta": "5y",
}
DEFAULT_OWNED = "NDA-FI.HE\nKNEBV.HE\nSAMPO.HE"
DEFAULT_CANDIDATES = "NOKIA.HE\nUPM.HE\nSPY\nAAPL"


def normalize_ticker_list(raw: str) -> list[str]:
    items = [x.strip().upper() for x in raw.replace(";", ",").replace("\n", ",").split(",")]
    items = [x for x in items if x]
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


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

    close_col = close_candidates[0]
    volume_col = volume_candidates[0] if volume_candidates else None

    out = pd.DataFrame(index=df.index)
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    out["Volume"] = pd.to_numeric(df[volume_col], errors="coerce") if volume_col else np.nan
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


def analyze_from_df(df: pd.DataFrame) -> dict:
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

    return {
        "close": close,
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_12m": ret_12m,
        "vol_30d": vol_30d,
        "score": score,
        "action": action,
        "trend_text": trend_text,
        "box_type": box_type,
        "explanation": explanation,
    }


def build_watchlist_df(symbols: list[str], selected_period: str, category_label: str) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        df = load_data(symbol, selected_period)
        if df.empty:
            rows.append(
                {
                    "Lista": category_label,
                    "Ticker": symbol,
                    "Toimintatulkinta": "Ei dataa",
                    "Trendi": "-",
                    "Trendipisteet": -1,
                    "Kurssi": np.nan,
                    "1 kk %": np.nan,
                    "3 kk %": np.nan,
                    "12 kk %": np.nan,
                    "30 pv vol %": np.nan,
                }
            )
            continue

        r = analyze_from_df(df)
        rows.append(
            {
                "Lista": category_label,
                "Ticker": symbol,
                "Toimintatulkinta": r["action"],
                "Trendi": r["trend_text"],
                "Trendipisteet": r["score"],
                "Kurssi": round(r["close"], 2) if pd.notna(r["close"]) else np.nan,
                "1 kk %": round(r["ret_1m"] * 100, 1) if pd.notna(r["ret_1m"]) else np.nan,
                "3 kk %": round(r["ret_3m"] * 100, 1) if pd.notna(r["ret_3m"]) else np.nan,
                "12 kk %": round(r["ret_12m"] * 100, 1) if pd.notna(r["ret_12m"]) else np.nan,
                "30 pv vol %": round(r["vol_30d"] * 100, 1) if pd.notna(r["vol_30d"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sort_watchlist_df(df: pd.DataFrame) -> pd.DataFrame:
    signal_order = {"Osta": 0, "Pidä": 1, "Varo": 2, "Myy": 3, "Ei dataa": 4}
    temp = df.copy()
    temp["_signal_rank"] = temp["Toimintatulkinta"].map(signal_order).fillna(99)
    temp = temp.sort_values(
        ["_signal_rank", "Trendipisteet", "12 kk %"],
        ascending=[True, False, False],
    ).drop(columns=["_signal_rank"])
    return temp


def style_watchlist(df: pd.DataFrame):
    def color_signal(val):
        if val == "Osta":
            return "background-color: #d1fae5; color: #065f46; font-weight: bold;"
        if val == "Pidä":
            return "background-color: #dbeafe; color: #1e3a8a; font-weight: bold;"
        if val == "Varo":
            return "background-color: #fef3c7; color: #92400e; font-weight: bold;"
        if val == "Myy":
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
        return "background-color: #f3f4f6; color: #374151;"

    def color_score(val):
        try:
            if val >= 5:
                return "background-color: #d1fae5;"
            if val >= 3:
                return "background-color: #dbeafe;"
            if val >= 2:
                return "background-color: #fef3c7;"
            if val >= 0:
                return "background-color: #fee2e2;"
        except Exception:
            pass
        return ""

    def color_list(val):
        if val == "Omat":
            return "background-color: #ede9fe; color: #5b21b6; font-weight: bold;"
        if val == "Harkinnassa":
            return "background-color: #ecfeff; color: #155e75; font-weight: bold;"
        return ""

    return (
        df.style
        .map(color_list, subset=["Lista"])
        .map(color_signal, subset=["Toimintatulkinta"])
        .map(color_score, subset=["Trendipisteet"])
    )


def count_signals(df: pd.DataFrame, signal: str) -> int:
    return int((df["Toimintatulkinta"] == signal).sum()) if not df.empty else 0


st.sidebar.header("Asetukset")
ticker = st.sidebar.text_input("Yksittäinen ticker", DEFAULT_TICKER).strip().upper()
period_label = st.sidebar.selectbox("Aikaväli", list(PERIOD_OPTIONS.keys()), index=2)
period = PERIOD_OPTIONS[period_label]
show_volume = st.sidebar.checkbox("Näytä volyymi", value=False)

st.sidebar.header("Omat kohteet")
owned_raw = st.sidebar.text_area(
    "Syötä omat tickerit",
    value=DEFAULT_OWNED,
    height=110,
    help="Erottele pilkulla, puolipisteellä tai rivinvaihdolla.",
)
owned_list = normalize_ticker_list(owned_raw)

st.sidebar.header("Harkinnassa")
candidate_raw = st.sidebar.text_area(
    "Syötä harkinnassa olevat tickerit",
    value=DEFAULT_CANDIDATES,
    height=110,
    help="Erottele pilkulla, puolipisteellä tai rivinvaihdolla.",
)
candidate_list = normalize_ticker_list(candidate_raw)

st.subheader("Yksittäinen analyysi")
df = load_data(ticker, period)

if df.empty:
    st.error(f"Dataa ei löytynyt tickerille: {ticker}")
    st.info("Kokeile esimerkiksi: NOKIA.HE, KNEBV.HE, SAMPO.HE, UPM.HE, SPY, AAPL")
else:
    res = analyze_from_df(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Viimeisin kurssi", f"{res['close']:.2f}" if pd.notna(res["close"]) else "-")
    c2.metric("Trendipisteet", f"{res['score']}/6")
    c3.metric("Trendi", res["trend_text"])
    c4.metric("Toimintatulkinta", res["action"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("1 kk tuotto", f"{res['ret_1m'] * 100:.1f} %" if pd.notna(res["ret_1m"]) else "-")
    c6.metric("3 kk tuotto", f"{res['ret_3m'] * 100:.1f} %" if pd.notna(res["ret_3m"]) else "-")
    c7.metric("6 kk tuotto", f"{res['ret_6m'] * 100:.1f} %" if pd.notna(res["ret_6m"]) else "-")
    c8.metric("12 kk tuotto", f"{res['ret_12m'] * 100:.1f} %" if pd.notna(res["ret_12m"]) else "-")

    st.metric("30 pv volatiliteetti", f"{res['vol_30d'] * 100:.1f} %" if pd.notna(res["vol_30d"]) else "-")

    if res["box_type"] == "success":
        st.success(res["explanation"])
    elif res["box_type"] == "info":
        st.info(res["explanation"])
    elif res["box_type"] == "warning":
        st.warning(res["explanation"])
    else:
        st.error(res["explanation"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Kurssi"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))

    if show_volume and "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volyymi", opacity=0.25))

    fig.update_layout(
        title=f"{ticker} – kurssi ja liukuvat keskiarvot",
        xaxis_title="Päivä",
        yaxis_title="Hinta",
        legend_title="Sarjat",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Watchlist")
st.caption("Omat ja harkinnassa olevat kohteet erillään, väritettynä ja parhaasta heikoimpaan järjestettynä")

owned_df = sort_watchlist_df(build_watchlist_df(owned_list, period, "Omat")) if owned_list else pd.DataFrame()
candidate_df = sort_watchlist_df(build_watchlist_df(candidate_list, period, "Harkinnassa")) if candidate_list else pd.DataFrame()
combined_df = pd.concat([owned_df, candidate_df], ignore_index=True) if not owned_df.empty or not candidate_df.empty else pd.DataFrame()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Omat yhteensä", len(owned_df) if not owned_df.empty else 0)
m2.metric("Harkinnassa yhteensä", len(candidate_df) if not candidate_df.empty else 0)
m3.metric("Osta-signaalit", count_signals(combined_df, "Osta"))
m4.metric("Myy-signaalit", count_signals(combined_df, "Myy"))

all_tab, owned_tab, candidate_tab = st.tabs(["Kaikki", "Omat", "Harkinnassa"])

with all_tab:
    if combined_df.empty:
        st.info("Watchlist on tyhjä.")
    else:
        st.dataframe(style_watchlist(combined_df), use_container_width=True, hide_index=True)

with owned_tab:
    if owned_df.empty:
        st.info("Omat-lista on tyhjä.")
    else:
        st.dataframe(style_watchlist(owned_df), use_container_width=True, hide_index=True)

with candidate_tab:
    if candidate_df.empty:
        st.info("Harkinnassa-lista on tyhjä.")
    else:
        st.dataframe(style_watchlist(candidate_df), use_container_width=True, hide_index=True)
