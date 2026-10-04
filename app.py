import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# SETTINGS
# =========================================================

st.set_page_config(
    page_title="Live Trading Signal Bot",
    page_icon="📊",
    layout="centered"
)

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")

PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD"
]

TIMEFRAMES = {
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min"
}


# =========================================================
# GET MARKET DATA
# =========================================================

def get_market_data(symbol, interval):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "apikey": API_KEY
    }

    response = requests.get(url, params=params, timeout=15)
    data = response.json()

    if "values" not in data:
        return None, data.get("message", "Market data error")

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(df["datetime"])

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    return df, None


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(df):

    last = df.iloc[-1]
    previous = df.iloc[-2]

    up_score = 0
    down_score = 0

    reasons_up = []
    reasons_down = []

    # EMA trend
    if last["ema9"] > last["ema21"]:
        up_score += 2
        reasons_up.append("EMA9 above EMA21")
    else:
        down_score += 2
        reasons_down.append("EMA9 below EMA21")

    if last["ema21"] > last["ema50"]:
        up_score += 2
        reasons_up.append("EMA21 above EMA50")
    else:
        down_score += 2
        reasons_down.append("EMA21 below EMA50")

    # RSI
    if 50 < last["rsi"] < 70:
        up_score += 2
        reasons_up.append("RSI bullish")
    elif 30 < last["rsi"] < 50:
        down_score += 2
        reasons_down.append("RSI bearish")
    elif last["rsi"] >= 70:
        down_score += 1
        reasons_down.append("RSI overbought")
    elif last["rsi"] <= 30:
        up_score += 1
        reasons_up.append("RSI oversold")

    # MACD
    if last["macd"] > last["macd_signal"]:
        up_score += 2
        reasons_up.append("MACD bullish")
    else:
        down_score += 2
        reasons_down.append("MACD bearish")

    # Price momentum
    if last["close"] > previous["close"]:
        up_score += 1
        reasons_up.append("Price momentum UP")
    else:
        down_score += 1
        reasons_down.append("Price momentum DOWN")

    # Candle direction
    if last["close"] > last["open"]:
        up_score += 1
        reasons_up.append("Bullish candle")
    else:
        down_score += 1
        reasons_down.append("Bearish candle")

    # Recent structure
    recent_high = df["high"].iloc[-6:-1].max()
    recent_low = df["low"].iloc[-6:-1].min()

    if last["close"] > recent_high:
        up_score += 2
        reasons_up.append("Recent high breakout")

    if last["close"] < recent_low:
        down_score += 2
        reasons_down.append("Recent low breakdown")

    total = up_score + down_score

    if up_score >= down_score:
        signal = "UP"
        score = round((up_score / total) * 100, 1)
        reasons = reasons_up
    else:
        signal = "DOWN"
        score = round((down_score / total) * 100, 1)
        reasons = reasons_down

    return signal, score, up_score, down_score, reasons


# =========================================================
# BACKTEST
# =========================================================

def backtest(df):

    wins = 0
    losses = 0

    results = []

    start_index = 60

    for i in range(start_index, len(df) - 1):

        test_df = df.iloc[:i + 1].copy()

        signal, score, _, _, _ = generate_signal(test_df)

        current_close = df.iloc[i]["close"]
        next_close = df.iloc[i + 1]["close"]

        if signal == "UP":
            win = next_close > current_close
        else:
            win = next_close < current_close

        if win:
            wins += 1
            result = "WIN"
        else:
            losses += 1
            result = "LOSS"

        results.append({
            "time": df.iloc[i]["datetime"],
            "signal": signal,
            "score": score,
            "result": result
        })

    total = wins + losses

    accuracy = (wins / total * 100) if total > 0 else 0

    return accuracy, wins, losses, pd.DataFrame(results)


# =========================================================
# UI
# =========================================================

st.title("📊 Live Trading Signal Bot")

st.write(
    "Real-time market analysis with historical accuracy testing."
)

if not API_KEY:
    st.error("Twelve Data API key is missing.")
    st.stop()

pair = st.selectbox(
    "Select Pair",
    PAIRS
)

timeframe_name = st.selectbox(
    "Select Timeframe",
    list(TIMEFRAMES.keys())
)

interval = TIMEFRAMES[timeframe_name]

if st.button("🚀 START ANALYZE", use_container_width=True):

    with st.spinner("Analyzing live market..."):

        df, error = get_market_data(pair, interval)

        if error:
            st.error(error)
            st.stop()

        df = calculate_indicators(df)

        signal, score, up_score, down_score, reasons = generate_signal(df)

        accuracy, wins, losses, history = backtest(df)

        current_price = df.iloc[-1]["close"]

    st.success("Market analysis completed.")

    st.metric(
        "CURRENT PRICE",
        f"{current_price:.6f}"
    )

    if signal == "UP":
        st.success("🟢 SIGNAL: UP")
    else:
        st.error("🔴 SIGNAL: DOWN")

    st.metric(
        "SIGNAL STRENGTH",
        f"{score}%"
    )

    st.write("### 📊 Signal Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "UP SCORE",
            up_score
        )

    with col2:
        st.metric(
            "DOWN SCORE",
            down_score
        )

    st.write("### 🧠 Reasons")

    for reason in reasons:
        st.write("•", reason)

    st.divider()

    st.write("### 🧪 Historical Backtest")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )

    with col2:
        st.metric(
            "WIN",
            wins
        )

    with col3:
        st.metric(
            "LOSS",
            losses
        )

    st.caption(
        f"Backtest based on {len(history)} historical signals."
    )

    st.write("### 🕒 Signal Time")

    st.write(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    st.warning(
        "Backtest accuracy is historical performance only. "
        "It does not guarantee future results."
)
