import streamlit as st
import requests
import pandas as pd
import numpy as np
import websocket
import json
import time
import threading
from datetime import datetime

st.set_page_config(
    page_title="Live Trading Signal Bot",
    page_icon="📊"
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
    "5 Seconds": 5,
    "10 Seconds": 10,
    "15 Seconds": 15,
    "30 Seconds": 30,
    "1 Minute": 60
}

if "ticks" not in st.session_state:
    st.session_state.ticks = []

if "ws_started" not in st.session_state:
    st.session_state.ws_started = False

if "ws_symbol" not in st.session_state:
    st.session_state.ws_symbol = None


# =========================================================
# LIVE TICK DATA
# =========================================================

def start_websocket(symbol):

    if st.session_state.ws_started:
        return

    def on_message(ws, message):
        try:
            data = json.loads(message)

            price = None

            if isinstance(data, dict):
                if "price" in data:
                    price = float(data["price"])

                elif "close" in data:
                    price = float(data["close"])

                elif "data" in data and isinstance(data["data"], dict):
                    if "price" in data["data"]:
                        price = float(data["data"]["price"])

            if price is not None:
                st.session_state.ticks.append({
                    "time": datetime.now(),
                    "price": price
                })

                if len(st.session_state.ticks) > 3000:
                    st.session_state.ticks = st.session_state.ticks[-3000:]

        except Exception:
            pass

    def on_error(ws, error):
        pass

    def on_close(ws, close_status_code, close_msg):
        pass

    def on_open(ws):

        subscribe_message = {
            "action": "subscribe",
            "params": {
                "symbols": symbol
            }
        }

        ws.send(json.dumps(subscribe_message))

    def run():
        try:
            ws = websocket.WebSocketApp(
                "wss://ws.twelvedata.com/v1/quotes/price",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            ws.run_forever()
        except Exception:
            pass

    thread = threading.Thread(
        target=run,
        daemon=True
    )

    thread.start()

    st.session_state.ws_started = True
    st.session_state.ws_symbol = symbol


# =========================================================
# CREATE SHORT-TIMEFRAME CANDLES
# =========================================================

def build_short_candles(seconds):

    if len(st.session_state.ticks) < 20:
        return None

    tick_df = pd.DataFrame(
        st.session_state.ticks
    )

    tick_df["time"] = pd.to_datetime(
        tick_df["time"]
    )

    tick_df["price"] = pd.to_numeric(
        tick_df["price"],
        errors="coerce"
    )

    tick_df = tick_df.dropna()

    if tick_df.empty:
        return None

    tick_df = tick_df.set_index("time")

    rule = f"{seconds}s"

    candles = tick_df["price"].resample(rule).ohlc()

    candles = candles.dropna()

    candles = candles.reset_index()

    candles = candles.rename(
        columns={
            "time": "datetime"
        }
    )

    if len(candles) < 80:
        return None

    candles = candles.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return candles


# =========================================================
# NORMAL TWELVE DATA CANDLES
# =========================================================

def get_data(symbol, interval):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 500,
        "apikey": API_KEY
    }

    try:
        r = requests.get(
            url,
            params=params,
            timeout=15
        )

        data = r.json()

    except Exception as e:
        return None, str(e)

    if "values" not in data:
        return None, data.get(
            "message",
            "Market data error"
        )

    df = pd.DataFrame(
        data["values"]
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    for c in [
        "open",
        "high",
        "low",
        "close"
    ]:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna()

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df, None


# =========================================================
# INDICATORS
# =========================================================

def indicators(df):

    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df[
        "macd"
    ].ewm(
        span=9,
        adjust=False
    ).mean()

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = abs(
        df["high"] -
        df["close"].shift()
    )

    tr3 = abs(
        df["low"] -
        df["close"].shift()
    )

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(
        14
    ).mean()

    df["atr_avg"] = df[
        "atr"
    ].rolling(
        30
    ).mean()

    df["body"] = abs(
        df["close"] -
        df["open"]
    )

    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(
            0,
            np.nan
        )
    )

    df["momentum"] = (
        df["close"].pct_change(3)
    )

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def signal_engine(df):

    last = df.iloc[-1]

    up = 0
    down = 0

    reasons = []

    # Trend
    if (
        last["ema9"] >
        last["ema21"] >
        last["ema50"]
    ):

        up += 3

        reasons.append(
            "Strong bullish EMA trend"
        )

    elif (
        last["ema9"] <
        last["ema21"] <
        last["ema50"]
    ):

        down += 3

        reasons.append(
            "Strong bearish EMA trend"
        )

    # RSI
    if 52 <= last["rsi"] <= 68:

        up += 2

        reasons.append(
            "Bullish RSI zone"
        )

    elif 32 <= last["rsi"] <= 48:

        down += 2

        reasons.append(
            "Bearish RSI zone"
        )

    # MACD
    if (
        last["macd"] >
        last["macd_signal"]
    ):

        up += 2

        reasons.append(
            "MACD bullish"
        )

    elif (
        last["macd"] <
        last["macd_signal"]
    ):

        down += 2

        reasons.append(
            "MACD bearish"
        )

    # Momentum
    if last["momentum"] > 0:

        up += 2

        reasons.append(
            "Positive momentum"
        )

    elif last["momentum"] < 0:

        down += 2

        reasons.append(
            "Negative momentum"
        )

    # Candle strength
    if pd.notna(
        last["body_ratio"]
    ):

        if last["body_ratio"] >= 0.55:

            if (
                last["close"] >
                last["open"]
            ):

                up += 2

                reasons.append(
                    "Strong bullish candle"
                )

            else:

                down += 2

                reasons.append(
                    "Strong bearish candle"
                )

    # Recent structure
    if len(df) >= 12:

        recent_high = df[
            "high"
        ].iloc[-11:-1].max()

        recent_low = df[
            "low"
        ].iloc[-11:-1].min()

        if (
            last["close"] >
            recent_high
        ):

            up += 2

            reasons.append(
                "Recent high breakout"
            )

        elif (
            last["close"] <
            recent_low
        ):

            down += 2

            reasons.append(
                "Recent low breakdown"
            )

    # Volatility
    if (
        pd.notna(last["atr"]) and
        pd.notna(last["atr_avg"])
    ):

        if (
            last["atr"] >=
            last["atr_avg"] * 0.75
        ):

            if up > down:

                up += 1

            elif down > up:

                down += 1

    total = up + down

    if total == 0:

        return (
            "UP",
            50.0,
            up,
            down,
            ["No clear bias; forced direction"]
        )

    if up >= down:

        direction = "UP"

        strength = round(
            up / total * 100,
            1
        )

    else:

        direction = "DOWN"

        strength = round(
            down / total * 100,
            1
        )

    return (
        direction,
        strength,
        up,
        down,
        reasons
    )


# =========================================================
# BACKTEST
# =========================================================

def backtest(df):

    wins = 0
    losses = 0

    rows = []

    if len(df) < 62:
        return (
            0,
            0,
            0,
            pd.DataFrame()
        )

    for i in range(
        60,
        len(df) - 1
    ):

        historical = df.iloc[
            :i + 1
        ].copy()

        signal, strength, up, down, reasons = signal_engine(
            historical
        )

        current = df.iloc[i]["close"]

        following = df.iloc[
            i + 1
        ]["close"]

        if signal == "UP":

            result = following > current

        else:

            result = following < current

        if result:

            wins += 1

            outcome = "WIN"

        else:

            losses += 1

            outcome = "LOSS"

        rows.append({

            "time":
                df.iloc[i]["datetime"],

            "signal":
                signal,

            "strength":
                strength,

            "result":
                outcome
        })

    total = wins + losses

    accuracy = (
        wins / total * 100
        if total
        else 0
    )

    return (
        accuracy,
        wins,
        losses,
        pd.DataFrame(rows)
    )


# =========================================================
# UI
# =========================================================

st.title(
    "📊 Live Trading Signal Bot"
)

st.caption(
    "Real-time market analysis + historical backtesting"
)

if not API_KEY:

    st.error(
        "Twelve Data API key is missing."
    )

    st.stop()


pair = st.selectbox(
    "Pair",
    PAIRS
)


timeframe_name = st.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys())
)


seconds = TIMEFRAMES[
    timeframe_name
]


# =========================================================
# START LIVE CONNECTION
# =========================================================

if seconds <= 30:

    if (
        not st.session_state.ws_started
        or
        st.session_state.ws_symbol != pair
    ):

        st.session_state.ws_started = False

        start_websocket(pair)

        st.info(
            "Live tick data collect ho raha hai. "
            "Kuch seconds wait karke START ANALYZE dabao."
        )


# =========================================================
# ANALYZE
# =========================================================

if st.button(
    "🚀 START ANALYZE",
    use_container_width=True
):

    with st.spinner(
        "Analyzing market..."
    ):

        # Short timeframe
        if seconds <= 30:

            df = build_short_candles(
                seconds
            )

            if df is None:

                st.warning(
                    "Abhi enough live tick data available nahi hai. "
                    "10-30 seconds wait karke dobara START ANALYZE dabao."
                )

                st.stop()

            error = None

        # 1 minute
        else:

            df, error = get_data(
                pair,
                "1min"
            )

        if error:

            st.error(error)

            st.stop()

        df = indicators(df)

        signal, strength, up, down, reasons = signal_engine(
            df
        )

        accuracy, wins, losses, history = backtest(
            df
        )

        price = df.iloc[-1]["close"]


    st.success(
        "Analysis completed"
    )


    st.metric(
        "Live Price",
        f"{price:.6f}"
    )


    if signal == "UP":

        st.success(
            "🟢 SIGNAL: UP"
        )

    else:

        st.error(
            "🔴 SIGNAL: DOWN"
        )


    st.metric(
        "Signal Strength",
        f"{strength}%"
    )


    st.divider()


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "UP Score",
            up
        )


    with col2:

        st.metric(
            "DOWN Score",
            down
        )


    st.write(
        "### 🧠 Analysis"
    )


    for reason in reasons:

        st.write(
            "•",
            reason
        )


    st.divider()


    st.write(
        "### 🧪 Historical Backtest"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )


    with c2:

        st.metric(
            "WIN",
            wins
        )


    with c3:

        st.metric(
            "LOSS",
            losses
        )


    st.caption(
        f"Tested on {len(history)} historical signals."
    )


    st.write(
        "### 🕒 Signal Time"
    )


    st.write(
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    st.warning(
        "Historical accuracy does not guarantee future results."
    )


# =========================================================
# AUTO REFRESH
# =========================================================

if seconds <= 30:

    time.sleep(1)

    st.rerun()
