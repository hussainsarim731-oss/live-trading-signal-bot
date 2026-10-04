import streamlit as st
import websocket
import json
import threading
import time
from collections import deque
from datetime import datetime

import requests
import pandas as pd
import numpy as np


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Live Trading Signal Bot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Live Trading Signal System")


# =========================================================
# SETTINGS
# =========================================================

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


# =========================================================
# GLOBAL LIVE DATA
# =========================================================

if "tick_buffer" not in st.session_state:
    st.session_state.tick_buffer = {}

if "ws_started" not in st.session_state:
    st.session_state.ws_started = False

if "live_signal" not in st.session_state:
    st.session_state.live_signal = None

if "signal_time" not in st.session_state:
    st.session_state.signal_time = None

if "signal_price" not in st.session_state:
    st.session_state.signal_price = None

if "results" not in st.session_state:
    st.session_state.results = []

if "last_update" not in st.session_state:
    st.session_state.last_update = None


# =========================================================
# SELECTORS
# =========================================================

col1, col2 = st.columns(2)

with col1:
    pair = st.selectbox(
        "Pair",
        PAIRS
    )

with col2:
    timeframe_name = st.selectbox(
        "Candle",
        list(TIMEFRAMES.keys()),
        index=2
    )

TIMEFRAME = TIMEFRAMES[timeframe_name]


# =========================================================
# WEBSOCKET
# =========================================================

def start_websocket(symbol):

    if not API_KEY:
        return

    if symbol not in st.session_state.tick_buffer:
        st.session_state.tick_buffer[symbol] = deque(maxlen=20000)

    def on_open(ws):
        message = {
            "action": "subscribe",
            "params": {
                "symbols": symbol
            }
        }

        ws.send(json.dumps(message))

    def on_message(ws, message):

        try:
            data = json.loads(message)

            if "price" in data:

                price = float(data["price"])

                tick = {
                    "price": price,
                    "time": time.time()
                }

                st.session_state.tick_buffer[symbol].append(tick)

        except Exception:
            pass

    def on_error(ws, error):
        pass

    def on_close(ws, close_status_code, close_msg):
        pass

    url = (
        "wss://ws.twelvedata.com/v1/quotes/price?"
        + "apikey="
        + API_KEY
    )

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    thread = threading.Thread(
        target=ws.run_forever,
        daemon=True
    )

    thread.start()


if not st.session_state.ws_started:

    start_websocket(pair)

    st.session_state.ws_started = True


# =========================================================
# GET LIVE PRICE
# =========================================================

def get_ticks(symbol):

    if symbol not in st.session_state.tick_buffer:
        return []

    return list(st.session_state.tick_buffer[symbol])


ticks = get_ticks(pair)


# =========================================================
# LIVE PRICE
# =========================================================

if len(ticks) > 0:

    live_price = ticks[-1]["price"]

    st.metric(
        "Live Price",
        f"{live_price:.6f}"
    )

    st.session_state.last_update = datetime.now().strftime(
        "%H:%M:%S"
    )

    st.caption(
        "Last update: "
        + str(st.session_state.last_update)
    )

else:

    st.warning(
        "Live price abhi receive nahi hui. "
        "Kuch seconds wait karein."
    )


# =========================================================
# BUILD LOCAL CANDLES
# =========================================================

def make_candles(tick_data, seconds):

    if len(tick_data) == 0:
        return pd.DataFrame()

    rows = []

    for tick in tick_data:

        ts = tick["time"]
        price = tick["price"]

        bucket = int(ts // seconds) * seconds

        rows.append(
            {
                "bucket": bucket,
                "price": price
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    candles = (
        df.groupby("bucket")["price"]
        .agg(
            open="first",
            high="max",
            low="min",
            close="last"
        )
        .reset_index()
    )

    candles["datetime"] = pd.to_datetime(
        candles["bucket"],
        unit="s"
    )

    return candles


candles = make_candles(
    ticks,
    TIMEFRAME
)


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):

    data = df.copy()

    data["ema5"] = (
        data["close"]
        .ewm(span=5, adjust=False)
        .mean()
    )

    data["ema9"] = (
        data["close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["ema21"] = (
        data["close"]
        .ewm(span=21, adjust=False)
        .mean()
    )

    data["ema50"] = (
        data["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    delta = data["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.rolling(14)
        .mean()
    )

    avg_loss = (
        loss.rolling(14)
        .mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["rsi"] = 100 - (
        100 / (1 + rs)
    )

    ema12 = (
        data["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        data["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    data["macd"] = ema12 - ema26

    data["macd_signal"] = (
        data["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["macd_hist"] = (
        data["macd"]
        - data["macd_signal"]
    )

    previous_close = data["close"].shift(1)

    tr1 = data["high"] - data["low"]

    tr2 = (
        data["high"]
        - previous_close
    ).abs()

    tr3 = (
        data["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    data["atr"] = (
        true_range
        .rolling(14)
        .mean()
    )

    data["momentum"] = (
        data["close"]
        - data["close"].shift(5)
    )

    data["previous_high"] = (
        data["high"]
        .rolling(10)
        .max()
        .shift(1)
    )

    data["previous_low"] = (
        data["low"]
        .rolling(10)
        .min()
        .shift(1)
    )

    data["body"] = (
        data["close"]
        - data["open"]
    )

    data["range"] = (
        data["high"]
        - data["low"]
    )

    data["body_ratio"] = (
        data["body"].abs()
        / data["range"].replace(0, np.nan)
    )

    return data


if len(candles) >= 60:

    candles = calculate_indicators(candles)


# =========================================================
# MARKET REGIME
# =========================================================

def get_market_regime(row):

    if pd.isna(row["ema21"]) or pd.isna(row["ema50"]):
        return "WARMING UP"

    distance = abs(
        row["ema21"] - row["ema50"]
    )

    atr = row["atr"]

    if pd.isna(atr) or atr == 0:
        return "WARMING UP"

    ratio = distance / atr

    if ratio > 1.2:

        if row["ema21"] > row["ema50"]:
            return "BULL TREND"

        return "BEAR TREND"

    return "SIDEWAYS"


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(data):

    if len(data) < 60:
        return None

    row = data.iloc[-1]

    required = [
        "ema5",
        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "momentum",
        "body_ratio"
    ]

    for item in required:

        if pd.isna(row[item]):
            return None

    up_score = 0
    down_score = 0

    reasons_up = []
    reasons_down = []

    # -----------------------------------------
    # EMA TREND
    # -----------------------------------------

    if (
        row["ema5"] > row["ema9"]
        and row["ema9"] > row["ema21"]
        and row["ema21"] > row["ema50"]
    ):

        up_score += 2
        reasons_up.append("EMA bullish")

    elif (
        row["ema5"] < row["ema9"]
        and row["ema9"] < row["ema21"]
        and row["ema21"] < row["ema50"]
    ):

        down_score += 2
        reasons_down.append("EMA bearish")

    else:

        if row["ema5"] > row["ema21"]:
            up_score += 1
            reasons_up.append("short EMA up")

        else:
            down_score += 1
            reasons_down.append("short EMA down")

    # -----------------------------------------
    # RSI
    # -----------------------------------------

    if row["rsi"] >= 50 and row["rsi"] < 75:

        up_score += 1
        reasons_up.append("RSI bullish")

    elif row["rsi"] < 50 and row["rsi"] > 25:

        down_score += 1
        reasons_down.append("RSI bearish")

    # -----------------------------------------
    # MACD
    # -----------------------------------------

    if (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    ):

        up_score += 2
        reasons_up.append("MACD bullish")

    elif (
        row["macd"] < row["macd_signal"]
        and row["macd_hist"] < 0
    ):

        down_score += 2
        reasons_down.append("MACD bearish")

    # -----------------------------------------
    # MOMENTUM
    # -----------------------------------------

    if row["momentum"] > 0:

        up_score += 1
        reasons_up.append("momentum up")

    elif row["momentum"] < 0:

        down_score += 1
        reasons_down.append("momentum down")

    # -----------------------------------------
    # CANDLE STRENGTH
    # -----------------------------------------

    if row["body"] > 0 and row["body_ratio"] >= 0.55:

        up_score += 1
        reasons_up.append("strong bullish candle")

    elif row["body"] < 0 and row["body_ratio"] >= 0.55:

        down_score += 1
        reasons_down.append("strong bearish candle")

    # -----------------------------------------
    # BREAKOUT
    # -----------------------------------------

    if not pd.isna(row["previous_high"]):

        if row["close"] > row["previous_high"]:

            up_score += 2
            reasons_up.append("high breakout")

        elif row["close"] < row["previous_low"]:

            down_score += 2
            reasons_down.append("low breakout")

    # -----------------------------------------
    # REGIME
    # -----------------------------------------

    regime = get_market_regime(row)

    if regime == "BULL TREND":

        up_score += 1
        reasons_up.append("bull trend")

    elif regime == "BEAR TREND":

        down_score += 1
        reasons_down.append("bear trend")

    # -----------------------------------------
    # FINAL DIRECTION
    # -----------------------------------------

    if up_score >= down_score:

        direction = "UP"
        score = up_score
        reasons = reasons_up

    else:

        direction = "DOWN"
        score = down_score
        reasons = reasons_down

    max_score = 10

    strength = min(
        100,
        int((score / max_score) * 100)
    )

    return {
        "direction": direction,
        "score": score,
        "strength": strength,
        "regime": regime,
        "rsi": float(row["rsi"]),
        "macd": float(row["macd"]),
        "atr": float(row["atr"]),
        "price": float(row["close"]),
        "reasons": reasons
    }


# =========================================================
# DISPLAY CANDLES
# =========================================================

st.subheader("🕯️ Local Candles")

if len(candles) > 0:

    st.write(
        "Available candles:",
        len(candles)
    )

    st.dataframe(
        candles.tail(10),
        use_container_width=True
    )

else:

    st.info(
        "Candles abhi collect ho rahi hain."
    )


# =========================================================
# SIGNAL
# =========================================================

st.subheader("🎯 Current Direction")


if len(candles) >= 60:

    result = generate_signal(candles)

    if result is not None:

        st.session_state.live_signal = result["direction"]
        st.session_state.signal_price = result["price"]
        st.session_state.signal_time = time.time()

        if result["direction"] == "UP":

            st.success(
                "🟢 UP"
            )

        else:

            st.error(
                "🔴 DOWN"
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Direction",
                result["direction"]
            )

        with c2:
            st.metric(
                "Signal Strength",
                str(result["strength"]) + "%"
            )

        with c3:
            st.metric(
                "Market Regime",
                result["regime"]
            )

        st.write(
            "Live analysis price:",
            f"{result['price']:.6f}"
        )

        st.write(
            "RSI:",
            round(result["rsi"], 2)
        )

        st.write(
            "MACD:",
            round(result["macd"], 6)
        )

        st.write(
            "ATR:",
            round(result["atr"], 6)
        )

        st.write(
            "Reasons:"
        )

        for reason in result["reasons"]:
            st.write("•", reason)

else:

    remaining = max(
        0,
        60 - len(candles)
    )

    st.warning(
        str(remaining)
        + " candles aur collect honi hain."
    )


# =========================================================
# SIGNAL RESULT CHECKER
# =========================================================

st.subheader("⏱️ Signal Result")

if (
    st.session_state.signal_time is not None
    and st.session_state.signal_price is not None
    and st.session_state.live_signal is not None
):

    elapsed = (
        time.time()
        - st.session_state.signal_time
    )

    if elapsed >= TIMEFRAME:

        current_price = None

        if len(ticks) > 0:
            current_price = ticks[-1]["price"]

        if current_price is not None:

            entry = st.session_state.signal_price
            direction = st.session_state.live_signal

            if direction == "UP":

                win = current_price > entry

            else:

                win = current_price < entry

            result_text = "WIN" if win else "LOSS"

            record = {
                "time": datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "direction": direction,
                "entry": entry,
                "exit": current_price,
                "result": result_text
            }

            if len(st.session_state.results) == 0:

                st.session_state.results.append(
                    record
                )

            else:

                last = st.session_state.results[-1]

                if (
                    last["time"] != record["time"]
                    or last["entry"] != record["entry"]
                ):

                    st.session_state.results.append(
                        record
                    )

            st.session_state.signal_time = None

            if win:
                st.success(
                    "✅ WIN | "
                    + direction
                )
            else:
                st.error(
                    "❌ LOSS | "
                    + direction
                )

    else:

        remaining_seconds = int(
            TIMEFRAME - elapsed
        )

        st.info(
            "Signal active hai. "
            + str(remaining_seconds)
            + " seconds remaining."
        )


# =========================================================
# RESULT HISTORY
# =========================================================

st.subheader("📈 Live Result History")

if len(st.session_state.results) > 0:

    result_df = pd.DataFrame(
        st.session_state.results
    )

    st.dataframe(
        result_df.tail(20),
        use_container_width=True
    )

    wins = sum(
        1
        for x in st.session_state.results
        if x["result"] == "WIN"
    )

    losses = sum(
        1
        for x in st.session_state.results
        if x["result"] == "LOSS"
    )

    total = wins + losses

    if total > 0:

        accuracy = (
            wins / total
        ) * 100

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Wins",
                wins
            )

        with b:
            st.metric(
                "Losses",
                losses
            )

        with c:
            st.metric(
                "Accuracy",
                f"{accuracy:.2f}%"
            )

else:

    st.info(
        "Abhi live results collect nahi hue."
    )


# =========================================================
# AUTO REFRESH
# =========================================================

time.sleep(1)

st.rerun()
