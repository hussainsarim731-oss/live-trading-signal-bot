import streamlit as st
import websocket
import json
import threading
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Adaptive Live Trading Signal Bot",
    page_icon="📊",
    layout="centered"
)

# =========================================================
# API
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
# SESSION STATE
# =========================================================

if "ticks" not in st.session_state:
    st.session_state.ticks = {}

if "ws_status" not in st.session_state:
    st.session_state.ws_status = "Disconnected"

if "ws_started" not in st.session_state:
    st.session_state.ws_started = False

if "pending_signal" not in st.session_state:
    st.session_state.pending_signal = None

if "live_results" not in st.session_state:
    st.session_state.live_results = []


# =========================================================
# WEBSOCKET STORAGE
# =========================================================

def store_tick(symbol, price):

    if symbol not in st.session_state.ticks:
        st.session_state.ticks[symbol] = []

    st.session_state.ticks[symbol].append({
        "time": datetime.now(),
        "price": float(price)
    })

    # Keep latest 5000 ticks
    st.session_state.ticks[symbol] = (
        st.session_state.ticks[symbol][-5000:]
    )


# =========================================================
# WEBSOCKET CALLBACKS
# =========================================================

def websocket_worker(symbol):

    ws_url = (
        "wss://ws.twelvedata.com/v1/quotes/price"
        "?apikey=" + API_KEY
    )

    def on_open(ws):

        st.session_state.ws_status = "Connected"

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

            if data.get("event") == "price":

                price = data.get("price")

                if price is not None:

                    store_tick(
                        symbol,
                        float(price)
                    )

        except Exception:
            pass

    def on_error(ws, error):

        st.session_state.ws_status = "Error"

    def on_close(ws, close_status_code, close_msg):

        st.session_state.ws_status = "Disconnected"

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()


# =========================================================
# START WEBSOCKET
# =========================================================

def start_websocket(symbol):

    if st.session_state.ws_started:
        return

    st.session_state.ws_started = True

    thread = threading.Thread(
        target=websocket_worker,
        args=(symbol,),
        daemon=True
    )

    thread.start()


# =========================================================
# BUILD LOCAL CANDLES
# =========================================================

def build_candles(symbol, seconds):

    if symbol not in st.session_state.ticks:
        return pd.DataFrame()

    ticks = st.session_state.ticks[symbol]

    if len(ticks) < 10:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)

    df["time"] = pd.to_datetime(
        df["time"]
    )

    df = df.set_index("time")

    candles = df["price"].resample(
        f"{seconds}s"
    ).agg([
        "first",
        "max",
        "min",
        "last"
    ])

    candles.columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    candles = candles.dropna()

    candles = candles.reset_index()

    return candles


# =========================================================
# HISTORICAL DATA
# =========================================================

def get_historical(symbol):

    url = (
        "https://api.twelvedata.com/time_series"
    )

    params = {
        "symbol": symbol,
        "interval": "1min",
        "outputsize": 500,
        "apikey": API_KEY
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        data = response.json()

        if "values" not in data:
            return None

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
        ).reset_index(
            drop=True
        )

        return df

    except Exception:

        return None


# =========================================================
# INDICATORS
# =========================================================

def indicators(df):

    df = df.copy()

    # EMA
    df["ema5"] = df["close"].ewm(
        span=5,
        adjust=False
    ).mean()

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

    # RSI
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

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # ATR
    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = abs(
        df["high"]
        - df["close"].shift()
    )

    tr3 = abs(
        df["low"]
        - df["close"].shift()
    )

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(
        14
    ).mean()

    # Candle
    df["body"] = abs(
        df["close"]
        - df["open"]
    )

    df["range"] = (
        df["high"]
        - df["low"]
    )

    df["body_ratio"] = (
        df["body"]
        / df["range"].replace(
            0,
            np.nan
        )
    )

    # Momentum
    df["momentum"] = (
        df["close"]
        .pct_change(3)
    )

    return df


# =========================================================
# MARKET REGIME
# =========================================================

def market_regime(df):

    if len(df) < 50:
        return "WARMING UP"

    last = df.iloc[-1]

    ema_distance = abs(
        last["ema9"]
        - last["ema21"]
    )

    volatility = last["atr"]

    if pd.isna(volatility):
        return "UNKNOWN"

    if ema_distance > volatility * 0.30:

        if last["ema9"] > last["ema21"]:
            return "BULL TREND"

        return "BEAR TREND"

    return "SIDEWAYS"


# =========================================================
# ADAPTIVE SIGNAL
# =========================================================

def adaptive_signal(df):

    if len(df) < 60:

        return (
            "UP",
            50,
            0,
            0,
            "WARMING UP",
            ["Collecting enough candles"]
        )

    last = df.iloc[-1]

    regime = market_regime(df)

    up = 0
    down = 0

    up_reasons = []
    down_reasons = []

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if (
        last["ema5"]
        > last["ema9"]
        > last["ema21"]
        > last["ema50"]
    ):

        up += 4

        up_reasons.append(
            "EMA trend bullish"
        )

    elif (
        last["ema5"]
        < last["ema9"]
        < last["ema21"]
        < last["ema50"]
    ):

        down += 4

        down_reasons.append(
            "EMA trend bearish"
        )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 52 <= last["rsi"] <= 68:

        up += 3

        up_reasons.append(
            "RSI bullish"
        )

    elif 32 <= last["rsi"] <= 48:

        down += 3

        down_reasons.append(
            "RSI bearish"
        )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if (
        last["macd"]
        > last["macd_signal"]
        and last["macd_hist"] > 0
    ):

        up += 3

        up_reasons.append(
            "MACD bullish"
        )

    elif (
        last["macd"]
        < last["macd_signal"]
        and last["macd_hist"] < 0
    ):

        down += 3

        down_reasons.append(
            "MACD bearish"
        )

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    if last["momentum"] > 0:

        up += 2

        up_reasons.append(
            "Positive momentum"
        )

    elif last["momentum"] < 0:

        down += 2

        down_reasons.append(
            "Negative momentum"
        )

    # -----------------------------------------------------
    # CANDLE
    # -----------------------------------------------------

    if (
        pd.notna(last["body_ratio"])
        and last["body_ratio"] >= 0.55
    ):

        if last["close"] > last["open"]:

            up += 2

            up_reasons.append(
                "Strong bullish candle"
            )

        elif last["close"] < last["open"]:

            down += 2

            down_reasons.append(
                "Strong bearish candle"
            )

    # -----------------------------------------------------
    # REGIME ADAPTATION
    # -----------------------------------------------------

    if regime == "BULL TREND":

        up += 2

        up_reasons.append(
            "Bull trend regime"
        )

    elif regime == "BEAR TREND":

        down += 2

        down_reasons.append(
            "Bear trend regime"
        )

    elif regime == "SIDEWAYS":

        # Reduce aggressive trend signals
        if up > down:
            up -= 1

        elif down > up:
            down -= 1

    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    total = max(
        up + down,
        1
    )

    if up >= down:

        signal = "UP"

        strength = (
            up / total
        ) * 100

        reasons = up_reasons

    else:

        signal = "DOWN"

        strength = (
            down / total
        ) * 100

        reasons = down_reasons

    return (
        signal,
        round(strength, 1),
        round(up, 1),
        round(down, 1),
        regime,
        reasons
    )


# =========================================================
# PAGE
# =========================================================

st.title(
    "📊 Adaptive Live Trading Signal Bot"
)

st.caption(
    "WebSocket ticks + local candles + adaptive analysis"
)

if not API_KEY:

    st.error(
        "TWELVE_DATA_API_KEY missing."
    )

    st.stop()

# =========================================================
# CONTROLS
# =========================================================

pair = st.selectbox(
    "Pair",
    PAIRS
)

timeframe = st.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys())
)

seconds = TIMEFRAMES[
    timeframe
]

# =========================================================
# START WEBSOCKET
# =========================================================

start_websocket(pair)

st.write(
    f"WebSocket status: **{st.session_state.ws_status}**"
)

# =========================================================
# ANALYZE
# =========================================================

if st.button(
    "🚀 ANALYZE LIVE MARKET",
    use_container_width=True
):

    # Get live price from REST
    price, error = get_historical(pair), None

    if price is None:

        st.error(
            "Historical market data unavailable."
        )

        st.stop()

    # Local candles
    df = build_candles(
        pair,
        seconds
    )

    if len(df) < 60:

        st.warning(
            f"{timeframe} ke liye abhi "
            f"{len(df)} candles available hain. "
            f"Kam az kam 60 candles collect hone dein."
        )

        st.info(
            "WebSocket live ticks background mein "
            "collect ho rahe hain."
        )

        st.stop()

    df = indicators(df)

    (
        signal,
        strength,
        up_score,
        down_score,
        regime,
        reasons
    ) = adaptive_signal(df)

    live_price = df.iloc[-1]["close"]

    # =====================================================
    # SIGNAL
    # =====================================================

    st.metric(
        "💰 LIVE PRICE",
        f"{live_price:.6f}"
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
        "SIGNAL STRENGTH",
        f"{strength}%"
    )

    st.metric(
        "MARKET REGIME",
        regime
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "UP SCORE",
            up_score
        )

    with c2:

        st.metric(
            "DOWN SCORE",
            down_score
        )

    st.divider()

    st.subheader(
        "🧠 Signal Reasons"
    )

    for reason in reasons:

        st.write(
            "• " + reason
        )

    st.divider()

    st.subheader(
        "📈 Indicators"
    )

    last = df.iloc[-1]

    st.write(
        f"EMA 5: {last['ema5']:.6f}"
    )

    st.write(
        f"EMA 9: {last['ema9']:.6f}"
    )

    st.write(
        f"EMA 21: {last['ema21']:.6f}"
    )

    st.write(
        f"EMA 50: {last['ema50']:.6f}"
    )

    st.write(
        f"RSI: {last['rsi']:.2f}"
    )

    st.write(
        f"MACD: {last['macd']:.6f}"
    )

    st.write(
        f"ATR: {last['atr']:.6f}"
    )

    st.divider()

    st.subheader(
        "⚠️ Important"
    )

    st.warning(
        "5s/10s/15s/30s signals ke liye "
        "WebSocket ko enough ticks collect karne "
        "dene honge. Strength % guaranteed "
        "win probability nahi hai."
    )

# =========================================================
# LIVE TICK COUNTER
# =========================================================

tick_count = len(
    st.session_state.ticks.get(
        pair,
        []
    )
)

st.divider()

st.write(
    f"📡 Live ticks collected: **{tick_count}**"
)

st.write(
    "WebSocket: **"
    + st.session_state.ws_status
    + "**"
)
