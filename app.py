import streamlit as st
import websocket
import json
import threading
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, deque

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Adaptive Live Trading Signal Bot",
    page_icon="📊",
    layout="centered"
)

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
# GLOBAL WEBSOCKET STORAGE
# =========================================================
# Important:
# WebSocket data yahan store hota hai, Streamlit session_state
# mein nahi. Isse rerun par tick buffer normally disappear nahi
# hota.

if "GLOBAL_TICKS" not in st.session_state:
    st.session_state.GLOBAL_TICKS = {}

if "WS_THREADS" not in st.session_state:
    st.session_state.WS_THREADS = {}

if "WS_STATUS" not in st.session_state:
    st.session_state.WS_STATUS = {}

if "PENDING" not in st.session_state:
    st.session_state.PENDING = None

if "RESULTS" not in st.session_state:
    st.session_state.RESULTS = []


# =========================================================
# INITIALIZE SYMBOL BUFFER
# =========================================================

def init_symbol(symbol):

    if symbol not in st.session_state.GLOBAL_TICKS:
        st.session_state.GLOBAL_TICKS[symbol] = deque(
            maxlen=10000
        )


# =========================================================
# SAVE TICK
# =========================================================

def save_tick(symbol, price):

    init_symbol(symbol)

    st.session_state.GLOBAL_TICKS[symbol].append(
        {
            "time": datetime.now(),
            "price": float(price)
        }
    )


# =========================================================
# WEBSOCKET
# =========================================================

def websocket_loop(symbol):

    url = (
        "wss://ws.twelvedata.com/v1/quotes/price"
        "?apikey=" + API_KEY
    )

    def on_open(ws):

        st.session_state.WS_STATUS[symbol] = "CONNECTED"

        subscribe = {
            "action": "subscribe",
            "params": {
                "symbols": symbol
            }
        }

        ws.send(
            json.dumps(subscribe)
        )

    def on_message(ws, message):

        try:

            data = json.loads(message)

            event = data.get("event")

            if event == "price":

                price = data.get("price")

                if price is not None:

                    save_tick(
                        symbol,
                        float(price)
                    )

        except Exception:
            pass

    def on_error(ws, error):

        st.session_state.WS_STATUS[
            symbol
        ] = "ERROR"

    def on_close(
        ws,
        close_status_code,
        close_msg
    ):

        st.session_state.WS_STATUS[
            symbol
        ] = "DISCONNECTED"

    while True:

        try:

            st.session_state.WS_STATUS[
                symbol
            ] = "CONNECTING"

            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            ws.run_forever(
                ping_interval=20,
                ping_timeout=10
            )

        except Exception:

            st.session_state.WS_STATUS[
                symbol
            ] = "RECONNECTING"

        time.sleep(3)


# =========================================================
# START WEBSOCKET
# =========================================================

def start_websocket(symbol):

    init_symbol(symbol)

    if symbol in st.session_state.WS_THREADS:
        thread = st.session_state.WS_THREADS[symbol]

        if thread.is_alive():
            return

    thread = threading.Thread(
        target=websocket_loop,
        args=(symbol,),
        daemon=True
    )

    thread.start()

    st.session_state.WS_THREADS[
        symbol
    ] = thread


# =========================================================
# LOCAL CANDLES FROM WEBSOCKET TICKS
# =========================================================

def make_local_candles(symbol, seconds):

    init_symbol(symbol)

    ticks = list(
        st.session_state.GLOBAL_TICKS[
            symbol
        ]
    )

    if len(ticks) < 5:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)

    df["time"] = pd.to_datetime(
        df["time"]
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = df.dropna()

    if df.empty:
        return pd.DataFrame()

    df = df.set_index("time")

    candles = df["price"].resample(
        f"{seconds}s"
    ).agg(
        [
            "first",
            "max",
            "min",
            "last"
        ]
    )

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
# HISTORICAL 1-MINUTE DATA
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

        for col in [
            "open",
            "high",
            "low",
            "close"
        ]:

            df[col] = pd.to_numeric(
                df[col],
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

def calculate_indicators(df):

    df = df.copy()

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
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
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
        df["macd"].ewm(
            span=9,
            adjust=False
        ).mean()
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
        /
        df["range"].replace(
            0,
            np.nan
        )
    )

    # Momentum

    df["momentum"] = (
        df["close"]
        .pct_change(3)
    )

    # Recent structure

    df["recent_high"] = (
        df["high"]
        .rolling(10)
        .max()
        .shift(1)
    )

    df["recent_low"] = (
        df["low"]
        .rolling(10)
        .min()
        .shift(1)
    )

    return df


# =========================================================
# MARKET REGIME
# =========================================================

def detect_regime(df):

    if len(df) < 30:
        return "WARMING UP"

    last = df.iloc[-1]

    if pd.isna(last["atr"]):
        return "UNKNOWN"

    ema_gap = abs(
        last["ema9"]
        - last["ema21"]
    )

    atr = last["atr"]

    if atr <= 0:
        return "SIDEWAYS"

    ratio = ema_gap / atr

    if ratio > 0.35:

        if last["ema9"] > last["ema21"]:
            return "BULL TREND"

        return "BEAR TREND"

    return "SIDEWAYS"


# =========================================================
# MULTI STRATEGY ENGINE
# =========================================================

def generate_signal(df):

    if len(df) < 30:

        return (
            "UP",
            50.0,
            0,
            0,
            "WARMING UP",
            ["Collecting candles"]
        )

    last = df.iloc[-1]

    up = 0.0
    down = 0.0

    up_reasons = []
    down_reasons = []

    regime = detect_regime(df)

    # -----------------------------------------------------
    # 1. EMA TREND
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
    # 2. RSI
    # -----------------------------------------------------

    if 52 <= last["rsi"] <= 68:

        up += 3

        up_reasons.append(
            "RSI bullish zone"
        )

    elif 32 <= last["rsi"] <= 48:

        down += 3

        down_reasons.append(
            "RSI bearish zone"
        )

    elif last["rsi"] < 30:

        up += 2

        up_reasons.append(
            "RSI oversold"
        )

    elif last["rsi"] > 70:

        down += 2

        down_reasons.append(
            "RSI overbought"
        )

    # -----------------------------------------------------
    # 3. MACD
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
    # 4. MOMENTUM
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
    # 5. CANDLE STRENGTH
    # -----------------------------------------------------

    if (
        pd.notna(last["body_ratio"])
        and last["body_ratio"] >= 0.55
    ):

        if last["close"] > last["open"]:

            up += 3

            up_reasons.append(
                "Strong bullish candle"
            )

        elif last["close"] < last["open"]:

            down += 3

            down_reasons.append(
                "Strong bearish candle"
            )

    # -----------------------------------------------------
    # 6. BREAKOUT
    # -----------------------------------------------------

    if (
        pd.notna(last["recent_high"])
        and last["close"]
        > last["recent_high"]
    ):

        up += 4

        up_reasons.append(
            "Recent high breakout"
        )

    elif (
        pd.notna(last["recent_low"])
        and last["close"]
        < last["recent_low"]
    ):

        down += 4

        down_reasons.append(
            "Recent low breakdown"
        )

    # -----------------------------------------------------
    # 7. PRICE STRUCTURE
    # -----------------------------------------------------

    if len(df) >= 4:

        a = df.iloc[-1]["close"]
        b = df.iloc[-2]["close"]
        c = df.iloc[-3]["close"]

        if a > b > c:

            up += 2

            up_reasons.append(
                "Higher close structure"
            )

        elif a < b < c:

            down += 2

            down_reasons.append(
                "Lower close structure"
            )

    # -----------------------------------------------------
    # 8. REGIME
    # -----------------------------------------------------

    if regime == "BULL TREND":

        up += 3

        up_reasons.append(
            "Bull trend regime"
        )

    elif regime == "BEAR TREND":

        down += 3

        down_reasons.append(
            "Bear trend regime"
        )

    elif regime == "SIDEWAYS":

        # Sideways mein trend signals ko
        # thora reduce karte hain.
        if up > down:
            up -= 1

        elif down > up:
            down -= 1

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    total = up + down

    if total <= 0:

        return (
            "UP",
            50.0,
            up,
            down,
            regime,
            ["No clear technical bias"]
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
# MAIN
# =========================================================

st.title(
    "📊 Adaptive Live Trading Signal Bot"
)

st.caption(
    "WebSocket + 5s/10s/15s/30s/1min + Multi-Strategy"
)

if not API_KEY:

    st.error(
        "TWELVE_DATA_API_KEY missing."
    )

    st.stop()

# =========================================================
# SELECT
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

status = st.session_state.WS_STATUS.get(
    pair,
    "STARTING"
)

tick_count = len(
    st.session_state.GLOBAL_TICKS.get(
        pair,
        []
    )
)

st.info(
    f"📡 WebSocket: **{status}** | "
    f"Live ticks: **{tick_count}**"
)

# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🚀 ANALYZE LIVE MARKET",
    use_container_width=True
):

    # Build local candles
    local_df = make_local_candles(
        pair,
        seconds
    )

    candle_count = len(local_df)

    st.write(
        f"Local {timeframe} candles: "
        f"**{candle_count}**"
    )

    # -----------------------------------------------------
    # 5s / 10s / 15s / 30s
    # -----------------------------------------------------

    if seconds < 60:

        if candle_count >= 30:

            df = local_df

            st.success(
                "✅ Real WebSocket candles available."
            )

        else:

            # Warm-up from historical data.
            # Signal will be based on 1-minute data
            # until enough local candles arrive.

            historical = get_historical(
                pair
            )

            if historical is None:

                st.error(
                    "Market data unavailable."
                )

                st.stop()

            df = historical

            st.warning(
                f"{timeframe} ke liye abhi "
                f"{candle_count} local candles hain. "
                f"Warm-up ke liye temporary 1-minute "
                f"historical data use ho raha hai."
            )

    # -----------------------------------------------------
    # 1 MINUTE
    # -----------------------------------------------------

    else:

        historical = get_historical(
            pair
        )

        if historical is None:

            st.error(
                "Historical data unavailable."
            )

            st.stop()

        df = historical

    # =====================================================
    # INDICATORS
    # =====================================================

    df = calculate_indicators(
        df
    )

    (
        signal,
        strength,
        up_score,
        down_score,
        regime,
        reasons
    ) = generate_signal(df)

    live_price = df.iloc[-1]["close"]

    # =====================================================
    # SIGNAL
    # =====================================================

    st.divider()

    st.metric(
        "💰 LIVE PRICE",
        f"{live_price:.6f}"
    )

    if signal == "UP":

        st.success(
            "🟢 TRADE DIRECTION: UP"
        )

    else:

        st.error(
            "🔴 TRADE DIRECTION: DOWN"
        )

    st.metric(
        "📊 SIGNAL STRENGTH",
        f"{strength}%"
    )

    st.metric(
        "📈 MARKET REGIME",
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

    # =====================================================
    # REASONS
    # =====================================================

    st.divider()

    st.subheader(
        "🧠 Strategy Confirmations"
    )

    for reason in reasons:

        st.write(
            "• " + reason
        )

    # =====================================================
    # INDICATORS
    # =====================================================

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
        f"MACD: 
