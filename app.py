import streamlit as st
import requests
import pandas as pd
import numpy as np
import websocket
import threading
import json
import time
from collections import deque
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


# =========================================================
# LIVE TICK COLLECTOR
# =========================================================

@st.cache_resource
def create_collector(api_key):

    class Collector:

        def __init__(self):
            self.ticks = deque(maxlen=20000)
            self.lock = threading.Lock()
            self.ws = None
            self.connected = False
            self.symbol = None
            self.last_price = None
            self.last_tick = None

            thread = threading.Thread(
                target=self.connect,
                daemon=True
            )

            thread.start()

        def connect(self):

            while True:

                try:

                    url = (
                        "wss://ws.twelvedata.com/"
                        "v1/quotes/price?apikey="
                        + self.api_key
                    )

                    self.ws = websocket.WebSocketApp(
                        url,
                        on_open=self.on_open,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close
                    )

                    self.ws.run_forever(
                        ping_interval=10,
                        ping_timeout=5
                    )

                except Exception:
                    pass

                self.connected = False

                time.sleep(3)

        def on_open(self, ws):

            self.connected = True

            if self.symbol:

                self.subscribe(
                    self.symbol
                )

        def subscribe(self, symbol):

            try:

                message = {
                    "action": "subscribe",
                    "params": {
                        "symbols": symbol
                    }
                }

                self.ws.send(
                    json.dumps(message)
                )

            except Exception:
                pass

        def change_symbol(self, symbol):

            self.symbol = symbol

            with self.lock:
                self.ticks.clear()

            self.last_price = None
            self.last_tick = None

            if self.connected:
                self.subscribe(symbol)

        def on_message(self, ws, message):

            try:
                data = json.loads(message)
            except Exception:
                return

            if data.get("event") != "price":
                return

            symbol = data.get("symbol")
            price = data.get("price")
            timestamp = data.get("timestamp")

            if symbol is None or price is None:
                return

            try:
                price = float(price)
            except Exception:
                return

            try:

                if timestamp is not None:
                    ts = pd.to_datetime(
                        float(timestamp),
                        unit="s",
                        utc=True
                    )
                else:
                    ts = pd.Timestamp.now(
                        tz="UTC"
                    )

            except Exception:

                ts = pd.Timestamp.now(
                    tz="UTC"
                )

            with self.lock:

                self.ticks.append({
                    "datetime": ts,
                    "price": price
                })

            self.last_price = price
            self.last_tick = ts

        def on_error(self, ws, error):
            self.connected = False

        def on_close(
            self,
            ws,
            close_status_code,
            close_msg
        ):
            self.connected = False

        def get_ticks(self):

            with self.lock:
                return list(self.ticks)

        def count(self):

            with self.lock:
                return len(self.ticks)

    return Collector()


# =========================================================
# BUILD REAL CANDLES FROM TICKS
# =========================================================

def make_candles(collector, seconds):

    ticks = collector.get_ticks()

    if len(ticks) < 2:
        return None

    df = pd.DataFrame(ticks)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["datetime", "price"]
    )

    if len(df) < 2:
        return None

    df = df.sort_values(
        "datetime"
    )

    df = df.set_index(
        "datetime"
    )

    candles = df["price"].resample(
        f"{seconds}s",
        label="left",
        closed="left"
    ).ohlc()

    candles = candles.dropna()

    if candles.empty:
        return None

    candles = candles.reset_index()

    candles.columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    return candles


# =========================================================
# 1 MINUTE DATA
# =========================================================

def get_data(symbol):

    try:

        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": "1min",
                "outputsize": 500,
                "apikey": API_KEY
            },
            timeout=15
        )

        data = response.json()

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

    for column in [
        "open",
        "high",
        "low",
        "close"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    df = df.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    return df, None


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

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
        14,
        min_periods=1
    ).mean()

    avg_loss = loss.rolling(
        14,
        min_periods=1
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    df["rsi"] = df["rsi"].fillna(50)

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
        14,
        min_periods=1
    ).mean()

    df["atr_avg"] = df["atr"].rolling(
        30,
        min_periods=1
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
    ).fillna(0)

    df["momentum"] = (
        df["close"].pct_change(3)
        .fillna(0)
    )

    return df


# =========================================================
# SIGNAL
# =========================================================

def get_signal(df):

    if df is None or len(df) == 0:

        return (
            "UP",
            50.0,
            0,
            0,
            ["Waiting for live ticks"]
        )

    last = df.iloc[-1]

    up = 0
    down = 0
    reasons = []

    # EMA
    if (
        last["ema9"] >
        last["ema21"] >
        last["ema50"]
    ):

        up += 3
        reasons.append(
            "Bullish EMA trend"
        )

    elif (
        last["ema9"] <
        last["ema21"] <
        last["ema50"]
    ):

        down += 3
        reasons.append(
            "Bearish EMA trend"
        )

    # RSI
    if 52 <= last["rsi"] <= 68:

        up += 2
        reasons.append(
            "Bullish RSI"
        )

    elif 32 <= last["rsi"] <= 48:

        down += 2
        reasons.append(
            "Bearish RSI"
        )

    # MACD
    if last["macd"] > last["macd_signal"]:

        up += 2
        reasons.append(
            "MACD bullish"
        )

    elif last["macd"] < last["macd_signal"]:

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

    # Candle
    if last["body_ratio"] >= 0.55:

        if last["close"] > last["open"]:

            up += 2
            reasons.append(
                "Strong bullish candle"
            )

        elif last["close"] < last["open"]:

            down += 2
            reasons.append(
                "Strong bearish candle"
            )

    # Recent structure
    if len(df) >= 3:

        recent_high = df[
            "high"
        ].iloc[-11:-1].max()

        recent_low = df[
            "low"
        ].iloc[-11:-1].min()

        if (
            pd.notna(recent_high)
            and
            last["close"] > recent_high
        ):

            up += 2
            reasons.append(
                "High breakout"
            )

        elif (
            pd.notna(recent_low)
            and
            last["close"] < recent_low
        ):

            down += 2
            reasons.append(
                "Low breakdown"
            )

    # Volatility
    if (
        pd.notna(last["atr"])
        and
        pd.notna(last["atr_avg"])
        and
        last["atr"] >=
        last["atr_avg"] * 0.75
    ):

        if up > down:
            up += 1

        elif down > up:
            down += 1

    total = up + down

    # Always UP or DOWN
    if total == 0:

        if last["close"] >= last["open"]:

            return (
                "UP",
                50.0,
                up,
                down,
                ["Bullish live candle bias"]
            )

        return (
            "DOWN",
            50.0,
            up,
            down,
            ["Bearish live candle bias"]
        )

    if up >= down:

        return (
            "UP",
            round(
                up / total * 100,
                1
            ),
            up,
            down,
            reasons
        )

    return (
        "DOWN",
        round(
            down / total * 100,
            1
        ),
        up,
        down,
        reasons
    )


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(df):

    if df is None or len(df) < 62:

        return None, 0, 0

    wins = 0
    losses = 0

    for i in range(
        60,
        len(df) - 1
    ):

        historical = df.iloc[
            :i + 1
        ].copy()

        signal, _, _, _, _ = get_signal(
            historical
        )

        current = df.iloc[i]["close"]
        next_price = df.iloc[
            i + 1
        ]["close"]

        if signal == "UP":

            win = next_price > current

        else:

            win = next_price < current

        if win:
            wins += 1
        else:
            losses += 1

    total = wins + losses

    if total == 0:
        return None, wins, losses

    accuracy = (
        wins /
        total *
        100
    )

    return accuracy, wins, losses


# =========================================================
# APP
# =========================================================

st.title(
    "📊 Live Trading Signal Bot"
)

st.caption(
    "Real live tick analysis + UP/DOWN signal"
)

if not API_KEY:

    st.error(
        "TWELVE_DATA_API_KEY missing."
    )

    st.stop()


pair = st.selectbox(
    "Pair",
    PAIRS
)

timeframe = st.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys())
)


collector = create_collector(
    API_KEY
)

collector.change_symbol(
    pair
)


# =========================================================
# START
# =========================================================

if "live" not in st.session_state:

    st.session_state.live = False


c1, c2 = st.columns(2)

with c1:

    if st.button(
        "🚀 START ANALYZE",
        use_container_width=True
    ):

        st.session_state.live = True

with c2:

    if st.button(
        "⛔ STOP",
        use_container_width=True
    ):

        st.session_state.live = False


# =========================================================
# LIVE DISPLAY
# =========================================================

if st.session_state.live:

    @st.fragment(run_every=1)
    def live_screen():

        st.divider()

        st.write(
            "### 📡 Live Connection"
        )

        s1, s2, s3 = st.columns(3)

        with s1:

            if collector.connected:

                st.success(
                    "🟢 Connected"
                )

            else:

                st.warning(
                    "🟡 Connecting..."
                )

        with s2:

            st.metric(
                "Ticks",
                collector.count()
            )

        with s3:

            if collector.last_price:

                st.metric(
                    "Live Price",
                    f"{collector.last_price:.6f}"
                )

            else:

                st.metric(
                    "Live Price",
                    "---"
                )

        seconds = TIMEFRAMES[
            timeframe
        ]

        # -------------------------------------------------
        # 1 MINUTE
        # -------------------------------------------------

        if seconds == 60:

            df, error = get_data(
                pair
            )

            if error:

                st.error(error)
                return

        # -------------------------------------------------
        # SHORT TIMEFRAME
        # -------------------------------------------------

        else:

            df = make_candles(
                collector,
                seconds
            )

            if df is None:

                st.info(
                    "⏳ Live ticks collect ho rahe hain..."
                )

                st.caption(
                    "Real ticks milte hi "
                    f"{seconds}-second candle banegi."
                )

                return

        # -------------------------------------------------
        # ANALYSIS
        # -------------------------------------------------

        df = add_indicators(
            df
        )

        signal, strength, up, down, reasons = (
            get_signal(df)
        )

        price = (
            collector.last_price
            if seconds != 60
            and collector.last_price
            else df.iloc[-1]["close"]
        )

        accuracy, wins, losses = (
            run_backtest(df)
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.write(
            f"### ⚡ {pair} — {timeframe}"
        )

        if signal == "UP":

            st.success(
                "🟢 SIGNAL: UP"
            )

        else:

            st.error(
                "🔴 SIGNAL: DOWN"
            )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Live Price",
                f"{price:.6f}"
            )

        with m2:

            st.metric(
                "Signal Strength",
                f"{strength}%"
            )

        with m3:

            st.metric(
                "Candles",
                len(df)
            )

        # -------------------------------------------------
        # SCORES
        # -------------------------------------------------

        m4, m5 = st.columns(2)

        with m4:

            st.metric(
                "🟢 UP Score",
                up
            )

        with m5:

            st.metric(
                "🔴 DOWN Score",
                down
            )

        # -------------------------------------------------
        # REASONS
        # -------------------------------------------------

        st.write(
            "### 🧠 Analysis"
        )

        for reason in reasons:

            st.write(
                "• " + reason
            )

        # -------------------------------------------------
        # BACKTEST
        # -------------------------------------------------

        st.write(
            "### 🧪 Historical Backtest"
        )

        if accuracy is not None:

            b1, b2, b3 = st.columns(3)

            with b1:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.2f}%"
                )

            with b2:

                st.metric(
                    "WIN",
                    wins
                )

            with b3:

                st.metric(
                    "LOSS",
                    losses
                )

        else:

            st.info(
                "5-second historical backtest ke liye "
                "real candles abhi collect ho rahi hain."
            )

        st.caption(
            "Last update: "
            + datetime.now().strftime(
                "%H:%M:%S"
            )
        )

    live_screen()
