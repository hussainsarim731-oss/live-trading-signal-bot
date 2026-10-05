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
    page_icon="📊",
    layout="wide"
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
def get_tick_collector(api_key):

    class TickCollector:

        def __init__(self):

            self.api_key = api_key

            self.ticks = deque(maxlen=20000)

            self.lock = threading.Lock()

            self.ws = None
            self.thread = None

            self.connected = False
            self.subscribed = False

            self.current_symbol = None

            self.last_tick_time = None
            self.last_price = None

            self.status_message = "Connecting..."

            self.start()

        # -------------------------------------------------
        # CONNECT
        # -------------------------------------------------

        def start(self):

            if self.thread is not None:
                return

            self.thread = threading.Thread(
                target=self.run,
                daemon=True
            )

            self.thread.start()

        # -------------------------------------------------
        # WEBSOCKET RUN
        # -------------------------------------------------

        def run(self):

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

                    self.connected = False
                    self.subscribed = False

                    self.status_message = "Connecting..."

                    self.ws.run_forever(
                        ping_interval=None
                    )

                except Exception as e:

                    self.connected = False
                    self.subscribed = False

                    self.status_message = (
                        "Connection error: "
                        + str(e)[:100]
                    )

                time.sleep(3)

        # -------------------------------------------------
        # OPEN
        # -------------------------------------------------

        def on_open(self, ws):

            self.connected = True
            self.status_message = "Connected"

            symbol = self.current_symbol

            if symbol:

                self.subscribe(symbol)

            # Heartbeat thread
            threading.Thread(
                target=self.heartbeat,
                args=(ws,),
                daemon=True
            ).start()

        # -------------------------------------------------
        # SUBSCRIBE
        # -------------------------------------------------

        def subscribe(self, symbol):

            if not self.connected or self.ws is None:
                return

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

                self.current_symbol = symbol
                self.subscribed = True

                self.status_message = (
                    "Live ticks: " + symbol
                )

            except Exception as e:

                self.status_message = (
                    "Subscribe error: "
                    + str(e)[:100]
                )

        # -------------------------------------------------
        # CHANGE SYMBOL
        # -------------------------------------------------

        def set_symbol(self, symbol):

            if symbol == self.current_symbol:
                return

            self.current_symbol = symbol

            with self.lock:
                self.ticks.clear()

            self.last_price = None
            self.last_tick_time = None

            if self.connected and self.ws:

                try:

                    reset_message = {
                        "action": "reset"
                    }

                    self.ws.send(
                        json.dumps(reset_message)
                    )

                    time.sleep(0.2)

                    self.subscribe(symbol)

                except Exception:
                    pass

        # -------------------------------------------------
        # HEARTBEAT
        # -------------------------------------------------

        def heartbeat(self, ws):

            while self.connected:

                try:

                    time.sleep(10)

                    if not self.connected:
                        break

                    heartbeat_message = {
                        "action": "heartbeat"
                    }

                    ws.send(
                        json.dumps(
                            heartbeat_message
                        )
                    )

                except Exception:
                    break

        # -------------------------------------------------
        # MESSAGE
        # -------------------------------------------------

        def on_message(self, ws, message):

            try:

                data = json.loads(message)

            except Exception:
                return

            # Subscribe status
            if data.get("event") == "subscribe-status":

                self.status_message = (
                    "Subscription active"
                )

                return

            # Only real price event
            if data.get("event") != "price":

                return

            symbol = data.get("symbol")
            price = data.get("price")
            timestamp = data.get("timestamp")

            if symbol is None:
                return

            if price is None:
                return

            try:

                price = float(price)

            except Exception:
                return

            # Timestamp
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

            tick = {
                "datetime": ts,
                "price": price
            }

            with self.lock:

                self.ticks.append(tick)

            self.last_price = price
            self.last_tick_time = ts

            self.status_message = (
                f"Receiving live ticks: {symbol}"
            )

        # -------------------------------------------------
        # ERROR
        # -------------------------------------------------

        def on_error(self, ws, error):

            self.connected = False

            self.status_message = (
                "WebSocket error"
            )

        # -------------------------------------------------
        # CLOSE
        # -------------------------------------------------

        def on_close(
            self,
            ws,
            close_status_code,
            close_msg
        ):

            self.connected = False
            self.subscribed = False

            self.status_message = (
                "Disconnected - reconnecting..."
            )

        # -------------------------------------------------
        # GET TICKS
        # -------------------------------------------------

        def get_ticks(self):

            with self.lock:

                return list(self.ticks)

        def tick_count(self):

            with self.lock:

                return len(self.ticks)

    return TickCollector()


# =========================================================
# BUILD REAL SHORT CANDLES FROM REAL TICKS
# =========================================================

def build_tick_candles(
    collector,
    seconds
):

    ticks = collector.get_ticks()

    if not ticks:
        return None

    df = pd.DataFrame(ticks)

    if df.empty:
        return None

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
        subset=[
            "datetime",
            "price"
        ]
    )

    if df.empty:
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
# 1 MINUTE REST DATA
# =========================================================

def get_data(
    symbol,
    interval
):

    url = (
        "https://api.twelvedata.com/"
        "time_series"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
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

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    return df, None


# =========================================================
# INDICATORS
# =========================================================

def indicators(df):

    df = df.copy()

    # EMA
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
        14,
        min_periods=1
    ).mean()

    avg_loss = loss.rolling(
        14,
        min_periods=1
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

    df["rsi"] = df["rsi"].fillna(50)

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

    # ATR
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
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(
        14,
        min_periods=1
    ).mean()

    df["atr_avg"] = (
        df["atr"].rolling(
            30,
            min_periods=1
        ).mean()
    )

    # Candle
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

    df["body_ratio"] = (
        df["body_ratio"]
        .fillna(0)
    )

    # Momentum
    df["momentum"] = (
        df["close"].pct_change(
            3
        )
    )

    df["momentum"] = (
        df["momentum"]
        .fillna(0)
    )

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def signal_engine(df):

    if df is None or len(df) == 0:

        return (
            "UP",
            50.0,
            0,
            0,
            ["Waiting for live market data"]
        )

    last = df.iloc[-1]

    up = 0
    down = 0

    reasons = []

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if (
        pd.notna(last["ema9"])
        and
        pd.notna(last["ema21"])
        and
        pd.notna(last["ema50"])
    ):

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

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CANDLE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RECENT STRUCTURE
    # -----------------------------------------------------

    if len(df) >= 3:

        recent_high = (
            df["high"]
            .iloc[-11:-1]
            .max()
        )

        recent_low = (
            df["low"]
            .iloc[-11:-1]
            .min()
        )

        if pd.notna(recent_high):

            if (
                last["close"] >
                recent_high
            ):

                up += 2

                reasons.append(
                    "Recent high breakout"
                )

        if pd.notna(recent_low):

            if (
                last["close"] <
                recent_low
            ):

                down += 2

                reasons.append(
                    "Recent low breakdown"
                )

    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

    if (
        pd.notna(last["atr"])
        and
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

    # -----------------------------------------------------
    # ALWAYS UP OR DOWN
    # -----------------------------------------------------

    total = up + down

    if total == 0:

        # First candle / neutral indicators
        # use actual candle direction

        if (
            last["close"] >=
            last["open"]
        ):

            return (
                "UP",
                50.0,
                up,
                down,
                ["Current live candle is bullish"]
            )

        else:

            return (
                "DOWN",
                50.0,
                up,
                down,
                ["Current live candle is bearish"]
            )

    if up >= down:

        direction = "UP"

        strength = round(
            up /
            total *
            100,
            1
        )

    else:

        direction = "DOWN"

        strength = round(
            down /
            total *
            100,
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

    if (
        df is None
        or
        len(df) < 62
    ):

        return (
            None,
            0,
            0,
            pd.DataFrame()
        )

    wins = 0
    losses = 0

    rows = []

    for i in range(
        60,
        len(df) - 1
    ):

        historical = (
            df.iloc[:i + 1]
            .copy()
        )

        signal, strength, up, down, reasons = (
            signal_engine(
                historical
            )
        )

        current = df.iloc[i]["close"]

        following = (
            df.iloc[i + 1]["close"]
        )

        if signal == "UP":

            result = (
                following > current
            )

        else:

            result = (
                following < current
            )

        if result:

            wins += 1
            outcome = "WIN"

        else:

            losses += 1
            outcome = "LOSS"

        rows.append(
            {
                "time": df.iloc[i]["datetime"],
                "signal": signal,
                "strength": strength,
            
