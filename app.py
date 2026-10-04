import streamlit as st
import requests
import pandas as pd
import numpy as np
import websocket
import threading
import json
import time
from collections import defaultdict, deque
from datetime import datetime


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Live Trading Signal Bot",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# API KEY
# =========================================================

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")


# =========================================================
# PAIRS
# =========================================================

PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD"
]


# =========================================================
# TIMEFRAMES
# =========================================================

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

class TickCollector:

    def __init__(self, api_key, symbols):

        self.api_key = api_key
        self.symbols = symbols

        self.ticks = defaultdict(lambda: deque(maxlen=30000))

        self.lock = threading.Lock()

        self.ws = None
        self.thread = None
        self.running = False

        self.last_error = ""
        self.connected = False

        self.start()


    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()


    def _run(self):

        while self.running:

            try:

                url = (
                    "wss://ws.twelvedata.com/v1/quotes/price"
                    f"?apikey={self.api_key}"
                )

                self.ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )

                self.ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10
                )

            except Exception as e:

                self.last_error = str(e)
                self.connected = False

            time.sleep(5)


    def _on_open(self, ws):

        self.connected = True
        self.last_error = ""

        subscribe_message = {
            "action": "subscribe",
            "params": {
                "symbols": ",".join(self.symbols)
            }
        }

        ws.send(
            json.dumps(subscribe_message)
        )


    def _on_message(self, ws, message):

        try:

            data = json.loads(message)

            if data.get("event") != "price":
                return

            symbol = data.get("symbol")
            price = data.get("price")
            timestamp = data.get("timestamp")

            if symbol is None or price is None:
                return

            price = float(price)

            if timestamp is None:
                timestamp = time.time()

            timestamp = float(timestamp)

            with self.lock:

                self.ticks[symbol].append({
                    "timestamp": timestamp,
                    "price": price
                })

        except Exception as e:

            self.last_error = str(e)


    def _on_error(self, ws, error):

        self.connected = False
        self.last_error = str(error)


    def _on_close(self, ws, close_status_code, close_msg):

        self.connected = False


    def get_ticks(self, symbol):

        with self.lock:

            data = list(
                self.ticks.get(symbol, [])
            )

        if not data:
            return pd.DataFrame(
                columns=["timestamp", "price"]
            )

        return pd.DataFrame(data)


    def count(self, symbol):

        with self.lock:
            return len(self.ticks.get(symbol, []))


# =========================================================
# START ONE PERSISTENT COLLECTOR
# =========================================================

@st.cache_resource
def get_collector(api_key):

    return TickCollector(
        api_key,
        PAIRS
    )


# =========================================================
# HISTORICAL DATA
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

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df, None


# =========================================================
# BUILD SHORT CANDLES FROM LIVE TICKS
# =========================================================

def build_short_candles(tick_df, seconds):

    if tick_df is None or tick_df.empty:

        return pd.DataFrame()

    df = tick_df.copy()

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
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
        return pd.DataFrame()

    df = df.sort_values(
        "datetime"
    )

    df = df.set_index(
        "datetime"
    )

    rule = f"{seconds}s"

    candles = df["price"].resample(
        rule,
        label="left",
        closed="left"
    ).ohlc()

    candles = candles.dropna()

    candles = candles.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close"
        }
    )

    candles = candles.reset_index()

    return candles


# =========================================================
# INDICATORS
# =========================================================

def indicators(df):

    df = df.copy()

    df["ema9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["ema21"] = (
        df["close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

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

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

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

    df["atr"] = (
        tr.rolling(14).mean()
    )

    df["atr_avg"] = (
        df["atr"]
        .rolling(30)
        .mean()
    )

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

    if len(df) < 20:

        return (
            "UP",
            50.0,
            0,
            0,
            [
                "Collecting live candles..."
            ]
        )

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

    if pd.notna(last["rsi"]):

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
        pd.notna(last["macd"]) and
        pd.notna(last["macd_signal"])
    ):

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

    if pd.notna(last["momentum"]):

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

    if pd.notna(last["body_ratio"]):

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

    if len(df) >= 11:

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

        if last["close"] > recent_high:

            up += 2

            reasons.append(
                "Recent high breakout"
            )

        elif last["close"] < recent_low:

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
            [
                "No clear bias; forced direction"
            ]
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

    if len(df) < 65:

        return (
            0,
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

        current = (
            df.iloc[i]["close"]
        )

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
# APP
# =========================================================

st.title(
    "📊 Live Trading Signal Bot"
)

st.caption(
    "Real-time market analysis + live short-timeframe candles"
)


if not API_KEY:

    st.error(
        "Twelve Data API key is missing."
    )

    st.stop()


# Start persistent collector

collector = get_collector(
    API_KEY
)


# =========================================================
# SETTINGS
# =========================================================

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
# CONNECTION STATUS
# =========================================================

tick_count = collector.count(
    pair
)

if collector.connected:

    st.success(
        f"🟢 Live data connected — "
        f"{tick_count} ticks collected"
    )

else:

    st.warning(
        "🟡 Connecting to live market data..."
    )


# =========================================================
# ANALYZE
# =========================================================

if st.button(
    "🚀 START ANALYZE",
    use_container_width=True
):

    with st.spinner(
        "Analyzing live market..."
    ):

        # Give collector a moment to receive latest tick

        time.sleep(1)

        ticks = collector.get_ticks(
            pair
        )

        candles = build_short_candles(
            ticks,
            seconds
        )


        if candles.empty:

            st.error(
                "Live ticks abhi receive nahi ho rahe. "
                "10 seconds wait karke dobara START ANALYZE dabao."
            )

            st.stop()


        # Need enough candles for meaningful indicators

        if len(candles) < 20:

            st.warning(
                f"Live candles abhi {len(candles)} hain. "
                f"Kam az kam 20 candles collect hone dein."
            )

            st.info(
                "App background mein ticks collect kar raha hai. "
                "Thori der baad START ANALYZE dobara dabao."
            )

            st.stop()


        candles = indicators(
            candles
        )


        signal, strength, up, down, reasons = (
            signal_engine(
                candles
            )
        )


        accuracy, wins, losses, history = (
            backtest(
                candles
            )
        )


        price = (
            candles.iloc[-1]["close"]
        )


    # =====================================================
    # RESULT
    # =====================================================

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


    # =====================================================
    # LIVE CANDLE INFORMATION
    # =====================================================

    st.write(
        "### 🕯️ Live Candle Data"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Timeframe",
            timeframe_name
        )


    with c2:

        st.metric(
            "Live Candles",
            len(candles)
        )


    with c3:

        st.metric(
            "Live Ticks",
            len(ticks)
        )


    # =====================================================
    # BACKTEST
    # =====================================================

    st.divider()

    st.write(
        "### 🧪 Historical Backtest"
    )


    if len(history) > 0:

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
            f"Tested on {len(history)} "
            "short-timeframe signals."
        )

    else:

        st.info(
            "Is short timeframe ke liye "
            "abhi enough candles collect nahi hui "
            "ke historical backtest calculate ho sake."
        )


    # =====================================================
    # TIME
    # =====================================================

    st.write(
        "### 🕒 Signal Time"
    )

    st.write(
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    st.warning(
        "Signal strength indicator agreement hai, "
        "guaranteed probability nahi. Historical accuracy "
        "future result ki guarantee nahi deti."
    )


# =========================================================
# AUTO REFRESH BUTTON
# =========================================================

st.divider()

st.caption(
    "Live ticks background mein continuously collect ho rahe hain."
)

if st.button(
    "🔄 REFRESH LIVE DATA",
    use_container_width=True
):

    st.rerun()
