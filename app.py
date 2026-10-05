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
def get_tick_collector(api_key):

    class TickCollector:

        def __init__(self):
            self.ticks = defaultdict(lambda: deque(maxlen=10000))
            self.connected = False
            self.lock = threading.Lock()
            self.ws = None
            self.thread = None

            self.start()

        def on_message(self, ws, message):

            try:
                data = json.loads(message)

                symbol = data.get("symbol")
                price = data.get("price")

                if symbol is None or price is None:
                    return

                try:
                    price = float(price)
                except:
                    return

                timestamp = data.get("timestamp")

                if timestamp is not None:
                    try:
                        ts = pd.to_datetime(
                            float(timestamp),
                            unit="s"
                        )
                    except:
                        ts = pd.Timestamp.utcnow()
                else:
                    ts = pd.Timestamp.utcnow()

                with self.lock:
                    self.ticks[symbol].append(
                        {
                            "datetime": ts,
                            "price": price
                        }
                    )

            except Exception:
                pass

        def on_open(self, ws):

            self.connected = True

            subscribe_message = {
                "action": "subscribe",
                "params": {
                    "symbols": ",".join(PAIRS)
                }
            }

            try:
                ws.send(json.dumps(subscribe_message))
            except:
                pass

        def on_close(self, ws, close_status_code, close_msg):
            self.connected = False

        def on_error(self, ws, error):
            self.connected = False

        def run(self):

            while True:

                try:

                    self.ws = websocket.WebSocketApp(
                        "wss://ws.twelvedata.com/v1/quotes/price?apikey="
                        + api_key,
                        on_open=self.on_open,
                        on_message=self.on_message,
                        on_close=self.on_close,
                        on_error=self.on_error
                    )

                    self.ws.run_forever(
                        ping_interval=20,
                        ping_timeout=10
                    )

                except Exception:
                    pass

                self.connected = False

                time.sleep(3)

        def start(self):

            if self.thread is None:

                self.thread = threading.Thread(
                    target=self.run,
                    daemon=True
                )

                self.thread.start()

        def get_ticks(self, symbol):

            with self.lock:
                return list(self.ticks[symbol])

    return TickCollector()


# =========================================================
# BUILD SHORT TIMEFRAME CANDLES
# =========================================================

def build_short_candles(collector, symbol, seconds):

    ticks = collector.get_ticks(symbol)

    if len(ticks) < 2:
        return None

    df = pd.DataFrame(ticks)

    if df.empty:
        return None

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
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

    df = df.sort_values("datetime")

    df = df.set_index("datetime")

    rule = f"{seconds}s"

    candles = df["price"].resample(
        rule,
        label="left",
        closed="left"
    ).ohlc()

    candles = candles.dropna()

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
# ORIGINAL 1 MINUTE DATA
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

    df = pd.DataFrame(data["values"])

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

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

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

    df["rsi"] = 100 - (
        100 / (1 + rs)
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

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    tr1 = df["high"] - df["low"]

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
    )

    df["body_ratio"] = (
        df["body_ratio"]
        .fillna(0)
    )

    df["momentum"] = (
        df["close"].pct_change(3)
    )

    df["momentum"] = (
        df["momentum"].fillna(0)
    )

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def signal_engine(df):

    if df is None or len(df) < 2:

        return (
            "UP",
            50.0,
            0,
            0,
            ["Waiting for live candle data"]
        )

    last = df.iloc[-1]

    up = 0
    down = 0

    reasons = []

    # -----------------------------------------------------
    # TREND
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

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if (
        pd.notna(last["macd"])
        and
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
    # CANDLE STRENGTH
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

        recent_high = df[
            "high"
        ].iloc[-11:-1].max()

        recent_low = df[
            "low"
        ].iloc[-11:-1].min()

        if pd.notna(recent_high):

            if last["close"] > recent_high:

                up += 2

                reasons.append(
                    "Recent high breakout"
                )

        if pd.notna(recent_low):

            if last["close"] < recent_low:

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

        if last["atr"] >= (
            last["atr_avg"] * 0.75
        ):

            if up > down:

                up += 1

            elif down > up:

                down += 1

    # -----------------------------------------------------
    # ALWAYS RETURN UP OR DOWN
    # -----------------------------------------------------

    total = up + down

    if total == 0:

        if last["close"] >= last["open"]:

            return (
                "UP",
                50.0,
                up,
                down,
                [
                    "Bullish candle bias"
                ]
            )

        else:

            return (
                "DOWN",
                50.0,
                up,
                down,
                [
                    "Bearish candle bias"
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

    if df is None or len(df) < 62:

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

        historical = df.iloc[
            :i + 1
        ].copy()

        signal, strength, up, down, reasons = (
            signal_engine(historical)
        )

        current = df.iloc[i]["close"]

        following = df.iloc[
            i + 1
        ]["close"]

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
                "result": outcome
            }
        )

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

timeframe = st.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys())
)


# Start live collector once
collector = get_tick_collector(
    API_KEY
)


if st.button(
    "🚀 START ANALYZE",
    use_container_width=True
):

    seconds = TIMEFRAMES[timeframe]

    # =====================================================
    # 1 MINUTE
    # =====================================================

    if seconds == 60:

        with st.spinner(
            "Analyzing 1 Minute market..."
        ):

            df, error = get_data(
                pair,
                "1min"
            )

            if error:

                st.error(error)
                st.stop()

            df = indicators(df)

            signal, strength, up, down, reasons = (
                signal_engine(df)
            )

            accuracy, wins, losses, history = (
                backtest(df)
            )

            price = df.iloc[-1]["close"]


    # =====================================================
    # SHORT TIMEFRAME
    # =====================================================

    else:

        with st.spinner(
            f"Analyzing live {seconds}-second market..."
        ):

            df = build_short_candles(
                collector,
                pair,
                seconds
            )

            if df is None or len(df) < 2:

                st.warning(
                    "Live tick data abhi collect ho raha hai. "
                    "Kuch seconds baad START ANALYZE dobara dabao."
                )

                st.stop()

            df = indicators(df)

            signal, strength, up, down, reasons = (
                signal_engine(df)
            )

            # Backtest sirf tab jab enough real candles hon
            accuracy, wins, losses, history = (
                backtest(df)
            )

            price = df.iloc[-1]["close"]


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

    if reasons:

        for reason in reasons:

            st.write(
                "•",
                reason
            )

    st.divider()

    st.write(
        "### 🧪 Historical Backtest"
    )

    if accuracy is not None:

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

    else:

        st.info(
            "Short timeframe ka historical backtest "
            "real live candles collect hone ke baad "
            "available hoga."
        )

    st.write(
        "### 🕒 Signal Time"
    )

    st.write(
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    st.caption(
        f"Live candles available: {len(df)}"
    )

    st.warning(
        "Historical accuracy future results ki guarantee nahi hai."
)
