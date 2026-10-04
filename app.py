import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Live Trading Signal System",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "pair" not in st.session_state:
    st.session_state.pair = "EUR/USD"

if "timeframe" not in st.session_state:
    st.session_state.timeframe = "1min"

if "data" not in st.session_state:
    st.session_state.data = None

if "signal" not in st.session_state:
    st.session_state.signal = None

if "signal_time" not in st.session_state:
    st.session_state.signal_time = None

if "signal_price" not in st.session_state:
    st.session_state.signal_price = None

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# API
# =========================================================

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")

BASE_URL = "https://api.twelvedata.com/time_series"


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


TIMEFRAMES = {
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min",
    "30 Minutes": "30min",
    "1 Hour": "1h"
}


# =========================================================
# GET DATA
# =========================================================

@st.cache_data(ttl=10)
def get_data(symbol, interval, outputsize=500):

    if not API_KEY:
        return None, "Twelve Data API key missing."

    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY,
            "format": "JSON"
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=20
        )

        data = response.json()

        if "values" not in data:
            return None, data.get("message", "No market data received.")

        df = pd.DataFrame(data["values"])

        required = ["datetime", "open", "high", "low", "close"]

        for col in required:
            if col not in df.columns:
                return None, "Required market data missing."

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna().copy()

        df = df.sort_values("datetime").reset_index(drop=True)

        return df, None

    except Exception as e:
        return None, str(e)


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):

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

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

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

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # True Range
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = df["tr"].rolling(14).mean()

    # Momentum
    df["momentum5"] = (
        df["close"] - df["close"].shift(5)
    )

    df["momentum10"] = (
        df["close"] - df["close"].shift(10)
    )

    # Candle
    df["body"] = (
        df["close"] - df["open"]
    )

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body_ratio"] = (
        df["body"].abs() /
        df["range"].replace(0, np.nan)
    )

    df["body_ratio"] = (
        df["body_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Recent support/resistance
    df["recent_high"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["recent_low"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    # Structure
    df["high_change"] = (
        df["high"] - df["high"].shift(5)
    )

    df["low_change"] = (
        df["low"] - df["low"].shift(5)
    )

    return df


# =========================================================
# MARKET REGIME
# =========================================================

def market_regime(df):

    last = df.iloc[-1]

    atr = float(last["atr"]) if pd.notna(last["atr"]) else 0
    avg_atr = (
        df["atr"].rolling(50).mean().iloc[-1]
        if len(df) >= 50
        else atr
    )

    if pd.isna(avg_atr):
        avg_atr = atr

    ema_gap = abs(
        last["ema21"] - last["ema50"]
    )

    if atr > avg_atr * 1.8:
        return "HIGH VOLATILITY"

    if (
        last["ema21"] > last["ema50"]
        and ema_gap > atr * 0.20
    ):
        return "BULL TREND"

    if (
        last["ema21"] < last["ema50"]
        and ema_gap > atr * 0.20
    ):
        return "BEAR TREND"

    if (
        last["ema5"] > last["ema9"]
        and last["ema9"] > last["ema21"]
    ):
        return "BULL MOMENTUM"

    if (
        last["ema5"] < last["ema9"]
        and last["ema9"] < last["ema21"]
    ):
        return "BEAR MOMENTUM"

    return "SIDEWAYS"


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(df):

    last = df.iloc[-1]

    up_score = 0.0
    down_score = 0.0

    reasons_up = []
    reasons_down = []

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if (
        last["ema5"] >
        last["ema9"] >
        last["ema21"] >
        last["ema50"]
    ):
        up_score += 2.0
        reasons_up.append("EMA alignment bullish")

    elif (
        last["ema5"] <
        last["ema9"] <
        last["ema21"] <
        last["ema50"]
    ):
        down_score += 2.0
        reasons_down.append("EMA alignment bearish")

    else:

        if last["ema5"] > last["ema9"]:
            up_score += 0.5

        else:
            down_score += 0.5

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    rsi = float(last["rsi"])

    if 50 < rsi < 70:
        up_score += 1.0
        reasons_up.append("RSI bullish zone")

    elif 30 < rsi < 50:
        down_score += 1.0
        reasons_down.append("RSI bearish zone")

    elif rsi >= 70:
        down_score += 0.5
        reasons_down.append("RSI overbought")

    elif rsi <= 30:
        up_score += 0.5
        reasons_up.append("RSI oversold")

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if (
        last["macd"] >
        last["macd_signal"]
        and last["macd_hist"] > 0
    ):
        up_score += 1.5
        reasons_up.append("MACD bullish")

    elif (
        last["macd"] <
        last["macd_signal"]
        and last["macd_hist"] < 0
    ):
        down_score += 1.5
        reasons_down.append("MACD bearish")

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    if (
        last["momentum5"] > 0
        and last["momentum10"] > 0
    ):
        up_score += 1.0
        reasons_up.append("Momentum bullish")

    elif (
        last["momentum5"] < 0
        and last["momentum10"] < 0
    ):
        down_score += 1.0
        reasons_down.append("Momentum bearish")

    # -----------------------------------------------------
    # CANDLE STRENGTH
    # -----------------------------------------------------

    body_ratio = float(last["body_ratio"])

    if body_ratio >= 0.55:

        if last["body"] > 0:
            up_score += 0.75
            reasons_up.append("Strong bullish candle")

        elif last["body"] < 0:
            down_score += 0.75
            reasons_down.append("Strong bearish candle")

    # -----------------------------------------------------
    # BREAKOUT
    # -----------------------------------------------------

    if pd.notna(last["recent_high"]):

        if last["close"] > last["recent_high"]:
            up_score += 1.25
            reasons_up.append("Resistance breakout")

    if pd.notna(last["recent_low"]):

        if last["close"] < last["recent_low"]:
            down_score += 1.25
            reasons_down.append("Support breakdown")

    # -----------------------------------------------------
    # PRICE STRUCTURE
    # -----------------------------------------------------

    if (
        last["high_change"] > 0
        and last["low_change"] > 0
    ):
        up_score += 1.0
        reasons_up.append("Higher high / higher low")

    elif (
        last["high_change"] < 0
        and last["low_change"] < 0
    ):
        down_score += 1.0
        reasons_down.append("Lower high / lower low")

    # -----------------------------------------------------
    # REGIME
    # -----------------------------------------------------

    regime = market_regime(df)

    if regime in [
        "BULL TREND",
        "BULL MOMENTUM"
    ]:
        up_score += 0.75
        reasons_up.append("Bullish market regime")

    elif regime in [
        "BEAR TREND",
        "BEAR MOMENTUM"
    ]:
        down_score += 0.75
        reasons_down.append("Bearish market regime")

    # -----------------------------------------------------
    # FINAL DIRECTION
    # -----------------------------------------------------

    if up_score >= down_score:
        direction = "UP"
    else:
        direction = "DOWN"

    total = up_score + down_score

    if total > 0:
        strength = (
            max(up_score, down_score) /
            total
        ) * 100
    else:
        strength = 50

    strength = max(
        50,
        min(95, strength)
    )

    return {
        "direction": direction,
        "up_score": round(up_score, 2),
        "down_score": round(down_score, 2),
        "strength": round(strength, 1),
        "regime": regime,
        "reasons_up": reasons_up,
        "reasons_down": reasons_down
    }


# =========================================================
# MTF ANALYSIS
# =========================================================

def get_mtf_signal(symbol):

    intervals = [
        ("1min", "1 Minute"),
        ("5min", "5 Minutes"),
        ("15min", "15 Minutes")
    ]

    results = []

    for interval, label in intervals:

        df, error = get_data(
            symbol,
            interval,
            250
        )

        if df is None:
            results.append({
                "Timeframe": label,
                "Direction": "ERROR",
                "Strength": 0
            })
            continue

        df = calculate_indicators(df)

        signal = generate_signal(df)

        results.append({
            "Timeframe": label,
            "Direction": signal["direction"],
            "Strength": signal["strength"]
        })

    return pd.DataFrame(results)


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(df):

    if len(df) < 120:
        return None

    wins = 0
    losses = 0

    start = 100
    end = len(df) - 1

    for i in range(start, end):

        sample = df.iloc[:i].copy()

        signal = generate_signal(sample)

        direction = signal["direction"]

        current_close = df.iloc[i]["close"]
        next_close = df.iloc[i + 1]["close"]

        if direction == "UP":
            correct = next_close > current_close
        else:
            correct = next_close < current_close

        if correct:
            wins += 1
        else:
            losses += 1

    total = wins + losses

    if total == 0:
        return None

    accuracy = (
        wins / total
    ) * 100

    return {
        "wins": wins,
        "losses": losses,
        "total": total,
        "accuracy": accuracy
    }


# =========================================================
# HEADER
# =========================================================

st.title("📊 Live Trading Signal System")

st.caption(
    "Real-time market analysis • UP/DOWN direction • "
    "Multi-strategy confirmation"
)


# =========================================================
# CONTROL PANEL
# =========================================================

st.subheader("🎛️ Trading Controls")

col1, col2, col3 = st.columns([1.3, 1.3, 1])

with col1:

    selected_pair = st.selectbox(
        "Currency Pair",
        PAIRS,
        index=PAIRS.index(
            st.session_state.pair
        )
    )

with col2:

    timeframe_names = list(TIMEFRAMES.keys())

    selected_name = st.selectbox(
        "Timeframe",
        timeframe_names,
        index=timeframe_names.index(
            next(
                name
                for name, value
                in TIMEFRAMES.items()
                if value == st.session_state.timeframe
            )
        )
    )

with col3:

    candles = st.number_input(
        "Historical Candles",
        min_value=200,
        max_value=1000,
        value=500,
        step=100
    )


st.session_state.pair = selected_pair
st.session_state.timeframe = TIMEFRAMES[selected_name]


# =========================================================
# CLICKABLE TIMEFRAME BUTTONS
# =========================================================

st.write("### ⏱️ Quick Timeframe")

tf1, tf2, tf3, tf4, tf5 = st.columns(5)

if tf1.button(
    "1 MIN",
    use_container_width=True
):
    st.session_state.timeframe = "1min"
    st.rerun()

if tf2.button(
    "5 MIN",
    use_container_width=True
):
    st.session_state.timeframe = "5min"
    st.rerun()

if tf3.button(
    "15 MIN",
    use_container_width=True
):
    st.session_state.timeframe = "15min"
    st.rerun()

if tf4.button(
    "30 MIN",
    use_container_width=True
):
    st.session_state.timeframe = "30min"
    st.rerun()

if tf5.button(
    "1 HOUR",
    use_container_width=True
):
    st.session_state.timeframe = "1h"
    st.rerun()


# =========================================================
# MAIN BUTTONS
# =========================================================

b1, b2, b3 = st.columns(3)

analyze_clicked = b1.button(
    "🔄 Analyze Market",
    use_container_width=True,
    type="primary"
)

signal_clicked = b2.button(
    "🎯 Generate New Signal",
    use_container_width=True
)

clear_clicked = b3.button(
    "🗑️ Clear",
    use_container_width=True
)


if clear_clicked:

    st.session_state.data = None
    st.session_state.signal = None
    st.session_state.signal_price = None
    st.session_state.signal_time = None

    st.rerun()


# =========================================================
# FETCH MAIN DATA
# =========================================================

if (
    analyze_clicked
    or signal_clicked
    or st.session_state.data is None
):

    with st.spinner("Market data analyze ho raha hai..."):

        df, error = get_data(
            st.session_state.pair,
            st.session_state.timeframe,
            int(candles)
        )

    if error:

        st.error(
            "Market data error: " + str(error)
        )

    else:

        df = calculate_indicators(df)

        st.session_state.data = df

        signal = generate_signal(df)

        st.session_state.signal = signal


# =========================================================
# DISPLAY SIGNAL
# =========================================================

if st.session_state.data is not None:

    df = st.session_state.data
    signal = st.session_state.signal

    last = df.iloc[-1]

    price = float(last["close"])

    direction = signal["direction"]

    strength = signal["strength"]

    regime = signal["regime"]


    st.divider()

    st.subheader(
        f"📈 {st.session_state.pair} — "
        f"{selected_name}"
    )


    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Price",
        f"{price:.6f}"
    )

    c2.metric(
        "Direction",
        direction
    )

    c3.metric(
        "Signal Strength",
        f"{strength:.1f}%"
    )

    c4.metric(
        "Market Regime",
        regime
    )


    # -----------------------------------------------------
    # BIG SIGNAL
    # -----------------------------------------------------

    st.subheader(
        "🎯 Current Trading Direction"
    )

    if direction == "UP":

        st.success(
            "🟢 UP\n\n"
            f"Signal Strength: {strength:.1f}%"
        )

    else:

        st.error(
            "🔴 DOWN\n\n"
            f"Signal Strength: {strength:.1f}%"
        )


    st.caption(
        "Signal Strength indicator agreement score hai, "
        "guaranteed win probability nahi."
    )


    # -----------------------------------------------------
    # GENERATE SIGNAL RECORD
    # -----------------------------------------------------

    if signal_clicked:

        st.session_state.signal_price = price

        st.session_state.signal_time = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        st.session_state.history.append({
            "Time": st.session_state.signal_time,
            "Pair": st.session_state.pair,
            "Timeframe": selected_name,
            "Price": price,
            "Direction": direction,
            "Strength": strength
        })

        st.success(
            f"New {direction} signal generated at "
            f"{price:.6f}"
        )


    # -----------------------------------------------------
    # TECHNICAL ANALYSIS
    # -----------------------------------------------------

    with st.expander(
        "📊 Technical Analysis",
        expanded=True
    ):

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "RSI",
            f"{last['rsi']:.2f}"
        )

        a2.metric(
            "MACD",
            f"{last['macd']:.6f}"
        )

        a3.metric(
            "ATR",
            f"{last['atr']:.6f}"
        )

        a4.metric(
            "Momentum",
            f"{last['momentum5']:.6f}"
        )


        st.write(
            f"**UP Score:** {signal['up_score']}"
        )

        st.write(
            f"**DOWN Score:** {signal['down_score']}"
        )


    # ----------------------------
