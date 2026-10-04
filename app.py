import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Live Trading Signal System",
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
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min",
    "30 Minutes": "30min",
    "1 Hour": "1h"
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")

pair = st.sidebar.selectbox(
    "Pair",
    PAIRS
)

timeframe_name = st.sidebar.selectbox(
    "Main Timeframe",
    list(TIMEFRAMES.keys()),
    index=0
)

timeframe = TIMEFRAMES[timeframe_name]

lookback = st.sidebar.slider(
    "Historical Candles",
    200,
    1000,
    500,
    100
)

run_button = st.sidebar.button(
    "🔄 Analyze Market",
    use_container_width=True
)


# =========================================================
# API DATA
# =========================================================

@st.cache_data(ttl=10)
def get_data(symbol, interval, outputsize):

    if not API_KEY:
        return pd.DataFrame(), "API key missing"

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        data = response.json()

        if "values" not in data:

            message = data.get(
                "message",
                "No market data received."
            )

            return pd.DataFrame(), message

        df = pd.DataFrame(
            data["values"]
        )

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna()

        df = df.sort_values(
            "datetime"
        ).reset_index(drop=True)

        return df, ""

    except Exception as e:

        return pd.DataFrame(), str(e)


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    data = df.copy()

    # EMA
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

    # RSI
    delta = data["close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    data["rsi"] = (
        100
        - (100 / (1 + rs))
    )

    # MACD
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

    # ATR
    previous_close = data["close"].shift(1)

    tr1 = (
        data["high"]
        - data["low"]
    )

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
        .ewm(span=14, adjust=False)
        .mean()
    )

    # Momentum
    data["momentum5"] = (
        data["close"]
        - data["close"].shift(5)
    )

    data["momentum10"] = (
        data["close"]
        - data["close"].shift(10)
    )

    # Candle
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
        / data["range"].replace(
            0,
            np.nan
        )
    )

    # Support / resistance
    data["recent_high"] = (
        data["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    data["recent_low"] = (
        data["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    # Structure
    data["high_change"] = (
        data["high"]
        - data["high"].shift(5)
    )

    data["low_change"] = (
        data["low"]
        - data["low"].shift(5)
    )

    # Volatility
    data["atr_average"] = (
        data["atr"]
        .rolling(30)
        .mean()
    )

    return data


# =========================================================
# MARKET REGIME
# =========================================================

def market_regime(row):

    if pd.isna(row["atr"]) or row["atr"] == 0:
        return "WARMING UP"

    ema_distance = abs(
        row["ema21"]
        - row["ema50"]
    )

    trend_strength = (
        ema_distance
        / row["atr"]
    )

    if trend_strength >= 1.20:

        if row["ema21"] > row["ema50"]:
            return "BULL TREND"

        return "BEAR TREND"

    if (
        not pd.isna(row["atr_average"])
        and row["atr"] > row["atr_average"] * 1.30
    ):
        return "HIGH VOLATILITY"

    return "SIDEWAYS"


# =========================================================
# SIGNAL ENGINE
# =========================================================

def calculate_signal(data):

    if len(data) < 80:
        return None

    row = data.iloc[-1]

    fields = [
        "ema5",
        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "momentum5",
        "momentum10",
        "body_ratio"
    ]

    for field in fields:

        if pd.isna(row[field]):
            return None

    up = 0.0
    down = 0.0

    up_reasons = []
    down_reasons = []

    regime = market_regime(row)

    # -----------------------------------------------------
    # 1. EMA TREND
    # -----------------------------------------------------

    if (
        row["ema5"] > row["ema9"]
        and row["ema9"] > row["ema21"]
        and row["ema21"] > row["ema50"]
    ):

        up += 2.5

        up_reasons.append(
            "EMA alignment bullish"
        )

    elif (
        row["ema5"] < row["ema9"]
        and row["ema9"] < row["ema21"]
        and row["ema21"] < row["ema50"]
    ):

        down += 2.5

        down_reasons.append(
            "EMA alignment bearish"
        )

    else:

        if row["ema5"] > row["ema21"]:

            up += 0.75

            up_reasons.append(
                "Short EMA above EMA21"
            )

        else:

            down += 0.75

            down_reasons.append(
                "Short EMA below EMA21"
            )

    # -----------------------------------------------------
    # 2. RSI
    # -----------------------------------------------------

    if 52 <= row["rsi"] <= 68:

        up += 1.5

        up_reasons.append(
            "RSI bullish zone"
        )

    elif 32 <= row["rsi"] < 48:

        down += 1.5

        down_reasons.append(
            "RSI bearish zone"
        )

    elif row["rsi"] > 68:

        down += 0.5

        down_reasons.append(
            "RSI extended high"
        )

    elif row["rsi"] < 32:

        up += 0.5

        up_reasons.append(
            "RSI extended low"
        )

    # -----------------------------------------------------
    # 3. MACD
    # -----------------------------------------------------

    if (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    ):

        up += 2.0

        up_reasons.append(
            "MACD bullish"
        )

    elif (
        row["macd"] < row["macd_signal"]
        and row["macd_hist"] < 0
    ):

        down += 2.0

        down_reasons.append(
            "MACD bearish"
        )

    # -----------------------------------------------------
    # 4. MOMENTUM
    # -----------------------------------------------------

    if (
        row["momentum5"] > 0
        and row["momentum10"] > 0
    ):

        up += 1.5

        up_reasons.append(
            "Momentum bullish"
        )

    elif (
        row["momentum5"] < 0
        and row["momentum10"] < 0
    ):

        down += 1.5

        down_reasons.append(
            "Momentum bearish"
        )

    # -----------------------------------------------------
    # 5. CANDLE STRENGTH
    # -----------------------------------------------------

    if row["body_ratio"] >= 0.55:

        if row["body"] > 0:

            up += 1.0

            up_reasons.append(
                "Strong bullish candle"
            )

        elif row["body"] < 0:

            down += 1.0

            down_reasons.append(
                "Strong bearish candle"
            )

    # -----------------------------------------------------
    # 6. BREAKOUT
    # -----------------------------------------------------

    if not pd.isna(row["recent_high"]):

        if row["close"] > row["recent_high"]:

            up += 2.0

            up_reasons.append(
                "Resistance breakout"
            )

        elif row["close"] < row["recent_low"]:

            down += 2.0

            down_reasons.append(
                "Support breakdown"
            )

    # -----------------------------------------------------
    # 7. PRICE STRUCTURE
    # -----------------------------------------------------

    if (
        row["high_change"] > 0
        and row["low_change"] > 0
    ):

        up += 1.0

        up_reasons.append(
            "Higher-high / higher-low structure"
        )

    elif (
        row["high_change"] < 0
        and row["low_change"] < 0
    ):

        down += 1.0

        down_reasons.append(
            "Lower-high / lower-low structure"
        )

    # -----------------------------------------------------
    # 8. MARKET REGIME ADAPTATION
    # -----------------------------------------------------

    if regime == "BULL TREND":

        if up >= down:

            up += 1.5

            up_reasons.append(
                "Bull trend confirmation"
            )

        else:

            down += 0.5

    elif regime == "BEAR TREND":

        if down >= up:

            down += 1.5

            down_reasons.append(
                "Bear trend confirmation"
            )

        else:

            up += 0.5

    elif regime == "SIDEWAYS":

        # In sideways markets, breakout/momentum
        # evidence gets more weight than trend.

        if row["momentum5"] > 0:

            up += 0.5

        elif row["momentum5"] < 0:

            down += 0.5

    # -----------------------------------------------------
    # FINAL DIRECTION
    # -----------------------------------------------------

    if up >= down:

        direction = "UP"

        winning_score = up

        losing_score = down

        reasons = up_reasons

    else:

        direction = "DOWN"

        winning_score = down

        losing_score = up

        reasons = down_reasons

    total = up + down

    if total <= 0:

        agreement = 50

    else:

        agreement = (
            winning_score
            / total
        ) * 100

    strength = int(
        min(
            99,
            max(
                50,
                agreement
            )
        )
    )

    return {
        "direction": direction,
        "strength": strength,
        "up_score": round(up, 2),
        "down_score": round(down, 2),
        "regime": regime,
        "price": float(row["close"]),
        "rsi": float(row["rsi"]),
        "macd": float(row["macd"]),
        "atr": float(row["atr"]),
        "reasons": reasons
    }


# =========================================================
# MULTI-TIMEFRAME CONFIRMATION
# =========================================================

def timeframe_signal(symbol, interval):

    df, error = get_data(
        symbol,
        interval,
        250
    )

    if df.empty:
        return None

    df = add_indicators(df)

    return calculate_signal(df)


def get_multi_timeframe(symbol, main_interval):

    results = {}

    intervals = [
        "1min",
        "5min",
        "15min"
    ]

    for interval in intervals:

        if interval == main_interval:

            continue

        result = timeframe_signal(
            symbol,
            interval
        )

        if result is not None:

            results[interval] = result

    return results


# =========================================================
# WALK FORWARD BACKTEST
# =========================================================

def walk_forward_test(data):

    if len(data) < 160:

        return {
            "wins": 0,
            "losses": 0,
            "accuracy": 0.0,
            "total": 0
        }

    wins = 0
    losses = 0

    start = 100

    end = len(data) - 1

    for i in range(start, end):

        training_data = data.iloc[:i].copy()

        signal = calculate_signal(
            training_data
        )

        if signal is None:
            continue

        current_close = float(
            data.iloc[i]["close"]
        )

        next_close = float(
            data.iloc[i + 1]["close"]
        )

        if signal["direction"] == "UP":

            correct = (
                next_close
                > current_close
            )

        else:

            correct = (
                next_close
                < current_close
            )

        if correct:
            wins += 1
        else:
            losses += 1

    total = wins + losses

    if total == 0:

        accuracy = 0.0

    else:

        accuracy = (
            wins / total
        ) * 100

    return {
        "wins": wins,
        "losses": losses,
        "accuracy": accuracy,
        "total": total
    }


# =========================================================
# HISTORICAL SIGNAL TEST
# =========================================================

def historical_test(data):

    if len(data) < 120:

        return {
            "wins": 0,
            "losses": 0,
            "accuracy": 0.0,
            "total": 0
        }

    wins = 0
    losses = 0

    start = 80

    for i in range(
        start,
        len(data) - 1
    ):

        sample = data.iloc[:i + 1]

        signal = calculate_signal(
            sample
        )

        if signal is None:
            continue

        current_close = float(
            data.iloc[i]["close"]
        )

        next_close = float(
            data.iloc[i + 1]["close"]
        )

        if signal["direction"] == "UP":

            if next_close > current_close:
                wins += 1
            else:
                losses += 1

        else:

            if next_close < current_close:
                wins += 1
            else:
                losses += 1

    total = wins + losses

    if total:

        accuracy = (
            wins / total
        ) * 100

    else:

        accuracy = 0.0

    return {
        "wins": wins,
        "losses": losses,
        "accuracy": accuracy,
        "total": total
    }


# =========================================================
# MAIN DATA
# =========================================================

if not API_KEY:

    st.error(
        "TWELVE_DATA_API_KEY Streamlit Secrets mein nahi mili."
    )

    st.stop()


with st.spinner(
    "Market data analyze ho raha hai..."
):

    main_df, error = get_data(
        pair,
        timeframe,
        lookback
    )


if main_df.empty:

    st.error(
        "Market data nahi mili."
    )

    st.code(error)

    st.stop()


main_df = add_indicators(
    main_df
)


# =========================================================
# CURRENT MARKET
# =========================================================

signal = calculate_signal(
    main_df
)

if signal is None:

    st.warning(
        "Indicators calculate karne ke liye "
        "aur candles chahiye."
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.subheader(
    pair
    + " — "
    + timeframe_name
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Live Price",
        f"{signal['price']:.6f}"
    )

with c2:

    st.metric(
        "Direction",
        signal["direction"]
    )

with c3:

    st.metric(
        "Signal Strength",
        str(signal["strength"])
        + "%"
    )

with c4:

    st.metric(
        "Market Regime",
        signal["regime"]
    )


# =========================================================
# BIG SIGNAL
# =========================================================

st.subheader(
    "🎯 Current Trading Direction"
)

if signal["direction"] == "UP":

    st.success(
        "🟢 UP"
    )

else:

    st.error(
        "🔴 DOWN"
    )


# =========================================================
# INDICATORS
# =========================================================

st.subheader(
    "📊 Technical Analysis"
)

a, b, c, d, e = st.columns(5)

with a:

    st.metric(
        "RSI",
        f"{signal['rsi']:.2f}"
    )

with b:

    st.metric(
        "MACD",
        f"{signal['macd']:.6f}"
    )

with c:

    st.metric(
        "ATR",
        f"{signal['atr']:.6f}"
    )

with d:

    st.metric(
        "UP Score",
        str(signal["up_score"])
    )

with e:

    st.metric(
        "DOWN Score",
        str(signal["down_score"])
    )


# =========================================================
# REASONS
# =========================================================

st.subheader(
    "🧠 Signal Reasons"
)

for reason in signal["reasons"]:

    st.write(
        "•",
        reason
    )


# =========================================================
# MULTI TIMEFRAME
# =========================================================

st.subheader(
    "🔄 Multi-Timeframe Confirmation"
)

with st.spinner(
    "Higher/lower timeframe confirmation..."
):

    mtf = get_multi_timeframe(
        pair,
        timeframe
    )

if mtf:

    mtf_rows = []

    for interval, result in mtf.items():

        mtf_rows.append(
            {
                "Timeframe": interval,
                "Direction": result["direction"],
                "Strength": str(
                    result["strength"]
                ) + "%",
                "Regime": result["regime"]
            }
        )

    mtf_df = pd.DataFrame(
        mtf_rows
    )

    st.dataframe(
        mtf_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Multi-timeframe data available nahi."
    )


# =========================================================
# HISTORICAL BACKTEST
# =========
