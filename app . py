import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(
    page_title="Live Market Signal",
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
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min"
}


# =========================================================
# DATA
# =========================================================

def get_market_data(symbol, interval, outputsize=200):

    if not API_KEY:
        return None, "API key missing."

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
            timeout=15
        )

        data = response.json()

        if "values" not in data:
            return None, data.get(
                "message",
                "Market data unavailable."
            )

        df = pd.DataFrame(
            data["values"]
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

    except Exception as e:

        return None, str(e)


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    close = df["close"]

    # EMA
    df["ema9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema21"] = close.ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
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
    fast = close.ewm(
        span=12,
        adjust=False
    ).mean()

    slow = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = fast - slow

    df["macd_signal"] = df[
        "macd"
    ].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # ATR
    previous_close = close.shift(1)

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
    ).abs()

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

    # Average volume is only used
    # when the data provider supplies it.
    if "volume" in df.columns:

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        )

        df["volume_avg"] = df[
            "volume"
        ].rolling(
            20
        ).mean()

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(df):

    df = add_indicators(
        df
    )

    if len(df) < 60:

        return {
            "signal": "UP",
            "score": 50,
            "bullish": 0,
            "bearish": 0,
            "reason": "Building market history."
        }

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    bullish = 0
    bearish = 0

    up_reasons = []
    down_reasons = []

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if (
        latest["ema9"]
        > latest["ema21"]
        > latest["ema50"]
    ):

        bullish += 3
        up_reasons.append(
            "EMA trend"
        )

    elif (
        latest["ema9"]
        < latest["ema21"]
        < latest["ema50"]
    ):

        bearish += 3
        down_reasons.append(
            "EMA trend"
        )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    rsi_value = latest["rsi"]

    if 52 <= rsi_value <= 68:

        bullish += 2
        up_reasons.append(
            "RSI momentum"
        )

    elif 32 <= rsi_value <= 48:

        bearish += 2
        down_reasons.append(
            "RSI momentum"
        )

    elif rsi_value < 30:

        bullish += 1
        up_reasons.append(
            "RSI oversold"
        )

    elif rsi_value > 70:

        bearish += 1
        down_reasons.append(
            "RSI overbought"
        )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if (
        latest["macd_hist"] > 0
        and latest["macd_hist"]
        > previous["macd_hist"]
    ):

        bullish += 2
        up_reasons.append(
            "MACD"
        )

    elif (
        latest["macd_hist"] < 0
        and latest["macd_hist"]
        < previous["macd_hist"]
    ):

        bearish += 2
        down_reasons.append(
            "MACD"
        )

    # -----------------------------------------------------
    # PRICE MOMENTUM
    # -----------------------------------------------------

    if (
        latest["close"]
        > previous["close"]
    ):

        bullish += 1
        up_reasons.append(
            "Price momentum"
        )

    elif (
        latest["close"]
        < previous["close"]
    ):

        bearish += 1
        down_reasons.append(
            "Price momentum"
        )

    # -----------------------------------------------------
    # BREAKOUT
    # -----------------------------------------------------

    previous_high = df[
        "high"
    ].iloc[-21:-1].max()

    previous_low = df[
        "low"
    ].iloc[-21:-1].min()

    if latest["close"] > previous_high:

        bullish += 3
        up_reasons.append(
            "Breakout"
        )

    elif latest["close"] < previous_low:

        bearish += 3
        down_reasons.append(
            "Breakdown"
        )

    # -----------------------------------------------------
    # CANDLE BODY
    # -----------------------------------------------------

    candle_body = abs(
        latest["close"]
        - latest["open"]
    )

    candle_range = (
        latest["high"]
        - latest["low"]
    )

    if candle_range > 0:

        body_ratio = (
            candle_body
            / candle_range
        )

        if body_ratio >= 0.60:

            if (
                latest["close"]
                > latest["open"]
            ):

                bullish += 1
                up_reasons.append(
                    "Strong bullish candle"
                )

            elif (
                latest["close"]
                < latest["open"]
            ):

                bearish += 1
                down_reasons.append(
                    "Strong bearish candle"
                )

    # -----------------------------------------------------
    # RECENT STRUCTURE
    # -----------------------------------------------------

    recent_close = df[
        "close"
    ].iloc[-6:]

    if (
        recent_close.iloc[-1]
        > recent_close.iloc[0]
    ):

        bullish += 1
        up_reasons.append(
            "Short-term structure"
        )

    elif (
        recent_close.iloc[-1]
        < recent_close.iloc[0]
    ):

        bearish += 1
        down_reasons.append(
            "Short-term structure"
        )

    # -----------------------------------------------------
    # FINAL DIRECTION
    # -----------------------------------------------------

    total = (
        bullish
        + bearish
    )

    if total == 0:

        signal = "UP"
        score = 50
        reason = (
            "Market direction is balanced; "
            "UP selected as the forced direction."
        )

    elif bullish >= bearish:

        signal = "UP"

        score = int(
            bullish
            / total
            * 100
        )

        reason = ", ".join(
            up_reasons
        )

        if not reason:

            reason = (
                "Bullish side has the stronger score."
            )

    else:

        signal = "DOWN"

        score = int(
            bearish
            / total
            * 100
        )

        reason = ", ".join(
            down_reasons
        )

        if not reason:

            reason = (
                "Bearish side has the stronger score."
            )

    return {
        "signal": signal,
        "score": score,
        "bullish": bullish,
        "bearish": bearish,
        "reason": reason
    }


# =========================================================
# UI
# =========================================================

st.title(
    "📊 Live Market Signal Bot"
)

st.caption(
    "Real market data • Multi-indicator analysis"
)

pair = st.selectbox(
    "Select Pair",
    PAIRS
)

timeframe_name = st.selectbox(
    "Select Timeframe",
    list(
        TIMEFRAMES.keys()
    )
)

interval = TIMEFRAMES[
    timeframe_name
]

analyze = st.button(
    "🚀 START ANALYZE",
    use_container_width=True
)


# =========================================================
# ANALYZE
# =========================================================

if analyze:

    status = st.empty()

    for seconds_left in range(
        5,
        0,
        -1
    ):

        status.markdown(
            f"# ⏱️ Live analysis: {seconds_left}"
        )

        time.sleep(1)

    status.empty()

    df, error = get_market_data(
        pair,
        interval,
        200
    )

    if error:

        st.error(
            f"Market data error: {error}"
        )

    else:

        result = generate_signal(
            df
        )

        st.divider()

        if result[
            "signal"
        ] == "UP":

            st.success(
                "# 🟢 UP"
            )

        else:

            st.error(
                "# 🔴 DOWN"
            )

        st.metric(
            "Current Price",
            f"{df['close'].iloc[-1]:.6f}"
        )

        st.write(
            f"**Pair:** {pair}"
        )

        st.write(
            f"**Timeframe:** "
            f"{timeframe_name}"
        )

        st.write(
            f"**Bullish Score:** "
            f"{result['bullish']}"
        )

        st.write(
            f"**Bearish Score:** "
            f"{result['bearish']}"
        )

        st.write(
            f"**Agreement Score:** "
            f"{result['score']}%"
        )

        st.info(
            f"Analysis: {result['reason']}"
        )

        st.caption(
            "Agreement score is a strategy score, "
            "not a guaranteed winning probability."
        )


# =========================================================
# STATUS
# =========================================================

st.divider()

st.write(
    f"**Selected Pair:** {pair}"
)

st.write(
    f"**Selected Timeframe:** {timeframe_name}"
)

st.caption(
    f"Last update: "
    f"{datetime.now().strftime('%H:%M:%S')}"
)
