import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(
    page_title="Live Trading Signal Bot",
    page_icon="📊",
    layout="centered"
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
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min"
}


def get_data(symbol, interval, outputsize=500):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
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
            "Market data unavailable"
        )

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df, None


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

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

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
        df["high"] - df["close"].shift()
    )

    tr3 = abs(
        df["low"] - df["close"].shift()
    )

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    df["body"] = abs(
        df["close"] - df["open"]
    )

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, np.nan)
    )

    df["momentum"] = (
        df["close"].pct_change(3)
    )

    return df


def generate_signal(df):
    last = df.iloc[-1]

    up = 0
    down = 0

    reasons = []

    # EMA trend
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

    # Candle strength
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

    # Recent breakout
    recent_high = df[
        "high"
    ].iloc[-11:-1].max()

    recent_low = df[
        "low"
    ].iloc[-11:-1].min()

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

    total = up + down

    if total == 0:
        return (
            "UP",
            50.0,
            up,
            down,
            ["No clear bias"]
        )

    if up >= down:
        signal = "UP"
        strength = round(
            up / total * 100,
            1
        )

    else:
        signal = "DOWN"
        strength = round(
            down / total * 100,
            1
        )

    return (
        signal,
        strength,
        up,
        down,
        reasons
    )


def historical_backtest(df):

    wins = 0
    losses = 0

    for i in range(60, len(df) - 1):

        historical = df.iloc[
            :i + 1
        ].copy()

        signal, _, _, _, _ = (
            generate_signal(historical)
        )

        current_price = df.iloc[
            i
        ]["close"]

        next_price = df.iloc[
            i + 1
        ]["close"]

        if signal == "UP":
            win = next_price > current_price
        else:
            win = next_price < current_price

        if win:
            wins += 1
        else:
            losses += 1

    total = wins + losses

    if total == 0:
        accuracy = 0
    else:
        accuracy = (
            wins / total
        ) * 100

    return accuracy, wins, losses


def check_live_result():

    pending = st.session_state.get(
        "pending_signal"
    )

    if not pending:
        st.info(
            "No pending signal."
        )
        return

    signal_time = pending[
        "signal_time"
    ]

    elapsed = (
        datetime.now() -
        signal_time
    ).total_seconds()

    remaining = max(
        60 - int(elapsed),
        0
    )

    st.write(
        f"⏳ Remaining: "
        f"{remaining} seconds"
    )

    if remaining > 0:
        st.warning(
            "1-minute expiry complete hone ka wait karein."
        )
        return

    df, error = get_data(
        pending["pair"],
        "1min",
        5
    )

    if error:
        st.error(error)
        return

    result_price = df.iloc[-1]["close"]

    entry_price = pending[
        "entry_price"
    ]

    signal = pending[
        "signal"
    ]

    if signal == "UP":
        win = result_price > entry_price
    else:
        win = result_price < entry_price

    if win:
        result = "WIN"
        st.success(
            f"✅ {result}"
        )
    else:
        result = "LOSS"
        st.error(
            f"❌ {result}"
        )

    st.write(
        f"Entry Price: {entry_price:.6f}"
    )

    st.write(
        f"Result Price: {result_price:.6f}"
    )

    st.write(
        f"Direction: {signal}"
    )

    if "live_results" not in st.session_state:
        st.session_state.live_results = []

    st.session_state.live_results.append(
        {
            "time": signal_time.strftime(
                "%H:%M:%S"
            ),
            "signal": signal,
            "entry": entry_price,
            "result_price": result_price,
            "result": result
        }
    )

    st.session_state.pending_signal = None


st.title(
    "📊 Live Trading Signal Bot"
)

st.caption(
    "Live market analysis + "
    "1-minute signal tracking"
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

timeframe_name = st.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys())
)

interval = TIMEFRAMES[
    timeframe_name
]


if st.button(
    "🚀 START ANALYZE",
    use_container_width=True
):

    with st.spinner(
        "Analyzing live market..."
    ):

        df, error = get_data(
            pair,
            interval,
            500
        )

        if error:
            st.error(error)
            st.stop()

        df = add_indicators(df)

        (
            signal,
            strength,
            up_score,
            down_score,
            reasons
        ) = generate_signal(df)

        accuracy, wins, losses = (
            historical_backtest(df)
        )

        price = df.iloc[-1]["close"]

    st.session_state.pending_signal = {
        "pair": pair,
        "signal": signal,
        "entry_price": price,
        "signal_time": datetime.now()
    }

    st.success(
        "New signal generated."
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

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "UP Score",
            up_score
        )

    with col2:
        st.metric(
            "DOWN Score",
            down_score
        )

    st.write(
        "### 🧠 Signal Reasons"
    )

    for reason in reasons:
        st.write(
            "• " + reason
        )

    st.divider()

    st.write(
        "### 🧪 Historical Backtest"
    )

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

    st.divider()

    st.write(
        "### ⏱️ 1-Minute Live Result"
    )

    st.info(
        "Signal ke baad 1 minute wait karein, "
        "phir Check Result dabayein."
    )


if st.button(
    "🔎 CHECK 1-MINUTE RESULT",
    use_container_width=True
):

    check_live_result()


if st.session_state.get(
    "pending_signal"
):

    st.divider()

    pending = st.session_state[
        "pending_signal"
    ]

    elapsed = (
        datetime.now() -
        pending["signal_time"]
    ).total_seconds()

    remaining = max(
        60 - int(elapsed),
        0
    )

    st.write(
        f"⏳ Pending signal: "
        f"**{pending['signal']}**"
    )

    st.write(
        f"Entry price: "
        f"**{pending['entry_price']:.6f}**"
    )

    st.write(
        f"Time remaining: "
        f"**{remaining} seconds**"
    )


if st.session_state.get(
    "live_results"
):

    st.divider()

    st.write(
        "### 📋 Live Results"
    )

    results_df = pd.DataFrame(
        st.session_state.live_results
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    total = len(results_df)

    live_wins = len(
        results_df[
            results_df["result"] == "WIN"
        ]
    )

    live_accuracy = (
        live_wins / total * 100
        if total > 0
        else 0
    )

    st.metric(
        "Live Accuracy",
        f"{live_accuracy:.2f}%"
    )

st.caption(
    "Historical/live results are measurements, "
    "not guarantees of future performance."
    )
