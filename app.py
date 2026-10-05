import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Live Trading Signal Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min"
}

# =========================================================
# ANIMATED UI / CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 255, 170, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(0, 120, 255, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #050816 0%,
                #08111f 45%,
                #030711 100%
            );
        color: #f5f7ff;
    }

    /* Animated glow */
    .stApp::before {
        content: "";
        position: fixed;
        width: 450px;
        height: 450px;
        border-radius: 50%;
        background: rgba(0, 255, 170, 0.035);
        filter: blur(80px);
        top: 5%;
        left: -120px;
        animation: floatGlow 8s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::after {
        content: "";
        position: fixed;
        width: 400px;
        height: 400px;
        border-radius: 50%;
        background: rgba(50, 100, 255, 0.04);
        filter: blur(80px);
        bottom: 5%;
        right: -100px;
        animation: floatGlow2 10s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes floatGlow {
        from {
            transform: translateY(-20px) translateX(0);
        }
        to {
            transform: translateY(80px) translateX(80px);
        }
    }

    @keyframes floatGlow2 {
        from {
            transform: translateY(30px) translateX(0);
        }
        to {
            transform: translateY(-70px) translateX(-70px);
        }
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 10px 0 25px 0;
        position: relative;
        z-index: 2;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 1px;
        background: linear-gradient(
            90deg,
            #ffffff,
            #7affd5,
            #73a7ff,
            #ffffff
        );
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleGradient 6s ease infinite;
    }

    @keyframes titleGradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    .subtitle {
        color: #9aa8c2;
        font-size: 15px;
        margin-top: 5px;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(10, 20, 38, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 22px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.30),
            inset 0 1px 0 rgba(255,255,255,0.04);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(122, 255, 213, 0.20);
        box-shadow:
            0 15px 45px rgba(0, 0, 0, 0.38),
            0 0 25px rgba(0, 255, 170, 0.05);
    }

    .small-label {
        color: #8190aa;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 7px;
    }

    .big-value {
        font-size: 30px;
        font-weight: 750;
        color: #ffffff;
    }

    /* Live status */
    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 8px 14px;
        border-radius: 30px;
        background: rgba(0, 255, 150, 0.08);
        border: 1px solid rgba(0, 255, 150, 0.20);
        color: #7affc7;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .live-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #00ff99;
        box-shadow: 0 0 8px #00ff99;
        animation: pulseDot 1.2s infinite;
    }

    @keyframes pulseDot {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.5);
            opacity: 0.45;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }

    /* Signal cards */
    .signal-up {
        text-align: center;
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(
            145deg,
            rgba(0, 255, 140, 0.12),
            rgba(0, 100, 80, 0.08)
        );
        border: 1px solid rgba(0, 255, 150, 0.30);
        box-shadow: 0 0 35px rgba(0, 255, 150, 0.08);
        animation: signalPulse 2.2s infinite;
    }

    .signal-down {
        text-align: center;
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(
            145deg,
            rgba(255, 60, 90, 0.12),
            rgba(100, 20, 35, 0.08)
        );
        border: 1px solid rgba(255, 70, 90, 0.30);
        box-shadow: 0 0 35px rgba(255, 50, 80, 0.08);
        animation: signalPulseDown 2.2s infinite;
    }

    @keyframes signalPulse {
        0% {
            box-shadow: 0 0 20px rgba(0,255,150,0.05);
        }
        50% {
            box-shadow: 0 0 45px rgba(0,255,150,0.18);
        }
        100% {
            box-shadow: 0 0 20px rgba(0,255,150,0.05);
        }
    }

    @keyframes signalPulseDown {
        0% {
            box-shadow: 0 0 20px rgba(255,50,80,0.05);
        }
        50% {
            box-shadow: 0 0 45px rgba(255,50,80,0.18);
        }
        100% {
            box-shadow: 0 0 20px rgba(255,50,80,0.05);
        }
    }

    .signal-icon {
        font-size: 48px;
        margin-bottom: 5px;
    }

    .signal-text-up {
        color: #54ffc0;
        font-size: 38px;
        font-weight: 900;
        letter-spacing: 2px;
    }

    .signal-text-down {
        color: #ff6b82;
        font-size: 38px;
        font-weight: 900;
        letter-spacing: 2px;
    }

    .signal-strength {
        color: #aab6ca;
        font-size: 14px;
        margin-top: 8px;
    }

    /* Progress */
    .progress-bg {
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.07);
        border-radius: 20px;
        overflow: hidden;
        margin-top: 10px;
    }

    .progress-up {
        height: 100%;
        background: linear-gradient(
            90deg,
            #00c878,
            #66ffd0
        );
        border-radius: 20px;
        box-shadow: 0 0 14px rgba(0,255,160,0.30);
        transition: width 0.8s ease;
    }

    .progress-down {
        height: 100%;
        background: linear-gradient(
            90deg,
            #ff405d,
            #ff8798
        );
        border-radius: 20px;
        box-shadow: 0 0 14px rgba(255,50,80,0.30);
        transition: width 0.8s ease;
    }

    /* Reason cards */
    .reason {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 15px;
        margin: 7px 0;
        color: #dce4f4;
        transition: all 0.25s ease;
    }

    .reason:hover {
        background: rgba(255,255,255,0.065);
        transform: translateX(4px);
    }

    /* Section headings */
    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #f3f6ff;
        margin: 25px 0 13px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64738d;
        font-size: 12px;
        margin-top: 35px;
        padding: 15px;
    }

    /* Mobile */
    @media (max-width: 700px) {

        .main-title {
            font-size: 30px;
        }

        .big-value {
            font-size: 24px;
        }

        .signal-text-up,
        .signal-text-down {
            font-size: 31px;
        }

        .signal-icon {
            font-size: 38px;
        }

        .block-container {
            padding-left: 12px;
            padding-right: 12px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DATA FUNCTION
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
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = 100 - (
        100 / (1 + rs)
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
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(
        14
    ).mean()

    df["atr_avg"] = df["atr"].rolling(
        30
    ).mean()

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

    # Momentum
    df["momentum"] = (
        df["close"].pct_change(3)
    )

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def signal_engine(df):

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

        else:

            down += 2

            reasons.append(
                "Strong bearish candle"
            )

    # Recent structure
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

    total = up + down

    if total == 0:

        return (
            "UP",
            50.0,
            up,
            down,
            ["No clear bias; forced direction"]
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

        (
            signal,
            strength,
            up,
            down,
            reasons
        ) = signal_engine(
            historical
        )

        current = df.iloc[i]["close"]

        following = (
            df.iloc[i + 1]["close"]
        )

        if signal == "UP":

            result = (
                following >
                current
            )

        else:

            result = (
                following <
                current
            )

        if result:

            wins += 1
            outcome = "WIN"

        else:

            losses += 1
            outcome = "LOSS"

        rows.append({
            "time": df.iloc[i]["datetime"],
            "signal": signal,
            "strength": strength,
            "result": outcome
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
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">

        <div class="main-title">
            📊 LIVE TRADING SIGNAL
        </div>

        <div class="subtitle">
            Real-time market analysis • Technical indicators • Historical backtesting
        </div>

        <br>

        <div class="live-status">
            <span class="live-dot"></span>
            MARKET SYSTEM ONLINE
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# API CHECK
# =========================================================

if not API_KEY:

    st.error(
        "Twelve Data API key is missing."
    )

    st.stop()


# =========================================================
# CONTROLS
# =========================================================

st.markdown(
    '<div class="section-title">⚙️ Market Settings</div>',
    unsafe_allow_html=True
)

control1, control2 = st.columns(2)

with control1:

    pair = st.selectbox(
        "Trading Pair",
        PAIRS
    )

with control2:

    timeframe = st.selectbox(
        "Timeframe",
        list(TIMEFRAMES.keys())
    )


# =========================================================
# START BUTTON
# =========================================================

start = st.button(
    "🚀  START MARKET ANALYSIS",
    use_container_width=True
)


if start:

    with st.spinner(
        "Scanning market conditions..."
    ):

        df, error = get_data(
            pair,
            TIMEFRAMES[timeframe]
        )

        if error:

            st.error(error)
            st.stop()

        if df is None or len(df) < 2:

            st.error(
                "Not enough market data available."
            )

            st.stop()

        df = indicators(df)

        (
            signal,
            strength,
            up,
            down,
            reasons
        ) = signal_engine(df)

        (
            accuracy,
            wins,
            losses,
            history
        ) = backtest(df)

        price = df.iloc[-1]["close"]

        rsi_value = df.iloc[-1]["rsi"]

        macd_value = df.iloc[-1]["macd"]

        momentum_value = df.iloc[-1]["momentum"]


    # =====================================================
    # MARKET INFO
    # =====================================================

    st.markdown(
        '<div class="section-title">📡 Market Overview</div>',
        unsafe_allow_html=True
    )

    info1, info2, info3, info4 = st.columns(4)

    with info1:

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-label">
                    Trading Pair
                </div>

                <div class="big-value">
                    {pair}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-label">
                    Timeframe
                </div>

              
