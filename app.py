    with info2:

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-label">
                    Timeframe
                </div>

                <div class="big-value">
                    {timeframe}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info3:

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-label">
                    Live Price
                </div>

                <div class="big-value">
                    {price:.6f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info4:

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="small-label">
                    Signal Time
                </div>

                <div class="big-value">
                    {datetime.now().strftime("%H:%M:%S")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # SIGNAL
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Current Trading Direction</div>',
        unsafe_allow_html=True
    )

    signal_class = (
        "signal-up"
        if signal == "UP"
        else "signal-down"
    )

    signal_text_class = (
        "signal-text-up"
        if signal == "UP"
        else "signal-text-down"
    )

    signal_icon = "🟢" if signal == "UP" else "🔴"

    st.markdown(
        f"""
        <div class="{signal_class}">

            <div class="signal-icon">
                {signal_icon}
            </div>

            <div class="{signal_text_class}">
                {signal}
            </div>

            <div class="signal-strength">
                Signal Strength: {strength}%
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # SIGNAL STRENGTH
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Signal Strength</div>',
        unsafe_allow_html=True
    )

    progress_class = (
        "progress-up"
        if signal == "UP"
        else "progress-down"
    )

    st.markdown(
        f"""
        <div class="glass-card">

            <div class="small-label">
                {signal} CONFIDENCE SCORE
            </div>

            <div class="big-value">
                {strength}%
            </div>

            <div class="progress-bg">

                <div
                    class="{progress_class}"
                    style="width:{strength}%;">
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # UP / DOWN SCORES
    # =====================================================

    st.markdown(
        '<div class="section-title">⚖️ Market Direction Scores</div>',
        unsafe_allow_html=True
    )

    score1, score2 = st.columns(2)

    total_score = up + down

    if total_score > 0:
        up_percent = up / total_score * 100
        down_percent = down / total_score * 100
    else:
        up_percent = 50
        down_percent = 50

    with score1:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    🟢 UP SCORE
                </div>

                <div class="big-value">
                    {up}
                </div>

                <div class="progress-bg">

                    <div
                        class="progress-up"
                        style="width:{up_percent:.1f}%;">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with score2:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    🔴 DOWN SCORE
                </div>

                <div class="big-value">
                    {down}
                </div>

                <div class="progress-bg">

                    <div
                        class="progress-down"
                        style="width:{down_percent:.1f}%;">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # TECHNICAL ANALYSIS
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Technical Analysis</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        rsi_display = (
            f"{rsi_value:.2f}"
            if pd.notna(rsi_value)
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    RSI
                </div>

                <div class="big-value">
                    {rsi_display}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with tech2:

        macd_display = (
            f"{macd_value:.6f}"
            if pd.notna(macd_value)
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    MACD
                </div>

                <div class="big-value">
                    {macd_display}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with tech3:

        momentum_display = (
            f"{momentum_value * 100:.4f}%"
            if pd.notna(momentum_value)
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    MOMENTUM
                </div>

                <div class="big-value">
                    {momentum_display}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # ANALYSIS REASONS
    # =====================================================

    st.markdown(
        '<div class="section-title">🧠 Market Analysis Reasons</div>',
        unsafe_allow_html=True
    )

    reason_html = ""

    for reason in reasons:

        reason_html += f"""
        <div class="reason">
            ✓ {reason}
        </div>
        """

    st.markdown(
        f"""
        <div class="glass-card">
            {reason_html}
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # BACKTEST
    # =====================================================

    st.markdown(
        '<div class="section-title">📚 Historical Backtest</div>',
        unsafe_allow_html=True
    )

    bt1, bt2, bt3, bt4 = st.columns(4)

    with bt1:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    ACCURACY
                </div>

                <div class="big-value">
                    {accuracy:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with bt2:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    WINS
                </div>

                <div class="big-value">
                    {wins}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with bt3:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    LOSSES
                </div>

                <div class="big-value">
                    {losses}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with bt4:

        tested = wins + losses

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="small-label">
                    TESTED SIGNALS
                </div>

                <div class="big-value">
                    {tested}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # BACKTEST CHART
    # =====================================================

    if not history.empty:

        st.markdown(
            '<div class="section-title">📉 Historical Signal Strength</div>',
            unsafe_allow_html=True
        )

        chart_data = history[
            ["time", "strength"]
        ].copy()

        chart_data = chart_data.set_index(
            "time"
        )

        st.line_chart(
            chart_data,
            use_container_width=True
        )


    # =====================================================
    # SIGNAL DETAILS
    # =====================================================

    st.markdown(
        '<div class="section-title">ℹ️ Signal Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="glass-card">

            <div class="reason">
                💱 Pair: <b>{pair}</b>
            </div>

            <div class="reason">
                ⏱️ Timeframe: <b>{timeframe}</b>
            </div>

            <div class="reason">
                💰 Current Price: <b>{price:.6f}</b>
            </div>

            <div class="reason">
                🎯 Direction: <b>{signal}</b>
            </div>

            <div class="reason">
                📊 Strength: <b>{strength}%</b>
            </div>

            <div class="reason">
                🕐 Analysis Time:
                <b>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</b>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "Signal strength indicator agreement ko show karta hai; "
        "ye guaranteed probability ya guaranteed profit nahi hai."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Live Trading Signal Bot • Technical Analysis Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
