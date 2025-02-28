import streamlit as st
import yfinance as yf
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Stonks", layout="wide")


##############################
# Indicator Computations
##############################
def sma(series, window=20):
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def bollinger_bands(series, window=20, num_std=2):
    """Compute Bollinger Bands: upper, middle, lower."""
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    return upper, mid, lower


def rsi(series, period=14):
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD = EMA(fastperiod) - EMA(slowperiod)."""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def cci(close_series, high_series, low_series, period=20):
    """Commodity Channel Index."""
    typical_price = (high_series + low_series + close_series) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_dev = (typical_price - sma_tp).abs().rolling(window=period).mean()
    return (typical_price - sma_tp) / (0.015 * mean_dev)


def roc(series, period=10):
    """Rate of Change."""
    shifted = series.shift(period)
    return ((series - shifted) / shifted) * 100


##############################
# Formatting & Indicator Arrows
##############################
def format_number(num):
    """Format large numbers with K, M, or B suffix."""
    try:
        num = float(num)
    except (ValueError, TypeError):
        return "N/A"
    if abs(num) >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def get_indicator(metric_name, financials):
    """
    Return a small arrow (green ▲ if current > previous, red ▼ if current < previous)
    based on the metric's values in the latest two columns of the financials DataFrame.
    """
    if metric_name in financials.index and financials.shape[1] > 1:
        try:
            current_val = float(financials.loc[metric_name].iloc[0])
            previous_val = float(financials.loc[metric_name].iloc[1])
        except (ValueError, TypeError):
            return ""
        if current_val > previous_val:
            return "<span style='color:green; font-size:100%'>&#9650;</span>"
        elif current_val < previous_val:
            return "<span style='color:red; font-size:100%'>&#9660;</span>"
        else:
            return ""
    return ""


##############################
# Price Targets Function
##############################
def display_price_targets(info_data):
    """
    Displays a dark-themed Plotly chart showing Low, Current, Average, and High
    analyst price targets on a horizontal line, with 'card-like' callouts for
    Current and Average to match the reference screenshot.
    """

    st.markdown(
        "<h3 style='color:#fff; background:#1b1b1b; padding:5px; border-radius:8px;"
        "margin-bottom:15px;'>Analyst Price Targets</h3>",
        unsafe_allow_html=True
    )

    # Extract relevant data from the info dictionary
    target_mean = info_data.get("targetMeanPrice")
    target_high = info_data.get("targetHighPrice")
    target_low = info_data.get("targetLowPrice")
    current_price = info_data.get("currentPrice")

    # Check if the essential data exists
    if not target_mean or not current_price:
        st.write("No price target data available.")
        return

    # Provide fallback if low/high are missing
    low_value = target_low if target_low else current_price
    high_value = target_high if target_high else current_price

    # Prepare a small buffer so markers are not right at the edges
    chart_min = min(low_value, high_value, current_price, target_mean) * 0.95
    chart_max = max(low_value, high_value, current_price, target_mean) * 1.05

    # Create a Plotly figure
    fig = go.Figure()

    # 1) Add a horizontal line (shape) from low_value to high_value
    fig.add_shape(
        type="line",
        x0=low_value,
        y0=0,
        x1=high_value,
        y1=0,
        line=dict(color="#888", width=3),
    )

    # 2) Low marker + text
    fig.add_trace(go.Scatter(
        x=[low_value],
        y=[0],
        mode="markers+text",
        text=[f"${low_value:,.2f} Low"],
        textposition="bottom center",
        marker=dict(color="white", size=10),
        name="Low"
    ))

    # 3) High marker + text
    fig.add_trace(go.Scatter(
        x=[high_value],
        y=[0],
        mode="markers+text",
        text=[f"${high_value:,.2f} High"],
        textposition="bottom center",
        marker=dict(color="white", size=10),
        name="High"
    ))

    # 4) Current marker (no inline text, we'll use an annotation)
    fig.add_trace(go.Scatter(
        x=[current_price],
        y=[0],
        mode="markers",
        marker=dict(color="white", size=10),
        name="Current"
    ))

    # 5) Average marker (no inline text, we'll use an annotation)
    fig.add_trace(go.Scatter(
        x=[target_mean],
        y=[0],
        mode="markers",
        marker=dict(color="#4c9cff", size=10),
        name="Average"
    ))

    # 6) Add "card-like" annotations for Current (below) and Average (above)

    # -- Current annotation (below the line)
    fig.add_annotation(
        x=current_price,
        y=-0.13,  # offset below the line
        xanchor="center",
        yanchor="top",
        showarrow=False,
        align="center",
        text=(
            f"<span style='font-weight:bold;'>${current_price:,.2f}</span><br>"
            f"<span style='font-size:11px;'>Current</span>"
        ),
        font=dict(color="white"),
        bordercolor="white",
        borderwidth=1,
        borderpad=5,
        bgcolor="#1b1b1b",
        opacity=1
    )

    # -- Average annotation (above the line)
    fig.add_annotation(
        x=target_mean,
        y=0.13,  # offset above the line
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        align="center",
        text=(
            f"<span style='font-weight:bold;'>${target_mean:,.2f}</span><br>"
            f"<span style='font-size:11px;'>Average</span>"
        ),
        font=dict(color="white"),
        bordercolor="#4c9cff",
        borderwidth=1,
        borderpad=5,
        bgcolor="#1b1b1b",
        opacity=1
    )

    # Configure the layout for a clean, dark look
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            range=[chart_min, chart_max],
            showgrid=False,
            showline=False,
            zeroline=False,
            visible=False  # hides axis numbers and lines
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            visible=False,
            # Expand the y-range so we have room for annotations above/below
            range=[-0.3, 0.3],
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="#1b1b1b",  # match your dark background
        paper_bgcolor="#1b1b1b",
        showlegend=False,
        height=300
    )

    # Render the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


##############################
# Main Streamlit App
##############################
def main():
    st.title("Stonks")

    # Sidebar: inputs for ticker, date range, and timeframe
    st.sidebar.header("Select Ticker & Date Range")
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.now().date())

    # Timeframe dropdown with only daily and weekly options
    timeframe_options = ["daily", "weekly"]
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", timeframe_options)
    timeframe_mapping = {"daily": "1d", "weekly": "1wk"}
    interval = timeframe_mapping[selected_timeframe]

    # Multi-select for technical indicators
    st.sidebar.header("Select Indicators")
    all_indicators = ["SMA (20)", "Bollinger Bands", "RSI", "MACD", "CCI", "ROC"]
    selected_indicators = st.sidebar.multiselect("Choose one or more:", all_indicators, default=[])

    # Fetch Data button
    if st.sidebar.button("Fetch Data"):
        # ------------------------
        # 1) Fetch chart data from yfinance
        # ------------------------
        chart_df = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df.columns = chart_df.columns.get_level_values(0)
        if chart_df.empty:
            st.warning("No data found for the specified range.")
            return
        chart_df.dropna(inplace=True)
        chart_df.index = pd.to_datetime(chart_df.index)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        chart_df = chart_df[numeric_cols].copy()
        chart_df.dropna(inplace=True)

        # 2) Fetch additional data for overview, holders, analysis
        ticker_obj = yf.Ticker(ticker_symbol)
        financials = ticker_obj.financials
        if financials is not None and not financials.empty:
            latest_col = financials.columns[0]
            year = str(latest_col.year) if hasattr(latest_col, "year") else str(latest_col)
            ebitda = financials.loc["EBITDA"].iloc[0] if "EBITDA" in financials.index else None
            net_income = financials.loc["Net Income"].iloc[0] if "Net Income" in financials.index else None
            operating_income = financials.loc["Operating Income"].iloc[
                0] if "Operating Income" in financials.index else None
            total_revenue = financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in financials.index else None

            if "Operating Expenses" in financials.index:
                operating_expenses = financials.loc["Operating Expenses"].iloc[0]
            elif total_revenue is not None and operating_income is not None:
                operating_expenses = total_revenue - operating_income
            else:
                operating_expenses = None

            operating_revenue = financials.loc["Operating Revenue"].iloc[
                0] if "Operating Revenue" in financials.index else total_revenue
        else:
            year = "N/A"
            ebitda = net_income = operating_income = operating_expenses = total_revenue = operating_revenue = None

        # 3) Create 4 tabs: Company Overview, Chart(s), Institutional Holders, and Analysis
        tabs = st.tabs(["Company Overview", "Chart(s)", "Institutional Holders", "Analysis"])

        ###################################################################
        # TAB 1: Company Overview
        ###################################################################
        with tabs[0]:
            st.header("Company Overview")

            # First row of three columns
            row1 = st.columns(3, gap="medium")
            # Second row of three columns
            row2 = st.columns(3, gap="medium")

            tile_style = (
                "border:1px solid #ddd; padding:10px; border-radius:5px; "
                "text-align:center; margin:10px; height:120px; "
                "display:flex; flex-direction:column; justify-content:center;"
            )

            with row1[0]:
                st.markdown(
                    f"<div style='{tile_style}'>"
                    f"<strong style='font-size:120%;'>EBITDA <sub style='font-size:60%;'>{year}</sub></strong><br>"
                    f"{format_number(ebitda)} {get_indicator('EBITDA', financials)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with row1[1]:
                st.markdown(
                    f"<div style='{tile_style}'>"
                    f"<strong style='font-size:120%;'>Net Income <sub style='font-size:60%;'>{year}</sub></strong><br>"
                    f"{format_number(net_income)} {get_indicator('Net Income', financials)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with row1[2]:
                st.markdown(
                    f"<div style='{tile_style}'>"
                    f"<strong style='font-size:120%;'>Operating Income <sub style='font-size:60%;'>{year}</sub></strong><br>"
                    f"{format_number(operating_income)} {get_indicator('Operating Income', financials)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with row2[0]:
                st.markdown(
                    f"<div style='{tile_style}'>"
                    f"<strong style='font-size:120%;'>Operating Expenses <sub style='font-size:60%;'>{year}</sub></strong><br>"
                    f"{format_number(operating_expenses)} {get_indicator('Operating Expenses', financials)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with row2[1]:
                st.markdown(
                    f"<div style='{tile_style}'>"
                    f"<strong style='font-size:120%;'>Operating Revenue <sub style='font-size:60%;'>{year}</sub></strong><br>"
                    f"{format_number(operating_revenue)} {get_indicator('Operating Revenue', financials)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with row2[2]:
                st.markdown(
                    f"<div style='{tile_style}'>"
                    f"<strong style='font-size:120%;'>Total Revenue <sub style='font-size:60%;'>{year}</sub></strong><br>"
                    f" {format_number(total_revenue)} {get_indicator('Total Revenue', financials)} "
                    f"</div>",
                    unsafe_allow_html=True
                )

        ###################################################################
        # TAB 2: Chart(s) with Indicators (Synced Zoom/Pan)
        ###################################################################
        with tabs[1]:
            st.header("Interactive Chart with Synced Zoom/Pan")
            # Prepare chart data
            chart_df.reset_index(inplace=True)
            chart_df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)

            # Compute selected indicators
            if "SMA (20)" in selected_indicators:
                chart_df["sma20"] = sma(chart_df["close"], window=20)
            if "Bollinger Bands" in selected_indicators:
                upper, mid, lower = bollinger_bands(chart_df["close"], window=20, num_std=2)
                chart_df["bb_upper"] = upper
                chart_df["bb_mid"] = mid
                chart_df["bb_lower"] = lower
            if "RSI" in selected_indicators:
                chart_df["rsi"] = rsi(chart_df["close"], period=14)
            if "MACD" in selected_indicators:
                macd_line, signal_line, hist = macd(chart_df["close"])
                chart_df["macd_line"] = macd_line
                chart_df["macd_signal"] = signal_line
                chart_df["macd_hist"] = hist
            if "CCI" in selected_indicators:
                chart_df["cci"] = cci(chart_df["close"], chart_df["high"], chart_df["low"])
            if "ROC" in selected_indicators:
                chart_df["roc"] = roc(chart_df["close"])

            # Price-based chart (top panel)
            base = alt.Chart(chart_df).encode(x="date:T")
            rule = base.mark_rule().encode(
                y="low:Q",
                y2="high:Q"
            )
            bar = base.mark_bar().encode(
                y="open:Q",
                y2="close:Q",
                color=alt.condition("datum.open <= datum.close", alt.value("green"), alt.value("red"))
            )
            layers = [rule, bar]
            if "SMA (20)" in selected_indicators:
                sma_line = base.mark_line(color="blue").encode(y="sma20:Q")
                layers.append(sma_line)
            if "Bollinger Bands" in selected_indicators:
                bb_upper_line = base.mark_line(color="orange").encode(y="bb_upper:Q")
                bb_mid_line = base.mark_line(color="gray").encode(y="bb_mid:Q")
                bb_lower_line = base.mark_line(color="orange").encode(y="bb_lower:Q")
                layers += [bb_upper_line, bb_mid_line, bb_lower_line]

            price_chart = alt.layer(*layers).properties(
                width=900,
                height=400,
                title=f"{ticker_symbol.upper()} Candlestick Chart ({selected_timeframe})"
            )

            # Oscillator charts (lower panels)
            osc_charts = []
            if "RSI" in selected_indicators:
                rsi_chart = alt.Chart(chart_df).mark_line(color="purple").encode(
                    x="date:T",
                    y=alt.Y("rsi:Q", title="RSI (14)")
                ).properties(width=900, height=120, title="RSI")
                osc_charts.append(rsi_chart)

            if "MACD" in selected_indicators:
                macd_base = alt.Chart(chart_df).encode(x="date:T")
                macd_line_chart = macd_base.mark_line(color="blue").encode(y=alt.Y("macd_line:Q", title="MACD"))
                signal_line_chart = macd_base.mark_line(color="red").encode(y="macd_signal:Q")
                hist_chart = macd_base.mark_bar(color="green").encode(y="macd_hist:Q")
                macd_layer = alt.layer(macd_line_chart, signal_line_chart, hist_chart).properties(
                    width=900, height=120, title="MACD (12,26,9)"
                )
                osc_charts.append(macd_layer)

            if "CCI" in selected_indicators:
                cci_chart = alt.Chart(chart_df).mark_line(color="orange").encode(
                    x="date:T",
                    y=alt.Y("cci:Q", title="CCI")
                ).properties(width=900, height=120, title="CCI")
                osc_charts.append(cci_chart)

            if "ROC" in selected_indicators:
                roc_chart = alt.Chart(chart_df).mark_line(color="green").encode(
                    x="date:T",
                    y=alt.Y("roc:Q", title="ROC (10)")
                ).properties(width=900, height=120, title="ROC")
                osc_charts.append(roc_chart)

            if osc_charts:
                final_chart = alt.vconcat(price_chart, *osc_charts).resolve_scale(x='shared').interactive()
            else:
                final_chart = price_chart.interactive()

            st.altair_chart(final_chart, use_container_width=True)

        ###################################################################
        # TAB 3: Institutional Holders
        ###################################################################
        with tabs[2]:
            st.header("Institutional Holders")
            info = ticker_obj.info

            # Major Holders Breakdown
            held_insiders = info.get("heldPercentInsiders")
            held_institutions = info.get("heldPercentInstitutions")
            num_institutions = info.get("institutionCount") or info.get("numberOfInstitutions")
            if held_insiders is not None and held_institutions is not None:
                pct_insiders = f"{held_insiders * 100:.2f}%"
                pct_institutions = f"{held_institutions * 100:.2f}%"
                if held_insiders < 1:
                    pct_float_institutions = f"{(held_institutions / (1 - held_insiders)) * 100:.2f}%"
                else:
                    pct_float_institutions = "N/A"

                st.subheader("Major Holders Breakdown")
                st.write(f"% of Shares Held by All Insider: {pct_insiders}")
                st.write(f"% of Shares Held by Institutions: {pct_institutions}")
                st.write(f"% of Float Held by Institutions: {pct_float_institutions}")
                if num_institutions is not None:
                    st.write(f"Number of Institutions Holding Shares: {num_institutions}")
            else:
                st.write("No breakdown data available.")

            # Top Institutional Holders
            st.subheader("Top Institutional Holders")
            inst_df = ticker_obj.institutional_holders
            if inst_df is None or inst_df.empty:
                st.write("No institutional holders data available.")
            else:
                st.dataframe(inst_df)

            # Top Mutual Fund Holders
            st.subheader("Top Mutual Fund Holders")
            mf_df = ticker_obj.mutualfund_holders
            if mf_df is None or mf_df.empty:
                st.write("No mutual fund holders data available.")
            else:
                st.dataframe(mf_df)

        ###################################################################
        # TAB 4: Analysis
        ###################################################################
        with tabs[3]:
            st.header("Analysis")
            # st.write("Below are four example tiles showing typical analysis data—two tiles per row for a clean look.")

            colA, colB = st.columns(2)
            # ----------------------
            # TILE 1: Revenue vs. Earnings (Quarterly, last 4 quarters)
            # ----------------------

            with colA:
                st.markdown(
                    """
                    <style>
                    .recommendations-panel {
                        background-color: #1e1e1e;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                    }
                    .panel-title {
                        color: white;
                        font-size: 20px;
                        font-weight: 500;
                        margin-bottom: 5px;
                    }
                    </style>
                    <div class="recommendations-panel">
                        <div class="panel-title">
                            <h3 style="margin: 0; padding: 0;">Revenue V Earnings</h3>
                        </div>
                    """,
                    unsafe_allow_html=True
                )

                # Use quarterly_financials (Yahoo's quarterly income statement)
                q_fin = ticker_obj.quarterly_financials
                if q_fin is not None and not q_fin.empty:
                    if "Total Revenue" in q_fin.index and "Net Income" in q_fin.index:
                        # Transpose so rows represent quarters and columns the metrics
                        df_q = q_fin.loc[["Total Revenue", "Net Income"]].T

                        # Convert the index to datetime (if not already) and sort chronologically,
                        # then take the last 4 quarters.
                        df_q.index = pd.to_datetime(df_q.index)
                        df_q = df_q.sort_index(ascending=True).tail(4)

                        # Format the index as "Mon YYYY" (e.g. "Jan 2025") and store in a new column
                        df_q["Quarter"] = df_q.index.strftime("%b %Y")

                        # Rename columns for clarity
                        df_q.rename(columns={"Total Revenue": "Revenue", "Net Income": "Earnings"}, inplace=True)

                        # Melt (pivot) the DataFrame into long format for plotting
                        df_melt = df_q.melt(id_vars="Quarter", value_vars=["Revenue", "Earnings"],
                                            var_name="Indicator", value_name="Amount")

                        # Helper function to format numbers with appropriate SI suffixes
                        def format_si(n):
                            abs_n = abs(n)
                            if abs_n >= 1e9:
                                return f"{n / 1e9:.2f}B"
                            elif abs_n >= 1e6:
                                return f"{n / 1e6:.2f}M"
                            elif abs_n >= 1e3:
                                return f"{n / 1e3:.2f}K"
                            else:
                                return f"{n:.2f}"

                        # Create a new column with the formatted numbers
                        df_melt["Formatted"] = df_melt["Amount"].apply(format_si)

                        # Create a grouped bar chart using Plotly Express.
                        # Pass hover_data so the raw "Amount" is hidden and only our formatted value shows.
                        fig = px.bar(
                            df_melt,
                            x="Quarter",
                            y="Amount",
                            color="Indicator",
                            barmode="group",
                            color_discrete_map={"Revenue": "#4c78a8", "Earnings": "#f58518"},
                            labels={"Amount": "USD"},
                            text_auto=True,
                            height=300,
                            hover_data={"Formatted": True, "Amount": False}
                        )

                        # Update hovertemplate for all bar traces to show the formatted value.
                        fig.update_traces(hovertemplate='%{x}<br>Amount: %{customdata[0]}')

                        # Manually compute y-axis tick values and tick text.
                        import numpy as np
                        y_min = df_melt["Amount"].min()
                        y_max = df_melt["Amount"].max()
                        tickvals = np.linspace(y_min, y_max, 5)
                        ticktext = [format_si(val) for val in tickvals]
                        fig.update_yaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)

                        # Update layout for dark theme aesthetics.
                        fig.update_layout(
                            plot_bgcolor="#1e1e1e",
                            paper_bgcolor="#1e1e1e",
                            font_color="white",
                            margin=dict(l=20, r=20, t=40, b=20),
                            title_text=None,
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Quarterly financials missing 'Total Revenue' or 'Net Income'.")
                else:
                    st.write("No quarterly financials data available.")

            # ----------------------
            # TILE 2: Analyst Recommendations (STACKED CHART)
            # ----------------------
            with colB:
                st.markdown(
                    """
                    <style>
                    .recommendations-panel {
                        background-color: #1e1e1e;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                    }

                    }
                    .panel-title {
                        color: white;
                        font-size: 20px;
                        font-weight: 500;
                        margin-bottom: 5px;
                    }
                    </style>
                    <div class="recommendations-panel">
                        <div class="panel-title"><h3 border-radius: 1px; padding: 1px;margin-bottom: 5px;>Analyst Outlook</h3></div>
                    """,
                    unsafe_allow_html=True
                )

                recs_df = ticker_obj.recommendations
                if recs_df is not None and not recs_df.empty:
                    required_cols = ['period', 'strongBuy', 'buy', 'hold', 'sell', 'strongSell']
                    if all(c in recs_df.columns for c in required_cols):

                        # Update date conversion to properly handle 0m and negative offsets.
                        def get_period_date(period_val):
                            if isinstance(period_val, str) and period_val.endswith('m'):
                                try:
                                    offset = int(period_val[:-1])
                                except ValueError:
                                    return None
                                # For negative offsets (e.g. "-1m"), subtract the absolute number of months.
                                if offset < 0:
                                    return datetime.now() - pd.DateOffset(months=abs(offset))
                                else:
                                    # For "0m" or positive values, treat them as months ago.
                                    return datetime.now() - pd.DateOffset(months=offset)
                            else:
                                try:
                                    return pd.to_datetime(period_val)
                                except Exception:
                                    return None

                        recs_df['period_date'] = recs_df['period'].apply(get_period_date)
                        recs_df = recs_df[recs_df['period_date'].notnull()]

                        # Define the window: current month (0m) and the previous three months.
                        current_date = datetime.now()
                        three_months_ago = current_date - pd.DateOffset(months=3)
                        recs_df = recs_df[
                            (recs_df['period_date'] >= three_months_ago) & (recs_df['period_date'] <= current_date)]

                        recs_df = recs_df.sort_values('period_date', ascending=True)
                        recs_df['display_period'] = recs_df['period_date'].apply(lambda d: d.strftime('%b %Y'))
                        recs_df['total'] = recs_df[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].sum(axis=1)

                        melted = recs_df.melt(
                            id_vars=['display_period', 'total'],
                            value_vars=['strongBuy', 'buy', 'hold', 'sell', 'strongSell'],
                            var_name='Rating',
                            value_name='Count'
                        )

                        rating_map = {
                            'strongBuy': 'Strong Buy',
                            'buy': 'Buy',
                            'hold': 'Hold',
                            'sell': 'Sell',
                            'strongSell': 'Strong Sell'
                        }
                        melted['Rating_Display'] = melted['Rating'].map(rating_map)

                        # Define the desired order for stacking (top to bottom in the legend)
                        rating_order = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
                        # Sort display_period chronologically
                        sorted_periods = sorted(recs_df['display_period'].unique(),
                                                key=lambda x: datetime.strptime(x, '%b %Y'))

                        fig = px.bar(
                            melted,
                            x="display_period",
                            y="Count",
                            color="Rating_Display",
                            category_orders={
                                "Rating_Display": rating_order,
                                "display_period": sorted_periods
                            },
                            hover_data={
                                "display_period": False,  # <--- Hide from tooltip
                                "Rating_Display": False,
                                "Count": True,
                                "total": False  # optionally hide 'total' if it’s in your DataFrame
                            },
                            color_discrete_sequence=['#00b894', '#78cd51', '#f1c40f', '#e67e22', '#e74c3c'],
                            labels={"display_period": "Period", "Count": "Number of Analysts",
                                    "Rating_Display": "Rating"}
                        )
                        fig.update_layout(
                            barmode="stack",
                            hovermode="x unified",
                            paper_bgcolor="#1e1e1e",
                            plot_bgcolor="#1e1e1e",
                            font_color="white",
                            xaxis_title=None,
                            yaxis_title="Number of Analysts",
                            legend_title=None,
                            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
                            width=200,
                            height=280,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        # Add total counts as text annotations
                        for i, row in recs_df.iterrows():
                            fig.add_annotation(
                                x=row['display_period'],
                                y=row['total'] * 1.05,
                                text=str(row['total']),
                                showarrow=False,
                                font=dict(color="white", size=12)
                            )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='color:white; background:#1e1e1e; padding:10px; border-radius:5px;'>Recommendations DataFrame missing required columns.</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<div style='color:white; background:#1e1e1e; padding:10px; border-radius:5px;'>No analyst recommendations data available.</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            # Next row
            colC, colD = st.columns(2)

            # ----------------------
            # ----------------------
            # TILE 3: EPS Trend - Modern Dark UI With Timeline
            # ----------------------
            with colC:
                st.markdown(
                    "<h3 style='color:#fff; background:#1b1b1b; padding:5px; border-radius:8px;'>Earnings Per Share</h3>",
                    unsafe_allow_html=True
                )

                # Retrieve earnings history from the ticker
                eh = ticker_obj.earnings_history
                if eh is None:
                    st.write("No earnings history data available.")
                elif isinstance(eh, pd.DataFrame):
                    df_earn = eh.copy()
                elif isinstance(eh, list):
                    if len(eh) == 0:
                        st.write("No earnings history data available.")
                    else:
                        df_earn = pd.DataFrame(eh)
                else:
                    st.write("No earnings history data available.")

                # Proceed only if we have a DataFrame
                if 'df_earn' in locals() and not df_earn.empty:
                    # 1) Identify EPS columns
                    possible_actual = ["epsActual", "epsactual"]
                    possible_est = ["epsEstimate", "epsestimate"]
                    actual_col = next((c for c in possible_actual if c in df_earn.columns), None)
                    est_col = next((c for c in possible_est if c in df_earn.columns), None)

                    # If the index holds the date info and its name is "quarte", reset and rename it
                    if df_earn.index.name == "quarter":
                        df_earn = df_earn.reset_index().rename(columns={"quarter": "quarter"})
                    elif df_earn.index.name is None:
                        # If the index has no name but contains date information, you might use it anyway:
                        df_earn = df_earn.reset_index()
                        # Optionally, rename the default column "index" to "quarter":
                        df_earn = df_earn.rename(columns={"index": "quarter"})

                    if not actual_col or not est_col:
                        st.write("EPS columns not found in earnings history.")
                    else:
                        # 2) Sort data by the 'quarter' column (oldest first)
                        #    Assuming 'quarter' is a string like "2024-03-31 00:00:00"
                        if 'quarter' in df_earn.columns:
                            df_earn['quarter'] = pd.to_datetime(df_earn['quarter'])
                            df_earn.sort_values(by='quarter', ascending=True, inplace=True)
                        else:
                            st.write("No 'quarter' column found; cannot label quarters.")
                            return

                        # 3) Take the last 4 quarters (past 3 + upcoming 1)
                        last_3 = df_earn.tail(3).copy()  # Last 3 completed quarters

                        # 4) Create a dummy next quarter
                        next_qtr = last_3.iloc[-1:].copy()
                        if 'quarter' in next_qtr.columns:
                            next_qtr['quarter'] = next_qtr['quarter'] + pd.DateOffset(months=3)
                        # Clear actual EPS for the future quarter
                        next_qtr[actual_col] = np.nan

                        # Combine past 3 + future 1
                        all_qtrs = pd.concat([last_3, next_qtr])

                        # 5) Create a user-friendly label for each quarter
                        def quarter_label(row):
                            dt = row["quarter"]
                            if pd.isnull(dt):
                                return "Unknown"
                            q = (dt.month - 1) // 3 + 1
                            return f"Q{q} '{str(dt.year)[-2:]}"  # e.g. "Q1 '24"

                        all_qtrs['Quarter'] = all_qtrs.apply(quarter_label, axis=1)

                        # 6) Calculate EPS Surprise
                        all_qtrs['epsSurprise'] = all_qtrs[actual_col] - all_qtrs[est_col]

                        def beat_or_miss(row):
                            if pd.isnull(row['epsSurprise']):
                                return "Upcoming"
                            return "Beat" if row['epsSurprise'] > 0 else ("Miss" if row['epsSurprise'] < 0 else "Met")

                        all_qtrs['beatOrMiss'] = all_qtrs.apply(beat_or_miss, axis=1)

                        # 7) Format EPS surprise for display
                        def format_surprise(row):
                            if pd.isnull(row['epsSurprise']):
                                return ""
                            return f"+${row['epsSurprise']:.2f}" if row[
                                                                        'epsSurprise'] > 0 else f"${row['epsSurprise']:.2f}"

                        all_qtrs['formattedSurprise'] = all_qtrs.apply(format_surprise, axis=1)

                        # 8) Get date for the upcoming quarter (if available)
                        upcoming_date = ""
                        if 'startdatetime' in all_qtrs.columns and not pd.isnull(all_qtrs['startdatetime'].iloc[-1]):
                            upcoming_date = all_qtrs['startdatetime'].iloc[-1].strftime("%b %d")

                        # 9) Define next_estimate so the code won't fail
                        next_estimate = all_qtrs[est_col].iloc[-1]
                        if pd.isnull(next_estimate):
                            next_estimate = 0.0

                        # 10) Create the custom dark-themed UI using HTML/CSS
                        html_content = f"""
                                 <div style="background-color:#1e1e1e; border-radius:8px; padding:15px; margin-top:15px; color:white; font-family:sans-serif;">
                                    <div style="background-color:#000000; border-radius:5px; padding:10px; margin-bottom:15px;">
                                        <span style="color:#00cc66; font-size:18px; font-weight:bold;">+{next_estimate:.2f}</span>
                                        <span style="color:white; margin-left:8px;">Estimate</span>
                                 </div>
                                     <div style="position:relative; height:200px;">
                                    <!-- Dotted timeline -->
                                    <div style="position:absolute; top:21px; left:5%; right:5%; height:1px; border-top:2px dotted #555555;"></div>
                                </div>
                            """

                        # Add circles and labels for each quarter
                        quarters = all_qtrs['Quarter'].tolist()
                        statuses = all_qtrs['beatOrMiss'].tolist()
                        surprises = all_qtrs['formattedSurprise'].tolist()

                        # The issue is in this loop:
                        for i, (qtr, status, surprise) in enumerate(zip(quarters, statuses, surprises)):
                            # Position (evenly distributed)
                            left_pos = 5 + (i * 25)  # percentage

                            # Determine circle style based on status
                            if status == "Beat":
                                circle_style = "background-color:#00cc66; border:2px solid #00cc66;"
                                status_color = "#00cc66"
                            elif status == "Miss":
                                circle_style = "background-color:#ff3333; border:2px solid #ff3333;"
                                status_color = "#ff3333"
                            elif status == "Met":
                                circle_style = "background-color:#aaaaaa; border:2px solid #aaaaaa;"
                                status_color = "#aaaaaa"
                            else:  # Upcoming
                                circle_style = "background-color:transparent; border:2px solid #555555;"
                                status_color = "#aaaaaa"

                            timeline_y = 60
                            # Get the surprise value as a number (default to 0 if not available)
                            surprise_value = 0
                            if status != "Upcoming" and surprise:
                                # Extract numeric value from formatted surprise string
                                try:
                                    # Handle both "+$0.06" and "$-0.03" formats
                                    surprise_value = float(surprise.replace('+$', '').replace('$', ''))
                                except:
                                    surprise_value = 0

                            # Scale the surprise value for visual representation
                            # Adjust the multiplier (30) to control how much the circles move up/down
                            # Higher value = more vertical movement per dollar of surprise
                            vertical_offset = surprise_value * 30

                            html_content += f"""
                            <!-- Quarter {i + 1} -->
                            <div style="position:absolute; top:105px; left:{left_pos}%;">
                                <!-- Circle -->
                                <div style="width:20px; height:20px; border-radius:50%; {circle_style} margin:0 auto;"></div>
                                <!-- Quarter label -->
                                <div style="text-align:center; margin-top:10px; color:#bbbbbb; font-size:12px;">{qtr}</div>
                                <!-- Status label -->
                                <div style="text-align:center; margin-top:5px; color: {status_color} ; font-size:12px;"> {status if status != "Upcoming" else ""} <span style="display:block;">{surprise}</span> </div>
                            </div>
                            """

                        # Add upcoming date if available
                        if upcoming_date:
                            html_content += f'''
                                        <!-- Upcoming date -->
                                            <div style="position:absolute; top:0px; right:5%; color:#aaaaaa; font-size:12px;"> {upcoming_date} </div>
                            '''

                        # Close the container divs
                        html_content += '''</div></div>'''

                        # Render the HTML content
                        st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.write("No earnings history data available.")

            # ----------------------
            # TILE 4: Price Targets (Now uses our function)
            # ----------------------
            with colD:
                # Simply call the function we defined earlier
                display_price_targets(ticker_obj.info)


if __name__ == "__main__":
    main()
