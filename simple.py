import streamlit as st
import yfinance as yf
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Stonks", layout="wide")


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
# Sidebar Inputs
##############################
st.sidebar.header("Select Ticker & Date Range")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now().date())
timeframe_options = ["daily", "weekly"]
selected_timeframe = st.sidebar.selectbox("Select Timeframe", timeframe_options)
timeframe_mapping = {"daily": "1d", "weekly": "1wk"}
interval = timeframe_mapping[selected_timeframe]

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
    else:
        chart_df.dropna(inplace=True)
        chart_df.index = pd.to_datetime(chart_df.index)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        chart_df = chart_df[numeric_cols].copy()
        chart_df.dropna(inplace=True)

    # ------------------------
    # 2) Fetch additional data for Company Overview
    # ------------------------
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

    ##############################
    # Create Tabs: Company Overview & Chart(s)
    ##############################
    tabs = st.tabs(["Company Overview", "Chart(s)"])

    ###################################################################
    # TAB 1: Company Overview
    ###################################################################
    with tabs[0]:
        st.header("Company Overview")
        # We'll use two rows of three columns each for six tiles.
        row1 = st.columns(3, gap="medium")
        row2 = st.columns(3, gap="medium")

        # Fixed tile style for consistency.
        tile_style = (
            "border:1px solid #ddd; padding:10px; border-radius:5px; "
            "text-align:center; margin:5px; height:120px; "
            "display:flex; flex-direction:column; justify-content:center;"
        )

        with row1[0]:
            st.markdown(
                f"""
                <div style="{tile_style}">
                    <strong>EBITDA <sub style='font-size:60%;'>{year}</sub></strong><br>
                    {format_number(ebitda)} {get_indicator('EBITDA', financials)}
                </div>
                """,
                unsafe_allow_html=True
            )
        with row1[1]:
            st.markdown(
                f"""
                <div style="{tile_style}">
                    <strong>Net Income <sub style='font-size:60%;'>{year}</sub></strong><br>
                    {format_number(net_income)} {get_indicator('Net Income', financials)}
                </div>
                """,
                unsafe_allow_html=True
            )
        with row1[2]:
            st.markdown(
                f"""
                <div style="{tile_style}">
                    <strong>Operating Income <sub style='font-size:60%;'>{year}</sub></strong><br>
                    {format_number(operating_income)} {get_indicator('Operating Income', financials)}
                </div>
                """,
                unsafe_allow_html=True
            )
        with row2[0]:
            st.markdown(
                f"""
                <div style="{tile_style}">
                    <strong>Operating Expenses <sub style='font-size:60%;'>{year}</sub></strong><br>
                    {format_number(operating_expenses)} {get_indicator('Operating Expenses', financials)}
                </div>
                """,
                unsafe_allow_html=True
            )
        with row2[1]:
            st.markdown(
                f"""
                <div style="{tile_style}">
                    <strong>Operating Revenue <sub style='font-size:60%;'>{year}</sub></strong><br>
                    {format_number(operating_revenue)} {get_indicator('Operating Revenue', financials)}
                </div>
                """,
                unsafe_allow_html=True
            )
        with row2[2]:
            st.markdown(
                f"""
                <div style="{tile_style}">
                    <strong>Total Revenue <sub style='font-size:60%;'>{year}</sub></strong><br>
                    {format_number(total_revenue)} {get_indicator('Total Revenue', financials)}
                </div>
                """,
                unsafe_allow_html=True
            )

    ###################################################################
    # TAB 2: Chart(s)
    ###################################################################
    with tabs[1]:
        st.header("Chart(s)")
        # Reset index and rename columns so that Altair sees a 'date' column
        chart_df = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df.columns = chart_df.columns.get_level_values(0)
        if chart_df.empty:
            st.warning("No data found for the specified range.")
        else:
            chart_df.dropna(inplace=True)
            chart_df.index = pd.to_datetime(chart_df.index)
            chart_df.reset_index(inplace=True)  # <-- Reset index to bring Date into a column
            # Rename columns so that they match the ones used in Altair
            chart_df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)

            # Create a basic candlestick chart using Altair
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
            candlestick = alt.layer(rule, bar).properties(
                width=900,
                height=400,
                title=f"{ticker_symbol.upper()} Candlestick Chart ({selected_timeframe})"
            )
            st.altair_chart(candlestick, use_container_width=True)
