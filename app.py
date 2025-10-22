import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go

# ============================
# 🎨 Page Config & Custom CSS
# ============================
st.set_page_config(
    page_title="📈 Stock Price Predictor (Machine Learning)",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for Dark/Light Mode Readability
st.markdown("""
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color, #FFFFFF);
        }
        .stApp {
            background-color: var(--bg-color, #0E1117);
            color: #FAFAFA;
        }
        /* Card style for sections */
        .stCard {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        /* Adjust for Light Mode */
        @media (prefers-color-scheme: light) {
            .stApp {
                background-color: #FFFFFF;
                color: #111111;
            }
            .stCard {
                background-color: #F7F9FB;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
        }
    </style>
""", unsafe_allow_html=True)

# ============================
# Sidebar Navigation
# ============================
page = st.sidebar.selectbox("🌍 Navigate", ["Home", "About", "Model Info"])

# ============================
# PAGE: HOME
# ============================
if page == "Home":
    st.title("📈 Stock Price Predictor (Machine Learning)")
    st.markdown(
        "This app uses **historical stock data** to predict the **next day's closing price** "
        "using a Linear Regression model trained on Open, High, Low, Close, and Volume data."
    )

    symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, MSFT, TSLA):", "AAPL")

    if symbol:
        try:
            df = yf.download(symbol, period="6mo", interval="1d")
            if df.empty:
                st.warning(f"No data found for {symbol}. Please check the symbol and try again.")
            else:
                st.success(f"✅ Successfully fetched data for {symbol}")

                # Prepare training data
                df["Next Close"] = df["Close"].shift(-1)
                df.dropna(inplace=True)

                X = df[["Open", "High", "Low", "Close", "Volume"]]
                y = df["Next Close"]

                model = LinearRegression()
                model.fit(X, y)

                # Predict next close
                last_row = df.iloc[-1][["Open", "High", "Low", "Close", "Volume"]].values.reshape(1, -1)
                predicted_price = float(model.predict(last_row)[0])  # ensure it's float, not Series
                last_close = float(df["Close"].iloc[-1])
                change = predicted_price - last_close
                pct_change = (change / last_close) * 100

                # Display results
                st.markdown(f"### Predicted Next Closing Price for {symbol}: **${predicted_price:.2f}**")
                st.markdown(f"Change from last close (${last_close:.2f}): **{pct_change:+.2f}%**")

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["Close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="#00BFFF", width=2)
                ))
                fig.update_layout(
                    title=f"📊 Stock Price Trend (Last 6 Months) - {symbol}",
                    xaxis_title="Date",
                    yaxis_title="Close Price (USD)",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander(f"📂 Raw Data for {symbol}"):
                    st.dataframe(df.tail(20))
        except Exception as e:
            st.error(f"⚠️ Error fetching or processing data: {e}")

# ============================
# PAGE: ABOUT
# ============================
elif page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
        This **Stock Price Predictor** was built by **Kevin Kiplangat Mutai** 🚀  
        It uses real market data from Yahoo Finance and applies a **Linear Regression Model** 
        to estimate the next day's closing price.

        **Technologies used:**
        - Streamlit for the UI
        - yFinance for stock data
        - scikit-learn for the ML model
        - Plotly for interactive charts

        💡 *Disclaimer:* Predictions are for **educational purposes only** and should not be used for trading decisions.
    """)

# ============================
# PAGE: MODEL INFO
# ============================
elif page == "Model Info":
    st.title("🧠 Model Information")
    st.markdown("""
        The prediction model uses a **Linear Regression** approach:
        - **Features:** Open, High, Low, Close, Volume  
        - **Target:** Next day’s Close price  
        - **Training Data:** Last 6 months of stock history  

        This simple model demonstrates how Machine Learning can be applied in financial analysis.
    """)

    st.info("Future upgrade: Integrate LSTM for time-series learning and multi-day forecasting.")

st.markdown("<hr><center>Built by <b>Kevin Kiplangat Mutai</b> | Machine Learning Stock Prediction Demo 🚀</center>", unsafe_allow_html=True)
