import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import date, timedelta

# ---------------------------
# 🏷️ Page Configuration
# ---------------------------
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📊",
    layout="centered"
)

# ---------------------------
# 🎨 App Header
# ---------------------------
st.title("📈 Stock Price Predictor (Machine Learning)")
st.markdown("""
This app uses **historical stock data** to predict the **next day's closing price**
using a **Linear Regression model** trained on Open, High, Low, Close, and Volume data.
""")

# ---------------------------
# 🧠 Input Section
# ---------------------------
symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, MSFT, TSLA):", "AAPL").upper()

# ---------------------------
# 📦 Fetch Data
# ---------------------------
if st.button("Predict"):
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years of data

        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            st.error(f"❌ No data found for {symbol}. Please check the symbol and try again.")
        else:
            st.success(f"✅ Successfully fetched data for {symbol}")

            # ---------------------------
            # 🧮 Prepare Data
            # ---------------------------
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data['Target'] = data['Close'].shift(-1)
            data = data.dropna()

            X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = data['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # ---------------------------
            # 🤖 Train Model
            # ---------------------------
            model = LinearRegression()
            model.fit(X_train, y_train)

            # ---------------------------
            # 🔮 Predict Next Day
            # ---------------------------
            latest_data = data.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1)
            predicted_price = model.predict(latest_data)
            predicted_price = float(predicted_price[0])  # ✅ Convert to float

            last_close = float(data['Close'].iloc[-1])  # ✅ Convert to float
            change = ((predicted_price - last_close) / last_close) * 100

            # ---------------------------
            # 📊 Display Results
            # ---------------------------
            st.subheader(f"Predicted Next Closing Price for {symbol}: ${predicted_price:.2f}")
            st.write(f"Change from last close (${last_close:.2f}): {change:+.2f}%")

            # ---------------------------
            # 📈 Plot Stock Trend
            # ---------------------------
            st.subheader("📊 Stock Price Trend (Last 6 Months)")
            last_6_months = data.tail(180)
            st.line_chart(last_6_months['Close'])

            # ---------------------------
            # 📋 Show Data
            # ---------------------------
            with st.expander(f"Raw Data for {symbol}"):
                st.dataframe(data.tail(10))

    except Exception as e:
        st.error(f"⚠️ Error fetching or processing data: {e}")

# ---------------------------
# 👤 Footer
# ---------------------------
st.markdown("""
---
Built by **Kevin Kiplangat Mutai** | Machine Learning Stock Prediction Demo 🚀
""")
