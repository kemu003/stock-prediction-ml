import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# 🎨 Custom Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📊",
    layout="centered",
)

# Inject Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #1a73e8;
        }
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #155ab6;
            color: #fff;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🌍 Sidebar Navigation
# -----------------------------
page = st.sidebar.selectbox(
    "Navigate",
    ["🏠 Home", "📘 About", "🧠 Model Info"]
)

# -----------------------------
# 🏠 HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.title("📈 Stock Price Predictor (Machine Learning)")
    st.write(
        "This app uses **historical stock data** to predict the next day's closing price "
        "using a **Linear Regression model** trained on Open, High, Low, Close, and Volume data."
    )

    # User input
    symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, MSFT, TSLA):").upper()

    if symbol:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            data = yf.download(symbol, start=start_date, end=end_date)

            if data.empty:
                st.warning(f"⚠️ No data found for {symbol}. Please check the stock symbol.")
            else:
                st.success(f"✅ Successfully fetched data for {symbol}")

                # Prepare data
                data['Prediction'] = data['Close'].shift(-1)
                X = data[['Open', 'High', 'Low', 'Close', 'Volume']][:-1]
                y = data['Prediction'][:-1]

                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict next close
                last_row = data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
                next_pred = model.predict(last_row)[0]
                last_close = data['Close'].iloc[-1]
                change_pct = ((next_pred - last_close) / last_close) * 100

                st.subheader(f"Predicted Next Closing Price for {symbol}: ${next_pred:.2f}")
                st.write(f"Change from last close (${last_close:.2f}): {change_pct:+.2f}%")

                # Plot
                st.subheader("📊 Stock Price Trend (Last 6 Months)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data.index, data['Close'], label="Actual Close", linewidth=2)
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.set_title(f"{symbol} Closing Price History")
                ax.legend()
                st.pyplot(fig)

                st.subheader(f"📄 Raw Data for {symbol}")
                st.dataframe(data.tail())

        except Exception as e:
            st.error(f"⚠️ Error fetching or processing data: {e}")

    st.markdown("---")
    st.markdown("**Built by Kevin Kiplangat Mutai | Machine Learning Stock Prediction Demo 🚀**")

# -----------------------------
# 📘 ABOUT PAGE
# -----------------------------
elif page == "📘 About":
    st.title("📘 About This App")
    st.write("""
    The **Stock Price Predictor** is a demo machine learning app that uses real stock data 
    from Yahoo Finance to estimate future prices.
    
    - **Tech Stack:** Streamlit, Python, yFinance, scikit-learn  
    - **Model:** Linear Regression  
    - **Goal:** Demonstrate ML for financial forecasting.
    
    ⚙️ *This project was built and deployed by* **Kevin Kiplangat Mutai**.
    """)

# -----------------------------
# 🧠 MODEL INFO PAGE
# -----------------------------
elif page == "🧠 Model Info":
    st.title("🧠 Model Information")
    st.write("""
    The app uses **Linear Regression**, a fundamental machine learning algorithm for regression tasks.
    
    **Model Details:**
    - **Inputs:** Open, High, Low, Close, Volume
    - **Output:** Next day's Close price
    - **Training Split:** 80% Train, 20% Test
    
    **Formula:**  
    `Close_next_day = a*Open + b*High + c*Low + d*Close + e*Volume + bias`
    
    Future versions may include:
    - Random Forest Regression  
    - LSTM Neural Networks for time series  
    - Real-time prediction APIs
    """)
