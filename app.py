import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import requests
from textblob import TextBlob

# Load models
lstm_model = tf.keras.models.load_model("lstm_gold_model.h5")
xgb_model = joblib.load("xgb_gold_model.pkl")
scaler = joblib.load("scaler.pkl")

# Get today's date dynamically
today_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Streamlit UI Customization
st.set_page_config(page_title="Gold Price Prediction", layout="wide")
st.title("📈 Gold Price Prediction Dashboard")

# Allow user to select how many days ahead to predict
days_ahead = st.slider("⏳ Predict how many days ahead?", min_value=1, max_value=7, value=1)
prediction_date = (datetime.datetime.today() + datetime.timedelta(days=days_ahead)).strftime('%Y-%m-%d')

# Fetch the latest gold price (ensures updated data)
gold_data = yf.download("GC=F", period="200d", interval="1d")  # Get 200 days for trend analysis

if gold_data.empty:
    st.error("Error: No data received for gold prices. Try again later.")
else:
    # Extract the latest available price (last row)
    gold_latest = gold_data.tail(1)

    # Extract latest available date dynamically
    latest_available_date = gold_latest.index[-1].strftime('%Y-%m-%d')

    # Extract and convert current price to float BEFORE using it
    if 'Close' in gold_latest.columns:
        current_price = float(gold_latest['Close'].values[0])  # Convert NumPy array to float
    else:
        st.error("Error: 'Close' price data is missing in the dataset.")
        st.stop()

    # Preprocess latest data
    latest_features = gold_latest[['Open', 'High', 'Low', 'Close', 'Volume']].values
    latest_features = scaler.transform(latest_features)

    # Reshape for LSTM
    latest_features_lstm = latest_features.reshape((1, 1, latest_features.shape[1]))

    # Predict price movement with LSTM
    lstm_prediction = lstm_model.predict(latest_features_lstm)

    # Combine LSTM output with features for XGBoost
    final_input = np.hstack((latest_features, lstm_prediction.reshape(1, -1)))

    # Predict movement with XGBoost
    xgb_prediction = xgb_model.predict(final_input)

    # Define predicted price change (convert to float)
    predicted_price_change = float(lstm_prediction[0][0] * (current_price * 0.02))  # Assuming 2% fluctuation
    predicted_price = float(current_price + (predicted_price_change if xgb_prediction[0] == 1 else -predicted_price_change))

    # Define breakpoint price (convert to float)
    breakpoint_price = float(current_price * 1.01 if xgb_prediction[0] == 1 else current_price * 0.99)

    # **Calculate Adjusted Predictions (-20)**
    adjusted_current_price = current_price - 20
    adjusted_predicted_price = predicted_price - 20
    adjusted_breakpoint_price = breakpoint_price - 20

    # **Show Metrics**
    st.subheader("📊 Predictions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="📌 Current Gold Price", value=f"${current_price:.2f}")
        st.metric(label="📊 Predicted Price", value=f"${predicted_price:.2f}")
        st.metric(label="🔴 Breakpoint Price", value=f"${breakpoint_price:.2f}")
    
    with col2:
        st.metric(label="📌 Adjusted Current Price (-20)", value=f"${adjusted_current_price:.2f}")
        st.metric(label="📊 Adjusted Predicted Price (-20)", value=f"${adjusted_predicted_price:.2f}")
        st.metric(label="🔴 Adjusted Breakpoint (-20)", value=f"${adjusted_breakpoint_price:.2f}")

    # **Confidence Interval**
    confidence_interval = predicted_price * 0.015
    upper_bound = predicted_price + confidence_interval
    lower_bound = predicted_price - confidence_interval
    st.write(f"📉 **Confidence Interval:** ${lower_bound:.2f} - ${upper_bound:.2f}")

    # **Moving Averages**
    gold_data["50_MA"] = gold_data["Close"].rolling(window=50).mean()
    gold_data["200_MA"] = gold_data["Close"].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gold_data.index, gold_data["Close"], label="Gold Price", color='blue')
    ax.plot(gold_data.index, gold_data["50_MA"], label="50-Day MA", linestyle='dashed', color='orange')
    ax.plot(gold_data.index, gold_data["200_MA"], label="200-Day MA", linestyle='dashed', color='red')

    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (USD)")
    ax.set_title("Gold Price with Moving Averages")
    ax.legend()
    st.pyplot(fig)

    # **RSI Indicator**
    def compute_rsi(data, window=14):
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    gold_data["RSI"] = compute_rsi(gold_data)
    latest_rsi = gold_data["RSI"].iloc[-1]
    st.write(f"📊 **RSI Indicator (Last Close):** {latest_rsi:.2f}")

    if latest_rsi > 70:
        st.warning("🚨 Overbought! Possible Price Drop Ahead")
    elif latest_rsi < 30:
        st.success("💹 Oversold! Possible Price Increase Ahead")

    # **Historical Prices Table**
    st.subheader("📜 Historical Gold Prices (Last 30 Days)")
    st.dataframe(gold_data.tail(30)[['Close']])

    # **Real-Time Gold News Sentiment Analysis**
    st.subheader("📰 Latest Gold Market News & Sentiment")
    news_api_url = "https://newsapi.org/v2/everything?q=gold+price&sortBy=publishedAt&apiKey=YOUR_API_KEY"
    response = requests.get(news_api_url)
    news_data = response.json()

    for article in news_data.get("articles", [])[:5]:
        title = article["title"]
        sentiment = TextBlob(title).sentiment.polarity
        sentiment_text = "🔴 Negative" if sentiment < -0.1 else "🟢 Positive" if sentiment > 0.1 else "⚪ Neutral"
        st.write(f"📌 {title} - {sentiment_text}")
