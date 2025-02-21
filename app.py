import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.figure_factory as ff
from textblob import TextBlob

# ----------------------------
# Load Pre-trained Models & Scaler
# ----------------------------
lstm_model = tf.keras.models.load_model("lstm_gold_model.h5")
xgb_model = joblib.load("xgb_gold_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Utility Functions
# ----------------------------
@st.cache_data
def fetch_gold_data(period: str, interval: str) -> pd.DataFrame:
    """Fetch gold price data from Yahoo Finance."""
    data = yf.download("GC=F", period=period, interval=interval)
    return data

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators (SMA and EMA) and add them to the DataFrame."""
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def predict_tomorrow_price(gold_data: pd.DataFrame) -> dict:
    """
    Predict tomorrow's gold price using the LSTM and XGBoost models.
    Returns a dictionary with various price metrics.
    """
    gold_latest = gold_data.tail(1)
    latest_available_date = gold_latest.index[-1].strftime('%Y-%m-%d')

    if 'Close' not in gold_latest.columns:
        st.error("Error: 'Close' price data is missing in the dataset.")
        st.stop()

    current_price = float(gold_latest['Close'].values[0])
    tomorrow_date = (datetime.datetime.strptime(latest_available_date, '%Y-%m-%d') +
                     datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    # Preprocess latest data for prediction
    latest_features = gold_latest[['Open', 'High', 'Low', 'Close', 'Volume']].values
    latest_features_scaled = scaler.transform(latest_features)
    latest_features_lstm = latest_features_scaled.reshape((1, 1, latest_features_scaled.shape[1]))

    # LSTM prediction
    lstm_prediction = lstm_model.predict(latest_features_lstm)

    # Combine LSTM output with features for XGBoost
    final_input = np.hstack((latest_features_scaled, lstm_prediction.reshape(1, -1)))
    xgb_prediction = xgb_model.predict(final_input)

    # Calculate predicted price change using a dynamic 2% factor
    predicted_price_change = float(lstm_prediction[0][0] * (current_price * 0.02))
    predicted_price = float(
        current_price + (predicted_price_change if xgb_prediction[0] == 1 else -predicted_price_change)
    )
    breakpoint_price = float(
        current_price * 1.01 if xgb_prediction[0] == 1 else current_price * 0.99
    )

    # Adjusted predictions (subtracting 20 for alternate scenario)
    adjusted_current_price = current_price - 20
    adjusted_predicted_price = predicted_price - 20
    adjusted_breakpoint_price = breakpoint_price - 20

    return {
        "today_date": datetime.datetime.today().strftime('%Y-%m-%d'),
        "latest_available_date": latest_available_date,
        "tomorrow_date": tomorrow_date,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "breakpoint_price": breakpoint_price,
        "adjusted_current_price": adjusted_current_price,
        "adjusted_predicted_price": adjusted_predicted_price,
        "adjusted_breakpoint_price": adjusted_breakpoint_price,
    }

def plot_price_movement(predictions: dict) -> go.Figure:
    """Create an interactive Plotly line chart to compare normal and adjusted predictions."""
    categories = ["Current Price", "Breakpoint", "Predicted Price"]
    normal_prices = [predictions["current_price"], predictions["breakpoint_price"], predictions["predicted_price"]]
    adjusted_prices = [
        predictions["adjusted_current_price"],
        predictions["adjusted_breakpoint_price"],
        predictions["adjusted_predicted_price"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=categories,
        y=normal_prices,
        mode='lines+markers',
        name='Normal Prediction',
        line=dict(dash='dash', color='blue'),
        marker=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=categories,
        y=adjusted_prices,
        mode='lines+markers',
        name='Adjusted Prediction (-20)',
        line=dict(dash='dash', color='red'),
        marker=dict(color='red')
    ))
    fig.update_layout(
        title="Predicted Gold Price Movement",
        xaxis_title="Price Movement Stages",
        yaxis_title="Gold Price (USD)",
        template="plotly_white"
    )
    return fig

def plot_candlestick_chart(gold_data: pd.DataFrame) -> go.Figure:
    """Create an interactive candlestick chart for historical gold prices."""
    fig = go.Figure(data=[go.Candlestick(
        x=gold_data.index,
        open=gold_data['Open'],
        high=gold_data['High'],
        low=gold_data['Low'],
        close=gold_data['Close'],
        name="Gold Price"
    )])
    fig.update_layout(
        title="Historical Gold Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
    return fig

@st.cache_data
def fetch_news_data() -> dict:
    """Fetch the latest gold market news using NewsAPI."""
    news_api_url = ("https://newsapi.org/v2/everything?q=gold+price&"
                    "sortBy=publishedAt&apiKey=93f93957d19546dc872e4c8bcd05763e")
    response = requests.get(news_api_url)
    if response.status_code == 200:
        return response.json()
    return {}

# ----------------------------
# Streamlit App Layout & Sidebar
# ----------------------------
st.set_page_config(page_title="Gold Price Prediction", layout="wide")
st.title("📈 Gold Price Prediction Dashboard")

# Sidebar settings
st.sidebar.title("Settings")
historical_days = st.sidebar.slider("Select Historical Days for Data", min_value=7, max_value=90, value=30)

# Fetch data for predictions and historical display
with st.spinner("Fetching latest gold data for predictions..."):
    gold_data_prediction = fetch_gold_data(period="7d", interval="1d")
with st.spinner("Fetching historical gold data..."):
    gold_data_history = fetch_gold_data(period=f"{historical_days}d", interval="1d")
    # Optionally, add technical indicators for display purposes
    gold_data_history_indicators = add_technical_indicators(gold_data_history.copy())

# ----------------------------
# Main Prediction and Visualization Section
# ----------------------------
if st.button("🔮 Predict Tomorrow's Gold Price"):
    predictions = predict_tomorrow_price(gold_data_prediction)

    # Display basic information and metrics
    st.subheader("📊 Tomorrow's Prediction")
    st.write(f"📅 **Today's Date:** {predictions['today_date']}")
    st.write(f"📅 **Latest Available Data:** {predictions['latest_available_date']}")
    st.write(f"🔮 **Prediction for:** {predictions['tomorrow_date']}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📌 Current Gold Price", value=f"${predictions['current_price']:.2f}")
        st.metric(label="📊 Predicted Price", value=f"${predictions['predicted_price']:.2f}")
        st.metric(label="🔴 Breakpoint Price", value=f"${predictions['breakpoint_price']:.2f}")
    with col2:
        st.metric(label="📌 Adjusted Current Price (-20)", value=f"${predictions['adjusted_current_price']:.2f}")
        st.metric(label="📊 Adjusted Predicted Price (-20)", value=f"${predictions['adjusted_predicted_price']:.2f}")
        st.metric(label="🔴 Adjusted Breakpoint (-20)", value=f"${predictions['adjusted_breakpoint_price']:.2f}")

    # Plot interactive prediction chart
    st.plotly_chart(plot_price_movement(predictions), use_container_width=True)

# ----------------------------
# Historical Data Visualization & Table
# ----------------------------
st.subheader("📜 Historical Gold Prices (Last {} Days)".format(historical_days))
st.dataframe(gold_data_history[['Close']].tail(30))
st.plotly_chart(plot_candlestick_chart(gold_data_history), use_container_width=True)

# ----------------------------
# Gold Market News & Sentiment Analysis
# ----------------------------
st.subheader("📰 Latest Gold Market News & Sentiment")
news_data = fetch_news_data()
if news_data and news_data.get("articles"):
    for article in news_data.get("articles", [])[:5]:
        title = article.get("title", "No Title")
        url = article.get("url", "#")
        sentiment = TextBlob(title).sentiment.polarity
        sentiment_text = "🔴 Negative" if sentiment < -0.1 else "🟢 Positive" if sentiment > 0.1 else "⚪ Neutral"
        st.markdown(f"[📌 {title}]({url}) - {sentiment_text}")
else:
    st.write("No news articles available at the moment.")
