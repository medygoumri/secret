import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
import requests
import plotly.graph_objects as go
from textblob import TextBlob

# ----------------------------
# Load Pre-trained Models & Scaler (for LSTM/XGBoost predictions)
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

@st.cache_data
def fetch_gold_api_price() -> float:
    """
    Fetch the current gold price from GoldAPI.
    Returns the price as a float.
    """
    headers = {
        "x-access-token": "goldapi-1ppsm6y8o4sp-io",
        "Content-Type": "application/json"
    }
    url = "https://www.goldapi.io/api/XAU/USD"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # st.write("GoldAPI Response:", data)  # Uncomment for debugging
        return data.get("price")
    else:
        st.error("Error fetching data from GoldAPI")
        return None

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators (SMA and EMA) and add them to the DataFrame."""
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def predict_tomorrow_price(gold_data: pd.DataFrame) -> dict:
    """
    Predict tomorrow's gold price using the LSTM and XGBoost models.
    Also fetches the current gold price from GoldAPI to compute the difference.
    Returns a dictionary with various price metrics.
    """
    # Get the latest Yahoo Finance data
    gold_latest = gold_data.tail(1)
    latest_available_date = gold_latest.index[-1].strftime('%Y-%m-%d')

    if 'Close' not in gold_latest.columns:
        st.error("Error: 'Close' price data is missing in the dataset.")
        st.stop()

    current_price = float(gold_latest['Close'].values[0])
    
    # Fetch current price from GoldAPI
    goldapi_price = fetch_gold_api_price()
    if goldapi_price is None:
        st.error("Error: Unable to fetch current gold price from GoldAPI.")
        st.stop()
    
    # Calculate the difference between Yahoo and GoldAPI prices
    diff = current_price - goldapi_price

    # Get tomorrow's date
    tomorrow_date = (datetime.datetime.strptime(latest_available_date, '%Y-%m-%d') +
                     datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    # Preprocess features for prediction
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

    # Adjusted prices using the difference between Yahoo and GoldAPI prices
    adjusted_current_price = current_price - diff      # This equals the GoldAPI price
    adjusted_predicted_price = predicted_price - diff
    adjusted_breakpoint_price = breakpoint_price - diff

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
        "diff": diff  # For debugging/reference
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
        name='Adjusted Prediction (GoldAPI Diff)',
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

def plot_predicted_trading_chart(predictions: dict) -> go.Figure:
    """
    Plot a candlestick chart for tomorrow's predicted prices in a trading platform style.
    Assumptions:
      - Predicted Open: current price (from Yahoo Finance)
      - Predicted Close: predicted price from the model
      - Predicted High: maximum of current, predicted, and breakpoint prices
      - Predicted Low: minimum of current, predicted, and breakpoint prices
    """
    predicted_open = predictions["current_price"]
    predicted_close = predictions["predicted_price"]
    predicted_high = max(predictions["current_price"], predictions["predicted_price"], predictions["breakpoint_price"])
    predicted_low = min(predictions["current_price"], predictions["predicted_price"], predictions["breakpoint_price"])
    
    fig = go.Figure(data=[go.Candlestick(
        x=[predictions["tomorrow_date"]],
        open=[predicted_open],
        high=[predicted_high],
        low=[predicted_low],
        close=[predicted_close],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(
        title="Predicted Trading View for Tomorrow",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )
    return fig

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
# Main Prediction and Visualization Section (LSTM + XGBoost)
# ----------------------------
if st.button("🔮 Predict Tomorrow's Gold Price"):
    predictions = predict_tomorrow_price(gold_data_prediction)

    # Display basic information and metrics
    st.subheader("📊 Tomorrow's Prediction (LSTM/XGBoost)")
    st.write(f"📅 **Today's Date:** {predictions['today_date']}")
    st.write(f"📅 **Latest Available Data:** {predictions['latest_available_date']}")
    st.write(f"🔮 **Prediction for:** {predictions['tomorrow_date']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📌 Current Gold Price (Yahoo)", value=f"${predictions['current_price']:.2f}")
        st.metric(label="📊 Predicted Price (Yahoo)", value=f"${predictions['predicted_price']:.2f}")
        st.metric(label="🔴 Breakpoint Price (Yahoo)", value=f"${predictions['breakpoint_price']:.2f}")
    with col2:
        st.metric(label="📌 Adjusted Current Price (GoldAPI)", value=f"${predictions['adjusted_current_price']:.2f}")
        st.metric(label="📊 Adjusted Predicted Price", value=f"${predictions['adjusted_predicted_price']:.2f}")
        st.metric(label="🔴 Adjusted Breakpoint Price", value=f"${predictions['adjusted_breakpoint_price']:.2f}")
    
    st.write(f"**Price Difference (Yahoo - GoldAPI):** ${predictions['diff']:.2f}")
    st.plotly_chart(plot_price_movement(predictions), use_container_width=True)

# ----------------------------
# Historical Data Visualization & Table
# ----------------------------
st.subheader(f"📜 Historical Gold Prices (Last {historical_days} Days)")
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

# ----------------------------
# Predicted Trading View Chart & Details
# ----------------------------
if 'predictions' in locals():
    st.subheader("📈 Predicted Trading View for Tomorrow (LSTM/XGBoost)")
    trading_chart = plot_predicted_trading_chart(predictions)
    st.plotly_chart(trading_chart, use_container_width=True)
    
    predicted_open = predictions["current_price"]
    predicted_close = predictions["predicted_price"]
    predicted_high = max(predictions["current_price"], predictions["predicted_price"], predictions["breakpoint_price"])
    predicted_low = min(predictions["current_price"], predictions["predicted_price"], predictions["breakpoint_price"])
    
    st.markdown("### Predicted Prices Details (LSTM/XGBoost)")
    st.write(f"**Predicted Open:** ${predicted_open:.2f}")
    st.write(f"**Predicted High:** ${predicted_high:.2f}")
    st.write(f"**Predicted Low:** ${predicted_low:.2f}")
    st.write(f"**Predicted Close:** ${predicted_close:.2f}")

# ----------------------------
# Transformer Model Prediction Section
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section (Tomorrow Prediction)
# ----------------------------
if st.button("🔮 Predict Tomorrow's Gold Price (Transformer Model)"):
    # Load custom objects (if your Transformer model uses any custom layers)
    try:
        from models.transformer_model import transformer_encoder, build_transformer_model
        custom_objects = {
            "transformer_encoder": transformer_encoder,
            "build_transformer_model": build_transformer_model,
            "mse": tf.keras.losses.MeanSquaredError()
        }
    except ImportError:
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    
    # Load the Transformer model for inference (compile=False)
    transformer_model = tf.keras.models.load_model(
        "models/transformer_gold_model.h5",
        custom_objects=custom_objects,
        compile=False
    )
    
    # Load historical gold data from CSV (daily data)
    transformer_data = pd.read_csv("data/gold_data.csv", index_col=0)
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    transformer_data[cols] = transformer_data[cols].apply(pd.to_numeric, errors='coerce')
    transformer_data.dropna(inplace=True)
    features_transformer = transformer_data[cols].values.astype(np.float32)
    
    window_size = 30  # Use the last 30 days for prediction
    if len(features_transformer) < window_size:
        st.error("Not enough data to perform transformer prediction.")
        st.stop()
    
    # Prepare the input window (last 30 days)
    window_data = features_transformer[-window_size:]
    df_window = pd.DataFrame(window_data, columns=cols)
    
    # Load the scaler used during training for input features.
    # (If not available, fit one on the available features—but this may not match training exactly.)
    try:
        scaler_transformer = joblib.load("scaler_transformer.pkl")
        st.write("Loaded saved input scaler.")
    except Exception as e:
        st.warning("Input scaler not found. Fitting scaler on current data (may differ from training).")
        from sklearn.preprocessing import StandardScaler
        scaler_transformer = StandardScaler()
        scaler_transformer.fit(features_transformer)
    
    # Transform the window data (pass a DataFrame to preserve feature names)
    window_data_scaled = scaler_transformer.transform(df_window.values)
    
    # Add batch dimension: shape becomes (1, window_size, num_features)
    window_data_scaled = window_data_scaled.reshape(1, window_size, len(cols))
    
    # Make prediction using the Transformer model on the scaled data
    transformer_prediction = transformer_model.predict(window_data_scaled)
    raw_pred = transformer_prediction[0][0]
    
    # Now load the target scaler (used to scale the 'Close' during training) so we can invert the scaling.
    try:
        target_scaler = joblib.load("scaler_target.pkl")
        st.write("Loaded saved target scaler.")
    except Exception as e:
        st.warning("Target scaler not found. Fitting target scaler on current 'Close' values (may differ from training).")
        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler()
        target_scaler.fit(transformer_data[['Close']])
    
    # Inverse transform the raw prediction to get the actual price.
    # Note: target_scaler expects a 2D array.
    predicted_price_transformer = float(target_scaler.inverse_transform([[raw_pred]])[0][0])
    
    # Get the current gold price from the CSV (Yahoo 'Close' from last row)
    current_price_transformer = float(features_transformer[-1, 3])
    
    st.subheader("📊 Transformer Model Prediction (Daily)")
    st.write(f"📌 Current Gold Price (Yahoo): ${current_price_transformer:.2f}")
    st.write(f"🔮 Predicted Gold Price for Tomorrow (Transformer): ${predicted_price_transformer:.2f}")

# ----------------------------
# Transformer Model Next Hour Prediction Section (Hourly Data)
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section (Daily)
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section (Daily Live Data)
# ----------------------------
# ----------------------------
# Transformer Model Prediction Section (Daily Live Data)
# ----------------------------
if st.button("🔮 Predict Tomorrow's Gold Price (Transformer Model - Live Data)"):
    # Load custom objects (if your Transformer model uses any custom layers)
    try:
        from models.transformer_model import transformer_encoder, build_transformer_model
        custom_objects = {
            "transformer_encoder": transformer_encoder,
            "build_transformer_model": build_transformer_model,
            "mse": tf.keras.losses.MeanSquaredError()
        }
    except ImportError:
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    
    # Load the Transformer model for inference (without compiling)
    transformer_model = tf.keras.models.load_model(
        "models/transformer_gold_model.h5",
        custom_objects=custom_objects,
        compile=False
    )
    
    # Fetch live daily data using yfinance (increasing period to 60d to ensure sufficient data)
    live_data = yf.download("GC=F", period="60d", interval="1d")
    if live_data.empty:
        st.error("Failed to fetch live data from Yahoo Finance.")
        st.stop()
    
    st.write("Live data columns:", live_data.columns.tolist())
    
    # Define the required columns for our model
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in live_data.columns:
            st.warning(f"Column '{col}' not found in live data. Filling with default value (0.0).")
            live_data[col] = 0.0
    
    # Select and clean the required columns
    live_data = live_data[required_cols]
    live_data = live_data.apply(pd.to_numeric, errors='coerce')
    live_data.dropna(inplace=True)
    
    # Convert to NumPy array
    features_live = live_data.values.astype(np.float32)
    
    window_size = 30  # Use the last 30 days for prediction
    if len(features_live) < window_size:
        st.error(f"Not enough live data to perform prediction. Data available: {len(features_live)} days.")
        st.stop()
    
    # Prepare the input window (last 30 days)
    window_data = features_live[-window_size:]
    df_window = pd.DataFrame(window_data, columns=required_cols)
    
    # Load or fit the input scaler (ideally, use the scaler from training)
    try:
        scaler_transformer = joblib.load("scaler_transformer.pkl")
        st.write("Loaded saved input scaler.")
    except Exception as e:
        st.warning("Input scaler not found. Fitting scaler on live data (may differ from training).")
        from sklearn.preprocessing import StandardScaler
        scaler_transformer = StandardScaler()
        scaler_transformer.fit(features_live)
    
    # Transform the window data and add batch dimension
    window_data_scaled = scaler_transformer.transform(df_window.values)
    window_data_scaled = window_data_scaled.reshape(1, window_size, len(required_cols))
    
    # Make prediction using the Transformer model on the scaled data
    transformer_prediction = transformer_model.predict(window_data_scaled)
    raw_pred = transformer_prediction[0][0]  # ensure raw_pred is computed
    
    # Try loading the target scaler to inverse-transform the prediction
    try:
        target_scaler = joblib.load("scaler_target.pkl")
        st.write("Loaded saved target scaler.")
        predicted_price_transformer = float(target_scaler.inverse_transform([[raw_pred]])[0][0])
    except Exception as e:
        st.warning("Target scaler not found or error in inverse transforming. Using raw model output.")
        predicted_price_transformer = float(raw_pred)
    
    # Get the current price from live data (Yahoo 'Close' of the last day)
    current_price_transformer = float(features_live[-1, 3])
    
    st.subheader("📊 Transformer Model Prediction (Daily Live Data)")
    st.write(f"📌 Current Gold Price (Yahoo Live): ${current_price_transformer:.2f}")
    st.write(f"🔮 Predicted Gold Price for Tomorrow (Transformer): ${predicted_price_transformer:.2f}")


