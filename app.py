import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load models
lstm_model = tf.keras.models.load_model("lstm_gold_model.h5")
xgb_model = joblib.load("xgb_gold_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit Title
st.title("📈 Gold Price Prediction (XAU/USD)")

# Button to Predict
if st.button("🔮 Predict Gold Price for Tomorrow"):

    # Fetch latest Gold price
    gold_latest = yf.download("GC=F", period="5d", interval="1d")
    gold_latest = gold_latest.tail(1)

    # Extract current price
    current_price = gold_latest['Close'].values[0]

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

    # Define predicted price change
    predicted_price_change = lstm_prediction[0][0] * (current_price * 0.02)  # Assuming 2% price fluctuation
    predicted_price = current_price + (predicted_price_change if xgb_prediction[0] == 1 else -predicted_price_change)

    # Define breakpoint price
    breakpoint_price = current_price * 1.01 if xgb_prediction[0] == 1 else current_price * 0.99

    # Display predictions
    st.write(f"📌 **Current Gold Price:** **${current_price:.2f}**")
    st.write(f"📊 **Predicted Gold Price for Tomorrow:** **${predicted_price:.2f}**")
    st.write(f"🔴 **Breakpoint Price:** **${breakpoint_price:.2f}**")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(["Current Price", "Breakpoint", "Predicted Price"],
            [current_price, breakpoint_price, predicted_price], marker='o', linestyle='dashed', color='b')

    # Add labels
    ax.set_xlabel("Price Movement Stages")
    ax.set_ylabel("Gold Price (USD)")
    ax.set_title("Predicted Gold Price Movement and Breakpoint")

    # Annotate points
    ax.text(0, current_price, f"${current_price:.2f}", ha='right', fontsize=10)
    ax.text(1, breakpoint_price, f"${breakpoint_price:.2f}", ha='center', fontsize=10, color='red')
    ax.text(2, predicted_price, f"${predicted_price:.2f}", ha='left', fontsize=10, color='green')

    # Show the plot in Streamlit
    st.pyplot(fig)
