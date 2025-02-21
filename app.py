import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

# Load models
lstm_model = tf.keras.models.load_model("lstm_gold_model.h5")
xgb_model = joblib.load("xgb_gold_model.pkl")
scaler = joblib.load("scaler.pkl")

# Get today's date dynamically
today_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Streamlit Title
st.title("📈 Gold Price Prediction (XAU/USD)")

# Button to Predict
if st.button("🔮 Predict Gold Price for Tomorrow"):

    # Fetch the latest gold price (ensures updated data)
    gold_data = yf.download("GC=F", period="7d", interval="1d")  # Get last 7 days for safety

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

        # Get tomorrow's date dynamically
        tomorrow_date = (datetime.datetime.strptime(latest_available_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

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

        # **Calculate Adjusted Predictions (Subtract 20)**
        adjusted_current_price = current_price - 20
        adjusted_predicted_price = predicted_price - 20
        adjusted_breakpoint_price = breakpoint_price - 20

        # **Display Predictions**
        st.write(f"📅 **Today's Date:** {today_date}")
        st.write(f"📅 **Latest Available Data:** {latest_available_date}")
        st.write(f"🔮 **Prediction for:** {tomorrow_date}")
        
        st.subheader("📌 Normal Predictions")
        st.write(f"📌 **Current Gold Price:** **${current_price:.2f}**")
        st.write(f"📊 **Predicted Gold Price for Tomorrow:** **${predicted_price:.2f}**")
        st.write(f"🔴 **Breakpoint Price:** **${breakpoint_price:.2f}**")

        st.subheader("📉 Adjusted Predictions (-20)")
        st.write(f"📌 **Adjusted Current Gold Price:** **${adjusted_current_price:.2f}**")
        st.write(f"📊 **Adjusted Predicted Gold Price for Tomorrow:** **${adjusted_predicted_price:.2f}**")
        st.write(f"🔴 **Adjusted Breakpoint Price:** **${adjusted_breakpoint_price:.2f}**")

        # **Visualization**
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(["Current Price", "Breakpoint", "Predicted Price"],
                [current_price, breakpoint_price, predicted_price], marker='o', linestyle='dashed', color='b', label="Normal Prediction")

        ax.plot(["Current Price", "Breakpoint", "Predicted Price"],
                [adjusted_current_price, adjusted_breakpoint_price, adjusted_predicted_price], marker='o', linestyle='dashed', color='r', label="Adjusted Prediction (-20)")

        # Add labels
        ax.set_xlabel("Price Movement Stages")
        ax.set_ylabel("Gold Price (USD)")
        ax.set_title("Predicted Gold Price Movement and Breakpoint")

        # Annotate points
        ax.text(0, current_price, f"${current_price:.2f}", ha='right', fontsize=10)
        ax.text(1, breakpoint_price, f"${breakpoint_price:.2f}", ha='center', fontsize=10, color='red')
        ax.text(2, predicted_price, f"${predicted_price:.2f}", ha='left', fontsize=10, color='green')

        # Annotate adjusted points
        ax.text(0, adjusted_current_price, f"${adjusted_current_price:.2f}", ha='right', fontsize=10, color='darkred')
        ax.text(1, adjusted_breakpoint_price, f"${adjusted_breakpoint_price:.2f}", ha='center', fontsize=10, color='purple')
        ax.text(2, adjusted_predicted_price, f"${adjusted_predicted_price:.2f}", ha='left', fontsize=10, color='orange')

        ax.legend()
        
        # Show the plot in Streamlit
        st.pyplot(fig)
