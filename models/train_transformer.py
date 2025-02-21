import numpy as np
import pandas as pd
import tensorflow as tf
from transformer_model import build_transformer_model

# Example: Load your preprocessed data
data = pd.read_csv("data/gold_data.csv")  # Your CSV file with historical data

# Function to create windowed sequences
def create_windows(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 3])  # Assuming column 3 is 'Close'
    return np.array(X), np.array(y)

# Assume your features are normalized and in numpy array format
features = data[['Open','High','Low','Close','Volume']].values  
X, y = create_windows(features, window_size=30)

input_shape = (X.shape[1], X.shape[2])
model = build_transformer_model(
    input_shape, head_size=64, num_heads=4, ff_dim=128,
    num_transformer_blocks=2, mlp_units=[64], dropout=0.1, mlp_dropout=0.1
)
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Save the model for later inference
model.save("models/transformer_gold_model.h5")
