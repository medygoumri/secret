import tensorflow as tf
from tensorflow.keras import layers, Model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x_attn = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Add()([x, x_attn])
    x_ff = layers.LayerNormalization(epsilon=1e-6)(x)
    x_ff = layers.Dense(ff_dim, activation="relu")(x_ff)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    x = layers.Add()([x, x_ff])
    return x

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    for units in mlp_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    model = Model(inputs, outputs)
    return model
