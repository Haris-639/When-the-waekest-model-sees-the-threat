"""
Temporal attention for NIDS.

Learns importance weights for each time interval.
"""

import tensorflow as tf
from tensorflow import keras


class TemporalAttention(keras.layers.Layer):
    """
    Temporal Attention Mechanism.

    Output shape: (batch, 1, time, 1)
    """

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1d = keras.layers.Conv1D(
            1, self.kernel_size, padding='same', activation='sigmoid'
        )
        super().build(input_shape)

    def call(self, x):
        # Collapse spatial dimensions: (B, H, W, C) -> (B, W, H*C)
        batch_size = keras.backend.shape(x)[0]
        height = keras.backend.shape(x)[1]
        width = keras.backend.shape(x)[2]
        channels = keras.backend.shape(x)[3]

        x_reshaped = keras.backend.reshape(x, [batch_size, width, height * channels])
        temporal_weights = self.conv1d(x_reshaped)
        temporal_weights = keras.backend.reshape(temporal_weights, [batch_size, 1, width, 1])

        self.temporal_attention_weights = temporal_weights
        return x * temporal_weights

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
