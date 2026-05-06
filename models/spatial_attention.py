"""
Spatial attention for NIDS.

Learns importance weights for each IP (spatial location).
"""

import tensorflow as tf
from tensorflow import keras


class SpatialAttention(keras.layers.Layer):
    """
    Spatial Attention Mechanism.

    Output shape: (batch, height, width, 1)
    """

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv2D(
            1, kernel_size, padding='same', activation='sigmoid'
        )

    def call(self, x):
        # Channel-wise statistics
        avg_pool = keras.backend.mean(x, axis=-1, keepdims=True)
        max_pool = keras.backend.max(x, axis=-1, keepdims=True)

        concat = keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_weights = self.conv(concat)

        self.spatial_attention_weights = spatial_weights
        return x * spatial_weights

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
