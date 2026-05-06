"""
Channel attention for NIDS.

Learns importance weights for each learner channel (LOF, OCSVM, COPOD).
"""

import tensorflow as tf
from tensorflow import keras


class ChannelAttention(keras.layers.Layer):
    """
    Channel Attention Mechanism.

    Output shape: (batch, 1, 1, channels)
    """

    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = keras.layers.Dense(
            max(1, channels // self.reduction_ratio), activation='relu'
        )
        self.fc2 = keras.layers.Dense(channels)
        super().build(input_shape)

    def call(self, x):
        # Global Average Pooling
        avg_pool = keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        # Global Max Pooling
        max_pool = keras.layers.GlobalMaxPooling2D(keepdims=True)(x)

        # Shared FC layers
        avg_feat = self.fc2(self.fc1(avg_pool))
        max_feat = self.fc2(self.fc1(max_pool))

        channel_weights = keras.activations.sigmoid(avg_feat + max_feat)
        self.channel_attention_weights = channel_weights

        return x * channel_weights

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config
