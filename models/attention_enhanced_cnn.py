"""
Attention-Enhanced CNN for NIDS with XAI.

Uses three attention mechanisms:
- Channel: which learner scores matter
- Spatial: which IPs are suspicious
- Temporal: which time intervals are critical
"""

import tensorflow as tf
from tensorflow import keras

from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention
from .temporal_attention import TemporalAttention


class AttentionEnhancedCNN(keras.Model):
    """
    Dual-branch CNN (src, dst) with attention blocks for XAI.
    """

    def __init__(self, n_src_ips=451, n_dst_ips=451,
                 n_intervals=15, n_channels=3,
                 filters=32, dense_units=64, dropout=0.3,
                 name='AttentionEnhancedCNN', **kwargs):
        super().__init__(name=name, **kwargs)

        self.n_src_ips = n_src_ips
        self.n_dst_ips = n_dst_ips
        self.n_intervals = n_intervals
        self.n_channels = n_channels

        # Source branch
        self.src_conv1 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name='src_conv1'
        )
        self.src_channel_attn1 = ChannelAttention(name='src_channel_attn1')
        self.src_spatial_attn1 = SpatialAttention(name='src_spatial_attn1')
        self.src_temporal_attn1 = TemporalAttention(name='src_temporal_attn1')

        self.src_conv2 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same',
            strides=(2, 2), name='src_conv2'
        )
        self.src_conv3 = keras.layers.Conv2D(
            filters * 2, (3, 3), activation='relu', padding='same', name='src_conv3'
        )
        self.src_channel_attn2 = ChannelAttention(name='src_channel_attn2')
        self.src_spatial_attn2 = SpatialAttention(name='src_spatial_attn2')
        self.src_conv4 = keras.layers.Conv2D(
            filters * 2, (3, 3), activation='relu', padding='same',
            strides=(2, 2), name='src_conv4'
        )
        self.src_gap = keras.layers.GlobalAveragePooling2D(name='src_gap')

        # Destination branch
        self.dst_conv1 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name='dst_conv1'
        )
        self.dst_channel_attn1 = ChannelAttention(name='dst_channel_attn1')
        self.dst_spatial_attn1 = SpatialAttention(name='dst_spatial_attn1')
        self.dst_temporal_attn1 = TemporalAttention(name='dst_temporal_attn1')

        self.dst_conv2 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same',
            strides=(2, 2), name='dst_conv2'
        )
        self.dst_conv3 = keras.layers.Conv2D(
            filters * 2, (3, 3), activation='relu', padding='same', name='dst_conv3'
        )
        self.dst_channel_attn2 = ChannelAttention(name='dst_channel_attn2')
        self.dst_spatial_attn2 = SpatialAttention(name='dst_spatial_attn2')
        self.dst_conv4 = keras.layers.Conv2D(
            filters * 2, (3, 3), activation='relu', padding='same',
            strides=(2, 2), name='dst_conv4'
        )
        self.dst_gap = keras.layers.GlobalAveragePooling2D(name='dst_gap')

        # Classifier head
        self.merge = keras.layers.Concatenate(name='merge')
        self.dense = keras.layers.Dense(dense_units, activation='relu', name='dense')
        self.dropout_layer = keras.layers.Dropout(dropout)
        self.output_layer = keras.layers.Dense(1, activation='sigmoid', name='output')

        self.attention_weights = {}

    def call(self, inputs, training=False, return_attention=False):
        src_x, dst_x = inputs

        # Source branch
        src_x = self.src_conv1(src_x)
        src_x = self.src_channel_attn1(src_x)
        src_x = self.src_spatial_attn1(src_x)
        src_x = self.src_temporal_attn1(src_x)
        self.attention_weights['src_channel_attn1'] = self.src_channel_attn1.channel_attention_weights
        self.attention_weights['src_spatial_attn1'] = self.src_spatial_attn1.spatial_attention_weights
        self.attention_weights['src_temporal_attn1'] = self.src_temporal_attn1.temporal_attention_weights

        src_x = self.src_conv2(src_x)
        src_x = self.src_conv3(src_x)
        src_x = self.src_channel_attn2(src_x)
        src_x = self.src_spatial_attn2(src_x)
        self.attention_weights['src_channel_attn2'] = self.src_channel_attn2.channel_attention_weights
        self.attention_weights['src_spatial_attn2'] = self.src_spatial_attn2.spatial_attention_weights

        src_x = self.src_conv4(src_x)
        src_feat = self.src_gap(src_x)

        # Destination branch
        dst_x = self.dst_conv1(dst_x)
        dst_x = self.dst_channel_attn1(dst_x)
        dst_x = self.dst_spatial_attn1(dst_x)
        dst_x = self.dst_temporal_attn1(dst_x)
        self.attention_weights['dst_channel_attn1'] = self.dst_channel_attn1.channel_attention_weights
        self.attention_weights['dst_spatial_attn1'] = self.dst_spatial_attn1.spatial_attention_weights
        self.attention_weights['dst_temporal_attn1'] = self.dst_temporal_attn1.temporal_attention_weights

        dst_x = self.dst_conv2(dst_x)
        dst_x = self.dst_conv3(dst_x)
        dst_x = self.dst_channel_attn2(dst_x)
        dst_x = self.dst_spatial_attn2(dst_x)
        self.attention_weights['dst_channel_attn2'] = self.dst_channel_attn2.channel_attention_weights
        self.attention_weights['dst_spatial_attn2'] = self.dst_spatial_attn2.spatial_attention_weights

        dst_x = self.dst_conv4(dst_x)
        dst_feat = self.dst_gap(dst_x)

        merged = self.merge([src_feat, dst_feat])
        x = self.dense(merged)
        x = self.dropout_layer(x, training=training)
        output = self.output_layer(x)

        if return_attention:
            return output, self.attention_weights

        return output

    def get_attention_weights(self, perspective='src', block=1, attention_type='channel'):
        key = f'{perspective}_{attention_type}_attn{block}'
        return self.attention_weights.get(key, None)

    def get_config(self):
        return {
            'n_src_ips': self.n_src_ips,
            'n_dst_ips': self.n_dst_ips,
            'n_intervals': self.n_intervals,
            'n_channels': self.n_channels,
        }
