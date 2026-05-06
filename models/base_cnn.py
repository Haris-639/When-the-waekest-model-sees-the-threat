"""
Base CNN model for NIDS (Network Intrusion Detection System).
Refactored from the original dual-branch architecture.
"""

import tensorflow as tf
from tensorflow import keras


class BaseCNN(keras.Model):
    """
    Dual-branch CNN for source and destination traffic anomaly detection.
    
    Architecture:
    - Two parallel branches (src, dst) with identical structure
    - Each branch: 2x Conv2D with strided pooling + GAP + Dense
    - Merged feature concatenation
    - Final classifier head
    
    Attributes:
        n_src_ips (int): Number of source IPs (typically 451)
        n_dst_ips (int): Number of destination IPs (typically 451)
        n_intervals (int): Number of time intervals (typically 15)
        n_channels (int): Number of learner channels (typically 3: LOF, OCSVM, COPOD)
    """
    
    def __init__(self, n_src_ips=451, n_dst_ips=451, 
                 n_intervals=15, n_channels=3, 
                 filters=32, dense_units=64, dropout=0.3,
                 name='BaseCNN', **kwargs):
        """
        Initialize Base CNN.
        
        Args:
            n_src_ips: Source IP dimension
            n_dst_ips: Destination IP dimension
            n_intervals: Time interval dimension
            n_channels: Learner/feature dimension
            filters: Initial Conv2D filter count
            dense_units: Dense layer units before output
            dropout: Dropout rate
        """
        super().__init__(name=name, **kwargs)
        
        self.n_src_ips = n_src_ips
        self.n_dst_ips = n_dst_ips
        self.n_intervals = n_intervals
        self.n_channels = n_channels
        
        # Source branch
        self.src_input = keras.layers.Input(
            shape=(n_src_ips, n_intervals, n_channels), name='src_input')
        self.src_conv1 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name='src_conv1')
        self.src_conv2 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', 
            strides=(2, 2), name='src_conv2')
        self.src_conv3 = keras.layers.Conv2D(
            filters*2, (3, 3), activation='relu', padding='same', name='src_conv3')
        self.src_conv4 = keras.layers.Conv2D(
            filters*2, (3, 3), activation='relu', padding='same',
            strides=(2, 2), name='src_conv4')
        self.src_gap = keras.layers.GlobalAveragePooling2D(name='src_gap')
        
        # Destination branch
        self.dst_input = keras.layers.Input(
            shape=(n_dst_ips, n_intervals, n_channels), name='dst_input')
        self.dst_conv1 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name='dst_conv1')
        self.dst_conv2 = keras.layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same',
            strides=(2, 2), name='dst_conv2')
        self.dst_conv3 = keras.layers.Conv2D(
            filters*2, (3, 3), activation='relu', padding='same', name='dst_conv3')
        self.dst_conv4 = keras.layers.Conv2D(
            filters*2, (3, 3), activation='relu', padding='same',
            strides=(2, 2), name='dst_conv4')
        self.dst_gap = keras.layers.GlobalAveragePooling2D(name='dst_gap')
        
        # Merger and classifier head
        self.merge = keras.layers.Concatenate(name='merge')
        self.dense = keras.layers.Dense(dense_units, activation='relu', name='dense')
        self.dropout = keras.layers.Dropout(dropout)
        self.output_layer = keras.layers.Dense(1, activation='sigmoid', name='output')
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: [src_data, dst_data] - two tensors
            training: Boolean for dropout behavior
            
        Returns:
            Logits for binary classification
        """
        src_x, dst_x = inputs
        
        # Source branch
        src_x = self.src_conv1(src_x)
        src_x = self.src_conv2(src_x)
        src_x = self.src_conv3(src_x)
        src_x = self.src_conv4(src_x)
        src_feat = self.src_gap(src_x)
        
        # Destination branch
        dst_x = self.dst_conv1(dst_x)
        dst_x = self.dst_conv2(dst_x)
        dst_x = self.dst_conv3(dst_x)
        dst_x = self.dst_conv4(dst_x)
        dst_feat = self.dst_gap(dst_x)
        
        # Merge and classify
        merged = self.merge([src_feat, dst_feat])
        x = self.dense(merged)
        x = self.dropout(x, training=training)
        output = self.output_layer(x)
        
        return output
    
    def get_config(self):
        """Serialize model for saving."""
        return {
            'n_src_ips': self.n_src_ips,
            'n_dst_ips': self.n_dst_ips,
            'n_intervals': self.n_intervals,
            'n_channels': self.n_channels,
        }
