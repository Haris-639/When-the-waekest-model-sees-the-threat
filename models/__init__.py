"""
Models module for NIDS with XAI.
Includes base CNN, Attention-Enhanced CNN, and explainability tools.
"""

from .base_cnn import BaseCNN
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention
from .temporal_attention import TemporalAttention
from .attention_enhanced_cnn import AttentionEnhancedCNN
from .xai_methods import saliency_maps, integrated_gradients
from .explainability import AttentionVisualizer

__all__ = [
    'BaseCNN',
    'ChannelAttention',
    'SpatialAttention',
    'TemporalAttention',
    'AttentionEnhancedCNN',
    'AttentionVisualizer',
    'saliency_maps',
    'integrated_gradients',
]
