"""
Models module for NIDS with XAI.
Includes base CNN, Attention-Enhanced CNN, and explainability tools.
"""

from .base_cnn import BaseCNN
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention
from .temporal_attention import TemporalAttention
from .attention_enhanced_cnn import AttentionEnhancedCNN
from .responsible_ai_pipeline import ResponsibleAIPipeline
from .network_feature_engineer import NetworkFeatureEngineer
from .anomaly_scoring_engine import AnomalyScoringEngine
from .heatmap_builder import HeatmapBuilder
from .model_factory import ModelFactory
from .xai_methods import saliency_maps, integrated_gradients
from .explainability import AttentionVisualizer

__all__ = [
    'BaseCNN',
    'ChannelAttention',
    'SpatialAttention',
    'TemporalAttention',
    'AttentionEnhancedCNN',
    'ResponsibleAIPipeline',
    'NetworkFeatureEngineer',
    'AnomalyScoringEngine',
    'HeatmapBuilder',
    'ModelFactory',
    'AttentionVisualizer',
    'saliency_maps',
    'integrated_gradients',
]
