"""
Explainability tools for Attention-Enhanced CNN.

Visualizes:
1. Channel Attention: Which learner scores (LOF/OCSVM/COPOD) matter?
2. Spatial Attention: Which IPs are anomalous?
3. Temporal Attention: Which time intervals are critical?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns


class AttentionVisualizer:
    """
    Visualize attention weights from AttentionEnhancedCNN.
    """
    
    def __init__(self, learner_names=None, ip_names=None):
        """
        Initialize visualizer.
        
        Args:
            learner_names: List of learner names, default ['LOF', 'OCSVM', 'COPOD']
            ip_names: List of IP names (or None for numeric indices)
        """
        self.learner_names = learner_names or ['LOF', 'OCSVM', 'COPOD']
        self.ip_names = ip_names

    @staticmethod
    def _to_numpy(values):
        if hasattr(values, 'numpy'):
            return values.numpy()
        return np.asarray(values)
    
    def visualize_channel_attention(self, channel_weights, perspective='src', block=1,
                                    figsize=(8, 4), cmap='RdYlGn'):
        """
        Visualize channel attention weights.
        
        Shows importance of each learner (LOF, OCSVM, COPOD).
        
        Args:
            channel_weights: Tensor of shape (batch, 1, 1, channels) or (1, 1, 1, channels)
            perspective: 'src' or 'dst' for title
            block: Attention block number
            figsize: Figure size
            cmap: Colormap name
            
        Returns:
            fig, ax
        """
        # Extract weights
        if len(channel_weights.shape) == 4:
            weights = channel_weights[0, 0, 0, :]  # (channels,)
        else:
            weights = channel_weights.flatten()

        weights = self._to_numpy(weights)
        
        # Normalize for display
        weights_norm = weights / (weights.sum() + 1e-7)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot
        if len(weights_norm) == len(self.learner_names):
            labels = self.learner_names
        else:
            labels = [f'C{i}' for i in range(len(weights_norm))]
        colors = plt.cm.get_cmap(cmap)(weights_norm)
        bars = ax.bar(labels, weights_norm, color=colors, edgecolor='black', linewidth=1.5)
        
        # Annotations
        for bar, weight in zip(bars, weights_norm):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Channel Attention — {perspective.upper()} Block {block}\n'
                    'Which learner scores matter?',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(weights_norm) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_spatial_attention(self, spatial_weights, n_top_ips=20,
                                    perspective='src', block=1, figsize=(12, 4), cmap='YlOrRd'):
        """
        Visualize spatial attention weights (IP importance).
        
        Shows which IPs are most anomalous.
        
        Args:
            spatial_weights: Tensor of shape (batch, height, width, 1) or (batch, height, 1, 1)
            n_top_ips: Show top N IPs
            perspective: 'src' or 'dst'
            block: Attention block number
            figsize: Figure size
            cmap: Colormap name
            
        Returns:
            fig, ax
        """
        # Extract weights along IP dimension (first spatial axis)
        if len(spatial_weights.shape) == 4:
            # Average over time: (batch, height, width, 1) -> (height,)
            weights = spatial_weights[0, :, :, 0].mean(axis=1)
        else:
            weights = spatial_weights[0, :, 0]

        weights = self._to_numpy(weights)
        
        # Get top IPs
        top_indices = np.argsort(weights)[-n_top_ips:][::-1]
        top_weights = weights[top_indices]
        top_ips = [f'IP_{i}' if self.ip_names is None else self.ip_names[i] 
                   for i in top_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Horizontal bar plot
        colors = plt.cm.get_cmap(cmap)(top_weights / top_weights.max())
        ax.barh(range(len(top_ips)), top_weights, color=colors, edgecolor='black', linewidth=1)
        
        # Annotations
        for i, (ip, weight) in enumerate(zip(top_ips, top_weights)):
            ax.text(weight, i, f'  {weight:.4f}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(top_ips)))
        ax.set_yticklabels(top_ips, fontsize=10)
        ax.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Spatial Attention (Top {n_top_ips} IPs) — {perspective.upper()} Block {block}\n'
                    'Which IPs are most suspicious?',
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_temporal_attention(self, temporal_weights, perspective='src', block=1,
                                     figsize=(10, 4), cmap='Blues'):
        """
        Visualize temporal attention weights (time importance).
        
        Shows which time intervals are critical.
        
        Args:
            temporal_weights: Tensor of shape (batch, 1, time, 1)
            perspective: 'src' or 'dst'
            block: Attention block number
            figsize: Figure size
            cmap: Colormap name
            
        Returns:
            fig, ax
        """
        # Extract weights along time dimension
        if len(temporal_weights.shape) == 4:
            weights = temporal_weights[0, 0, :, 0]  # (time,)
        else:
            weights = temporal_weights[0, :, 0]

        weights = self._to_numpy(weights)
        
        # Normalize
        weights_norm = weights / (weights.sum() + 1e-7)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Line plot with fill
        time_steps = np.arange(len(weights_norm))
        ax.fill_between(time_steps, weights_norm, alpha=0.5, color='steelblue')
        ax.plot(time_steps, weights_norm, 'o-', color='darkblue', linewidth=2, markersize=8)
        
        # Annotations for peaks
        peak_idx = np.argmax(weights_norm)
        ax.annotate(f'Peak: {weights_norm[peak_idx]:.4f}',
                   xy=(peak_idx, weights_norm[peak_idx]),
                   xytext=(peak_idx, weights_norm[peak_idx] + 0.05),
                   ha='center', fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        ax.set_xlabel('Time Interval', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Temporal Attention — {perspective.upper()} Block {block}\n'
                    'Which time intervals are critical?',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(time_steps)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_all_attentions(self, model_outputs_dict, perspective='src',
                                 figsize=(16, 12)):
        """
        Create a comprehensive visualization of all attention types and blocks.
        
        Args:
            model_outputs_dict: Dict from model.attention_weights
            perspective: 'src' or 'dst'
            figsize: Overall figure size
            
        Returns:
            fig, axes
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        for block in [1, 2]:
            # Channel attention
            ax = fig.add_subplot(gs[0, block-1])
            ch_key = f'{perspective}_channel_attn{block}'
            if ch_key in model_outputs_dict:
                weights = model_outputs_dict[ch_key][0, 0, 0, :]
                weights = self._to_numpy(weights)
                weights_norm = weights / (weights.sum() + 1e-7)
                colors = plt.cm.RdYlGn(weights_norm)
                if len(weights_norm) == len(self.learner_names):
                    labels = self.learner_names
                else:
                    labels = [f'C{i}' for i in range(len(weights_norm))]
                ax.bar(labels, weights_norm, color=colors, edgecolor='black')
                ax.set_title(f'Channel Attn Block {block}', fontweight='bold')
                ax.set_ylabel('Importance')
                ax.grid(axis='y', alpha=0.3)
            
            # Spatial attention (heatmap)
            ax = fig.add_subplot(gs[1, block-1])
            sp_key = f'{perspective}_spatial_attn{block}'
            if sp_key in model_outputs_dict:
                weights = model_outputs_dict[sp_key][0, :, :, 0]
                weights = self._to_numpy(weights)
                sns.heatmap(weights, cmap='YlOrRd', ax=ax, cbar=True, square=False)
                ax.set_title(f'Spatial Attn Block {block} (IP x Time)', fontweight='bold')
                ax.set_xlabel('Time Intervals')
                ax.set_ylabel('IPs (Sample)')
            
            # Temporal attention
            ax = fig.add_subplot(gs[2, block-1])
            tm_key = f'{perspective}_temporal_attn{block}'
            if tm_key in model_outputs_dict:
                weights = model_outputs_dict[tm_key][0, 0, :, 0]
                weights = self._to_numpy(weights)
                weights_norm = weights / (weights.sum() + 1e-7)
                ax.fill_between(range(len(weights_norm)), weights_norm, alpha=0.5, color='steelblue')
                ax.plot(weights_norm, 'o-', color='darkblue', linewidth=2)
                ax.set_title(f'Temporal Attn Block {block}', fontweight='bold')
                ax.set_xlabel('Time Interval')
                ax.set_ylabel('Importance')
                ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Attention Visualization — {perspective.upper()} Branch',
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    @staticmethod
    def attention_summary_table(model_outputs_dict, learner_names=None):
        """
        Create a summary table of average attention values across perspectives.
        
        Args:
            model_outputs_dict: Dict from model.attention_weights
            learner_names: List of learner names
            
        Returns:
            Formatted summary as string
        """
        learner_names = learner_names or ['LOF', 'OCSVM', 'COPOD']
        
        summary = "=" * 70 + "\n"
        summary += "ATTENTION WEIGHTS SUMMARY\n"
        summary += "=" * 70 + "\n\n"
        
        for perspective in ['src', 'dst']:
            summary += f"{perspective.upper()} Branch:\n"
            summary += "-" * 70 + "\n"
            
            for block in [1, 2]:
                ch_key = f'{perspective}_channel_attn{block}'
                if ch_key in model_outputs_dict:
                    weights = model_outputs_dict[ch_key][0, 0, 0, :]
                    weights = AttentionVisualizer._to_numpy(weights)
                    weights_norm = weights / (weights.sum() + 1e-7)
                    
                    summary += f"  Block {block} Channel Attention:\n"
                    for learner, weight in zip(learner_names, weights_norm):
                        summary += f"    {learner:10s}: {weight:.4f}\n"
                    summary += "\n"
            
            summary += "\n"
        
        return summary
