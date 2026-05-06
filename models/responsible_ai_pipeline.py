"""
Helpers to keep responsible-ai notebook output-focused.
"""

import numpy as np
import tensorflow as tf

from .attention_enhanced_cnn import AttentionEnhancedCNN
from .xai_methods import integrated_gradients, saliency_maps


class ResponsibleAIPipeline:
    """Build and explain AttentionEnhancedCNN predictions."""

    def __init__(self, learner_names=None):
        self.learner_names = learner_names or ["LOF", "OCSVM", "COPOD"]

    def build_attention_cnn(
        self,
        n_src_ips,
        n_dst_ips,
        n_intervals,
        n_channels,
        filters=32,
        dense_units=64,
        dropout=0.3,
        learning_rate=1e-4,
    ):
        model = AttentionEnhancedCNN(
            n_src_ips=n_src_ips,
            n_dst_ips=n_dst_ips,
            n_intervals=n_intervals,
            n_channels=n_channels,
            filters=filters,
            dense_units=dense_units,
            dropout=dropout,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return model

    def summarize_attention(self, attn_weights):
        lines = []
        for perspective in ["src", "dst"]:
            perspective_upper = perspective.upper()
            for block in [1, 2]:
                ch_key = f"{perspective}_channel_attn{block}"
                if ch_key in attn_weights:
                    weights = attn_weights[ch_key][0, 0, 0, :]
                    weights_norm = weights / (weights.sum() + 1e-7)
                    learner_txt = " ".join(
                        f"{learner}={w:.3f}"
                        for learner, w in zip(self.learner_names, weights_norm)
                    )
                    lines.append(
                        f"{perspective_upper} Block {block} channel: {learner_txt}"
                    )

                tm_key = f"{perspective}_temporal_attn{block}"
                if tm_key in attn_weights:
                    weights = attn_weights[tm_key][0, 0, :, 0]
                    weights_norm = weights / (weights.sum() + 1e-7)
                    peak_idx = int(np.argmax(weights_norm))
                    lines.append(
                        f"{perspective_upper} Block {block} temporal peak: "
                        f"interval {peak_idx} (weight={weights_norm[peak_idx]:.4f})"
                    )

                sp_key = f"{perspective}_spatial_attn{block}"
                if sp_key in attn_weights:
                    weights = attn_weights[sp_key][0, :, :, 0]
                    ip_importance = weights.mean(axis=1)
                    top_3_indices = np.argsort(ip_importance)[-3:][::-1]
                    top_txt = " ".join(
                        f"IP_{idx}={ip_importance[idx]:.3f}" for idx in top_3_indices
                    )
                    lines.append(
                        f"{perspective_upper} Block {block} spatial top: {top_txt}"
                    )
        return lines

    def gradient_xai(self, model, sample_src, sample_dst, steps=24):
        sal_src, sal_dst = saliency_maps(model, sample_src, sample_dst)
        ig_src, ig_dst = integrated_gradients(
            model, sample_src, sample_dst, steps=steps
        )
        return {
            "saliency_src": sal_src,
            "saliency_dst": sal_dst,
            "ig_src": ig_src,
            "ig_dst": ig_dst,
        }
