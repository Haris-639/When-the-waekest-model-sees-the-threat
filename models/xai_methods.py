"""
GPU-friendly xAI methods for dual-input models.
"""

import tensorflow as tf


def saliency_maps(model, src_batch, dst_batch):
    """Compute saliency maps via input gradients."""
    src = tf.convert_to_tensor(src_batch)
    dst = tf.convert_to_tensor(dst_batch)

    with tf.GradientTape() as tape:
        tape.watch([src, dst])
        preds = model([src, dst], training=False)
    grads = tape.gradient(preds, [src, dst])
    return grads[0].numpy(), grads[1].numpy()


def integrated_gradients(model, src, dst, steps=24):
    """Integrated gradients for dual-input model."""
    src = tf.convert_to_tensor(src, dtype=tf.float32)
    dst = tf.convert_to_tensor(dst, dtype=tf.float32)

    baseline_src = tf.zeros_like(src)
    baseline_dst = tf.zeros_like(dst)

    alphas = tf.linspace(0.0, 1.0, steps)
    src_grads = []
    dst_grads = []

    for alpha in alphas:
        src_step = baseline_src + alpha * (src - baseline_src)
        dst_step = baseline_dst + alpha * (dst - baseline_dst)

        with tf.GradientTape() as tape:
            tape.watch([src_step, dst_step])
            preds = model([src_step, dst_step], training=False)
        grads = tape.gradient(preds, [src_step, dst_step])
        src_grads.append(grads[0])
        dst_grads.append(grads[1])

    avg_src_grads = tf.reduce_mean(tf.stack(src_grads), axis=0)
    avg_dst_grads = tf.reduce_mean(tf.stack(dst_grads), axis=0)

    ig_src = (src - baseline_src) * avg_src_grads
    ig_dst = (dst - baseline_dst) * avg_dst_grads

    return ig_src.numpy(), ig_dst.numpy()
