"""
Model factories for baseline and tunable NIDS models.
"""

import tensorflow as tf


class ModelFactory:
    def __init__(self, n_intervals=15):
        self.n_intervals = n_intervals

    def build_tunable_model(self, hp, n_src_ips, n_dst_ips):
        src_input = tf.keras.Input(shape=(n_src_ips, self.n_intervals, 3), name="src_input")
        dst_input = tf.keras.Input(shape=(n_dst_ips, self.n_intervals, 3), name="dst_input")

        def tunable_branch(inp, prefix):
            x = inp
            n_conv = hp.Int(f"{prefix}_n_conv", min_value=1, max_value=2, step=1)
            for i in range(n_conv):
                filters = hp.Int(f"{prefix}_filters_{i}", 32, 128, step=32)
                x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
                x = tf.keras.layers.Conv2D(
                    filters, (3, 3), activation="relu", padding="same", strides=(2, 2)
                )(x)
            return tf.keras.layers.GlobalAveragePooling2D()(x)

        src_feat = tunable_branch(src_input, "src")
        dst_feat = tunable_branch(dst_input, "dst")
        merged = tf.keras.layers.Concatenate()([src_feat, dst_feat])

        dense_units = hp.Int("dense_units", 64, 256, step=64)
        x = tf.keras.layers.Dense(dense_units, activation="relu")(merged)
        dropout = hp.Float("dropout", 0.2, 0.5, step=0.1)
        x = tf.keras.layers.Dropout(dropout)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=[src_input, dst_input], outputs=output, name="NIDS_CNN_Tuned")
        lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return model

    def build_single_branch_model(self, input_name, n_ips):
        inp = tf.keras.Input(shape=(n_ips, self.n_intervals, 3), name=input_name)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs=inp, outputs=out, name=f"cnn_{input_name}")
