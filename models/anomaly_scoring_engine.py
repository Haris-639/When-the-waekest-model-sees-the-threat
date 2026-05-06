"""
Anomaly scoring and evaluation utilities.
"""

import numpy as np
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class AnomalyScoringEngine:
    def __init__(self, contamination=0.05, window_size=2000):
        self.contamination = contamination
        self.window_size = window_size

    def score_windowed(self, df, feature_cols, feature_pairs, learner_name, window_size=None):
        used_window = window_size or self.window_size
        x_raw = df[feature_cols].values.astype(np.float64)
        x_raw = np.where(np.isfinite(x_raw), x_raw, 0.0)
        n_total = len(x_raw)
        scores = np.zeros(n_total, dtype=np.float32)
        n_pairs = len(feature_pairs)

        for start in range(0, n_total, used_window):
            end = min(start + used_window, n_total)
            x_win = x_raw[start:end]
            if len(x_win) < 5:
                continue

            scaler = StandardScaler()
            x_sc = scaler.fit_transform(x_win).astype(np.float32)
            x_sc = np.nan_to_num(x_sc, nan=0.0, posinf=0.0, neginf=0.0)
            n = x_sc.shape[0]
            flag_counts = np.zeros(n, dtype=np.float32)

            for fa, fb in feature_pairs:
                ia = feature_cols.index(fa)
                ib = feature_cols.index(fb)
                x2d = x_sc[:, [ia, ib]]
                if np.std(x2d[:, 0]) < 1e-9 and np.std(x2d[:, 1]) < 1e-9:
                    continue
                try:
                    if learner_name == "LOF":
                        clf = LocalOutlierFactor(
                            n_neighbors=min(3, n - 1),
                            contamination=self.contamination,
                            novelty=False,
                        )
                        flags = (clf.fit_predict(x2d) == -1).astype(np.float32)
                    elif learner_name == "OCSVM":
                        clf = OneClassSVM(
                            nu=self.contamination,
                            kernel="rbf",
                            max_iter=300,
                        )
                        clf.fit(x2d)
                        flags = (clf.predict(x2d) == -1).astype(np.float32)
                    elif learner_name == "KNN":
                        clf = KNN(
                            n_neighbors=min(3, n - 1),
                            contamination=self.contamination,
                        )
                        clf.fit(x2d)
                        flags = clf.labels_.astype(np.float32)
                    elif learner_name == "COPOD":
                        clf = COPOD(contamination=self.contamination)
                        clf.fit(x2d)
                        flags = clf.labels_.astype(np.float32)
                    else:
                        continue
                    flag_counts += flags
                except Exception:
                    continue

            scores[start:end] = flag_counts / max(n_pairs, 1)
        return scores

    @staticmethod
    def evaluate_learner(scored_df, learner_name, perspective=None):
        col = f"score_{learner_name}"
        if col not in scored_df.columns:
            return None, None

        attack_mask = scored_df["label"] == 1
        benign_mask = scored_df["label"] == 0
        flagged = scored_df[col] > 0
        tp = (flagged & attack_mask).sum()
        fn = (~flagged & attack_mask).sum()
        fp = (flagged & benign_mask).sum()
        tn = (~flagged & benign_mask).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return round(tpr, 4), round(fpr, 4)

    @staticmethod
    def evaluate_ensemble(scored_df, learner_list):
        score_cols = [f"score_{l}" for l in learner_list if f"score_{l}" in scored_df.columns]
        ensemble_flag = (scored_df[score_cols] > 0).any(axis=1)
        attack_mask = scored_df["label"] == 1
        benign_mask = scored_df["label"] == 0
        tp = (ensemble_flag & attack_mask).sum()
        fn = (~ensemble_flag & attack_mask).sum()
        fp = (ensemble_flag & benign_mask).sum()
        tn = (~ensemble_flag & benign_mask).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return round(tpr, 4), round(fpr, 4)
