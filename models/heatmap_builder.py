"""
Heatmap/image builders for source-destination anomaly tensors.
"""

import numpy as np


class HeatmapBuilder:
    def __init__(self, n_intervals=15, n_channels=3):
        self.n_intervals = n_intervals
        self.n_channels = n_channels

    @staticmethod
    def build_ip_mapping(ip_values, max_ips=451):
        all_ips = sorted(ip_values)
        n_ips = min(len(all_ips), max_ips)
        return n_ips, {ip: idx for idx, ip in enumerate(all_ips[:n_ips])}

    def make_image(self, chunk, learners, ip_to_row, n_ips):
        img = np.zeros((n_ips, self.n_intervals, self.n_channels), dtype=np.float32)
        for t, (_, row) in enumerate(chunk.iterrows()):
            if t >= self.n_intervals:
                break
            ip = str(row.get("ip", ""))
            if ip not in ip_to_row:
                continue
            ip_row = ip_to_row[ip]
            for c, learner in enumerate(learners):
                score = float(row.get(f"score_{learner}", 0.0))
                img[ip_row, t, c] = score
        return img

    def build_heatmap_pair(
        self,
        src_chunk,
        dst_chunk,
        src_learners,
        dst_learners,
        src_ip_to_row,
        dst_ip_to_row,
        n_src_ips,
        n_dst_ips,
    ):
        src_img = self.make_image(src_chunk, src_learners, src_ip_to_row, n_src_ips)
        dst_img = self.make_image(dst_chunk, dst_learners, dst_ip_to_row, n_dst_ips)
        return src_img, dst_img
