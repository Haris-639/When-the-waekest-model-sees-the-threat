"""
Feature-engineering utilities for responsible AI NIDS workflow.
"""

import ipaddress
import numpy as np
import pandas as pd


class NetworkFeatureEngineer:
    def __init__(self, bucket_midpoints=None):
        self.bucket_midpoints = bucket_midpoints or {
            "NUM_PKTS_UP_TO_128_BYTES": 64,
            "NUM_PKTS_128_TO_256_BYTES": 192,
            "NUM_PKTS_256_TO_512_BYTES": 384,
            "NUM_PKTS_512_TO_1024_BYTES": 768,
            "NUM_PKTS_1024_TO_1514_BYTES": 1269,
        }

    @staticmethod
    def is_internal_ip(ip_str):
        try:
            return ipaddress.ip_address(str(ip_str).strip()).is_private
        except ValueError:
            return False

    def compute_pkt_std(self, group_df):
        all_sizes = []
        all_counts = []
        for col, mid in self.bucket_midpoints.items():
            if col in group_df.columns:
                counts = group_df[col].values
                all_sizes.extend([mid] * len(counts))
                all_counts.extend(counts.tolist())

        if not all_counts or sum(all_counts) == 0:
            return 0.0

        sizes = np.array(all_sizes, dtype=float)
        counts = np.array(all_counts, dtype=float)
        total = counts.sum()
        mean = (sizes * counts).sum() / total
        var = (counts * (sizes - mean) ** 2).sum() / total
        return float(np.sqrt(var))

    def compute_src_features(self, group, ip):
        sizes = group["IN_BYTES"] + group["OUT_BYTES"]
        return {
            "f1_n_fwd_pkts": group["IN_PKTS"].sum(),
            "f2_n_bwd_pkts": group["OUT_PKTS"].sum(),
            "f3_sum_flx_dur": group["FLOW_DURATION_MILLISECONDS"].sum(),
            "f4_sum_pkts_size": sizes.sum(),
            "f5_std_pkt_size": self.compute_pkt_std(group),
            "f6_tot_flx": len(group),
            "f7_n_dst_ports": group["L4_DST_PORT"].nunique(),
            "f8_proto_div": group["PROTOCOL"].nunique(),
            "f9_dur_var": group["FLOW_DURATION_MILLISECONDS"].std() if len(group) > 1 else 0.0,
            "f10_n_dst_ip": group["IPV4_DST_ADDR"].nunique(),
            "ip": ip,
            "perspective": "src",
            "label": int(group["Label"].any()),
            "attack_types": ",".join(group["Attack"].astype(str).unique()),
        }

    def compute_dst_features(self, group, ip):
        sizes = group["IN_BYTES"] + group["OUT_BYTES"]
        return {
            "f1_n_fwd_pkts": group["IN_PKTS"].sum(),
            "f2_n_bwd_pkts": group["OUT_PKTS"].sum(),
            "f3_sum_flx_dur": group["FLOW_DURATION_MILLISECONDS"].sum(),
            "f4_sum_pkts_size": sizes.sum(),
            "f5_std_pkt_size": self.compute_pkt_std(group),
            "f6_tot_flx": len(group),
            "f7_n_dst_ports": group["L4_DST_PORT"].nunique(),
            "f8_proto_div": group["PROTOCOL"].nunique(),
            "f9_dur_var": group["FLOW_DURATION_MILLISECONDS"].std() if len(group) > 1 else 0.0,
            "f10_n_src_ip": group["IPV4_SRC_ADDR"].nunique(),
            "ip": ip,
            "perspective": "dst",
            "label": int(group["Label"].any()),
            "attack_types": ",".join(group["Attack"].astype(str).unique()),
        }

    @staticmethod
    def safe_clean(df):
        for col in df.columns:
            if hasattr(df[col], "cat"):
                df[col] = df[col].astype(str)
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    @staticmethod
    def get_category_distribution(agg_df, perspective):
        total = len(agg_df)
        benign_pct = (agg_df["label"] == 0).sum() / total * 100
        attack_rows = agg_df[agg_df["label"] == 1].copy()

        type_counts = {}
        for _, row in attack_rows.iterrows():
            types = str(row["attack_types"]).split(",")
            for t in types:
                t = t.strip()
                if t and t.lower() not in ["0", "benign", "nan"]:
                    type_counts[t] = type_counts.get(t, 0) + 1

        print(f"\n  {perspective} Perspective:")
        print(f"  {'Category':<35} {'Our %':>10}  {'Paper %':>10}")
        print(f"  {'-' * 57}")

        paper_ref = {
            "Brute-force": (0.004455, 0.009547),
            "DoS": (0.001412, 0.004231),
            "Web Attack": (0.0, 0.013089),
            "Infiltration": (0.003477, 0.007480),
            "Bot": (0.162978, 0.0),
            "DDoS": (0.006302, 0.006102),
            "Benign": (99.821376, 99.956894),
        }

        p_idx = 0 if "src" in perspective.lower() else 1
        print(
            f"  {'Benign':<35} {benign_pct:>9.4f}%  "
            f"{paper_ref['Benign'][p_idx]:>9.4f}%"
        )

        for attack_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            our_pct = count / total * 100
            paper_pct = "N/A"
            for pkey, pvals in paper_ref.items():
                if pkey.lower() in attack_type.lower() or attack_type.lower() in pkey.lower():
                    paper_pct = f"{pvals[p_idx]:>9.4f}%"
                    break
            print(f"  {attack_type:<35} {our_pct:>9.4f}%  {paper_pct:>10}")

        print(f"\n  Total aggregates: {total:,}")
        return type_counts
