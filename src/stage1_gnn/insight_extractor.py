"""
Stage 1.3: 风险洞察提取 — 将 GNN 的数值输出转化为 LLM 可理解的结构化描述

这是 Nancy 部分的 **核心创新模块**:
- 从 GNN 输出中提取高风险节点
- 对异常节点嵌入进行聚类，发现不同的风险模式
- 分析每个风险聚类的图结构特征 (度、连通性、局部密度等)
- 输出结构化的 risk_insights.json，供 LLM 策略层使用
"""

import json
import os
import torch
import numpy as np
import dgl
from sklearn.cluster import KMeans
from collections import Counter


class InsightExtractor:
    """将 GNN 的数值预测转化为 LLM 可消费的结构化风险洞察"""

    def __init__(self, config):
        self.config = config['gnn']['insight']
        self.top_k = self.config['top_k_anomaly']
        self.threshold = self.config['anomaly_threshold']
        self.n_clusters = self.config['n_clusters']
        self.k_hop = self.config['k_hop']

    def extract(self, graph, probs, embeddings):
        """
        主函数: 提取结构化风险洞察

        Args:
            graph: DGL 图
            probs: [N, 2] 异常预测概率
            embeddings: [N, D] 节点嵌入
        Returns:
            List[dict]: 风险洞察列表
        """
        anomaly_scores = probs[:, 1].numpy()
        labels = graph.ndata['label'].numpy()

        # Step 1: 筛选高风险节点
        anomaly_nodes = self._select_anomaly_nodes(anomaly_scores)
        print(f"[Insight] 筛选出 {len(anomaly_nodes)} 个高风险节点")

        # Step 2: 对高风险节点嵌入进行聚类
        cluster_labels = self._cluster_anomalies(embeddings, anomaly_nodes)

        # Step 3: 分析每个聚类的图结构特征
        insights = []
        for cid in range(self.n_clusters):
            cluster_nodes = anomaly_nodes[cluster_labels == cid]
            if len(cluster_nodes) == 0:
                continue

            insight = self._analyze_cluster(
                graph, cluster_nodes, anomaly_scores, embeddings, cid
            )
            insights.append(insight)

        # Step 4: 生成全局统计摘要
        summary = self._global_summary(graph, anomaly_scores, anomaly_nodes, insights)

        return {
            'summary': summary,
            'risk_patterns': insights,
        }

    def _select_anomaly_nodes(self, anomaly_scores):
        """筛选高风险节点: 阈值 + Top-K"""
        above_threshold = np.where(anomaly_scores > self.threshold)[0]

        if len(above_threshold) > self.top_k:
            # 取 Top-K 最高风险的
            top_indices = np.argsort(anomaly_scores[above_threshold])[-self.top_k:]
            return above_threshold[top_indices]
        return above_threshold

    def _cluster_anomalies(self, embeddings, anomaly_nodes):
        """对高风险节点的嵌入向量进行 KMeans 聚类"""
        emb = embeddings[anomaly_nodes].numpy()

        n_clusters = min(self.n_clusters, len(anomaly_nodes))
        if n_clusters < 2:
            return np.zeros(len(anomaly_nodes), dtype=int)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(emb)
        return cluster_labels

    def _analyze_cluster(self, graph, cluster_nodes, anomaly_scores, embeddings, cluster_id):
        """分析单个风险聚类的图结构特征"""
        node_list = cluster_nodes.tolist()

        # 度统计
        degrees = graph.in_degrees().numpy()[cluster_nodes]
        # 异常分数统计
        scores = anomaly_scores[cluster_nodes]
        # 原始特征统计
        features = graph.ndata['feature'][cluster_nodes].numpy()

        # 提取 k-hop 子图分析局部结构
        subgraph_stats = self._analyze_subgraph(graph, cluster_nodes)

        # 特征维度上的统计: 找出哪些特征维度在异常节点上显著偏高/偏低
        all_feat_mean = graph.ndata['feature'].numpy().mean(axis=0)
        all_feat_std = graph.ndata['feature'].numpy().std(axis=0) + 1e-8
        cluster_feat_mean = features.mean(axis=0)
        z_scores = (cluster_feat_mean - all_feat_mean) / all_feat_std
        salient_features = np.where(np.abs(z_scores) > 1.5)[0].tolist()

        insight = {
            'cluster_id': int(cluster_id),
            'num_nodes': len(node_list),
            'avg_anomaly_score': float(np.mean(scores)),
            'max_anomaly_score': float(np.max(scores)),
            'degree_stats': {
                'mean': float(np.mean(degrees)),
                'max': int(np.max(degrees)),
                'min': int(np.min(degrees)),
                'std': float(np.std(degrees)),
            },
            'salient_feature_dims': salient_features,
            'feature_z_scores': {int(d): float(z_scores[d]) for d in salient_features},
            'subgraph_stats': subgraph_stats,
            'risk_description': self._generate_description(
                cluster_id, len(node_list), degrees, scores, salient_features, z_scores, subgraph_stats
            ),
            'sample_nodes': node_list[:20],  # 采样节点 ID (供调试)
        }
        return insight

    def _analyze_subgraph(self, graph, cluster_nodes):
        """分析异常节点的局部子图结构"""
        if len(cluster_nodes) > 200:
            # 采样避免计算量过大
            sample_idx = np.random.choice(len(cluster_nodes), 200, replace=False)
            cluster_nodes = cluster_nodes[sample_idx]

        node_tensor = torch.tensor(cluster_nodes, dtype=torch.int64)

        # k-hop 子图
        sg_nodes, sg_edges = dgl.sampling.sample_neighbors(
            graph, node_tensor, fanout=-1
        ).edges()

        # 子图内部边数 (异常节点之间的连接数)
        cluster_set = set(cluster_nodes.tolist())
        internal_edges = sum(
            1 for s, d in zip(sg_nodes.tolist(), sg_edges.tolist())
            if s in cluster_set and d in cluster_set
        )

        n = len(cluster_set)
        max_internal = n * (n - 1) if n > 1 else 1
        density = internal_edges / max_internal

        return {
            'internal_edges': internal_edges,
            'local_density': float(density),
            'total_neighbors': len(set(sg_nodes.tolist()) | set(sg_edges.tolist())),
        }

    def _generate_description(self, cid, n_nodes, degrees, scores, salient_features, z_scores, subgraph_stats):
        """为每个风险聚类生成自然语言描述 (供 LLM 消费)"""
        desc_parts = [f"Risk Pattern #{cid}: {n_nodes} anomalous nodes detected."]

        # 度特征描述
        avg_deg = np.mean(degrees)
        if avg_deg > 50:
            desc_parts.append(f"High connectivity (avg degree={avg_deg:.0f}), suggesting organized group behavior.")
        elif avg_deg < 5:
            desc_parts.append(f"Low connectivity (avg degree={avg_deg:.0f}), suggesting isolated suspicious accounts.")
        else:
            desc_parts.append(f"Moderate connectivity (avg degree={avg_deg:.0f}).")

        # 密度描述
        if subgraph_stats['local_density'] > 0.3:
            desc_parts.append("Densely connected subgraph — potential fraud ring.")
        elif subgraph_stats['local_density'] > 0.1:
            desc_parts.append("Moderately connected subgraph — possible coordinated activity.")

        # 显著特征描述
        if salient_features:
            high_feats = [d for d in salient_features if z_scores[d] > 0]
            low_feats = [d for d in salient_features if z_scores[d] < 0]
            if high_feats:
                desc_parts.append(f"Abnormally high on feature dims {high_feats}.")
            if low_feats:
                desc_parts.append(f"Abnormally low on feature dims {low_feats}.")

        # 置信度
        desc_parts.append(f"Avg anomaly score: {np.mean(scores):.3f}.")

        return " ".join(desc_parts)

    def _global_summary(self, graph, anomaly_scores, anomaly_nodes, insights):
        """全局风险概况总结"""
        n_total = graph.num_nodes()
        n_anomaly = len(anomaly_nodes)
        return {
            'total_nodes': n_total,
            'total_anomaly_detected': n_anomaly,
            'anomaly_ratio': n_anomaly / n_total,
            'num_risk_patterns': len(insights),
            'avg_anomaly_score': float(np.mean(anomaly_scores[anomaly_nodes])),
            'description': (
                f"Detected {n_anomaly} anomalous nodes ({n_anomaly/n_total:.2%}) "
                f"across {len(insights)} distinct risk patterns in a graph with "
                f"{n_total:,} nodes and {graph.num_edges():,} edges."
            ),
        }

    def save(self, insights, output_dir='outputs/'):
        """保存风险洞察到 JSON"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'risk_insights.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        print(f"[Insight] Saved to {path}")
        return path
