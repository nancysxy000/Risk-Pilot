"""
数据加载与预处理模块
支持: tfinance, tsocial, yelp, amazon 数据集
"""

import os
import torch
import numpy as np
import dgl
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs


class GraphDataset:
    """统一的图数据集加载器"""

    SUPPORTED = ['tfinance', 'tsocial', 'yelp', 'amazon']

    def __init__(self, name='tfinance', data_dir='dataset/', homo=True,
                 anomaly_alpha=None, anomaly_std=None):
        assert name in self.SUPPORTED, f"数据集 {name} 不支持，可选: {self.SUPPORTED}"

        self.name = name
        self.data_dir = data_dir
        self.graph = None
        self.stats = {}

        if name == 'tfinance':
            self.graph = self._load_tfinance(anomaly_alpha, anomaly_std)
        elif name == 'tsocial':
            self.graph = self._load_tsocial()
        elif name == 'yelp':
            self.graph = self._load_yelp(homo)
        elif name == 'amazon':
            self.graph = self._load_amazon(homo)

        # 统一格式
        self.graph.ndata['label'] = self.graph.ndata['label'].long().squeeze(-1)
        self.graph.ndata['feature'] = self.graph.ndata['feature'].float()

        # 计算数据集统计信息
        self._compute_stats()

    def _load_tfinance(self, anomaly_alpha, anomaly_std):
        path = os.path.join(self.data_dir, 'tfinance')
        graph, _ = load_graphs(path)
        graph = graph[0]

        if anomaly_std:
            feat = graph.ndata['feature'].numpy()
            anomaly_id = graph.ndata['label'][:, 1].nonzero().squeeze(1)
            feat = (feat - np.average(feat, 0)) / np.std(feat, 0)
            feat[anomaly_id] = anomaly_std * feat[anomaly_id]
            graph.ndata['feature'] = torch.tensor(feat)

        if anomaly_alpha:
            import random
            feat = graph.ndata['feature'].numpy()
            anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
            normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
            label = graph.ndata['label'].argmax(1)
            diff = anomaly_alpha * len(label) - len(anomaly_id)
            new_id = random.sample(normal_id, int(diff))
            for idx in new_id:
                aid = random.choice(anomaly_id)
                feat[idx] = feat[aid]
                label[idx] = 1

        graph.ndata['label'] = graph.ndata['label'].argmax(1)
        return graph

    def _load_tsocial(self):
        path = os.path.join(self.data_dir, 'tsocial')
        graph, _ = load_graphs(path)
        return graph[0]

    def _load_yelp(self, homo):
        dataset = FraudYelpDataset()
        graph = dataset[0]
        if homo:
            graph = dgl.to_homogeneous(
                graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask']
            )
            graph = dgl.add_self_loop(graph)
        return graph

    def _load_amazon(self, homo):
        dataset = FraudAmazonDataset()
        graph = dataset[0]
        if homo:
            graph = dgl.to_homogeneous(
                graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask']
            )
            graph = dgl.add_self_loop(graph)
        return graph

    def _compute_stats(self):
        labels = self.graph.ndata['label']
        n_nodes = self.graph.num_nodes()
        n_edges = self.graph.num_edges()
        n_anomaly = (labels == 1).sum().item()
        n_normal = (labels == 0).sum().item()

        self.stats = {
            'name': self.name,
            'num_nodes': n_nodes,
            'num_edges': n_edges,
            'num_features': self.graph.ndata['feature'].shape[1],
            'num_anomaly': n_anomaly,
            'num_normal': n_normal,
            'anomaly_ratio': n_anomaly / n_nodes if n_nodes > 0 else 0,
        }

    def summary(self):
        """打印数据集摘要"""
        print(f"\n{'='*50}")
        print(f"  Dataset: {self.stats['name']}")
        print(f"  Nodes: {self.stats['num_nodes']:,}")
        print(f"  Edges: {self.stats['num_edges']:,}")
        print(f"  Features: {self.stats['num_features']}")
        print(f"  Anomaly: {self.stats['num_anomaly']:,} ({self.stats['anomaly_ratio']:.2%})")
        print(f"  Normal:  {self.stats['num_normal']:,}")
        print(f"{'='*50}\n")
