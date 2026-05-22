"""
传统异常检测 Baseline 对比实验

在 tfinance / yelp / amazon 上运行 4 种传统方法，与 BWGNN 对比:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- PCA + Mahalanobis Distance

公平对比: 相同数据划分、相同特征、相同评估指标。
"""

import os
import json
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

import torch

from ..stage1_gnn.dataset_loader import GraphDataset


def get_best_f1_threshold(labels, scores):
    """
    阈值搜索: 找到使 Macro-F1 最高的阈值

    与 GNN 训练中的 get_best_f1() 使用相同的搜索策略:
    在异常分数上搜索 19 个阈值点 [P5, P10, ..., P95]
    """
    best_f1, best_thre = 0, 0
    for percentile in range(5, 100, 5):
        thres = np.percentile(scores, percentile)
        preds = (scores >= thres).astype(int)
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def compute_metrics(labels, scores, threshold):
    """计算 Recall, Precision, Macro-F1, AUC"""
    preds = (scores >= threshold).astype(int)

    # 处理全 0 或全 1 预测
    if preds.sum() == 0 or preds.sum() == len(preds):
        return {
            'recall': 0.0,
            'precision': 0.0,
            'macro_f1': 0.0,
            'auc': 0.0,
            'threshold': float(threshold),
        }

    return {
        'recall': float(recall_score(labels, preds)),
        'precision': float(precision_score(labels, preds)),
        'macro_f1': float(f1_score(labels, preds, average='macro')),
        'auc': float(roc_auc_score(labels, scores)),
        'threshold': float(threshold),
    }


class TraditionalBaselineRunner:
    """
    传统异常检测方法 Baseline 运行器

    对每个数据集运行 4 种传统方法，复用 GNN 的数据划分和评估逻辑。
    """

    def __init__(self, config):
        self.config = config
        self.features = None
        self.labels = None
        self.degrees = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None

    def load_data(self):
        """加载图数据，提取特征 + 标签 + 度数"""
        ds_cfg = self.config['dataset']
        dataset = GraphDataset(
            name=ds_cfg['name'],
            data_dir=ds_cfg['path'],
            homo=ds_cfg['homo'],
            anomaly_alpha=ds_cfg.get('anomaly_alpha'),
            anomaly_std=ds_cfg.get('anomaly_std'),
        )
        graph = dataset.graph

        self.features = graph.ndata['feature'].numpy()
        self.labels = graph.ndata['label'].numpy()
        self.degrees = graph.in_degrees().float().numpy()
        self.dataset_name = ds_cfg['name']

        print(f"\n{'='*50}")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Nodes: {len(self.labels):,}")
        print(f"  Features: {self.features.shape[1]}")
        print(f"  Anomaly: {self.labels.sum():,} ({self.labels.mean():.2%})")
        print(f"{'='*50}")

        return self

    def _get_split(self):
        """复用 train.py 的划分逻辑"""
        index = list(range(len(self.labels)))
        if self.dataset_name == 'amazon':
            index = list(range(3305, len(self.labels)))

        train_ratio = self.config['dataset'].get('train_ratio', 0.4)

        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index, self.labels[index], stratify=self.labels[index],
            train_size=train_ratio, random_state=42, shuffle=True
        )
        _, idx_test, _, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest,
            test_size=0.67, random_state=42, shuffle=True
        )

        self.idx_train = np.array(idx_train)
        self.idx_test = np.array(idx_test)
        self.y_train = self.labels[self.idx_train]
        self.y_test = self.labels[self.idx_test]

    def _prepare_features(self):
        """拼接 [原始特征, degree_zscore]，并标准化"""
        self._get_split()

        # 度数标准化
        degree_zscore = (self.degrees - self.degrees.mean()) / (self.degrees.std() + 1e-8)
        degree_zscore = degree_zscore.reshape(-1, 1)

        # 拼接
        X = np.hstack([self.features, degree_zscore])

        # 标准化 (对每个 baseline 都很重要)
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.idx_train])
        self.X_train = self.scaler.transform(X[self.idx_train])
        self.X_test = self.scaler.transform(X[self.idx_test])

    def run_isolation_forest(self):
        """Isolation Forest"""
        print("\n  [IF] Isolation Forest ...", end=" ", flush=True)
        t0 = time.time()

        contamination = self.y_train.mean()
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(self.X_train)
        scores = -model.score_samples(self.X_test)  # 取负: 越大越异常

        best_f1, threshold = get_best_f1_threshold(self.y_test, scores)
        metrics = compute_metrics(self.y_test, scores, threshold)
        elapsed = time.time() - t0

        print(f"F1={metrics['macro_f1']:.2%}, AUC={metrics['auc']:.2%} ({elapsed:.1f}s)")
        return metrics

    def run_lof(self):
        """Local Outlier Factor"""
        print("  [LOF] Local Outlier Factor ...", end=" ", flush=True)
        t0 = time.time()

        contamination = self.y_train.mean()
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        model.fit(self.X_train)
        scores = -model.score_samples(self.X_test)

        best_f1, threshold = get_best_f1_threshold(self.y_test, scores)
        metrics = compute_metrics(self.y_test, scores, threshold)
        elapsed = time.time() - t0

        print(f"F1={metrics['macro_f1']:.2%}, AUC={metrics['auc']:.2%} ({elapsed:.1f}s)")
        return metrics

    def run_one_class_svm(self):
        """One-Class SVM (只用正常样本训练)"""
        print("  [SVM] One-Class SVM ...", end=" ", flush=True)
        t0 = time.time()

        X_train_normal = self.X_train[self.y_train == 0]
        nu = min(0.5, self.y_train.mean() * 2)  # nu 参数

        model = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
        model.fit(X_train_normal)
        scores = -model.score_samples(self.X_test)

        best_f1, threshold = get_best_f1_threshold(self.y_test, scores)
        metrics = compute_metrics(self.y_test, scores, threshold)
        elapsed = time.time() - t0

        print(f"F1={metrics['macro_f1']:.2%}, AUC={metrics['auc']:.2%} ({elapsed:.1f}s)")
        return metrics

    def run_pca_mahalanobis(self):
        """PCA + Mahalanobis 距离"""
        print("  [PCA] PCA + Mahalanobis ...", end=" ", flush=True)
        t0 = time.time()

        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)

        mean = X_train_pca.mean(axis=0)
        cov = np.cov(X_train_pca.T)
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

        # Mahalanobis 距离
        diff = X_test_pca - mean
        scores = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        best_f1, threshold = get_best_f1_threshold(self.y_test, scores)
        metrics = compute_metrics(self.y_test, scores, threshold)
        elapsed = time.time() - t0

        print(f"F1={metrics['macro_f1']:.2%}, AUC={metrics['auc']:.2%} ({elapsed:.1f}s)")
        return metrics

    def run_all(self):
        """运行全部 4 个 baseline"""
        self.load_data()
        self._prepare_features()

        results = {
            'dataset': self.dataset_name,
            'num_nodes': int(len(self.labels)),
            'num_features': int(self.features.shape[1]),
            'anomaly_ratio': float(self.labels.mean()),
            'methods': {},
        }

        results['methods']['IsolationForest'] = self.run_isolation_forest()
        results['methods']['LOF'] = self.run_lof()
        results['methods']['OneClassSVM'] = self.run_one_class_svm()
        results['methods']['PCA_Mahalanobis'] = self.run_pca_mahalanobis()

        return results

    @staticmethod
    def save_results(results, output_dir):
        """保存结果为 JSON"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'traditional_results.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {path}")

    @staticmethod
    def print_comparison(results, gnn_reference=None):
        """打印对比表格"""
        dataset = results['dataset']
        print(f"\n{'='*70}")
        print(f"  {dataset.upper()} — 传统方法 vs BWGNN 对比")
        print(f"{'='*70}")
        print(f"  {'Method':<20} {'Recall':>8} {'Precision':>10} {'Macro-F1':>10} {'AUC':>8}")
        print(f"  {'-'*56}")

        for name, m in results['methods'].items():
            print(f"  {name:<20} {m['recall']:>8.2%} {m['precision']:>10.2%} "
                  f"{m['macro_f1']:>10.2%} {m['auc']:>8.2%}")

        if gnn_reference:
            print(f"  {'-'*56}")
            print(f"  {'BWGNN (ours)':<20} {gnn_reference['recall']:>8.2%} "
                  f"{gnn_reference['precision']:>10.2%} "
                  f"{gnn_reference['macro_f1']:>10.2%} "
                  f"{gnn_reference['auc']:>8.2%}")
        print(f"{'='*70}")
