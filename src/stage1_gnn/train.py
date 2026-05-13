"""
Stage 1: GNN 模型训练 — 异常检测与嵌入提取
"""

import os
import time
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score
)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from .bwgnn_model import BWGNN, BWGNN_Hetero
from .dataset_loader import GraphDataset


def get_best_f1(labels, probs):
    """通过阈值搜索获取最佳 Macro-F1"""
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


class GNNTrainer:
    """GNN 异常检测训练器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.graph = None
        self.dataset = None

    def load_data(self):
        """加载数据集"""
        ds_cfg = self.config['dataset']
        self.dataset = GraphDataset(
            name=ds_cfg['name'],
            data_dir=ds_cfg['path'],
            homo=ds_cfg['homo'],
            anomaly_alpha=ds_cfg.get('anomaly_alpha'),
            anomaly_std=ds_cfg.get('anomaly_std'),
        )
        self.graph = self.dataset.graph
        self.dataset.summary()
        return self.dataset

    def build_model(self):
        """构建 BWGNN 模型"""
        gnn_cfg = self.config['gnn']
        in_feats = self.graph.ndata['feature'].shape[1]
        h_feats = gnn_cfg['hidden_dim']
        num_classes = 2
        d = gnn_cfg['order']

        if self.config['dataset']['homo']:
            self.model = BWGNN(in_feats, h_feats, num_classes, self.graph, d=d)
        else:
            self.model = BWGNN_Hetero(in_feats, h_feats, num_classes, self.graph, d=d)

        device = gnn_cfg.get('device', 'cpu')
        self.model = self.model.to(device)
        return self.model

    def train(self):
        """训练模型，返回最佳指标和训练好的模型"""
        gnn_cfg = self.config['gnn']
        features = self.graph.ndata['feature']
        labels = self.graph.ndata['label']

        # 数据划分
        index = list(range(len(labels)))
        if self.config['dataset']['name'] == 'amazon':
            index = list(range(3305, len(labels)))

        train_ratio = self.config['dataset']['train_ratio']
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index, labels[index], stratify=labels[index],
            train_size=train_ratio, random_state=42, shuffle=True
        )
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest,
            test_size=0.67, random_state=42, shuffle=True
        )

        train_mask = torch.zeros(len(labels)).bool()
        val_mask = torch.zeros(len(labels)).bool()
        test_mask = torch.zeros(len(labels)).bool()
        train_mask[idx_train] = True
        val_mask[idx_valid] = True
        test_mask[idx_test] = True

        print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

        # 类别不平衡权重
        weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print(f"Class weight (neg/pos): {weight:.2f}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=gnn_cfg['learning_rate'])
        best_f1 = 0.
        best_metrics = {}

        t_start = time.time()
        pbar = tqdm(range(gnn_cfg['epochs']), desc="Training", unit="epoch",
                    bar_format='{l_bar}{bar:30}{r_bar}')

        for epoch in pbar:
            # ---- Train ----
            self.model.train()
            logits = self.model(features)
            loss = F.cross_entropy(
                logits[train_mask], labels[train_mask],
                weight=torch.tensor([1., weight])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Eval ----
            self.model.eval()
            with torch.no_grad():
                probs = logits.softmax(1)
            f1, thres = get_best_f1(labels[val_mask].numpy(), probs[val_mask].numpy())

            preds = np.zeros_like(labels.numpy())
            preds[probs[:, 1].numpy() > thres] = 1

            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    'epoch': epoch,
                    'val_f1': f1,
                    'test_recall': recall_score(labels[test_mask], preds[test_mask]),
                    'test_precision': precision_score(labels[test_mask], preds[test_mask]),
                    'test_macro_f1': f1_score(labels[test_mask], preds[test_mask], average='macro'),
                    'test_auc': roc_auc_score(labels[test_mask], probs[test_mask][:, 1].numpy()),
                    'threshold': thres,
                }
                # 保存最佳模型参数
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'val_F1': f'{f1:.4f}',
                'best_F1': f'{best_f1:.4f}',
            })

        elapsed = time.time() - t_start
        print(f"\nTraining done in {elapsed:.1f}s")
        print(f"Best — Recall {best_metrics['test_recall']:.2%} | "
              f"Precision {best_metrics['test_precision']:.2%} | "
              f"Macro-F1 {best_metrics['test_macro_f1']:.2%} | "
              f"AUC {best_metrics['test_auc']:.2%}")

        # 恢复最佳模型
        self.model.load_state_dict(best_state)
        return best_metrics

    def get_predictions(self):
        """获取全量节点的异常预测分数"""
        self.model.eval()
        with torch.no_grad():
            features = self.graph.ndata['feature']
            logits = self.model(features)
            probs = logits.softmax(1)
        return probs  # [N, 2]: [:, 1] 为异常概率

    def get_embeddings(self):
        """获取全量节点的嵌入向量 (传递给 Stage 1.3 洞察提取)"""
        self.model.eval()
        with torch.no_grad():
            features = self.graph.ndata['feature']
            embeddings = self.model.get_embeddings(features)
        return embeddings

    def save_outputs(self, output_dir='outputs/'):
        """保存预测结果和嵌入向量"""
        os.makedirs(output_dir, exist_ok=True)

        probs = self.get_predictions()
        embeddings = self.get_embeddings()

        torch.save(probs, os.path.join(output_dir, 'gnn_predictions.pt'))
        torch.save(embeddings, os.path.join(output_dir, 'node_embeddings.pt'))
        print(f"Saved predictions shape={probs.shape} and embeddings shape={embeddings.shape}")
        return probs, embeddings
