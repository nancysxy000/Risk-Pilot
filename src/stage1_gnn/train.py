"""
Stage 1: GNN 模型训练 — 异常检测与嵌入提取

本模块负责:
1. 加载金融交易图数据 (tfinance 等)
2. 训练 BWGNN 模型进行节点级异常检测
3. 输出异常预测概率 → gnn_predictions.pt
4. 输出节点嵌入向量  → node_embeddings.pt
   (嵌入向量将传递给 insight_extractor.py 做聚类分析)

数据集说明:
- tfinance: 39,357 节点 / 42,445,086 边 / 10 维特征 / 异常比例 4.58%
  节点 = 账户/交易实体，边 = 交易关系，标签由论文作者标注 (ICML 2022)
- 标签来源: graph.ndata['label'] — 数据集自带，0=正常, 1=异常(欺诈)
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

# 屏蔽 sklearn 在早期 epoch 中因无预测样本产生的 UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from .bwgnn_model import BWGNN, BWGNN_Hetero
from .dataset_loader import GraphDataset


def get_best_f1(labels, probs):
    """
    通过阈值搜索获取最佳 Macro-F1

    因为异常检测任务中正负样本极度不平衡 (正常:异常 ≈ 20:1)，
    默认 0.5 阈值效果差，需要搜索最优阈值。
    在 [0.05, 0.95] 区间以 0.05 为步长搜索，返回最佳 F1 和对应阈值。
    """
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
    """
    GNN 异常检测训练器

    完整流程:
      load_data() → build_model() → train() → save_outputs()
      输出: gnn_predictions.pt, node_embeddings.pt
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.graph = None
        self.dataset = None

    def load_data(self):
        """
        加载数据集

        从 DGL 二进制文件加载图结构，读取节点特征和标签。
        数据集中的异常比例 (如 tfinance 的 4.58%) 来自 graph.ndata['label'] 统计，
        这些标签是数据集发布时已标注好的真实欺诈标记。
        """
        ds_cfg = self.config['dataset']
        self.dataset = GraphDataset(
            name=ds_cfg['name'],
            data_dir=ds_cfg['path'],
            homo=ds_cfg['homo'],
            anomaly_alpha=ds_cfg.get('anomaly_alpha'),
            anomaly_std=ds_cfg.get('anomaly_std'),
        )
        self.graph = self.dataset.graph
        self.dataset.summary()  # 打印: 节点数、边数、异常比例等统计信息
        return self.dataset

    def build_model(self):
        """
        构建 BWGNN 模型

        - BWGNN: 同构图版本，适用于 tfinance / tsocial
        - BWGNN_Hetero: 异构图版本，适用于 yelp / amazon
        模型参数量约 10 万，CPU 上即可训练。
        """
        gnn_cfg = self.config['gnn']
        in_feats = self.graph.ndata['feature'].shape[1]  # 特征维度 (tfinance=10)
        h_feats = gnn_cfg['hidden_dim']   # 隐藏层维度 (默认 64)
        num_classes = 2                    # 二分类: 正常 / 异常
        d = gnn_cfg['order']              # Beta 小波阶数 (默认 2，产生 d+1=3 个基)

        if self.config['dataset']['homo']:
            self.model = BWGNN(in_feats, h_feats, num_classes, self.graph, d=d)
        else:
            self.model = BWGNN_Hetero(in_feats, h_feats, num_classes, self.graph, d=d)

        device = gnn_cfg.get('device', 'cpu')
        self.model = self.model.to(device)
        return self.model

    def train(self):
        """
        训练模型，返回最佳指标和训练好的模型

        训练策略:
        - 数据划分: 40% 训练 / 20% 验证 / 40% 测试 (分层抽样保持异常比例)
        - 损失函数: 加权交叉熵，权重 = 正常样本数/异常样本数 (约 20x)，缓解类别不平衡
        - 早停: 基于验证集 Macro-F1 保存最佳模型
        - 输出指标: Recall, Precision, Macro-F1, AUC
        """
        gnn_cfg = self.config['gnn']
        features = self.graph.ndata['feature']  # [N, feat_dim] 节点特征矩阵
        labels = self.graph.ndata['label']      # [N] 节点标签: 0=正常, 1=异常

        # ---- 数据划分 (分层抽样) ----
        index = list(range(len(labels)))
        if self.config['dataset']['name'] == 'amazon':
            index = list(range(3305, len(labels)))  # Amazon 数据集前 3305 个节点无特征

        train_ratio = self.config['dataset']['train_ratio']
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index, labels[index], stratify=labels[index],  # stratify: 保持正负样本比例一致
            train_size=train_ratio, random_state=42, shuffle=True
        )
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest,
            test_size=0.67, random_state=42, shuffle=True
        )

        # 创建 mask 矩阵标记训练/验证/测试节点
        train_mask = torch.zeros(len(labels)).bool()
        val_mask = torch.zeros(len(labels)).bool()
        test_mask = torch.zeros(len(labels)).bool()
        train_mask[idx_train] = True
        val_mask[idx_valid] = True
        test_mask[idx_test] = True

        print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

        # ---- 类别不平衡处理 ----
        # weight = 正常样本数 / 异常样本数，用于加权交叉熵损失
        # 例如 tfinance: weight ≈ 20.8，意味着异常样本的 loss 权重是正常的 20.8 倍
        weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print(f"Class weight (neg/pos): {weight:.2f}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=gnn_cfg['learning_rate'])
        best_f1 = 0.
        best_metrics = {}

        # ---- 训练循环 (带 tqdm 进度条) ----
        t_start = time.time()
        pbar = tqdm(range(gnn_cfg['epochs']), desc="Training", unit="epoch",
                    bar_format='{l_bar}{bar:30}{r_bar}')

        for epoch in pbar:
            # ---- 前向传播 + 反向传播 ----
            self.model.train()
            logits = self.model(features)  # [N, 2] 每个节点的 logits
            loss = F.cross_entropy(
                logits[train_mask], labels[train_mask],
                weight=torch.tensor([1., weight])  # 异常类权重放大
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- 验证集评估 ----
            self.model.eval()
            with torch.no_grad():
                probs = logits.softmax(1)  # [N, 2] 转为概率，[:, 1] 为异常概率
            # 搜索最优阈值下的 Macro-F1
            f1, thres = get_best_f1(labels[val_mask].numpy(), probs[val_mask].numpy())

            preds = np.zeros_like(labels.numpy())
            preds[probs[:, 1].numpy() > thres] = 1  # 用最优阈值生成预测

            # ---- 保存最佳模型 (基于验证集 F1) ----
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
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # 进度条实时显示: loss / 当前 F1 / 历史最佳 F1
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
        """
        获取全量节点的异常预测分数

        Returns:
            probs: [N, 2] tensor
              - probs[:, 0] = 正常概率
              - probs[:, 1] = 异常概率 (用于排序和阈值判定)
        输出文件: outputs/gnn_predictions.pt
        """
        self.model.eval()
        with torch.no_grad():
            features = self.graph.ndata['feature']
            logits = self.model(features)
            probs = logits.softmax(1)
        return probs

    def get_embeddings(self):
        """
        获取全量节点的嵌入向量

        嵌入向量是 BWGNN 倒数第二层的输出 [N, hidden_dim]，
        编码了节点的结构特征和属性特征，将传递给 InsightExtractor
        做 KMeans 聚类，发现不同的异常模式。
        输出文件: outputs/node_embeddings.pt
        """
        self.model.eval()
        with torch.no_grad():
            features = self.graph.ndata['feature']
            embeddings = self.model.get_embeddings(features)
        return embeddings

    def save_outputs(self, output_dir='outputs/'):
        """
        保存预测结果和嵌入向量到磁盘

        输出:
          - outputs/gnn_predictions.pt  — 异常概率 [N, 2]
          - outputs/node_embeddings.pt  — 节点嵌入 [N, hidden_dim]
        这两个文件会被 Stage 1.3 (insight_extractor) 和 Stage 3 (evaluator) 使用。
        """
        os.makedirs(output_dir, exist_ok=True)

        probs = self.get_predictions()
        embeddings = self.get_embeddings()

        torch.save(probs, os.path.join(output_dir, 'gnn_predictions.pt'))
        torch.save(embeddings, os.path.join(output_dir, 'node_embeddings.pt'))
        print(f"Saved predictions shape={probs.shape} and embeddings shape={embeddings.shape}")
        return probs, embeddings
