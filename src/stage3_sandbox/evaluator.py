"""
Stage 3: 沙盒验证层 — 规则执行与评估

在标注数据集上回测 LLM 生成的规则，计算召回率、精确率、误杀率等指标。

回测逻辑:
  1. 将每条 JSON 规则 (如 "degree > 50 AND feature_0 > 2.0") 解析为可执行函数
  2. 在 tfinance 全量节点上执行规则，得到 "命中/未命中" 的 mask
  3. 与真实标签对比，计算混淆矩阵和指标

评估指标:
  - Recall (召回率): 真正的欺诈账户中，规则能抓到多少? 越高越好
  - Precision (精确率): 规则标记为欺诈的账户中，有多少真的是? 越高越好
  - FPR (误杀率): 正常用户中，有多少被误标为欺诈? 必须 < 5%
  - F1: Precision 和 Recall 的调和均值

合格标准 (default.yaml 配置):
  - F1 >= 0.60, Recall >= 0.50, Precision >= 0.40, FPR <= 0.05
  - 不合格的规则会反馈给 LLM 重新优化 (闭环迭代)

实际结果示例 (Mock 规则, tfinance):
  - "高度数异常交易检测": Recall=90.6% 但 FPR=83.9% → 不合格 (阈值太宽松)
  - "孤立高风险账户检测": Recall=0.5%, FPR=1.3% → 不合格 (条件太苛刻)
  - 接入真实 LLM 后，系统会自动迭代优化这些阈值
"""

import json
import os
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, classification_report
)
from .rule_parser import RuleParser


class RuleEvaluator:
    """规则沙盒回测与评估"""

    def __init__(self, config):
        self.config = config['sandbox']
        self.parser = RuleParser()
        self.thresholds = self.config['thresholds']  # 合格标准阈值

    def evaluate(self, rules, graph):
        """
        在标注数据集上评估所有规则

        Args:
            rules: list[dict], LLM 生成的规则 (来自 generated_rules.json)
            graph: DGL 图 (含 label 和 feature，即 tfinance 原始数据)
        Returns:
            dict: 评估报告 → 写入 evaluation_report.json
        """
        features = graph.ndata['feature']       # [N, feat_dim] 节点特征
        labels = graph.ndata['label'].numpy()   # [N] 真实标签: 0=正常, 1=异常
        degrees = graph.in_degrees().float()    # [N] 节点度数 (规则中 node_degree 字段对应)

        # 将 JSON 规则解析为可执行函数
        parsed_rules = self.parser.parse_all(rules)

        # 逐条规则单独评估
        rule_results = []
        for parsed in parsed_rules:
            result = self._evaluate_single_rule(parsed, features, degrees, labels)
            rule_results.append(result)

        # 所有规则组合评估 (OR 逻辑: 任意规则命中即标记)
        combined_result = self._evaluate_combined(parsed_rules, features, degrees, labels)

        # 按阈值筛选合格规则
        qualified = self._filter_qualified(rule_results)

        report = {
            'total_rules': len(rules),
            'qualified_rules': len(qualified),
            'individual_results': rule_results,
            'combined_result': combined_result,
            'qualified_rule_ids': [r['rule_id'] for r in qualified],
        }

        return report

    def _evaluate_single_rule(self, parsed_rule, features, degrees, labels):
        """
        评估单条规则

        混淆矩阵:
                        预测=正常(0)  预测=异常(1)
          真实=正常(0)     TN           FP (误杀)
          真实=异常(1)     FN (漏检)     TP (正确抓到)

        关键指标:
          Recall = TP / (TP + FN) — 异常账户中抓到了多少
          Precision = TP / (TP + FP) — 标记为异常的里面有多少是对的
          FPR = FP / (FP + TN) — 正常用户被误杀的比例
        """
        mask = parsed_rule['fn'](features, degrees)
        preds = mask.numpy().astype(int) if isinstance(mask, torch.Tensor) else np.array(mask, dtype=int)

        # 基础指标
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        total_pos = int((labels == 1).sum())
        total_neg = int((labels == 0).sum())

        recall = tp / total_pos if total_pos > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'rule_id': parsed_rule['rule_id'],
            'name': parsed_rule['name'],
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'recall': recall,
            'precision': precision,
            'fpr': fpr,
            'f1': f1,
            'flagged': int(preds.sum()),
            'qualified': self._is_qualified(recall, precision, fpr, f1),
        }

    def _evaluate_combined(self, parsed_rules, features, degrees, labels):
        """评估所有规则的组合效果 (OR 逻辑)"""
        combined_preds = np.zeros(len(labels), dtype=int)

        for parsed in parsed_rules:
            mask = parsed['fn'](features, degrees)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            combined_preds = np.maximum(combined_preds, mask.astype(int))

        total_pos = int((labels == 1).sum())
        total_neg = int((labels == 0).sum())
        tp = int(((combined_preds == 1) & (labels == 1)).sum())
        fp = int(((combined_preds == 1) & (labels == 0)).sum())

        recall = tp / total_pos if total_pos > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0

        return {
            'combined_recall': recall,
            'combined_precision': precision,
            'combined_fpr': fpr,
            'total_flagged': int(combined_preds.sum()),
            'coverage': recall,  # 与 recall 等价
        }

    def _is_qualified(self, recall, precision, fpr, f1):
        """判断规则是否达标"""
        return (
            f1 >= self.thresholds['min_f1']
            and recall >= self.thresholds['min_recall']
            and fpr <= self.thresholds['max_fpr']
            and precision >= self.thresholds['min_precision']
        )

    def _filter_qualified(self, rule_results):
        """筛选合格规则"""
        return [r for r in rule_results if r['qualified']]

    def get_rules_to_optimize(self, rules, rule_results):
        """获取需要优化的规则 (不合格的)"""
        failed_ids = {r['rule_id'] for r in rule_results if not r['qualified']}
        return [r for r in rules if r.get('rule_id') in failed_ids]

    def save_report(self, report, output_dir='outputs/'):
        """保存评估报告"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'evaluation_report.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Saved report to {path}")

        # 打印摘要
        print(f"\n{'='*50}")
        print(f"  Evaluation Report")
        print(f"  Total rules:     {report['total_rules']}")
        print(f"  Qualified rules: {report['qualified_rules']}")
        combined = report['combined_result']
        print(f"  Combined Recall:    {combined['combined_recall']:.2%}")
        print(f"  Combined Precision: {combined['combined_precision']:.2%}")
        print(f"  Combined FPR:       {combined['combined_fpr']:.2%}")
        print(f"{'='*50}\n")

        return path
