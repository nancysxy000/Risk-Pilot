"""
Stage 3: 沙盒验证层 — 规则执行与评估

在标注数据集上回测规则，计算召回率、精确率、误杀率等指标。
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
        self.thresholds = self.config['thresholds']

    def evaluate(self, rules, graph):
        """
        在标注数据集上评估所有规则

        Args:
            rules: list[dict], LLM 生成的规则
            graph: DGL 图 (含 label 和 feature)
        Returns:
            dict: 评估报告
        """
        features = graph.ndata['feature']
        labels = graph.ndata['label'].numpy()
        degrees = graph.in_degrees().float()

        parsed_rules = self.parser.parse_all(rules)

        # 逐条规则评估
        rule_results = []
        for parsed in parsed_rules:
            result = self._evaluate_single_rule(parsed, features, degrees, labels)
            rule_results.append(result)

        # 组合规则评估 (所有规则 OR)
        combined_result = self._evaluate_combined(parsed_rules, features, degrees, labels)

        # 筛选合格规则
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
        """评估单条规则"""
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
