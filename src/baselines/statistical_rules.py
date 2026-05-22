"""
统计规则自动生成器 (不使用 LLM)

从 risk_insights 的风险模式直接生成规则，作为 LLM 规则生成的对比基线。
对每个 risk_pattern，利用显著特征维度的 z-score 和度数统计自动生成条件。
"""

import time
from datetime import datetime


class StatisticalRuleGenerator:
    """
    基于统计的规则生成器

    对每个风险模式 (cluster)，根据其显著特征维度和 z-score 自动生成规则。
    完全不调用 LLM，用于对比 LLM 规则生成的增量价值。
    """

    def __init__(self, config=None):
        self.config = config or {}

    def generate(self, risk_insights, dataset_stats=None):
        """
        从 risk_insights 自动生成规则

        对每个 risk_pattern:
        1. gnn_anomaly_score 条件 (基于聚类平均分数)
        2. 每个显著特征维度的 z-score 条件
        3. 度数条件 (如果聚类平均度数远高于全局)

        Args:
            risk_insights: dict, 来自 InsightExtractor 的输出
            dataset_stats: dict, 数据集统计信息 (可选)

        Returns:
            list[dict]: 规则列表，格式与 LLM 生成的规则完全一致
        """
        patterns = risk_insights.get('risk_patterns', [])
        rules = []

        for pattern in patterns:
            conditions = []

            # 条件 1: GNN 异常分数阈值
            avg_score = pattern['avg_anomaly_score']
            score_threshold = round(avg_score * 0.8, 4)
            conditions.append({
                "field": "gnn_anomaly_score",
                "operator": ">=",
                "value": score_threshold
            })

            # 条件 2: 每个显著特征维度的 z-score 条件
            for dim, z_score in pattern.get('feature_z_scores', {}).items():
                dim = int(dim)
                field = f"feature_dim_{dim}_zscore"
                # 放宽阈值: 用 z * 0.7 而不是 z 本身，留一些泛化空间
                if z_score > 0:
                    conditions.append({
                        "field": field,
                        "operator": ">=",
                        "value": round(z_score * 0.7, 2)
                    })
                else:
                    conditions.append({
                        "field": field,
                        "operator": "<=",
                        "value": round(z_score * 0.7, 2)
                    })

            # 条件 3: 度数条件 (如果聚类平均度数异常高)
            degree_mean = pattern.get('degree_stats', {}).get('mean', 0)
            if degree_mean > 100:  # 度数较高时添加条件
                conditions.append({
                    "field": "node_degree",
                    "operator": ">=",
                    "value": int(degree_mean * 0.5)
                })

            # 生成规则
            cluster_id = pattern['cluster_id']
            num_nodes = pattern['num_nodes']
            salient_dims = pattern.get('salient_feature_dims', [])

            rule = {
                "name": f"Statistical-Cluster{cluster_id}",
                "description": (
                    f"Auto-generated rule for risk pattern #{cluster_id} "
                    f"({num_nodes} nodes, avg anomaly score {avg_score:.3f}). "
                    f"Salient feature dims: {salient_dims}. "
                    f"Conditions derived from cluster z-scores and degree statistics."
                ),
                "risk_type": "集团欺诈",
                "severity": "high" if num_nodes > 50 else "medium",
                "conditions": conditions,
                "logic": "AND",
                "action": "review",
                "explanation": (
                    f"Statistical rule for cluster {cluster_id} with {len(conditions)} conditions. "
                    f"Derived from z-scores of {len(salient_dims)} salient feature dimensions."
                ),
                "rule_id": f"R-STAT-{datetime.now().strftime('%Y%m%d')}-{cluster_id:03d}",
                "generated_at": datetime.now().isoformat(),
                "source": "statistical_auto_gen",
            }
            rules.append(rule)

        return rules

    def threshold_search(self, rules, graph, gnn_outputs=None):
        """
        统计规则不需要单独的 threshold_search，
        因为 run_baselines.py 会调用 RuleGenerator.threshold_search() 来处理。
        这里只是提供一个 passthrough 接口。
        """
        return rules
