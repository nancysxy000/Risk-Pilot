"""
Stage 2: LLM 规则自动生成模块

核心流程:
1. 读取 GNN 输出的 risk_insights.json
2. 从知识库 RAG 检索相关历史规则
3. 组装 Prompt + 调用 LLM
4. 解析 LLM 输出为结构化规则 JSON
5. (可选) 阈值微调搜索: 在 LLM 规则基础上贪心搜索最优阈值
"""

import json
import os
from datetime import datetime
import numpy as np
import torch


class RuleGenerator:
    """LLM 驱动的风控规则生成器"""

    def __init__(self, config, rag_retriever=None):
        self.config = config['llm']
        self.rag = rag_retriever
        self.llm_client = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM 客户端，支持 DashScope / OpenAI / Ollama"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        provider = self.config.get('provider', 'openai')
        if provider == 'openai':
            try:
                from openai import OpenAI
                api_key = os.environ.get(self.config.get('api_key_env', 'OPENAI_API_KEY'))
                base_url = os.environ.get(self.config.get('base_url_env', 'OPENAI_BASE_URL'))
                if api_key:
                    kwargs = {"api_key": api_key}
                    if base_url:
                        kwargs["base_url"] = base_url
                    self.llm_client = OpenAI(**kwargs)
                    print(f"[LLM] Connected: model={self.config['model']}, base_url={base_url or 'default'}")
                else:
                    print("[WARNING] OPENAI_API_KEY not set. LLM calls will use mock mode.")
                    self.llm_client = None
            except ImportError:
                print("[WARNING] openai not installed. LLM calls will use mock mode.")
                self.llm_client = None

    def generate(self, risk_insights, dataset_stats=None, data_distribution="未提供"):
        """
        基于 GNN 风险洞察生成风控规则

        Args:
            risk_insights: dict, 来自 InsightExtractor 的输出
            dataset_stats: dict, 数据集统计信息
            data_distribution: str, 度数和特征的分布统计 (供 LLM 设定阈值)
        Returns:
            list[dict]: 结构化规则列表
        """
        from .prompt_templates import SYSTEM_PROMPT, RULE_GENERATION_PROMPT

        # 1. RAG 检索历史相关规则
        historical_rules = "无历史规则 (首次运行)"
        if self.rag:
            historical_rules = self.rag.retrieve(risk_insights)

        # 2. 组装 Prompt
        summary = risk_insights.get('summary', {})
        user_prompt = RULE_GENERATION_PROMPT.format(
            risk_insights=json.dumps(risk_insights['risk_patterns'], indent=2, ensure_ascii=False),
            dataset_name=dataset_stats.get('name', 'unknown') if dataset_stats else 'unknown',
            total_nodes=summary.get('total_nodes', 'N/A'),
            feature_dims=dataset_stats.get('num_features', 'N/A') if dataset_stats else 'N/A',
            anomaly_ratio=f"{summary.get('anomaly_ratio', 0):.2%}",
            historical_rules=historical_rules,
            data_distribution=data_distribution,
        )

        # 3. 调用 LLM
        raw_response = self._call_llm(SYSTEM_PROMPT, user_prompt)

        # 4. 解析输出
        rules = self._parse_rules(raw_response)

        # 5. 添加元数据
        for i, rule in enumerate(rules):
            rule['rule_id'] = f"R-{datetime.now().strftime('%Y%m%d')}-{i+1:03d}"
            rule['generated_at'] = datetime.now().isoformat()
            rule['source'] = 'llm_auto_gen'

        return rules

    def optimize(self, rules, evaluation_results, data_distribution="未提供"):
        """
        基于定向反馈优化规则

        改进点:
        - 不再把整个评估报告丢给 LLM，而是为每条规则生成具体的问题诊断
        - 要求 LLM 小幅调整阈值 (不超过 20%)
        """
        from .prompt_templates import SYSTEM_PROMPT, RULE_OPTIMIZATION_PROMPT

        historical_rules = "无历史规则"
        if self.rag:
            historical_rules = self.rag.retrieve_by_text(
                json.dumps(evaluation_results, ensure_ascii=False)
            )

        # 生成定向反馈
        targeted_feedback = self._build_targeted_feedback(rules, evaluation_results)

        user_prompt = RULE_OPTIMIZATION_PROMPT.format(
            targeted_feedback=targeted_feedback,
            historical_rules=historical_rules,
            data_distribution=data_distribution,
        )

        raw_response = self._call_llm(SYSTEM_PROMPT, user_prompt)
        optimized_rules = self._parse_rules(raw_response)

        for i, rule in enumerate(optimized_rules):
            rule['rule_id'] = f"R-{datetime.now().strftime('%Y%m%d')}-OPT-{i+1:03d}"
            rule['generated_at'] = datetime.now().isoformat()
            rule['source'] = 'llm_optimized'

        return optimized_rules

    def _build_targeted_feedback(self, rules, evaluation_results):
        """
        为每条规则生成针对性的反馈，而非传递整个评估报告

        对每条规则，明确指出:
        - 当前指标和差距
        - 具体调整方向
        - 建议调整幅度
        """
        # 建立 rule_id → result 映射
        results_map = {}
        for res in evaluation_results.get('individual_results', []):
            results_map[res['rule_id']] = res

        feedback_lines = []
        for rule in rules:
            rule_id = rule.get('rule_id', 'unknown')
            result = results_map.get(rule_id)

            if result is None:
                feedback_lines.append(f"### 规则: {rule.get('name', rule_id)}\n(无评估数据)")
                continue

            recall = result['recall']
            precision = result['precision']
            fpr = result['fpr']
            f1 = result['f1']

            # 诊断问题
            problems = []
            suggestions = []

            if recall < 0.50:
                gap = 0.50 - recall
                problems.append(f"召回率过低: {recall:.1%} (目标 >= 50%，差 {gap:.1%})")
                suggestions.append("降低条件阈值约 10-20%，或去掉最严格的条件")

            if fpr > 0.05:
                gap = fpr - 0.05
                problems.append(f"误杀率过高: {fpr:.1%} (目标 <= 5%，超出 {gap:.1%})")
                suggestions.append("提高条件阈值约 10-20%，或增加一个 AND 条件")

            if precision < 0.40:
                problems.append(f"精确率过低: {precision:.1%} (目标 >= 40%)")
                suggestions.append("收紧条件以减少误报")

            if not problems:
                problems.append("各项指标接近目标但 F1 未达标")

            # 当前的条件值
            current_conditions = []
            for cond in rule.get('conditions', []):
                current_conditions.append(f"{cond['field']} {cond['operator']} {cond['value']}")

            feedback = (
                f"### 规则: {rule.get('name', rule_id)}\n"
                f"当前条件: {', '.join(current_conditions)} (逻辑: {rule.get('logic', 'AND')})\n"
                f"当前指标: Recall={recall:.1%}, Precision={precision:.1%}, "
                f"FPR={fpr:.1%}, F1={f1:.3f}\n"
                f"问题: {'; '.join(problems)}\n"
                f"建议: {'; '.join(suggestions)}\n"
            )
            feedback_lines.append(feedback)

        return "\n".join(feedback_lines)

    def threshold_search(self, rules, graph, gnn_outputs=None):
        """
        阈值微调搜索: 对 LLM 生成的规则做贪心搜索，自动微调阈值

        对每个条件的 value 在 ±30% 范围内搜索，选择 F1 最高的组合。
        这层搜索在 LLM 生成规则后执行，作为规则质量的保底优化。

        Args:
            rules: list[dict], LLM 生成的规则
            graph: DGL 图，用于评估
            gnn_outputs: dict, GNN 输出 (可选) {'probs': [N,2], 'embeddings': [N,dim]}
        Returns:
            list[dict]: 微调后的规则
        """
        from src.stage3_sandbox.rule_parser import RuleParser
        from src.stage3_sandbox.evaluator import RuleEvaluator

        labels = graph.ndata['label'].numpy()
        features = graph.ndata['feature']
        degrees = graph.in_degrees().float()
        feature_zscores = (features - features.mean(0)) / (features.std(0) + 1e-8)
        degree_zscore = (degrees - degrees.mean()) / (degrees.std() + 1e-8)

        ctx = {
            'features': features,
            'degrees': degrees,
            'feature_zscores': feature_zscores,
            'degree_zscore': degree_zscore,
        }

        # 注入 GNN 输出 (与 evaluator 保持一致)
        if gnn_outputs is not None:
            probs = gnn_outputs.get('probs')
            embeddings = gnn_outputs.get('embeddings')
            if probs is not None:
                ctx['gnn_anomaly_scores'] = probs[:, 1]
            else:
                ctx['gnn_anomaly_scores'] = torch.zeros(features.shape[0])
            if embeddings is not None:
                from sklearn.cluster import KMeans
                emb_np = embeddings.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                cluster_ids = kmeans.fit_predict(emb_np)
                anomaly_scores = ctx['gnn_anomaly_scores'].numpy()
                cluster_ids[anomaly_scores < 0.5] = -1
                ctx['embedding_cluster_ids'] = torch.tensor(cluster_ids, dtype=torch.long)
            else:
                ctx['embedding_cluster_ids'] = torch.full((features.shape[0],), -1, dtype=torch.long)
        else:
            ctx['gnn_anomaly_scores'] = torch.zeros(features.shape[0])
            ctx['embedding_cluster_ids'] = torch.full((features.shape[0],), -1, dtype=torch.long)

        total_pos = int((labels == 1).sum())
        total_neg = int((labels == 0).sum())
        parser = RuleParser()

        optimized_rules = []
        for rule in rules:
            best_rule = rule
            best_f1 = -1

            # 收集可调条件
            conditions = rule.get('conditions', [])
            if not conditions:
                optimized_rules.append(rule)
                continue

            # 对每个条件尝试微调
            search_values_list = []
            for cond in conditions:
                value = cond.get('value', 0)
                # 在 ±30% 范围内搜索 11 个点
                values = np.linspace(value * 0.7, value * 1.3, 11)
                # 确保包含原始值
                values = np.unique(np.append(values, value))
                search_values_list.append(values)

            # 贪心搜索: 逐条件优化
            current_rule = dict(rule)
            current_conditions = [dict(c) for c in conditions]
            current_rule['conditions'] = current_conditions

            for cond_idx, search_values in enumerate(search_values_list):
                best_val = current_conditions[cond_idx]['value']
                best_cond_f1 = -1

                for val in search_values:
                    # 临时修改条件值
                    trial_conditions = [dict(c) for c in current_conditions]
                    trial_conditions[cond_idx]['value'] = float(val)
                    trial_rule = dict(current_rule)
                    trial_rule['conditions'] = trial_conditions

                    # 解析并评估
                    try:
                        parsed = parser.parse(trial_rule)
                        mask = parsed(ctx)
                        preds = mask.numpy().astype(int) if hasattr(mask, 'numpy') else np.array(mask, dtype=int)

                        tp = int(((preds == 1) & (labels == 1)).sum())
                        fp = int(((preds == 1) & (labels == 0)).sum())
                        recall = tp / total_pos if total_pos > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                        # 加权评分: F1 为主，惩罚高 FPR
                        fpr = fp / total_neg if total_neg > 0 else 0
                        score = f1 - max(0, fpr - 0.05) * 0.5  # FPR 超过 5% 时扣分

                        if score > best_cond_f1:
                            best_cond_f1 = score
                            best_val = float(val)
                    except Exception:
                        continue

                current_conditions[cond_idx]['value'] = best_val

            # 最终评估
            try:
                parsed = parser.parse(current_rule)
                mask = parsed(ctx)
                preds = mask.numpy().astype(int) if hasattr(mask, 'numpy') else np.array(mask, dtype=int)
                tp = int(((preds == 1) & (labels == 1)).sum())
                fp = int(((preds == 1) & (labels == 0)).sum())
                recall = tp / total_pos if total_pos > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                print(f"  [ThresholdSearch] {rule.get('name', '?')}: "
                      f"original -> Recall={recall:.1%}, Precision={precision:.1%}, F1={f1:.3f}")
            except Exception:
                pass

            current_rule['source'] = rule.get('source', '') + '+threshold_search'
            optimized_rules.append(current_rule)

        return optimized_rules

    def _call_llm(self, system_prompt, user_prompt):
        """调用 LLM API"""
        if self.llm_client is None:
            print("[MOCK] LLM not available, returning mock rules")
            return self._mock_response()

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config['model'],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return self._mock_response()

    def _parse_rules(self, raw_response):
        """从 LLM 原始输出中解析结构化规则"""
        try:
            rules = json.loads(raw_response)
            if isinstance(rules, list):
                return rules
        except json.JSONDecodeError:
            pass

        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_response)
        if json_match:
            try:
                rules = json.loads(json_match.group(1))
                if isinstance(rules, list):
                    return rules
            except json.JSONDecodeError:
                pass

        print("[WARNING] Failed to parse LLM output, returning empty rules")
        return []

    def _mock_response(self):
        """Mock 规则 (用于无 LLM API 时的测试)"""
        mock_rules = [
            {
                "name": "高度数异常交易检测",
                "description": "检测交易网络中度数异常偏高的节点，可能是集团欺诈的核心账户",
                "risk_type": "集团欺诈",
                "severity": "high",
                "conditions": [
                    {"field": "node_degree", "operator": ">", "value": 50},
                    {"field": "feature_dim_0", "operator": ">", "value": 2.0}
                ],
                "logic": "AND",
                "action": "review",
                "explanation": "基于 GNN Cluster-0 模式: 高连接度 + 异常特征组合"
            },
            {
                "name": "孤立高风险账户检测",
                "description": "检测低度数但特征异常的节点，可能是新型单点欺诈",
                "risk_type": "盗刷",
                "severity": "medium",
                "conditions": [
                    {"field": "node_degree", "operator": "<", "value": 5},
                    {"field": "feature_dim_1", "operator": ">", "value": 3.0}
                ],
                "logic": "AND",
                "action": "alert",
                "explanation": "基于 GNN Cluster-1 模式: 低连接度 + 单特征维度极端值"
            }
        ]
        return json.dumps(mock_rules, ensure_ascii=False)

    def save(self, rules, output_dir='outputs/'):
        """保存生成的规则"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'generated_rules.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        print(f"[RuleGen] Saved {len(rules)} rules to {path}")
        return path
