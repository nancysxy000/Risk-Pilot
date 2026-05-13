"""
Stage 2: LLM 规则自动生成模块

核心流程:
1. 读取 GNN 输出的 risk_insights.json
2. 从知识库 RAG 检索相关历史规则
3. 组装 Prompt + 调用 LLM
4. 解析 LLM 输出为结构化规则 JSON
"""

import json
import os
from datetime import datetime


class RuleGenerator:
    """LLM 驱动的风控规则生成器"""

    def __init__(self, config, rag_retriever=None):
        self.config = config['llm']
        self.rag = rag_retriever
        self.llm_client = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM 客户端，支持 DashScope / OpenAI / Ollama"""
        # 加载 .env 文件
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # 没有 python-dotenv 时依赖手动设置环境变量

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
        """基于沙盒回测结果优化规则"""
        from .prompt_templates import SYSTEM_PROMPT, RULE_OPTIMIZATION_PROMPT

        historical_rules = "无历史规则"
        if self.rag:
            historical_rules = self.rag.retrieve_by_text(
                json.dumps(evaluation_results, ensure_ascii=False)
            )

        user_prompt = RULE_OPTIMIZATION_PROMPT.format(
            rules_to_optimize=json.dumps(rules, indent=2, ensure_ascii=False),
            evaluation_results=json.dumps(evaluation_results, indent=2, ensure_ascii=False),
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
            # 尝试直接解析 JSON
            rules = json.loads(raw_response)
            if isinstance(rules, list):
                return rules
        except json.JSONDecodeError:
            pass

        # 尝试从 Markdown 代码块中提取 JSON
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
