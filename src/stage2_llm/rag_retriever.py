"""
Stage 2: RAG 检索模块 — 从知识库中检索相关历史规则
"""

import json


class RAGRetriever:
    """从规则知识库中检索相关历史规则"""

    def __init__(self, config, knowledge_base=None):
        self.config = config['llm']['rag']
        self.kb = knowledge_base
        self.top_k = self.config['top_k_retrieve']

    def retrieve(self, risk_insights):
        """
        基于 GNN 风险洞察检索相关历史规则

        Args:
            risk_insights: dict, 风险洞察
        Returns:
            str: 格式化的历史规则文本
        """
        if self.kb is None or self.kb.is_empty():
            return "暂无历史规则 (知识库为空)"

        # 构造检索 query: 每个风险模式的描述
        results = []
        for pattern in risk_insights.get('risk_patterns', []):
            desc = pattern.get('risk_description', '')
            if desc:
                similar_rules = self.kb.search(desc, top_k=self.top_k)
                results.extend(similar_rules)

        if not results:
            return "未检索到相关历史规则"

        # 去重 + 格式化
        seen = set()
        unique_rules = []
        for rule in results:
            rule_id = rule.get('rule_id', '')
            if rule_id not in seen:
                seen.add(rule_id)
                unique_rules.append(rule)

        return json.dumps(unique_rules[:self.top_k], indent=2, ensure_ascii=False)

    def retrieve_by_text(self, query_text):
        """基于文本直接检索"""
        if self.kb is None or self.kb.is_empty():
            return "暂无历史规则"

        results = self.kb.search(query_text, top_k=self.top_k)
        return json.dumps(results, indent=2, ensure_ascii=False)
