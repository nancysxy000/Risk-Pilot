"""
Stage 4: 规则知识库 — 规则数据模型
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime


@dataclass
class RuleCondition:
    field: str
    operator: str
    value: float

    def to_dict(self):
        return asdict(self)


@dataclass
class Rule:
    """风控规则数据模型"""
    rule_id: str
    name: str
    description: str = ""
    risk_type: str = ""
    severity: str = "medium"             # high / medium / low
    conditions: List[dict] = field(default_factory=list)
    logic: str = "AND"                   # AND / OR
    action: str = "alert"                # block / review / alert / limit
    explanation: str = ""
    source: str = "llm_auto_gen"

    # 性能指标 (来自沙盒评估)
    recall: float = 0.0
    precision: float = 0.0
    fpr: float = 0.0
    f1: float = 0.0

    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    is_active: bool = True

    def to_dict(self):
        return asdict(self)

    def to_embedding_text(self):
        """生成用于向量化的文本描述"""
        cond_text = " ".join(
            f"{c['field']} {c['operator']} {c['value']}" for c in self.conditions
        )
        return (
            f"{self.name}. {self.description}. "
            f"Risk type: {self.risk_type}. "
            f"Conditions: {cond_text}. "
            f"Severity: {self.severity}. "
            f"F1: {self.f1:.3f}, Recall: {self.recall:.3f}."
        )
