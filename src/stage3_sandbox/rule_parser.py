"""
Stage 3: 沙盒验证层 — 规则解析器

将 LLM 生成的 JSON 规则转化为可执行的 Python 函数。
"""

import operator


OPERATOR_MAP = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
}


class RuleParser:
    """将 JSON 规则解析为可执行的过滤函数"""

    def parse(self, rule_dict):
        """
        解析单条规则为可执行函数

        Args:
            rule_dict: 规则 JSON dict
        Returns:
            callable: 接受 (features, degrees) 返回 bool mask 的函数
        """
        conditions = rule_dict.get('conditions', [])
        logic = rule_dict.get('logic', 'AND').upper()

        condition_fns = []
        for cond in conditions:
            fn = self._parse_condition(cond)
            if fn:
                condition_fns.append(fn)

        if not condition_fns:
            return lambda features, degrees: features.new_zeros(features.shape[0]).bool()

        if logic == 'AND':
            def rule_fn(features, degrees):
                mask = condition_fns[0](features, degrees)
                for fn in condition_fns[1:]:
                    mask = mask & fn(features, degrees)
                return mask
        else:  # OR
            def rule_fn(features, degrees):
                mask = condition_fns[0](features, degrees)
                for fn in condition_fns[1:]:
                    mask = mask | fn(features, degrees)
                return mask

        return rule_fn

    def _parse_condition(self, cond):
        """解析单个条件"""
        field = cond.get('field', '')
        op_str = cond.get('operator', '')
        value = cond.get('value')

        if op_str not in OPERATOR_MAP:
            print(f"[WARNING] Unsupported operator: {op_str}")
            return None

        op_fn = OPERATOR_MAP[op_str]

        if field == 'node_degree':
            return lambda features, degrees, op=op_fn, v=float(value): op(degrees, v)
        elif field.startswith('feature_dim_'):
            dim = int(field.replace('feature_dim_', ''))
            return lambda features, degrees, op=op_fn, d=dim, v=float(value): op(features[:, d], v)
        else:
            print(f"[WARNING] Unknown field: {field}")
            return None

    def parse_all(self, rules_list):
        """批量解析规则列表"""
        parsed = []
        for rule in rules_list:
            fn = self.parse(rule)
            parsed.append({
                'rule_id': rule.get('rule_id', 'unknown'),
                'name': rule.get('name', ''),
                'severity': rule.get('severity', 'medium'),
                'action': rule.get('action', 'alert'),
                'fn': fn,
                'meta': rule,
            })
        return parsed
