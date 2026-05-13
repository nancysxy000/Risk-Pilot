"""
Stage 3: 沙盒验证层 — 规则解析器

将 LLM 生成的 JSON 规则转化为可执行的 Python 函数。
支持的字段:
  - node_degree: 节点度数 (原始值)
  - node_degree_zscore: 节点度数的 z-score
  - feature_dim_X: 第 X 维原始特征值
  - feature_dim_X_zscore: 第 X 维特征的 z-score
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
            callable: 接受 ctx dict 返回 bool mask 的函数
        """
        conditions = rule_dict.get('conditions', [])
        logic = rule_dict.get('logic', 'AND').upper()

        condition_fns = []
        for cond in conditions:
            fn = self._parse_condition(cond)
            if fn:
                condition_fns.append(fn)

        if not condition_fns:
            return lambda ctx: ctx['degrees'].new_zeros(ctx['degrees'].shape[0]).bool()

        if logic == 'AND':
            def rule_fn(ctx):
                mask = condition_fns[0](ctx)
                for fn in condition_fns[1:]:
                    mask = mask & fn(ctx)
                return mask
        else:  # OR
            def rule_fn(ctx):
                mask = condition_fns[0](ctx)
                for fn in condition_fns[1:]:
                    mask = mask | fn(ctx)
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
            return lambda ctx, op=op_fn, v=float(value): op(ctx['degrees'], v)
        elif field == 'node_degree_zscore':
            return lambda ctx, op=op_fn, v=float(value): op(ctx['degree_zscore'], v)
        elif field.startswith('feature_dim_') and field.endswith('_zscore'):
            dim = int(field.replace('feature_dim_', '').replace('_zscore', ''))
            return lambda ctx, op=op_fn, d=dim, v=float(value): op(ctx['feature_zscores'][:, d], v)
        elif field.startswith('feature_dim_'):
            dim = int(field.replace('feature_dim_', ''))
            return lambda ctx, op=op_fn, d=dim, v=float(value): op(ctx['features'][:, d], v)
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
