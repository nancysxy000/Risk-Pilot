"""
RiskPilot — 主 Pipeline 编排

完整闭环:
  Stage 1 (GNN 感知) → Stage 2 (LLM 策略) → Stage 3 (沙盒验证) → Stage 4 (规则沉淀)
  ↑                                                                      │
  └──────────────────── 闭环反馈 (知识库 RAG) ──────────────────────────────┘
"""

import os
import json
import argparse
import yaml


def load_config(config_path='configs/default.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline(config, mode='full'):
    """
    运行 RiskPilot Pipeline

    Args:
        config: dict, 配置
        mode: str, 运行模式
            - 'full': 完整 4 阶段闭环
            - 'gnn_only': 仅 Stage 1 (GNN 训练 + 洞察提取)
            - 'eval_only': 仅 Stage 3 (评估已有规则)
    """
    output_dir = config['pipeline']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # Stage 1: GNN 感知层 — 异常检测与风险洞察提取
    # ================================================================
    print("\n" + "=" * 60)
    print("  Stage 1: GNN 感知层")
    print("=" * 60)

    from src.stage1_gnn.train import GNNTrainer
    from src.stage1_gnn.insight_extractor import InsightExtractor

    trainer = GNNTrainer(config)
    trainer.load_data()
    trainer.build_model()
    metrics = trainer.train()

    # 保存预测结果和嵌入向量
    probs, embeddings = trainer.save_outputs(output_dir)

    # 提取结构化风险洞察
    extractor = InsightExtractor(config)
    risk_insights = extractor.extract(trainer.graph, probs, embeddings)
    extractor.save(risk_insights, output_dir)

    if mode == 'gnn_only':
        print("\n[Pipeline] GNN-only mode complete.")
        return {'stage1': metrics, 'risk_insights': risk_insights}

    # ================================================================
    # Stage 4 (预加载): 初始化知识库 (供 Stage 2 RAG 使用)
    # ================================================================
    print("\n" + "=" * 60)
    print("  Stage 4: 初始化规则知识库")
    print("=" * 60)

    from src.stage4_kb.knowledge_base import RuleKnowledgeBase
    kb = RuleKnowledgeBase(config)

    # ================================================================
    # Stage 2: LLM 策略层 — 规则生成
    # ================================================================
    print("\n" + "=" * 60)
    print("  Stage 2: LLM 规则生成")
    print("=" * 60)

    from src.stage2_llm.rag_retriever import RAGRetriever
    from src.stage2_llm.rule_generator import RuleGenerator

    rag = RAGRetriever(config, knowledge_base=kb)
    generator = RuleGenerator(config, rag_retriever=rag)

    rules = generator.generate(
        risk_insights,
        dataset_stats=trainer.dataset.stats
    )
    generator.save(rules, output_dir)
    print(f"  Generated {len(rules)} rules")

    # ================================================================
    # Stage 3: 沙盒验证层 — 规则回测
    # ================================================================
    print("\n" + "=" * 60)
    print("  Stage 3: 沙盒验证 — 规则回测")
    print("=" * 60)

    from src.stage3_sandbox.evaluator import RuleEvaluator
    evaluator = RuleEvaluator(config)
    report = evaluator.evaluate(rules, trainer.graph)
    evaluator.save_report(report, output_dir)

    # ================================================================
    # 闭环迭代: 优化不合格规则
    # ================================================================
    max_iterations = config['pipeline']['max_iterations']

    for iteration in range(1, max_iterations):
        # 检查是否有需要优化的规则
        rules_to_optimize = evaluator.get_rules_to_optimize(rules, report['individual_results'])
        if not rules_to_optimize:
            print(f"\n[Pipeline] 所有规则均已合格，无需继续优化")
            break

        print(f"\n{'='*60}")
        print(f"  闭环迭代 #{iteration}: 优化 {len(rules_to_optimize)} 条不合格规则")
        print(f"{'='*60}")

        # LLM 重新优化
        optimized_rules = generator.optimize(rules_to_optimize, report)
        if not optimized_rules:
            print("[Pipeline] LLM 未返回优化结果，停止迭代")
            break

        # 重新评估
        report = evaluator.evaluate(optimized_rules, trainer.graph)
        evaluator.save_report(report, output_dir)

        # 合并合格规则
        qualified_new = [
            r for r, result in zip(optimized_rules, report['individual_results'])
            if result['qualified']
        ]
        rules.extend(qualified_new)

    # ================================================================
    # Stage 4 (沉淀): 将合格规则写入知识库
    # ================================================================
    print("\n" + "=" * 60)
    print("  Stage 4: 规则沉淀至知识库")
    print("=" * 60)

    qualified_rules = [
        r for r in rules
        if any(
            res['rule_id'] == r.get('rule_id') and res['qualified']
            for res in report['individual_results']
        )
    ]

    if qualified_rules:
        kb.add_rules(qualified_rules)
        # 保存合格规则
        qualified_path = os.path.join(output_dir, 'qualified_rules.json')
        with open(qualified_path, 'w', encoding='utf-8') as f:
            json.dump(qualified_rules, f, indent=2, ensure_ascii=False)
        print(f"  {len(qualified_rules)} 条优质规则已沉淀至知识库")
    else:
        print("  本轮无合格规则沉淀")

    # ================================================================
    # 最终报告
    # ================================================================
    print("\n" + "=" * 60)
    print("  RiskPilot Pipeline 完成!")
    print("=" * 60)
    print(f"  GNN Metrics:     Macro-F1={metrics['test_macro_f1']:.2%}, AUC={metrics['test_auc']:.2%}")
    print(f"  Risk Patterns:   {len(risk_insights['risk_patterns'])} 种")
    print(f"  Rules Generated: {len(rules)} 条")
    print(f"  Rules Qualified: {len(qualified_rules)} 条")
    print(f"  Knowledge Base:  {kb.collection.count() if kb.collection else len(kb._rules_cache)} 条")
    print(f"  Output Dir:      {output_dir}")
    print("=" * 60)

    return {
        'stage1_metrics': metrics,
        'risk_insights': risk_insights,
        'rules': rules,
        'evaluation': report,
        'qualified_rules': qualified_rules,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RiskPilot Pipeline')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset name (tfinance/tsocial/yelp/amazon)')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'gnn_only', 'eval_only'])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset:
        config['dataset']['name'] = args.dataset

    run_pipeline(config, mode=args.mode)
