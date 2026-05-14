"""
RiskPilot — 主 Pipeline 编排

完整闭环:
  Stage 1 (GNN 感知) → Stage 2 (LLM 策略) → Stage 3 (沙盒验证) → Stage 4 (规则沉淀)
  ↑                                                                      │
  └──────────────────── 闭环反馈 (知识库 RAG) ──────────────────────────────┘

运行方式:
  python -m src.pipeline --dataset tfinance --mode full

输出文件 (outputs/ 目录):
  - gnn_predictions.pt      Stage 1 → 每个节点的异常概率 [N, 2]
  - node_embeddings.pt      Stage 1 → 节点嵌入向量 [N, hidden_dim]
  - risk_insights.json      Stage 1 → 结构化风险洞察 (5 种风险模式)
  - generated_rules.json    Stage 2 → LLM 生成的风控规则
  - evaluation_report.json  Stage 3 → 沙盒回测指标 (Recall/Precision/FPR)
  - qualified_rules.json    Stage 4 → 通过回测的优质规则

tfinance 数据集运行结果:
  - GNN 检测到 500 个高风险节点 (Top-K)，聚类为 5 种异常模式
  - Mock 模式生成 2 条规则，均因 FPR 过高或 Recall 过低未通过
  - 接入真实 LLM 后，闭环迭代会自动优化规则直到达标
"""

import os
import json
import argparse
import yaml
import numpy as np


def load_config(config_path='configs/default.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_data_distribution(graph):
    """
    计算图级数据分布统计，供 LLM 设定规则阈值参考

    Returns:
        str: 格式化的分布统计文本
    """
    features = graph.ndata['feature'].numpy()
    degrees = graph.in_degrees().float().numpy()

    lines = ["### 节点度数分布"]
    lines.append(f"- mean={degrees.mean():.1f}, std={degrees.std():.1f}, "
                 f"median={np.median(degrees):.1f}")
    for p in [90, 95, 99]:
        lines.append(f"- P{p}={np.percentile(degrees, p):.1f}")
    lines.append(f"- min={degrees.min():.0f}, max={degrees.max():.0f}")

    lines.append("\n### 各特征维度分布")
    for i in range(features.shape[1]):
        col = features[:, i]
        lines.append(f"- feature_dim_{i}: mean={col.mean():.3f}, std={col.std():.3f}, "
                      f"min={col.min():.3f}, max={col.max():.3f}")

    return "\n".join(lines)


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

    # 计算数据分布统计 (供 LLM 参考设定阈值)
    data_distribution = compute_data_distribution(trainer.graph)

    rules = generator.generate(
        risk_insights,
        dataset_stats=trainer.dataset.stats,
        data_distribution=data_distribution,
    )

    # 阈值微调搜索: 在 LLM 生成规则后自动搜索最优阈值
    gnn_outputs = {'probs': probs, 'embeddings': embeddings}
    print(f"  LLM 生成 {len(rules)} 条规则，开始阈值微调搜索...")
    rules = generator.threshold_search(rules, trainer.graph, gnn_outputs=gnn_outputs)
    generator.save(rules, output_dir)
    print(f"  最终保留 {len(rules)} 条规则 (含阈值优化)")

    # ================================================================
    # Stage 3: 沙盒验证层 — 规则回测
    # ================================================================
    print("\n" + "=" * 60)
    print("  Stage 3: 沙盒验证 — 规则回测")
    print("=" * 60)

    from src.stage3_sandbox.evaluator import RuleEvaluator
    evaluator = RuleEvaluator(config)

    # 评估所有规则
    report = evaluator.evaluate(rules, trainer.graph, gnn_outputs=gnn_outputs)
    evaluator.save_report(report, output_dir)

    # ================================================================
    # 闭环迭代: 优化不合格规则 (含稳定性保障)
    # ================================================================
    max_iterations = config['pipeline']['max_iterations']

    # 用 all_report_results 累积所有轮次的评估结果，防止 report 覆盖丢失合格规则
    all_report_results = list(report['individual_results'])

    # 追踪历史最优，防止迭代退化
    best_rules = list(rules)
    best_qualified_count = report['qualified_rules']
    best_combined_f1 = 0
    combined = report.get('combined_result', {})
    if combined.get('combined_recall', 0) > 0 and combined.get('combined_precision', 0) > 0:
        r = combined['combined_recall']
        p = combined['combined_precision']
        best_combined_f1 = 2 * p * r / (p + r)

    no_improvement_count = 0

    for iteration in range(1, max_iterations):
        # 检查是否有需要优化的规则 (只看原始规则 + 已累积的合格规则)
        rules_to_optimize = evaluator.get_rules_to_optimize(rules, all_report_results)
        if not rules_to_optimize:
            print(f"\n[Pipeline] 所有规则均已合格，无需继续优化")
            break

        # Early stop: 连续 2 轮无改善
        if no_improvement_count >= 2:
            print(f"\n[Pipeline] 连续 {no_improvement_count} 轮无改善，停止迭代")
            # 回退到历史最优
            if best_qualified_count > sum(1 for r in all_report_results if r['qualified']):
                print(f"  回退到历史最优: {best_qualified_count} 条合格规则")
                rules = best_rules
            break

        print(f"\n{'='*60}")
        print(f"  闭环迭代 #{iteration}: 优化 {len(rules_to_optimize)} 条不合格规则")
        print(f"{'='*60}")

        # LLM 定向优化 (使用定向反馈而非全量报告)
        optimized_rules = generator.optimize(rules_to_optimize, report, data_distribution)
        if not optimized_rules:
            print("[Pipeline] LLM 未返回优化结果，停止迭代")
            break

        # 对优化后的规则再做阈值搜索
        optimized_rules = generator.threshold_search(optimized_rules, trainer.graph, gnn_outputs=gnn_outputs)

        # 重新评估
        new_report = evaluator.evaluate(optimized_rules, trainer.graph, gnn_outputs=gnn_outputs)

        # 累积本轮结果 (不覆盖历史)
        all_report_results.extend(new_report['individual_results'])

        # 计算本轮综合 F1
        new_combined = new_report.get('combined_result', {})
        new_f1 = 0
        if new_combined.get('combined_recall', 0) > 0 and new_combined.get('combined_precision', 0) > 0:
            nr = new_combined['combined_recall']
            np_ = new_combined['combined_precision']
            new_f1 = 2 * np_ * nr / (np_ + nr)

        # 比较是否改善 (合格规则数 + 综合 F1)
        improved = (new_report['qualified_rules'] > best_qualified_count or
                    new_f1 > best_combined_f1)

        if improved:
            best_rules = list(rules)
            best_qualified_count = max(best_qualified_count, new_report['qualified_rules'])
            best_combined_f1 = max(best_combined_f1, new_f1)
            no_improvement_count = 0
            print(f"  本轮有改善: qualified={new_report['qualified_rules']}, F1={new_f1:.3f}")
        else:
            no_improvement_count += 1
            print(f"  本轮无改善 (连续 {no_improvement_count} 次): qualified={new_report['qualified_rules']}, F1={new_f1:.3f}")

        # 更新 report 为最新结果 (供下一轮 optimize 使用)
        report = new_report
        evaluator.save_report(report, output_dir)

        # 合并合格规则到 rules 列表
        qualified_new = [
            r for r, result in zip(optimized_rules, new_report['individual_results'])
            if result['qualified']
        ]
        # 同时把所有优化后的规则加入 rules (下一轮可以再次评估)
        rules.extend(optimized_rules)

    # ================================================================
    # Stage 4 (沉淀): 将合格规则写入知识库
    # ================================================================
    print("\n" + "=" * 60)
    print("  Stage 4: 规则沉淀至知识库")
    print("=" * 60)

    # 从累积的所有评估结果中筛选合格规则
    qualified_rules = [
        r for r in rules
        if any(
            res['rule_id'] == r.get('rule_id') and res['qualified']
            for res in all_report_results
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
