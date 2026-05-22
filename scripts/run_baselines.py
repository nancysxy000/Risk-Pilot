"""
Baseline 对比实验运行脚本

支持三种对比模式:
1. 传统方法 (IF/LOF/OCSVM/PCA):   python scripts/run_baselines.py --traditional --all
2. 统计规则 (不用 LLM):           python scripts/run_baselines.py --statistical --all
3. 无迭代 (LLM 只跑 1 轮):        python scripts/run_baselines.py --no-iteration --all

也可以混合: python scripts/run_baselines.py --traditional --statistical --no-iteration --all
"""

import os
import sys
import json
import argparse

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 惰性导入：只在需要时导入，避免 DGL 在非传统方法模式下报错
# from src.baselines.traditional import TraditionalBaselineRunner
# from src.baselines.statistical_rules import StatisticalRuleGenerator

# GNN 基线参考结果 (来自之前的实验)
GNN_REFERENCE = {
    'tfinance': {
        'recall': 0.7600,
        'precision': 0.8816,
        'macro_f1': 0.9041,
        'auc': 0.9604,
    },
    'yelp': {
        'recall': 0.5632,
        'precision': 0.6697,
        'macro_f1': 0.7069,
        'auc': 0.8310,
    },
    'amazon': {
        'recall': 0.8456,
        'precision': 0.8817,
        'macro_f1': 0.9236,
        'auc': 0.9649,
    },
}

# 基线 Pipeline 规则结果 (LLM + 闭环迭代, 来自之前实验)
PIPELINE_REFERENCE = {
    'tfinance': {'recall': 0.6985, 'precision': 0.7604, 'fpr': 0.0350, 'qualified': 12},
    'yelp':     {'recall': 0.2462, 'precision': 0.3945, 'fpr': 0.0642, 'qualified': 0},
    'amazon':   {'recall': 0.7978, 'precision': 0.8875, 'fpr': 0.0075, 'qualified': 6},
}

DATASETS = ['tfinance', 'yelp', 'amazon']


def get_config(dataset_name):
    """获取数据集配置"""
    return {
        'dataset': {
            'name': dataset_name, 'path': 'dataset/', 'homo': True,
            'train_ratio': 0.4, 'anomaly_alpha': None, 'anomaly_std': None,
        },
        'gnn': {
            'model': 'BWGNN', 'hidden_dim': 64, 'order': 2,
            'epochs': 100, 'learning_rate': 0.01, 'device': 'cpu',
            'insight': {'top_k_anomaly': 500, 'anomaly_threshold': 0.5,
                        'n_clusters': 5, 'k_hop': 2},
        },
        'llm': {
            'provider': 'openai', 'model': 'qwen-plus', 'temperature': 0.3,
            'max_tokens': 4096, 'api_key_env': 'OPENAI_API_KEY',
            'base_url_env': 'OPENAI_BASE_URL',
            'rag': {'embedding_model': 'text-embedding-3-small', 'top_k_retrieve': 5,
                    'similarity_threshold': 0.7},
            'rule_gen': {'max_rules_per_round': 10, 'max_optimization_rounds': 3},
        },
        'sandbox': {
            'thresholds': {'min_f1': 0.60, 'min_recall': 0.50,
                           'max_fpr': 0.05, 'min_precision': 0.40},
            'backtest_ratio': 1.0,
        },
        'knowledge_base': {
            'vector_store': 'chroma', 'persist_dir': '/tmp/kb_baseline', 'collection_name': 'baseline',
        },
        'pipeline': {
            'max_iterations': 5, 'output_dir': 'outputs_baselines/', 'log_level': 'INFO',
        },
    }


def run_traditional(dataset_name):
    """运行传统方法 baseline"""
    from src.baselines.traditional import TraditionalBaselineRunner

    config = get_config(dataset_name)
    output_dir = f'outputs_baselines/traditional_{dataset_name}/'

    runner = TraditionalBaselineRunner(config)
    results = runner.run_all()
    results['gnn_reference'] = GNN_REFERENCE.get(dataset_name)

    runner.save_results(results, output_dir)
    runner.print_comparison(results, GNN_REFERENCE.get(dataset_name))
    return results


def run_statistical(dataset_name):
    """
    运行统计规则实验 (不用 LLM)

    流程: Stage 1 (GNN + Insight) → StatisticalRuleGenerator → threshold_search → evaluate
    """
    from src.stage1_gnn.train import GNNTrainer
    from src.stage1_gnn.insight_extractor import InsightExtractor
    from src.stage2_llm.rule_generator import RuleGenerator
    from src.stage3_sandbox.evaluator import RuleEvaluator

    config = get_config(dataset_name)
    output_dir = f'outputs_baselines/statistical_{dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  统计规则实验 — {dataset_name}")
    print(f"{'='*60}")

    # Stage 1: GNN + Insight (与基线完全一致)
    print("\n  [Stage 1] GNN 训练...")
    trainer = GNNTrainer(config)
    trainer.load_data()
    trainer.build_model()
    gnn_metrics = trainer.train()
    probs, embeddings = trainer.save_outputs(output_dir)

    extractor = InsightExtractor(config)
    risk_insights = extractor.extract(trainer.graph, probs, embeddings)
    extractor.save(risk_insights, output_dir)

    # 用统计规则生成器替代 LLM
    print("\n  [统计规则] 从 risk_insights 自动生成规则 (不调用 LLM)...")
    from src.baselines.statistical_rules import StatisticalRuleGenerator
    stat_gen = StatisticalRuleGenerator(config)
    rules = stat_gen.generate(risk_insights, dataset_stats=trainer.dataset.stats)
    print(f"  生成 {len(rules)} 条统计规则")

    # 仍然跑 threshold_search (公平对比)
    gnn_outputs = {'probs': probs, 'embeddings': embeddings}
    # 借用 RuleGenerator 的 threshold_search
    llm_gen = RuleGenerator(config)
    rules = llm_gen.threshold_search(rules, trainer.graph, gnn_outputs=gnn_outputs)

    # 保存规则
    rules_path = os.path.join(output_dir, 'generated_rules.json')
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    # Stage 3: 评估
    print("\n  [评估] 沙盒回测...")
    evaluator = RuleEvaluator(config)
    report = evaluator.evaluate(rules, trainer.graph, gnn_outputs=gnn_outputs)
    evaluator.save_report(report, output_dir)

    # 输出结果
    combined = report.get('combined_result', {})
    print(f"\n  结果: Recall={combined.get('combined_recall', 0):.2%}, "
          f"Precision={combined.get('combined_precision', 0):.2%}, "
          f"FPR={combined.get('combined_fpr', 0):.2%}")
    print(f"  合格规则: {report['qualified_rules']}/{report['total_rules']}")

    return {
        'dataset': dataset_name,
        'mode': 'statistical',
        'recall': combined.get('combined_recall', 0),
        'precision': combined.get('combined_precision', 0),
        'fpr': combined.get('combined_fpr', 0),
        'qualified_rules': report['qualified_rules'],
        'total_rules': report['total_rules'],
        'gnn_metrics': gnn_metrics,
    }


def run_no_iteration(dataset_name):
    """
    运行无迭代实验 (LLM 只跑 1 轮，不闭环优化)

    流程: Stage 1 → LLM generate (1 轮) → threshold_search → evaluate
    """
    # 使用 pipeline 的 max_iterations=1
    from src import pipeline as pipeline_mod

    config = get_config(dataset_name)
    config['pipeline']['max_iterations'] = 1
    config['pipeline']['output_dir'] = f'outputs_baselines/no_iteration_{dataset_name}/'

    # 知识库配置
    config['knowledge_base']['persist_dir'] = f'/tmp/kb_no_iter_{dataset_name}'

    print(f"\n{'='*60}")
    print(f"  无迭代实验 — {dataset_name}")
    print(f"{'='*60}")

    result = pipeline_mod.run_pipeline(config, mode='full')
    return result


def main():
    parser = argparse.ArgumentParser(description='Run baseline comparison experiments')
    parser.add_argument('--traditional', action='store_true',
                        help='Run traditional method baselines (IF/LOF/OCSVM/PCA)')
    parser.add_argument('--statistical', action='store_true',
                        help='Run statistical rule generation (no LLM)')
    parser.add_argument('--no-iteration', action='store_true',
                        help='Run LLM with no iteration (1 round only)')
    parser.add_argument('--all-modes', action='store_true',
                        help='Run all three modes')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name: tfinance, yelp, amazon')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run on all three datasets')
    args = parser.parse_args()

    if args.all_modes:
        args.traditional = args.statistical = args.no_iteration = True

    if not any([args.traditional, args.statistical, args.no_iteration]):
        parser.print_help()
        return

    datasets = DATASETS if args.all_datasets else ([args.dataset] if args.dataset else [])
    if not datasets:
        print("请指定 --dataset tfinance 或 --all-datasets")
        return

    all_results = {}

    for ds in datasets:
        all_results[ds] = {}

        if args.traditional:
            print(f"\n{'#'*70}")
            print(f"  传统方法 — {ds}")
            print(f"{'#'*70}")
            all_results[ds]['traditional'] = run_traditional(ds)

        if args.statistical:
            print(f"\n{'#'*70}")
            print(f"  统计规则 — {ds}")
            print(f"{'#'*70}")
            all_results[ds]['statistical'] = run_statistical(ds)

        if args.no_iteration:
            print(f"\n{'#'*70}")
            print(f"  无迭代 — {ds}")
            print(f"{'#'*70}")
            all_results[ds]['no_iteration'] = run_no_iteration(ds)

    # 汇总打印
    if len(datasets) > 1 or sum([args.traditional, args.statistical, args.no_iteration]) > 1:
        print_summary(all_results)


def print_summary(all_results):
    """打印所有实验的汇总对比"""
    print(f"\n{'='*80}")
    print(f"  实验汇总对比")
    print(f"{'='*80}")
    print(f"  {'Dataset':<12} {'Method':<20} {'Recall':>8} {'Precision':>10} {'Qualified':>10}")
    print(f"  {'-'*60}")

    for ds, modes in all_results.items():
        for mode, res in modes.items():
            if mode == 'traditional':
                # 传统方法取最佳
                best = max(res['methods'].values(), key=lambda x: x['macro_f1'])
                name = 'Best Traditional'
                recall = best['recall']
                precision = best['precision']
                qualified = '-'
            elif mode == 'statistical':
                name = 'Statistical Rules'
                recall = res.get('recall', 0)
                precision = res.get('precision', 0)
                qualified = str(res.get('qualified_rules', 0))
            elif mode == 'no_iteration':
                name = 'LLM (1 round)'
                # 从 no_iteration 结果中提取
                recall = '-'
                precision = '-'
                qualified = '-'
            else:
                continue

            print(f"  {ds:<12} {name:<20} {recall:>8} {precision:>10} {qualified:>10}")

        # 打印基线
        ref = PIPELINE_REFERENCE.get(ds, {})
        if ref:
            print(f"  {ds:<12} {'LLM + Iteration':<20} {ref['recall']:>8.2%} "
                  f"{ref['precision']:>10.2%} {ref['qualified']:>10}")
        print(f"  {'-'*60}")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
