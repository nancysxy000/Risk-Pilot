"""
论文图表生成脚本 — 基于 outputs/ 下的实际运行数据生成所有论文和 PPT 所需图表

用法:
  cd Risk-Pilot
  python report/paper/figures/generate_figures.py

输出: report/paper/figures/ 下的 PDF 和 PNG 文件
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ---- 路径配置 ----
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = ROOT  # 图表输出到 figures/ 目录
INSIGHTS_PATH = os.path.join(ROOT, '../../../outputs/risk_insights.json')
EVAL_PATH = os.path.join(ROOT, '../../../outputs/evaluation_report.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 加载数据 ----
with open(INSIGHTS_PATH, 'r') as f:
    insights = json.load(f)
with open(EVAL_PATH, 'r') as f:
    eval_report = json.load(f)


# ============================================================
# Figure 1: Pipeline 架构图
# ============================================================
def fig1_pipeline():
    """RiskPilot 四阶段闭环 Pipeline 架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # 四个 Stage 方框
    stages = [
        (1.0, 2.5, 'Stage 1\nGNN Perception', '#4C72B0'),
        (4.0, 2.5, 'Stage 2\nLLM Strategy', '#55A868'),
        (7.0, 2.5, 'Stage 3\nSandbox Validation', '#C44E52'),
        (10.0, 2.5, 'Stage 4\nRule Knowledge Base', '#8172B2'),
    ]

    boxes = []
    for x, y, label, color in stages:
        box = FancyBboxPatch((x - 0.9, y - 0.6), 1.8, 1.2,
                             boxstyle="round,pad=0.1", facecolor=color,
                             edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        boxes.append((x, y))

    # 正向箭头
    arrow_style = dict(arrowstyle='->', color='#333333', lw=2, mutation_scale=20)
    for i in range(3):
        x1, y1 = boxes[i]
        x2, y2 = boxes[i + 1]
        ax.annotate('', xy=(x2 - 0.9, y2), xytext=(x1 + 0.9, y1),
                     arrowprops=arrow_style)

    # 反馈箭头 (Stage 4 → Stage 1)
    ax.annotate('', xy=(boxes[0][0], boxes[0][1] - 0.8),
                xytext=(boxes[3][0], boxes[3][1] - 0.8),
                arrowprops=dict(arrowstyle='->', color='#DD8452', lw=2,
                                connectionstyle='arc3,rad=0.3', mutation_scale=20))
    ax.text(5.5, 1.0, 'Feedback Loop (RAG)', ha='center', va='center',
            fontsize=9, fontstyle='italic', color='#DD8452')

    # 下方标注
    labels_below = [
        'Anomaly Detection\n+ Risk Insights',
        'Auto Rule\nGeneration',
        'Backtest\nMetrics',
        'Rule\nAccumulation',
    ]
    for (x, y), label in zip(boxes, labels_below):
        ax.text(x, y + 1.0, label, ha='center', va='center', fontsize=8, color='#555555')

    ax.set_title('RiskPilot: Four-Stage Closed-Loop Pipeline', fontsize=14, fontweight='bold', pad=15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'pipeline.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'pipeline.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Figure 1] pipeline.pdf / pipeline.png')


# ============================================================
# Figure 2: 风险聚类特征对比 (度数 + 密度 + 节点数)
# ============================================================
def fig2_cluster_comparison():
    """风险聚类对比: 平均度数 vs 局部密度 (气泡大小 = 节点数)"""
    patterns = insights['risk_patterns']

    cluster_ids = [p['cluster_id'] for p in patterns]
    avg_degrees = [p['degree_stats']['mean'] for p in patterns]
    densities = [p['subgraph_stats']['local_density'] for p in patterns]
    num_nodes = [p['num_nodes'] for p in patterns]

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#DD8452']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i, (deg, den, n, cid) in enumerate(zip(avg_degrees, densities, num_nodes, cluster_ids)):
        ax.scatter(deg, den, s=n * 3 + 50, c=colors[i], alpha=0.75, edgecolors='white', linewidth=1.5, zorder=3)
        ax.annotate(f'C{cid}\n({n} nodes)', (deg, den), textcoords='offset points',
                    xytext=(10, 5), fontsize=9, fontweight='bold', color=colors[i])

    ax.set_xlabel('Average Degree', fontsize=12)
    ax.set_ylabel('Local Subgraph Density', fontsize=12)
    ax.set_title('Risk Pattern Clusters: Degree vs Density\n(Bubble size ∝ cluster size)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(400, 1300)
    ax.set_ylim(0.3, 1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cluster_comparison.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'cluster_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Figure 2] cluster_comparison.pdf / cluster_comparison.png')


# ============================================================
# Figure 3: 各聚类度数分布 (箱线图)
# ============================================================
def fig3_degree_boxplot():
    """各聚类的度数统计对比 (使用 mean ± std 近似)"""
    patterns = insights['risk_patterns']

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    cluster_labels = []
    means = []
    stds = []
    mins = []
    maxs = []

    for p in patterns:
        cid = p['cluster_id']
        d = p['degree_stats']
        cluster_labels.append(f"Cluster {cid}\n({p['num_nodes']} nodes)")
        means.append(d['mean'])
        stds.append(d['std'])
        mins.append(d['min'])
        maxs.append(d['max'])

    x = np.arange(len(cluster_labels))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#DD8452']

    # 用误差棒表示 mean ± std, 标注 min/max
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=1.5)

    for i, (mn, mx) in enumerate(zip(mins, maxs)):
        ax.plot([i, i], [mn, mx], 'k-', linewidth=1, alpha=0.5)
        ax.plot(i, mn, 'v', color='black', markersize=5, alpha=0.5)
        ax.plot(i, mx, '^', color='black', markersize=5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels, fontsize=10)
    ax.set_ylabel('Node Degree', fontsize=12)
    ax.set_title('Degree Distribution by Risk Cluster\n(Bar = Mean ± Std, Arrows = Min/Max)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'degree_boxplot.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'degree_boxplot.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Figure 3] degree_boxplot.pdf / degree_boxplot.png')


# ============================================================
# Figure 4: 沙盒回测结果对比 (Recall vs FPR)
# ============================================================
def fig4_eval_results():
    """规则回测结果: Recall / Precision / FPR 对比"""
    results = eval_report['individual_results']
    combined = eval_report['combined_result']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左图: 各规则的 Recall / Precision / FPR ---
    ax = axes[0]
    # 英文名避免中文字体渲染问题
    name_map = {'高度数异常交易检测': 'High-Degree Anomaly', '孤立高风险账户检测': 'Isolated High-Risk'}
    rule_names = [name_map.get(r['name'], r['name']) for r in results]
    recalls = [r['recall'] * 100 for r in results]
    precisions = [r['precision'] * 100 for r in results]
    fprs = [r['fpr'] * 100 for r in results]

    x = np.arange(len(rule_names))
    width = 0.25

    ax.bar(x - width, recalls, width, label='Recall (%)', color='#4C72B0', alpha=0.85)
    ax.bar(x, precisions, width, label='Precision (%)', color='#55A868', alpha=0.85)
    ax.bar(x + width, fprs, width, label='FPR (%)', color='#C44E52', alpha=0.85)

    # 合格线
    ax.axhline(y=50, color='#4C72B0', linestyle='--', alpha=0.5, label='Min Recall (50%)')
    ax.axhline(y=5, color='#C44E52', linestyle='--', alpha=0.5, label='Max FPR (5%)')

    ax.set_xticks(x)
    ax.set_xticklabels(rule_names, fontsize=9)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Individual Rule Evaluation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # --- 右图: 混淆矩阵热力图 (规则 1) ---
    ax2 = axes[1]
    r = results[0]  # 第一条规则
    cm = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
    im = ax2.imshow(cm, cmap='Blues', alpha=0.8)

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Pred: Normal', 'Pred: Anomaly'], fontsize=10)
    ax2.set_yticklabels(['True: Normal', 'True: Anomaly'], fontsize=10)
    ax2.set_title(f'Confusion Matrix: {name_map.get(r["name"], r["name"])}', fontsize=12, fontweight='bold')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax2.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                     fontsize=14, fontweight='bold', color=color)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'eval_results.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'eval_results.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Figure 4] eval_results.pdf / eval_results.png')


# ============================================================
# Figure 5: 聚类节点数分布 (饼图)
# ============================================================
def fig5_cluster_pie():
    """风险聚类节点数占比饼图"""
    patterns = insights['risk_patterns']

    labels = [f"Cluster {p['cluster_id']}" for p in patterns]
    sizes = [p['num_nodes'] for p in patterns]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#DD8452']

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75, textprops={'fontsize': 11}
    )
    for t in autotexts:
        t.set_fontweight('bold')
        t.set_fontsize(10)

    # 中心圆 (甜甜圈效果)
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'Total\n{sum(sizes)}', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#333333')

    ax.set_title('Anomalous Node Distribution by Risk Cluster', fontsize=13, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cluster_pie.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'cluster_pie.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Figure 5] cluster_pie.pdf / cluster_pie.png')


# ============================================================
# Figure 6: 聚类特征雷达图
# ============================================================
def fig6_cluster_radar():
    """各聚类的多维特征雷达图对比"""
    patterns = insights['risk_patterns']

    categories = ['Avg Degree\n(÷1200)', 'Density', 'Cluster Size\n(÷300)', 'Anomaly\nScore']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#DD8452']

    for i, p in enumerate(patterns):
        values = [
            p['degree_stats']['mean'] / 1200,
            p['subgraph_stats']['local_density'],
            p['num_nodes'] / 300,
            p['avg_anomaly_score'],
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=f"Cluster {p['cluster_id']}", alpha=0.8)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Risk Cluster Feature Radar', fontsize=13, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cluster_radar.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'cluster_radar.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Figure 6] cluster_radar.pdf / cluster_radar.png')


# ============================================================
# 主函数: 一键生成所有图表
# ============================================================
if __name__ == '__main__':
    print('=' * 50)
    print('  Generating RiskPilot figures...')
    print('=' * 50)

    fig1_pipeline()
    fig2_cluster_comparison()
    fig3_degree_boxplot()
    fig4_eval_results()
    fig5_cluster_pie()
    fig6_cluster_radar()

    print('=' * 50)
    print(f'  All figures saved to: {OUTPUT_DIR}')
    print('=' * 50)
