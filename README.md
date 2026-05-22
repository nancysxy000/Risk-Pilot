# RiskPilot — 金融风控规则自主进化引擎

## 项目概览

RiskPilot 是一个金融风控规则的自主进化引擎，通过 **GNN 感知 → LLM 策略生成 → 沙盒回测 → 规则沉淀** 的闭环，让风控体系越用越聪明。

```
┌─────────────────────────────────────────────────────────────────┐
│                      RiskPilot Pipeline                        │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Stage 1  │───>│ Stage 2  │───>│ Stage 3  │───>│ Stage 4  │  │
│  │ GNN 感知 │    │ LLM 策略 │    │ 沙盒验证 │    │ 规则沉淀 │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                │               │               │        │
│       ▼                ▼               ▼               ▼        │
│  异常检测+        规则自动生成     回测评估指标     知识库RAG     │
│  风险洞察提取     /优化           精度/召回/误杀   反馈下一轮     │
└─────────────────────────────────────────────────────────────────┘
```

## 分工架构

| 层       | 负责人 | 核心任务                                       |
|----------|--------|------------------------------------------------|
| GNN 感知层 | Nancy  | 图异常检测 (BWGNN)、风险特征提取、结构化洞察输出 |
| LLM 策略层 | Yifei  | LLM Agent 规则生成/优化、Prompt 工程            |
| 沙盒验证层 | 共同   | 规则回测、指标评估 (召回率/误杀率/覆盖范围)      |
| 规则知识库 | 共同   | 历史规则存储、RAG 检索、闭环反馈                 |

---

## Pipeline 详细设计

### Stage 1: GNN 感知层 — 图异常检测与风险洞察 (Nancy)

**目标**: 在交易图上识别异常节点，并提取结构化的风险洞察（供 LLM 使用）。

**技术栈**: BWGNN (Beta Wavelet GNN) + DGL + PyTorch

**流程**:
```
原始交易图 (tfinance/yelp/amazon)
    │
    ▼
[1.1] 数据预处理 & 图构建
    │   - 加载 tfinance/yelp/amazon 数据集
    │   - 特征标准化、类别不平衡处理
    │   - 图结构分析（度分布、连通性）
    ▼
[1.2] BWGNN 异常检测模型训练
    │   - Beta 小波变换: 多尺度频谱分析
    │   - 捕获异常节点的"频谱差异"
    │   - 输出: 每个节点的异常概率 + 嵌入向量
    ▼
[1.3] 风险洞察提取 (本项目关键创新)
    │   - 异常子图提取: 高风险节点的k-hop邻域
    │   - 异常模式聚类: 对异常嵌入做聚类，发现不同风险类型
    │   - 结构化输出: JSON格式的风险描述
    ▼
输出: risk_insights.json
    {
        "cluster_id": 0,
        "num_nodes": 298,
        "degree_stats": {"mean": 201.3, "max": 2577, "min": 9},
        "salient_feature_dims": [7, 12, 13, 15, 16, 17, 18, 19],
        "feature_z_scores": {"7": 2.16, "12": 2.13, ...},
        "risk_description": "Risk Pattern #0: 298 anomalous nodes..."
    }
```

**关键输出文件**:
- `outputs/gnn_predictions.pt` — 节点级异常分数
- `outputs/node_embeddings.pt` — 节点嵌入向量
- `outputs/risk_insights.json` — 结构化风险洞察 (传递给 LLM)

---

### Stage 2: LLM 策略层 — 规则自动生成/优化 (Yifei)

**目标**: 基于 GNN 感知到的风险洞察 + 历史规则库，利用 LLM 自动生成/优化风控规则。

**技术栈**: 阿里云 DashScope (qwen-plus) / OpenAI API + RAG

**流程**:
```
risk_insights.json + 规则知识库 + 数据分布统计
    │
    ▼
[2.1] RAG 检索: 从知识库中检索相关历史规则
    │
    ▼
[2.2] Prompt 构造: 组装风险洞察 + 历史规则 + 数据分布 + 约束条件
    │   System Prompt: "你是金融风控专家..."
    │   Context: {risk_insights} + {data_distribution} + {similar_rules} + {constraints}
    │   Chain-of-Thought 引导: 分布对比 → 分界点识别 → 阈值设定
    │
    ▼
[2.3] LLM 规则生成
    │   输出结构化规则 JSON:
    │   {
    │       "rule_id": "R-20260514-001",
    │       "name": "高连接度欺诈环检测",
    │       "conditions": [
    │           {"field": "gnn_anomaly_score", "operator": ">=", "value": 0.68},
    │           {"field": "feature_dim_15_zscore", "operator": "<=", "value": -1.95}
    │       ],
    │       "logic": "AND",
    │       "action": "review",
    │       "severity": "high",
    │       "explanation": "基于GNN发现的Cluster-0模式..."
    │   }
    ▼
[2.4] 阈值微调搜索: 对 LLM 生成的规则做贪心 ±30% 搜索，自动微调阈值
    │   评分: F1 - max(0, FPR - 0.05) * 0.5
    │
    ▼
[2.5] 规则自优化: 基于定向反馈迭代优化
    │   - 为每条规则生成针对性诊断 (Recall/FPR/Precision 具体差距)
    │   - 约束 LLM 小幅调整 (阈值变动 <= 20%)
    │
    ▼
输出: generated_rules.json
```

**规则支持的字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `node_degree` | float | 节点度数，即该节点的连接数/交易对手数 |
| `node_degree_zscore` | float | 节点度数的标准分数，>2 表示远高于平均水平 |
| `feature_dim_X` | float | 第 X 维原始特征值 (X 从 0 开始) |
| `feature_dim_X_zscore` | float | 第 X 维特征的标准分数，\|z\|>1.5 表示显著偏离全局均值 |
| `gnn_anomaly_score` | float | **GNN 输出的异常概率**，范围 [0, 1]，>0.5 表示可能异常，>0.8 表示高置信异常 |
| `embedding_cluster_id` | int | 嵌入聚类 ID，-1 表示非异常节点，0~4 表示不同风险模式聚类 |

---

### Stage 3: 沙盒验证层 — 规则回测与评估 (共同)

**目标**: 在标注数据集上回测 LLM 生成的规则，量化评估规则质量。

**流程**:
```
generated_rules.json + 标注数据集 + GNN 输出
    │
    ▼
[3.1] 规则解析: 将 JSON 规则转为可执行的 Python 函数
    │   - 支持字段: node_degree, node_degree_zscore, feature_dim_X,
    │     feature_dim_X_zscore, gnn_anomaly_score, embedding_cluster_id
    │   - 支持操作符: >, <, >=, <=, ==, !=
    │   - 支持逻辑: AND / OR
    │
    ▼
[3.2] 预计算派生特征
    │   - feature_zscores: 各特征维度的 z-score
    │   - degree_zscore: 度数的 z-score
    │   - gnn_anomaly_scores: GNN 输出的异常概率 (probs[:, 1])
    │   - embedding_cluster_ids: KMeans 聚类全量嵌入，异常节点标记聚类 ID
    │
    ▼
[3.3] 规则执行 & 指标计算
    │   - Recall (召回率): 能抓到多少真正的异常?
    │   - Precision (精确率): 抓到的里面有多少是对的?
    │   - FPR (误杀率): 误伤了多少正常用户?
    │   - F1: Precision 和 Recall 的调和均值
    ▼
[3.4] 规则筛选: 通过阈值筛选高质量规则
    │   - F1 >= 0.60 且 Recall >= 0.50 且 FPR <= 0.05 且 Precision >= 0.40 → 通过
    │   - 否则 → 反馈给 LLM 重新优化 (闭环迭代)
    ▼
[3.5] 闭环迭代 (含稳定性保障)
    │   - all_report_results 累积器: 跨轮次累积评估结果，不覆盖
    │   - 历史最优追踪: best_rules / best_qualified_count / best_combined_f1
    │   - Early stop: 连续 2 轮无改善则停止，回退到历史最优
    │
    ▼
输出: evaluation_report.json, qualified_rules.json
```

---

### Stage 4: 规则知识库 — RAG 沉淀与闭环 (共同)

**目标**: 将验证通过的优质规则存入知识库，供下一轮 LLM 检索使用，形成闭环。

**流程**:
```
qualified_rules.json
    │
    ▼
[4.1] 规则向量化: 使用 Embedding 模型对规则描述编码
    │
    ▼
[4.2] 存入向量数据库 (ChromaDB / FAISS)
    │
    ▼
[4.3] 知识库更新: 维护规则版本、有效期、性能指标
    │
    ▼
[4.4] 闭环反馈: 新一轮 GNN 检测 → RAG 检索历史规则 → LLM 增量优化
    │
    ▼
规则知识库 (持续积累)
```

---

## 项目结构

```
Risk-Pilot-main/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包
├── .env                               # API 密钥配置 (不纳入版本控制)
├── configs/
│   ├── default.yaml                   # 全局配置 (tfinance 基线)
│   ├── baseline_yelp.yaml             # Yelp 基线配置
│   ├── baseline_amazon.yaml           # Amazon 基线配置
│   ├── ablation_yelp_no_gnn.yaml      # Yelp 消融配置 (无 GNN 字段)
│   ├── ablation_amazon_no_gnn.yaml    # Amazon 消融配置 (无 GNN 字段)
│   └── ablation_no_gnn.yaml           # tfinance 消融配置 (无 GNN 字段)
├── dataset/
│   ├── tfinance                       # T-Finance 数据集 (DGL 二进制)
│   └── tsocial                        # T-Social 数据集
│
├── src/
│   ├── __init__.py
│   │
│   ├── stage1_gnn/                    # Stage 1: GNN 感知层 (Nancy)
│   │   ├── __init__.py
│   │   ├── bwgnn_model.py             # BWGNN 模型定义
│   │   ├── dataset_loader.py          # 数据加载与预处理
│   │   ├── train.py                   # 模型训练
│   │   ├── insight_extractor.py       # 风险洞察提取 (核心创新)
│   │   └── utils.py                   # 工具函数
│   │
│   ├── stage2_llm/                    # Stage 2: LLM 策略层 (Yifei)
│   │   ├── __init__.py
│   │   ├── rule_generator.py          # LLM 规则生成 (含阈值搜索 + 定向反馈 + 消融支持)
│   │   ├── prompt_templates.py        # Prompt 模板 (含 CoT + GNN 字段 + 消融变体)
│   │   └── rag_retriever.py           # RAG 检索模块
│   │
│   ├── stage3_sandbox/                # Stage 3: 沙盒验证层 (共同)
│   │   ├── __init__.py
│   │   ├── rule_parser.py             # 规则解析器 (支持 GNN 字段)
│   │   └── evaluator.py               # 指标评估 (注入 GNN 输出)
│   │
│   ├── stage4_kb/                     # Stage 4: 规则知识库 (共同)
│   │   ├── __init__.py
│   │   ├── knowledge_base.py          # 知识库管理
│   │   └── rule_schema.py             # 规则数据模型
│   │
│   └── pipeline.py                    # 主 Pipeline 编排 (含闭环迭代 + 消融支持)
│
├── src/baselines/                    # 传统方法 Baseline 对比
│   ├── __init__.py
│   ├── traditional.py                # IF / LOF / OCSVM / PCA+Mahalanobis 对比实验
│   └── statistical_rules.py          # 统计规则自动生成 (不使用 LLM)
│
├── outputs/                           # 运行输出 (默认 tfinance)
├── outputs_yelp/                      # Yelp 基线输出
├── outputs_amazon/                    # Amazon 基线输出
├── outputs_yelp_ablation/             # Yelp 消融实验输出
├── outputs_amazon_ablation/           # Amazon 消融实验输出
├── outputs_ablation_no_gnn/           # T-Finance 消融实验输出
├── outputs_baselines/                 # 传统方法 Baseline 输出
│
├── scripts/
│   └── run_baselines.py               # Baseline 对比实验脚本 (传统方法/统计规则/无迭代)
├── notebooks/                         # 实验 Notebook
└── tests/
```

---

## 快速开始

### Step 1: 创建 Conda 环境

```bash
conda create -n riskpilot python=3.11 -y
conda activate riskpilot
```

> **注意**: DGL 对 PyTorch 版本极其敏感。当前配置已在 **Python 3.11 + torch 2.2.1 + dgl 2.1.0** 上验证通过。请严格使用 `requirements.txt` 中的版本。

### Step 2: 安装依赖

```bash
pip install -r requirements.txt --index-url https://pypi.org/simple/
```

如果需要 LLM 规则生成 (Stage 2)，还需安装:

```bash
pip install openai python-dotenv
```

### Step 3: 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

`.env` 配置示例 (阿里云 DashScope):

```bash
# ===== LLM API (Stage 2: 规则生成) =====
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

支持的 LLM 服务商 (均通过 OpenAI 兼容接口接入):
- **阿里云 DashScope**: `base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`
- **OpenAI 官方**: `base_url` 留空或设为 `https://api.openai.com/v1`
- **Azure OpenAI**: `base_url=https://your-endpoint.openai.azure.com/`
- **本地 Ollama**: `base_url=http://localhost:11434/v1`

> **不配置 API 也能运行**: Stage 2 会自动使用 Mock 规则走完全流程，Stage 1 (GNN) 和 Stage 3 (沙盒回测) 均为真实计算。

### Step 4: 运行

```bash
# 完整 Pipeline — 使用默认配置 (tfinance)
python -m src.pipeline

# 指定数据集
python -m src.pipeline --dataset yelp --mode full
python -m src.pipeline --dataset amazon --mode full

# 使用自定义配置文件
python -m src.pipeline --config configs/baseline_yelp.yaml

# 仅运行 GNN 感知层 (不需要 API)
python -m src.pipeline --dataset tfinance --mode gnn_only

# 消融实验 (禁用 GNN 字段)
python -m src.pipeline --config configs/ablation_amazon_no_gnn.yaml
```

---

## 数据集

| 数据集     | 节点数    | 边数         | 异常比例 | 特征维度 | 来源          |
|-----------|----------|-------------|---------|---------|--------------|
| T-Finance | 39,357   | 42,445,086  | 4.58%   | 10      | ICML 2022    |
| T-Social  | 5,781,065| 73,105,508  | 3.01%   | 2       | ICML 2022    |
| Yelp      | 45,954   | 8,097,302   | 14.53%  | 32      | DGL Built-in |
| Amazon    | 11,944   | 9,569,592   | 6.87%   | 25      | DGL Built-in |

### 各数据集运行方案

#### T-Finance (推荐首选)
```bash
python -m src.pipeline --dataset tfinance --mode full
```
- **数据准备**: 从 [Google Drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr) 下载 `tfinance` 文件，放入 `dataset/` 目录
- **算力要求**: Mac CPU 约 22 分钟 (100 epochs)，内存 >= 2GB
- **适用场景**: 开发调试、论文实验、完整 Pipeline 演示

#### Yelp (欺诈评论检测)
```bash
python -m src.pipeline --dataset yelp --mode full
```
- **数据准备**: 首次运行自动下载 (~150MB)，后续使用缓存 (`~/.dgl/`)
- **算力要求**: Mac CPU 约 5 分钟，内存 >= 2GB

#### Amazon (虚假评论检测)
```bash
python -m src.pipeline --dataset amazon --mode full
```
- **数据准备**: 首次运行自动下载 (~30MB)
- **算力要求**: Mac CPU 约 5 分钟，内存 >= 1GB

#### T-Social (大规模社交网络)
```bash
python -m src.pipeline --dataset tsocial --mode full
```
- **数据准备**: 从 [Google Drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr) 下载 `tsocial` 文件
- **算力要求**: 内存 >= 16GB，CPU 训练约 30~60 分钟
- **注意**: 570 万节点，Mac 8GB 内存可能不够，建议在 16GB+ 机器上运行

---

## 实验结果

### Stage 1: GNN 感知层 — 跨数据集对比

BWGNN 在三个数据集上训练 100 epochs 的结果:

| 数据集 | 节点数 | 异常比例 | Macro-F1 | AUC | 训练时长 |
|--------|--------|---------|----------|-----|---------|
| **T-Finance** | 39,357 | 4.58% | **88.89%** | **95.02%** | ~22 min |
| **Yelp** | 45,954 | 14.53% | 70.69% | 83.10% | ~4 min |
| **Amazon** | 11,944 | 6.87% | **92.36%** | **96.49%** | ~5 min |

**关键发现**: Amazon 的 GNN 检测效果最好 (F1=92.36%)，Yelp 最弱 (F1=70.69%)。GNN 质量直接决定了下游规则的上限。

### Stage 2+3: 规则生成与回测 — 跨数据集结果

#### T-Finance (GNN F1=88.89%)

GNN 从 39,357 个节点中检测出 **500 个高风险节点**，聚类为 **5 种风险模式**:

| 聚类 ID | 节点数 | 平均度数 | 局部密度 | 风险描述 |
|---------|--------|---------|---------|---------|
| Cluster 0 | 254 | 522 | 0.414 | 中大规模欺诈环，高度连接 |
| Cluster 1 | 29 | 1,194 | 0.936 | 极小团伙，极高内聚性 |
| Cluster 2 | 159 | 885 | 0.844 | 中等规模高密度欺诈环 |
| Cluster 3 | 46 | 936 | 0.847 | 紧密小型欺诈团伙 |
| Cluster 4 | 12 | 1,043 | 0.500 | 高连接 + 特征异常 |

闭环迭代优化过程:

| 迭代轮次 | 合格规则数 | 综合 Recall | 综合 Precision | 综合 FPR | 状态 |
|---------|-----------|------------|---------------|---------|------|
| 初始 | 0/5 | 44.0% | 78.1% | 0.59% | 起点 |
| 第 1 轮 | 1/5 | 57.7% | 68.2% | 1.29% | 改善 |
| 第 2 轮 | 2/4 | 81.6% | 70.2% | 1.66% | 改善 |
| 第 3 轮 | 0/4 | 31.4% | 72.3% | 0.58% | 退化 |
| 第 4 轮 | 3/5 | **81.5%** | **71.0%** | **1.60%** | 恢复最优 |

**最终**: 从 23 条规则中筛选出 **12 条合格规则**沉淀至知识库。

#### Amazon (GNN F1=92.36%)

GNN 检测到 500 个高风险节点，聚类为 5 种风险模式。

闭环迭代优化过程:

| 迭代轮次 | 合格规则数 | 综合 Recall | 综合 Precision | 综合 FPR | 状态 |
|---------|-----------|------------|---------------|---------|------|
| 初始 | 0/3 | 48.4% | 91.5% | 0.33% | 起点 |
| 第 1 轮 | 0/3 | 54.6% | 95.1% | 0.21% | 改善 |
| 第 2 轮 | 1/3 | 67.1% | 94.8% | 0.27% | 改善 |
| 第 3 轮 | 1/6 | 77.6% | 69.1% | 2.56% | 无改善 |
| 第 4 轮 | 1/5 | **79.8%** | **88.8%** | **0.75%** | 改善 |

**最终**: 沉淀 **6 条合格规则**至知识库。

#### Yelp (GNN F1=70.69%)

GNN 检测到 500 个高风险节点，聚类为 5 种风险模式。

闭环迭代优化过程:

| 迭代轮次 | 合格规则数 | 综合 Recall | 综合 Precision | 综合 FPR | 状态 |
|---------|-----------|------------|---------------|---------|------|
| 初始 | 0/5 | 20.0% | 54.7% | 2.81% | 起点 |
| 第 1 轮 | 0/5 | 23.8% | 40.4% | 5.96% | 改善 |
| 第 2 轮 | 0/10 | 23.9% | 39.8% | 6.14% | 无改善 |
| 第 3 轮 | 0/15 | 24.6% | 39.3% | 6.48% | 改善 |
| 第 4 轮 | 0/20 | **24.6%** | **39.5%** | **6.42%** | 改善 |

**结果**: **0 条合格规则**。Yelp 的 GNN 模型偏弱 (F1=70.7%)，导致规则质量不足以达标。

### 跨数据集结果汇总

| 数据集 | GNN Macro-F1 | GNN AUC | 规则 Recall | 规则 Precision | 规则 FPR | 合格规则 |
|--------|-------------|---------|------------|---------------|---------|---------|
| **T-Finance** | 88.89% | 95.02% | **81.5%** | 71.0% | 1.60% | **12** |
| **Amazon** | 92.36% | 96.49% | **79.8%** | **88.8%** | **0.75%** | **6** |
| Yelp | 70.69% | 83.10% | 24.6% | 39.5% | 6.42% | 0 |

**关键发现**:
- GNN F1 越高 → 合格规则越多。Amazon (92.4%) 和 T-Finance (88.9%) 产生了大量合格规则，而 Yelp (70.7%) 完全失败
- Pipeline 的上限被 GNN 模型质量锁死

---

## 传统方法 Baseline 对比

为了验证 BWGNN 的优越性，我们在三个数据集上对比了 4 种传统异常检测方法。所有方法使用相同的数据划分 (40% train / 20% val / 40% test, stratified, random_state=42)、相同的特征输入 (原始特征 + degree) 和相同的评估指标。

### 对比方法

| 方法 | 类型 | 说明 |
|------|------|------|
| Isolation Forest | 无监督集成 | 基于随机划分的异常隔离 |
| LOF | 无监督密度 | 局部密度偏差检测 |
| One-Class SVM | 半监督边界 | 只用正常样本训练 |
| PCA + Mahalanobis | 无监督统计 | 降维后马氏距离 |

### T-Finance 对比

| 方法 | Recall | Precision | Macro-F1 | AUC |
|------|--------|-----------|----------|-----|
| Isolation Forest | 0.14% | 0.13% | 47.55% | 35.06% |
| LOF | 2.48% | 2.27% | 48.73% | 41.57% |
| One-Class SVM | 32.41% | 9.90% | 52.98% | 72.86% |
| PCA + Mahalanobis | 0.14% | 0.13% | 47.55% | 41.80% |
| **BWGNN (ours)** | **76.00%** | **88.16%** | **90.41%** | **96.04%** |

### Yelp 对比

| 方法 | Recall | Precision | Macro-F1 | AUC |
|------|--------|-----------|----------|-----|
| Isolation Forest | 28.73% | 27.82% | 57.92% | 63.25% |
| LOF | 26.53% | 19.27% | 53.06% | 56.55% |
| One-Class SVM | 30.22% | 29.27% | 58.78% | 61.77% |
| PCA + Mahalanobis | 29.92% | 28.98% | 58.61% | 64.15% |
| **BWGNN (ours)** | **56.32%** | **66.97%** | **70.69%** | **83.10%** |

### Amazon 对比

| 方法 | Recall | Precision | Macro-F1 | AUC |
|------|--------|-----------|----------|-----|
| Isolation Forest | 74.24% | 28.19% | 64.27% | 82.85% |
| LOF | 33.33% | 31.61% | 62.57% | 66.51% |
| One-Class SVM | 81.21% | 38.56% | 72.02% | 88.42% |
| PCA + Mahalanobis | 82.73% | 26.20% | 62.49% | 82.13% |
| **BWGNN (ours)** | **84.56%** | **88.17%** | **92.36%** | **96.49%** |

### 汇总

| 数据集 | 最佳传统方法 F1 | BWGNN F1 | 领先幅度 |
|--------|---------------|----------|---------|
| T-Finance | 52.98% (OCSVM) | **90.41%** | **+37.4%** |
| Yelp | 58.78% (OCSVM) | **70.69%** | **+11.9%** |
| Amazon | 72.02% (OCSVM) | **92.36%** | **+20.3%** |

**关键发现**:
- BWGNN 在所有数据集上都大幅领先，F1 平均高出 **23.2%**
- 传统方法 Precision 普遍很低 (Amazon 上 IF 只有 28%)，大量误杀正常用户
- GNN 利用图结构信息，能有效区分结构上相似的异常和正常节点
- 运行脚本: `python scripts/run_baselines.py --all`

---

## 统计规则 vs LLM 规则生成

为了验证 **LLM Agent 在规则生成中的不可替代性**，我们对比了不用 LLM 的纯统计规则生成方法: 对每个 risk_pattern，直接将其显著特征维度的 z-score 机械拼接为 AND 条件，然后仍然经过 threshold_search 微调。

### 方法说明

| 方法 | 规则生成方式 | 迭代优化 |
|------|-----------|---------|
| **统计规则** | 按 z-score 机械拼接所有显著特征维度 | threshold_search |
| **LLM 规则 (基线)** | LLM 理解风险模式后智能选择条件 | threshold_search + 闭环迭代 |

### T-Finance 统计规则

| 指标 | 统计规则 | LLM 规则 (基线) | 变化 |
|------|---------|---------------|------|
| Recall | 61.31% | **69.85%** | -8.54% |
| Precision | 85.80% | 76.04% | +9.76% |
| FPR | 0.49% | 3.50% | -3.01% |
| 合格规则 | 2 | **12** | **-10** |

### Yelp 统计规则

| 指标 | 统计规则 | LLM 规则 (基线) | 变化 |
|------|---------|---------------|------|
| Recall | 17.96% | **24.62%** | -6.66% |
| Precision | 46.82% | 39.45% | +7.37% |
| FPR | 3.47% | 6.42% | -2.95% |
| 合格规则 | 0 | 0 | - |

### Amazon 统计规则

| 指标 | 统计规则 | LLM 规则 (基线) | 变化 |
|------|---------|---------------|------|
| Recall | 33.13% | **79.78%** | **-46.65%** |
| Precision | 96.11% | 88.75% | +7.36% |
| FPR | 0.10% | 0.75% | -0.65% |
| 合格规则 | 0 | **6** | **-6** |

### 统计规则实验结论

1. **LLM 规则的 Recall 显著高于统计规则**: Amazon 上差距最大 (-46.65%)，因为统计规则机械拼接所有显著特征导致条件过严
2. **统计规则过于保守**: Precision 很高 (Amazon 96%) 但代价是漏掉大量异常 (Recall 仅 33%)
3. **LLM 的核心优势是条件选择**: LLM 像风控专家一样选择最有效的条件组合，而不是无脑拼接所有显著特征
4. **合格规则数量差异巨大**: T-Finance 上统计规则只有 2 条合格 vs LLM 的 12 条

---

## LLM Agent 价值总结

综合传统方法、统计规则、消融实验三组对比，完整证明 RiskPilot 各组件的增量价值:

### 对比矩阵

| 方法 | T-Finance Recall | Amazon Recall | Amazon 合格规则 |
|------|-----------------|---------------|---------------|
| 传统方法 (最佳) | 32.41% | 81.21% | — |
| 统计规则 (无 LLM) | 61.31% | 33.13% | 0 |
| LLM 无 GNN 字段 | 0.28% | 49.21% | 0 |
| **RiskPilot (完整)** | **69.85%** | **79.78%** | **6** |

### 各组件贡献

| 组件 | 消除后影响 | 结论 |
|------|----------|------|
| **GNN 模型** | 传统方法 F1 -23%~37% | 图结构信息是检测的基础 |
| **GNN 字段注入规则** | Recall -70% (tfinance) | 规则依赖 GNN 分数作为核心过滤条件 |
| **LLM 规则生成** | 合格规则 12→2 (tfinance) | LLM 的智能条件选择不可替代 |
| **闭环迭代** | 待补充 | 迭代优化的增量价值 |

---

## 消融实验

为了验证 `gnn_anomaly_score` 字段在规则中的实际贡献，我们在 T-Finance、Yelp 和 Amazon 三个数据集上进行了消融实验: 去掉 GNN 字段，只允许 LLM 使用 `node_degree`、`node_degree_zscore`、`feature_dim_X`、`feature_dim_X_zscore` 生成规则。

### 消融实验设置

- **基线 (Baseline)**: 规则可使用全部 6 个字段 (含 gnn_anomaly_score, embedding_cluster_id)
- **消融 (No GNN Fields)**: 规则只能使用 4 个图统计字段 (degree + feature)
- **方法**: 通过 `configs/ablation_*_no_gnn.yaml` 配置文件控制，pipeline 自动选择对应的 Prompt 模板并禁用 GNN 输出注入

### T-Finance 消融结果

| 指标 | 基线 (含 GNN 字段) | 消融 (无 GNN 字段) | 变化 |
|------|-------------------|-------------------|------|
| Combined Recall | **69.85%** | 0.28% | **-69.57%** |
| Combined Precision | **76.04%** | 0.35% | **-75.69%** |
| FPR | 3.50% | 3.74% | +0.24% |
| 合格规则数 | **12** | **0** | **-12** |

**解读**: T-Finance 的消融效果最为极端 — Recall 从 70% 断崖式跌至 0.28%，说明纯特征阈值规则在 10 维特征上几乎完全失效。GNN 分数是规则有效性的唯一支柱。

### Yelp 消融结果

| 指标 | 基线 (含 GNN 字段) | 消融 (无 GNN 字段) | 变化 |
|------|-------------------|-------------------|------|
| Combined Recall | 24.62% | 28.07% | +3.5% |
| Combined Precision | 39.45% | 22.16% | **-17.3%** |
| FPR | 6.42% | 16.76% | **+10.3%** |
| 合格规则数 | 0 | 0 | - |

### Amazon 消融结果

| 指标 | 基线 (含 GNN 字段) | 消融 (无 GNN 字段) | 变化 |
|------|-------------------|-------------------|------|
| Combined Recall | **79.78%** | 49.21% | **-30.6%** |
| Combined Precision | **88.75%** | 42.80% | **-46.0%** |
| FPR | **0.75%** | 4.85% | +4.1% |
| 合格规则数 | **6** | **0** | **-6** |

### 消融实验结论

1. **`gnn_anomaly_score` 是规则质量的核心支柱**: 三个数据集上一致验证 — T-Finance Recall -69.6%，Amazon Recall -30.6%，合格规则全部归零
2. **GNN 字段的核心作用是精确过滤**: Yelp 消融中 Recall 反而微升 (+3.5%)，但 Precision 暴跌 (-17.3%)，FPR 飙升 (+10.3%)。没有 GNN 分数作为过滤条件，规则会放宽到误杀大量正常节点
3. **规则本质上是 GNN 判断的可解释包装**: 所有合格规则的第一条件都是 `gnn_anomaly_score >= 0.68`，然后加 degree/feature 条件细分风险类型。规则继承了 GNN 的检测能力，并增加了业务可解释性
4. **特征维度越低越依赖 GNN**: T-Finance (10 维) 消融后 Recall 仅 0.28%，Amazon (25 维) 还有 49.2%，说明低维特征本身不足以区分异常

---

## 核心优化总结

### 优化 1: GNN 嵌入驱动规则

将 GNN 输出的异常概率 (`gnn_anomaly_score`) 和嵌入聚类 ID (`embedding_cluster_id`) 作为规则可用字段。

- 修改 `rule_parser.py`: 新增两个字段的解析逻辑
- 修改 `evaluator.py`: 注入 GNN 输出到评估上下文，含 KMeans 聚类
- 修改 `prompt_templates.py`: 更新可用字段表，设计原则改为优先使用 GNN 信号
- 修改 `rule_generator.py`: `threshold_search()` 支持 GNN 字段
- 修改 `pipeline.py`: 全链路传递 `gnn_outputs`

效果: Recall 从 2.4% 跃升至 44%+，因为规则可写 `gnn_anomaly_score >= 0.8` 直接继承 GNN 检测能力。

### 优化 2: Prompt 工程 (Chain-of-Thought)

重写 `prompt_templates.py`，引入结构化思考流程:

- **SYSTEM_PROMPT**: 增加思考流程 (分布对比 → 分界点识别 → 阈值设定)
- **RULE_GENERATION_PROMPT**: 4 步 CoT (理解模式 → 参考分布 → 分析分界点 → 生成规则)
- **RULE_OPTIMIZATION_PROMPT**: 定向反馈替代全量报告，增量约束 (阈值调整 <= 20%)

### 优化 3: 阈值微调搜索

新增 `threshold_search()` 方法 (`rule_generator.py`):

- 对 LLM 生成的每条规则，在 ±30% 范围内贪心搜索最优阈值
- 评分: `F1 - max(0, FPR - 0.05) * 0.5`
- 在 LLM 生成规则后、沙盒评估前自动执行，作为规则质量保底

### 优化 4: 闭环迭代稳定性

修改 `pipeline.py` 的迭代逻辑:

- **`all_report_results` 累积器**: 跨轮次累积所有评估结果，永不覆盖，防止丢失已合格规则
- **历史最优追踪**: `best_rules` / `best_qualified_count` / `best_combined_f1`
- **Early stop**: 连续 2 轮无改善则停止迭代
- **退化回退**: 如果迭代后整体变差，自动回退到历史最优规则集

### 优化 5: 跨数据集兼容性修复

解决 yelp (32 维特征) 和 amazon (25 维特征) 数据集上的 LLM 解析失败问题:

- **max_tokens 扩容**: 2000 → 4096，避免高维特征数据集的 LLM 输出截断
- **动态异常比例**: 移除硬编码的 tfinance 异常比例，改为 `{anomaly_ratio}` 占位符
- **LLM 输出解析增强**: 三层 JSON 解析 + 截断恢复 (找最后一个 `}` 并补 `]`)
- **Prompt 字段一致性**: 约束行与可用字段表保持同步

---

## 效果演进对比

| 版本 | Recall | Precision | FPR | 合格规则 | 关键改动 |
|------|--------|-----------|-----|---------|---------|
| 初始版本 | 2.4% | 0.75% | 15.6% | **0 条** | 仅 degree/feature 阈值规则 |
| 第二次优化 | **81.5%** | **71.0%** | **1.60%** | **12 条** | +GNN 嵌入字段 +CoT Prompt +阈值搜索 +迭代稳定性 |
| 跨数据集扩展 | 79.8% | 88.8% | 0.75% | **18 条** (TF+AM) | +Yelp/Amazon 兼容 +消融实验 |
| 消融实验验证 | — | — | — | 0 条 (无GNN) | 证明 GNN 字段是规则质量的核心 |

---

## 输出文件说明

| 文件 | 说明 | 生成阶段 |
|------|------|---------|
| `outputs/gnn_predictions.pt` | 每个节点的异常概率 `[N, 2]`，`[:, 1]` 为异常分数 | Stage 1 |
| `outputs/node_embeddings.pt` | 每个节点的 GNN 嵌入向量 `[N, hidden_dim]` | Stage 1 |
| `outputs/risk_insights.json` | 结构化风险洞察: 聚类 + 图结构特征 + 自然语言描述 | Stage 1 |
| `outputs/generated_rules.json` | LLM 生成的风控规则 (含阈值优化) | Stage 2 |
| `outputs/evaluation_report.json` | 沙盒回测指标: TP/FP/FN/TN, Recall, Precision, FPR | Stage 3 |
| `outputs/qualified_rules.json` | 通过回测的优质规则 (沉淀至知识库) | Stage 4 |
| `outputs/_llm_raw_response.txt` | LLM 原始响应 (调试用，含 token 统计) | Stage 2 |

---

## 评估指标

| 指标              | 含义                     | 合格标准   |
|-------------------|--------------------------|-----------|
| Macro-F1          | 整体分类平衡性            | > 0.70    |
| AUC-ROC           | 排序质量                  | > 0.85    |
| Recall (召回率)    | 异常节点被抓到的比例       | >= 0.50   |
| Precision (精确率) | 标记为异常中真正异常的比例 | >= 0.40   |
| FPR (误杀率)       | 正常用户被误伤的比例       | <= 0.05   |
| F1 (规则级)        | 单条规则的 F1            | >= 0.60   |

---

## 技术亮点 & 创新点

1. **GNN + LLM 协同**: 将图神经网络的结构化风险感知与 LLM 的规则生成能力结合，规则直接继承 GNN 的检测精度
2. **自主进化闭环**: GNN 感知 → LLM 生成 → 沙盒验证 → 知识库沉淀 → 反馈优化，规则越迭代越精准
3. **GNN 嵌入驱动规则**: 规则可直接使用 `gnn_anomaly_score` 字段，解决了简单阈值规则无法有效表达图结构模式的核心瓶颈
4. **消融实验验证**: 在三个数据集上通过禁用 GNN 字段的对照实验，量化证明了 `gnn_anomaly_score` 对规则质量的贡献 (T-Finance Recall -69.6%, Amazon Recall -30.6%, 三数据集合格规则全部归零)
5. **Chain-of-Thought Prompt**: 引导 LLM 先分析分布差异再设定阈值，避免随意出值
6. **定向反馈优化**: 不再传整份评估报告给 LLM，而是为每条规则生成具体诊断 (差距 + 方向 + 幅度)
7. **阈值微调搜索**: LLM 生成规则后自动贪心搜索最优阈值，作为规则质量保底
8. **迭代稳定性保障**: 累积器 + 历史最优追踪 + Early stop + 退化回退，防止闭环迭代反而变差
9. **跨数据集通用**: 支持 tfinance / yelp / amazon / tsocial 四个数据集，含数据集自适应的 Prompt 和解析逻辑
10. **多 LLM 后端支持**: 统一的 OpenAI 兼容接口，一键切换 DashScope / OpenAI / Ollama

---

## 开发环境

| 项目 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.2.1 |
| DGL | 2.1.0 |
| scikit-learn | 用于 KMeans 聚类 |
| LLM | 阿里云 DashScope qwen-plus |
| OS | macOS (Apple Silicon / Intel) |

> DGL 对 PyTorch 版本敏感，请严格使用上述版本组合。
