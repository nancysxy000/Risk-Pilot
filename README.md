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
原始交易图 (tfinance/tsocial)
    │
    ▼
[1.1] 数据预处理 & 图构建
    │   - 加载 tfinance/tsocial 数据集
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
        "risk_type": "集团欺诈",
        "key_features": ["高度数", "密集子图", "短时间大量交易"],
        "anomaly_nodes": [1234, 5678, ...],
        "confidence": 0.92
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
    │
    ▼
[2.3] LLM 规则生成
    │   输出结构化规则 JSON:
    │   {
    │       "rule_id": "R-20260513-001",
    │       "name": "超高连接度欺诈环检测",
    │       "conditions": [
    │           {"field": "node_degree_zscore", "operator": ">", "value": 2.0}
    │       ],
    │       "action": "block",
    │       "severity": "high",
    │       "explanation": "基于GNN发现的Cluster-0模式..."
    │   }
    ▼
[2.4] 规则自优化: 基于沙盒回测反馈迭代优化
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

---

### Stage 3: 沙盒验证层 — 规则回测与评估 (共同)

**目标**: 在标注数据集上回测 LLM 生成的规则，量化评估规则质量。

**流程**:
```
generated_rules.json + 标注数据集
    │
    ▼
[3.1] 规则解析: 将 JSON 规则转为可执行的 Python 函数
    │   - 支持字段: node_degree, node_degree_zscore, feature_dim_X, feature_dim_X_zscore
    │   - 支持操作符: >, <, >=, <=, ==, !=
    │   - 支持逻辑: AND / OR
    │
    ▼
[3.2] 预计算派生特征
    │   - feature_zscores: 各特征维度的 z-score
    │   - degree_zscore: 度数的 z-score
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
│   └── default.yaml                   # 全局配置
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
│   │   ├── rule_generator.py          # LLM 规则生成 (支持 DashScope/OpenAI)
│   │   ├── prompt_templates.py        # Prompt 模板 (含可用字段、数据分布)
│   │   └── rag_retriever.py           # RAG 检索模块
│   │
│   ├── stage3_sandbox/                # Stage 3: 沙盒验证层 (共同)
│   │   ├── __init__.py
│   │   ├── rule_parser.py             # 规则解析器 (支持 z-score 字段)
│   │   └── evaluator.py               # 指标评估 (预计算派生特征)
│   │
│   ├── stage4_kb/                     # Stage 4: 规则知识库 (共同)
│   │   ├── __init__.py
│   │   ├── knowledge_base.py          # 知识库管理
│   │   └── rule_schema.py             # 规则数据模型
│   │
│   └── pipeline.py                    # 主 Pipeline 编排 (含数据分布统计)
│
├── outputs/                           # 运行输出
│   ├── gnn_predictions.pt
│   ├── node_embeddings.pt
│   ├── risk_insights.json
│   ├── generated_rules.json
│   ├── evaluation_report.json
│   └── qualified_rules.json
│
├── notebooks/                         # 实验 Notebook
│   ├── 01_eda.ipynb
│   ├── 02_gnn_training.ipynb
│   ├── 03_llm_rule_gen.ipynb
│   └── 04_full_pipeline.ipynb
│
└── tests/
    ├── test_gnn.py
    ├── test_sandbox.py
    └── test_pipeline.py
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

`.env` 配置示例 (OpenAI 官方):

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# OPENAI_BASE_URL 留空则使用 OpenAI 官方地址
```

支持的 LLM 服务商 (均通过 OpenAI 兼容接口接入):
- **阿里云 DashScope**: `base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`
- **OpenAI 官方**: `base_url` 留空或设为 `https://api.openai.com/v1`
- **Azure OpenAI**: `base_url=https://your-endpoint.openai.azure.com/`
- **本地 Ollama**: `base_url=http://localhost:11434/v1`

> **不配置 API 也能运行**: Stage 2 会自动使用 Mock 规则走完全流程，Stage 1 (GNN) 和 Stage 3 (沙盒回测) 均为真实计算。

### Step 4: 运行

```bash
# 完整 Pipeline (GNN → LLM → 沙盒 → 知识库)
python -m src.pipeline --dataset tfinance --mode full

# 仅运行 GNN 感知层 (不需要 API)
python -m src.pipeline --dataset tfinance --mode gnn_only

# 切换数据集
python -m src.pipeline --dataset yelp --mode full
python -m src.pipeline --dataset amazon --mode full
```

---

## 数据集

| 数据集     | 节点数    | 边数       | 异常比例 | 特征维度 | 来源          |
|-----------|----------|-----------|---------|---------|--------------|
| T-Finance | 39,357   | 42,445,086 | 4.58%   | 10      | ICML 2022    |
| T-Social  | 5,781,065 | 73,105,508 | 3.01%   | 2       | ICML 2022    |
| Yelp      | 45,954   | 3,846,979  | 6.67%   | 32      | DGL Built-in |
| Amazon    | 11,944   | 4,398,392  | 9.50%   | 25      | DGL Built-in |

### 各数据集运行方案

#### T-Finance (推荐首选)
```bash
# 需要手动下载数据集
python -m src.pipeline --dataset tfinance --mode full
```
- **数据准备**: 从 [Google Drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr) 下载 `tfinance` 文件，放入 `dataset/` 目录
- **算力要求**: Mac CPU 约 18 分钟 (100 epochs)，内存 >= 2GB
- **推荐参数**: `hidden_dim=64, order=2, epochs=100` (默认配置即可)
- **适用场景**: 开发调试、论文实验、完整 Pipeline 演示

#### Yelp (欺诈评论检测)
```bash
# 自动从 DGL 下载，无需手动准备
python -m src.pipeline --dataset yelp --mode full
```
- **数据准备**: 首次运行自动下载 (~150MB)，后续使用缓存 (`~/.dgl/`)
- **算力要求**: Mac CPU < 1 分钟，内存 < 1GB
- **推荐参数**: `hidden_dim=64, order=2, epochs=100`

#### Amazon (虚假评论检测)
```bash
# 自动从 DGL 下载
python -m src.pipeline --dataset amazon --mode full
```
- **数据准备**: 首次运行自动下载 (~30MB)
- **算力要求**: Mac CPU < 1 分钟，内存 < 1GB

#### T-Social (大规模社交网络)
```bash
# 需要手动下载数据集，谨慎使用
python -m src.pipeline --dataset tsocial --mode full
```
- **数据准备**: 从 [Google Drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr) 下载 `tsocial` 文件
- **算力要求**: 内存 >= 16GB，CPU 训练约 30~60 分钟
- **推荐参数**: `hidden_dim=10, order=5, epochs=100` (特征维度仅 2，需更高阶小波)
- **注意**: 570 万节点，Mac 8GB 内存可能不够，建议在 16GB+ 机器上运行

---

## 实际运行结果 (tfinance 数据集)

以下为接入真实 LLM (阿里云 DashScope qwen-plus) 后在 T-Finance 数据集上的完整运行结果。

### Stage 1: GNN 风险感知

BWGNN 在 tfinance (39,357 节点 / 42M 边) 上训练 100 epochs 的结果:

| 指标 | 值 |
|------|-----|
| **Test Macro-F1** | **89.95%** |
| **Test AUC** | **94.89%** |
| 训练时长 (Mac CPU) | ~18 分钟 |

GNN 从 39,357 个节点中检测出 **500 个高风险节点**，聚类为 **5 种风险模式**:

| 聚类 ID | 节点数 | 平均度数 | 局部密度 | 异常分数 | 风险描述 |
|---------|--------|---------|---------|---------|---------|
| Cluster 0 | 256 | 547 | 0.416 | 1.000 | 中大规模欺诈环，高度连接 |
| Cluster 1 | 16 | 1,126 | 0.683 | 1.000 | 极小团伙但极高内聚性 |
| Cluster 2 | 181 | 838 | 0.627 | 1.000 | 中等规模欺诈环 |
| Cluster 3 | 38 | 1,133 | 0.879 | 1.000 | 高密度小型欺诈团伙 |
| Cluster 4 | 9 | 1,180 | 0.833 | 1.000 | 高连接 + feature_dim_0 z-score=1.72 异常 |

**关键发现**:
- 所有异常聚类均呈现**高连接度 + 高局部密度**特征，符合金融欺诈中"集团作案"的典型图结构模式
- Cluster 4 是唯一携带显著特征异常的聚类，在 feature_dim_0 上 z-score 达 1.72

### Stage 2: LLM 规则生成

基于 GNN 风险洞察 + 数据分布统计，qwen-plus 自动生成了 **4 条风控规则**:

| 规则 ID | 名称 | 条件 | 动作 | 目标风险 |
|---------|------|------|------|---------|
| R-20260513-001 | 超高连接度欺诈环检测 | `node_degree_zscore > 2.0` | block | Cluster 0 |
| R-20260513-002 | 极端高连接度小团伙检测 | `node_degree_zscore > 2.5` | block | Cluster 1 |
| R-20260513-003 | 高连通欺诈环通用检测 | `node_degree > 800` | review | Cluster 2 & 3 |
| R-20260513-004 | 特征+高连接双重检测 | `feature_dim_0_zscore > 1.5 AND node_degree > 850` | block | Cluster 4 |

### Stage 3: 沙盒回测结果

**第一轮回测** (LLM 初始生成规则):

| 规则 | Recall | Precision | FPR | F1 | 合格 |
|------|--------|-----------|-----|-----|------|
| 超高连接度欺诈环检测 | 27.49% | 1.42% | 20.65% | 0.027 | 否 |
| 极端高连接度小团伙检测 | 0.89% | 0.57% | 0.39% | 0.007 | 否 |
| 高连通欺诈环通用检测 | 27.49% | 1.42% | 20.65% | 0.027 | 否 |
| 特征+高连接双重检测 | 0.06% | 0.03% | 0.08% | 0.001 | 否 |
| **组合 (OR)** | **27.49%** | **1.42%** | **20.65%** | — | — |

**闭环优化迭代后** (LLM 自动优化):

| 规则 | Recall | Precision | FPR | F1 | 合格 |
|------|--------|-----------|-----|-----|------|
| 超高连接度欺诈环优化版 | 1.94% | 0.83% | 11.10% | 0.012 | 否 |
| 极端高连接度小团伙优化版 | 0.00% | 0.00% | 0.39% | 0.000 | 否 |
| 中等规模欺诈环优化版 | 2.44% | 0.79% | 14.65% | 0.012 | 否 |
| 特征+高连接双重检测优化版 | 0.00% | 0.00% | 0.09% | 0.000 | 否 |
| 低连接度特征异常检测 (新增) | 0.06% | 0.22% | 1.22% | 0.001 | 否 |
| **组合 (OR)** | **2.44%** | **0.75%** | **15.57%** | — | — |

**合格标准**: F1 >= 0.60, Recall >= 0.50, FPR <= 0.05, Precision >= 0.40

### 问题分析

当前规则质量未达标的核心原因:

1. **规则表达能力不足**: 简单的阈值规则 (`degree > X AND feature_Y > Z`) 无法有效表达图结构模式 (如密集子图、欺诈环)。GNN 通过频谱分析达到 90% F1，但将这些信息蒸馏为 if-else 规则会损失大量信息。

2. **特征维度信息缺失**: tfinance 的 10 维特征为匿名化特征，LLM 难以基于 feature_dim_0~9 理解业务含义，导致阈值设定缺乏直觉。

3. **闭环迭代"过度修正"**: LLM 在收到"Recall 低"反馈后倾向于大幅放宽条件，导致 FPR 暴增；收到"FPR 高"后又大幅收紧，导致 Recall 接近 0。

### 下一步优化方向

详见项目文档或与开发者讨论，核心方向包括:
- **扩展规则字段**: 增加聚类系数、k-core、PageRank 等图结构派生特征
- **GNN 嵌入驱动**: 利用 GNN 输出的异常分数和嵌入向量作为规则输入
- **混合架构**: GNN 初筛 + LLM 规则精炼的两阶段方案
- **智能反馈**: 定向反馈 + 贝叶斯阈值搜索，替代当前的全量报告反馈

---

## 输出文件说明

| 文件 | 说明 | 生成阶段 |
|------|------|---------|
| `outputs/gnn_predictions.pt` | 每个节点的异常概率 `[N, 2]`，`[:, 1]` 为异常分数 | Stage 1 |
| `outputs/node_embeddings.pt` | 每个节点的 GNN 嵌入向量 `[N, hidden_dim]` | Stage 1 |
| `outputs/risk_insights.json` | 结构化风险洞察: 聚类 + 图结构特征 + 自然语言描述 | Stage 1 |
| `outputs/generated_rules.json` | LLM 生成的风控规则 (JSON 格式) | Stage 2 |
| `outputs/evaluation_report.json` | 沙盒回测指标: TP/FP/FN/TN, Recall, Precision, FPR | Stage 3 |
| `outputs/qualified_rules.json` | 通过回测的优质规则 (沉淀至知识库) | Stage 4 |

---

## 评估指标

| 指标              | 含义                     | 目标   |
|-------------------|--------------------------|--------|
| Macro-F1          | 整体分类平衡性            | > 0.70 |
| AUC-ROC           | 排序质量                  | > 0.85 |
| Recall@TopK       | 高风险节点召回率           | > 0.80 |
| FPR (误杀率)       | 误伤正常用户比例           | < 0.05 |
| Rule Coverage     | 规则覆盖的异常类型数       | > 80%  |
| Rule Precision    | 生成规则中有效规则占比      | > 60%  |

---

## 技术亮点 & 创新点

1. **GNN + LLM 协同**: 首次将图神经网络的结构化风险感知与 LLM 的规则生成能力结合
2. **自主进化闭环**: GNN 感知 → LLM 生成 → 沙盒验证 → 知识库沉淀 → 反馈优化
3. **Beta 小波频谱分析**: 利用 BWGNN 的多尺度频谱特性，捕获传统 GNN 难以识别的异常模式
4. **可解释规则输出**: LLM 生成的规则具有自然语言解释，便于风控运营人员理解和审核
5. **数据分布感知**: 自动计算图级统计分布，辅助 LLM 设定合理的规则阈值
6. **Z-score 派生特征**: 支持标准化后的特征比较，让规则在跨数据集场景下更稳定
7. **多 LLM 后端支持**: 统一的 OpenAI 兼容接口，一键切换 DashScope / OpenAI / Ollama

---

## 开发环境

| 项目 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.2.1 |
| DGL | 2.1.0 |
| LLM | 阿里云 DashScope qwen-plus |
| OS | macOS (Apple Silicon / Intel) |

> DGL 对 PyTorch 版本敏感，请严格使用上述版本组合。
