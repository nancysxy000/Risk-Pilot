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

**技术栈**: OpenAI API / LangChain + RAG

**流程**:
```
risk_insights.json + 规则知识库
    │
    ▼
[2.1] RAG 检索: 从知识库中检索相关历史规则
    │
    ▼
[2.2] Prompt 构造: 组装风险洞察 + 历史规则 + 约束条件
    │   System Prompt: "你是金融风控专家..."
    │   Context: {risk_insights} + {similar_rules} + {constraints}
    │
    ▼
[2.3] LLM 规则生成
    │   输出结构化规则 JSON:
    │   {
    │       "rule_id": "R-20260512-001",
    │       "name": "集团欺诈-密集交易检测",
    │       "condition": "degree > 50 AND tx_count_1h > 20 AND ...",
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

---

### Stage 3: 沙盒验证层 — 规则回测与评估 (共同)

**目标**: 在标注数据集上回测 LLM 生成的规则，量化评估规则质量。

**流程**:
```
generated_rules.json + 标注数据集
    │
    ▼
[3.1] 规则解析: 将 JSON 规则转为可执行的 Python 函数
    │
    ▼
[3.2] 规则执行: 在 tfinance/tsocial 数据集上逐条执行规则
    │
    ▼
[3.3] 指标计算:
    │   - Recall (召回率): 能抓到多少真正的异常?
    │   - Precision (精确率): 抓到的里面有多少是对的?
    │   - FPR (误杀率): 误伤了多少正常用户?
    │   - Coverage (覆盖范围): 规则覆盖了多少种异常类型?
    │   - Macro-F1, AUC
    ▼
[3.4] 规则筛选: 通过阈值筛选高质量规则
    │   - F1 > 0.6 且 FPR < 0.05 → 通过
    │   - 否则 → 反馈给 LLM 重新优化
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
independent-project/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包
├── configs/
│   └── default.yaml                   # 全局配置
├── dataset/
│   ├── tfinance                       # T-Finance 数据集 (DGL 二进制)
│   ├── tfinance.zip
│   ├── tsocial                        # T-Social 数据集
│   └── tsocial.zip
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
│   │   ├── rule_generator.py          # LLM 规则生成
│   │   ├── prompt_templates.py        # Prompt 模板
│   │   ├── rag_retriever.py           # RAG 检索模块
│   │   └── rule_optimizer.py          # 规则优化 (基于反馈)
│   │
│   ├── stage3_sandbox/                # Stage 3: 沙盒验证层 (共同)
│   │   ├── __init__.py
│   │   ├── rule_parser.py             # 规则解析器
│   │   ├── rule_executor.py           # 规则执行引擎
│   │   ├── evaluator.py               # 指标评估
│   │   └── report_generator.py        # 报告生成
│   │
│   ├── stage4_kb/                     # Stage 4: 规则知识库 (共同)
│   │   ├── __init__.py
│   │   ├── knowledge_base.py          # 知识库管理
│   │   ├── embedding_store.py         # 向量存储 (FAISS/Chroma)
│   │   └── rule_schema.py             # 规则数据模型
│   │
│   └── pipeline.py                    # 主 Pipeline 编排
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
│   ├── 01_eda.ipynb                   # 数据探索
│   ├── 02_gnn_training.ipynb          # GNN 训练实验
│   ├── 03_llm_rule_gen.ipynb          # LLM 规则生成实验
│   └── 04_full_pipeline.ipynb         # 完整 Pipeline 演示
│
├── tests/                             # 测试
│   ├── test_gnn.py
│   ├── test_sandbox.py
│   └── test_pipeline.py
│
└── scripts/
    ├── run_pipeline.sh                # 一键运行全流程
    └── run_stage1.sh                  # 单独运行 Stage 1
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行完整 Pipeline
python -m src.pipeline --dataset tfinance --mode full

# 3. 单独运行 GNN 感知层
python -m src.stage1_gnn.train --dataset tfinance --epochs 100

# 4. 单独运行沙盒回测
python -m src.stage3_sandbox.evaluator --rules outputs/generated_rules.json
```

---

## 数据集

| 数据集     | 节点数   | 边数      | 异常比例 | 特征维度 | 来源          |
|-----------|---------|----------|---------|---------|--------------|
| T-Finance | ~40万   | ~80万    | ~4.6%   | 10      | ICML 2022    |
| T-Social  | ~570万  | ~7300万  | ~3.0%   | 2       | ICML 2022    |
| Yelp      | ~45K    | ~3.8M   | ~6.7%   | 32      | DGL Built-in |
| Amazon    | ~11K    | ~4.4M   | ~9.5%   | 25      | DGL Built-in |

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
