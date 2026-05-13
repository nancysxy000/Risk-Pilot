# RiskPilot — Presentation Outline

## PPT 结构 (建议 15~20 页)

---

### Slide 1: Title Slide
- **RiskPilot: An Autonomous Evolution Engine for Financial Risk Control Rules via GNN-LLM Collaboration**
- 作者: Nancy Su, Yifei XX
- 学校 / 日期

---

### Slide 2: Background & Motivation (1/2)
- 移动支付风险挑战 (3 个痛点)
  - 风险形态迭代加速 → 传统规则跟不上
  - 风险手法多样且长尾 → 需要多维数据挖掘
  - 风控运营效率不足 → 逐案核查成本高
- 配图: 微信支付场景示意 / 欺诈手法演变时间线

---

### Slide 3: Background & Motivation (2/2)
- 现有方案的局限:
  - GNN: 能检测异常，但输出是数值 → 不能直接变成规则
  - LLM: 能生成规则，但不理解图结构 → 规则质量差
- **核心 Gap**: GNN 和 LLM 之间缺少桥梁
- 配图: GNN output (数字) vs LLM input (文字) 的鸿沟

---

### Slide 4: Our Solution — RiskPilot Overview
- 四阶段闭环 Pipeline 架构图
  ```
  GNN 感知 → LLM 策略 → 沙盒验证 → 规则沉淀
       ↑                                │
       └────── 知识库 RAG 闭环反馈 ──────┘
  ```
- **核心理念**: 让风控体系"越用越聪明"
- 配图: Pipeline 架构图 (README 中的 ASCII → 美化版)

---

### Slide 5: Stage 1 — GNN Perception Layer (1/3)
- 技术选择: BWGNN (Beta Wavelet GNN)
- 为什么选 BWGNN 而不是 GCN/GAT?
  - 异常节点在图谱空间有独特的**频谱特征**
  - Beta 小波基实现多尺度频谱分析
- 配图: BWGNN 的频谱分解示意 (论文 Figure 1)

---

### Slide 6: Stage 1 — GNN Perception Layer (2/3)
- 数据集: T-Finance
  - 39,357 节点 / 42M 边 / 10 维特征 / 4.58% 异常
- 模型结构: Linear → ReLU → PolyConv × 3 → Linear → Softmax
- 训练策略: 加权交叉熵 (weight ≈ 20.8x) + 最优阈值搜索

---

### Slide 7: Stage 1 — Risk Insight Extraction (3/3) ⭐ 核心创新
- **关键贡献**: GNN 数值输出 → 结构化风险描述
- 流程:
  1. 筛选 Top-500 高风险节点
  2. KMeans 聚类 → 5 种风险模式
  3. 分析每个聚类: 度统计 / 局部密度 / 特征 z-score
  4. 生成自然语言描述 (供 LLM 消费)
- 配图: 聚类可视化 + risk_insights.json 示例

---

### Slide 8: Stage 1 — Experimental Results
- 风险聚类结果表格:

| Cluster | Nodes | Avg Degree | Density | Description |
|---------|-------|-----------|---------|-------------|
| 0 | 37 | 1,131 | 0.871 | 密集欺诈核心 |
| 1 | 305 | 594 | 0.453 | 大规模有组织行为 |
| ... | ... | ... | ... | ... |

- **关键发现**: 所有异常聚类都呈现高连接度 + 高密度

---

### Slide 9: Stage 2 — LLM Strategy Layer (1/2)
- 输入: risk_insights.json + 历史规则 (RAG 检索)
- Prompt 设计:
  - System: "你是金融风控专家..."
  - Context: GNN 洞察 + 相似历史规则 + 约束条件
- 输出: 结构化 JSON 规则

---

### Slide 10: Stage 2 — LLM Strategy Layer (2/2)
- 规则示例:
  ```json
  {
    "name": "高度数异常交易检测",
    "conditions": [{"field":"degree", "op":">", "value":50}],
    "action": "review",
    "explanation": "基于 GNN Cluster-0..."
  }
  ```
- 闭环优化: 不合格规则 → 反馈给 LLM 重新生成
- 配图: Prompt 组装流程图

---

### Slide 11: Stage 3 — Sandbox Validation
- 规则解析: JSON → 可执行 Python 函数
- 回测指标:
  - Recall / Precision / FPR / F1
  - 合格标准: F1 ≥ 0.6, FPR ≤ 5%
- 当前结果 (Mock 规则):

| Rule | Recall | FPR | Qualified? |
|------|--------|-----|-----------|
| 高度数检测 | 90.6% | 83.9% | ✗ |
| 孤立账户 | 0.5% | 1.3% | ✗ |

- 分析: Mock 规则阈值需要 LLM 迭代优化

---

### Slide 12: Stage 4 — Rule Knowledge Base
- 合格规则 → 向量化 → 存入 ChromaDB
- 下一轮 GNN 检测 → RAG 检索历史规则 → LLM 增量优化
- 知识库越积越多 → 规则质量越来越好
- 配图: 闭环反馈流程 + 知识库增长曲线 (conceptual)

---

### Slide 13: System Architecture (总览)
- 完整的模块关系图:
  ```
  dataset_loader → BWGNN → insight_extractor
                                    ↓
  knowledge_base ← RAG ← rule_generator
                                    ↓
  knowledge_base ←── evaluator ← rule_parser
  ```
- 代码结构: `src/stage1_gnn/`, `src/stage2_llm/`, `src/stage3_sandbox/`, `src/stage4_kb/`

---

### Slide 14: Technical Highlights
1. **GNN + LLM 协同**: 首次将图频谱分析与 LLM 规则生成结合
2. **结构化洞察桥梁**: GNN 数值 → 聚类 + 图特征 + 自然语言 → LLM 可理解
3. **自主进化闭环**: 无需人工干预，系统自动迭代优化
4. **可解释输出**: 每条规则附带推理依据，运营人员可审核

---

### Slide 15: Comparison with Existing Approaches
| 方法 | GNN 异常检测 | 规则自动生成 | 闭环优化 | 可解释性 |
|------|:---------:|:---------:|:------:|:------:|
| 传统规则系统 | ✗ | ✗ | ✗ | ✓ |
| GNN (BWGNN) | ✓ | ✗ | ✗ | △ |
| LLM 直接生成 | ✗ | ✓ | ✗ | ✓ |
| **RiskPilot** | **✓** | **✓** | **✓** | **✓** |

---

### Slide 16: Limitations & Future Work
- **当前**: Mock LLM → 接入真实 GPT-4 / Qwen2 验证效果
- **聚类**: KMeans 假设球形 → 尝试 DBSCAN 捕获不规则欺诈模式
- **时序**: 当前为静态图 → 加入时序交易序列建模
- **可扩展性**: T-Social 570 万节点 → 需要 mini-batch 训练
- **更多数据集**: SynthAML / AMLworld 反洗钱数据集

---

### Slide 17: Conclusion
- RiskPilot = GNN 感知 + LLM 策略 + 沙盒验证 + 知识库沉淀
- 核心创新: 结构化风险洞察桥梁 (GNN → LLM)
- 在 T-Finance 上验证了 Pipeline 的可行性
- 闭环设计让风控体系**自主进化**

---

### Slide 18: Q&A
- Thank you!
- 开源代码: `github.com/xxx/RiskPilot`

---

## 设计建议
- **配色**: 深蓝 + 白 + 橙色点缀 (金融科技风格)
- **字体**: 标题 Bold, 正文 Regular, 代码用等宽字体
- **图表**: 用 draw.io 或 Figma 画 Pipeline 架构图
- **动画**: Pipeline 的 4 个 Stage 按顺序逐个出现
- **时间分配**: 每页约 1~1.5 分钟，总计 ~20 分钟
