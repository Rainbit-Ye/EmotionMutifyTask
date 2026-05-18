# AAC2Text — AAC 象形图符号到自然语言翻译

## 项目概述

本项目实现了一个完整的 AAC（Augmentative and Alternative Communication）象形图符号序列到自然语言句子的翻译系统。输入为一组 AAC 象形图标签（如 `["I", "want_to", "water"]`），输出为自然流畅的英文句子（如 `"I want water."`）。

核心技术路线：**LLM 数据生成 + LoRA 微调**，利用 Meta-Llama-3-8B-Instruct 作为基座模型，通过 LoRA 参数高效微调实现 AAC 符号翻译。

---

## 目录结构

```
AAC2Text/
├── config.yaml                          # 主配置文件（数据/模型/训练/LoRA/测试）
├── config/
│   ├── prompts.yaml                     # Agent 提示词模板（翻译Agent + 验证Agent）
│   └── ds_config_zero2.json             # DeepSpeed ZeRO-2 配置
├── data/
│   ├── processed/
│   │   ├── aac_full_ontology.json       # 3,295 条象形图语义本体
│   │   ├── training_data.json           # 30,000 条训练数据
│   │   ├── val_data.json                # 3,000 条验证数据（10% 划分）
│   │   ├── emotions.csv                 # 情感类符号子集
│   │   ├── objects.csv                  # 物体类符号子集
│   │   └── persons.csv                  # 人物类符号子集
│   └── beta1/
│       ├── training_data_01.json        # 早期实验数据（50,000 条）
│       └── training_data_10000.json     # 早期实验数据（10,000 条）
├── scripts/
│   ├── build_full_ontology.py           # 阶段1: 构建语义本体
│   ├── generate_training_data.py        # 阶段2a: 语义约束数据生成（多模型流水线）
│   ├── generate_random_data.py          # 阶段2b: 随机数据生成（单模型方案）
│   ├── train.py                         # 阶段3: LoRA 微调 + DeepSpeed ZeRO-2
│   ├── test.py                          # 阶段4: 模型评估（7项指标）
│   ├── bleu/                            # 本地 BLEU 实现
│   │   ├── __init__.py
│   │   ├── bleu.py                      # BLEU 封装类
│   │   ├── bleu_.py                     # 核心 BLEU 算法（TensorFlow NMT）
│   │   └── tokenizer_13a.py             # WMT mteval-v13a 分词器
│   └── nltk_data/                       # 本地 NLTK 数据
│       ├── corpora/wordnet.zip
│       ├── taggers/averaged_perceptron_tagger_eng/
│       └── tokenizers/punkt_tab/
└── checkpoints/
    └── aac_model/                       # LoRA 微调后的模型权重
        ├── adapter_model.safetensors    # LoRA 权重（~160MB）
        ├── adapter_config.json          # LoRA 配置
        ├── tokenizer.json / tokenizer_config.json
        ├── chat_template.jinja          # Llama-3 Chat Template
        └── checkpoint-{2200,2400,2532}/ # 训练中间 checkpoint
```

---

## 系统架构与原理

### 整体流水线

```
外部 AAC 数据集
       │
       ▼
[阶段1] build_full_ontology.py ──→ aac_full_ontology.json (3,295 条语义本体)
       │                              语义类型 × 语法角色 × 组合关系
       ▼
[阶段2a] generate_training_data.py ──→ training_data.json (30,000 条)
       │    语义约束组合 + 三模型流水线（翻译+CoT验证+公式化评分）
       ▼
[阶段3] train.py ──→ checkpoints/aac_model/ (LoRA 权重)
       │   Meta-Llama-3-8B-Instruct + LoRA 微调 + DeepSpeed ZeRO-2
       ▼
[阶段4] test.py ──→ 评估报告 (7 项指标)
       │   推理 + 多维度评估
       ▼
   自然语言输出
```

### 阶段1：语义本体构建 (`build_full_ontology.py`)

从外部 AAC 数据集读取象形图符号映射，使用 LLM 自动推断每个符号的语义信息。

- **11 种语义类型**：ACTION, ENTITY, EMOTION, PLACE, TIME, QUALITY, PERSON, FOOD, DRINK, BODY, ABSTRACT
- **7 种语法角色**：SUBJECT, PREDICATE, OBJECT, MODIFIER, COMPLEMENT, LOCATION, TIME
- **额外字段**：can_combine_with, super_concept, typical_objects, typical_modifiers
- **方法**：批处理调用 Qwen2.5-1.5B-Instruct，每批最多 15 个符号，含详细分类规则提示

### 阶段2：训练数据生成

提供两种方案：

| | 语义约束生成 (`generate_training_data.py`) | 随机生成 (`generate_random_data.py`) |
|---|---|---|
| 组合策略 | SVO/SV/SVO_EMO 等 7 种语法模式 | 完全随机选 1-7 个符号 |
| 翻译方式 | Translation Agent (Qwen2.5-1.5B) | 单步 LLM 判断有效性+翻译 |
| 质量控制 | CoT Validation Agent (Llama-3-8B) + BERT 公式化评分 | LLM 输出 INVALID 过滤 |
| GPU 需求 | 2 卡（翻译 + CoT 验证） | 1 卡 |
| 输出格式 | `{labels, sentence, type, validation, cot_reasoning}` | `{labels, sentence}` |
| 实际使用 | ✅ 30,000 条训练数据来自此方案 | 备选方案 |

语义约束方案的核心架构：
- **翻译模型**：Qwen2.5-1.5B-Instruct — 生成 AAC 符号翻译
- **CoT 验证模型**：Llama-3-8B-Instruct — 4 维度评分（语义连贯性、语法自然性、标签整合度、整体自然度）
- **公式化评分模型**：BERT 回归器 — 预测 formulaicness 分数（权重：自然性 0.40, 覆盖度 0.25, 公式化 0.15）
- **流水线并行**：翻译线程 + CoT 验证线程通过 Queue 并发，batch_size=4 的 CoT 推理

### 阶段3：LoRA 微调 (`train.py`)

**基座模型**：Meta-Llama-3-8B-Instruct

**LoRA 配置**：
| 参数 | 值 |
|---|---|
| r | 32 |
| alpha | 64 |
| dropout | 0.05 |
| 目标层 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

**训练格式**：Llama-3 Chat Template
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Translate these AAC symbols into ONE simple English sentence: I want_to water<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I want water.<|eot_id|>
```

**关键技术点**：
- Label Masking：user 部分 token 的 loss 设为 -100，只在 assistant 回复上计算 loss
- 句子清理：去除引号，只保留第一行第一句（句号截断）
- 90/10 训练/验证集划分
- DeepSpeed ZeRO-2 多卡训练

**训练超参数**：
| 参数 | 值 |
|---|---|
| epochs | 3 |
| batch_size | 2 (per GPU) |
| gradient_accumulation | 8 |
| effective_batch | 32 (2 GPU × 2 × 8) |
| learning_rate | 2e-4 |
| optimizer | AdamW |
| scheduler | cosine |
| bf16 | True |
| max_length | 128 |
| warmup_ratio | 0.05 |

**训练结果**：
- 总步数：2,532 steps
- 最佳 checkpoint：step 2400, eval_loss = 0.3458

### 阶段4：模型评估 (`test.py`)

#### 评估指标

| 指标 | 说明 | 依赖库 |
|---|---|---|
| BLEU | 4-gram 重叠分数 | 本地 bleu 模块 |
| chrF | 字符级 n-gram F-score，对形态变化更鲁棒 | sacrebleu |
| ROUGE-L | 最长公共子序列 F1，容许语序差异 | rouge-score |
| METEOR | 对齐+同义词+词干匹配，比 BLEU 更宽容 | nltk |
| BERTScore | 基于 RoBERTa-large 的语义相似度（P/R/F1） | bert-score |
| Exact Match | 预测与参考完全相同的比例 | — |
| Partial Match | 预测与参考至少共享一个词的比例 | — |

#### 推理参数

| 参数 | 值 |
|---|---|
| max_new_tokens | 20 |
| do_sample | False（贪心解码） |
| stop_strings | `<\|im_end\|>`, `<\|eot_id\|>`, `\n` |
| 句号截断 | 取第一个句号结束的完整句子，去掉多余续写 |

---

## 运行方式

### 环境依赖

```bash
conda activate AgentPipeine
```

核心依赖：
- Python 3.10
- PyTorch（CUDA）
- transformers, peft, accelerate, deepspeed
- rouge-score, nltk, sacrebleu, bert-score
- NLTK 数据已置于 `scripts/nltk_data/`（无需联网下载）
- BERTScore 模型置于 `/home/user1/liuduanye/EmotionClassify/bertscore_model`（roberta-large）

### 阶段1：构建语义本体

```bash
python scripts/build_full_ontology.py
```

输出：`data/processed/aac_full_ontology.json`

### 阶段2：生成训练数据

```bash
# 语义约束方案（实际使用的方案，需 2 卡 GPU）
python scripts/generate_training_data.py --num 30000

# 随机生成方案（备选，1 卡即可）
python scripts/generate_random_data.py --num 50000
```

输出：`data/processed/training_data.json`

### 阶段3：训练模型

```bash
# 多卡 DeepSpeed 训练
deepspeed --num_gpus 3 scripts/train.py

# 指定配置文件
deepspeed --num_gpus 3 scripts/train.py --config /path/to/config.yaml

# 覆盖参数
deepspeed --num_gpus 3 scripts/train.py --num 10000 --epochs 5 --batch 2 --lr 2e-4
```

训练完成后自动运行测试。也可单独测试：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --test
```

### 阶段4：评估模型

```bash
# 默认 50 条采样
python scripts/test.py

# 指定采样数量
python scripts/test.py --num 100

# 指定配置文件
python scripts/test.py --config /path/to/config.yaml
```

---

## 关键设计决策

1. **语义约束生成作为主方案**：7 种 SVO 语法模式 + 三模型流水线（翻译 + CoT 验证 + 公式化评分）产生的数据质量更高，30K 训练数据采用此方案
2. **Llama-3-8B 作为基座模型**：比 Qwen2.5-1.5B 更强的英语生成能力，配合 LoRA r=32 覆盖 7 个目标模块
3. **Chat Template 格式**：训练和推理统一使用 Llama-3 的 `<|start_header_id|>` 格式，prompt 为 `"Translate these AAC symbols into ONE simple English sentence:"`
4. **句号截断策略**：推理输出取第一个句号结束的完整句子，避免 LLM 惯性续写产生的多余内容
5. **多维度评估**：7 项指标互补——BLEU/chrF 衡量字面匹配，METEOR/ROUGE-L 容许同义变换，BERTScore 衡量语义相似度
6. **本地化依赖**：BLEU 实现、NLTK 数据、BERTScore 模型均本地化，无需运行时联网

---

## 配置文件说明

### config.yaml

| 配置段 | 关键字段 | 说明 |
|---|---|---|
| data | train_data, val_data, num_train, val_ratio | 数据路径与划分 |
| model | base_model, output_dir, max_length | 基座模型路径与输出 |
| training | epochs, batch_size, learning_rate, warmup_ratio 等 | 训练超参数 |
| lora | r, lora_alpha, lora_dropout, target_modules | LoRA 配置 |
| test | test_samples | 手动测试用例 |

### config/prompts.yaml

- **translation_agent / translation_prompt**：Translation Expert 角色，将 AAC 符号列表翻译为英文句子，要求使用所有符号并自然重组，无法组合时输出 REJECT
- **validation_agent / validation_cot_prompt**：CoT Quality Evaluator 角色，4 维度评分（Semantic Coherence, Grammatical Naturalness, Label Integration, Overall Naturalness），每项 1-5 分

### config/ds_config_zero2.json

DeepSpeed ZeRO-2 配置：bf16 自动混合精度，AdamW 优化器，WarmupLR 调度器，ZeRO Stage 2 无 offloading
