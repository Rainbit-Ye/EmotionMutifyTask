# AAC2Text — AAC 象形图符号到自然语言翻译

## 项目概述

本项目实现了一个完整的 AAC（Augmentative and Alternative Communication）象形图符号序列到自然语言句子的翻译系统。输入为一组 AAC 象形图标签（如 `["I", "want_to", "water"]`），输出为自然流畅的英文句子（如 `"I want water."`）。

核心技术路线：**LLM 数据生成 + LoRA 微调**，利用 Qwen2.5-1.5B-Instruct 作为基座模型，通过 LoRA 参数高效微调实现 AAC 符号翻译。

---

## 目录结构

```
AAC2Text/
├── config.yaml                          # 主配置文件（数据/模型/训练/LoRA/测试）
├── config/
│   └── prompts.yaml                     # Agent 提示词模板（翻译Agent + 验证Agent）
├── data/
│   ├── processed/
│   │   ├── aac_full_ontology.json       # 3,295 条象形图语义本体
│   │   ├── training_data.json           # 50,000 条训练数据
│   │   └── val_data.json                # 5,000 条验证数据
│   └── raw/                             # 原始数据（预留目录）
├── scripts/
│   ├── build_full_ontology.py           # 阶段1: 构建语义本体
│   ├── generate_training_data.py        # 阶段2a: 语义约束数据生成
│   ├── generate_random_data.py          # 阶段2b: 随机数据生成（实际使用方案）
│   ├── train.py                         # 阶段3: LoRA 微调
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
        ├── adapter_model.safetensors    # LoRA 权重（~17MB）
        ├── adapter_config.json          # LoRA 配置
        ├── tokenizer.json / tokenizer_config.json
        ├── chat_template.jinja
        └── checkpoint-{3600,4000,4221}/ # 训练中间 checkpoint
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
[阶段2] generate_random_data.py ──→ training_data.json (50,000 条)
       │   随机符号组合 + LLM 翻译/过滤
       ▼
[阶段3] train.py ──→ checkpoints/aac_model/ (LoRA 权重)
       │   Qwen2.5-1.5B-Instruct + LoRA 微调
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

提供两种方案，**实际使用随机生成方案**（`generate_random_data.py`）：

| | 语义约束生成 (`generate_training_data.py`) | 随机生成 (`generate_random_data.py`) |
|---|---|---|
| 组合策略 | SVO/SV/SVO_EMO/SV_EMO 语法模式 | 完全随机选 1-7 个符号 |
| 翻译方式 | Translation Agent 翻译，REJECT 过滤 | 单步 LLM 判断有效性+翻译 |
| 质量控制 | Validation Agent（已定义未调用） | LLM 输出 INVALID 过滤 |
| 数据量 | — | 50,000 条 |

数据格式：
```json
{"labels": ["pilot", "wrap_to", "daydream_to"], "sentence": "Pilot wraps up daydreaming.", "type": "sv_emo"}
```

### 阶段3：LoRA 微调 (`train.py`)

**基座模型**：Qwen2.5-1.5B-Instruct

**LoRA 配置**：
| 参数 | 值 |
|---|---|
| r | 16 |
| alpha | 32 |
| dropout | 0.1 |
| 目标层 | q_proj, k_proj, v_proj, o_proj |

**训练格式**：Qwen2 Chat Template
```
<|im_start|>user
Translate these AAC symbols to a sentence: I want_to water<|im_end|>
<|im_start|>assistant
I want water.<|im_end|>
```

**关键技术点**：
- Label Masking：user 部分 token 的 loss 设为 -100，只在 assistant 回复上计算 loss
- 句子清理：去除引号，只保留第一行第一句
- 90/10 训练/验证集划分

**训练超参数**：
| 参数 | 值 |
|---|---|
| epochs | 3 |
| batch_size | 8 |
| gradient_accumulation | 4 |
| learning_rate | 2e-5 |
| optimizer | AdamW |
| fp16 | True |
| max_length | 128 |

**训练结果**：
- 总步数：4,221 steps
- 最佳 checkpoint：step 4200, eval_loss = 0.2796

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

#### 实测结果（50 条验证集采样）

```
BLEU:                  0.5304
chrF:                  85.21
ROUGE-L:               0.7665
METEOR:                0.7932
BERTScore-Precision:   0.9597
BERTScore-Recall:      0.9606
BERTScore-F1:          0.9601
Exact Match:           0.2400
Partial Match:         1.0000
```

#### 结果解读

- **语义质量优秀**：BERTScore-F1 = 0.96，模型输出语义与参考高度一致
- **字面匹配中等**：BLEU = 0.53，Exact Match = 0.24，说明模型倾向同义换表达
- **覆盖度好**：Partial Match = 1.0，每条预测至少与参考共享一个词

---

## 运行方式

### 环境依赖

```bash
conda activate AgentPipeine
```

核心依赖：
- Python 3.10
- PyTorch（CUDA）
- transformers, peft, accelerate
- rouge-score, nltk, sacrebleu, bert-score
- NLTK 数据已置于 `scripts/nltk_data/`（无需联网下载）
- BERTScore 模型置于 `/home/user1/liuduanye/AgentPipeline/bertscore_model`（roberta-large）

### 阶段1：构建语义本体

```bash
python scripts/build_full_ontology.py
```

输出：`data/processed/aac_full_ontology.json`

### 阶段2：生成训练数据

```bash
# 随机生成方案（实际使用的方案）
python scripts/generate_random_data.py

# 语义约束方案（备选）
python scripts/generate_training_data.py
```

输出：`data/processed/training_data.json`

### 阶段3：训练模型

```bash
# 完整训练
CUDA_VISIBLE_DEVICES=2 python scripts/train.py

# 指定配置文件
python scripts/train.py --config /path/to/config.yaml

# 覆盖参数
python scripts/train.py --num 10000 --epochs 5 --batch 16 --lr 3e-5
```

训练完成后自动运行测试。也可单独测试：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --test
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

1. **随机生成优于语义约束**：随机组合 + LLM 过滤产生的数据多样性更好，最终 50K 训练数据采用此方案
2. **Chat Template 格式**：训练和推理统一使用 Qwen2 的 `<|im_start|>` 格式，而非简单的 "Input: ... Output:" 拼接
3. **LoRA 参数高效微调**：仅微调注意力层的 4 个投影矩阵，可训练参数约 17MB
4. **本地化依赖**：BLEU 实现、NLTK 数据、BERTScore 模型均本地化，无需运行时联网

---

## 配置文件说明

### config.yaml

| 配置段 | 关键字段 | 说明 |
|---|---|---|
| data | train_data, val_data, num_train, val_ratio | 数据路径与划分 |
| model | base_model, output_dir, max_length | 基座模型路径与输出 |
| training | epochs, batch_size, lr, warmup_steps 等 | 训练超参数 |
| lora | r, lora_alpha, lora_dropout, target_modules | LoRA 配置 |
| test | test_samples | 手动测试用例 |

### config/prompts.yaml

- **translation_agent / translation_prompt**：将 AAC 符号列表翻译为英文句子，要求使用所有符号，无法组合时输出 REJECT
- **validation_agent / validation_prompt**：评估翻译质量（所有标签是否使用、语法是否正确、语义是否合理），输出 JSON 格式的 accept/revise/reject 判定
