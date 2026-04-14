# AAC EmotionClassify - 完整AAC交流系统

基于 AAC象形图 → 自然语言翻译 → 情感分析 → 下一轮情感预测 → Emotional RAG图标推荐 的完整流程。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AAC 完整交流系统 (Emotional RAG)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用户输入              翻译输出            情感分析           图标推荐       │
│   ┌──────────┐         ┌──────────┐        ┌──────────┐      ┌──────────┐   │
│   │ AAC符号   │  ──►   │ 自然语言  │  ──►  │ 情感识别  │ ──► │Emotional │   │
│   │[I,happy] │         │ "I am    │        │ happiness │     │   RAG    │   │
│   │          │         │  happy." │        │   (92%)   │     │ 推荐     │   │
│   └──────────┘         └──────────┘        └─────┬────┘      └──────────┘   │
│                                                  │                          │
│                                                  ▼                          │
│                                           ┌──────────┐                      │
│                                           │ 下一轮   │                      │
│                                           │ 情感预测 │                      │
│                                           │ surprise │  ──► 生成情感引导词   │
│                                           │  (65%)   │                      │
│                                           └──────────┘                      │
│                                                                             │
│   Emotional RAG: S(i) = λ·cos(E(Q_emo), E(i)) + (1-λ)·cos(E(Q_orig), E(i))  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 模块组成

| 模块 | 功能 | 基座模型 | 技术 |
|------|------|---------|------|
| **AAC2Text** | 象形图→自然语言 | Qwen2.5-1.5B-Instruct | LoRA生成式微调 |
| **EmotionClassify** | 情感分类+预测 | RoBERTa-base | 多任务LoRA微调 |

---

## 快速开始

### 环境依赖

```bash
# 基础依赖
pip install torch transformers peft scikit-learn numpy tqdm

# AAC2Text 额外依赖
pip install rouge-score nltk sacrebleu bert-score

# 图标推荐语义嵌入
pip install sentence-transformers
```

### 运行方式

```bash
# 1. 交互模式（推荐）
python aac_emotion_pipeline.py --interactive

# 2. 命令行单次输入
python aac_emotion_pipeline.py --symbols "I" "am" "happy"

# 3. 演示模式
python aac_emotion_pipeline.py
```

### 交互模式示例

```
AAC Emotion Pipeline - Interactive Mode
============================================================

Enter AAC symbols separated by spaces (e.g., 'I want_to water')
Enter 'quit' to exit, 'reset' to clear history
============================================================

AAC symbols> I want_to water

────────────────────────────────────────────────────────────
📌 Turn 1: I want water.
   😊 Single: anger (65%)
   🎯 Theme: anger (model predicted)
   🔮 Next Emotion: happiness (72%)
   📈 Trend: stable

   🎯 Recommended Next Icons (Emotional RAG):
      λ=0.3, Target: happiness
      Actions: drink_to(sim:0.42), celebrate_to(sim:0.35)
      Entities: water_bowl(sim:0.40), happy_man(sim:0.38)
────────────────────────────────────────────────────────────

AAC symbols> I feel sad

────────────────────────────────────────────────────────────
📌 Turn 2: I feel sad.
   😊 Single: sadness (88%)
   📍 Current State: anger (recent 3 turns)
   🎯 Theme: sadness (model predicted)
   🔮 Next Emotion: neutral (65%)
   📈 Trend: declining

   🎯 Recommended Next Icons (Emotional RAG):
      λ=0.3, Target: neutral
      Actions: help_to(sim:0.38), relax_to(sim:0.32)
      Entities: sad_lady(sim:0.42), friend(sim:0.35)
────────────────────────────────────────────────────────────
```

---

## 文件结构

```
EmotionClassify/
├── aac_emotion_pipeline.py         # 整合Pipeline（主入口）
├── test_emotional_rag.py           # Emotional RAG 测试脚本
├── config.json                      # 配置文件
│
├── # 情感分类模块
├── cls_trainer.py                   # LoRA训练
├── cls_multitask_trainer.py         # 多任务训练
├── simple_trainer.py                # 全参数训练
├── cls_evaluate.py                  # 模型评估
├── cls_inference.py                 # 模型推理
├── dynamic_emotion_analyzer.py      # 动态情感分析
├── evaluate_full_comparison.py      # 完整对比评估
├── evaluate_next_emotion.py         # 下一轮预测评估
│
├── data/
│   ├── sft_train.json              # 训练数据
│   ├── sft_val.json                # 验证数据
│   ├── sft_test.json               # 测试数据
│   └── emotion_weights.json        # 类别权重
│
├── Model/
│   ├── roberta-base/               # RoBERTa基座模型
│   └── all-MiniLM-L6-v2/           # 语义嵌入模型
│
└── output/
    ├── cls_best/                   # LoRA最佳模型
    ├── cls_final/                  # 多任务最终模型
    └── simple_best/                # 全参数最佳模型
```

---

## 核心算法

### 1. AAC符号翻译 (AAC2Text)

```
输入: ["I", "want_to", "water"]
      ↓
Qwen2.5-1.5B-Instruct + LoRA
      ↓
输出: "I want water."
```

**训练数据生成**：随机符号组合 + LLM翻译/过滤
**评估指标**：BLEU=0.53, BERTScore-F1=0.96

### 2. 情感分析 (EmotionClassify)

**多任务损失函数**：
```
L = L_main + α·L_turn + β·L_consistency + γ·L_contrastive + δ·L_next
```

| 损失 | 权重 | 说明 |
|------|------|------|
| L_main | 1.0 | 整体情绪分类 (Focal Loss) |
| L_turn | 0.3 | 每轮情绪分类 |
| L_consistency | 0.2 | 一致性约束 |
| L_contrastive | 0.1 | 对比学习 |
| L_next | 0.2 | 下一轮预测 |

**损失函数设计原理**：

| 损失项 | 作用 | 说明 |
|--------|------|------|
| **L_main** | 主任务损失 | 预测对话整体情感主题（交叉熵） |
| **L_turn** | 单轮损失 | 预测每一轮的情感，增强细粒度理解 |
| **L_consistency** | 一致性损失 | 让单轮情感和主题情感分布保持一致 |
| **L_contrastive** | 对比损失 | 让易混淆的情感对（如anger/disgust）在表示空间更分离 |
| **L_next** | 下一轮预测损失 | 预测下一轮可能出现的情感 |

这是典型的**多任务辅助学习**策略：主任务 + 多个辅助任务 → 共享表示 → 更好的泛化能力。

### 3. 下一轮情感预测

**模型预测**：`next_classifier` 直接输出

**趋势预测**（对比基线）：
```python
# 效价映射
valence = {"happiness": 1.0, "anger": -0.9, ...}

# 线性趋势预测
predicted_valence = mean_valence + slope
```

### 4. 下一轮图标推荐（Emotional RAG）

**原理**：基于 Emotional RAG 方法，用预测的下一个情感生成引导词，增强语义检索。

**核心公式**：

```
S(i) = λ·cos(E(Q_emo), E(i)) + (1-λ)·cos(E(Q_orig), E(i))
```

| 符号 | 含义 |
|------|------|
| `Q_orig` | 用户原始查询文本（翻译后的自然语言） |
| `Q_emo` | 情感引导的增强查询（原始 + 情感引导词） |
| `E(·)` | 语义嵌入模型（all-MiniLM-L6-v2） |
| `i` | AAC 图标（纯语义，无情感标签） |
| `λ` | 平衡系数（默认0.3，可做消融实验） |

**完整流程**：

```
用户选择图标 → 翻译 → 情感识别 → 预测下一情感 E
                                           ↓
                                    用 E 生成情感引导词
                                           ↓
                    ┌──────────────────────┴──────────────────────┐
                    ↓                                             ↓
              Q_orig (原始)                              Q_emo (增强)
            "I want water"                    "I want water happy joyful"
                    ↓                                             ↓
              E(Q_orig)                                   E(Q_emo)
                    ↓                                             ↓
                    └──────────────────────┬──────────────────────┘
                                           ↓
                              S(i) = λ·sim_emo + (1-λ)·sim_orig
                                           ↓
                                    推荐图标排序
```

**示例**：

```python
# 用户输入: ["I", "want_to", "water"]
# 翻译: "I want water."
# 情感识别: anger
# 预测下一情感: happiness  ← 用这个生成引导词

Q_orig = "I want water."
Q_emo = "I want water. happy joyful celebrate"  # happiness 引导词

# 分别计算相似度
sim_orig = cos(E(Q_orig), E(icon))  # 原始语义匹配
sim_emo = cos(E(Q_emo), E(icon))    # 情感增强匹配

# 融合（λ=0.3）
S(icon) = 0.3 * sim_emo + 0.7 * sim_orig
```

**情感引导词配置**：

| 预测情感 | 情感引导词 (emotion_prompts) | 关联关键词 |
|----------|------------------------------|------------|
| happiness | happy, joyful, excited, celebrate | fun, play, smile, love |
| sadness | sad, need comfort, support | help, comfort, friend, care |
| anger | frustrated, need calm, relax | calm, relax, breathe, peace |
| fear | scared, need safety, protection | safe, protect, help, security |
| surprise | surprised, curious, wonder | look, see, find, discover |

**核心设计**：

| 组件 | 说明 |
|------|------|
| 嵌入模型 | `all-MiniLM-L6-v2` (384维) |
| 图标库 | 3154个AAC图标（过滤135个空图标） |
| 图标文本 | `label: core_semantic super_concept` |
| 平衡系数 | λ = 0.3（可调参） |

**返回分解信息**：

```python
{
    'icon_id': 'drink_to',
    'label': 'to drink',
    'sim_combined': 0.42,  # 融合分数
    'sim_orig': 0.39,      # 原始相似度
    'sim_emo': 0.51        # 情感增强相似度
}
```

**情感转换场景增强**：

当预测情感与当前情感不同时，额外添加转换引导词：

| 转换 | 额外引导词 |
|------|-----------|
| sadness → happiness | cheer up, hope |
| anger → neutral | calm down |
| fear → neutral | feel safe |
| neutral → happiness | excited |

---

## 性能指标

### 情感分类

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Simple | 92.74% | 91.19% |
| MultiTask | 90.73% | 89.11% |

### 下一轮情感预测

| Method | Accuracy | vs Baseline |
|--------|----------|-------------|
| Model (MultiTask) | 62.36% | +14.84% |
| Trend Baseline | 47.52% | - |
| Random | 14.29% | - |

### AAC翻译

| Metric | Score |
|--------|-------|
| BLEU | 0.53 |
| BERTScore-F1 | 0.96 |
| chrF | 85.21 |

---

## 使用场景

### 场景1: 完整AAC交互流程

```python
from aac_emotion_pipeline import AACEmotionPipeline

pipeline = AACEmotionPipeline(...)

# 用户选择AAC图标
result = pipeline.process(["I", "want_to", "water"])

# 输出
print(result['translation']['sentence'])     # "I want water."
print(result['emotion']['single'])           # "anger" (当前情感)
print(result['prediction']['next_emotion'])  # "happiness" (预测下一情感)

# Emotional RAG 图标推荐
icons = result['icon_recommendations']
print(icons['emotional_rag']['Q_orig'])      # "I want water."
print(icons['emotional_rag']['Q_emo'])       # "I want water. happy joyful celebrate"
print(icons['emotional_rag']['lambda'])      # 0.3

# 推荐的图标（按融合相似度排序）
for action in icons['actions'][:3]:
    print(f"{action['label']}: combined={action['sim_combined']:.3f}, "
          f"orig={action['sim_orig']:.3f}, emo={action['sim_emo']:.3f}")
# drink_to: combined=0.42, orig=0.39, emo=0.51
```

### 场景2: 多轮对话追踪

```python
# 追踪多轮对话
conversation = [
    {"symbols": ["I", "am", "happy"], "role": "user"},
    {"symbols": ["that", "is", "good"], "role": "assistant"},
    {"symbols": ["but", "I", "feel", "tired"], "role": "user"},
]

result = pipeline.process_conversation(conversation)
print(result['summary']['dominant_emotion'])  # "neutral"
```

### 场景3: 仅情感分析

```python
from dynamic_emotion_analyzer import DynamicEmotionAnalyzer

analyzer = DynamicEmotionAnalyzer(...)
result = analyzer.analyze_conversation([
    {"role": "user", "content": "I'm so happy!"},
    {"role": "assistant", "content": "That's great!"},
])
```

---

## 测试

### 交互模式测试

```bash
python aac_emotion_pipeline.py --interactive
```

输入示例（用空格分隔图标，多词符号用下划线）：
```
I am happy
I want_to water
I feel sad
I am scared
I need help
```

### Emotional RAG 效果测试

```bash
python test_emotional_rag.py
```

测试内容：
- 单样本测试：显示完整的 RAG 流程和相似度分解
- 多轮对话测试：模拟连续对话场景
- 情感场景测试：测试快乐、悲伤、恐惧、愤怒四种场景

### 测试输出示例

```
输入图标: ['I', 'want_to', 'water']
翻译: I want water.
情感: anger (置信度: 65%)
预测下一情感: happiness

Emotional RAG (λ=0.3):
  Q_orig: I want water.
  Q_emo: I want water. happy joyful celebrate fun play

推荐图标:
  [Action] drink_to: sim_combined=0.420 (orig=0.390, emo=0.510)
  [Action] water_plants_to: sim_combined=0.405 (orig=0.410, emo=0.380)
  [Entity] water_bowl: sim_combined=0.395 (orig=0.380, emo=0.450)
```

---

## API 参考

### AACEmotionPipeline

```python
pipeline = AACEmotionPipeline(
    aac_model_path="path/to/aac_model",        # AAC2Text模型
    aac_base_model_path="path/to/qwen",        # Qwen基座
    emotion_model_path="path/to/emotion",      # 情感模型
    emotion_base_model_path="path/to/roberta", # RoBERTa基座
    device="cuda"
)

# 处理单次输入
result = pipeline.process(symbols=["I", "am", "happy"])

# 处理完整对话
result = pipeline.process_conversation(conversation)

# 重置历史
pipeline.reset()
```

### 返回格式

```python
{
    'input': {
        'symbols': ['I', 'am', 'happy'],
        'role': 'user'
    },
    'translation': {
        'sentence': 'I am happy.'
    },
    'emotion': {
        'single': 'happiness',           # 单句情感
        'current': 'happiness',          # 当前状态（最近3轮众数）
        'theme': 'happiness',            # 主题情感（模型预测）
        'confidence': 0.9235,
        'probabilities': {...}
    },
    'prediction': {
        'next_emotion': 'neutral',
        'confidence': 0.4520,
        'probabilities': {...}
    },
    'icon_recommendations': {            # 下一轮图标推荐
        'actions': [
            {'icon_id': 'celebrate_to', 'label': 'celebrate', 'similarity': 0.35}
        ],
        'entities': [
            {'icon_id': 'laughing_lady', 'label': 'happy person', 'similarity': 0.42}
        ],
        'emotions': [...],
        'combinations': [
            {'action': 'celebrate_to', 'entity': 'laughing_lady', 'label': 'celebrate + happy person'}
        ]
    },
    'trend': {
        'trend': 'improving',
        'direction': 0.15
    },
    'conversation_turn': 1
}
```

---

## 模型下载

| 模型 | 来源 | 大小 |
|------|------|------|
| RoBERTa-base | HuggingFace | ~500MB |
| Qwen2.5-1.5B-Instruct | HuggingFace | ~3GB |
| all-MiniLM-L6-v2 | HuggingFace | ~90MB |
| AAC LoRA权重 | 已训练 | ~17MB |
| Emotion LoRA权重 | 已训练 | ~20MB |

**嵌入模型下载**：
```bash
# 方法1: huggingface-cli
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir Model/all-MiniLM-L6-v2

# 方法2: modelscope镜像（国内）
modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --local_dir Model/all-MiniLM-L6-v2
```

---

## 引用

```bibtex
@misc{aac_emotion_classify_2024,
  title={AAC EmotionClassify: Complete AAC Communication System with Emotion Prediction},
  year={2024}
}
```

---

## License

MIT License
