# EmotionClassify - 基于RoBERTa的情绪分类系统

基于 RoBERTa 的多任务情绪分类系统，支持 **7种情绪分类** 和 **下一轮情绪预测**。

---

## 项目概述

| 项目 | 说明 |
|------|------|
| **任务** | 对话情绪分类 + 下一轮情绪预测 |
| **基座模型** | RoBERTa-base (125M参数) |
| **情绪类别** | neutral, anger, disgust, fear, happiness, sadness, surprise |
| **训练数据** | DailyDialog (约12K对话) |
| **核心技术** | LoRA微调、多任务学习、Focal Loss、对比学习 |

---

## 目录结构

```
EmotionClassify/
├── config.json                    # 配置文件
├── cls_trainer.py                 # 基础LoRA训练
├── cls_multitask_trainer.py       # 多任务训练（推荐）
├── simple_trainer.py              # 全参数微调训练
├── cls_evaluate.py                # 模型评估
├── cls_inference.py               # 模型推理
├── dynamic_emotion_analyzer.py    # 动态情感分析
├── evaluate_full_comparison.py    # 完整对比评估
├── evaluate_next_emotion.py       # 下一轮预测评估
├── data/
│   ├── sft_train.json            # 训练集
│   ├── sft_val.json              # 验证集
│   ├── sft_test.json             # 测试集
│   └── emotion_weights.json      # 类别权重
├── Model/
│   └── roberta-base/             # 基座模型
└── output/
    ├── cls_best/                 # LoRA最佳模型
    ├── cls_final/                # 多任务最终模型
    └── simple_best/              # 全参数最佳模型
```

---

## 快速开始

### 环境依赖

```bash
pip install torch transformers peft scikit-learn matplotlib seaborn tqdm numpy
```

### 训练模型

```bash
# 方式1: 多任务训练（推荐，支持下一轮预测）
python cls_multitask_trainer.py

# 方式2: 基础LoRA训练
python cls_trainer.py

# 方式3: 全参数微调
python simple_trainer.py
```

### 评估模型

```bash
# 完整对比评估（推荐）
python evaluate_full_comparison.py

# 单独评估下一轮预测
python evaluate_next_emotion.py
```

### 推理使用

```bash
# 交互式推理
python cls_inference.py --interactive --model_path output/cls_final

# 动态情感分析
python dynamic_emotion_analyzer.py
```

---

## 模型架构

### 三种训练方法对比

| 方法 | 文件 | 特点 | 适用场景 |
|------|------|------|---------|
| **MultiTask** | `cls_multitask_trainer.py` | 多任务学习 + 下一轮预测 | 需要预测下一轮情绪 |
| **LoRA** | `cls_trainer.py` | LoRA微调，参数高效 | 快速迭代，资源有限 |
| **Simple** | `simple_trainer.py` | 全参数微调 | 追求最佳分类准确率 |

### 多任务模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    输入对话                                  │
│         "User: I'm so happy!\nAssistant: Great!"            │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 RoBERTa Encoder (LoRA)                      │
│              hidden_states [batch, seq_len, 768]            │
└─────────────────────────┬───────────────────────────────────┘
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ [CLS] 向量   │ │ 每轮末尾向量 │ │ 最后轮向量   │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │main_classifier│turn_classifier│next_classifier
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │整体情绪     │ │每轮情绪     │ │下一轮情绪   │
   │ (7类)      │ │ (n轮×7类)   │ │ (7类)       │
   └─────────────┘ └─────────────┘ └─────────────┘
```

---

## 核心算法

### 1. 损失函数

多任务总损失：
```
L = L_main + α·L_turn + β·L_consistency + γ·L_contrastive + δ·L_next
```

| 损失 | 权重 | 说明 |
|------|------|------|
| `L_main` | 1.0 | 主任务：整体情绪分类 (Focal Loss) |
| `L_turn` | 0.3 | 辅助任务：每轮情绪分类 (Focal Loss) |
| `L_consistency` | 0.2 | 一致性约束：每轮情绪众数应与整体一致 |
| `L_contrastive` | 0.1 | 对比学习：区分相似情绪对 |
| `L_next` | 0.2 | 下一轮情绪预测 (Focal Loss) |

### 2. Focal Loss

```python
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

# 默认参数
α = class_weights    # 类别权重
γ = 2.0             # 聚焦参数
```

**作用**：更关注难分类样本，减少简单样本的权重。

### 3. 对比学习损失

针对容易混淆的情绪对增强区分能力：
```python
CONFUSING_PAIRS = [
    ("anger", "disgust"),
    ("sadness", "surprise"),
    ("happiness", "surprise")
]
```

### 4. 动态样本权重

根据预测准确度动态调整样本权重：
```python
# 准确度低的样本权重高
sample_weight = 1.0 + 2.0 * (1.0 - accuracy)  # 范围 [1, 3]
```

### 5. 下一轮情绪预测

**模型预测**：使用 `next_classifier` 直接预测

**趋势预测**（对比基线）：
```python
# 1. 将情绪映射到效价值
valence = {
    "happiness": +1.0, "surprise": +0.3, "neutral": 0.0,
    "disgust": -0.6, "fear": -0.7, "sadness": -0.8, "anger": -0.9
}

# 2. 线性回归计算趋势
slope = np.polyfit(time_steps, valences, 1)[0]

# 3. 预测下一轮效价
predicted_valence = mean_valence + slope

# 4. 映射回情绪类别
```

---

## 实验结果

### 整体情绪分类

| Model | Accuracy | Macro F1 | Non-neutral Acc |
|-------|----------|----------|-----------------|
| **Simple** | **92.74%** | **91.19%** | **95.56%** |
| LoRA (cls) | 91.27% | 89.23% | 94.80% |
| MultiTask | 90.73% | 89.11% | 94.09% |

### 各类别准确率

| Emotion | cls | multitask | simple |
|---------|-----|-----------|--------|
| neutral | 48.94% | 50.53% | **59.04%** |
| anger | 97.65% | 97.23% | **97.87%** |
| disgust | **100%** | 99.60% | **100%** |
| fear | **100%** | **100%** | **100%** |
| happiness | 83.37% | 81.02% | **85.07%** |
| sadness | 98.51% | **98.72%** | 98.51% |
| surprise | 95.52% | 94.88% | **97.23%** |

### 下一轮情绪预测

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| **Model (MultiTask)** | **62.36%** | **45.90%** |
| Trend Baseline | 47.52% | 17.73% |
| Random Baseline | 14.29% | - |
| **Improvement** | **+14.84%** | **+28.17%** |

---

## 配置说明

### config.json

```json
{
  "model": {
    "model_path": "Model/roberta-base",
    "output_dir": "output"
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "value", "key", "dense"],
    "lora_dropout": 0.1
  },
  "cls": {
    "num_epochs": 30,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_length": 256,
    "loss_weights": {
      "main": 1.0,
      "turn": 0.3,
      "consistency": 0.2,
      "contrastive": 0.1,
      "next": 0.2
    }
  }
}
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora.r` | 8 | LoRA秩 |
| `lora.lora_alpha` | 16 | LoRA缩放参数 |
| `num_epochs` | 30 | 训练轮数 |
| `batch_size` | 8 | 批大小 |
| `learning_rate` | 2e-5 | 学习率 |
| `max_length` | 256 | 最大序列长度 |

---

## 数据格式

### 训练数据格式 (sft_*.json)

```json
{
  "conversation": [
    {"role": "user", "content": "I am tired of everything.", "emotion": "neutral"},
    {"role": "assistant", "content": "What? How happy you are!", "emotion": "surprise"}
  ],
  "main_emotion": "surprise",
  "has_non_neutral": true,
  "emotion_counts": {"neutral": 1, "surprise": 1}
}
```

### 类别权重 (emotion_weights.json)

```json
{
  "neutral": 2.0,
  "anger": 1.0,
  "disgust": 1.2,
  "fear": 1.5,
  "happiness": 1.2,
  "sadness": 1.0,
  "surprise": 1.0
}
```

---

## 使用建议

### 场景选择

| 需求 | 推荐模型 | 原因 |
|------|---------|------|
| 仅分类当前情绪 | Simple | 最高准确率 (92.74%) |
| 预测下一轮情绪 | MultiTask | 唯一支持 (62.36%) |
| 资源受限 | LoRA | 参数高效 |
| 两者都需要 | MultiTask | 略有取舍 |

### 已知问题

1. **neutral 类别准确率低** (48-59%)
   - 原因：数据不平衡，neutral 样本过多
   - 建议：增加其他情绪样本或调整类别权重

2. **多任务学习有 trade-off**
   - 整体分类略降 (~2%)，但获得下一轮预测能力
   - 可调整 `loss_weights` 平衡

---

## 评估指标说明

| 指标 | 说明 |
|------|------|
| Accuracy | 整体准确率 |
| Macro F1 | 各类别F1的平均，不受类别不平衡影响 |
| Weighted F1 | 按样本数加权的F1 |
| Non-neutral Acc | 非neutral类别的准确率（更重要） |

---

## 依赖版本

```
torch >= 1.12.0
transformers >= 4.20.0
peft >= 0.3.0
scikit-learn >= 1.0.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
tqdm >= 4.60.0
```

---

## 引用

```bibtex
@misc{emotionclassify2024,
  title={EmotionClassify: Multi-task Emotion Classification with Next-turn Prediction},
  author={Your Name},
  year={2024}
}
```

---

## License

MIT License
