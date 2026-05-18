# 情绪分类模块 (Emotion Classification)

基于 RoBERTa 的情绪分类训练与推理模块，支持 7 种情绪分类。

## 情绪类别

```
neutral, anger, disgust, fear, happiness, sadness, surprise
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `cls_trainer.py` | 基础分类训练器 — LoRA + `RobertaForSequenceClassification`，单任务 |
| `cls_multitask_trainer.py` | 多任务分类训练器 — 自定义多头模型 + Focal Loss + 对比学习 |
| `simple_trainer.py` | 全参数微调基线 — 无 LoRA，作为对比 |
| `cls_inference.py` | 推理/预测脚本，支持交互模式 |
| `cls_evaluate.py` | 评估脚本，支持 3 种模型对比 |
| `dynamic_emotion_analyzer.py` | 实时情绪分析器 — 滑动窗口趋势追踪 + 异常检测 + 下一轮情绪预测 |
| `evaluate_full_comparison.py` | 全量对比评估 — 主情绪 + 下一轮情绪预测对比 |
| `evaluate_next_emotion.py` | 下一轮情绪预测专项评估 |

---

## 三种训练方法

| 方法 | 文件 | 模型架构 | 保存格式 | 适用场景 |
|------|------|---------|---------|---------|
| 基础版 | `cls_trainer.py` | `RobertaForSequenceClassification` + LoRA | PEFT (adapter_model.safetensors) | 常规使用，推荐首选 |
| 多任务版 | `cls_multitask_trainer.py` | 自定义 `MultiTaskEmotionClassifier` + LoRA | torch.save (model.pt) | 需要逐轮情绪、下一轮预测、混淆情绪区分 |
| 简单基线 | `simple_trainer.py` | `RobertaForSequenceClassification` 全参数 | HF (model.safetensors) | 对比基线 |

### 基础版 (cls_trainer.py)

- 单任务学习，只预测整体情绪 (`main_emotion`)
- LoRA 微调 query, value, key, dense 层
- 加权 CrossEntropyLoss 处理类别不平衡
- 模型保存为 PEFT 标准格式

### 多任务版 (cls_multitask_trainer.py)

4 个训练目标 + 动态样本权重：

```
L = L_main + α*L_turn + β*L_consistency + γ*L_contrastive + δ*L_next
```

| 损失项 | 权重 | 说明 |
|--------|------|------|
| L_main | 1.0 | 主情绪分类（CrossEntropy + Focal Loss） |
| L_turn | 0.3 | 逐轮情绪分类 |
| L_consistency | 0.2 | 逐轮预测与主预测一致性约束 |
| L_contrastive | 0.1 | 易混淆情绪对对比学习（anger/disgust, sadness/surprise 等） |
| L_next | 0.2 | 下一轮情绪预测 |

**注意**：多任务版需要数据中每轮对话都有 `emotion` 标签。

---

## 数据格式

### 数据规模

| 数据集 | 条数 |
|--------|------|
| 训练集 | 19,491 |
| 验证集 | 2,438 |
| 测试集 | 2,439 |

### 数据格式 (sft_train.json / sft_val.json / sft_test.json)

```json
[
  {
    "conversation": [
      {"role": "user", "content": "今天工作怎么样？", "emotion": "neutral"},
      {"role": "assistant", "content": "太累了，老板又骂我", "emotion": "sadness"}
    ],
    "main_emotion": "sadness",
    "has_non_neutral": true,
    "emotion_counts": {"neutral": 1, "sadness": 1}
  }
]
```

- **基础版**：需要 `conversation` 和 `main_emotion`
- **多任务版**：额外需要每轮对话的 `emotion` 标签

### 类别权重

`data/emotion_weights.json`：
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

neutral 权重最高 (2.0)，用于抵消模型对多数类的偏好。

---

## 快速开始

### 1. 训练

```bash
# 基础版（推荐首选）
python emotion_classify/cls_trainer.py

# 多任务版
python emotion_classify/cls_multitask_trainer.py

# 简单基线
python emotion_classify/simple_trainer.py
```

或在 `config.json` 中配置：
```json
{
  "cls": {
    "enabled": true,
    "use_multitask": false
  }
}
```

### 2. 推理

```bash
# 默认模式
python emotion_classify/cls_inference.py

# 交互模式
python emotion_classify/cls_inference.py --interactive

# 指定模型
python emotion_classify/cls_inference.py --model_path output/cls_best --base_model_path Model/roberta-base
```

### 3. 评估

```bash
# 评估基础版 + 简单基线
python emotion_classify/cls_evaluate.py --eval_cls --eval_simple

# 评估多任务版
python emotion_classify/cls_evaluate.py --eval_multitask --multitask_model output/cls_final

# 全量对比（主情绪 + 下一轮预测）
python emotion_classify/evaluate_full_comparison.py

# 下一轮情绪预测专项
python emotion_classify/evaluate_next_emotion.py
```

---

## 配置说明

### config.json

```json
{
  "model": {
    "model_path": "/home/user1/liuduanye/EmotionClassify/Model/roberta-base",
    "output_dir": "/home/user1/liuduanye/EmotionClassify/output"
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "value", "key", "dense"],
    "lora_dropout": 0.1,
    "bias": "none"
  },
  "cls": {
    "enabled": true,
    "use_multitask": true,
    "num_epochs": 30,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_length": 256,
    "weight_decay": 0.01,
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

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enabled` | 是否启用分类训练 | true |
| `use_multitask` | 是否使用多任务训练器 | true |
| `num_epochs` | 训练轮数 | 30 |
| `batch_size` | 批次大小 | 8 |
| `learning_rate` | 学习率 | 2e-5 |
| `warmup_ratio` | 预热比例 | 0.1 |
| `max_length` | 最大序列长度 | 256 |
| `weight_decay` | 权重衰减 | 0.01 |
| `loss_weights` | 多任务损失权重 | 见上表 |

---

## 输出文件

训练后保存到 `output/` 目录：

```
output/
├── cls_best/                          # 最佳验证准确率模型（LoRA + PEFT 格式）
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── model.pt                       # 多任务版额外保存的完整权重
│   └── tokenizer files
├── cls_final/                         # 最终模型（同上格式）
├── cls_checkpoint_epoch_N/            # 每轮检查点
├── simple_best/                       # 全参数基线最佳模型
├── simple_final/                      # 全参数基线最终模型
├── simple_checkpoint_epoch_N/         # 基线检查点
├── evaluation/                        # 评估结果
└── comparison/                        # 对比结果
```

**注意**：基础版和多任务版共用 `cls_best`/`cls_final` 路径，多任务版额外保存 `model.pt`（自定义模型权重）。如果两个都训练过，后者会覆盖前者。

---

## 实时情绪分析 (dynamic_emotion_analyzer.py)

面向生产环境的实时情绪追踪组件：

- **EmotionTracker**：滑动窗口（size=10）趋势追踪，基于效价(Valence)线性回归预测趋势
- **异常检测**：效价突变 > 1.0 触发告警
- **下一轮预测**：模型预测 + 趋势外推双通道
- **自适应灵敏度**：根据预测准确率在线调整灵敏度参数

```python
from emotion_classify.dynamic_emotion_analyzer import DynamicEmotionAnalyzer

analyzer = DynamicEmotionAnalyzer(
    model_path="output/cls_final",
    base_model_path="Model/roberta-base"
)

result = analyzer.analyze_turn("I feel so frustrated today", "user", predict_next=True)
print(result["emotion"], result["confidence"])
print(result.get("next_emotion"))
```

---

## 关键技术

- **LoRA 微调**：r=8, alpha=16, 只训练 query/value/key/dense 层
- **类别权重**：neutral 权重 2.0，处理类别不平衡
- **Focal Loss**：多任务版聚焦困难样本 (gamma=2.0)
- **对比学习**：易混淆情绪对惩罚 (anger/disgust, sadness/surprise, happiness/surprise)
- **动态样本权重**：根据准确率自适应调整 (范围 [1, 3])
- **一致性约束**：JS 散度确保逐轮预测与主预测一致

---

## 建议调参顺序

1. 首次训练 → 基础版
2. 效果不佳 → 多任务版
3. 数据不平衡 → 调整 `emotion_weights.json`
4. 调参顺序 → learning_rate → batch_size → num_epochs → loss_weights
