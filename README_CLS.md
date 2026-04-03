# 情绪分类模块 (Classification)

基于 RoBERTa 的情绪分类训练与推理模块。

## 文件说明

| 文件 | 说明 |
|------|------|
| `cls_trainer.py` | 基础分类训练器（推荐） |
| `cls_multitask_trainer.py` | 多任务分类训练器（高级） |
| `cls_inference.py` | 分类推理脚本 |
| `cls_evaluate.py` | 分类评估脚本 |

## 训练器选择

### 基础版 (cls_trainer.py) - 推荐

**适用场景**：大多数情况下的首选

**特点**：
- 单任务学习，只预测整体情绪
- 使用 `RobertaForSequenceClassification`
- 训练快速，结构简单
- 模型保存为 transformers 标准格式

**优点**：
- 训练效率高
- 容易调试和维护
- 模型兼容性好

### 多任务版 (cls_multitask_trainer.py) - 高级

**适用场景**：
- 基础版效果不理想
- 数据中包含每轮情绪标签
- 需要解决情绪混淆问题

**特点**：
- 主任务：预测整体情绪
- 辅助任务：预测每轮情绪
- 一致性约束：确保预测一致性
- 动态样本权重：关注困难样本
- Focal Loss + 对比学习

**优点**：
- 更强的区分能力
- 解决易混淆情绪对

**注意**：需要数据中每轮对话都有 `emotion` 标签

## 快速开始

### 1. 训练

**使用基础版**：
```bash
python cls_trainer.py
```

**使用多任务版**：
```bash
python cls_multitask_trainer.py
```

或在 `train.py` 中配置：
```json
{
  "cls": {
    "enabled": true,
    "use_multitask": false  // true 为多任务版
  }
}
```

### 2. 推理

```bash
python cls_inference.py
```

### 3. 评估

```bash
python cls_evaluate.py
```

## 配置说明

在 `config.json` 中配置：

```json
{
  "cls": {
    "enabled": true,
    "use_multitask": false,
    "num_epochs": 30,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_length": 256,
    "weight_decay": 0.01
  }
}
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enabled` | 是否启用分类训练 | true |
| `use_multitask` | 是否使用多任务训练器 | false |
| `num_epochs` | 训练轮数 | 30 |
| `batch_size` | 批次大小 | 8 |
| `learning_rate` | 学习率 | 2e-5 |
| `warmup_ratio` | 预热比例 | 0.1 |
| `max_length` | 最大序列长度 | 256 |
| `weight_decay` | 权重衰减 | 0.01 |

## 数据格式

训练数据格式 (`sft_train.json`)：

```json
[
  {
    "conversation": [
      {"role": "user", "content": "今天工作怎么样？"},
      {"role": "assistant", "content": "太累了，老板又骂我", "emotion": "sadness"}
    ],
    "main_emotion": "sadness"
  }
]
```

**基础版**：需要 `conversation` 和 `main_emotion`

**多任务版**：额外需要每轮对话的 `emotion` 标签

## 情绪类别

支持 7 种情绪：

```
neutral, anger, disgust, fear, happiness, sadness, surprise
```

## 输出文件

训练后保存到 `output/` 目录：

```
output/
├── cls_best/              # 最佳验证准确率模型
├── cls_final/             # 最终模型
└── cls_checkpoint_epoch_N/ # 每轮检查点
```

## 类别权重

可选的类别权重文件 `data/emotion_weights.json`：

```json
{
  "neutral": 0.5,
  "anger": 1.5,
  "disgust": 2.0,
  "fear": 1.8,
  "happiness": 1.0,
  "sadness": 1.2,
  "surprise": 1.5
}
```

用于处理类别不平衡问题。

## 建议

1. **首次训练**：使用基础版
2. **效果不佳**：尝试多任务版
3. **数据不平衡**：添加类别权重文件
4. **调参顺序**：learning_rate → batch_size → num_epochs

## 性能对比

| 训练器 | 训练速度 | 模型大小 | 推荐场景 |
|--------|---------|---------|---------|
| 基础版 | 快 | 标准 | 常规使用 |
| 多任务版 | 慢 | 标准+ | 困难样本、混淆情绪 |
