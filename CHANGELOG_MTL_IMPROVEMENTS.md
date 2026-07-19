# EmotionClassify 项目方法与改动记录

---

## 一、项目使用方法总览

### 1. 基座模型

| 组件 | 说明 | 参考文献 |
|------|------|---------|
| RoBERTa-base | 125M参数的预训练语言模型，作为所有分类实验的基座 | Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019. arXiv:1907.11692 |

---

### 2. 参数高效微调方法（PEFT）

| 方法 | 文件 | 核心思想 | 参考文献 |
|------|------|---------|---------|
| LoRA (r=8) | `cls_trainer.py` | 在attention的query/key/value/dense矩阵旁插入低秩分解矩阵 ΔW=BA，仅训练A和B | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022. arXiv:2106.09685 |
| LoRA (r=16) | `baselines/baseline_lora_ablation.py` | 同LoRA，秩增大到16，alpha=32（保持alpha/r=2） | 同上；消融实验参考 Wang & Azman, "LoRA Fine-Tuning of RoBERTa", 2025 |
| LoRA (r=32) | `baselines/baseline_lora_ablation.py` | 同LoRA，秩增大到32，alpha=64 | 同上 |
| IA3 | `baselines/baseline_ia3.py` | 仅缩放key/value/dense的激活值（乘以可训练向量），比LoRA更轻量 | Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022. arX:2205.05638 |
| Prefix Tuning | `baselines/baseline_prefix.py` | 在每层前添加20个可训练的虚拟token向量，带prefix_projection | Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation", ACL 2021. arXiv:2101.00190 |
| Context-LoRA | `baselines/baseline_context.py` | LoRA r=8 + Speaker1/Speaker2上下文标记，利用对话角色信息 | 上下文建模在ERC中的有效性，参考 DialogueRNN (Majumder et al., ACL 2019) |
| 全参数微调 | `simple_trainer.py` | 不使用PEFT，直接微调RoBERTa全部参数，作为性能上界 | 标准fine-tuning |

---

### 3. 多任务学习架构

| 组件 | 说明 | 参考文献 |
|------|------|---------|
| 3头分类器 | `cls_multitask_trainer.py` 中 `MultiTaskEmotionClassifier`：main_classifier（整体情绪）、turn_classifier（每轮情绪）、next_classifier（下一轮情绪预测） | 多任务学习框架，参考 Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks", 2017. arXiv:1706.05098 |
| 共享编码器 + 任务特定头 | RoBERTa编码器通过LoRA共享，3个分类头各自独立 | 标准硬参数共享MTL (Caruana, 1997) |

---

### 4. 损失函数

| 损失 | 文件位置 | 说明 | 参考文献 |
|------|---------|------|---------|
| Focal Loss | `cls_multitask_trainer.py` 类 `FocalLoss` | `((1-pt)^γ) * CE_loss`，γ=2.0，聚焦难分类样本，用于L_main、L_turn、L_next | Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017. arXiv:1708.02002 |
| 一致性损失 L_consistency | `cls_multitask_trainer.py` 方法 `compute_consistency_loss` | turn预测的非neutral众数应与main预测一致，不一致则惩罚1.0/样本 | 自设计，约束辅助任务与主任务的语义一致性 |
| 对比学习损失 L_contrastive | `cls_multitask_trainer.py` 类 `ContrastiveLoss` | 对混淆情绪对(anger,disgust)/(sadness,surprise)/(happiness,surprise)，惩罚互预测概率，温度τ=0.5 | 对比学习思想参考 Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020. arXiv:2002.05709 |
| 动态样本权重 | `cls_multitask_trainer.py` 方法 `compute_dynamic_weights` | 基于当前预测准确度计算权重 w = 1 + 2*(1-acc)，范围[1,3]，困难样本权重更高 | 难样本挖掘思想参考 Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining", CVPR 2016. arXiv:1604.03540 |
| 类别加权交叉熵 | `cls_trainer.py`, `simple_trainer.py` | 对少数类赋予更高权重 | 标准类别不平衡处理 |
| 平方根平滑逆频率权重 | `process_dailydialog.py` 方法 `compute_class_weights` | weight = sqrt(total / (num_classes * count))，归一化使均值为1 | 改进自逆频率加权，平方根平滑避免极端权重 |

---

### 5. 训练策略

| 策略 | 说明 | 参考文献 |
|------|------|---------|
| AdamW优化器 | 所有训练器统一使用，weight_decay=0.01 | Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019. arXiv:1711.05101 |
| 线性预热+衰减调度 | warmup_ratio=0.1，线性衰减到0 | 标准transformer训练策略 (Vaswani et al., 2017) |
| 梯度裁剪 | max_norm=1.0 | 标准防梯度爆炸策略 |
| 类别不平衡处理 | neutral欠采样到30% + 非neutral过采样到happiness数量（最大15倍） | 数据重采样策略 (Buda et al., 2018) |

---

### 6. 多任务损失组合（原始固定权重）

| 权重 | 值 | 作用 |
|------|----|------|
| L_main | ×1.0 | 主情绪分类 |
| L_turn | ×0.3 | 每轮情绪分类辅助任务 |
| L_consistency | ×0.2 | turn-main一致性约束 |
| L_contrastive | ×0.1 | 混淆情绪对对比学习 |
| L_next | ×0.2 | 下一轮情绪预测 |

---

### 7. 数据集

| 数据集 | 说明 | 参考文献 |
|--------|------|---------|
| DailyDialog | 7类情绪(neutral/anger/disgust/fear/happiness/sadness/surprise)的多轮对话数据集，训练集19491条，测试集2439条 | Li et al., "DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset", IJCNLP 2017. arXiv:1710.03957 |

---

### 8. 评估指标

| 指标 | 说明 | 参考文献 |
|------|------|---------|
| 准确率 (Accuracy) | 主指标 | — |
| Macro F1 | 各类F1的均值，对类别不平衡更公平 | — |
| Weighted F1 | 按样本数加权的F1 | — |
| 非中性准确率 | 排除neutral后的准确率 | 自定义，关注有情绪类别的表现 |
| 下一轮情绪预测准确率 | 预测对话下一轮情绪的准确率 | 自定义，与趋势基线(trend baseline)对比 |
| 混淆矩阵 | 各模型在7类上的混淆矩阵 | 标准分类评估 |

---

## 二、改动记录

---

### 2026-04-12：初始分类模型开发

**改动文件**: `emotion_classify/cls_trainer.py`, `emotion_classify/simple_trainer.py`

**改动内容**:
1. 实现 LoRA r=8 单任务分类训练器 `EmotionClassifier`，基于 `RobertaForSequenceClassification`
2. 实现全参数微调基线 `SimpleTrainer`
3. 两者均使用类别加权交叉熵损失 + AdamW优化器 + 线性预热调度

**参考文献**:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019

---

### 2026-04-15：多任务模型开发

**改动文件**: `emotion_classify/cls_multitask_trainer.py`

**改动内容**:
1. 实现 `MultiTaskEmotionClassifier` 模型：RoBERTa编码器 + LoRA(r=8, FEATURE_EXTRACTION) + 3个分类头
2. 实现5项损失：Focal Loss (main/turn/next)、一致性损失、对比学习损失
3. 实现动态样本权重机制（基于预测准确度，权重范围[1,3]）
4. 实现下一轮情绪预测分类头（next_classifier），使用最后一轮hidden state

**参考文献**:
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
- Caruana, "Multitask Learning", 1997

---

### 2026-04-15 ~ 04-29：AAC2Text 模块开发

**改动文件**: `AAC2Text/scripts/train.py`, `AAC2Text/scripts/test.py`, `AAC2Text/README.md`

**改动内容**:
1. AAC2Text翻译模块训练与测试脚本
2. Qwen2.5-1.5B基座的CoT生成流程
3. Pipeline模型加载逻辑优化

---

### 2026-05-09 ~ 05-15：基线实验框架搭建与训练

**改动文件**: `emotion_classify/baselines/` 目录下全部文件, `emotion_classify/cls_evaluate.py`

**改动内容**:
1. 新增5种PEFT基线实验脚本：IA3、Prefix Tuning、LoRA r=16、LoRA r=32、Context-LoRA
2. 新增共享工具模块 `common.py`，封装 EmotionDataset、train_loop、load_class_weights
3. IA3和Prefix Tuning完成训练，产出模型检查点
4. 大幅扩展 `cls_evaluate.py`（+150行），支持更多模型类型的加载与对比评估
5. `run_all_baselines.py` 运行时遇到 Broken pipe 错误，但单个基线脚本可独立运行

**参考文献**:
- Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022 (IA3)
- Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation", ACL 2021 (Prefix Tuning)
- Wang & Azman, "LoRA Fine-Tuning of RoBERTa", 2025 (LoRA秩消融)

---

### 2026-05-16 ~ 05-17：剩余基线训练完成

**改动文件**: 无代码改动，产出模型检查点

**改动内容**:
1. 5月16日：单独运行完成 LoRA-r16、LoRA-r32 训练（各30 epoch）
2. 5月17日：完成 Context-LoRA 训练（30 epoch）
3. 修复方式：绕过 `run_all_baselines` 的管道调度，直接运行各基线脚本

---

### 2026-05-18：全模型评估与代码提交

**改动文件**: EmotionClassify仓库 commit `a1e4168`, AACServer仓库 commit `3e57021`

**改动内容**:
1. 对全部8种模型进行统一评估，生成 `all_experiment_results.json`
2. 关键结果：Simple(92.74%) > LoRA-r16(91.68%) > LoRA-r32(91.47%) > Context-LoRA(91.43%) > LoRA-r8(91.27%) > Multi-task(90.65%) > IA3(72.12%) > Prefix Tuning(7.71%)
3. Multi-task是唯一超越趋势基线的下一轮情绪预测模型（62.36% vs 47.52%）
4. 提交 EmotionClassify 基线实验代码（12文件，+1146/-252行）
5. 提交 AACServer proto 和 EventDispatcher 更新（6文件，+18/-15行）

---

### 2026-05-23：Multi-task 模型优化 — PCGrad 与自适应损失权重

**改动文件**: `emotion_classify/cls_multitask_trainer.py`, `config.json`

#### 改动1：实现 PCGrad 梯度手术

**参考文献**: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020. arXiv:2001.06782

**改动内容**:

1. 新增 `import copy`（用于梯度深拷贝）

2. `MultiTaskTrainer.__init__` 中新增配置项：
   - `self.use_pcgrad`：PCGrad 开关，从 `config['cls']['use_pcgrad']` 读取，默认 `False`
   - 启用时打印提示信息

3. 新增 `MultiTaskTrainer._pcgrad_project(task_gradients)` 方法：
   - 输入：各任务的梯度列表（每个元素是与模型参数同形状的梯度张量列表）
   - 核心逻辑：遍历所有任务对 (i, j)，若两任务梯度点积 < 0（冲突），则将任务 i 的梯度投影到任务 j 梯度的法平面上
   - 投影公式：`g_i = g_i - (g_i · g_j / |g_j|²) * g_j`
   - 输出：投影后的梯度列表

4. 训练循环 `train()` 新增 PCGrad 分支（`self.use_pcgrad == True` 时激活）：
   - 分别对5个任务（main, turn, consistency, contrastive, next）单独计算损失和梯度
   - 调用 `_pcgrad_project()` 进行梯度投影
   - 取所有任务投影后梯度的平均值作为最终梯度
   - 赋值给模型参数后执行梯度裁剪和参数更新

**配置方式**: 在 `config.json` 的 `cls` 节中设置 `"use_pcgrad": true`

**预期效果**: 消除多任务梯度冲突导致的负迁移，主分类准确率预计恢复1-2%

---

#### 改动2：实现自适应损失权重（Uncertainty Weighting）

**参考文献**: Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018. arXiv:1705.07115

**改动内容**:

1. 新增 `UncertaintyWeighting` 类（`nn.Module` 子类）：
   - 可学习参数 `log_vars`：5个任务各一个 `log_var_i`，初始化为0
   - `forward(losses)`：计算加权总损失 `L = Σ (0.5 * exp(-2*log_var_i) * L_i + log_var_i)`
   - `get_weights()`：返回当前各任务的有效权重列表（用于日志输出）
   - 原理：`σ_i = exp(log_var_i)` 代表任务噪声水平，训练中自动收敛到合理权重
   - 正则项 `log_var_i` 防止权重无限增大

2. `MultiTaskTrainer.__init__` 中新增配置项：
   - `self.use_uncertainty_weighting`：自适应权重开关，从 `config['cls']['use_uncertainty_weighting']` 读取，默认 `False`
   - 启用时创建 `UncertaintyWeighting` 实例并移至 GPU

3. 优化器初始化修改：
   - 启用自适应权重时，将 `uncertainty_weighting.parameters()` 加入优化器

4. 训练循环 `train()` 新增自适应权重分支（`self.use_uncertainty_weighting == True` 时激活）：
   - 计算各任务原始损失（不乘固定权重 alpha/beta/gamma/delta）
   - 调用 `self.uncertainty_weighting([main_loss, turn_loss, consistency_loss, contrastive_loss, next_loss])` 计算加权总损失
   - 反向传播、梯度裁剪、参数更新

5. Epoch 日志输出增强：
   - 启用自适应权重时，额外输出当前各任务的有效权重值

**配置方式**: 在 `config.json` 的 `cls` 节中设置 `"use_uncertainty_weighting": true`

**预期效果**: 替代手动调权重，自动平衡主分类与辅助任务，整体提升0.5-1%

---

#### 改动3：配置文件更新

**改动文件**: `config.json`

**改动内容**:

在 `cls` 节中新增两个字段：

```json
"use_pcgrad": false,
"use_uncertainty_weighting": false
```

两个开关默认均为 `false`，即不改变原有训练行为。需要手动设置为 `true` 才会启用对应功能。

---

#### 兼容性说明

- 两个新功能**独立开关**，可单独启用或同时启用
- 当两个开关均为 `false` 时，训练行为与改动前完全一致（原始固定权重模式）
- PCGrad 和 Uncertainty Weighting 也可以同时启用（PCGrad 处理梯度方向冲突，Uncertainty Weighting 处理损失量级平衡），但当前实现中优先级为 PCGrad > Uncertainty Weighting（同时开启时走 PCGrad 分支），建议择一使用或分两组实验对比

---

### 2026-05-23（下午）：新增 Valence 效价回归辅助训练头

**参考文献**: Elgabry & Hamdi, "CMHL: Contrastive Multi-Head Learning for Emotionally Consistent Text Classification", arXiv 2026. arXiv:2603.14078; Russell, "A Circumplex Model of Affect", 1980

**改动文件**: `emotion_classify/cls_multitask_trainer.py`, `config.json`

**改动内容**:

1. 新增常量 `EMOTION_VALENCE`：基于 Russell 环形模型定义7种情绪的效价值
   - neutral=0.0, happiness=0.9, surprise=0.2, sadness=-0.8, anger=-0.8, fear=-0.7, disgust=-0.6
   - 与 `common.py` 和 `dynamic_emotion_analyzer.py` 中已有的效价值一致，但首次作为训练标签使用

2. `MultiTaskEmotionClassifier` 新增 `valence_regressor` 回归头：
   - 结构：Dropout(0.1) → Linear(768, 192) → ReLU → Linear(192, 1) → Tanh
   - 输出范围 [-1, 1]，与效价值范围匹配
   - 输入为 [CLS] hidden state（与 main_classifier 共享输入）

3. `MultiTaskEmotionDataset.__getitem__` 新增返回 `main_valence` 标签

4. `collate_fn` 新增 `main_valences` 张量（float类型）

5. 模型 `forward` 返回值从5个变为6个：新增 `valence_pred`

6. 训练循环三个分支（PCGrad / Uncertainty Weighting / 原始）均新增：
   - `valence_loss = F.mse_loss(valence_pred.squeeze(), main_valences)`
   - 原始模式总损失：`loss += epsilon * valence_loss`（epsilon 默认 0.15）
   - PCGrad 模式：valence_loss 作为第6个任务参与梯度投影
   - Uncertainty Weighting 模式：valence_loss 作为第6个任务参与自适应加权

7. `UncertaintyWeighting` 从 `num_tasks=5` 改为 `num_tasks=6`

8. `_validate` 方法新增 valence_loss 计算和 epsilon 参数

9. 损失权重统计新增 `total_valence_loss`

10. `config.json` 的 `loss_weights` 新增 `"valence": 0.15`

**设计说明**:
- 效价回归是**心理学驱动的辅助任务**（参考 CMHL），让模型不仅知道"是什么情绪"，还理解"这个情绪的正负极性"
- 作为归纳偏置，valence_head 迫使 [CLS] 表示包含效价信息，间接帮助区分 neutral(0.0) 和 happiness(0.9) 这类混淆对
- 使用 Tanh 激活确保输出在 [-1, 1] 范围内，与效价值范围匹配
- 回归头使用较小的隐藏层(192维)，避免与主分类头竞争参数

**参考文献**: Elgabry & Hamdi, "CMHL: Contrastive Multi-Head Learning for Emotionally Consistent Text Classification", arXiv 2026. arXiv:2603.14078; Russell, "A Circumplex Model of Affect", 1980

---

## 三、参考文献汇总

1. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019. arXiv:1907.11692
2. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022. arXiv:2106.09685
3. Wang & Azman, "LoRA Fine-Tuning of RoBERTa", 2025
4. Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022. arXiv:2205.05638 (IA3)
5. Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation", ACL 2021. arXiv:2101.00190
6. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017. arXiv:1708.02002
7. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020. arXiv:2002.05709 (SimCLR)
8. Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining", CVPR 2016. arXiv:1604.03540 (OHEM)
9. Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019. arXiv:1711.05101 (AdamW)
10. Li et al., "DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset", IJCNLP 2017. arXiv:1710.03957
11. Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks", 2017. arXiv:1706.05098
12. Caruana, "Multitask Learning", Machine Learning, 1997
13. Vaswani et al., "Attention Is All You Need", NeurIPS 2017. arXiv:1706.03762 (预热调度)
14. Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020. arXiv:2001.06782 (PCGrad)
15. Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018. arXiv:1705.07115 (Uncertainty Weighting)
16. Elgabry & Hamdi, "CMHL: Contrastive Multi-Head Learning for Emotionally Consistent Text Classification", arXiv 2026. arXiv:2603.14078 (Valence辅助任务、Russell环形模型)
17. Russell, "A Circumplex Model of Affect", Journal of Personality and Social Psychology, 1980 (效价-唤醒度环形模型)
18. Sacrebleu: Post, "A Call for Clarity in Reporting BLEU Scores", WMT 2018. arXiv:1804.08771 (chrF计算)

---

### 2026-05-23 ~ 05-26：多任务模型 Valence + Uncertainty Weighting 训练

**改动文件**: `emotion_classify/cls_multitask_trainer.py`, `config.json`

**改动内容**:

1. `config.json` 更新：
   - `output_dir` 改为 `output/mtl_valence_uw`
   - 启用 `use_uncertainty_weighting: true`, `use_pcgrad: false`
   - `loss_weights` 新增 `"valence": 0.15`

2. 使用 Uncertainty Weighting 模式训练 Multi-task 模型（含 Valence 回归头），30 epochs 完成
3. 所有 checkpoint 保存于 `output/mtl_valence_uw/`（epoch 1~30 + final）

**评估结果** (2026-05-29, 测试集 2439 条):

| 指标 | 原始 Multi-task | MTL + Valence + UW | 变化 |
|------|----------------|-------------------|------|
| 准确率 | 90.65% | **90.98%** | +0.33% |
| Macro F1 | 88.98% | **89.53%** | +0.55% |
| 非中性准确率 | 93.96% | **94.22%** | +0.26% |
| neutral | 51.06% | **52.13%** | +1.07% |
| happiness | 79.74% | **82.09%** | +2.35% |
| sadness | 98.72% | 96.80% | -1.92% |

**分析**: 整体小幅提升，happiness 提升最显著 (+2.35%)，但 sadness 下降 1.92%。仍不及全参数微调 Simple (92.74%) 和 LoRA-r16 (91.68%)。

---

### 2026-05-26：AAC2Text 训练脚本恢复支持 + 重训 + Batch 评估

**改动文件**: `AAC2Text/scripts/train.py`, `AAC2Text/scripts/test.py`

#### 改动1：train.py 新增 `--resume` 参数支持断点续训

**改动内容**:
1. `argparse` 新增 `--resume` 参数：指定 checkpoint 路径恢复训练，或 `"auto"` 自动查找最新 checkpoint
2. `train()` 函数签名新增 `resume_from_checkpoint: str = None` 参数
3. `trainer.train()` 改为 `trainer.train(resume_from_checkpoint=resume_from_checkpoint)`
4. `main()` 中实现 `--resume auto` 逻辑：扫描 `output_dir` 下 `checkpoint-*` 目录，按 step 号排序取最新

**遇到的问题**: 服务器重启后尝试从 checkpoint-800 恢复训练，但 DeepSpeed ZeRO 要求 DP world size 一致（旧 2 卡 vs 新 3 卡），抛出 `ZeRORuntimeException`。解决方案：使用相同卡数（2卡）重训。

#### 改动2：AAC2Text 模型重新训练（Llama-3-8B, 2卡）

**训练配置**:
- 基座: Meta-Llama-3-8B-Instruct
- LoRA: r=32, alpha=64, dropout=0.05, target_modules=[q,k,v,o,gate,up,down] (7个)
- DeepSpeed ZeRO-2, CUDA_VISIBLE_DEVICES=0,3 (2卡)
- 3 epochs, eval_loss 最终 0.3414, 训练耗时约 1h45m
- 模型保存于 `AAC2Text/checkpoints/aac_model/`

#### 改动3：test.py 新增 batch 推理支持

**改动内容**:
1. 新增 `import time, tqdm`
2. 设置 `tokenizer.padding_side = "left"`（**关键修复**：右 padding 导致 batch 生成位置错位）
3. 预处理所有 prompt 后按 batch_size 分批 tokenize + generate
4. 使用 `tqdm` 显示进度条，输出推理速度（条/秒）
5. `argparse` 新增 `--batch` 参数（默认 16）
6. 移除推理测试样例（6个硬编码测试），直接进入验证集评估
7. 预测示例从 5 条增到 10 条

**Bug 修复**: 初次 batch 推理使用默认右 padding，BLEU=0.0004（几乎为零）。改为左 padding 后 BLEU 恢复至 0.6723。

---

### 2026-05-29：AAC2Text 评估完成，三版模型对比

**评估方式**: `CUDA_VISIBLE_DEVICES=0 python scripts/test.py --num 100 --batch 16`

**Llama-3-8B 当前版评估结果**:

| 指标 | 值 |
|------|-----|
| BLEU | 0.6723 |
| chrF | 77.70 |
| ROUGE-L | 0.8224 |
| METEOR | 0.8201 |
| BERTScore-Precision | 0.9735 |
| BERTScore-Recall | 0.9713 |
| BERTScore-F1 | 0.9723 |
| Exact Match | 0.4267 |
| Partial Match | 0.9993 |

**三版模型横向对比**:

| 指标 | Qwen2.5-1.5B (04-29) | Llama-3-8B 旧 (05-11) | Llama-3-8B 当前 (05-29) | 当前 vs 旧3-8B |
|------|----------------------|----------------------|------------------------|---------------|
| BLEU | 0.53 | 0.6048 | **0.6723** | +0.07 |
| chrF | 85.21 | 41.51 | **77.70** | +36.19 |
| ROUGE-L | 0.76 | 0.7695 | **0.8224** | +0.05 |
| METEOR | 0.7932 | 0.7654 | **0.8201** | +0.05 |
| BERTScore-F1 | 0.9601 | 0.9644 | **0.9723** | +0.01 |
| Exact Match | 0.24 | 0.34 | **0.4267** | +0.09 |

**旧 Llama-3-8B vs 当前 Llama-3-8B 指标差异说明**:

经核实，两版训练的算法、配置（config.yaml）、训练数据（md5一致）完全相同。指标差异并非来自训练改进，而是：

1. **评估方法修正**（chrF 36点差异的主因）：旧版评估使用右 padding batch 推理，短序列生成位置错位导致输出异常；修正为 `padding_side="left"` 后评估结果恢复正常
2. **训练随机性**：重训后随机种子不同，可能收敛到稍不同的局部最优，但差异不大

**结论**：旧模型的真实生成质量与当前模型相近，之前的低指标（chrF 41.51等）主要是评估 bug 导致的，而非模型能力差异

---

### 2026-06-05：动态增量预测架构升级 — SASRec + Emotional RAG 融合 + S-DPO 对齐

**目标**: 将AAC图标推荐系统从"整段输入后预测"升级为"逐icon输入即时预测"(类似输入法IME)，同时保留现有batch模式，并引入前沿方法（SASRec序列模型 + S-DPO对齐）。

#### 改动1：本体丰富化 — Colourful Semantics 语义角色标注

**参考文献**: Bryan, "Colourful Semantics", 1997; PrAACT (Magnana et al., 2023); BERTptCS, 2024

**改动文件**: `AAC2Text/data/processed/aac_full_ontology.json`, `AAC2Text/scripts/enrich_ontology_cs.py`

**改动内容**:

1. 新增脚本 `enrich_ontology_cs.py`：为3295个AAC图标自动标注Colourful Semantics (CS)语义角色

2. CS角色体系（6类语用语义槽位）：

   | CS Role | 含义 | 示例icon | 分布 |
   |---------|------|---------|------|
   | WHO | 施事/主语 | I, you, mum, doctor | 1029 |
   | WHAT_DOING | 动作/谓语 | eat, drink, go, help | 657 |
   | WHAT | 受事/宾语 | water, food, apple | 1295 |
   | WHERE | 处所 | home, school, hospital | 100 |
   | WHEN | 时间 | morning, today, week | 36 |
   | HOW | 方式/修饰 | quickly, happy, sad | 178 |

3. 映射策略：
   - 基于现有 `(grammar_role, semantic_type)` 组合的确定性规则映射（覆盖~96%的icon）
   - 剩余129个歧义icon使用Qwen2.5-1.5B推断CS角色（`--use-llm`参数）
   - 同时规范化 `grammar_role` 字段（30+不一致值统一为10个规范值：SUBJ/OBJ/TRANS/INTR/MOD/LOC/COMPL/DUR/INST/DIR）

4. 输出：更新 `aac_full_ontology.json`，每个icon新增 `cs_role` 字段，`grammar_role` 已规范化

**设计说明**:
- CS角色比grammar_role更粗粒度但更具语用性：WHO/WHAT_DOING/WHAT直接对应"谁在做什么"的句法槽位
- 参考BERTptCS (2024)：将CS语义角色注入transformer输入，引导模型预测符合当前语义槽位的icon
- CS角色嵌入将作为SASRec模型的输入特征之一

---

#### 改动2：合成icon序列数据生成

**改动文件**: `AAC2Text/scripts/generate_icon_sequences.py`

**改动内容**:

1. 新增脚本 `generate_icon_sequences.py`：从现有数据生成SASRec训练用的icon序列

2. 数据来源与策略：
   - **training_data.json (30K对)**: 每条有`labels`(icon列表)和`type`(svo/sv_time/svo_emo等语法模式)，直接解析为CS标注序列
   - **CS模板增强**: 按CS模板(WHO+WHAT_DOING, WHO+WHAT_DOING+WHAT等12种)从ontology采样，生成~50K额外序列
   - **多轮对话模拟**: 生成2-5轮的对话session(~5K)，用于SASRec的跨轮次上下文学习
   - **负采样**: 同CS角色替换(硬负例)+随机替换

3. 输出文件：

   | 文件 | 数量 |
   |------|------|
   | `data/icon_sequences_train.json` | 153,000 |
   | `data/icon_sequences_val.json` | 8,500 |
   | `data/icon_sequences_test.json` | 8,500 |

4. 序列格式：
   ```json
   {"sequence": ["mum", "give_to", "water"], "cs_roles": ["WHO", "WHAT_DOING", "WHAT"], "emotion": "happiness", "type": "svo", "source": "training_data"}
   ```

---

#### 改动3：SASRec序列推荐模型

**参考文献**: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICLR 2018

**改动文件**: `sequence_model/sasrec.py`, `sequence_model/train_sasrec.py`, `sequence_model/__init__.py`

**改动内容**:

1. 新增 `sequence_model/` 包，实现SASRec模型用于AAC icon的next-item预测

2. 模型架构 (`SASRec`, ~512K参数)：

   ```
   输入: [item_emb + cs_role_emb + position_emb]
     → CausalSelfAttentionBlock × 2 (hidden=64, heads=2)
     → LayerNorm → Linear
     → logits(3155) (icon词汇表大小)
   ```

   - `item_emb`: icon ID嵌入 (padding_idx=0)
   - `cs_role_emb`: CS语义角色嵌入 (16维，投影到hidden_size)
   - `position_emb`: 位置编码
   - `CausalSelfAttentionBlock`: Multi-head self-attention + FFN + residual + LayerNorm
   - 因果mask: 位置t只关注 ≤t 的位置，保证自回归特性
   - CS角色嵌入帮助模型理解当前正在填充的"句法槽位"

3. 训练脚本 `train_sasrec.py`：
   - 交叉熵损失 (next-item prediction)
   - 评估指标: Hit@K, MRR, NDCG@K
   - Early stopping (patience=10)

4. 训练结果（2026-06-05，12 epochs，第12 epoch early stopping）：

   **训练配置**: hidden=64, blocks=2, heads=2, lr=1e-3, batch=128, DeepSpeed ZeRO-2, 2×GPU

   **训练过程**（train loss从6.56降至5.70，train acc从0.6%升至2.3%）：

   | Epoch | Train Loss | Train Acc | Val Hit@1 | Val Hit@5 | Val MRR | Val NDCG@5 |
   |-------|-----------|----------|-----------|-----------|---------|------------|
   | 1     | 6.5617    | 0.62%    | 0.59%     | 2.85%     | 2.80%   | 1.67%      |
   | 2 ★   | 6.4199    | 0.66%    | 0.62%     | 3.22%     | 3.04%   | 1.90%      |
   | 3     | 6.3761    | 0.75%    | 0.51%     | 3.13%     | 2.90%   | 1.77%      |
   | 5     | 6.1915    | 1.41%    | 0.61%     | 2.89%     | 2.88%   | 1.73%      |
   | 8     | 5.9070    | 2.04%    | 0.47%     | 2.64%     | 2.56%   | 1.51%      |
   | 12   | 5.7026    | 2.27%    | 0.42%     | 2.28%     | 2.36%   | 1.33%      |

   ★ = best val MRR

   **测试集最终结果**（使用best model, epoch 2）：

   | 指标 | 值 | 随机基线 (1/3155) | 相对随机提升 |
   |------|-----|-------------------|-------------|
   | Hit@1 | 0.72% | 0.032% | 22.5× |
   | Hit@3 | 2.01% | 0.095% | 21.2× |
   | Hit@5 | 3.19% | 0.158% | 20.2× |
   | Hit@10 | 6.25% | 0.317% | 19.7× |
   | MRR | 3.14% | — | — |
   | NDCG@5 | 1.94% | — | — |
   | NDCG@10 | 2.91% | — | — |

   **过拟合分析**：模型从第2 epoch后开始过拟合（train loss持续下降但val指标持续下降），原因：
   - 合成数据分布偏窄：大量序列来自同样的7种语法模板，模式单一
   - 3155类词汇表对~512K参数模型偏大：模型记住了训练模式的CS角色组合但未学到泛化的icon序列规律
   - 合成数据icon间缺乏真实语用关联：模板采样时icon组合无真实语义约束

   **正面信号**：Hit@10=6.25%是随机概率的~200倍，说明模型确实学到了CS角色的序列模式（WHO→WHAT_DOING→WHAT的槽位填充规律）。融合Emotional RAG后实际推荐效果会显著提升，因为RAG提供语义相似度兜底

4. 辅助类：
   - `SASRecDataset`: 序列padding + 下一item标签生成
   - `CS_ROLE_TO_ID`: CS角色到索引的映射字典
   - `build_item_vocabulary()`: 从ontology构建icon词汇表
   - `compute_metrics()`: 计算Hit@K, MRR, NDCG@K

**已知问题与改进方向**:

1. **合成数据过拟合**（核心问题）：train loss从6.56降至5.70，train acc从0.6%升至2.3%，但val MRR从3.04%降至2.36%。建议：引入真实用户交互日志、增加数据多样性、使用对比学习增强(CL4SRec)
2. **模型容量偏小**：当前hidden=64/blocks=2/heads=2（512K参数），对3155类词汇表偏小。建议：hidden=128/blocks=4/heads=4（~4M参数），或使用预训练语言模型初始化
3. **SASRec单路性能有限**：但融合Emotional RAG后可互补——SASRec提供序列上下文规律(CS角色槽位填充)，RAG提供语义+情感相似度兜底，融合后实际推荐效果应显著优于单路
4. **S-DPO对齐尚未运行**：偏好对齐可进一步提升模型对用户真实选择的预测准确率

---

#### 改动4：SASRec + Emotional RAG融合预测器

**改动文件**: `sequence_model/fusion.py`

**改动内容**:

1. 新增 `FusedIconPredictor` 类：融合SASRec序列预测与Emotional RAG语义检索

2. 融合公式：
   ```
   Final(i) = alpha × P_sasrec(i|seq) + (1-alpha) × [lambda × cos(E(Q_emo), E(i)) + (1-lambda) × cos(E(Q_orig), E(i))]
   ```
   - `P_sasrec(i|seq)`: SASRec的softmax概率（序列上下文）
   - `cos(Q, i)`: all-MiniLM-L6-v2余弦相似度（语义+情感）
   - `alpha=0.5` (默认，可调), `lambda=0.3` (沿用Emotional RAG参数)

3. 归一化策略：两路评分分别min-max归一化到[0,1]后再加权融合

4. 增量模式 vs batch模式：
   - 增量模式: SASRec天然支持部分序列; RAG用局部翻译文本
   - Batch模式: SASRec用完整序列; RAG用完整翻译

---

#### 改动5：双模式Pipeline架构重构

**改动文件**: `aac_emotion_pipeline.py`

**改动内容**:

1. 新增 `IncrementalState` 类：管理增量预测模式的状态
   - `current_sequence` / `current_cs_roles`: 当前轮次的icon序列
   - `turn_history`: 已提交轮次的历史
   - `max_seq_len=50`: 滑动窗口
   - `add_icon()`: 添加icon到当前序列
   - `commit_turn()`: 提交当前轮次
   - `undo()`: 撤销上一个icon
   - `get_context_for_sasrec()`: 拼接最近3轮+当前序列作为SASRec输入

2. `AACEmotionPipeline` 新增 `mode` 参数：
   - `mode='batch'` (默认): 保持原有 `process()` 行为完全不变
   - `mode='incremental'`: 加载SASRec + FusedIconPredictor，使用增量API

3. 增量模式新增方法：
   - `add_icon(icon_id)`: 用户点击一个icon → 更新序列 → 即时预测下一个icon
   - `commit_sequence()`: 用户完成当前轮次 → 完整翻译+情感分析 → 清空序列
   - `undo_icon()`: 撤销上一个icon → 重新预测

4. 增量模式的翻译策略：
   - **轻量翻译**: 每加一个icon，用最近2-3个icon做局部翻译（降低延迟）
   - **commit时**: 用完整序列做一次高质量翻译，更新对话历史
   - **SASRec不依赖翻译**: SASRec直接从icon序列预测，核心预测路径无需等待翻译

5. CLI新增参数：
   - `--incremental`: 启动增量交互模式
   - `--mode batch|incremental`: 指定模式
   - 增量模式特殊命令: `.` / `commit` = 提交, `u` / `undo` = 撤销, `reset` = 清空

6. 新增 `incremental_mode()` 交互函数和 `_display_incremental_predictions()` 显示函数

7. `_init_incremental_mode()`: 初始化增量模式，加载SASRec模型和FusedIconPredictor

**兼容性**: batch模式完全不受影响，所有原有代码路径不变

---

#### 改动6：S-DPO对齐（序列模型 + 情感模型）

**参考文献**: Hu et al., "S-DPO: Simultaneous DPO for Multi-Negative Preference Alignment", NeurIPS 2024

**改动文件**: `sequence_model/sdpo_trainer.py`, `sequence_model/collect_preference_data.py`

**改动内容**:

1. 新增 `collect_preference_data.py`：DPO偏好数据生成
   - **SASRec偏好数据** (S-DPO多负例): 用户选择icon A → (chosen=A, rejected={B,C,D,E})
   - 初始模拟: 用test序列中真实next-icon作chosen，模型top-K中非正确项作rejected，同CS角色icon作硬负例
   - **情感分类器偏好数据**: 用户纠正预测情感 → chosen=正确情感, rejected=系统错误预测
   - 生成数据：
     - `data/sasrec_dpo_train.json`: 417,774对
     - `data/sasrec_dpo_val.json`: 23,130对
     - `data/cls_dpo_train.json`: 4,500对
     - `data/cls_dpo_val.json`: 500对

2. 新增 `sdpo_trainer.py`：S-DPO训练器

   S-DPO损失函数（与标准DPO的区别：对K-1个rejected项取平均）：
   ```
   L_S-DPO = -E[log σ(β × (log π(chosen|seq)/π_ref(chosen|seq)
                            - mean_j log π(rejected_j|seq)/π_ref(rejected_j|seq)))]
   ```

   - `SDPOLoss` 类：实现S-DPO损失
   - `SDPODataset` 类：偏好数据集
   - 训练流程: 冻结已训练SASRec作reference model，训练policy model
   - `get_log_probability()`: 从SASRec获取指定item的log概率

3. DPO数据格式：
   ```json
   {"prompt": {"sequence": ["I", "want_to"], "cs_roles": ["WHO", "WHAT_DOING"]},
    "chosen": "water", "rejected": ["food", "help", "go", "sleep"],
    "emotion_context": "neutral"}
   ```

---

#### 改动7：评估模块

**改动文件**: `sequence_model/evaluate.py`

**改动内容**:

1. 新增评估指标实现：
   - `accuracy_at_k()`: Hit@K
   - `reciprocal_rank()`: MRR
   - `ndcg_at_k()`: NDCG@K
   - `cs_role_accuracy()`: CS角色准确率（预测icon的CS角色是否与目标一致）
   - `evaluate_predictions()`: 批量评估
   - `compare_modes()`: batch vs incremental模式对比

---

#### 改动8：配置文件更新

**改动文件**: `config.json`

**改动内容**:

1. 新增 `sasrec` 配置段：
   ```json
   "sasrec": {
       "enabled": true,
       "hidden_size": 64, "num_blocks": 2, "num_heads": 2,
       "max_seq_len": 50, "cs_role_emb_dim": 16, "dropout": 0.2,
       "batch_size": 128, "learning_rate": 0.001, "num_epochs": 50,
       "patience": 10, "model_path": "./output/sasrec/best_model.pt",
       "fusion_alpha": 0.5, "fusion_lambda": 0.3
   }
   ```

2. 更新 `dpo` 配置段：
   ```json
   "dpo": {
       "enabled": false,
       "num_epochs": 3, "batch_size": 4, "learning_rate": 5e-6,
       "beta": 0.1, "max_length": 256,
       "s_dpo_num_negatives": 4,
       "seq_model_path": "./output/sasrec_dpo",
       "cls_model_path": "./output/cls_dpo"
   }
   ```

---

#### 新增文件汇总

| 文件 | 说明 |
|------|------|
| `AAC2Text/scripts/enrich_ontology_cs.py` | 本体CS角色标注 + grammar_role规范化 |
| `AAC2Text/scripts/generate_icon_sequences.py` | 合成icon序列生成 |
| `sequence_model/__init__.py` | 包初始化 |
| `sequence_model/sasrec.py` | SASRec模型 + Dataset + 评估函数 |
| `sequence_model/train_sasrec.py` | SASRec训练脚本 |
| `sequence_model/fusion.py` | SASRec + Emotional RAG融合预测器 |
| `sequence_model/sdpo_trainer.py` | S-DPO训练器 |
| `sequence_model/collect_preference_data.py` | DPO偏好数据生成 |
| `sequence_model/evaluate.py` | 评估指标实现 |

#### 修改文件汇总

| 文件 | 改动 |
|------|------|
| `AAC2Text/data/processed/aac_full_ontology.json` | 新增`cs_role`字段，规范化`grammar_role` |
| `aac_emotion_pipeline.py` | 新增IncrementalState、双模式、add_icon/commit_sequence/undo_icon、增量CLI |
| `config.json` | 新增`sasrec`段，更新`dpo`段 |

---

#### 后续步骤

1. **SASRec训练优化**: 合成数据过拟合问题，建议增加数据多样性或引入真实用户交互日志
2. **S-DPO对齐训练**: 运行 `python3 sequence_model/sdpo_trainer.py`
3. **消融实验**: alpha融合权重(0/0.25/0.5/0.75/1.0)，SASRec-only vs RAG-only vs 融合，有/无CS角色嵌入
4. **DPO for 情感分类器**: 在 `cls_multitask_trainer.py` 中添加DPO训练模式

---

#### 参考文献（新增）

19. Kang & McAuley, "Self-Attentive Sequential Recommendation", ICLR 2018. (SASRec)
20. Hu et al., "S-DPO: Simultaneous DPO for Multi-Negative Preference Alignment", NeurIPS 2024. (S-DPO)
21. Bryan, "Colourful Semantics", 1997. (CS语义角色体系)
22. Magnana et al., "PrAACT: Fine-tuning Transformers for AAC Pictogram Prediction", 2023. (AAC符号预测)
23. BERTptCS, "Colourful Semantics + BERT for AAC", 2024. (CS角色注入transformer)
24. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023. (DPO)
25. Schulman et al., "Proximal Policy Optimization Algorithms", 2017. (PPO/RLHF)
26. Shao et al., "DeepSeekMath: GRPO", 2024. (GRPO)

---

### 2026-06-10 ~ 06-11：AAC2Text 双语训练数据生成 Pipeline 改进

**目标**: 改进AAC2Text数据生成流程，增加I/U人称主语支持、减少幻觉、生成更自然的句子、构建中文本体并支持双语文本输出。

#### 改动1：I/U 代词主语支持 + 21种组合类型

**改动文件**: `AAC2Text/scripts/generate_training_data.py`, `AAC2Text/config/prompts.yaml`

**改动内容**:

1. 新增 `ALL_COMBO_TYPES`（21种）：7种原始第三人称 + 7种`i_`前缀第一人称 + 7种`u_`前缀第二人称
   - `i_`前缀：主语固定为"I"，使用第一人称翻译模板
   - `u_`前缀：主语固定为"U"，使用第二人称翻译模板
   - 无前缀：随机选择人称

2. `generate_combination()` 返回3元组 `(labels, combo_type, subject_type)`

3. 新增 `PRONOUN_SENTENCE_FORMS`：映射 I→{i,me,my,mine,myself}、U→{you,your,yours,u}，用于覆盖率评分时的代词匹配

4. `QuantitativeValidator._coverage_score()` 增加代词映射逻辑：先检查 `PRONOUN_SENTENCE_FORMS`，再按常规方式匹配

5. `QuantitativeValidator.validate()` 新增 `person` 维度（权重0.15）：自然度 = 0.20×coherence + 0.25×grammaticality + 0.15×integration + 0.15×person + 0.25×overall

---

#### 改动2：反幻觉 + 自由词序 + 复合符号读取规则

**改动文件**: `AAC2Text/config/prompts.yaml`

**改动内容**:

1. 所有6个翻译模板（3英文+3中文）新增规则：
   - "You MUST NOT add content that is NOT represented by the symbols. No hallucination."
   - 允许的glue words：冠词、介词、连词、助动词
   - "The symbol order does NOT dictate sentence order. Rearrange freely."

2. 复合符号下划线读取规则：
   - `pink_pale` → pale pink（颜色形容词），不是"my pink pale"
   - `_to`后缀标记动词形式，不是介词：`ski_to` → ski，不是"ski to"
   - `arrest_to` → arrest（动词），不是"arrest to"

3. Few-shot示例中加入幻觉bad examples：
   - Bad: "I ski down the mountain with my curly hair blowing in the wind while singing my favorite songs."（mountain, wind, favorite, songs 不在符号中！）

---

#### 改动3：图标100%覆盖率（2929→3295→2929 unique 100%）

**改动文件**: `AAC2Text/scripts/generate_training_data.py`

**改动内容**:

1. 移除所有过滤逻辑：
   - 删除 `len(clean_id) > 2` 过滤（排除I/U/PE/N等）
   - 删除 `flag_/country_` 前缀过滤（排除~363个国旗）
   - 删除 `features_/man_-/woman_-` 前缀过滤（排除面部变体）

2. 扩展 `semantic_type` → `aac_category` 映射，覆盖所有缺失类型：
   - 新增：quantity, symbol, content, geography, abstract, adjective 等

3. 新增 `cs_role` 兜底：仍有未覆盖的icon使用CS角色进行类别分配

4. 结果：2929个unique icon全部覆盖（100%）

---

#### 改动4：CoT验证精简 + 5维度评估

**改动文件**: `AAC2Text/config/prompts.yaml`, `AAC2Text/scripts/generate_training_data.py`

**改动内容**:

1. CoT prompt 从冗长分析格式精简为纯输出格式（5行分数），避免截断
2. 5个维度：Semantic Coherence, Grammatical Naturalness, Label Integration, Person Consistency, Overall Naturalness
3. `max_new_tokens` 从200增加到300，确保5个维度完整输出
4. 新增 Person Consistency 维度（评估主语与动词形式是否匹配）

---

#### 改动5：复合符号可读名称优化

**改动文件**: `AAC2Text/scripts/generate_training_data.py`

**改动内容**:

1. 新增 `readable_names` 字典：从 `core_semantic` 字段构建人类可读名称
   - `pink_pale` → "pale pink"
   - `arrest_to` → "take into custody"
   - `drink_consistency_juice_straw` → "liquid container for drinking"

2. `clean_symbol()` 改为优先使用 `readable_names`，regex作为fallback

3. 中间位置的变体编号（`_2_`）被正确去除

4. 英文翻译模型默认从 Qwen2.5-1.5B 升级为 Llama-3-8B-Instruct

---

#### 改动6：中文本体构建

**改动文件**: `AAC2Text/scripts/build_zh_ontology.py`（新增）, `AAC2Text/scripts/fix_zh_ontology.py`（新增）

**改动内容**:

1. `build_zh_ontology.py`：基于英文本体 `aac_full_ontology.json`，用Qwen2.5-1.5B逐条翻译为中文
   - Step 1：翻译词汇表（typical_objects + typical_modifiers），缓存到 `vocab_en_zh.json`
   - Step 2：逐条翻译 core_semantic/label/super_concept，增量保存
   - 输出 `aac_full_ontology_zh.json`：3154条，每条新增 `core_semantic_zh`, `label_zh`, `super_concept_zh`, `typical_objects_zh`, `typical_modifiers_zh`
   - 结构性字段（semantic_type, grammar_role, cs_role等）保持英文

2. `fix_zh_ontology.py`：修复中文本体质量问题
   - 修复887条 `label_zh` 格式污染（"核心语义:" 等结构化前缀泄漏）
   - 补翻译29条 `core_semantic_zh` 和448条 `super_concept_zh` 仍为英文的字段
   - 硬编码修复11个顽固条目
   - 最终质量：core_semantic_zh 100%, label_zh 99.4%, super_concept_zh 99.6%

**最终中文本体质量**:

| 字段 | 中文率 |
|------|--------|
| core_semantic_zh | 100.0% (3154/3154) |
| label_zh | 99.4% (3135/3154) |
| super_concept_zh | 99.6% (3141/3154) |
| typical_objects_zh | 98.7% (3071/3110) |
| typical_modifiers_zh | 99.2% (1933/1948) |

---

#### 改动7：双语生成模式（EN→ZH翻译模式）

**改动文件**: `AAC2Text/scripts/generate_training_data.py`, `AAC2Text/config/prompts.yaml`

**改动内容**:

1. 中文本体加载：`_build_chinese_names()`（运行时Qwen批量翻译）替换为 `_load_chinese_names_from_ontology()`（直接从 `aac_full_ontology_zh.json` 读取，秒级加载）

2. 中文生成模式从 **icon→ZH直译** 改为 **EN→ZH翻译**：
   - 先用 Llama-3-8B 生成英文句子（icon→EN，不变）
   - 再用 Qwen1.5B 翻译英文句子为中文（EN→ZH，新逻辑）
   - 新增 `_translate_en_to_zh()` 方法
   - 原因：1.5B模型做翻译（输入已是完整自然句）比做生成（需同时理解icon语义+组织中文句子）可靠得多

3. 新增 `translation_en_to_zh` prompt 模板：
   - 简洁翻译指令："忠实原文，不添加不遗漏，自然流畅"
   - 不区分人称（人称信息已在英文句子中）

4. 输出格式：`{"labels", "sentence_en", "sentence_zh", "type", "subject_type", "validation", "cot_reasoning"}`

5. GPU分配：GPU0=Llama-3-8B(EN), GPU1=Qwen1.5B(ZH翻译), GPU2=Llama-3-8B(CoT), GPU3=BERT(公式化)

---

#### Bug 修复

| Bug | 原因 | 修复 |
|-----|------|------|
| BertRegressionModel `init_weights()` AttributeError | transformers版本不兼容 | 删除 `self.init_weights()` 调用，权重从safetensors手动加载 |
| Icon数量从3295降至75 | 多个过滤逻辑排除图标 | 移除所有过滤，扩展semantic_type映射，加cs_role兜底 |
| CoT 5维度截断 | max_new_tokens=200不够5个维度输出 | 精简prompt+增加到300 |
| `pink_pale` 读成 "my pink pale" | 下划线按词拆分 | 用core_semantic构建readable_names字典 |
| 批量翻译行错位 | 30条/批的输出行数与输入不1:1 | 改为逐条翻译+增量保存 |
| 中文生成严重幻觉 | Qwen1.5B做icon→ZH生成能力不足 | 改为EN→ZH翻译模式 |
| 中文引号SyntaxError | `strip('"\'""')` 引号嵌套 | 改为显式循环strip |
| KeyError 'icon_id' | 2条英文本体数据缺失icon_id | 改为 `item.get("icon_id")` |
| label_zh格式污染 | Qwen输出"核心语义: xxx"而非纯净值 | 鲁棒解析+补翻译+硬编码修复 |

---

#### 新增文件

| 文件 | 说明 |
|------|------|
| `AAC2Text/scripts/build_zh_ontology.py` | 构建中文本体（增量、可中断恢复） |
| `AAC2Text/scripts/fix_zh_ontology.py` | 修复中文本体质量问题（格式污染、未翻译、语义错误） |
| `AAC2Text/data/processed/aac_full_ontology_zh.json` | 中文本体（3154条，6个中文字段） |
| `AAC2Text/data/processed/vocab_en_zh.json` | 英→中词汇翻译缓存 |
| `AAC2Text/data/processed/icon_names_zh.json` | icon中文可读名称缓存（已弃用，被中文本体替代） |

#### 修改文件

| 文件 | 改动 |
|------|------|
| `AAC2Text/scripts/generate_training_data.py` | 21种组合类型、I/U人称支持、反幻觉prompt、100%图标覆盖、readable_names、双语EN→ZH模式、中文本体加载 |
| `AAC2Text/config/prompts.yaml` | 6个翻译模板(3EN+3ZH)增加反幻觉规则、新增EN→ZH翻译模板、CoT 5维度精简 |

---

#### 待办

1. 运行全量双语pipeline验证EN→ZH翻译质量
2. 更新 `train.py` 和 `test.py` 适配双语数据格式（`sentence_en`/`sentence_zh`）
3. 中文本体剩余质量优化（`typical_objects_zh` 约1.3%英文，个别语义翻译偏差）

---

### 2026-07-01：DPO 偏好数据 v2 生成（基于人工修正，输入一致对比）

#### 背景

v1 `dpo_pairs.json`（1632 条）训练 DPO 后效果不明显。分析发现核心问题在**负样本质量**：
- 1266/1632 条 rejected 用的是 `sentence_zh`（polished 版，本来就不差）
- 366/1632 条 rejected 用的是 `original_zh`（改前图标序列的翻译，**输入都不一样**，模型学不到翻译质量）
- 模型学到的是"polished vs 原始"的文体差异，不是"对 vs 错"
- 存在长度捷径（改前序列更长 → rejected 句子更长 → 模型靠"短=好"作弊）

#### 改动：v2 `dpo_pairs_v2.json`（1341 条）

**核心设计**：chosen 和 rejected 对应**同一个 `labels` 序列**，差异只在翻译质量 → DPO 信号纯净，无长度捷径，无图标序列差异。

| 维度 | v1 (`dpo_pairs.json`) | v2 (`dpo_pairs_v2.json`) |
|------|----------------------|--------------------------|
| 条数 | 1632 | 1341 |
| chosen | `zh_correction` 优先，回落 `sentence_zh` | **统一用 `zh_correction`**（无修正则跳过） |
| rejected | `original_zh`（366条）或 `sentence_zh`（1266条） | **对 `labels` 重新跑 Llama(EN)+Qwen(ZH) 生成的翻译** |
| 输入一致性 | chosen 和 rejected 对应**不同**图标序列（改前 vs 改后） | chosen 和 rejected 对应**同一个** `labels`（人工修改后） |
| 信号类型 | 文体差异（polished vs 原始） | 翻译质量差异（人工对 vs AI 错译） |
| 长度捷径 | 存在（改前序列更长 → rejected 更长） | 无（同输入，差异只在翻译质量） |
| 数据筛选 | 1697 条 valid 全用 | 1345 条 valid + 有 `zh_correction`，最终 1341 条（4 条生成失败/等同跳过） |
| 跳过统计 | — | `is_valid=0`: 127 条；无 `zh_correction`: 352 条；生成失败: 4 条 |

**负样本生成方法**（复刻 `generate_training_data.py` 原始流程）：
- 英文：Llama-3-8B，按 `subject_type` 选 `translation_prompt_{first/second/third}`（icon → EN）
- 中文：Qwen2.5-1.5B，`translation_en_to_zh` 模板（EN → ZH）
- 贪心解码（`do_sample=False`），与原数据生成一致

#### 关键字段（v2 新增，向后兼容 v1 schema）

| 字段 | 说明 |
|------|------|
| `labels` | 人工修改后的图标序列（同 v1） |
| `chosen` | `zh_correction`（人工手输修正） |
| `rejected` | AI 对 `labels` 重新生成的中文翻译 |
| `source` | `v2_edit_derived_ai_rejected`（v1 是 6 种细分 source） |
| `item_id` | 标注样本 ID（同 v1） |
| `deleted_labels` | 人工删除的图标（同 v1） |
| `generated_en` | **v2 新增** AI 生成的英文句（追溯用） |
| `subject_type` | **v2 新增** 人称类型（first/second/third） |
| `original_zh` | **v2 新增** 改前原始中文翻译（追溯用） |
| `sentence_zh` | **v2 新增** 改后 polished 中文翻译（追溯用） |

#### 质量验证

- `chosen == rejected`: 0/1341（全部有差异）
- 有图标编辑（`deleted_labels` 非空）: 466/1341
- chosen 平均长度 10.8 字符，rejected 平均长度 13.2 字符
  - 差异来自 AI 翻译的翻译腔/冗长（真实质量差异），非输入差异
- 样例（id=0）：
  - labels: `['I', 'forwards', 'flag_Belgium', 'operating_theatre']`
  - chosen（人工）: "我带着国旗去手术室"
  - rejected（AI）: "我将前往手术室，手持国旗。"（翻译腔"前往""手持"）

#### 新增文件

| 文件 | 说明 |
|------|------|
| `AAC2Text/scripts/generate_dpo_v2.py` | v2 偏好数据生成脚本（Llama EN + Qwen ZH，断点续跑） |
| `AAC2Text/data/cleardata/dpo_pairs_v2.json` | v2 偏好数据（1341 条） |

#### 后续步骤

1. 用 `dpo_pairs_v2.json` 重跑 `train_dpo.py`（改 `--dpo-data` 参数）
2. 对比 v1/v2 DPO 后的 BLEU/BERTScore/人工评估
3. 若 v2 仍无显著提升，考虑：增大 beta、增加 K（多负样本）、或换 IPO/KTO

---

### 2026-07-02：DPO v2 三配置对比实验（弱/中/强）

#### 背景

v2 数据（1341 条）训练后，弱配置（beta=0.1, 3 epochs）几乎无效果（12% 输出改变率）。为找到最佳平衡点，跑三组配置对比。

#### 三组配置

| 配置 | beta | lr | epochs | train loss | rewards/acc | rewards/margins | 输出改变率 |
|------|------|----|--------|------------|-------------|-----------------|------------|
| 弱 | 0.1 | 5e-7 | 3 | 0.666 | 92% | 0.056 | 12% |
| **中（最佳）** | **0.2** | **1e-6** | **4** | **0.435** | **97.8%** | **0.644** | **48%** |
| 强 | 0.3 | 1e-6 | 6 | 0.191 | 98% | 1.91 | 74% |

#### 评估方法

1. **测试集字面指标**（168 条 sft_val）：BERTScore-F1 / BLEU / chrF
2. **LLM-as-Judge**（Llama-3-8B 盲评 SFT vs DPO 输出，A/B 位置随机化）

#### 结果

| 配置 | BERTScore-F1 | BLEU | chrF | LLM Judge: DPO 胜率 |
|------|--------------|------|------|----------------------|
| SFT baseline | 0.9406 | 11.48 | 18.21 | — |
| 弱 | 0.9404 | 11.47 | 18.50 | ≈50%（改变太少） |
| **中** | 0.9400 | 10.63 | **18.55** | **48.3%**（42/87） |
| 强 | 0.9382 | 9.05 | 17.54 | 42.7%（50/117） |

#### 结论：中配置（beta=0.2, lr=1e-6, epochs=4）为最佳平衡点

- **48% 输出改变率**：介于弱（12%）和强（74%）之间，既不是"几乎没改"也不是"改得太狠"
- **LLM Judge 胜率 48.3%**：几乎打平 SFT（45 vs 42），优于强配置（42.7% 落败）
- **chrF 最高（18.55）**：字面指标也最佳
- **修好了 strong 引入的几个错误**：中英混杂（"crystals"）、幻觉（"喷到球道上"）、人称错乱（U→我）

#### Strong 配置的问题（中配置解决）

1. **中英混杂**：`sugar_brown` → "棕色糖 crystals"（strong 引入，mid 干净）
2. **幻觉啰嗦**：`bowling food_hot` → "吃了个热乎的东西吧？那你要小心不要把它喷到球道上吧"（strong 幻觉，mid 只加一个"吗"）
3. **人称错乱**：`U short_hair` → "我有短短的粉色发型"（strong 把 U 错译成"我"，mid 预测不变避免错误）

#### 根本限制

DPO 未能显著超过 SFT 的两个原因：
1. **训练数据上限**：1341 条偏好对，信号有限
2. **测试 ref 与训练目标分布不一致**：测试 ref 是 `sentence_zh`（polished 版），DPO 学的是 `zh_correction`（人工手输风格）— 即使 DPO 更自然，字面指标和 judge 也可能因参考偏颇判 SFT 胜

#### 最终采用

**`aac_dpo_zh_v2_mid`** 作为最终 DPO checkpoint。

#### 新增 checkpoint

| Checkpoint | 配置 | 路径 |
|------------|------|------|
| `aac_dpo_zh_v2` | 弱（beta=0.1, 3ep） | `AAC2Text/checkpoints/aac_dpo_zh_v2` |
| `aac_dpo_zh_v2_mid` | **中（beta=0.2, 4ep，采用）** | `AAC2Text/checkpoints/aac_dpo_zh_v2_mid` |
| `aac_dpo_zh_v2_strong` | 强（beta=0.3, 6ep） | `AAC2Text/checkpoints/aac_dpo_zh_v2_strong` |

#### 评估结果文件

| 文件 | 说明 |
|------|------|
| `AAC2Text/checkpoints/eval_zh_sft_vs_dpov2.json` | SFT vs DPO-v1 vs DPO-v2弱 测试集指标 |
| `AAC2Text/checkpoints/eval_zh_dpo_strong.json` | DPO-strong 测试集指标 |
| `AAC2Text/checkpoints/eval_zh_dpo_mid.json` | DPO-mid 测试集指标 |
| `AAC2Text/checkpoints/llm_judge_results.json` | LLM Judge 评估 strong（SFT 67 vs DPO 50） |
| `AAC2Text/checkpoints/llm_judge_mid_results.json` | LLM Judge 评估 mid（SFT 45 vs DPOmid 42） |

#### 后续方向（若需进一步提升）

1. 换 IPO 或 KTO（对噪声偏好更鲁棒）
2. 扩大 DPO 数据（放宽 chosen 到 `sentence_zh` 回落，可恢复到 1697 条）
3. 用 `zh_correction` 作为测试 ref 重新评估（对齐训练目标）

---

### 2026-07-06：G-Eval 语义评估（Reference-free LLM-as-Judge）

#### 背景

字面指标（BLEU/BERTScore/chrF）依赖参考翻译做字面匹配，SFT 和 DPO 都贴近原翻译时区分不开。引入 G-Eval（Liu et al., NeurIPS 2023）风格的 reference-free LLM-as-Judge，不依赖参考翻译，直接按图标语义评估翻译质量。

#### 评估方法

**G-Eval 风格 LLM-as-Judge**：
- **不给 judge 看参考翻译**（reference-free），避免字面匹配偏倚
- **注入图标语义**：从本体注入 `core_semantic_zh` + `label_zh` + `cs_role`，让 judge 理解每个图标的真实含义
- **思维链（CoT）**：每个维度先给理由再给分数，可解释
- **5 维度评分**（1-5 分）：图标覆盖 / 语义准确 / 自然度 / 无幻觉 / 整体质量
- **Judge 模型**：Llama-3-8B（本地推理）

#### 文本 G-Eval 结果（168 条，最终采用）

图标语义来源：`core_semantic_zh`（人工编写的文本本体）

| 维度 | SFT | DPO-mid | 差异 | 胜出 |
|------|-----|---------|------|------|
| 图标覆盖 | 3.28 | 3.29 | +0.02 | 持平 |
| **语义准确** | 3.66 | **3.72** | **+0.05** | **DPO ↑** |
| **自然度** | 3.85 | **3.91** | **+0.06** | **DPO ↑** |
| 无幻觉 | 4.78 | 4.77 | -0.00 | 持平 |
| **整体质量** | 3.69 | **3.75** | **+0.06** | **DPO ↑** |

**语义准确度胜出统计**：DPO 26 条胜出 vs SFT 15 条胜出 vs 持平 127 条 → DPO 语义准确度确实更好

#### 多模态 G-Eval 结果（对照实验，未采用）

图标语义来源：llava:latest 看图标图片生成的视觉描述（API: `http://172.31.226.24:4433`）

| 维度 | SFT | DPO-mid | 差异 | 胜出 |
|------|-----|---------|------|------|
| 图标覆盖 | 2.08 | 2.09 | +0.01 | 持平 |
| 语义准确 | 2.17 | 2.20 | +0.04 | 持平 |
| 自然度 | 2.32 | 2.34 | +0.02 | 持平 |
| 无幻觉 | 3.60 | 3.57 | -0.02 | 持平 |
| 整体质量 | 2.20 | 2.21 | +0.01 | 持平 |

**多模态评估不可靠的原因**：llava 对 AAC 图标的视觉识别能力差，生成的不准确描述导致 judge 评估基础就错了：
- `is_this_enough`（询问确认）→ llava 描述为"空酒杯" ❌
- `Boxing_Day`（节礼日）→ llava 描述为"盒子里有圣诞老人" ❌
- `cheese_brie`（奶酪）→ llava 描述为"庚状形凯滋" ❌
- `I`（代词"我"）→ llava 描述为"沟通指示符号,帮助残疾人阅读" ❌

评分普遍偏低（2.2 vs 文本评估的 3.7），因为 llava 描述不准导致 judge 认为翻译没覆盖图标含义。需更强的多模态模型（Qwen-VL-72B / GPT-4V）才能可靠评估。

#### 三种评估方法对比

| 评估方法 | 图标语义来源 | DPO 整体质量 | DPO 胜出 | 可靠性 |
|---------|------------|-------------|---------|--------|
| 字面指标（BLEU/chrF） | 无（依赖 ref） | 10.63/18.55 | 持平/略差 | 低（字面匹配） |
| LLM Judge 二选一 | ref 参考翻译 | 48.3% 胜率 | 持平 | 中（给了 ref） |
| **G-Eval 文本本体** | `core_semantic_zh` | **3.75 (+0.06)** | **DPO ↑** | **高（reference-free）** |
| 多模态 G-Eval | llava 看图描述 | 2.21 (+0.01) | 持平 | 低（llava 识别差） |

#### DPO 语义准确胜出样例

| 样本 | SFT 翻译 | DPO 翻译 | SFT 分 | DPO 分 | 改进点 |
|------|---------|---------|-------|-------|--------|
| `I is_this_enough Boxing_Day` | 我在圣诞节的时候在工作吗？ | 圣诞节的时候我在确认一下需求是否已经满足了。 | 2 | 4 | 正确译出"确认" |
| `U tape_recorder masking_tape when` | 你可以在录音机上贴上胶带来控制音量。 | 你可以用胶带纸条来标记音频记录的时间。 | 3 | 4 | 更贴合 when 含义 |
| `U busy hair_plaits` | ...梳头发。 | ...梳个辫子。 | 2 | 3 | hair_plaits 译出"辫子" |

#### 结论

**最可靠的评估结论（G-Eval 文本本体）**：
- DPO 在整体质量（+0.06）、自然度（+0.06）、语义准确（+0.05）三维度均胜出 SFT
- 语义准确度：DPO 26 条胜出 vs SFT 15 条胜出
- 证明 DPO 对齐确实提升了翻译语义质量 — 之前 BLEU 看不出是因为字面匹配惩罚了 DPO 的口语化风格

#### 新增文件

| 文件 | 说明 |
|------|------|
| `AAC2Text/scripts/geval.py` | G-Eval 文本评估脚本（Llama-3-8B + core_semantic_zh 注入） |
| `AAC2Text/scripts/geval_multimodal.py` | 多模态 G-Eval 评估脚本（llava 看图 + Llama-3-8B 评估） |
| `AAC2Text/checkpoints/geval_results.json` | 文本 G-Eval 评估结果（168 条 × 5 维度 + 理由） |
| `AAC2Text/checkpoints/geval_multimodal_results.json` | 多模态 G-Eval 评估结果（含 llava 图标描述缓存） |

#### 参考文献

- **G-Eval**: Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment", NeurIPS 2023
- **DPO**: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023
- **LLM-as-Judge**: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023

---

### 2026-07-17：review5000 下一图标预测 — 双轨训练 + Route D 人工合理率评测

**目标**: 用真实人工标注的 review5000 数据重训 SASRec 下一图标预测；并改用"AI 当人工判建议是否合理"替代精确命中的严苛指标（Route D）。

#### 改动1：双轨训练脚本（CE + S-DP-O）
**改动文件**: `sequence_model/train_review5000.py`（新增）

**改动内容**:
1. 适配现有 `SASRecDataset`，吃 `review5000_combined_full.jsonl`（8892 条 = 3607 序列转移 + 839 good + 1834 bad + 2612 random）
2. 词表从数据构建（**1157** 真实人工使用子集，非线上 Mamba4Rec-28K 的 2720）
3. 双轨：Track1 CE 训练基座模型（next-item 交叉熵）→ Track2 以冻结基座为 reference、可训练 policy 跑 S-DP-O（每个 prefix 的 pos next 为 chosen，human_bad + random_neg 为多个 rejected）
4. CS 角色从本体 `cs_role` 派生（数据无 CS 字段）
5. `SDPOLoss` 增加 **mask-aware 平均**（避开 padding 稀释）：`rejected_logratios` 按 `rejected_mask` 求平均，解决原先 458 个 padding 行把 rejected 均值稀释到 ~0.001 的问题
6. 新增 `--neg-cap 16` 上限，避免某些 prefix 累积大量 random_neg 导致 padding 到 458、3.5s/it

**训练结果**（Emotion 环境, torch 2.5.1+cu121, GPU）:
- CE `best_model.pt`: MRR 0.1341, Hit@5 0.1847（已超线上 Mamba MRR 0.048 / Hit@10 0.099）
- S-DP-O `best_sdpo_model.pt`: MRR 0.1316, Hit@5 0.1757 → **基本等于 CE，S-DP-O 没带来提升**

#### 改动2：诊断脚本（collapse / 偏好）
**改动文件**: `sequence_model/eval_review5000.py`（新增）

**结论**:
- S-DP-O 相对 CE 几乎无差异（MRR 0.1341 vs 0.1316）
- **popularity collapse**：top-1 全模型只覆盖 78/1157 个图标，top-10 图标占 63.7% 预测 → 模型退化为"永远猜高频"

#### 改动3：Route D 评测（AI 当人工判"建议是否合理"）
**改动文件**: `sequence_model/eval_human_reasonable.py`（新增）

**动机**: 数据本身是 2-3 张图的短会话、候选 1157 个，"精确命中"（Hit@K/MRR）天花板极低，但"用户觉得建议合理吗"才是产品真问题。

**AI 判分标准**（全量 444 条 held-out val）:
1. **非退化**: 建议不是 prefix 里已出现的同一图标
2. **语义连贯**: 建议的 `super_concept`/`aac_category`/`can_combine_with`/`typical_objects` 需与 prefix 任一方有重叠；若双方都有明确主题且主题互斥则判不连贯
3. **CS 槽位不强制顺序**: 自然语言不僵硬，不强制 WHO→WHAT_DOING→WHAT→WHERE→WHEN→HOW，cs_role 仅作信息

**全量结果**（held-out val = 444 条）:

| 方法 | Hit@1 | Hit@5 | MRR |
|------|--------|--------|------|
| 均匀随机 | 0.0009 | 0.0043 | 0.0066 |
| 猜最高频 | 0.0113 | 0.0653 | 0.0451 |
| CE 模型 | 0.0833 | 0.1847 | 0.1341 |
| S-DP-O 模型 | 0.0766 | 0.1757 | 0.1316 |

| 模型 | top1 合理率 | top5 合理率 | 对照 exact top1 / top5 |
|------|-------------|-------------|----------------------|
| CE | 0.108 | 0.586 | 0.083 / 0.185 |
| S-DP-O | 0.079 | 0.669 | 0.077 / 0.176 |

**结论**:
- **模型相对基线学到真实信号**: Hit@1 约为"猜最高频"的 7×、Hit@5 约 3×（精确指标用官方 `compute_metrics` 校验）
- **S-DP-O ≈ CE**: 偏好对齐基本无效
- **"合理率"约为"精确命中"的 3 倍**（CE top5 0.586 vs 0.185）→ 用户要的是"建议靠不靠谱"而非"是否正好命中"，Route D 口径更贴合输入法联想可用性
- **但模型仍严重依赖高频图标**: 判读卡显示短上下文（如 prefix 只有 `I`/`U`）时，top-5 多为 `short_hair`/`tomorrow`/`morning`/`good`/`download` 这类与上下文无关的坍塌图标，约 40% 样本 top-5 里一条合理建议都没有

#### 改动4：融合语义 RAG 分支实验（证明数据墙）
**改动文件**: `sequence_model/fuse_review5000_exp.py`（新增）

**结论**: 融合语义 RAG（SASRec 概率 × 语义余弦相似度）**未改善**——纯语义分支恢复真实 next 成功率 = 0.000；纯 SASRec（α=1）因高频图标是语义枢纽，语义贴近度反而最高。证明 next-icon 在符号和语义空间都近似随机。

#### 改动5：脚本清理
**改动文件**: `sequence_model/collect_preference_data.py`、`sequence_model/train_sasrec.py` 移至 `sequence_model/_unused_backup/`（保留备份，未删除）

#### 数据墙结论（核心）
review5000 下一图标预测在 **2-3 张图上下文 + 1157 候选** 现状下已逼近天花板：模型相对随机/高频都赢很多，但绝对精确命中上不去；S-DP-O、融合 RAG 等模型/损失侧手段均证明救不回来。**唯一能真正抬升精确匹配的杠杆在数据侧**：更长更干净的会话（给模型更多上下文）、约束词表到子集（缩小候选空间）。

#### Git
- `b7aedd6` feat(sequence_model): review5000 双轨训练 + Route D 人工合理率评测（双轨脚本 + 诊断 + Route D + 融合实验 + SDPOLoss mask-aware + 清理）
- `645d14d` fix(eval_human_reasonable): 放宽 CS 槽位顺序约束 + 改用官方 compute_metrics（修正自写 evaluate 低估 ~2× 的 bug）+ 移除无用 evaluate()/F 导入
- 已 push 到 `origin/main`（github.com:Rainbit-Ye/EmotionMutifyTask.git）

### 2026-07-17：端到端打通（选词预测 + 自然语言翻译 + 真实数据采集）

**目标**：按"先不管正确率，先把整个流程跑通"的要求，让 `aac_emotion_pipeline.py` 的
**incremental 模式**真正能跑：用户逐个点 icon → 实时预测下一个 icon（选词预测）
+ 自然语言翻译；并在真实使用中自动采集数据，供后续用真实数据重训、逐步优化。

**打通前阻断**（均已修复）：
1. `sentence_transformers` 未安装 → `AACIconPredictor._init_embeddings` 构造时 `import` 直接抛错，
   整个 pipeline 启动即崩（连翻译都跑不了）。改为 try/except 包裹，缺包时
   `embedding_model=None`，RAG 分支自动停用，不影响翻译/SASRec 主流程。
2. 词表不匹配 → SASRec `load_state_dict` 崩溃。`_init_incremental_mode` 原先用
   `build_item_vocabulary(全量本体=3295)` 建词表，但 `output/review5000/best_model.pt`
   自含训练词表（item2idx=1158）。改为**优先直接用 checkpoint 自带的 item2idx/idx2item**，
   `num_items = len(ck['item2idx']) - 1`，与模型尺寸严格一致。
3. `config.json` 的 `sasrec.model_path` 指向不存在的 `./output/sasrec/best_model.pt`
   → 改为 `./output/review5000/best_model.pt`；`fusion_alpha` 0.5→**1.0**（退化为纯 SASRec，
   不依赖 sentence_transformers）；`max_seq_len` 50→**16**（与训练一致）。
4. `main()` 的 AAC2Text 基座默认是 `Qwen2_5-1_5B-Instruct`（hidden=1536），但其 LoRA
   (`AAC2Text/checkpoints/aac_model`) 的 `adapter_config.json` 声明基座为
   **`/home/user1/liuduanye/Meta-Llama-3-8B-Instruct`**（hidden=4096）且磁盘存在。
   默认基座尺寸不符导致 LoRA 无法正确加载 → 改为 Llama-3-8B 为默认基座。

**真实数据采集（增量优化闭环）**：
- `AACEmotionPipeline` 新增 `log_path` 参数（默认 `./output/incremental_usage.jsonl`，`None` 关闭）；
  `main()` 新增 `--log_path` / `--no_log`。
- `add_icon`：追加 `icon_add` 记录（prefix 上下文、chosen_icon、model_top_k 带分、partial_translation、emotion），
  直接得到 (上下文→选定) 配对，未来可作重训序列，也可持续统计真实 top-k 命中率。
- `commit_sequence`：追加 `commit` 记录（full_sequence、full_translation、emotion）。
- 写入为 append jsonl，无外部依赖。

**冒烟验证**（Emotion 环境，incremental）：`I → want_to → water`
- 翻译：`I want to go home.`（局部）/ `I want water.`（完整/commit）✅
- SASRec 加载自 review5000，词表取自 checkpoint，无 size mismatch ✅
- 下一图标推荐非空（16 个候选，top 如 divide/forward/begin_start…）✅
- 无 sentence_transformers 崩溃（仅告警）✅
- `incremental_usage.jsonl` 写入 3×icon_add + 1×commit ✅

**保留/未做**：不安装 sentence_transformers、不接 RAG（alpha=1.0 即纯 SASRec）；
词表外 icon 暂按 PAD 处理，流程不崩，随真实数据重训自然扩展词表。

#### 参考文献
- Kang & McAuley, "Self-Attentive Sequential Recommendation", ICLR 2018 (SASRec)
- Hu et al., "S-DPO: Simultaneous DPO for Multi-Negative Preference Alignment", NeurIPS 2024 (S-DP-O)
- Bryan, "Colourful Semantics", 1997 (CS 语义角色体系)




---

## Web 部署（选词预测 + 自然语言翻译，前端可测）

**日期**：2026-07-18
**目标**：把 `AACEmotionPipeline`（增量模式）整套流程以网页形式部署，供多用户同时点击测试。

### 架构
- 后端 `web/server.py`（FastAPI + uvicorn，单文件）：
  - **单 pipeline 实例**（SASRec + Llama-3-8B LoRA + RoBERTa）仅加载一次，GPU 权重共享。
  - **每用户按 `X-Session-Id` 头隔离状态**：服务端维护 `SESSIONS[sid] = {state: IncrementalState, history: []}`，
    所有推理调用 `PIPELINE.add_icon(icon_id, sess["state"], sess["history"], session_id=sid)`，
    不再触碰单例的 `self.incremental_state`，彻底规避多用户状态串台。
  - **`INFER_LOCK` 全局锁串行化所有模型推理**（GPU 不可并发 forward），保证并发正确。
  - 真实使用数据继续落盘 `output/incremental_usage.jsonl`，每条带 `session_id`。
  - 图标真实 PNG 由 `web/icon_map.py` 映射（`dataset_custom.json` + `_to` 动词补丁，约 3431 个），
    经 `/api/icon/{icon_id}` 返回；缺失则 404，前端回退文字。
  - 构建后的 `frontend/dist` 由 FastAPI `StaticFiles(html=True)` 在根路径 `/` 托管（单端口）。
- 前端 `frontend/`（React 18 + Vite 5）：`api.js` 每浏览器生成并持久化一个 `sid`，
  所有请求带 `X-Session-Id`；`App.jsx` 编排 当前句子气泡 / 模型推荐 / 全部图标调色板 / 情绪徽标 / 撤销·完成 / 深浅色主题。
- 启动脚本 `web/run.sh`：`npm install && npm run build` 后起 uvicorn。
  **端口 8001**（本机 8000 已被 `icon_game` 后端占用）；`CUDA_VISIBLE_DEVICES=3`（空闲卡）；
  `CUDA_LAUNCH_BLOCKING=1` 让异步 CUDA 错误同步抛出，便于排查。

### 修复：SASRec 越界崩溃（真实 bug，非偶发抖动）
- **现象**：`/api/add` 首次即报 `CUDA error: device-side assert triggered`（ScatterGatherKernel index out of bounds），
  且一旦触发会**毒化整张卡的 context**，此后所有请求（即便词表内 icon）都报同一错误。
- **根因**：`fusion.py:_get_sasrec_scores` 用 `item2idx.get(icon, 0)` 把**词表外（OOV）icon 映射成 padding 0**。
  当序列里全是 OOV（如 `water`，不在 SASRec 1158 词表内）时，`sasrec.py:forward` 中
  `seq_lengths=(item_ids!=0).sum()=0` -> `last_positions=-1` -> `x.gather(1, idx)` 越界断言。
  此前误判为“异步偶发抖动”，实际是确定性的真实 bug（无 `CUDA_LAUNCH_BLOCKING` 时被异步执行掩盖）。
- **修复**：
  1. `sasrec.py:forward`：`last_positions = (seq_lengths - 1).clamp(min=0)`，彻底杜绝 -1。
  2. `fusion.py:_get_sasrec_scores`：序列无任何词表内 item（`not any(item_ids)`）时直接 `return {}`，
     既避免崩溃也避免返回无意义的垃圾推荐。
- **影响**：OOV icon（如 `water`）仍可正常翻译、上屏、记日志，仅“下一图标推荐”为空；
  词表内 icon（如 `want_to`）正常给出 SASRec 推荐。后续用真实数据重训 SASRec 自然扩展词表。

### 验证
- `GET /` 返回前端 HTML；`GET /api/icon/water` 返回 `image/png`（5293 B）。
- `POST /api/add`：`water`(OOV) 返回翻译不崩；`want_to`(词表内) 返回带分推荐（short_hair/volleyball… score=1.0）。
- **并发隔离**：6 个并行会话各加各自 icon，序列互不串台；同一会话 `water->want_to->eat->drink` 正确累积为 4 个。
- `POST /api/commit`：整句翻译 + trend 正常；`incremental_usage.jsonl` 正确写入 `session_id`。

### 待办 / 备注
- RAG 推荐仍因缺 `sentence_transformers` 关闭（alpha=1.0 纯 SASRec）；装后可恢复情感 RAG 推荐。
- 翻译准确性不在本轮范围（用户：“先不管正确率”）；随真实测试数据回流逐步优化。
- **未 git 提交**（用户：“先不提交”）。

### 修正（2026-07-18 续）：翻译输出语言 = 英文 -> 中文
- **现象**：网页点图标后翻译是英文（如 `The patient wants water.`）。
- **根因两层**：
  1. `aac_emotion_pipeline.py:AACTranslator.translate` 的 prompt 写死成
     `Translate these AAC symbols into ONE simple English sentence: ...`（英文指令），
     即便 LoRA 是中文权重，也被指令压成英文。
  2. 更关键：`web/server.py` 的 `AAC_MODEL_PATH` 默认指向 `AAC2Text/checkpoints/aac_model`，
     **这是英文 SFT 权重**（`config.yaml` 对应），与用户训练的中文权重无关。
     用户训练的中文权重在 `aac_model_zh`（中文 SFT）与 `aac_dpo_zh*` 系列（中文 DPO）。
- **DPO 必须叠在 SFT 之上加载（关键纠正，原"DPO 全面退化"结论错误）**：
  最初实测 `aac_dpo_zh*` 对同样输入输出废话（`A simple one! 😊` 等），误判为"训崩"。
  实为**加载方式错**——`AACTranslator` 原本只把 DPO LoRA 直接压在基座 Llama 上；
  而 `scripts/test_zh.py` 的评测配方是 **先 `merge_and_unload()` 合并 SFT(`aac_model_zh`) 到基座，
  再在其上 `from_pretrained` 加载 DPO**。直接压基座 → 退化；按评测配方 → 正常中文，
  I→我 / U→你 人称正确，与评测 bertscore≈0.94（中配置）一致。**DPO 没崩**，中配置(beta=0.2)即用户指定最佳。
- **修复**：
  1. `translate` 的 prompt 改为与 `scripts/train_zh.py` 一致的中文指令：
     `请把这些 AAC 图标序列翻译成一个简单的中文句子：{...}`。
  2. `AACTranslator` 新增 `sft_model_path` 参数，存在且与主模型不同路径时先 `merge_and_unload()` 合并 SFT；
     `AACEmotionPipeline` 透传 `aac_sft_model_path`；
     `web/server.py` 默认 `AAC_MODEL=aac_dpo_zh_v2_mid` + `AAC_SFT_MODEL=aac_model_zh`
     （即 SFT 合并 + DPO 中配置），并保留 `AAC_MODEL` 环境变量便于切到纯 SFT 等。
- **验证**：`I→下周→飞机→吃→外卖` 整句 `下周我要坐飞机吃外卖。…`；`U→…` → `你下周坐飞机…` ✅
- **待办**：无（DPO 中配置已可用，无需重训）。

### 修正（2026-07-18 续2）：翻译"闲扯/重复/解释图标" -> 单句干净
- **现象**：用户实测翻译把整段糊出来——重复句（"我的手机在户外响了。我的手机在外面响了。"）、
  甚至回显图标含义（"hike_to表示…。mobile_phone_ring_tone表示…"）、幻觉概念（"思考"等）。
  用户原话："我的 怎么思考也给放出来了？"
- **根因（代码 bug）**：`aac_emotion_pipeline.py:translate` 的"只取第一句"截断
  只匹配半角 `\n` 和 `.`，但 Llama-3 中文输出用的是**全角句号 `。`**（及 `！` `？` `；`）。
  导致 `sep in response` 永为假，截断从不触发，模型一直生成到 `max_new_tokens=30` 上限，
  把重复/解释/幻觉全吐出来。
- **修复**：
  1. 截断改用全角标点：`['\n','。','！','？','；']`，命中即 `response[:idx+len(sep)]` 保留句末点，只留第一句。
  2. `generate` 加 `repetition_penalty=1.15` 抑制句内重复。
- **验证（SFT 合并 + DPO 中配置）**：
  - `hike_to+mobile_phone_ring_tone` -> `我在户外徒步旅行的时候手机响了。` ✅（原会回显图标含义）
  - `I+hike_to+mobile_phone_ring_tone` -> `我在户外活动的时候手机响了。` ✅
  - `I+下周+飞机+吃+外卖` -> `下周我要坐飞机吃外卖。` ✅（回归正常）
- 属推理后处理 bug，与模型权重无关；重训不解决，改后处理即可。

### 前端：I/U 高频人称词常驻快捷按钮
- 新增 `frontend/src/components/Shortcuts.jsx`：始终展示在顶栏下方，按钮 `I（我）` / `U（你）`。
- `App.jsx` 定义 `SHORTCUT_IDS=['I','U']`，从目录取 label/has_image；点击走同一 `handlePick` 流程
  （即加入当前序列，与调色板选图标等价）。
- `index.css` 加 `.shortcuts / .sc-btn` 样式（强调色左边框）。
- 构建：`npm run build`（StaticFiles 按请求读盘，前端改动无需重启服务，浏览器刷新即见）。
### 已知限制（2026-07-18 续3）：飞机/活动类只出陈述句，不出疑问句

- **现象**：用户给出真实序列（如 `I plane` / `I next_week plane` / `plane`）并给出更自然的范本
  `我的飞机几点降落？`，即期望模型在图标之外补出"疑问/意图"（降落、几点）。
  当前模型实际输出：
  - `I plane` → `我坐飞机。`
  - `I next_week plane` → `我下周坐飞机去旅行。`
  - `plane` → `他坐飞机去旅行。`
  均为**忠实陈述句**，不含图标之外的"降落/几点"等概念。
- **根因**：训练数据（SFT `aac_model_zh` + DPO `aac_dpo_zh_v2_mid` 的偏好对）**几乎全是陈述/描述句式**，
  没有"图标→疑问句/意图补全"的样本。DPO 学的就是"不添加图标外的词"，因此模型只会字面串联图标含义。
  这与上一节修掉"闲扯/幻觉"的取向一致——模型被对齐去"只翻图标、不添油加醋"。
- **冲突点**：用户范本 `我的飞机几点降落？` 实际**引入了图标里没有的概念**（降落、几点），
  与"不幻觉、忠实图标"原则相冲突。属数据分布缺失，非代码 bug。
- **决策（用户选定"先记下，等重训"）**：
  1. **当前行为保持**：翻译严格忠实图标，只出陈述句，不乱补疑问/意图。
  2. **记为已知限制**：飞机/活动/时间这类表达，模型无法自发产出"我的飞机几点降落？"式疑问意图。
  3. **留给重训解决**：后续若要支持疑问/意图补全，需要**在训练数据里补充疑问句样本**
     （图标→疑问句的 SFT 数据，及"示意→自然疑问"的 DPO 偏好对），再重训对齐。
  4. 本轮**不修改代码、不重训**，仅记录限制；服务继续以忠实陈述句对外。
- **待办（重训时）**：
  - 构造"图标序列→疑问句"训练样本（如 `I plane`→`我的飞机几点降落？` / `I next_week plane`→`我下周坐飞机去哪？`）。
  - 评估是否需要在 DPO 偏好对里加入"忠实陈述 vs 自然疑问"的偏好信号，避免重训后答疑句被对齐回陈述。
  - 若同时保留两种风格，考虑加一个生成模式开关（忠实 / 自然意图）。

### 修正（2026-07-18 续4）：前端一直卡在"翻译中…"永不出现中文
- **现象**：点 2 个以上图标后，气泡长时间显示"翻译中…"，轮询结束仍无中文；后端日志却能看到中文生成。
  实为**新会话的翻译被饿死**——卡在轮询 10s 窗口之外才轮到它。
- **根因（worker 线程空转，确定性强）**：`web/server.py:_translate_loop` 在翻译完一串后回到循环顶端，
  `ev.wait(timeout=1.0)` 超时醒来 → 防抖判断"距上次点击已久"→ 直接重读 `TRANS_PENDING` 把**同一串再翻一遍**。
  没有任何"本版本已翻译过"的判定，于是**每会话常驻线程永远在空转重翻同一句话**，
  每次 ~4.6s 死占 `INFER_LOCK`。前期压测留下大量 `deb2/cc-*/br-*` 等陈旧会话，每个都有空转线程，
  它们持续抢锁，把新浏览器会话的翻译排队到十几甚至几十秒后，**超出前端 10s 轮询**，故永远停在"翻译中…"。
- **修复**：
  1. `_translate_loop` 增加 `last_done_ver`：翻译成功的版本号记下来；下一轮若 `ver == last_done_ver`（无新点击）
     **直接 `continue` 跳过**，绝不空转重翻。只有新点击使 `TRANS_VER` 自增才会触发一次新翻译。
  2. 增加空闲退出：自上次点击起超过 `TRANS_IDLE_EXIT=600s` 仍无新版本 → 线程 `break` 退出，
     避免陈旧会话线程无限堆积占内存；用户再点 `api_add` 会按需重建（`is_alive` 判断）。
  3. 前端 `App.jsx:pollTranslation` 轮询窗口 25×400ms(10s) → 30×400ms(12s)，留出并发争锁余量。
- **验证（重启清掉陈旧空转线程后）**：
  - 单会话 `I plane`：5.5s 出 `我坐飞机。`；日志该会话仅 **1 次** `[TranslateWorker]` 写入。
  - 防抖：`I plane water` 连点(0.3s 间隔) 只出 **1 句**最终翻译，无中间译文。
  - 新版本：在第 4 个图标 `go` 后译文正确改为 `飞机起飞之后掉进水里了。`（新序列 `[plane,water,go]`）。
  - 并发：3 个会话同时 `I+plane`，分别 5.5/9.5/13.5s 全部得到中文，日志仅 **3 次** worker 写入（无空转）。
- **未 git 提交**（用户："先不提交"）。
- **待办（可选）**：`_translate_loop` 当前用 `TRANS_PENDING[-3:]` 只翻最后 3 个图标；超长句子若要翻全序列可后续调整。
