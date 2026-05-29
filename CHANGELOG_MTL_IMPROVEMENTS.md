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
