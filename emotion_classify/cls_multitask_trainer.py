#!/usr/bin/env python3
"""
情绪分类多任务学习训练模块

多任务架构：
1. 主任务：预测整体情绪 (main_emotion)
2. 辅助任务：预测每轮对话的情绪标签
3. 一致性约束：预测的每轮情绪分布与 main_emotion 应该一致
4. 动态惩罚：相似度低时增加样本权重

损失函数：
L = L_main + α * L_turn + β * L_consistency
其中一致性损失会对预测偏差大的样本增加惩罚
"""

import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from collections import Counter


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}

# 相似情绪分组（用于对比学习）
EMOTION_GROUPS = {
    "negative": ["anger", "disgust", "fear", "sadness"],
    "positive": ["happiness"],
    "neutral": ["neutral"],
    "surprise": ["surprise"]  # surprise独立，可能是正或负
}

# 容易混淆的情绪对
CONFUSING_PAIRS = [
    ("anger", "disgust"),
    ("sadness", "surprise"),
    ("happiness", "surprise"),
]


class MultiTaskEmotionDataset(Dataset):
    """多任务情绪分类数据集"""
    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"数据文件 {data_path} 不存在")
            return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item["conversation"]
        main_emotion = item["main_emotion"]

        # 获取下一轮情绪标签（最后一轮的情绪）
        # 训练时用前 n-1 轮预测第 n 轮
        next_emotion = conversation[-1].get("emotion", "neutral") if len(conversation) > 1 else main_emotion

        # 格式化对话并记录每轮的位置（使用前 n-1 轮进行训练）
        text, turn_positions, turn_emotions = self._format_conversation(conversation)

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()

        # 转换情绪标签
        main_label = LABEL2ID[main_emotion]
        turn_labels = [LABEL2ID[e] for e in turn_emotions]
        next_label = LABEL2ID[next_emotion]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'main_label': main_label,
            'turn_positions': turn_positions,  # 每轮最后一个token的位置
            'turn_labels': turn_labels,  # 每轮的情绪标签
            'next_label': next_label,  # 下一轮情绪标签
            'num_turns': len(turn_labels),
            'main_emotion': main_emotion
        }

    def _format_conversation(self, conversation):
        """
        格式化对话文本，返回：
        - text: 格式化后的文本
        - turn_positions: 每轮最后一个token的位置列表
        - turn_emotions: 每轮的情绪标签列表
        """
        text = ""
        turn_positions = []
        turn_emotions = []

        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            emotion = turn.get("emotion", "neutral")

            if role == "user":
                turn_text = f"User: {content}\n"
            else:
                turn_text = f"Assistant: {content}\n"

            # 记录该轮开始前的token数
            tokens_before = len(self.tokenizer.encode(text, add_special_tokens=False))
            text += turn_text
            # 该轮结束后的token数
            tokens_after = len(self.tokenizer.encode(text, add_special_tokens=False))

            # 该轮最后一个token的位置（考虑[CLS] token的位置偏移）
            # 实际位置 = tokens_after - 1 + 1 (因为第一个token是[CLS])
            turn_positions.append(tokens_after)  # 会在外部调整
            turn_emotions.append(emotion)

        return text, turn_positions, turn_emotions


def collate_fn(batch):
    """自定义批处理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    main_labels = torch.tensor([item['main_label'] for item in batch])
    next_labels = torch.tensor([item['next_label'] for item in batch])
    num_turns = [item['num_turns'] for item in batch]
    max_turns = max(num_turns)

    # 填充 turn_positions 和 turn_labels
    batch_size = len(batch)
    turn_positions = torch.zeros(batch_size, max_turns, dtype=torch.long)
    turn_labels = torch.zeros(batch_size, max_turns, dtype=torch.long)
    turn_mask = torch.zeros(batch_size, max_turns, dtype=torch.float)
    last_turn_idx = torch.zeros(batch_size, dtype=torch.long)  # 最后一轮的索引

    for i, item in enumerate(batch):
        # 调整位置（考虑[CLS] token）
        positions = [min(p + 1, 255) for p in item['turn_positions']]  # +1 for [CLS], clamp to max_length-1
        for j, (pos, label) in enumerate(zip(positions, item['turn_labels'])):
            turn_positions[i, j] = pos
            turn_labels[i, j] = label
            turn_mask[i, j] = 1.0
        # 记录最后一轮的索引
        last_turn_idx[i] = len(positions) - 1

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'main_labels': main_labels,
        'next_labels': next_labels,
        'turn_positions': turn_positions,
        'turn_labels': turn_labels,
        'turn_mask': turn_mask,
        'num_turns': num_turns,
        'last_turn_idx': last_turn_idx,
        'main_emotions': [item['main_emotion'] for item in batch]
    }


class MultiTaskEmotionClassifier(nn.Module):
    """多任务情绪分类模型"""
    def __init__(self, base_model_path, num_labels=7, lora_config=None):
        super().__init__()

        # 加载 RoBERTa 编码器
        self.roberta = RobertaModel.from_pretrained(base_model_path)

        # 应用 LoRA
        if lora_config is not None:
            self.roberta = get_peft_model(self.roberta, lora_config)

        hidden_size = self.roberta.config.hidden_size

        # 主任务分类头（整体情绪）
        self.main_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # 辅助任务分类头（每轮情绪）
        self.turn_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # 新增：下一轮情绪预测分类头
        self.next_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, turn_positions=None, turn_mask=None, last_turn_idx=None):
        """
        前向传播

        Returns:
            main_logits: 整体情绪的logits [batch_size, num_labels]
            turn_logits: 每轮情绪的logits [batch_size, max_turns, num_labels]
            next_logits: 下一轮情绪的logits [batch_size, num_labels]
            main_hidden: [CLS] hidden state
            turn_hiddens: 每轮的 hidden states
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 主任务：使用 [CLS] token
        main_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        main_logits = self.main_classifier(main_hidden)  # [batch_size, num_labels]

        # 辅助任务：使用每轮最后一个 token
        next_logits = None
        if turn_positions is not None:
            batch_size, max_turns = turn_positions.shape
            turn_hiddens = torch.zeros(batch_size, max_turns, hidden_states.size(-1), device=hidden_states.device)

            for i in range(batch_size):
                for j in range(max_turns):
                    if turn_mask[i, j] > 0:
                        pos = turn_positions[i, j].long()
                        turn_hiddens[i, j] = hidden_states[i, pos, :]

            turn_logits = self.turn_classifier(turn_hiddens)  # [batch_size, max_turns, num_labels]

            # 新增：下一轮情绪预测 - 使用最后一轮的 hidden state
            if last_turn_idx is not None:
                # 使用指定的最后一轮位置
                last_hiddens = torch.zeros(batch_size, hidden_states.size(-1), device=hidden_states.device)
                for i in range(batch_size):
                    idx = last_turn_idx[i]
                    last_hiddens[i] = turn_hiddens[i, idx]
                next_logits = self.next_classifier(last_hiddens)  # [batch_size, num_labels]
            else:
                # 默认使用 turn_hiddens 的最后一个有效位置
                last_hiddens = torch.zeros(batch_size, hidden_states.size(-1), device=hidden_states.device)
                for i in range(batch_size):
                    # 找到该样本最后一个有效的 turn
                    valid_turns = (turn_mask[i] > 0).sum().int().item()
                    if valid_turns > 0:
                        last_hiddens[i] = turn_hiddens[i, valid_turns - 1]
                next_logits = self.next_classifier(last_hiddens)  # [batch_size, num_labels]
        else:
            turn_logits = None
            turn_hiddens = None

        return main_logits, turn_logits, next_logits, main_hidden, turn_hiddens


class FocalLoss(nn.Module):
    """Focal Loss: 更关注难分类样本"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] logits
        targets: [batch_size] 类别索引
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # 正确类别的概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ContrastiveLoss(nn.Module):
    """
    对比学习损失：增强相似情绪的区分能力
    对于容易混淆的情绪对，增加惩罚
    """
    def __init__(self, confusing_pairs, temperature=0.5):
        super().__init__()
        self.confusing_pairs = confusing_pairs
        self.temperature = temperature

    def forward(self, logits, labels):
        """
        logits: [batch_size, num_classes]
        labels: [batch_size]
        """
        batch_size = logits.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=logits.device)

        # 归一化
        probs = F.softmax(logits / self.temperature, dim=-1)

        loss = 0.0
        count = 0

        for e1, e2 in self.confusing_pairs:
            idx1 = LABEL2ID[e1]
            idx2 = LABEL2ID[e2]

            # 找到真实标签为e1的样本
            mask1 = (labels == idx1)
            # 找到真实标签为e2的样本
            mask2 = (labels == idx2)

            if mask1.sum() > 0 and mask2.sum() > 0:
                # e1样本对e2的预测概率应该低
                probs_e1_to_e2 = probs[mask1, idx2].mean()
                # e2样本对e1的预测概率应该低
                probs_e2_to_e1 = probs[mask2, idx1].mean()

                # 惩罚混淆
                loss += probs_e1_to_e2 + probs_e2_to_e1
                count += 1

        if count > 0:
            loss = loss / count

        return torch.tensor(loss, device=logits.device, requires_grad=True)


class MultiTaskTrainer:
    """多任务训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 加载类别权重
        self.class_weights = self._load_class_weights()

        # 加载tokenizer
        print("加载tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(config['model']['model_path'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 配置LoRA
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias=config['lora']['bias'],
            task_type=TaskType.FEATURE_EXTRACTION  # 使用特征提取模式
        )

        # 创建多任务模型
        print("创建多任务模型...")
        self.model = MultiTaskEmotionClassifier(
            config['model']['model_path'],
            num_labels=len(EMOTION_LIST),
            lora_config=lora_config
        )
        self.model.to(self.device)

        # 打印参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    def _load_class_weights(self):
        """加载类别权重"""
        data_dir = os.path.dirname(self.config['data']['sft_train_path'])
        weights_path = os.path.join(data_dir, 'emotion_weights.json')

        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                emotion_weights = json.load(f)
            print(f"加载类别权重: {emotion_weights}")
            weights = [emotion_weights.get(emo, 1.0) for emo in EMOTION_LIST]
            return torch.tensor(weights, dtype=torch.float)
        else:
            return None

    def compute_similarity_loss(self, pred_distribution, true_distribution):
        """
        计算预测分布与真实分布的相似度损失

        使用 JS 散度（Jensen-Shannon Divergence）衡量分布差异
        差异越大，惩罚越重
        """
        # 计算预测的每轮情绪分布
        # pred_distribution: [batch_size, num_turns, num_labels]
        # true_distribution: [batch_size, num_turns, num_labels] (one-hot)

        # 平均 KL 散度
        eps = 1e-8
        pred = pred_distribution + eps
        true = true_distribution + eps

        # KL(true || pred)
        kl = true * (true.log() - pred.log())
        kl = kl.sum(dim=-1)  # [batch_size, num_turns]

        return kl.mean()

    def compute_consistency_loss(self, main_logits, turn_logits, turn_mask):
        """
        一致性损失：预测的每轮情绪众数应该与整体情绪一致

        思路：
        1. 从 turn_logits 计算每轮预测的情绪
        2. 计算非neutral的众数
        3. 与 main_logits 的预测比较，不一致则惩罚
        """
        batch_size = main_logits.size(0)

        # 获取每轮的预测
        turn_preds = torch.argmax(turn_logits, dim=-1)  # [batch_size, max_turns]

        # 计算一致性损失
        consistency_loss = 0.0
        for i in range(batch_size):
            num_turns = int(turn_mask[i].sum().item())
            if num_turns == 0:
                continue

            preds = turn_preds[i, :num_turns].cpu().numpy()
            turn_probs = F.softmax(turn_logits[i, :num_turns], dim=-1)

            # 计算非neutral的情绪分布
            non_neutral_mask = preds != 0  # 0 is neutral
            if non_neutral_mask.sum() > 0:
                # 获取非neutral的预测，计算众数
                non_neutral_preds = preds[non_neutral_mask]
                counter = Counter(non_neutral_preds.tolist())
                if counter:
                    majority_pred = counter.most_common(1)[0][0]

                    # 与主预测比较
                    main_pred = torch.argmax(main_logits[i]).item()
                    if majority_pred != main_pred:
                        # 不一致，增加惩罚
                        consistency_loss += 1.0

        return consistency_loss / max(batch_size, 1)

    def compute_dynamic_weights(self, main_logits, turn_logits, main_labels, turn_labels, turn_mask):
        """
        动态计算样本权重：相似度低的样本权重高

        返回每个样本的权重
        """
        batch_size = main_logits.size(0)
        weights = torch.ones(batch_size, device=self.device)

        with torch.no_grad():
            # 计算预测分布
            main_probs = F.softmax(main_logits, dim=-1)  # [batch_size, num_labels]
            turn_probs = F.softmax(turn_logits, dim=-1)  # [batch_size, max_turns, num_labels]

            # 获取真实标签的 one-hot
            main_one_hot = F.one_hot(main_labels, num_classes=len(EMOTION_LIST)).float()

            # 计算主任务的预测准确度
            main_preds = torch.argmax(main_probs, dim=-1)
            main_correct = (main_preds == main_labels).float()

            # 计算每轮任务的预测准确度
            turn_preds = torch.argmax(turn_probs, dim=-1)
            turn_correct = (turn_preds == turn_labels).float() * turn_mask
            turn_acc = turn_correct.sum(dim=1) / (turn_mask.sum(dim=1) + 1e-8)

            # 综合准确度
            overall_acc = 0.6 * main_correct + 0.4 * turn_acc

            # 准确度低的样本权重高
            # 使用 (1 - acc) 的平滑版本
            sample_weights = 1.0 + 2.0 * (1.0 - overall_acc)  # 权重范围 [1, 3]

        return sample_weights

    def train(self):
        """执行多任务训练"""
        cls_config = self.config['cls']

        # 加载数据集
        train_dataset = MultiTaskEmotionDataset(
            self.config['data']['sft_train_path'],
            self.tokenizer,
            max_length=cls_config['max_length']
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cls_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )

        # 验证集
        val_dataloader = None
        if os.path.exists(self.config['data'].get('sft_val_path', '')):
            val_dataset = MultiTaskEmotionDataset(
                self.config['data']['sft_val_path'],
                self.tokenizer,
                max_length=cls_config['max_length']
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cls_config['batch_size'],
                shuffle=False,
                collate_fn=collate_fn
            )

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cls_config['learning_rate'],
            weight_decay=cls_config.get('weight_decay', 0.01)
        )

        # 学习率调度器
        total_steps = len(train_dataloader) * cls_config['num_epochs']
        warmup_steps = int(total_steps * cls_config.get('warmup_ratio', 0.1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 损失函数 - 使用Focal Loss增强对难样本的关注
        if self.class_weights is not None:
            class_weights = self.class_weights.to(self.device)
        else:
            class_weights = None

        main_criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='none')
        turn_criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='none')
        next_criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='none')  # 新增
        contrastive_criterion = ContrastiveLoss(CONFUSING_PAIRS, temperature=0.5)

        # 损失权重（从配置文件读取，或使用默认值）
        loss_weights = cls_config.get('loss_weights', {})
        alpha = loss_weights.get('turn', 0.3)  # 辅助任务权重
        beta = loss_weights.get('consistency', 0.2)   # 一致性损失权重
        gamma = loss_weights.get('contrastive', 0.1)  # 对比学习损失权重
        delta = loss_weights.get('next', 0.2)  # 新增：下一轮预测损失权重

        print(f"损失权重: turn={alpha}, consistency={beta}, contrastive={gamma}, next={delta}")

        print(f"\n训练集大小: {len(train_dataset)}")
        if val_dataloader:
            print(f"验证集大小: {len(val_dataloader.dataset)}")
        print("开始多任务训练...")

        best_val_acc = 0.0

        for epoch in range(cls_config['num_epochs']):
            self.model.train()
            total_loss = 0
            total_main_loss = 0
            total_turn_loss = 0
            total_next_loss = 0  # 新增
            total_consistency_loss = 0
            total_contrastive_loss = 0
            total_correct = 0
            total_next_correct = 0  # 新增
            total_samples = 0
            hard_samples = 0  # 困难样本数

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{cls_config['num_epochs']}")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                main_labels = batch['main_labels'].to(self.device)
                next_labels = batch['next_labels'].to(self.device)  # 新增
                turn_positions = batch['turn_positions'].to(self.device)
                turn_labels = batch['turn_labels'].to(self.device)
                turn_mask = batch['turn_mask'].to(self.device)
                last_turn_idx = batch['last_turn_idx'].to(self.device)  # 新增

                # 前向传播
                main_logits, turn_logits, next_logits, _, _ = self.model(
                    input_ids, attention_mask, turn_positions, turn_mask, last_turn_idx
                )

                # 计算动态样本权重（基于相似度）
                sample_weights = self.compute_dynamic_weights(
                    main_logits, turn_logits, main_labels, turn_labels, turn_mask
                )

                # 统计困难样本
                hard_samples += (sample_weights > 1.5).sum().item()

                # 主任务损失（Focal Loss + 动态权重）
                main_loss_per_sample = main_criterion(main_logits, main_labels)
                main_loss = (main_loss_per_sample * sample_weights).mean()

                # 辅助任务损失（每轮情绪，Focal Loss）
                turn_loss_per_sample = turn_criterion(
                    turn_logits.view(-1, len(EMOTION_LIST)),
                    turn_labels.view(-1)
                ).view(main_labels.size(0), -1)
                turn_loss = (turn_loss_per_sample * turn_mask).sum() / (turn_mask.sum() + 1e-8)

                # 一致性损失
                consistency_loss = self.compute_consistency_loss(
                    main_logits, turn_logits, turn_mask
                )
                consistency_loss = torch.tensor(consistency_loss, device=self.device)

                # 对比学习损失（区分相似情绪）
                contrastive_loss = contrastive_criterion(main_logits, main_labels)

                # 新增：下一轮情绪预测损失
                next_loss = next_criterion(next_logits, next_labels).mean()

                # 总损失
                loss = main_loss + alpha * turn_loss + beta * consistency_loss + gamma * contrastive_loss + delta * next_loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # 统计
                total_loss += loss.item()
                total_main_loss += main_loss.item()
                total_turn_loss += turn_loss.item()
                total_next_loss += next_loss.item()
                total_consistency_loss += consistency_loss.item()
                total_contrastive_loss += contrastive_loss.item()

                preds = torch.argmax(main_logits, dim=1)
                total_correct += (preds == main_labels).sum().item()

                # 新增：统计下一轮预测准确率
                next_preds = torch.argmax(next_logits, dim=1)
                total_next_correct += (next_preds == next_labels).sum().item()

                total_samples += main_labels.size(0)

                # 更新进度条
                accuracy = total_correct / total_samples
                next_accuracy = total_next_correct / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'main': f'{main_loss.item():.4f}',
                    'next': f'{next_loss.item():.4f}',
                    'acc': f'{accuracy:.4f}',
                    'next_acc': f'{next_accuracy:.4f}'
                })

            # 验证
            if val_dataloader:
                val_acc, val_next_acc, val_loss = self._validate(
                    val_dataloader, main_criterion, turn_criterion, next_criterion,
                    contrastive_criterion, alpha, beta, gamma, delta
                )
                print(f"Epoch {epoch + 1} 完成 - "
                      f"训练损失: {total_loss / len(train_dataloader):.4f}, "
                      f"训练准确率: {accuracy:.4f}, "
                      f"下一轮预测准确率: {next_accuracy:.4f}, "
                      f"验证准确率: {val_acc:.4f}, "
                      f"验证下一轮准确率: {val_next_acc:.4f}, "
                      f"困难样本: {hard_samples}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch + 1, is_best=True)
            else:
                print(f"Epoch {epoch + 1} 完成 - "
                      f"平均损失: {total_loss / len(train_dataloader):.4f}, "
                      f"准确率: {accuracy:.4f}, "
                      f"下一轮预测准确率: {next_accuracy:.4f}")

            self._save_checkpoint(epoch + 1, is_epoch=True)

        self._save_final_model()
        print("多任务训练完成!")

        return self.model

    def _validate(self, val_dataloader, main_criterion, turn_criterion, next_criterion, contrastive_criterion, alpha, beta, gamma, delta):
        """验证集评估"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_next_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                main_labels = batch['main_labels'].to(self.device)
                next_labels = batch['next_labels'].to(self.device)
                turn_positions = batch['turn_positions'].to(self.device)
                turn_labels = batch['turn_labels'].to(self.device)
                turn_mask = batch['turn_mask'].to(self.device)
                last_turn_idx = batch['last_turn_idx'].to(self.device)

                main_logits, turn_logits, next_logits, _, _ = self.model(
                    input_ids, attention_mask, turn_positions, turn_mask, last_turn_idx
                )

                # 主任务损失
                main_loss = main_criterion(main_logits, main_labels).mean()

                # 辅助任务损失
                turn_loss_per_sample = turn_criterion(
                    turn_logits.view(-1, len(EMOTION_LIST)),
                    turn_labels.view(-1)
                ).view(main_labels.size(0), -1)
                turn_loss = (turn_loss_per_sample * turn_mask).sum() / (turn_mask.sum() + 1e-8)

                # 一致性损失
                consistency_loss = self.compute_consistency_loss(main_logits, turn_logits, turn_mask)
                consistency_loss = torch.tensor(consistency_loss, device=self.device)

                # 对比学习损失
                contrastive_loss = contrastive_criterion(main_logits, main_labels)

                # 下一轮预测损失
                next_loss = next_criterion(next_logits, next_labels).mean()

                loss = main_loss + alpha * turn_loss + beta * consistency_loss + gamma * contrastive_loss + delta * next_loss
                total_loss += loss.item()

                preds = torch.argmax(main_logits, dim=1)
                total_correct += (preds == main_labels).sum().item()

                # 下一轮预测准确率
                next_preds = torch.argmax(next_logits, dim=1)
                total_next_correct += (next_preds == next_labels).sum().item()

                total_samples += main_labels.size(0)

        self.model.train()
        main_acc = total_correct / total_samples
        next_acc = total_next_correct / total_samples
        return main_acc, next_acc, total_loss / len(val_dataloader)

    def _save_checkpoint(self, step, is_epoch=False, is_best=False):
        """保存检查点"""
        output_dir = self.config['model']['output_dir']
        if is_best:
            checkpoint_dir = os.path.join(output_dir, "cls_best")
        elif is_epoch:
            checkpoint_dir = os.path.join(output_dir, f"cls_checkpoint_epoch_{step}")
        else:
            checkpoint_dir = os.path.join(output_dir, f"cls_checkpoint_step_{step}")

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"检查点已保存到: {checkpoint_dir}")

    def _save_final_model(self):
        """保存最终模型"""
        output_dir = os.path.join(self.config['model']['output_dir'], "cls_final")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        self.tokenizer.save_pretrained(output_dir)
        print(f"最终模型已保存到: {output_dir}")


def run_multitask_training(config):
    """运行多任务训练的入口函数"""
    trainer = MultiTaskTrainer(config)
    model = trainer.train()
    return model


if __name__ == '__main__':
    with open('../config.json', 'r') as f:
        config = json.load(f)

    run_multitask_training(config)
