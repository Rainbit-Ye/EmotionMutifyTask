"""
共享工具模块 — 情绪分类基线实验通用组件
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# ========== 常量 ==========

EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}

# 效价值（用于趋势分析）
EMOTION_VALENCE = {
    "neutral": 0.0,
    "anger": -0.8,
    "disgust": -0.6,
    "fear": -0.7,
    "happiness": 0.9,
    "sadness": -0.8,
    "surprise": 0.2,
}

# ========== 数据集 ==========

class EmotionDataset(Dataset):
    """情绪分类数据集"""

    def __init__(self, data_path, tokenizer, max_length=256, include_context=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_context = include_context
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"数据文件 {data_path} 不存在")
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item["conversation"]
        main_emotion = item["main_emotion"]

        if self.include_context:
            text = self._format_with_context(conversation)
        else:
            text = self._format_conversation(conversation)

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = LABEL2ID[main_emotion]
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': label,
            'emotion': main_emotion
        }

    def _format_conversation(self, conversation):
        """标准格式化：每轮对话拼成文本"""
        text = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                text += f"User: {content}\n"
            else:
                text += f"Assistant: {content}\n"
        return text.strip()

    def _format_with_context(self, conversation):
        """上下文窗口格式化：拼接历史轮次 + 说话人标记"""
        text = ""
        for i, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]
            speaker = "Speaker1" if role == "user" else "Speaker2"
            text += f"{speaker}: {content} "
        return text.strip()


# ========== 工具函数 ==========

def load_class_weights(data_path):
    """加载类别权重"""
    data_dir = os.path.dirname(data_path)
    weights_path = os.path.join(data_dir, 'emotion_weights.json')

    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            emotion_weights = json.load(f)
        print(f"加载类别权重: {emotion_weights}")
        weights = [emotion_weights.get(emo, 1.0) for emo in EMOTION_LIST]
        return torch.tensor(weights, dtype=torch.float)
    else:
        print("未找到类别权重文件，使用均匀权重")
        return None


def train_loop(
    model, tokenizer, config, train_dataset, val_dataset=None,
    output_prefix="baseline", peft_type="unknown"
):
    """
    通用训练循环

    Args:
        model: 已配置好的模型
        tokenizer: 分词器
        config: 配置字典
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        output_prefix: 输出目录前缀 (如 "ia3", "adapter")
        peft_type: PEFT 方法名称（用于日志）
    """
    device = next(model.parameters()).device
    cls_config = config['cls']
    output_dir = config['model']['output_dir']

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=cls_config['batch_size'], shuffle=True
    )
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset, batch_size=cls_config['batch_size'], shuffle=False
        )

    # 类别权重
    class_weights = load_class_weights(config['data']['sft_train_path'])
    if class_weights is not None:
        class_weights = class_weights.to(device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fct = nn.CrossEntropyLoss()

    # 优化器 + 调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cls_config['learning_rate'],
        weight_decay=cls_config.get('weight_decay', 0.01)
    )
    total_steps = len(train_dataloader) * cls_config['num_epochs']
    warmup_steps = int(total_steps * cls_config.get('warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\n[{peft_type}] 训练集大小: {len(train_dataset)}")
    if val_dataloader:
        print(f"[{peft_type}] 验证集大小: {len(val_dataloader)}")

    best_val_acc = 0.0

    for epoch in range(cls_config['num_epochs']):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"[{peft_type}] Epoch {epoch + 1}/{cls_config['num_epochs']}"
        )

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            loss = loss_fct(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct / total_samples:.4f}'
            })

        train_acc = total_correct / total_samples

        # 验证
        if val_dataloader:
            val_loss, val_acc = _validate(model, val_dataloader, loss_fct, device)
            print(f"[{peft_type}] Epoch {epoch + 1} - "
                  f"训练 acc: {train_acc:.4f}, 验证 loss: {val_loss:.4f}, 验证 acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _save_model(model, tokenizer, os.path.join(output_dir, f"{output_prefix}_best"))
        else:
            print(f"[{peft_type}] Epoch {epoch + 1} - 训练 acc: {train_acc:.4f}")

        # 每 5 个 epoch 或最后一个 epoch 保存检查点
        if (epoch + 1) % 5 == 0 or epoch == cls_config['num_epochs'] - 1:
            _save_model(model, tokenizer, os.path.join(output_dir, f"{output_prefix}_checkpoint_epoch_{epoch + 1}"))

    # 保存最终模型
    _save_model(model, tokenizer, os.path.join(output_dir, f"{output_prefix}_final"))
    print(f"[{peft_type}] 训练完成! 最佳验证 acc: {best_val_acc:.4f}")

    return model, best_val_acc


def _validate(model, val_dataloader, loss_fct, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            loss = loss_fct(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    model.train()
    return total_loss / len(val_dataloader), total_correct / total_samples


def _save_model(model, tokenizer, save_dir):
    """保存模型和分词器"""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"模型已保存到: {save_dir}")
