#!/usr/bin/env python3
"""
简单RoBERTa分类模型训练器（单任务）
作为对比基线，不使用多任务学习、对比学习等高级技术
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
from sklearn.metrics import accuracy_score, f1_score


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}


class SimpleEmotionDataset(Dataset):
    """简单数据集"""
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

        # 格式化对话文本
        text = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                text += f"User: {content}\n"
            else:
                text += f"Assistant: {content}\n"
        text = text.strip()

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': LABEL2ID[main_emotion]
        }


class SimpleTrainer:
    """简单训练器"""
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

        # 加载分类模型
        print("加载RoBERTa分类模型...")
        self.model = RobertaForSequenceClassification.from_pretrained(
            config['model']['model_path'],
            num_labels=len(EMOTION_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        self.model.to(self.device)

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params:,}")

    def _load_class_weights(self):
        data_dir = os.path.dirname(self.config['data']['sft_train_path'])
        weights_path = os.path.join(data_dir, 'emotion_weights.json')

        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                emotion_weights = json.load(f)
            print(f"加载类别权重: {emotion_weights}")
            weights = [emotion_weights.get(emo, 1.0) for emo in EMOTION_LIST]
            return torch.tensor(weights, dtype=torch.float)
        else:
            print("未找到类别权重文件")
            return None

    def train(self):
        cls_config = self.config['cls']

        # 加载数据集
        train_dataset = SimpleEmotionDataset(
            self.config['data']['sft_train_path'],
            self.tokenizer,
            max_length=cls_config['max_length']
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cls_config['batch_size'],
            shuffle=True
        )

        # 验证集
        val_dataloader = None
        if os.path.exists(self.config['data'].get('sft_val_path', '')):
            val_dataset = SimpleEmotionDataset(
                self.config['data']['sft_val_path'],
                self.tokenizer,
                max_length=cls_config['max_length']
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cls_config['batch_size'],
                shuffle=False
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

        # 损失函数（带类别权重）
        if self.class_weights is not None:
            class_weights = self.class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        print(f"\n训练集大小: {len(train_dataset)}")
        if val_dataloader:
            print(f"验证集大小: {len(val_dataloader.dataset)}")
        print("开始简单分类模型训练...")

        best_val_acc = 0.0

        for epoch in range(cls_config['num_epochs']):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{cls_config['num_epochs']}")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # 如果使用类别权重，需要重新计算损失
                if self.class_weights is not None:
                    logits = outputs.logits
                    loss = criterion(logits, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # 统计
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                accuracy = total_correct / total_samples
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}'
                })

            # 验证
            if val_dataloader:
                val_acc, val_loss = self._validate(val_dataloader, criterion)
                print(f"Epoch {epoch + 1} 完成 - "
                      f"训练损失: {total_loss / len(train_dataloader):.4f}, "
                      f"训练准确率: {accuracy:.4f}, "
                      f"验证准确率: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch + 1, is_best=True)
            else:
                print(f"Epoch {epoch + 1} 完成 - "
                      f"平均损失: {total_loss / len(train_dataloader):.4f}, "
                      f"准确率: {accuracy:.4f}")

            self._save_checkpoint(epoch + 1, is_epoch=True)

        self._save_final_model()
        print("简单分类模型训练完成!")

        return self.model

    def _validate(self, val_dataloader, criterion):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        self.model.train()
        return total_correct / total_samples, total_loss / len(val_dataloader)

    def _save_checkpoint(self, step, is_epoch=False, is_best=False):
        output_dir = self.config['model']['output_dir']
        if is_best:
            checkpoint_dir = os.path.join(output_dir, "simple_best")
        elif is_epoch:
            checkpoint_dir = os.path.join(output_dir, f"simple_checkpoint_epoch_{step}")
        else:
            checkpoint_dir = os.path.join(output_dir, f"simple_checkpoint_step_{step}")

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"检查点已保存到: {checkpoint_dir}")

    def _save_final_model(self):
        output_dir = os.path.join(self.config['model']['output_dir'], "simple_final")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"最终模型已保存到: {output_dir}")


def run_simple_training(config):
    trainer = SimpleTrainer(config)
    model = trainer.train()
    return model


if __name__ == '__main__':
    with open('/home/user1/liuduanye/EmotionClassify/config.json', 'r') as f:
        config = json.load(f)

    run_simple_training(config)
