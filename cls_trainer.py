#!/usr/bin/env python3
"""
情绪分类模型训练模块
使用 RobertaForSequenceClassification 进行分类

相比生成式方法的优势：
1. 训练更快（单模型，无需生成）
2. 推理更高效（一次前向传播）
3. 更符合 RoBERTa 的设计初衷
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
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}


class EmotionDataset(Dataset):
    """情绪分类数据集"""
    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        """加载数据"""
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

        # 训练时不包含情绪标签，让模型纯粹从文本学习预测main_emotion
        text = self._format_conversation(conversation, include_emotion_labels=False)

        # Tokenize
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

    def _format_conversation(self, conversation, include_emotion_labels=True):
        """
        格式化对话文本

        Args:
            conversation: 对话列表
            include_emotion_labels: 是否包含情绪标签
                - 训练时为 True，帮助模型学习文本与情绪的关联
                - 推理时为 False，模型需要自己预测情绪
        """
        text = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                text += f"User: {content}\n"
            else:
                text += f"Assistant: {content}\n"

            # 训练时包含情绪标签，帮助模型学习
            if include_emotion_labels:
                emotion = turn.get("emotion", "neutral")
                text += f"[Emotion: {emotion}]\n"

        return text.strip()


class EmotionClassifier:
    """情绪分类训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 加载类别权重（处理不平衡）
        self.class_weights = self._load_class_weights()

        # 加载tokenizer
        print("加载tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(config['model']['model_path'])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载分类模型
        print("加载分类模型...")
        self.model = RobertaForSequenceClassification.from_pretrained(
            config['model']['model_path'],
            num_labels=len(EMOTION_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

        # 配置LoRA
        print("配置LoRA...")
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias=config['lora']['bias'],
            task_type=TaskType.SEQ_CLS
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

        self.model.print_trainable_parameters()

    def _load_class_weights(self):
        """加载类别权重"""
        data_dir = os.path.dirname(self.config['data']['sft_train_path'])
        weights_path = os.path.join(data_dir, 'emotion_weights.json')

        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                emotion_weights = json.load(f)
            print(f"加载类别权重: {emotion_weights}")

            # 转换为tensor，按标签顺序排列
            weights = [emotion_weights.get(emo, 1.0) for emo in EMOTION_LIST]
            return torch.tensor(weights, dtype=torch.float)
        else:
            print("未找到类别权重文件，使用均匀权重")
            return None

    def train(self):
        """执行训练"""
        cls_config = self.config['cls']

        # 加载数据集
        train_dataset = EmotionDataset(
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
            val_dataset = EmotionDataset(
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
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        print(f"\n训练集大小: {len(train_dataset)}")
        if val_dataloader:
            print(f"验证集大小: {len(val_dataloader.dataset)}")
        print("开始训练...")

        best_val_loss = float('inf')
        best_val_acc = 0.0

        for epoch in range(cls_config['num_epochs']):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{cls_config['num_epochs']}"
            )

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 计算损失（使用类别权重）
                loss = loss_fct(outputs.logits, labels)

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

                # 更新进度条
                avg_loss = total_loss / (total_samples / cls_config['batch_size'])
                accuracy = total_correct / total_samples
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'acc': f'{accuracy:.4f}'
                })

            # 验证
            if val_dataloader:
                val_loss, val_acc = self._validate(val_dataloader, loss_fct)
                print(f"Epoch {epoch + 1} 完成 - "
                      f"训练损失: {total_loss / len(train_dataloader):.4f}, "
                      f"训练准确率: {accuracy:.4f}, "
                      f"验证损失: {val_loss:.4f}, "
                      f"验证准确率: {val_acc:.4f}")

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch + 1, is_best=True)
            else:
                print(f"Epoch {epoch + 1} 完成 - "
                      f"平均损失: {total_loss / len(train_dataloader):.4f}, "
                      f"准确率: {accuracy:.4f}")

            # 保存每轮检查点
            self._save_checkpoint(epoch + 1, is_epoch=True)

        # 保存最终模型
        self._save_final_model()
        print("训练完成!")

        return self.model

    def _validate(self, val_dataloader, loss_fct):
        """验证集评估"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = loss_fct(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        self.model.train()
        return total_loss / len(val_dataloader), total_correct / total_samples

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
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"检查点已保存到: {checkpoint_dir}")

    def _save_final_model(self):
        """保存最终模型"""
        output_dir = os.path.join(self.config['model']['output_dir'], "cls_final")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"最终模型已保存到: {output_dir}")


def run_cls_training(config):
    """运行分类训练的入口函数"""
    trainer = EmotionClassifier(config)
    model = trainer.train()
    return model


if __name__ == '__main__':
    with open('/home/user1/liuduanye/EmotionClassify/config.json', 'r') as f:
        config = json.load(f)

    run_cls_training(config)
