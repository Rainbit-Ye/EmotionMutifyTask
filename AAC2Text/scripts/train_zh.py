#!/usr/bin/env python3
"""
AAC 中文 SFT 训练脚本 — Llama-3-8B + LoRA + DeepSpeed ZeRO-2

与 train.py 的区别:
  - 数据源: cleardata/sft_train.json (1689 条真实人工标注)
  - 目标字段: target_zh (中文)
  - Prompt 指令: 中文 (英文版本保留为注释)

Usage:
    python train_zh.py
    python train_zh.py --epochs 3 --batch 2
"""

import os
import json
import torch
import yaml
import random
import subprocess
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict


def find_available_gpus(min_free_gb: int = 5) -> str:
    """找到所有空闲显存 >= min_free_gb 的 GPU"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        gpus = []
        for line in r.stdout.strip().split("\n"):
            idx, free = line.strip().split(", ")
            if int(free) >= min_free_gb * 1024:
                gpus.append(idx)
        return ",".join(gpus) if gpus else "0"
    except Exception:
        return "0"


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class AACZhTrainDataset(Dataset):
    """AAC 中文 SFT 数据集 — 使用 target_zh 字段"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        labels = item["labels"]
        sentence = item["target_zh"]

        # 清理句子
        sentence = sentence.strip('"').strip("'").strip()
        sentence = sentence.split('\n')[0].strip()

        # 中文指令 (英文版本保留为注释供参考)
        # prompt = f"Translate these AAC symbols into ONE simple Chinese sentence: {' '.join(labels)}"
        prompt = f"请把这些 AAC 图标序列翻译成一个简单的中文句子：{' '.join(labels)}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sentence}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        labels_ids = input_ids.clone()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        # mask 掉 prompt 部分, 只在 assistant 回复上计算 loss
        assistant_start = -1
        for marker in ["<|im_start|>assistant", "<|start_header_id|>assistant<|end_header_id|>"]:
            pos = text.find(marker)
            if pos > 0:
                assistant_start = pos
                break
        if assistant_start > 0:
            prefix_text = text[:assistant_start]
            prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            labels_ids[:len(prefix_tokens)] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
        }


def load_data(data_path: str, val_ratio: float = 0.1, num_train: int = None):
    """加载中文 SFT 数据并划分训练/验证集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总数据量: {len(data)}")

    if num_train and num_train < len(data):
        data = data[:num_train]
        print(f"使用前 {num_train} 条数据")

    random.seed(42)
    random.shuffle(data)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size

    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

    return train_data, val_data


def load_model_and_tokenizer(model_path: str, lora_config: dict = None, use_lora: bool = True):
    print(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    if use_lora and lora_config:
        print("应用 LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 32),
            lora_alpha=lora_config.get("lora_alpha", 64),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train(config: dict, resume_from_checkpoint: str = None):
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["training"]
    lora_config = config["lora"]

    train_data, val_data = load_data(
        data_config["train_data"],
        data_config.get("val_ratio", 0.1),
        data_config.get("num_train")
    )

    val_path = data_config.get("val_data")
    if val_path:
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        print(f"验证集保存到: {val_path}")

    model, tokenizer = load_model_and_tokenizer(
        model_config["base_model"],
        lora_config,
        use_lora=True
    )

    print("\n准备数据集...")
    max_length = model_config.get("max_length", 128)
    train_dataset = AACZhTrainDataset(train_data, tokenizer, max_length)
    val_dataset = AACZhTrainDataset(val_data, tokenizer, max_length)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )

    training_args = TrainingArguments(
        output_dir=model_config["output_dir"],
        num_train_epochs=train_config.get("epochs", 3),
        per_device_train_batch_size=train_config.get("batch_size", 2),
        per_device_eval_batch_size=train_config.get("batch_size", 2),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
        learning_rate=train_config.get("learning_rate", 2e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.05),
        logging_steps=train_config.get("logging_steps", 50),
        eval_steps=train_config.get("eval_steps", 200),
        eval_strategy="steps",
        save_steps=train_config.get("save_steps", 200),
        save_total_limit=train_config.get("save_total_limit", 3),
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    num_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    print("\n" + "=" * 60)
    print(f"开始中文 SFT 训练 — DeepSpeed ZeRO-2, {num_gpus} GPU(s)")
    print(f"有效 batch size: {training_args.per_device_train_batch_size * num_gpus * training_args.gradient_accumulation_steps}")
    if resume_from_checkpoint:
        print(f"恢复训练: {resume_from_checkpoint}")
    print("=" * 60)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    output_dir = model_config["output_dir"]
    print(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("训练完成!")
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AAC 中文 SFT 训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--num', type=int, default=None, help='训练数据数量')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--resume', type=str, default=None, help='从checkpoint恢复训练')
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed local rank')
    parser.add_argument('--local-rank', type=int, default=-1, help='DeepSpeed local rank (alias)')
    args = parser.parse_args()

    config_path = args.config or "/home/user1/liuduanye/EmotionClassify/AAC2Text/config_zh.yaml"
    config = load_config(config_path)

    if args.num is not None:
        config["data"]["num_train"] = args.num
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch is not None:
        config["training"]["batch_size"] = args.batch
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    print("=" * 60)
    print("AAC 中文 SFT 训练 (Llama-3-8B + LoRA)")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"训练数据: {config['data'].get('num_train', '全部')}")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"学习率: {config['training']['learning_rate']}")

    resume_ckpt = None
    if args.resume:
        if args.resume == "auto":
            output_dir = config["model"]["output_dir"]
            ckpt_dirs = sorted(
                [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1])
            )
            if ckpt_dirs:
                resume_ckpt = os.path.join(output_dir, ckpt_dirs[-1])
                print(f"自动找到最新checkpoint: {resume_ckpt}")
            else:
                print("未找到可恢复的checkpoint，从头开始训练")
        else:
            resume_ckpt = args.resume
            print(f"从checkpoint恢复训练: {resume_ckpt}")

    train(config, resume_from_checkpoint=resume_ckpt)
