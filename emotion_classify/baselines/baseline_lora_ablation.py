"""
LoRA Rank Ablation 基线实验
- 测试不同 LoRA rank 对分类性能的影响
- r=8 (当前) vs r=16 vs r=32
- 文献显示 r=32 对 RoBERTa 分类任务最优
- 参考: Wang & Azman, "LoRA Fine-Tuning of RoBERTa" (2025)
"""

import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from common import (
    EMOTION_LIST, LABEL2ID, ID2LABEL,
    EmotionDataset, train_loop, load_class_weights,
)


def run_lora_ablation(config, rank=32):
    """运行 LoRA Rank Ablation 训练"""
    device = torch.device(config.get('device', 'cuda'))
    cls_config = config['cls']

    # 加载 tokenizer
    print(f"[LoRA-r{rank}] 加载 tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    print(f"[LoRA-r{rank}] 加载模型...")
    model = RobertaForSequenceClassification.from_pretrained(
        config['model']['model_path'],
        num_labels=len(EMOTION_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 配置 LoRA (使用指定 rank)
    alpha = rank * 2  # 保持 alpha/r = 2 的比例
    target_modules = config['lora']['target_modules']

    print(f"[LoRA-r{rank}] 配置 LoRA (r={rank}, alpha={alpha})...")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # 加载数据集
    train_dataset = EmotionDataset(
        config['data']['sft_train_path'], tokenizer,
        max_length=cls_config['max_length']
    )
    val_dataset = None
    if os.path.exists(config['data'].get('sft_val_path', '')):
        val_dataset = EmotionDataset(
            config['data']['sft_val_path'], tokenizer,
            max_length=cls_config['max_length']
        )

    # 训练
    model, best_acc = train_loop(
        model, tokenizer, config, train_dataset, val_dataset,
        output_prefix=f"cls_r{rank}", peft_type=f"LoRA-r{rank}"
    )

    return model, best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoRA Rank Ablation')
    parser.add_argument('--rank', type=int, default=32, choices=[8, 16, 32, 64],
                        help='LoRA rank')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    run_lora_ablation(config, rank=args.rank)
