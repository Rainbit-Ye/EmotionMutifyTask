"""
IA3 基线实验
- 只缩放 key/value/hidden 激活值，比 LoRA 更轻量
- 零推理开销（可合并回原模型）
- 参考论文: Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (2022)
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from peft import IA3Config, get_peft_model, TaskType
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from common import (
    EMOTION_LIST, LABEL2ID, ID2LABEL,
    EmotionDataset, train_loop, load_class_weights,
)


def run_ia3_training(config):
    """运行 IA3 训练"""
    device = torch.device(config.get('device', 'cuda'))
    cls_config = config['cls']

    # 加载 tokenizer
    print("[IA3] 加载 tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    print("[IA3] 加载模型...")
    model = RobertaForSequenceClassification.from_pretrained(
        config['model']['model_path'],
        num_labels=len(EMOTION_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 配置 IA3
    print("[IA3] 配置 IA3...")
    ia3_config = IA3Config(
        target_modules=["key", "value", "dense"],
        feedforward_modules=["dense"],
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, ia3_config)
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
        output_prefix="ia3", peft_type="IA3"
    )

    return model, best_acc


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    run_ia3_training(config)
