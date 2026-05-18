"""
Prefix Tuning 基线实验
- 在每层前添加可训练的 prefix 向量
- Prompt-based PEFT 方法，与 LoRA 的插入矩阵方式形成对比
- 参考论文: Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021)
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from peft import PrefixTuningConfig, get_peft_model, TaskType
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from common import (
    EMOTION_LIST, LABEL2ID, ID2LABEL,
    EmotionDataset, train_loop, load_class_weights,
)


def run_prefix_training(config):
    """运行 Prefix Tuning 训练"""
    device = torch.device(config.get('device', 'cuda'))
    cls_config = config['cls']

    # 加载 tokenizer
    print("[PrefixTuning] 加载 tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    print("[PrefixTuning] 加载模型...")
    model = RobertaForSequenceClassification.from_pretrained(
        config['model']['model_path'],
        num_labels=len(EMOTION_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 配置 Prefix Tuning (num_virtual_tokens=20)
    print("[PrefixTuning] 配置 Prefix Tuning...")
    prefix_config = PrefixTuningConfig(
        num_virtual_tokens=20,
        task_type=TaskType.SEQ_CLS,
        prefix_projection=True,
    )

    model = get_peft_model(model, prefix_config)
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
        output_prefix="prefix", peft_type="PrefixTuning"
    )

    return model, best_acc


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    run_prefix_training(config)
