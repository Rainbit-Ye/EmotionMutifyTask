"""
上下文窗口基线实验
- 将前 n-1 轮对话拼接到输入，利用对话上下文
- 添加说话人标记 (Speaker1/Speaker2) 增强角色区分
- 使用 LoRA r=8 + 上下文格式
- 参考论文: 上下文建模在 ERC 中的有效性 (+3-5% Micro F1)
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from common import (
    EMOTION_LIST, LABEL2ID, ID2LABEL,
    EmotionDataset, train_loop, load_class_weights,
)


def run_context_training(config):
    """运行上下文窗口训练"""
    device = torch.device(config.get('device', 'cuda'))
    cls_config = config['cls']

    # 加载 tokenizer
    print("[Context] 加载 tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    print("[Context] 加载模型...")
    model = RobertaForSequenceClassification.from_pretrained(
        config['model']['model_path'],
        num_labels=len(EMOTION_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 配置 LoRA (与基础版相同的 r=8)
    print("[Context] 配置 LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # 加载数据集 — 开启上下文窗口
    train_dataset = EmotionDataset(
        config['data']['sft_train_path'], tokenizer,
        max_length=cls_config['max_length'],
        include_context=True,  # 使用上下文格式
    )
    val_dataset = None
    if os.path.exists(config['data'].get('sft_val_path', '')):
        val_dataset = EmotionDataset(
            config['data']['sft_val_path'], tokenizer,
            max_length=cls_config['max_length'],
            include_context=True,
        )

    # 训练
    model, best_acc = train_loop(
        model, tokenizer, config, train_dataset, val_dataset,
        output_prefix="context", peft_type="Context-LoRA"
    )

    return model, best_acc


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    run_context_training(config)
