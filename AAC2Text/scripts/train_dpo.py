#!/usr/bin/env python3
"""
AAC2Text 中文 DPO 训练脚本 — 基于 trl.DPOTrainer

在中文 SFT checkpoint (aac_model_zh) 基础上做 DPO 对齐训练。
- chosen: 人工修正后的自然中文句
- rejected: 翻译腔原句
- reference: SFT checkpoint 冻结

Usage:
    python train_dpo.py
    python train_dpo.py --beta 0.1 --epochs 3 --lr 5e-7
"""

import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig


def load_dpo_data(dpo_path: str, tokenizer, val_ratio: float = 0.1, seed: int = 42):
    """
    加载 dpo_pairs.json 并转换为 trl DPO 格式.
    输出: {prompt, chosen, rejected}
    - prompt: Llama-3 chat template + 中文指令 + labels (不包含 assistant 回复)
    - chosen: chosen 字段 (人工修正自然句)
    - rejected: rejected 字段 (翻译腔原句)
    """
    with open(dpo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    import random
    rng = random.Random(seed)
    rng.shuffle(data)

    val_size = int(len(data) * val_ratio)
    train_data = data[val_size:]
    val_data = data[:val_size]

    def to_dpo_format(items):
        formatted = []
        for item in items:
            labels = item["labels"]
            # 中文指令 (英文版本注释保留)
            # prompt_text = f"Translate these AAC symbols into ONE simple Chinese sentence: {' '.join(labels)}"
            prompt_text = f"请把这些 AAC 图标序列翻译成一个简单的中文句子：{' '.join(labels)}"
            messages = [{"role": "user", "content": prompt_text}]
            # apply_chat_template with add_generation_prompt=True 得到 prompt 部分
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append({
                "prompt": prompt,
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
        return formatted

    train_formatted = to_dpo_format(train_data)
    val_formatted = to_dpo_format(val_data)

    print(f"DPO 数据: 训练 {len(train_formatted)} 条, 验证 {len(val_formatted)} 条")
    return train_formatted, val_formatted


def main():
    parser = argparse.ArgumentParser(description='AAC2Text 中文 DPO 训练')
    parser.add_argument('--sft-model', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_model_zh',
                        help='中文 SFT checkpoint 路径 (DPO 起点 + reference)')
    parser.add_argument('--base-model', type=str,
                        default='/home/user1/liuduanye/Meta-Llama-3-8B-Instruct',
                        help='基模路径 (用于加载 tokenizer)')
    parser.add_argument('--dpo-data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/data/cleardata/dpo_pairs.json')
    parser.add_argument('--output-dir', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_dpo_zh')
    parser.add_argument('--beta', type=float, default=0.1, help='DPO beta')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--max-prompt-length', type=int, default=128)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 加载 tokenizer
    print(f"加载 tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载 DPO 数据
    print(f"加载 DPO 数据: {args.dpo_data}")
    train_data, val_data = load_dpo_data(args.dpo_data, tokenizer, args.val_ratio, args.seed)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # 方案: 合并 SFT LoRA 到 base model, 让 DPOTrainer 用 peft_config 初始化新 LoRA
    # ref = disable_adapter() = SFT 合并后的 base model 权重
    # policy = SFT base + 新 LoRA (可训练)
    print(f"加载 SFT checkpoint 并合并 LoRA 到 base model: {args.sft_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    from peft import PeftModel
    policy_model = PeftModel.from_pretrained(base_model, args.sft_model)
    # 合并 LoRA 到 base model 并卸载 PeftModel 包装
    policy_model = policy_model.merge_and_unload()
    print("SFT LoRA 已合并到 base model, 现在是普通 PreTrainedModel")

    # ref_model=None: DPOTrainer 用 disable_adapter() 计算 ref logprobs
    # 但此时 model 不是 PeftModel, 所以需要传 peft_config 让 DPOTrainer 初始化新 LoRA
    ref_model = None

    # LoRA config (与 SFT 一致, DPOTrainer 会用 peft_config 在合并后的 model 上初始化新 LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # DPO 训练配置 (trl 1.7.0 API)
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_steps=50,
        eval_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        beta=args.beta,
        max_length=args.max_length,
        gradient_checkpointing=False,
    )

    # DPO Trainer (传 peft_config, trl 自动初始化新 LoRA; ref 用 disable_adapter)
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n" + "=" * 60)
    print(f"开始 DPO 训练")
    print(f"beta={args.beta}, lr={args.lr}, epochs={args.epochs}")
    print(f"有效 batch size: {args.batch_size * args.grad_accum}")
    print("=" * 60)

    trainer.train()

    # 保存最终模型
    print(f"\n保存 DPO 模型到: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("DPO 训练完成!")


if __name__ == "__main__":
    main()
