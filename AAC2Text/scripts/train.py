"""
AAC 标签序列到文本生成模型训练脚本

任务：将标签序列转换为自然语言文本
输入：["I", "want_to", "water"]
输出："I want water."

使用 Meta-Llama-3-8B-Instruct + LoRA + DeepSpeed ZeRO-2 多卡训练
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
import re
import evaluate


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
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class AACTrainDataset(Dataset):
    """AAC 训练数据集"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        labels = item["labels"]
        sentence = item["sentence"]

        # 清理句子，只保留第一句（以句号截断）
        sentence = sentence.strip('"').strip("'").strip()
        sentence = sentence.split('\n')[0].strip()
        dot_pos = sentence.find('.')
        if dot_pos != -1 and dot_pos < len(sentence) - 1:
            sentence = sentence[:dot_pos + 1]
        if '.' in sentence:
            sentence = sentence.split('.')[0] + '.'

        # 使用chat template格式
        prompt = f"Translate these AAC symbols into ONE simple English sentence: {' '.join(labels)}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sentence}
        ]

        # 应用chat template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize
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

        # 找到assistant开始的位置，之前的都mask掉
        # Qwen: <|im_start|>assistant, Llama-3: <|start_header_id|>assistant<|end_header_id|>
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
    """加载数据并划分训练集和验证集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总数据量: {len(data)}")

    if num_train and num_train < len(data):
        data = data[:num_train]
        print(f"使用前 {num_train} 条数据")

    # 随机划分
    random.shuffle(data)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size

    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

    return train_data, val_data


def load_model_and_tokenizer(model_path: str, lora_config: dict = None, use_lora: bool = True):
    """加载模型和 tokenizer"""

    print(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # DeepSpeed ZeRO-2 下不使用 device_map="auto"，由 DeepSpeed 管理设备
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 应用 LoRA
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


def compute_metrics(eval_preds, tokenizer):
    """计算评估指标"""
    bleu = evaluate.load("bleu")

    predictions, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def extract_output(text):
        if "Output:" in text:
            return text.split("Output:")[-1].strip()
        return text.strip()

    decoded_preds = [extract_output(p) for p in decoded_preds]
    decoded_labels = [extract_output(l) for l in decoded_labels]

    bleu_result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    exact_match = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
    exact_match_ratio = exact_match / len(decoded_preds) if decoded_preds else 0

    partial_match = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_words = set(pred.lower().split())
        label_words = set(label.lower().split())
        if pred_words & label_words:
            partial_match += 1
    partial_match_ratio = partial_match / len(decoded_preds) if decoded_preds else 0

    return {
        "bleu": bleu_result["bleu"],
        "exact_match": exact_match_ratio,
        "partial_match": partial_match_ratio,
    }


class CustomTrainer(Trainer):
    """自定义 Trainer，添加评估指标"""

    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.state.global_step % 100 == 0 and self.state.global_step > 0:
            print(f"\n[Step {self.state.global_step}] Training Loss: {loss.item():.4f}")

        return (loss, outputs) if return_outputs else loss


def train(config: dict):
    """训练函数 — DeepSpeed ZeRO-2 多卡"""

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["training"]
    lora_config = config["lora"]

    # 加载数据
    train_data, val_data = load_data(
        data_config["train_data"],
        data_config.get("val_ratio", 0.1),
        data_config.get("num_train")
    )

    # 保存验证集
    val_path = data_config.get("val_data")
    if val_path:
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        print(f"验证集保存到: {val_path}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_config["base_model"],
        lora_config,
        use_lora=True
    )

    # 创建数据集
    print("\n准备数据集...")
    max_length = model_config.get("max_length", 128)
    train_dataset = AACTrainDataset(train_data, tokenizer, max_length)
    val_dataset = AACTrainDataset(val_data, tokenizer, max_length)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )

    # DeepSpeed 配置
    ds_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "ds_config_zero2.json"
    )

    # 训练参数 — 针对多卡 + 8B 模型优化
    training_args = TrainingArguments(
        output_dir=model_config["output_dir"],
        num_train_epochs=train_config.get("epochs", 3),
        per_device_train_batch_size=train_config.get("batch_size", 4),
        per_device_eval_batch_size=train_config.get("batch_size", 4),
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
        # DeepSpeed
        deepspeed=ds_config_path,
    )

    trainer = CustomTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    num_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    print("\n" + "=" * 60)
    print(f"开始训练 — DeepSpeed ZeRO-2, {num_gpus} GPU(s)")
    print(f"有效 batch size: {training_args.per_device_train_batch_size * num_gpus * training_args.gradient_accumulation_steps}")
    print("=" * 60)
    trainer.train()

    # 保存最终模型
    output_dir = model_config["output_dir"]
    print(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("训练完成!")

    return trainer


def test_model(config: dict):
    """测试模型"""
    from peft import PeftModel

    print("\n" + "=" * 60)
    print("测试模型")
    print("=" * 60)

    model_config = config["model"]
    test_config = config["test"]

    # 设置可见GPU
    available_gpus = find_available_gpus(min_free_gb=15)
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus.split(",")[0]

    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    output_dir = model_config["output_dir"]
    if os.path.exists(output_dir):
        print(f"加载 LoRA: {output_dir}")
        model = PeftModel.from_pretrained(model, output_dir)

    model.eval()

    test_cases = test_config.get("test_samples", [
        ["I", "want_to", "water"],
        ["I", "am", "happy"],
        ["I", "eat_to", "apple"],
        ["I", "go_to", "school"],
    ])

    print("\n推理测试:")
    print("-" * 60)

    for labels in test_cases:
        prompt = f"Translate these AAC symbols into ONE simple English sentence: {' '.join(labels)}"
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                stop_strings=["<|im_end|>", "<|eot_id|>", "\n"],
                tokenizer=tokenizer,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().split('\n')[0].strip()
        dot_pos = response.find('.')
        if dot_pos != -1 and dot_pos < len(response) - 1:
            response = response[:dot_pos + 1]
        print(f"Labels: {labels}")
        print(f"Output: {response}")
        print()

    # BLEU 评估
    val_path = config["data"].get("val_data")
    if val_path and os.path.exists(val_path):
        print("\n计算 BLEU 分数 (采样50条)...")

        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        from bleu.bleu import Bleu

        bleu_scorer = Bleu()

        with open(val_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 兼容多种格式：标准JSON数组、JSONL、或多个JSON数组拼接
        try:
            test_data = json.loads(content)
        except json.JSONDecodeError:
            test_data = []
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                content_part = content[pos:].lstrip()
                if not content_part:
                    break
                try:
                    obj, end = decoder.raw_decode(content_part)
                    if isinstance(obj, list):
                        test_data.extend(obj)
                    else:
                        test_data.append(obj)
                    pos += len(content[pos:]) - len(content_part) + end
                except json.JSONDecodeError:
                    break

        # 使用全部验证集评估
        print(f"\n计算 BLEU 分数 (全部 {len(test_data)} 条)...")

        preds = []
        refs = []

        for item in tqdm(test_data, desc="计算BLEU"):
            labels = item["labels"]
            sentence = item["sentence"].strip('"').strip("'").strip()

            prompt = f"Translate these AAC symbols into ONE simple English sentence: {' '.join(labels)}"
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    stop_strings=["<|im_end|>", "<|eot_id|>", "\n"],
                    tokenizer=tokenizer,
                )

            pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = pred.strip().split('\n')[0].strip()
            dot_pos = pred.find('.')
            if dot_pos != -1 and dot_pos < len(pred) - 1:
                pred = pred[:dot_pos + 1]

            preds.append(pred)
            refs.append([sentence])

        result = bleu_scorer.compute(predictions=preds, references=refs)
        print(f"\nBLEU 分数: {result['bleu']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AAC 模型训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式')

    parser.add_argument('--num', type=int, default=None, help='训练数据数量')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed local rank')
    parser.add_argument('--local-rank', type=int, default=-1, help='DeepSpeed local rank (alias)')
    args = parser.parse_args()

    config_path = args.config or "/home/user1/liuduanye/EmotionClassify/AAC2Text/config.yaml"
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
    print("AAC 标签序列到文本生成模型训练")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"训练数据: {config['data'].get('num_train', '全部')}")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"学习率: {config['training']['learning_rate']}")

    if args.test:
        test_model(config)
    else:
        train(config)
        # DeepSpeed 多卡训练后显存被占，测试需单独运行:
        #   python scripts/train.py --test
        print("\n训练后测试请单独运行: python scripts/train.py --test")


if __name__ == "__main__":
    main()
