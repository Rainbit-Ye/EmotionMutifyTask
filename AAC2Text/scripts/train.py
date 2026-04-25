"""
AAC 标签序列到文本生成模型训练脚本

任务：将标签序列转换为自然语言文本
输入：["I", "want_to", "water"]
输出："I want water."

使用 Qwen2.5-1.5B-Instruct + LoRA 微调
"""

import os
# 只使用 GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
import yaml
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
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
        
        # 清理句子，只保留第一句
        sentence = sentence.strip('"').strip("'").strip()
        sentence = sentence.split('\n')[0].strip()  # 只取第一行
        if '.' in sentence:
            sentence = sentence.split('.')[0] + '.'  # 只取第一句
        
        # 使用chat template格式
        prompt = f"Translate these AAC symbols to a sentence: {' '.join(labels)}"
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
        
        # 只计算assistant部分的loss
        # 找到assistant回复的开始位置
        assistant_text = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": sentence}],
            tokenize=False,
            add_generation_prompt=False
        )
        
        labels_ids = input_ids.clone()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        
        # 找到assistant开始的位置，之前的都mask掉
        assistant_start = text.find("<|im_start|>assistant")
        if assistant_start > 0:
            # 计算token位置
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
    
    # 限制训练数据数量
    if num_train and num_train < len(data):
        data = data[:num_train]
        print(f"使用前 {num_train} 条数据")
    
    # 随机划分
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
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 确保 pad token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 应用 LoRA
    if use_lora and lora_config:
        print("应用 LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def compute_metrics(eval_preds, tokenizer):
    """计算评估指标"""
    # 加载评估指标
    bleu = evaluate.load("bleu")

    predictions, labels = eval_preds

    # 解码预测结果
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 将 -100 替换为 pad_token_id
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 提取 Output 后面的部分
    def extract_output(text):
        if "Output:" in text:
            return text.split("Output:")[-1].strip()
        return text.strip()

    decoded_preds = [extract_output(p) for p in decoded_preds]
    decoded_labels = [extract_output(l) for l in decoded_labels]

    # 计算 BLEU
    bleu_result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    # 计算精确匹配率
    exact_match = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
    exact_match_ratio = exact_match / len(decoded_preds) if decoded_preds else 0

    # 计算部分匹配率
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
        """
        重写 compute_loss 以在训练过程中打印更多信息
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 每 100 步打印一次训练 loss
        if self.state.global_step % 100 == 0 and self.state.global_step > 0:
            print(f"\n[Step {self.state.global_step}] Training Loss: {loss.item():.4f}")
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写评估循环，添加更多评估指标
        """
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # 如果有预测结果，计算额外指标
        if output.predictions is not None and self.tokenizer is not None:
            metrics = compute_metrics((output.predictions, output.label_ids), self.tokenizer)
            output.metrics.update(metrics)
            
            # 打印详细评估结果
            print(f"\n{'='*60}")
            print(f"评估结果 ({description}):")
            print(f"  Loss: {output.metrics.get('eval_loss', 'N/A'):.4f}" if 'eval_loss' in output.metrics else "")
            print(f"  BLEU: {metrics['bleu']:.4f}")
            print(f"  Exact Match: {metrics['exact_match']:.4f}")
            print(f"  Partial Match: {metrics['partial_match']:.4f}")
            print(f"{'='*60}\n")
        
        return output


def train(config: dict):
    """训练函数"""
    
    # 从配置获取参数
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
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=model_config["output_dir"],
        num_train_epochs=train_config.get("epochs", 3),
        per_device_train_batch_size=train_config.get("batch_size", 8),
        per_device_eval_batch_size=train_config.get("batch_size", 8),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        learning_rate=train_config.get("learning_rate", 2e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_steps=train_config.get("warmup_steps", 100),
        logging_steps=train_config.get("logging_steps", 50),
        eval_steps=train_config.get("eval_steps", 200),
        eval_strategy="steps",
        save_steps=train_config.get("save_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 3),
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Custom Trainer
    trainer = CustomTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    print(f"每隔 {train_config.get('eval_steps', 200)} 步评估一次")
    print("="*60)
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
    
    print("\n" + "="*60)
    print("测试模型")
    print("="*60)
    
    model_config = config["model"]
    test_config = config["test"]
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载 LoRA
    output_dir = model_config["output_dir"]
    if os.path.exists(output_dir):
        print(f"加载 LoRA: {output_dir}")
        model = PeftModel.from_pretrained(model, output_dir)
    
    model.eval()
    
    # 测试样例
    test_cases = test_config.get("test_samples", [
        ["I", "want_to", "water"],
        ["I", "am", "happy"],
        ["I", "eat_to", "apple"],
        ["I", "go_to", "school"],
    ])
    
    print("\n推理测试:")
    print("-" * 60)
    
    for labels in test_cases:
        # 使用与训练一致的chat template格式
        prompt = f"Translate these AAC symbols to a sentence: {' '.join(labels)}"
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                stop_strings=["<|im_end|>", "\n"],
                tokenizer=tokenizer,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # 只取第一句
        response = response.strip().split('\n')[0].strip()
        print(f"Labels: {labels}")
        print(f"Output: {response}")
        print()
    
    # 如果有验证集，计算 BLEU
    val_path = config["data"].get("val_data")
    if val_path and os.path.exists(val_path):
        print("\n计算 BLEU 分数 (采样50条)...")
        
        # 使用本地 BLEU 模块
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        from bleu.bleu import Bleu
        
        bleu_scorer = Bleu()
        
        with open(val_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 只采样50条测试
        import random
        random.seed(42)
        test_data = random.sample(test_data, min(50, len(test_data)))
        
        preds = []
        refs = []
        
        for item in tqdm(test_data, desc="计算BLEU"):
            labels = item["labels"]
            sentence = item["sentence"].strip('"').strip("'").strip()
            
            # 使用与训练一致的chat template格式
            prompt = f"Translate these AAC symbols to a sentence: {' '.join(labels)}"
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    stop_strings=["<|im_end|>", "\n"],
                    tokenizer=tokenizer,
                )
            
            pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = pred.strip().split('\n')[0].strip()
            
            preds.append(pred)
            refs.append([sentence])
        
        # 使用本地 BLEU 计算
        result = bleu_scorer.compute(predictions=preds, references=refs)
        print(f"\nBLEU 分数: {result['bleu']:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AAC 模型训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式')
    
    # 可覆盖配置的参数
    parser.add_argument('--num', type=int, default=None, help='训练数据数量')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config or "/home/user1/liuduanye/EmotionClassify/AAC2Text/config.yaml"
    config = load_config(config_path)
    
    # 命令行参数覆盖配置
    if args.num is not None:
        config["data"]["num_train"] = args.num
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch is not None:
        config["training"]["batch_size"] = args.batch
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    
    print("="*60)
    print("AAC 标签序列到文本生成模型训练")
    print("="*60)
    print(f"配置文件: {config_path}")
    print(f"训练数据: {config['data'].get('num_train', '全部')}")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"学习率: {config['training']['learning_rate']}")
    
    if args.test:
        # 测试模式
        test_model(config)
    else:
        # 训练模式
        train(config)
        
        # 训练后测试
        test_model(config)


if __name__ == "__main__":
    main()
