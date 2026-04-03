#!/usr/bin/env python3
"""
情绪分类模型评估脚本
支持单任务和多任务模型

评估时输入不包含情绪标签，模拟真实场景
"""

import json
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


class MultiTaskEmotionClassifier(nn.Module):
    """多任务情绪分类模型（用于评估）"""
    def __init__(self, base_model_path, num_labels=7, lora_config=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(base_model_path)
        if lora_config is not None:
            self.roberta = get_peft_model(self.roberta, lora_config)

        hidden_size = self.roberta.config.hidden_size
        self.main_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )
        self.turn_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        main_hidden = hidden_states[:, 0, :]
        main_logits = self.main_classifier(main_hidden)
        return main_logits


def load_test_data(data_path):
    """加载测试数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(model_path, base_model_path, device, use_multitask=True):
    """加载微调后的模型"""
    if use_multitask:
        print(f"加载多任务模型: {model_path}")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = MultiTaskEmotionClassifier(
            base_model_path,
            num_labels=len(EMOTION_LIST),
            lora_config=lora_config
        )
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=device), strict=False)
    else:
        print(f"加载基础模型: {base_model_path}")
        base_model = RobertaForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=len(EMOTION_LIST)
        )
        print(f"加载LoRA权重: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)

    model.to(device)
    model.eval()
    return model


def format_conversation(conversation):
    """
    格式化对话文本（评估时不包含情绪标签）
    模拟真实场景，模型需要自己预测情绪
    """
    text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"
    return text.strip()


def predict_emotion(model, tokenizer, conversation, device, max_length=256, use_multitask=True, neutral_threshold=0.4):
    """
    预测情绪

    Args:
        neutral_threshold: 如果预测的非neutral情绪概率低于此阈值，倾向于预测neutral
                          提高阈值减少误判为neutral（0.3 -> 0.4）
    """
    text = format_conversation(conversation)

    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if use_multitask:
            logits = outputs
        else:
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_prob = probs[pred_idx].item()

        neutral_idx = 0

        # 更保守的neutral判断：
        # 只在预测置信度很低（<0.4）且neutral概率较高（>0.3）时才改为neutral
        if pred_idx != neutral_idx:
            neutral_prob = probs[neutral_idx].item()

            # 条件：预测置信度很低 且 neutral概率较高
            if pred_prob < neutral_threshold and neutral_prob > 0.3:
                pred_idx = neutral_idx
                pred_prob = neutral_prob

    return EMOTION_LIST[pred_idx]


def evaluate(model, tokenizer, test_data, device, output_dir, use_multitask=True):
    """评估模型"""
    print("\n开始评估...")

    predictions = []
    labels = []

    for idx, item in enumerate(tqdm(test_data, desc="评估中")):
        conversation = item["conversation"]
        true_emotion = item["main_emotion"]

        try:
            pred_emotion = predict_emotion(model, tokenizer, conversation, device, use_multitask=use_multitask)
        except Exception as e:
            print(f"\n样本 {idx} 预测失败: {e}")
            pred_emotion = "neutral"

        predictions.append(pred_emotion)
        labels.append(true_emotion)

        if idx % 100 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    all_emotions = EMOTION_LIST

    # 整体准确率
    accuracy = accuracy_score(labels, predictions)
    print(f"\n整体准确率: {accuracy:.4f}")

    # 非neutral准确率（关键指标）
    non_neutral_mask = [l != "neutral" for l in labels]
    if any(non_neutral_mask):
        non_neutral_labels = [l for l, m in zip(labels, non_neutral_mask) if m]
        non_neutral_preds = [p for p, m in zip(predictions, non_neutral_mask) if m]
        non_neutral_acc = accuracy_score(non_neutral_labels, non_neutral_preds)
        print(f"非neutral准确率: {non_neutral_acc:.4f} ({len(non_neutral_labels)} 样本)")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(labels, predictions, target_names=all_emotions, zero_division=0))

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=all_emotions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_emotions, yticklabels=all_emotions)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'混淆矩阵 (准确率: {accuracy:.4f})')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'cls_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n混淆矩阵已保存: {cm_path}")

    # 各类别准确率
    print("\n各类别准确率:")
    for emotion in all_emotions:
        emotion_indices = [i for i, l in enumerate(labels) if l == emotion]
        if emotion_indices:
            emotion_preds = [predictions[i] for i in emotion_indices]
            emotion_labels = [labels[i] for i in emotion_indices]
            emotion_acc = accuracy_score(emotion_labels, emotion_preds)
            print(f"  {emotion}: {emotion_acc:.4f} ({len(emotion_indices)} 样本)")

    # 保存结果
    results = {
        "accuracy": accuracy,
        "predictions": predictions,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "emotion_labels": all_emotions,
        "total_samples": len(labels)
    }

    if any(non_neutral_mask):
        results["non_neutral_accuracy"] = non_neutral_acc

    results_path = os.path.join(output_dir, 'cls_evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"评估结果已保存: {results_path}")

    return accuracy, cm


def main():
    parser = argparse.ArgumentParser(description='情绪分类模型评估')
    parser.add_argument('--model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_final',
                        help='微调后的模型路径')
    parser.add_argument('--base_model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/Model/roberta-base',
                        help='基础模型路径')
    parser.add_argument('--test_data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/sft_test.json',
                        help='测试数据路径')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_evaluation',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--max_samples', type=int, default=0, help='最大评估样本数')
    parser.add_argument('--multitask', action='store_true', default=True,
                        help='使用多任务模型（默认True）')
    parser.add_argument('--single-task', dest='multitask', action='store_false',
                        help='使用单任务模型')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    print(f"模型类型: {'多任务' if args.multitask else '单任务'}")

    # 加载模型
    model = load_model(args.model_path, args.base_model_path, device, use_multitask=args.multitask)

    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载测试数据
    print(f"加载测试数据: {args.test_data}")
    test_data = load_test_data(args.test_data)

    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    print(f"测试样本数: {len(test_data)}")

    # 评估
    evaluate(model, tokenizer, test_data, device, args.output_dir, use_multitask=args.multitask)


if __name__ == '__main__':
    main()
