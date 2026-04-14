#!/usr/bin/env python3
"""
情绪分类模型评估脚本
支持三种模型对比评估：cls（基础版）、cls_multitask（多任务版）、simple（简单版）

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}


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
        # 新增：下一轮情绪预测分类头
        self.next_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, return_next=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        main_hidden = hidden_states[:, 0, :]
        main_logits = self.main_classifier(main_hidden)

        if return_next:
            next_logits = self.next_classifier(main_hidden)
            return main_logits, next_logits
        return main_logits


def load_test_data(data_path):
    """加载测试数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(model_path, base_model_path, device, model_type='cls'):
    """
    加载微调后的模型

    Args:
        model_type: 'cls' (基础LoRA), 'cls_multitask' (多任务), 'simple' (简单版)
    """
    print(f"加载模型 [{model_type}]: {model_path}")

    if model_type == 'cls':
        # 基础分类模型（LoRA + RobertaForSequenceClassification）
        base_model = RobertaForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=len(EMOTION_LIST)
        )
        model = PeftModel.from_pretrained(base_model, model_path)

    elif model_type == 'cls_multitask':
        # 多任务模型
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
        model.load_state_dict(
            torch.load(os.path.join(model_path, "model.pt"), map_location=device),
            strict=False
        )

    elif model_type == 'simple':
        # 简单模型（无LoRA）
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(EMOTION_LIST)
        )

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    model.to(device)
    model.eval()
    return model


def format_conversation(conversation):
    """格式化对话文本（评估时不包含情绪标签）"""
    text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"
    return text.strip()


def predict_emotion(model, tokenizer, conversation, device, max_length=256, model_type='cls'):
    """预测情绪"""
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

        # 根据模型类型获取logits
        if model_type in ['cls', 'simple']:
            logits = outputs.logits
        else:  # cls_multitask
            logits = outputs

        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

    return EMOTION_LIST[pred_idx]


def evaluate_single_model(model, tokenizer, test_data, device, model_name, model_type='cls'):
    """评估单个模型"""
    print(f"\n评估模型: {model_name}")

    predictions = []
    labels = []

    for idx, item in enumerate(tqdm(test_data, desc=f"评估中")):
        conversation = item["conversation"]
        true_emotion = item["main_emotion"]

        try:
            pred_emotion = predict_emotion(
                model, tokenizer, conversation, device, model_type=model_type
            )
        except Exception as e:
            print(f"\n样本 {idx} 预测失败: {e}")
            pred_emotion = "neutral"

        predictions.append(pred_emotion)
        labels.append(true_emotion)

        if idx % 100 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    # 非neutral准确率
    non_neutral_mask = [l != "neutral" for l in labels]
    non_neutral_acc = 0.0
    if any(non_neutral_mask):
        non_neutral_labels = [l for l, m in zip(labels, non_neutral_mask) if m]
        non_neutral_preds = [p for p, m in zip(predictions, non_neutral_mask) if m]
        non_neutral_acc = accuracy_score(non_neutral_labels, non_neutral_preds)

    # 各类别准确率
    emotion_accs = {}
    for emotion in EMOTION_LIST:
        emotion_indices = [i for i, l in enumerate(labels) if l == emotion]
        if emotion_indices:
            emotion_preds = [predictions[i] for i in emotion_indices]
            emotion_labels = [labels[i] for i in emotion_indices]
            emotion_accs[emotion] = accuracy_score(emotion_labels, emotion_preds)
        else:
            emotion_accs[emotion] = 0.0

    results = {
        'model_name': model_name,
        'model_type': model_type,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'non_neutral_accuracy': non_neutral_acc,
        'emotion_accuracies': emotion_accs,
        'predictions': predictions,
        'labels': labels,
        'confusion_matrix': confusion_matrix(labels, predictions, labels=EMOTION_LIST).tolist()
    }

    # 打印结果
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Non-neutral Accuracy: {non_neutral_acc:.4f}")

    return results


def compare_models(results_list, output_dir):
    """对比多个模型的结果"""
    print(f"\n{'='*80}")
    print("Model Comparison Results")
    print(f"{'='*80}")

    # 对比表格
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Non-neutral':<12}")
    print("-" * 60)

    for results in results_list:
        print(f"{results['model_name']:<20} "
              f"{results['accuracy']:<12.4f} "
              f"{results['macro_f1']:<12.4f} "
              f"{results['non_neutral_accuracy']:<12.4f}")

    # 各类别准确率对比
    print(f"\nPer-class Accuracy Comparison:")
    header = f"{'Emotion':<12}"
    for r in results_list:
        header += f"{r['model_name'][:10]:<12}"
    print(header)
    print("-" * len(header))

    for emotion in EMOTION_LIST:
        row = f"{emotion:<12}"
        for results in results_list:
            acc = results['emotion_accuracies'].get(emotion, 0.0)
            row += f"{acc:<12.4f}"
        print(row)

    # 生成可视化
    plot_comparison(results_list, output_dir)

    # 保存结果
    comparison = {
        'models': [r['model_name'] for r in results_list],
        'results': [{
            'model_name': r['model_name'],
            'accuracy': r['accuracy'],
            'macro_f1': r['macro_f1'],
            'weighted_f1': r['weighted_f1'],
            'non_neutral_accuracy': r['non_neutral_accuracy'],
            'emotion_accuracies': r['emotion_accuracies']
        } for r in results_list]
    }

    comparison_path = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {comparison_path}")


def plot_comparison(results_list, output_dir):
    """生成对比可视化图表"""

    # 1. 整体指标对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics = ['accuracy', 'macro_f1', 'non_neutral_accuracy']
    metric_names = ['准确率', 'Macro F1', '非neutral准确率']
    x = np.arange(len(metrics))
    width = 0.25

    for i, results in enumerate(results_list):
        values = [results[m] for m in metrics]
        axes[0].bar(x + i * width, values, width, label=results['model_name'])

    axes[0].set_xlabel('指标')
    axes[0].set_ylabel('分数')
    axes[0].set_title('整体指标对比')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metric_names)
    axes[0].legend()
    axes[0].set_ylim([0, 1.0])

    # 2. 各类别准确率对比
    x = np.arange(len(EMOTION_LIST))
    for i, results in enumerate(results_list):
        values = [results['emotion_accuracies'].get(e, 0) for e in EMOTION_LIST]
        axes[1].bar(x + i * width, values, width, label=results['model_name'])

    axes[1].set_xlabel('情绪类别')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('各类别准确率对比')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(EMOTION_LIST, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 混淆矩阵
    fig, axes = plt.subplots(1, len(results_list), figsize=(6 * len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]

    for i, results in enumerate(results_list):
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=EMOTION_LIST, yticklabels=EMOTION_LIST, ax=axes[i])
        axes[i].set_xlabel('预测标签')
        axes[i].set_ylabel('真实标签')
        axes[i].set_title(f"{results['model_name']}\n准确率: {results['accuracy']:.4f}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("可视化图表已保存")


def main():
    parser = argparse.ArgumentParser(description='情绪分类模型评估')

    parser.add_argument('--base_model_path', type=str,
                        default='../Model/roberta-base',
                        help='基础模型路径')
    parser.add_argument('--test_data', type=str,
                        default='../data/sft_test.json',
                        help='测试数据路径')
    parser.add_argument('--output_dir', type=str,
                        default='../output/evaluation',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--max_samples', type=int, default=0, help='最大评估样本数（0表示全部）')

    # 模型路径
    parser.add_argument('--cls_model', type=str,
                        default='../output/cls_best',
                        help='基础分类模型路径')
    parser.add_argument('--multitask_model', type=str,
                        default='../output/cls_best',
                        help='多任务模型路径')
    parser.add_argument('--simple_model', type=str,
                        default='../output/simple_best',
                        help='简单模型路径')

    # 选择评估哪些模型
    parser.add_argument('--eval_cls', action='store_true', default=True,
                        help='评估基础分类模型')
    parser.add_argument('--no-eval-cls', dest='eval_cls', action='store_false')
    parser.add_argument('--eval_multitask', action='store_true', default=False,
                        help='评估多任务模型')
    parser.add_argument('--eval_simple', action='store_true', default=True,
                        help='评估简单模型')

    # 兼容旧参数
    parser.add_argument('--model_path', type=str, default=None,
                        help='（旧参数）模型路径')
    parser.add_argument('--multitask', action='store_true', default=None,
                        help='（旧参数）使用多任务模型')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载测试数据
    print(f"\n加载测试数据: {args.test_data}")
    test_data = load_test_data(args.test_data)

    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    print(f"测试样本数: {len(test_data)}")

    results_list = []

    # 1. 评估基础分类模型 (cls)
    if args.eval_cls and os.path.exists(args.cls_model):
        try:
            model = load_model(args.cls_model, args.base_model_path, device, model_type='cls')
            results = evaluate_single_model(model, tokenizer, test_data, device, 'cls', 'cls')
            results_list.append(results)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"评估cls模型失败: {e}")

    # 2. 评估多任务模型
    if args.eval_multitask and os.path.exists(args.multitask_model):
        try:
            model = load_model(args.multitask_model, args.base_model_path, device, model_type='cls_multitask')
            results = evaluate_single_model(model, tokenizer, test_data, device, 'cls_multitask', 'cls_multitask')
            results_list.append(results)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"评估multitask模型失败: {e}")

    # 3. 评估简单模型
    if args.eval_simple and os.path.exists(args.simple_model):
        try:
            model = load_model(args.simple_model, args.base_model_path, device, model_type='simple')
            results = evaluate_single_model(model, tokenizer, test_data, device, 'simple', 'simple')
            results_list.append(results)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"评估simple模型失败: {e}")

    if not results_list:
        print("\n没有成功评估任何模型！")
        return

    # 对比模型
    compare_models(results_list, args.output_dir)

    print("\n评估完成！")


if __name__ == '__main__':
    main()
