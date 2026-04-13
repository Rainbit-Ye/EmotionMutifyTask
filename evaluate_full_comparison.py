#!/usr/bin/env python3
"""
完整模型对比评估脚本

对比内容：
1. 三种模型的整体情绪分类效果 (cls, multitask, simple)
2. 两种下一轮情绪预测方法 (model vs trend)
3. 完整的统计分析和可视化
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from collections import Counter, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}

EMOTION_VALENCE = {
    "neutral": 0.0, "happiness": 1.0, "surprise": 0.3,
    "sadness": -0.8, "anger": -0.9, "fear": -0.7, "disgust": -0.6
}


class MultiTaskEmotionClassifier(nn.Module):
    """多任务情绪分类模型"""
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


class TrendPredictor:
    """基于效价趋势的预测器"""
    def __init__(self):
        self.valence_history = deque(maxlen=10)

    def add(self, emotion):
        self.valence_history.append(EMOTION_VALENCE.get(emotion, 0.0))

    def predict_next(self):
        if len(self.valence_history) < 2:
            return 'neutral', 0.5

        valences = list(self.valence_history)
        x = np.arange(len(valences))
        slope = np.polyfit(x, valences, 1)[0]
        avg_valence = np.mean(valences)
        predicted_valence = avg_valence + slope

        if predicted_valence > 0.5:
            predicted_emotion = 'happiness'
        elif predicted_valence > 0.1:
            predicted_emotion = 'surprise'
        elif predicted_valence < -0.5:
            predicted_emotion = 'sadness'
        elif predicted_valence < -0.3:
            predicted_emotion = 'anger'
        else:
            predicted_emotion = 'neutral'

        confidence = min(0.9, 0.5 + abs(slope) * 2)
        return predicted_emotion, confidence

    def reset(self):
        self.valence_history.clear()


def format_conversation(conversation, include_last=True):
    """格式化对话文本"""
    text = ""
    turns = conversation if include_last else conversation[:-1]
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"
    return text.strip()


def load_model(model_path, base_model_path, device, model_type='cls'):
    """加载模型"""
    print(f"Loading model [{model_type}]: {model_path}")

    if model_type == 'cls':
        base_model = RobertaForSequenceClassification.from_pretrained(
            base_model_path, num_labels=len(EMOTION_LIST)
        )
        model = PeftModel.from_pretrained(base_model, model_path)

    elif model_type == 'multitask':
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )
        model = MultiTaskEmotionClassifier(base_model_path, num_labels=7, lora_config=lora_config)
        model.load_state_dict(
            torch.load(os.path.join(model_path, "model.pt"), map_location=device),
            strict=False
        )

    elif model_type == 'simple':
        model = RobertaForSequenceClassification.from_pretrained(
            model_path, num_labels=len(EMOTION_LIST)
        )

    model.to(device)
    model.eval()
    return model


def evaluate_main_emotion(model, tokenizer, test_data, device, model_type):
    """评估整体情绪分类"""
    predictions = []
    labels = []

    for item in tqdm(test_data, desc=f"Evaluating {model_type}"):
        conversation = item["conversation"]
        true_emotion = item["main_emotion"]

        text = format_conversation(conversation)
        inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if model_type == 'multitask':
                logits = model(**inputs)
            else:
                outputs = model(**inputs)
                logits = outputs.logits

            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_emotion = EMOTION_LIST[pred_idx]

        predictions.append(pred_emotion)
        labels.append(true_emotion)

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
        indices = [i for i, l in enumerate(labels) if l == emotion]
        if indices:
            correct = sum(1 for i in indices if predictions[i] == labels[i])
            emotion_accs[emotion] = correct / len(indices)
        else:
            emotion_accs[emotion] = 0.0

    return {
        'model_name': model_type,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'non_neutral_accuracy': non_neutral_acc,
        'emotion_accuracies': emotion_accs,
        'confusion_matrix': confusion_matrix(labels, predictions, labels=EMOTION_LIST).tolist()
    }


def evaluate_next_emotion(model, tokenizer, test_data, device, model_type):
    """评估下一轮情绪预测"""
    model_preds = []
    trend_preds = []
    true_labels = []

    trend_predictor = TrendPredictor()

    for item in tqdm(test_data, desc=f"Evaluating next emotion [{model_type}]"):
        conversation = item["conversation"]

        if len(conversation) < 2:
            continue

        history = conversation[:-1]
        next_turn = conversation[-1]
        true_emotion = next_turn.get("emotion", "neutral")

        # 模型预测
        text = format_conversation(conversation, include_last=False)
        inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if model_type == 'multitask':
                _, next_logits = model(**inputs, return_next=True)
                pred_idx = torch.argmax(next_logits, dim=-1).item()
                model_pred = EMOTION_LIST[pred_idx]
            else:
                # 非multitask模型，使用主分类器预测
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                pred_idx = torch.argmax(logits, dim=-1).item()
                model_pred = EMOTION_LIST[pred_idx]

        # 趋势预测
        trend_predictor.reset()
        for turn in history:
            emotion = turn.get("emotion", "neutral")
            trend_predictor.add(emotion)
        trend_pred, _ = trend_predictor.predict_next()

        model_preds.append(model_pred)
        trend_preds.append(trend_pred)
        true_labels.append(true_emotion)

    model_acc = accuracy_score(true_labels, model_preds)
    model_f1 = f1_score(true_labels, model_preds, average='macro', zero_division=0)
    trend_acc = accuracy_score(true_labels, trend_preds)
    trend_f1 = f1_score(true_labels, trend_preds, average='macro', zero_division=0)

    return {
        'model_name': model_type,
        'model_accuracy': model_acc,
        'model_f1': model_f1,
        'trend_accuracy': trend_acc,
        'trend_f1': trend_f1,
        'improvement_acc': model_acc - trend_acc,
        'improvement_f1': model_f1 - trend_f1
    }


def print_comparison_results(main_results, next_results):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    # 1. 整体情绪分类对比
    print("\n1. Main Emotion Classification")
    print("-" * 60)
    print(f"{'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Non-neutral':<12}")
    print("-" * 60)
    for r in main_results:
        print(f"{r['model_name']:<15} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['non_neutral_accuracy']:<12.4f}")

    # 2. 各类别准确率对比
    print("\n2. Per-class Accuracy")
    print("-" * 80)
    header = f"{'Emotion':<12}"
    for r in main_results:
        header += f"{r['model_name'][:10]:<12}"
    print(header)
    print("-" * len(header))

    for emotion in EMOTION_LIST:
        row = f"{emotion:<12}"
        for r in main_results:
            acc = r['emotion_accuracies'].get(emotion, 0.0)
            row += f"{acc:<12.4f}"
        print(row)

    # 3. 下一轮情绪预测对比
    if next_results:
        print("\n" + "=" * 80)
        print("NEXT EMOTION PREDICTION")
        print("=" * 80)
        print(f"\n{'Model':<15} {'Model Acc':<12} {'Trend Acc':<12} {'Improvement':<12}")
        print("-" * 60)
        for r in next_results:
            print(f"{r['model_name']:<15} {r['model_accuracy']:<12.4f} {r['trend_accuracy']:<12.4f} {r['improvement_acc']:+.4f}")


def plot_comparison(main_results, output_dir):
    """绘制对比图表"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 整体指标对比
    models = [r['model_name'] for r in main_results]
    x = np.arange(len(models))
    width = 0.25

    accs = [r['accuracy'] for r in main_results]
    f1s = [r['macro_f1'] for r in main_results]
    non_neutral = [r['non_neutral_accuracy'] for r in main_results]

    axes[0].bar(x - width, accs, width, label='Accuracy')
    axes[0].bar(x, f1s, width, label='Macro F1')
    axes[0].bar(x + width, non_neutral, width, label='Non-neutral')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Main Emotion Classification')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].set_ylim([0, 1.0])

    # 2. 各类别准确率对比
    x = np.arange(len(EMOTION_LIST))
    width = 0.8 / len(main_results)

    for i, r in enumerate(main_results):
        values = [r['emotion_accuracies'].get(e, 0) for e in EMOTION_LIST]
        axes[1].bar(x + i * width, values, width, label=r['model_name'])

    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-class Accuracy')
    axes[1].set_xticks(x + width * (len(main_results) - 1) / 2)
    axes[1].set_xticklabels(EMOTION_LIST, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, 'full_model_comparison.png')}")


def main():
    parser = argparse.ArgumentParser(description='Full model comparison evaluation')

    parser.add_argument('--base_model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/Model/roberta-base')
    parser.add_argument('--test_data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/sft_test.json')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=0)

    # 模型路径
    parser.add_argument('--cls_model', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_best')
    parser.add_argument('--multitask_model', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_final')
    parser.add_argument('--simple_model', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/simple_best')

    # 选择评估哪些模型
    parser.add_argument('--eval_cls', action='store_true', default=True)
    parser.add_argument('--eval_multitask', action='store_true', default=True)
    parser.add_argument('--eval_simple', action='store_true', default=True)
    parser.add_argument('--eval_next', action='store_true', default=True, help='Evaluate next emotion prediction')

    args = parser.parse_args()

    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载测试数据
    print(f"\nLoading test data: {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    print(f"Test samples: {len(test_data)}")

    main_results = []
    next_results = []

    # 评估 cls 模型
    if args.eval_cls and os.path.exists(args.cls_model):
        try:
            model = load_model(args.cls_model, args.base_model_path, device, 'cls')
            result = evaluate_main_emotion(model, tokenizer, test_data, device, 'cls')
            main_results.append(result)

            if args.eval_next:
                next_result = evaluate_next_emotion(model, tokenizer, test_data, device, 'cls')
                next_results.append(next_result)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Failed to evaluate cls model: {e}")

    # 评估 multitask 模型
    if args.eval_multitask and os.path.exists(args.multitask_model):
        try:
            model = load_model(args.multitask_model, args.base_model_path, device, 'multitask')
            result = evaluate_main_emotion(model, tokenizer, test_data, device, 'multitask')
            main_results.append(result)

            if args.eval_next:
                next_result = evaluate_next_emotion(model, tokenizer, test_data, device, 'multitask')
                next_results.append(next_result)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Failed to evaluate multitask model: {e}")

    # 评估 simple 模型
    if args.eval_simple and os.path.exists(args.simple_model):
        try:
            model = load_model(args.simple_model, args.base_model_path, device, 'simple')
            result = evaluate_main_emotion(model, tokenizer, test_data, device, 'simple')
            main_results.append(result)

            if args.eval_next:
                next_result = evaluate_next_emotion(model, tokenizer, test_data, device, 'simple')
                next_results.append(next_result)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Failed to evaluate simple model: {e}")

    if not main_results:
        print("\nNo models evaluated successfully!")
        return

    # 打印结果
    print_comparison_results(main_results, next_results)

    # 保存结果
    comparison = {
        'main_emotion_classification': main_results,
        'next_emotion_prediction': next_results
    }

    output_path = os.path.join(args.output_dir, 'full_comparison.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # 绘制图表
    plot_comparison(main_results, args.output_dir)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
