#!/usr/bin/env python3
"""
下一轮情绪预测评估脚本

评估两种预测方法的效果对比：
1. 模型预测（next_classifier）
2. 趋势预测（效价线性外推）
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter, deque


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}

# 情感效价值
EMOTION_VALENCE = {
    "neutral": 0.0,
    "happiness": 1.0,
    "surprise": 0.3,
    "sadness": -0.8,
    "anger": -0.9,
    "fear": -0.7,
    "disgust": -0.6
}


class MultiTaskEmotionClassifier(torch.nn.Module):
    """多任务情绪分类模型"""
    def __init__(self, base_model_path, num_labels=7, lora_config=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(base_model_path)
        if lora_config is not None:
            self.roberta = get_peft_model(self.roberta, lora_config)

        hidden_size = self.roberta.config.hidden_size
        self.main_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, num_labels)
        )
        self.next_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, num_labels)
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
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.valence_history = deque(maxlen=window_size)

    def add(self, emotion):
        """添加情绪记录"""
        self.valence_history.append(EMOTION_VALENCE.get(emotion, 0.0))

    def predict_next(self):
        """预测下一轮情绪"""
        if len(self.valence_history) < 2:
            return 'neutral', 0.5

        valences = list(self.valence_history)

        # 线性趋势
        x = np.arange(len(valences))
        slope = np.polyfit(x, valences, 1)[0]
        avg_valence = np.mean(valences)

        # 预测下一轮效价
        predicted_valence = avg_valence + slope

        # 映射回情绪
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


def format_conversation(conversation):
    """格式化对话文本"""
    text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"
    return text.strip()


def load_model(model_path, base_model_path, device):
    """加载模型"""
    print(f"Loading model from: {model_path}")

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["query", "value", "key", "dense"],
        lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
    )

    model = MultiTaskEmotionClassifier(base_model_path, num_labels=7, lora_config=lora_config)
    model.load_state_dict(
        torch.load(os.path.join(model_path, "model.pt"), map_location=device),
        strict=False
    )
    model.to(device)
    model.eval()

    return model


def evaluate_next_emotion_prediction(model, tokenizer, test_data, device, max_length=256):
    """评估下一轮情绪预测"""
    model_preds = []
    trend_preds = []
    true_labels = []

    trend_predictor = TrendPredictor()

    print(f"\nEvaluating next emotion prediction on {len(test_data)} samples...")

    for item in tqdm(test_data, desc="Evaluating"):
        conversation = item["conversation"]

        # 需要至少2轮对话才能预测下一轮
        if len(conversation) < 2:
            continue

        # 使用前 n-1 轮预测第 n 轮
        history = conversation[:-1]
        next_turn = conversation[-1]
        true_emotion = next_turn.get("emotion", "neutral")

        # 1. 模型预测
        text = format_conversation(history)
        inputs = tokenizer(
            text, return_tensors='pt', max_length=max_length,
            truncation=True, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            _, next_logits = model(**inputs, return_next=True)
            pred_idx = torch.argmax(next_logits, dim=-1).item()
            model_pred = EMOTION_LIST[pred_idx]

        # 2. 趋势预测
        trend_predictor.reset()
        for turn in history:
            # 简单预测每轮情绪（这里用真实标签，实际应用中需要预测）
            emotion = turn.get("emotion", "neutral")
            trend_predictor.add(emotion)
        trend_pred, _ = trend_predictor.predict_next()

        model_preds.append(model_pred)
        trend_preds.append(trend_pred)
        true_labels.append(true_emotion)

    # 计算指标
    results = {}

    # 模型预测指标
    model_acc = accuracy_score(true_labels, model_preds)
    model_f1 = f1_score(true_labels, model_preds, average='macro', zero_division=0)

    # 趋势预测指标
    trend_acc = accuracy_score(true_labels, trend_preds)
    trend_f1 = f1_score(true_labels, trend_preds, average='macro', zero_division=0)

    # 各类别准确率
    emotion_model_accs = {}
    emotion_trend_accs = {}

    for emotion in EMOTION_LIST:
        indices = [i for i, l in enumerate(true_labels) if l == emotion]
        if indices:
            model_correct = sum(1 for i in indices if model_preds[i] == true_labels[i])
            trend_correct = sum(1 for i in indices if trend_preds[i] == true_labels[i])
            emotion_model_accs[emotion] = model_correct / len(indices)
            emotion_trend_accs[emotion] = trend_correct / len(indices)
        else:
            emotion_model_accs[emotion] = 0.0
            emotion_trend_accs[emotion] = 0.0

    results = {
        'model_accuracy': model_acc,
        'model_f1': model_f1,
        'trend_accuracy': trend_acc,
        'trend_f1': trend_f1,
        'emotion_model_accs': emotion_model_accs,
        'emotion_trend_accs': emotion_trend_accs,
        'model_preds': model_preds,
        'trend_preds': trend_preds,
        'true_labels': true_labels
    }

    return results


def print_results(results):
    """打印结果"""
    print("\n" + "=" * 70)
    print("Next Emotion Prediction Evaluation Results")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Accuracy':<12} {'Macro F1':<12}")
    print("-" * 44)
    print(f"{'Model Prediction':<20} {results['model_accuracy']:<12.4f} {results['model_f1']:<12.4f}")
    print(f"{'Trend Prediction':<20} {results['trend_accuracy']:<12.4f} {results['trend_f1']:<12.4f}")
    print(f"{'Improvement':<20} {results['model_accuracy'] - results['trend_accuracy']:+.4f}      {results['model_f1'] - results['trend_f1']:+.4f}")

    print(f"\nPer-class Accuracy Comparison:")
    print(f"{'Emotion':<12} {'Model':<12} {'Trend':<12} {'Diff':<12}")
    print("-" * 48)
    for emotion in EMOTION_LIST:
        model_acc = results['emotion_model_accs'].get(emotion, 0)
        trend_acc = results['emotion_trend_accs'].get(emotion, 0)
        diff = model_acc - trend_acc
        print(f"{emotion:<12} {model_acc:<12.4f} {trend_acc:<12.4f} {diff:+.4f}")

    # 详细分类报告
    print(f"\nModel Prediction Classification Report:")
    print(classification_report(results['true_labels'], results['model_preds'],
                                target_names=EMOTION_LIST, zero_division=0))


def save_results(results, output_path):
    """保存结果"""
    # 移除不可序列化的字段
    save_data = {
        'model_accuracy': results['model_accuracy'],
        'model_f1': results['model_f1'],
        'trend_accuracy': results['trend_accuracy'],
        'trend_f1': results['trend_f1'],
        'emotion_model_accs': results['emotion_model_accs'],
        'emotion_trend_accs': results['emotion_trend_accs']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate next emotion prediction')

    parser.add_argument('--model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_final',
                        help='Model path')
    parser.add_argument('--base_model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/Model/roberta-base',
                        help='Base model path')
    parser.add_argument('--test_data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/sft_test.json',
                        help='Test data path')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/evaluation',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--max_samples', type=int, default=0, help='Max samples (0 for all)')

    args = parser.parse_args()

    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # 加载模型
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_path, args.base_model_path, device)

    # 加载测试数据
    print(f"\nLoading test data: {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    print(f"Test samples: {len(test_data)}")

    # 评估
    results = evaluate_next_emotion_prediction(model, tokenizer, test_data, device)

    # 打印结果
    print_results(results)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(results, os.path.join(args.output_dir, 'next_emotion_prediction.json'))


if __name__ == '__main__':
    main()
