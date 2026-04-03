#!/usr/bin/env python3
"""
情绪分类模型对比实验
比较三种模型：
1. 本文多任务模型 (RoBERTa + 多任务 + 对比学习)
2. 简单分类模型 (RoBERTaForSequenceClassification)
3. RoBERTa-base 本地模型 (未微调，随机初始化分类头)
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
ID2LABEL = {idx: emotion for idx, emotion in enumerate(EMOTION_LIST)}
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}

# 路径配置
BASE_MODEL_PATH = "/home/user1/liuduanye/EmotionClassify/Model/roberta-base"
MULTITASK_MODEL_PATH = "/home/user1/liuduanye/EmotionClassify/output/cls_final"
SIMPLE_MODEL_PATH = "/home/user1/liuduanye/EmotionClassify/output/simple_final"
TEST_DATA_PATH = "/home/user1/liuduanye/EmotionClassify/data/sft_test.json"
OUTPUT_DIR = "/home/user1/liuduanye/EmotionClassify/output/comparison"


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

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        main_hidden = hidden_states[:, 0, :]
        main_logits = self.main_classifier(main_hidden)
        return main_logits


def load_test_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_conversation(conversation):
    text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"
    return text.strip()


def evaluate_model(model, test_data, tokenizer, device, model_name="Model", is_multitask=True):
    """评估模型"""
    predictions = []
    labels = []

    for item in tqdm(test_data, desc=f"评估 {model_name}"):
        text = format_conversation(item["conversation"])
        true_label = LABEL2ID[item["main_emotion"]]

        inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if is_multitask:
                logits = model(**inputs)
                pred_idx = torch.argmax(logits, dim=-1).item()
            else:
                outputs = model(**inputs)
                pred_idx = torch.argmax(outputs.logits, dim=-1).item()

        predictions.append(pred_idx)
        labels.append(true_label)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')

    cm = confusion_matrix(labels, predictions)
    class_acc = {EMOTION_LIST[i]: cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(EMOTION_LIST))}

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'class_accuracy': class_acc
    }


def run_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载测试数据
    print(f"\n加载测试数据: {TEST_DATA_PATH}")
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"测试样本数: {len(test_data)}")

    tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ========== 1. 多任务模型 ==========
    if os.path.exists(os.path.join(MULTITASK_MODEL_PATH, "model.pt")):
        print("\n" + "=" * 70)
        print("1. 多任务模型 (RoBERTa + 多任务学习 + 对比学习 + Focal Loss)")
        print("=" * 70)

        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )
        multitask_model = MultiTaskEmotionClassifier(BASE_MODEL_PATH, num_labels=7, lora_config=lora_config)
        multitask_model.load_state_dict(
            torch.load(os.path.join(MULTITASK_MODEL_PATH, "model.pt"), map_location=device),
            strict=False
        )
        multitask_model.to(device)
        multitask_model.eval()

        results['Multi-task Model'] = evaluate_model(
            multitask_model, test_data, tokenizer, device, "多任务模型", is_multitask=True
        )
        print(f"准确率: {results['Multi-task Model']['accuracy']*100:.2f}%")
    else:
        print(f"\n多任务模型不存在: {MULTITASK_MODEL_PATH}")

    # ========== 2. 简单分类模型 ==========
    if os.path.exists(os.path.join(SIMPLE_MODEL_PATH, "config.json")):
        print("\n" + "=" * 70)
        print("2. 简单分类模型 (RoBERTaForSequenceClassification)")
        print("=" * 70)

        simple_model = RobertaForSequenceClassification.from_pretrained(SIMPLE_MODEL_PATH)
        simple_model.to(device)
        simple_model.eval()

        results['Simple Model'] = evaluate_model(
            simple_model, test_data, tokenizer, device, "简单分类模型", is_multitask=False
        )
        print(f"准确率: {results['Simple Model']['accuracy']*100:.2f}%")
    else:
        print(f"\n简单分类模型不存在: {SIMPLE_MODEL_PATH}")

    # ========== 3. RoBERTa-base 未微调 ==========
    print("\n" + "=" * 70)
    print("3. RoBERTa-base 未微调 (分类头随机初始化)")
    print("=" * 70)

    baseline_model = RobertaForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH, num_labels=len(EMOTION_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )
    baseline_model.to(device)
    baseline_model.eval()

    results['RoBERTa-base (unfine-tuned)'] = evaluate_model(
        baseline_model, test_data, tokenizer, device, "未微调RoBERTa", is_multitask=False
    )
    print(f"准确率: {results['RoBERTa-base (unfine-tuned)']['accuracy']*100:.2f}%")

    # ========== 打印对比结果 ==========
    print("\n" + "=" * 70)
    print("对比结果汇总")
    print("=" * 70)

    print(f"\n{'模型':<35} {'准确率':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 70)

    for model_name, metrics in results.items():
        print(f"{model_name:<35} {metrics['accuracy']*100:>9.2f}% {metrics['macro_f1']*100:>9.2f}% {metrics['weighted_f1']*100:>11.2f}%")

    # 各类别准确率对比
    if len(results) >= 2:
        print("\n" + "-" * 70)
        print("各类别准确率对比:")
        print(f"{'情绪':<12}", end="")
        for model_name in results.keys():
            print(f" {model_name[:12]:>12}", end="")
        print()
        print("-" * 70)

        for emotion in EMOTION_LIST:
            print(f"{emotion:<12}", end="")
            for metrics in results.values():
                acc = metrics['class_accuracy'].get(emotion, 0) * 100
                print(f" {acc:>11.2f}%", end="")
            print()

    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {results_path}")


if __name__ == '__main__':
    run_comparison()
