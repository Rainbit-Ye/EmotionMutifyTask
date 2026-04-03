#!/usr/bin/env python3
"""
数据预处理脚本
改进版：合并所有数据 + 过采样 + 重新划分

关键改进：
1. 合并train/val/test所有原始数据
2. 对非neutral情绪进行过采样（不使用数据增强）
3. 重新划分数据集（8:1:1）
4. 计算平衡权重
"""

import json
import os
import zipfile
import io
from itertools import zip_longest
import random
from collections import Counter
import math

# DailyDialog 情绪标签映射
EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

EMOTION_LIST = list(EMOTION_MAP.values())

# 数据路径配置
DATASET_DIR = "/home/user1/liuduanye/EmotionClassify/Dataset/dailydialog"
OUTPUT_DIR = "/home/user1/liuduanye/EmotionClassify/data"

SPLITS = {
    "train": "train.zip",
    "validation": "validation.zip",
    "test": "test.zip"
}


def load_dailydialog(zip_path):
    """加载 DailyDialog 数据"""
    data = []
    
    with zipfile.ZipFile(zip_path) as zip_file:
        files_list = list(map(str, zip_file.namelist()))
        
        acts_file = next((f for f in files_list if "act" in f.lower()))
        emotions_file = next((f for f in files_list if "emotion" in f.lower()))
        utterances_file = next(
            (f for f in files_list
             if "act" not in f.lower() and "emotion" not in f.lower() and "dialogues" in f.lower())
        )
        
        acts_file = io.TextIOWrapper(zip_file.open(acts_file), encoding="utf-8")
        emotions_file = io.TextIOWrapper(zip_file.open(emotions_file), encoding="utf-8")
        utterances_file = io.TextIOWrapper(zip_file.open(utterances_file), encoding="utf-8")
        
        sentinel = object()
        
        for idx, combo in enumerate(
            zip_longest(acts_file, emotions_file, utterances_file, fillvalue=sentinel)
        ):
            if sentinel in combo:
                break
            
            acts, emos, utts = combo
            
            acts_list = [int(a.strip()) for a in acts.strip().split(" ")]
            emos_list = [int(a.strip()) for a in emos.strip().split(" ")]
            utts_list = [a.strip() for a in utts.strip().strip("__eou__").split("__eou__")]
            
            if len(utts_list) == len(acts_list) and len(acts_list) == len(emos_list):
                data.append({
                    "id": idx,
                    "utterances": utts_list,
                    "emotions": emos_list,
                    "acts": acts_list
                })
    
    return data


def convert_to_sft_format_basic(data):
    """
    基础转换，不进行采样
    """
    sft_data = []
    
    for item in data:
        utterances = item["utterances"]
        emotions = item["emotions"]
        
        # 构建完整对话，每轮都标注情绪
        conversation = []
        for i, (utt, emo) in enumerate(zip(utterances, emotions)):
            role = "user" if i % 2 == 0 else "assistant"
            emotion_label = EMOTION_MAP.get(emo, "neutral")
            conversation.append({
                "role": role,
                "content": utt,
                "emotion": emotion_label
            })
        
        # 计算对话的主要情绪（非neutral的情绪）
        non_neutral_emotions = [EMOTION_MAP.get(e, "neutral") for e in emotions if e != 0]
        if non_neutral_emotions:
            main_emotion = Counter(non_neutral_emotions).most_common(1)[0][0]
        else:
            main_emotion = "neutral"
        
        sample = {
            "conversation": conversation,
            "main_emotion": main_emotion,
            "has_non_neutral": len(non_neutral_emotions) > 0,
            "emotion_counts": Counter([EMOTION_MAP.get(e, "neutral") for e in emotions])
        }
        
        sft_data.append(sample)
    
    return sft_data


def compute_class_weights(data):
    """
    计算平衡权重
    
    设计目标：让每个类别的总贡献相等
    权重 = C / 样本数，其中C为常数
    这样 权重 × 样本数 = C，每个类别的贡献相同
    
    同时使用平方根平滑，避免权重差异过大
    """
    emotion_counts = Counter()
    for item in data:
        emo = item["main_emotion"]
        emotion_counts[emo] += 1
    
    total = sum(emotion_counts.values())
    num_classes = len(EMOTION_LIST)
    
    # 计算平衡权重：让每个类别的贡献相等
    # 使用平方根平滑，避免极端权重
    weights = {}
    for emo in EMOTION_LIST:
        count = emotion_counts.get(emo, 1)
        weight = math.sqrt(total / (num_classes * count))
        weights[emo] = weight
    
    # 归一化权重（使平均权重为1）
    avg_weight = sum(weights.values()) / len(weights)
    for emo in weights:
        weights[emo] = weights[emo] / avg_weight
    
    print(f"\n平衡权重设计（每类贡献相等，平方根平滑）:")
    print(f"目标：每个类别对loss的贡献相同")
    for emo in EMOTION_LIST:
        count = emotion_counts.get(emo, 0)
        weight = weights[emo]
        contribution = count * weight
        print(f"  {emo}: count={count}, weight={weight:.3f}, contribution={contribution:.0f}")
    
    return weights


def convert_to_dpo_format(sft_data):
    """转换为DPO格式"""
    dpo_data = []
    
    for item in sft_data:
        conversation = item["conversation"]
        main_emotion = item["main_emotion"]
        
        # 只为有明确情绪的对话创建偏好对
        if main_emotion == "neutral":
            continue
        
        # 方式1：完整对话的情绪预测
        prompt_parts = []
        for turn in conversation[:-1]:
            role_prefix = "User" if turn["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role_prefix}: {turn['content']}")
        
        prompt = "\n".join(prompt_parts) + "\n"
        
        last_turn = conversation[-1]
        if last_turn["role"] == "assistant":
            chosen = f"Assistant: {last_turn['content']}\n[Emotion: {main_emotion}]"
            
            wrong_emotions = [e for e in EMOTION_MAP.values() if e != main_emotion]
            wrong_emotion = random.choice(wrong_emotions)
            rejected = f"Assistant: {last_turn['content']}\n[Emotion: {wrong_emotion}]"
            
            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "true_emotion": main_emotion,
                "type": "emotion_correctness"
            })
        
        # 方式2：为每个非neutral的轮次创建训练样本
        for i, turn in enumerate(conversation):
            if turn["emotion"] != "neutral":
                context_parts = []
                for j in range(i):
                    prev_turn = conversation[j]
                    role_prefix = "User" if prev_turn["role"] == "user" else "Assistant"
                    context_parts.append(f"{role_prefix}: {prev_turn['content']}")
                
                if context_parts:
                    context = "\n".join(context_parts) + "\n"
                    true_emo = turn["emotion"]
                    
                    chosen_response = f"{turn['role'].capitalize()}: {turn['content']}\n[Emotion: {true_emo}]"
                    
                    wrong_emos = [e for e in EMOTION_MAP.values() if e != true_emo]
                    wrong_emo = random.choice(wrong_emos)
                    rejected_response = f"{turn['role'].capitalize()}: {turn['content']}\n[Emotion: {wrong_emo}]"
                    
                    dpo_data.append({
                        "prompt": context,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                        "true_emotion": true_emo,
                        "type": "turn_level_emotion"
                    })
    
    return dpo_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("DailyDialog 数据预处理（合并+过采样版）")
    print("=" * 60)
    
    # 1. 加载所有原始数据
    print("\n加载所有原始数据...")
    all_data = []
    
    train_data = load_dailydialog(os.path.join(DATASET_DIR, SPLITS["train"]))
    print(f"训练集: {len(train_data)} 条对话")
    all_data.extend(train_data)
    
    val_data = load_dailydialog(os.path.join(DATASET_DIR, SPLITS["validation"]))
    print(f"验证集: {len(val_data)} 条对话")
    all_data.extend(val_data)
    
    test_data = load_dailydialog(os.path.join(DATASET_DIR, SPLITS["test"]))
    print(f"测试集: {len(test_data)} 条对话")
    all_data.extend(test_data)
    
    print(f"总计: {len(all_data)} 条对话")
    
    # 2. 转换为SFT格式
    print("\n转换为SFT格式...")
    all_sft = convert_to_sft_format_basic(all_data)
    
    # 统计原始分布
    emotion_counts = Counter([s["main_emotion"] for s in all_sft])
    print("\n原始情绪分布:")
    for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {emo}: {count} ({count/len(all_sft)*100:.1f}%)")
    
    # 3. 按情绪分组
    print("\n按情绪分组...")
    emotion_groups = {emo: [] for emo in EMOTION_LIST}
    for s in all_sft:
        emotion_groups[s["main_emotion"]].append(s)
    
    # 4. 计算目标数量
    # neutral保留原数量的30%（欠采样）
    # 其他情绪过采样到与happiness相同数量
    neutral_count = emotion_counts["neutral"]
    happiness_count = emotion_counts["happiness"]
    
    target_neutral = int(neutral_count * 0.3)
    target_non_neutral = happiness_count  # 与happiness相同
    
    print(f"\n目标分布:")
    print(f"  neutral: {target_neutral} (欠采样)")
    print(f"  其他情绪: {target_non_neutral} (过采样)")
    
    # 5. 过采样/欠采样
    print("\n进行过采样/欠采样...")
    balanced_data = []
    
    # neutral: 欠采样
    neutral_samples = emotion_groups["neutral"]
    if len(neutral_samples) > target_neutral:
        neutral_samples = random.sample(neutral_samples, target_neutral)
    balanced_data.extend(neutral_samples)
    print(f"neutral: {emotion_counts['neutral']} -> {len(neutral_samples)} (欠采样)")
    
    # 非neutral: 过采样（最大15倍）
    for emo in EMOTION_LIST:
        if emo == "neutral":
            continue
        
        samples = emotion_groups[emo]
        original_count = len(samples)
        
        if original_count < target_non_neutral:
            # 计算过采样倍数
            oversample_ratio = math.ceil(target_non_neutral / original_count)
            oversample_ratio = min(oversample_ratio, 15)  # 最大15倍
            
            # 过采样
            oversampled = samples * oversample_ratio
            # 随机采样到目标数量
            if len(oversampled) > target_non_neutral:
                oversampled = random.sample(oversampled, target_non_neutral)
            
            balanced_data.extend(oversampled)
            print(f"{emo}: {original_count} -> {len(oversampled)} (过采样 {oversample_ratio}x)")
        else:
            balanced_data.extend(samples)
            print(f"{emo}: {original_count} (无需过采样)")
    
    random.shuffle(balanced_data)
    
    # 统计平衡后分布
    balanced_counts = Counter([s["main_emotion"] for s in balanced_data])
    print(f"\n平衡后样本总数: {len(balanced_data)}")
    print("平衡后情绪分布:")
    for emo, count in sorted(balanced_counts.items(), key=lambda x: -x[1]):
        print(f"  {emo}: {count} ({count/len(balanced_data)*100:.1f}%)")
    
    # 6. 重新划分数据集 (80% train, 10% val, 10% test)
    print("\n重新划分数据集 (8:1:1)...")
    
    # 按情绪分层划分，保证每个情绪的比例一致
    emotion_data = {emo: [] for emo in EMOTION_LIST}
    for s in balanced_data:
        emotion_data[s["main_emotion"]].append(s)
    
    train_sft, val_sft, test_sft = [], [], []
    
    for emo in EMOTION_LIST:
        samples = emotion_data[emo]
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        train_sft.extend(samples[:train_end])
        val_sft.extend(samples[train_end:val_end])
        test_sft.extend(samples[val_end:])
    
    random.shuffle(train_sft)
    random.shuffle(val_sft)
    random.shuffle(test_sft)
    
    print(f"训练集: {len(train_sft)}")
    print(f"验证集: {len(val_sft)}")
    print(f"测试集: {len(test_sft)}")
    
    # 7. 计算类别权重（基于训练集）
    class_weights = compute_class_weights(train_sft)
    
    # 8. 转换为DPO格式
    print("\n转换为DPO格式...")
    train_dpo = convert_to_dpo_format(train_sft)
    val_dpo = convert_to_dpo_format(val_sft)
    print(f"训练DPO数据: {len(train_dpo)}")
    print(f"验证DPO数据: {len(val_dpo)}")
    
    # 9. 保存数据
    print("\n保存数据...")
    
    with open(os.path.join(OUTPUT_DIR, "sft_train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_sft, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "sft_val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_sft, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "sft_test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_sft, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "dpo_train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_dpo, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "dpo_val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_dpo, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "emotion_weights.json"), 'w', encoding='utf-8') as f:
        json.dump(class_weights, f, ensure_ascii=False, indent=2)
    
    # 10. 统计信息
    print("\n" + "=" * 60)
    print("最终数据统计")
    print("=" * 60)
    
    print("\n训练集情绪分布:")
    train_counts = Counter([s["main_emotion"] for s in train_sft])
    for emo, count in sorted(train_counts.items(), key=lambda x: -x[1]):
        print(f"  {emo}: {count} ({count/len(train_sft)*100:.1f}%)")
    
    print("\n验证集情绪分布:")
    val_counts = Counter([s["main_emotion"] for s in val_sft])
    for emo, count in sorted(val_counts.items(), key=lambda x: -x[1]):
        print(f"  {emo}: {count} ({count/len(val_sft)*100:.1f}%)")
    
    print("\n测试集情绪分布:")
    test_counts = Counter([s["main_emotion"] for s in test_sft])
    for emo, count in sorted(test_counts.items(), key=lambda x: -x[1]):
        print(f"  {emo}: {count} ({count/len(test_sft)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("输出文件")
    print("=" * 60)
    print(f"SFT训练集: {OUTPUT_DIR}/sft_train.json")
    print(f"SFT验证集: {OUTPUT_DIR}/sft_val.json")
    print(f"SFT测试集: {OUTPUT_DIR}/sft_test.json")
    print(f"DPO训练集: {OUTPUT_DIR}/dpo_train.json")
    print(f"DPO验证集: {OUTPUT_DIR}/dpo_val.json")
    print(f"情绪权重: {OUTPUT_DIR}/emotion_weights.json")
    print("=" * 60)


if __name__ == '__main__':
    random.seed(42)
    main()
