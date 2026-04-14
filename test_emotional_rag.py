#!/usr/bin/env python3
"""
测试 Emotional RAG 图标推荐效果

使用验证集测试:
- 输入: AAC图标序列
- 输出: 情感识别 + 下一情感预测 + 图标推荐
- 评估: 推荐的图标是否与后续输入相关
"""

import json
import sys
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from aac_emotion_pipeline import AACEmotionPipeline

def load_val_data(path, limit=100):
    """加载验证数据"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:limit]

def test_single_sample(pipeline, sample, show_detail=True):
    """测试单个样本"""
    labels = sample['labels']
    sentence = sample['sentence']
    
    # 处理
    result = pipeline.process(labels)
    
    if show_detail:
        print(f"\n{'='*60}")
        print(f"输入图标: {labels}")
        print(f"翻译: {result['translation']['sentence']}")
        print(f"情感: {result['emotion']['single']} (置信度: {result['emotion']['single_confidence']:.0%})")
        print(f"预测下一情感: {result['prediction']['next_emotion']}")
        
        icons = result['icon_recommendations']
        rag = icons.get('emotional_rag', {})
        
        print(f"\nEmotional RAG (λ={rag.get('lambda', 0.3):.1f}):")
        print(f"  Q_orig: {rag.get('Q_orig', '')}")
        print(f"  Q_emo: {rag.get('Q_emo', '')}")
        
        print(f"\n推荐图标:")
        for a in icons.get('actions', [])[:3]:
            print(f"  [Action] {a['label']}: sim_combined={a['sim_combined']:.3f} (orig={a['sim_orig']:.3f}, emo={a['sim_emo']:.3f})")
        for e in icons.get('entities', [])[:3]:
            print(f"  [Entity] {e['label']}: sim_combined={e['sim_combined']:.3f} (orig={e['sim_orig']:.3f}, emo={e['sim_emo']:.3f})")
    
    return result

def test_multi_turn(pipeline, samples, show_detail=True):
    """测试多轮对话场景"""
    pipeline.reset()
    
    print("\n" + "="*60)
    print("多轮对话测试")
    print("="*60)
    
    for i, sample in enumerate(samples[:5]):
        labels = sample['labels']
        
        result = pipeline.process(labels)
        
        print(f"\n--- Turn {i+1} ---")
        print(f"输入: {labels}")
        print(f"翻译: {result['translation']['sentence']}")
        print(f"情感: {result['emotion']['single']} → 预测下一情感: {result['prediction']['next_emotion']}")
        
        icons = result['icon_recommendations']
        rag = icons.get('emotional_rag', {})
        print(f"RAG: λ={rag.get('lambda', 0.3)}, target={rag.get('target_emotion')}")
        
        # 显示推荐的图标
        rec_labels = [a['label'] for a in icons.get('actions', [])[:2]]
        rec_labels += [e['label'] for e in icons.get('entities', [])[:2]]
        print(f"推荐: {rec_labels[:4]}")

def test_emotion_scenarios(pipeline):
    """测试不同情感场景"""
    pipeline.reset()
    
    scenarios = [
        {
            "name": "快乐场景",
            "inputs": [
                ["I", "am", "happy"],
                ["I", "want_to", "play"],
                ["I", "love_to", "celebrate"],
            ]
        },
        {
            "name": "悲伤场景",
            "inputs": [
                ["I", "feel", "sad"],
                ["I", "need", "help"],
                ["I", "want", "comfort"],
            ]
        },
        {
            "name": "恐惧场景",
            "inputs": [
                ["I", "am", "scared"],
                ["I", "need", "safety"],
                ["I", "want_to", "hide"],
            ]
        },
        {
            "name": "愤怒场景",
            "inputs": [
                ["I", "am", "angry"],
                ["I", "need_to", "calm_down"],
                ["I", "feel", "frustrated"],
            ]
        }
    ]
    
    print("\n" + "="*60)
    print("情感场景测试")
    print("="*60)
    
    for scenario in scenarios:
        pipeline.reset()
        print(f"\n### {scenario['name']} ###")
        
        for symbols in scenario['inputs']:
            result = pipeline.process(symbols)
            
            icons = result['icon_recommendations']
            rag = icons.get('emotional_rag', {})
            
            print(f"\n输入: {symbols}")
            print(f"翻译: {result['translation']['sentence']}")
            print(f"情感: {result['emotion']['single']} → 预测: {result['prediction']['next_emotion']}")
            print(f"RAG: λ={rag.get('lambda', 0.3)}, target={rag.get('target_emotion')}")
            
            recs = []
            for a in icons.get('actions', [])[:2]:
                recs.append(f"{a['label']}({a['sim_combined']:.2f})")
            for e in icons.get('entities', [])[:2]:
                recs.append(f"{e['label']}({e['sim_combined']:.2f})")
            print(f"推荐: {recs}")

def analyze_lambda_effect(pipeline, sample):
    """分析 λ 参数的影响"""
    labels = sample['labels']
    
    print("\n" + "="*60)
    print(f"λ 参数影响分析 - 输入: {labels}")
    print("="*60)
    
    for lambda_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        pipeline.reset()
        # 需要修改 pipeline 或 predictor 来传入 lambda
        # 这里简化展示
        result = pipeline.process(labels)
        icons = result['icon_recommendations']
        
        actions = [a['label'] for a in icons.get('actions', [])[:3]]
        entities = [e['label'] for e in icons.get('entities', [])[:3]]
        
        print(f"\nλ={lambda_val:.1f}:")
        print(f"  Actions: {actions}")
        print(f"  Entities: {entities}")

def main():
    # 加载验证数据
    val_path = os.path.join(SCRIPT_DIR, 'AAC2Text/data/processed/val_data.json')
    val_data = load_val_data(val_path, limit=20)
    
    print(f"加载验证数据: {len(val_data)} 条")
    
    # 初始化 Pipeline
    pipeline = AACEmotionPipeline(
        aac_model_path=os.path.join(SCRIPT_DIR, 'AAC2Text/checkpoints/aac_model'),
        aac_base_model_path='/home/user1/liuduanye/AgentPipeline/qwen/Qwen2_5-1_5B-Instruct',
        emotion_model_path=os.path.join(SCRIPT_DIR, 'output/cls_final'),
        emotion_base_model_path=os.path.join(SCRIPT_DIR, 'Model/roberta-base')
    )
    
    # 1. 测试单个样本
    print("\n" + "="*60)
    print("单样本测试")
    print("="*60)
    test_single_sample(pipeline, val_data[0])
    test_single_sample(pipeline, val_data[5])
    
    # 2. 测试多轮对话
    test_multi_turn(pipeline, val_data)
    
    # 3. 测试情感场景
    test_emotion_scenarios(pipeline)

if __name__ == '__main__':
    main()
