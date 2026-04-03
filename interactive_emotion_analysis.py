#!/usr/bin/env python3
"""
交互式动态情感分析
支持实时对话情感追踪和预测

使用方法：
python interactive_emotion_analysis.py --model_path output/cls_final
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
from collections import deque
import numpy as np


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EMOTION_VALENCE = {
    "neutral": 0.0,
    "happiness": 1.0,
    "surprise": 0.3,
    "sadness": -0.8,
    "anger": -0.9,
    "fear": -0.7,
    "disgust": -0.6
}


class MultiTaskEmotionClassifier(nn.Module):
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

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        main_hidden = hidden_states[:, 0, :]
        main_logits = self.main_classifier(main_hidden)
        return main_logits


class InteractiveEmotionAnalyzer:
    """交互式情感分析器"""
    
    def __init__(self, model_path, base_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("Loading model...")
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )
        
        self.model = MultiTaskEmotionClassifier(base_model_path, num_labels=7, lora_config=lora_config)
        self.model.load_state_dict(
            torch.load(f"{model_path}/model.pt", map_location=self.device),
            strict=False
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 对话历史
        self.conversation = []
        self.emotion_history = []
        self.valence_history = []
        
        print("Model loaded!\n")
    
    def predict_emotion(self, text):
        """预测情感"""
        inputs = self.tokenizer(
            text, return_tensors='pt', max_length=256,
            truncation=True, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs)
            probs = F.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            all_probs = {EMOTION_LIST[i]: probs[i].item() for i in range(len(EMOTION_LIST))}
        
        return EMOTION_LIST[pred_idx], confidence, all_probs
    
    def add_turn(self, role, content):
        """添加一轮对话"""
        self.conversation.append({"role": role, "content": content})
        
        # 预测情感
        if role == "user":
            text = f"User: {content}"
        else:
            text = f"Assistant: {content}"
        
        emotion, confidence, probs = self.predict_emotion(text)
        
        self.emotion_history.append({
            'role': role,
            'emotion': emotion,
            'confidence': confidence,
            'probs': probs
        })
        self.valence_history.append(EMOTION_VALENCE.get(emotion, 0.0))
        
        return emotion, confidence, probs
    
    def get_trend(self):
        """获取趋势"""
        if len(self.valence_history) < 2:
            return 'stable', 0.0
        
        valences = list(self.valence_history)
        x = np.arange(len(valences))
        slope = np.polyfit(x, valences, 1)[0]
        
        if slope > 0.1:
            trend = 'IMPROVING ↗'
        elif slope < -0.1:
            trend = 'DECLINING ↘'
        else:
            trend = 'STABLE →'
        
        return trend, slope
    
    def predict_next_emotion(self):
        """预测下一轮情感"""
        if len(self.emotion_history) < 2:
            return 'neutral', 0.5
        
        trend, slope = self.get_trend()
        avg_valence = np.mean(self.valence_history) if self.valence_history else 0.0
        
        predicted_valence = avg_valence + slope
        
        if predicted_valence > 0.5:
            return 'happiness', 0.7
        elif predicted_valence > 0.1:
            return 'surprise', 0.6
        elif predicted_valence < -0.5:
            return 'sadness', 0.7
        elif predicted_valence < -0.3:
            return 'anger', 0.6
        else:
            return 'neutral', 0.5
    
    def detect_anomaly(self):
        """检测异常"""
        if len(self.emotion_history) < 2:
            return None
        
        current = self.emotion_history[-1]
        previous = self.emotion_history[-2]
        
        curr_valence = EMOTION_VALENCE.get(current['emotion'], 0.0)
        prev_valence = EMOTION_VALENCE.get(previous['emotion'], 0.0)
        
        diff = abs(curr_valence - prev_valence)
        
        if diff > 1.0:
            return f"Anomaly detected: {previous['emotion']} → {current['emotion']} (change: {diff:.1f})"
        return None
    
    def get_overall_emotion(self):
        """获取整体情感"""
        if not self.conversation:
            return 'neutral', 0.0
        
        # 拼接完整对话
        text = ""
        for turn in self.conversation:
            if turn["role"] == "user":
                text += f"User: {turn['content']}\n"
            else:
                text += f"Assistant: {turn['content']}\n"
        
        return self.predict_emotion(text)
    
    def reset(self):
        """重置对话"""
        self.conversation = []
        self.emotion_history = []
        self.valence_history = []
        print("\n=== Conversation Reset ===\n")
    
    def print_summary(self):
        """打印摘要"""
        if not self.emotion_history:
            print("No conversation yet.")
            return
        
        print("\n" + "=" * 60)
        print("EMOTION ANALYSIS SUMMARY")
        print("=" * 60)
        
        # 整体情感
        overall, conf, _ = self.get_overall_emotion()
        print(f"\nOverall Emotion: {overall.upper()} (confidence: {conf:.2f})")
        
        # 情感轨迹
        trajectory = " → ".join([e['emotion'] for e in self.emotion_history])
        print(f"\nEmotion Trajectory:")
        print(f"  {trajectory}")
        
        # 趋势
        trend, slope = self.get_trend()
        print(f"\nTrend: {trend} (slope: {slope:.2f})")
        
        # 情感分布
        from collections import Counter
        emotions = [e['emotion'] for e in self.emotion_history]
        dist = Counter(emotions)
        print(f"\nEmotion Distribution:")
        for emo, count in dist.most_common():
            print(f"  {emo}: {count} ({count/len(emotions)*100:.0f}%)")
        
        # 下一轮预测
        next_emo, next_conf = self.predict_next_emotion()
        print(f"\nPredicted Next Emotion: {next_emo} (confidence: {next_conf:.2f})")
        
        print("=" * 60)


def print_help():
    """打印帮助"""
    print("\nCommands:")
    print("  u <text>  - Add user message")
    print("  a <text>  - Add assistant message")
    print("  s         - Show summary")
    print("  r         - Reset conversation")
    print("  h         - Show this help")
    print("  q         - Quit")
    print()


def interactive_mode(model_path, base_model_path, device='cuda'):
    """交互模式"""
    analyzer = InteractiveEmotionAnalyzer(model_path, base_model_path, device)
    
    print("=" * 60)
    print("INTERACTIVE EMOTION ANALYSIS")
    print("=" * 60)
    print_help()
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            
            # 解析命令
            if user_input == 'q' or user_input == 'quit':
                print("Goodbye!")
                break
            
            elif user_input == 'h' or user_input == 'help':
                print_help()
            
            elif user_input == 's' or user_input == 'summary':
                analyzer.print_summary()
            
            elif user_input == 'r' or user_input == 'reset':
                analyzer.reset()
            
            elif user_input.startswith('u '):
                content = user_input[2:]
                emotion, confidence, probs = analyzer.add_turn('user', content)
                
                # 打印结果
                top_3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
                anomaly = analyzer.detect_anomaly()
                
                print(f"\n  Emotion: {emotion.upper()} ({confidence:.2f})")
                print(f"  Top 3: {', '.join([f'{e}({c:.2f})' for e, c in top_3])}")
                
                trend, _ = analyzer.get_trend()
                print(f"  Trend: {trend}")
                
                if anomaly:
                    print(f"  ⚠️  {anomaly}")
                
                next_emo, next_conf = analyzer.predict_next_emotion()
                print(f"  Predicted next: {next_emo} ({next_conf:.2f})")
                print()
            
            elif user_input.startswith('a '):
                content = user_input[2:]
                emotion, confidence, probs = analyzer.add_turn('assistant', content)
                
                top_3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
                
                print(f"\n  Emotion: {emotion.upper()} ({confidence:.2f})")
                print(f"  Top 3: {', '.join([f'{e}({c:.2f})' for e, c in top_3])}")
                print()
            
            else:
                # 默认当作用户输入
                emotion, confidence, probs = analyzer.add_turn('user', user_input)
                
                top_3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
                anomaly = analyzer.detect_anomaly()
                
                print(f"\n  Emotion: {emotion.upper()} ({confidence:.2f})")
                print(f"  Top 3: {', '.join([f'{e}({c:.2f})' for e, c in top_3])}")
                
                trend, _ = analyzer.get_trend()
                print(f"  Trend: {trend}")
                
                if anomaly:
                    print(f"  ⚠️  {anomaly}")
                
                next_emo, next_conf = analyzer.predict_next_emotion()
                print(f"  Predicted next: {next_emo} ({next_conf:.2f})")
                print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def demo_mode(model_path, base_model_path, device='cuda'):
    """演示模式"""
    analyzer = InteractiveEmotionAnalyzer(model_path, base_model_path, device)
    
    print("=" * 60)
    print("DEMO MODE")
    print("=" * 60)
    
    # 演示对话
    demo_conversation = [
        ("user", "I'm having a terrible day, everything went wrong."),
        ("assistant", "I'm sorry to hear that. What happened?"),
        ("user", "My car broke down, and I lost my wallet."),
        ("assistant", "That sounds really frustrating. Is there anything I can help with?"),
        ("user", "Actually, someone found my wallet and returned it! I'm so relieved."),
        ("assistant", "That's wonderful news! It's great that there are honest people."),
        ("user", "Yes! And my car just needed a simple fix. I feel much better now!"),
    ]
    
    print("\nProcessing demo conversation...\n")
    
    for role, content in demo_conversation:
        emotion, confidence, probs = analyzer.add_turn(role, content)
        
        role_label = "USER" if role == "user" else "ASSISTANT"
        print(f"[{role_label}] {content}")
        print(f"  → Emotion: {emotion.upper()} ({confidence:.2f})")
        
        anomaly = analyzer.detect_anomaly()
        if anomaly:
            print(f"  → ⚠️  {anomaly}")
        
        print()
    
    analyzer.print_summary()


def main():
    parser = argparse.ArgumentParser(description='Interactive Emotion Analysis')
    parser.add_argument('--model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_final',
                        help='Model path')
    parser.add_argument('--base_model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/Model/roberta-base',
                        help='Base model path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mode(args.model_path, args.base_model_path, args.device)
    else:
        interactive_mode(args.model_path, args.base_model_path, args.device)


if __name__ == '__main__':
    main()
