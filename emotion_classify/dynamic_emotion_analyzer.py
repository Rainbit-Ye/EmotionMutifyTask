#!/usr/bin/env python3
"""
动态情感分析模块
支持实时情感追踪、走向预测、动态参数调整

核心功能：
1. 实时追踪每轮对话的情绪
2. 分析情感走向趋势
3. 动态调整模型敏感度
4. 支持在线学习快速适应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, get_peft_model, TaskType
from collections import deque
import numpy as np
from typing import List, Dict, Tuple, Optional


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
        # 新增：下一轮情绪预测分类头
        self.next_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, return_turn_hidden=False, return_next=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # 主任务：整体情绪
        main_hidden = hidden_states[:, 0, :]
        main_logits = self.main_classifier(main_hidden)

        # 新增：下一轮情绪预测
        next_logits = None
        if return_next:
            # 使用 [CLS] hidden state 预测下一轮情绪
            next_logits = self.next_classifier(main_hidden)

        if return_turn_hidden:
            return main_logits, hidden_states, next_logits
        return main_logits, next_logits


class EmotionTracker:
    """情感追踪器"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.valence_history = deque(maxlen=window_size)
        
    def add(self, emotion: str, confidence: float):
        """添加情感记录"""
        self.history.append({
            'emotion': emotion,
            'confidence': confidence,
            'valence': EMOTION_VALENCE.get(emotion, 0.0)
        })
        self.valence_history.append(EMOTION_VALENCE.get(emotion, 0.0))
    
    def get_trend(self) -> Dict:
        """获取情感趋势"""
        if len(self.history) < 2:
            return {'trend': 'stable', 'direction': 0.0}
        
        valences = list(self.valence_history)
        
        # 线性趋势
        x = np.arange(len(valences))
        if len(x) > 1:
            slope = np.polyfit(x, valences, 1)[0]
        else:
            slope = 0.0
        
        # 判断趋势
        if slope > 0.1:
            trend = 'improving'
        elif slope < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # 情感变化频率
        emotions = [h['emotion'] for h in self.history]
        unique_emotions = len(set(emotions))
        volatility = unique_emotions / len(emotions) if emotions else 0.0
        
        return {
            'trend': trend,
            'direction': slope,
            'volatility': volatility,
            'avg_valence': np.mean(valences),
            'recent_emotion': emotions[-1] if emotions else 'neutral'
        }
    
    def predict_next(self) -> Tuple[str, float]:
        """预测下一轮可能的情感"""
        if len(self.history) < 2:
            return 'neutral', 0.5
        
        trend = self.get_trend()
        avg_valence = trend['avg_valence']
        direction = trend['direction']
        
        # 基于趋势预测
        predicted_valence = avg_valence + direction
        
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
        
        confidence = min(0.9, 0.5 + abs(direction) * 2)
        
        return predicted_emotion, confidence
    
    def detect_anomaly(self) -> Optional[Dict]:
        """检测情感异常突变"""
        if len(self.history) < 3:
            return None
        
        emotions = [h['emotion'] for h in self.history]
        confidences = [h['confidence'] for h in self.history]
        
        # 检测突变：最近一个情感与之前的差异很大
        recent = emotions[-1]
        previous = emotions[-2]
        
        recent_valence = EMOTION_VALENCE.get(recent, 0.0)
        previous_valence = EMOTION_VALENCE.get(previous, 0.0)
        
        valence_diff = abs(recent_valence - previous_valence)
        
        if valence_diff > 1.0:  # 从积极突然变消极，或反之
            return {
                'type': 'sudden_change',
                'from': previous,
                'to': recent,
                'magnitude': valence_diff,
                'confidence': confidences[-1]
            }
        
        return None
    
    def get_summary(self) -> Dict:
        """获取情感摘要"""
        if not self.history:
            return {'dominant': 'neutral', 'count': 0}
        
        emotions = [h['emotion'] for h in self.history]
        from collections import Counter
        emotion_counts = Counter(emotions)
        
        return {
            'dominant': emotion_counts.most_common(1)[0][0],
            'distribution': dict(emotion_counts),
            'count': len(emotions),
            'trend': self.get_trend()
        }


class DynamicEmotionAnalyzer:
    """动态情感分析器"""
    def __init__(self, model_path, base_model_path, device='cuda', 
                 sensitivity=1.0, adaptation_rate=0.01):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sensitivity = sensitivity  # 模型敏感度
        self.adaptation_rate = adaptation_rate  # 在线学习速率
        
        # 加载模型
        print(f"加载动态情感分析模型...")
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
        
        # 情感追踪器
        self.tracker = EmotionTracker()
        
        # 动态参数
        self.confidence_threshold = 0.5
        self.context_weights = {}
        
        print("动态情感分析器加载完成")
    
    def analyze_turn(self, text: str, role: str = "user", predict_next: bool = True) -> Dict:
        """
        分析单轮对话的情感

        Args:
            text: 对话文本
            role: 说话者角色 (user/assistant)
            predict_next: 是否预测下一轮情绪

        Returns:
            情感分析结果
        """
        inputs = self.tokenizer(
            text, return_tensors='pt', max_length=256,
            truncation=True, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            main_logits, next_logits = self.model(**inputs, return_next=predict_next)

            # 应用敏感度调整
            adjusted_logits = main_logits * self.sensitivity
            probs = F.softmax(adjusted_logits, dim=-1)[0]

            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

            # 获取所有情绪概率
            all_probs = {EMOTION_LIST[i]: probs[i].item() for i in range(len(EMOTION_LIST))}

            # 预测下一轮情绪（使用模型）
            next_prediction = None
            if predict_next and next_logits is not None:
                next_probs = F.softmax(next_logits, dim=-1)[0]
                next_pred_idx = torch.argmax(next_probs).item()
                next_confidence = next_probs[next_pred_idx].item()
                next_all_probs = {EMOTION_LIST[i]: next_probs[i].item() for i in range(len(EMOTION_LIST))}
                next_prediction = {
                    'emotion': EMOTION_LIST[next_pred_idx],
                    'confidence': next_confidence,
                    'probabilities': next_all_probs
                }

        predicted_emotion = EMOTION_LIST[pred_idx]

        # 记录到追踪器
        self.tracker.add(predicted_emotion, confidence)

        # 检测异常
        anomaly = self.tracker.detect_anomaly()

        result = {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': all_probs,
            'role': role,
            'anomaly': anomaly,
            'trend': self.tracker.get_trend()
        }

        if next_prediction:
            result['model_next_prediction'] = next_prediction

        return result
    
    def analyze_conversation(self, conversation: List[Dict]) -> Dict:
        """
        分析完整对话的情感走向

        Args:
            conversation: 对话列表 [{"role": "user/assistant", "content": "..."}]

        Returns:
            完整分析结果
        """
        # 重置追踪器
        self.tracker = EmotionTracker()

        turn_results = []

        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            # 格式化文本
            if role == "user":
                text = f"User: {content}"
            else:
                text = f"Assistant: {content}"

            result = self.analyze_turn(text, role, predict_next=False)
            turn_results.append(result)

        # 获取整体预测（包含下一轮预测）
        full_text = ""
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                full_text += f"User: {content}\n"
            else:
                full_text += f"Assistant: {content}\n"

        overall_result = self.analyze_turn(full_text, "overall", predict_next=True)

        # 预测下一轮情感 - 基于效价趋势
        trend_next_emotion, trend_next_confidence = self.tracker.predict_next()

        # 获取模型的下一轮预测
        model_next_prediction = overall_result.get('model_next_prediction', None)

        return {
            'overall_emotion': overall_result['emotion'],
            'overall_confidence': overall_result['confidence'],
            'turn_by_turn': turn_results,
            'emotion_trajectory': [t['emotion'] for t in turn_results],
            'summary': self.tracker.get_summary(),
            'next_prediction': {
                'emotion': model_next_prediction['emotion'] if model_next_prediction else trend_next_emotion,
                'confidence': model_next_prediction['confidence'] if model_next_prediction else trend_next_confidence,
                'probabilities': model_next_prediction['probabilities'] if model_next_prediction else None,
                'method': 'model' if model_next_prediction else 'trend'
            },
            'trend_based_prediction': {
                'emotion': trend_next_emotion,
                'confidence': trend_next_confidence
            }
        }
    
    def adapt(self, true_emotion: str, learning_rate: Optional[float] = None):
        """
        在线适应：根据真实标签微调模型
        
        Args:
            true_emotion: 真实的情绪标签
            learning_rate: 学习率，默认使用 adaptation_rate
        """
        if learning_rate is None:
            learning_rate = self.adaptation_rate
        
        # 这里可以实现更复杂的在线学习
        # 目前简单实现：调整敏感度
        if len(self.tracker.history) > 0:
            last_pred = self.tracker.history[-1]['emotion']
            if last_pred != true_emotion:
                # 预测错误，增加敏感度
                self.sensitivity = min(2.0, self.sensitivity + learning_rate)
            else:
                # 预测正确，可以适当降低敏感度
                self.sensitivity = max(0.5, self.sensitivity - learning_rate * 0.5)
    
    def set_sensitivity(self, value: float):
        """设置模型敏感度"""
        self.sensitivity = max(0.1, min(3.0, value))
    
    def reset(self):
        """重置追踪器"""
        self.tracker = EmotionTracker()
        self.sensitivity = 1.0


def demo():
    """演示动态情感分析"""
    print("=" * 60)
    print("动态情感分析演示")
    print("=" * 60)
    
    # 示例对话
    conversation = [
        {"role": "user", "content": "I'm having a terrible day, everything went wrong."},
        {"role": "assistant", "content": "I'm sorry to hear that. What happened?"},
        {"role": "user", "content": "My car broke down, and I lost my wallet."},
        {"role": "assistant", "content": "That sounds really frustrating. Is there anything I can help with?"},
        {"role": "user", "content": "Actually, someone found my wallet and returned it! I'm so relieved."},
        {"role": "assistant", "content": "That's wonderful news! It's great that there are honest people."},
        {"role": "user", "content": "Yes! And my car just needed a simple fix. I feel much better now!"}
    ]
    
    # 初始化分析器
    analyzer = DynamicEmotionAnalyzer(
        model_path="../output/cls_final",
        base_model_path="../Model/roberta-base"
    )
    
    # 分析对话
    print("\n分析对话情感走向...")
    result = analyzer.analyze_conversation(conversation)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)
    
    print(f"\n整体情绪: {result['overall_emotion']} (置信度: {result['overall_confidence']:.2f})")
    
    print("\n逐轮情感追踪:")
    print("-" * 40)
    for i, turn in enumerate(result['turn_by_turn']):
        emotion = turn['emotion']
        confidence = turn['confidence']
        anomaly = " [异常突变!]" if turn.get('anomaly') else ""
        print(f"  第{i+1}轮 ({turn['role']}): {emotion} ({confidence:.2f}){anomaly}")
    
    print(f"\n情感轨迹: {' → '.join(result['emotion_trajectory'])}")
    
    print(f"\n趋势分析:")
    trend = result['summary']['trend']
    print(f"  趋势方向: {trend['trend']}")
    print(f"  平均效价: {trend['avg_valence']:.2f}")
    print(f"  波动性: {trend['volatility']:.2f}")

    print(f"\n预测下一轮情感:")
    next_pred = result['next_prediction']
    print(f"  方法: {next_pred.get('method', 'unknown')}")
    print(f"  预测情绪: {next_pred['emotion']} (置信度: {next_pred['confidence']:.2f})")
    if next_pred.get('probabilities'):
        print(f"  概率分布:")
        for emo, prob in sorted(next_pred['probabilities'].items(), key=lambda x: -x[1]):
            print(f"    {emo}: {prob:.4f}")

    # 对比趋势预测和模型预测
    if 'trend_based_prediction' in result:
        trend_pred = result['trend_based_prediction']
        print(f"\n  [对比] 基于趋势的预测: {trend_pred['emotion']} (置信度: {trend_pred['confidence']:.2f})")


if __name__ == '__main__':
    demo()
