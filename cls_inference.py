#!/usr/bin/env python3
"""
情绪分类推理脚本
支持单任务和多任务模型

推理时输入不包含情绪标签，模型需要自己预测
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from peft import PeftModel


EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


class MultiTaskEmotionClassifier(nn.Module):
    """多任务情绪分类模型（用于加载）"""
    def __init__(self, base_model_path, num_labels=7, lora_config=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(base_model_path)
        if lora_config is not None:
            from peft import get_peft_model
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


class EmotionPredictor:
    """情绪分类预测器"""

    def __init__(self, model_path, base_model_path, device='cuda', use_multitask=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.emotion_list = EMOTION_LIST
        self.use_multitask = use_multitask

        print(f"加载模型: {model_path}")
        print(f"基础模型: {base_model_path}")

        if use_multitask:
            # 加载多任务模型
            from peft import LoraConfig, TaskType
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value", "key", "dense"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.model = MultiTaskEmotionClassifier(base_model_path, num_labels=len(EMOTION_LIST), lora_config=lora_config)
            self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=self.device), strict=False)
        else:
            # 加载单任务模型（RobertaForSequenceClassification）
            from transformers import RobertaForSequenceClassification
            base_model = RobertaForSequenceClassification.from_pretrained(
                base_model_path,
                num_labels=len(EMOTION_LIST)
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)

        self.model.to(self.device)
        self.model.eval()

        print(f"加载tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path if use_multitask else base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"模型加载完成，使用设备: {self.device}")

    def _format_conversation(self, conversation):
        """
        格式化对话文本（推理时不包含情绪标签）
        模型需要自己预测情绪
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

    def predict(self, conversation):
        """
        预测对话的主要情绪

        Args:
            conversation: 对话列表，格式为 [{"role": "user/assistant", "content": "..."}]
                         或者字符串（单句话）

        Returns:
            predicted_emotion: 预测的情绪标签
            probabilities: 各情绪的概率分布
        """
        # 处理输入
        if isinstance(conversation, str):
            conversation = [{"role": "user", "content": conversation}]

        # 格式化输入（不包含情绪标签）
        text = self._format_conversation(conversation)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            predicted_emotion = self.emotion_list[pred_idx]

        # 获取概率分布
        probabilities = {
            emotion: probs[0][i].item()
            for i, emotion in enumerate(self.emotion_list)
        }

        return predicted_emotion, probabilities

    def predict_batch(self, conversations):
        """批量预测"""
        results = []
        for conv in conversations:
            emotion, probs = self.predict(conv)
            results.append({
                'emotion': emotion,
                'probabilities': probs
            })
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='情绪分类推理')
    parser.add_argument('--model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_final',
                        help='模型路径')
    parser.add_argument('--base_model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/Model/roberta-base',
                        help='基础模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--interactive', action='store_true', help='交互模式')

    args = parser.parse_args()

    # 加载模型
    predictor = EmotionPredictor(args.model_path, args.base_model_path, args.device)

    if args.interactive:
        # 交互模式
        print("\n" + "=" * 60)
        print("情绪分类交互模式")
        print("输入对话内容，模型将预测情绪")
        print("输入 'quit' 退出")
        print("=" * 60 + "\n")

        while True:
            text = input("请输入对话内容: ").strip()
            if text.lower() == 'quit':
                break

            if text:
                emotion, probs = predictor.predict(text)
                print(f"预测情绪: {emotion}")
                print("概率分布:")
                for emo, prob in sorted(probs.items(), key=lambda x: -x[1]):
                    print(f"  {emo}: {prob:.4f}")
                print()
    else:
        # 演示模式
        print("\n" + "=" * 60)
        print("演示模式")
        print("=" * 60)

        # 示例对话
        examples = [
            "I'm so happy today! Everything is going great!",
            "I'm really angry about what happened yesterday.",
            "I feel so sad, my best friend is moving away.",
            "That movie was disgusting, I hated every minute of it.",
            "I'm surprised by the news!",
            [
                {"role": "user", "content": "I finally got the promotion!"},
                {"role": "assistant", "content": "Congratulations! You must be thrilled!"},
                {"role": "user", "content": "Yes! I've been working so hard for this."}
            ]
        ]

        for i, example in enumerate(examples):
            print(f"\n示例 {i+1}:")
            if isinstance(example, list):
                for turn in example:
                    print(f"  {turn['role']}: {turn['content']}")
            else:
                print(f"  {example}")

            emotion, probs = predictor.predict(example)
            print(f"  → 预测情绪: {emotion}")
            top_3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
            print(f"  → Top 3: {', '.join([f'{e}({p:.2f})' for e, p in top_3])}")


if __name__ == '__main__':
    main()
