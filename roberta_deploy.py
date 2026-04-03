#!/usr/bin/env python3
"""
RoBERTa-base 模型部署脚本 (GPU 加速版)
用于文本特征提取和相似度计算
"""

import argparse
from transformers import RobertaTokenizer, RobertaModel
import torch


class RoBERTaDeploy:
    def __init__(self, model_path='/home/user1/liuduanye/EmotionClassify/Model/roberta-base', device=None):
        """初始化模型和tokenizer"""
        print(f"正在加载模型: {model_path}...")

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成!")

    def encode(self, text):
        """
        将文本编码为向量
        Args:
            text: 输入文本
        Returns:
            文本的向量表示 (last_hidden_state 的 [CLS] token)
        """
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # 将输入移到 GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 [CLS] token 的输出作为句子表示，移回 CPU
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def encode_batch(self, texts):
        """
        批量编码文本
        Args:
            texts: 文本列表
        Returns:
            向量数组
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # 将输入移到 GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def similarity(self, text1, text2):
        """
        计算两个文本的余弦相似度
        """
        from numpy import dot
        from numpy.linalg import norm

        vec1 = self.encode(text1).flatten()
        vec2 = self.encode(text2).flatten()

        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def main():
    parser = argparse.ArgumentParser(description='RoBERTa-base 模型部署 (GPU加速)')
    parser.add_argument('--text', type=str, help='要编码的文本')
    parser.add_argument('--text1', type=str, help='相似度计算文本1')
    parser.add_argument('--text2', type=str, help='相似度计算文本2')
    parser.add_argument('--interactive', action='store_true', help='进入交互模式')
    parser.add_argument('--device', type=str, default=None, help='指定设备 (cuda/cpu/cuda:0 等)')
    parser.add_argument('--model_path', type=str, default='/home/user1/liuduanye/EmotionClassify/Model/roberta-base', help='模型路径')

    args = parser.parse_args()

    deploy = RoBERTaDeploy(model_path=args.model_path, device=args.device)

    if args.interactive:
        print("\n进入交互模式，输入 'quit' 退出")
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() == 'quit':
                break
            vector = deploy.encode(text)
            print(f"向量维度: {vector.shape}")
            print(f"向量前10维: {vector[0][:10]}")

    elif args.text1 and args.text2:
        sim = deploy.similarity(args.text1, args.text2)
        print(f"\n文本1: {args.text1}")
        print(f"文本2: {args.text2}")
        print(f"相似度: {sim:.4f}")

    elif args.text:
        vector = deploy.encode(args.text)
        print(f"\n文本: {args.text}")
        print(f"向量维度: {vector.shape}")
        print(f"向量前10维: {vector[0][:10]}")

    else:
        # 默认演示
        print("\n=== 演示模式 ===")
        texts = [
            "I love programming.",
            "Coding is my passion.",
            "The weather is nice today."
        ]

        print("\n编码示例文本:")
        for text in texts:
            vector = deploy.encode(text)
            print(f"  '{text}' -> 向量维度: {vector.shape}")

        print("\n文本相似度:")
        sim1 = deploy.similarity(texts[0], texts[1])
        sim2 = deploy.similarity(texts[0], texts[2])
        print(f"  '{texts[0]}' vs '{texts[1]}' -> {sim1:.4f}")
        print(f"  '{texts[0]}' vs '{texts[2]}' -> {sim2:.4f}")


if __name__ == '__main__':
    main()
