"""
纯随机训练数据生成器

逻辑：
1. 从本体纯随机选取 1-7 个符号
2. 让 Qwen 判断是否语义合理
3. 合理的组合翻译成自然句子
4. 不合理的删除
"""

import os
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
from tqdm import tqdm
from collections import defaultdict


class RandomDataGenerator:
    """纯随机数据生成器"""
    
    def __init__(self, ontology_path: str, model_path: str):
        # 加载本体
        with open(ontology_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ontology = data.get("ontology", [])
        
        # 提取所有符号，去除编号后缀
        all_symbols_raw = [o.get("icon_id", "") for o in self.ontology if o.get("icon_id")]
        # 去重：去掉 _1a, _2b, _3c 等编号后缀
        import re
        self.all_symbols = []
        seen = set()
        for sym in all_symbols_raw:
            # 去掉编号后缀如 _1a, _2b, _123 等
            clean_sym = re.sub(r'_\d+[a-z]?$', '', sym)
            if clean_sym and clean_sym not in seen:
                self.all_symbols.append(clean_sym)
                seen.add(clean_sym)
        
        print(f"原始符号: {len(all_symbols_raw)} 个")
        print(f"去重后: {len(self.all_symbols)} 个")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("模型加载完成")
    
    def generate_random_combinations(self, num_combinations: int = 50000) -> List[Tuple[List[str], str]]:
        """纯随机生成组合"""
        results = []
        
        for _ in range(num_combinations):
            # 随机选择 1-7 个符号
            num_labels = random.randint(1, 7)
            labels = random.sample(self.all_symbols, num_labels)
            results.append((labels, ""))
        
        return results
    
    def validate_and_translate(self, labels: List[str]) -> Tuple[bool, str]:
        """
        让 Qwen 判断组合是否合理并翻译
        返回: (是否有效, 句子)
        """
        # 清理标签
        clean_labels = [l.replace("_,_to", "").replace("_to", "").replace("_", " ") for l in labels]
        
        prompt = f"""You are an AAC (Augmentative and Alternative Communication) translator.

AAC users select pictogram symbols to communicate. Your job is to:

1. Determine if these symbols can form a meaningful expression
2. If yes, translate them into a natural English sentence
3. If no (completely nonsensical), respond with "INVALID"

Symbols: {clean_labels}

Rules:
- Try to make sense of the symbols
- Add missing words (articles, prepositions, helping verbs) as needed
- Use first person ("I") if no subject is specified
- Keep sentences simple and natural
- Only say INVALID if absolutely impossible to interpret

Examples:
- ["want", "water"] → "I want water."
- ["happy"] → "I am happy."
- ["eat", "apple"] → "I eat an apple."
- ["doctor", "help"] → "The doctor helps." or "I need a doctor."
- ["go", "school"] → "I go to school."
- ["sleep", "computer"] → INVALID (nonsensical)
- ["drink", "table"] → INVALID (nonsensical)

Output only the sentence or INVALID:"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # 判断是否有效
        if response.upper() == "INVALID" or len(response) < 2:
            return False, ""
        
        # 清理响应
        response = response.strip('"\'')
        
        return True, response
    
    def generate_training_data(self, output_path: str, num_combinations: int = 50000):
        """生成训练数据"""
        
        print(f"\n步骤1: 随机生成 {num_combinations} 个组合...")
        combinations = self.generate_random_combinations(num_combinations)
        
        print(f"\n步骤2: Qwen 验证和翻译...")
        training_data = []
        valid_count = 0
        invalid_count = 0
        
        for labels, _ in tqdm(combinations, desc="处理数据"):
            is_valid, sentence = self.validate_and_translate(labels)
            
            if is_valid:
                training_data.append({
                    "labels": labels,
                    "sentence": sentence
                })
                valid_count += 1
            else:
                invalid_count += 1
        
        print(f"\n有效: {valid_count}, 无效: {invalid_count}")
        print(f"有效率: {valid_count / len(combinations) * 100:.1f}%")
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n保存 {len(training_data)} 条训练数据到: {output_path}")
        
        # 显示示例
        print("\n示例数据:")
        for item in training_data[:20]:
            print(f"  {item['labels']} → {item['sentence']}")
        
        return training_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成随机训练数据')
    parser.add_argument('--num', type=int, default=50000, help='生成组合数量 (默认: 50000)')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    args = parser.parse_args()
    
    ontology_path = "../../data/processed/aac_full_ontology.json"
    output_path = args.output or "../../data/processed/training_data.json"
    model_path = "/home/user1/liuduanye/qwen/Qwen2_5-1_5B-Instruct"
    
    print("="*60)
    print("纯随机训练数据生成")
    print("="*60)
    print(f"生成数量: {args.num}")
    
    generator = RandomDataGenerator(ontology_path, model_path)
    generator.generate_training_data(output_path, num_combinations=args.num)


if __name__ == '__main__':
    main()
