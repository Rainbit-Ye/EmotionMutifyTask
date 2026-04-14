"""
基于语义约束的 AAC 训练数据生成器

流程：
1. 组合标签（主语 + 动作 + 宾语）
2. 翻译Agent生成句子
3. 验证Agent评判质量（accept/revise/reject）
"""

import os
# 只使用 GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
import random
import re
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


class PromptsConfig:
    """人设配置管理"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_translation_prompt(self, labels: List[str]) -> str:
        template = self.config['translation_prompt']
        return template.format(labels=labels)
    
    def get_validation_prompt(self, labels: List[str], sentence: str) -> str:
        template = self.config['validation_prompt']
        return template.format(labels=labels, sentence=sentence)


class SemanticDataGenerator:
    """基于语义约束的训练数据生成器"""
    
    def __init__(self, ontology_path: str, model_path: str, prompts_config: PromptsConfig):
        self.prompts_config = prompts_config
        
        # 加载本体
        with open(ontology_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ontology = data.get("ontology", [])
        
        # 构建索引
        self.by_semantic_type = {}
        
        for item in self.ontology:
            icon_id = item.get("icon_id", "")
            clean_id = re.sub(r'_\d+[a-z]?$', '', icon_id)
            item["clean_id"] = clean_id
            
            st = item.get("semantic_type", "")
            if st:
                if st not in self.by_semantic_type:
                    self.by_semantic_type[st] = []
                self.by_semantic_type[st].append(item)
        
        # 提取关键类别（过滤掉不适合组合的标签）
        self.persons = [item for item in self._get_items_by_types(["person"]) 
                        if len(item["clean_id"]) > 2 and not item["clean_id"].startswith(("features", "man_-", "woman_-"))]
        self.actions = [item for item in self._get_items_by_types(["action", "verb"])
                        if len(item["clean_id"]) > 2]
        self.objects = [item for item in self._get_items_by_types(["entity", "object", "food", "drink", "body"])
                        if len(item["clean_id"]) > 2]
        self.emotions = [item for item in self._get_items_by_types(["emotion"])
                         if len(item["clean_id"]) > 2]
        self.places = [item for item in self._get_items_by_types(["place"])
                       if len(item["clean_id"]) > 2]
        
        print(f"人称: {len(self.persons)}, 动作: {len(self.actions)}, 物体: {len(self.objects)}, 情绪: {len(self.emotions)}, 地点: {len(self.places)}")
        
        # 加载模型
        print(f"\n加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        print("模型加载完成")
    
    def _get_items_by_types(self, types: List[str]) -> List[Dict]:
        """获取匹配语义类型的items"""
        items = []
        seen = set()
        for t in types:
            for item in self.by_semantic_type.get(t, []):
                clean_id = item["clean_id"]
                if clean_id not in seen:
                    seen.add(clean_id)
                    items.append(item)
        return items
    
    def generate_combination(self) -> Tuple[List[str], str]:
        """生成标签组合"""
        labels = []
        combo_type = random.choice(["svo", "sv", "svo_emo", "sv_emo"])
        
        subject = random.choice(self.persons) if self.persons else None
        action = random.choice(self.actions) if self.actions else None
        obj = random.choice(self.objects) if self.objects else None
        emotion = random.choice(self.emotions) if self.emotions else None
        
        if combo_type == "svo" and subject and action and obj:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"]]
        elif combo_type == "sv" and subject and action:
            labels = [subject["clean_id"], action["clean_id"]]
        elif combo_type == "svo_emo" and subject and action and obj and emotion:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"], emotion["clean_id"]]
        elif combo_type == "sv_emo" and subject and action and emotion:
            labels = [subject["clean_id"], action["clean_id"], emotion["clean_id"]]
        
        return labels, combo_type
    
    def clean_symbol(self, symbol: str) -> str:
        """清理符号"""
        symbol = re.sub(r'_\d+[a-z]?$', '', symbol)
        symbol = symbol.replace("_,_to", "").replace("_to", "")
        symbol = symbol.replace("_", " ")
        return symbol
    
    def call_model(self, prompt: str, max_tokens: int = 100) -> str:
        """调用模型"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    def translate(self, labels: List[str]) -> Tuple[bool, str]:
        """翻译Agent：标签 -> 句子"""
        clean_labels = [self.clean_symbol(l) for l in labels]
        
        prompt = f"""Translate these AAC symbols into ONE simple English sentence. You MUST use ALL symbols.

Symbols: {clean_labels}

If you cannot make a sensible sentence using ALL symbols, output: REJECT

Sentence:"""
        
        response = self.call_model(prompt, max_tokens=50)
        response = response.strip().strip('"\'').split('\n')[0].strip()
        
        # 如果模型返回REJECT，标记为失败
        if "REJECT" in response.upper():
            return False, ""
        
        return True, response
    
    def validate(self, labels: List[str], sentence: str) -> Dict:
        """验证Agent：评判语料质量"""
        clean_labels = [self.clean_symbol(l) for l in labels]
        
        prompt = f"""Check if this sentence uses all the labels:

Labels: {clean_labels}
Sentence: {sentence}

Does the sentence use ALL labels? Is it natural?

Output JSON only:
{{"action": "accept/revise/reject", "reason": "...", "revised_sentence": "..."}}"""
        
        response = self.call_model(prompt, max_tokens=150)
        
        # 提取JSON
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"action": "reject", "reason": "Failed to parse"}
    
    def generate_training_data(self, output_path: str, num_combinations: int = 1000, debug: bool = False):
        """生成训练数据（简化版，只用翻译Agent）"""
        training_data = []
        stats = {"accepted": 0, "rejected": 0}
        
        pbar = tqdm(total=num_combinations, desc="生成数据")
        attempts = 0
        max_attempts = num_combinations * 10
        
        while len(training_data) < num_combinations and attempts < max_attempts:
            attempts += 1
            
            labels, combo_type = self.generate_combination()
            if len(labels) < 2:
                continue
            
            # 翻译（模型会自动REJECT不合理的组合）
            success, sentence = self.translate(labels)
            
            if debug and attempts <= 5:
                print(f"\n[DEBUG] labels={labels}, success={success}, sentence={sentence}")
            
            if success and len(sentence) > 5 and "REJECT" not in sentence.upper():
                training_data.append({"labels": labels, "sentence": sentence, "type": combo_type})
                stats["accepted"] += 1
                pbar.update(1)
            else:
                stats["rejected"] += 1
        
        pbar.close()
        
        random.shuffle(training_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n保存 {len(training_data)} 条数据到: {output_path}")
        print(f"统计: accepted={stats['accepted']}, rejected={stats['rejected']}, attempts={attempts}")
        
        print("\n示例数据:")
        for item in training_data[:10]:
            print(f"  {item['labels']} → {item['sentence']}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    args = parser.parse_args()
    
    base_dir = "/home/user1/liuduanye/AgentPipeline/EmotionClassify/AAC2Text"
    ontology_path = f"{base_dir}/data/processed/aac_full_ontology.json"
    output_path = args.output or f"{base_dir}/data/processed/training_data.json"
    prompts_path = f"{base_dir}/config/prompts.yaml"
    model_path = "/home/user1/liuduanye/AgentPipeline/qwen/Qwen2_5-1_5B-Instruct"
    
    print("="*60)
    print("AAC 训练数据生成（带验证Agent）")
    print("="*60)
    
    prompts_config = PromptsConfig(prompts_path)
    generator = SemanticDataGenerator(ontology_path, model_path, prompts_config)
    generator.generate_training_data(output_path, num_combinations=args.num, debug=args.debug)


if __name__ == '__main__':
    main()
