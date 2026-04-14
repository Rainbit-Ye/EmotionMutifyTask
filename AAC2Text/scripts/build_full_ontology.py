"""
AAC 图标本体 (Ontology) 构建脚本

基于论文 PictOntology/Pictogrammar 方法，为每个象形图构建：

1. 图标 ID
2. 核心语义（核心含义）
3. 语义类型（动作/名词/情感/地点等）
4. 语法角色（主语/谓语/宾语/修饰语）
5. 语法约束（可和谁组合）
6. 上层语义概念（如：喝水 → 饮食行为）

使用 Qwen 大模型自动推断语义信息
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict
from enum import Enum
import re


# ============ 本体结构定义 ============

class SemanticType(Enum):
    """语义类型"""
    ACTION = "action"         # 动作：eat, drink, go, run
    ENTITY = "entity"         # 实体：物体、人物
    EMOTION = "emotion"       # 情感：happy, sad, angry
    PLACE = "place"           # 地点：home, school, hospital
    TIME = "time"             # 时间：morning, today
    QUALITY = "quality"       # 属性：big, small, hot, cold
    PERSON = "person"         # 人物：mum, dad, doctor
    FOOD = "food"             # 食物：apple, bread
    DRINK = "drink"           # 饮料：water, milk
    BODY = "body"             # 身体部位：hand, head
    ABSTRACT = "abstract"     # 抽象概念


class GrammarRole(Enum):
    """语法角色"""
    SUBJECT = "subject"       # 主语
    PREDICATE = "predicate"   # 谓语
    OBJECT = "object"         # 宾语
    MODIFIER = "modifier"     # 修饰语
    COMPLEMENT = "complement" # 补语
    LOCATION = "location"     # 地点状语
    TIME = "time"             # 时间状语


@dataclass
class PictogramOntology:
    """象形图本体结构"""
    icon_id: str                    # 图标ID
    label: str                      # 标签
    core_semantic: str              # 核心语义
    semantic_type: str              # 语义类型（动作/名词/情感/地点等）
    grammar_role: str               # 语法角色（主语/谓语/宾语/修饰语）
    can_combine_with: List[str]     # 可组合的语义类型
    super_concept: str              # 上层语义概念
    typical_objects: List[str]      # 动词专用：典型宾语
    typical_modifiers: List[str]    # 名词专用：典型修饰语


class OntologyBuilder:
    """本体构建器 - 使用 Qwen 推断语义信息"""
    
    def __init__(self, model_path: str = "/home/user1/liuduanye/AgentPipeline/qwen/Qwen2_5-1_5B-Instruct"):
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
    
    def generate_response(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """生成回复"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON"""
        try:
            # 尝试找到JSON块
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return None
    
    def infer_ontology_batch(self, symbols: List[Dict]) -> List[Dict]:
        """
        批量推断符号的本体信息
        
        输入: symbols = [{"icon_id": "xxx", "label": "xxx", "pos": "xxx", "meaning": "xxx"}]
        输出: 完整的本体信息
        """
        prompt = f"""You are an AAC ontology expert. Analyze these pictogram symbols and generate semantic ontology.

Symbols to analyze:
{json.dumps(symbols[:15], indent=2)}

For each symbol, output a JSON object with these 10 fields:

{{
  "icon_id": "original icon id",
  "label": "original label", 
  "core_semantic": "core meaning in one short phrase",
  "semantic_type": "one of: action, entity, emotion, place, time, quality, person, food, drink, body, abstract",
  "grammar_role": "primary role: subject, predicate, object, modifier, complement, location",
  "can_combine_with": ["what semantic types can combine with this, max 3"],
  "super_concept": "parent concept: e.g., 'drink water' -> 'eating behavior'",
  "typical_objects": ["for verbs: typical objects like 'water', 'food'"],
  "typical_modifiers": ["for nouns: typical modifiers like 'cold', 'big'"],
  "aac_category": "one of: subject, content, emotion"
}}

CRITICAL CLASSIFICATION RULES:

1. semantic_type MUST follow these rules:
   - "action": verbs - eat, drink, go, run, help, want, need
   - "person": HUMANS only - I, you, mum, dad, doctor, teacher, nurse, husband, friend
   - "food": edible items - apple, bread, rice, pancake, cake, meat
   - "drink": beverages - water, milk, juice, coffee, tea
   - "body": body parts - hand, head, eye, ear, arm, leg
   - "place": locations - home, school, hospital, park, kitchen
   - "emotion": FEELINGS only - happy, sad, angry, worried, tired, hungry
   - "quality": ADJECTIVES describing properties - big, small, hot, cold, red, blue
   - "abstract": concepts, shapes, symbols - triangle, number, color, letter
   - "entity": objects, animals, things NOT in above categories - cow, dog, chair, book, car

2. aac_category rules:
   - "subject": ONLY pronouns (I, you) and person roles (doctor, teacher, mum, dad)
   - "content": actions, objects, places, food, drinks, animals, things
   - "emotion": ONLY emotion words (happy, sad, angry, worried, scared)

3. Animals (cow, dog, cat, bird) → semantic_type: "entity", NOT "person"
4. Geometric shapes (triangle, circle, square) → semantic_type: "abstract", NOT "quality/emotion"
5. Food items (pancake, cake, bread) → semantic_type: "food", NOT "person"
6. Colors (red, blue, green) → semantic_type: "quality", NOT "emotion"

Examples:
- "doctor" → semantic_type: "person", aac_category: "subject"
- "cow" → semantic_type: "entity", aac_category: "content"  
- "pancake" → semantic_type: "food", aac_category: "content"
- "triangle_right-angled" → semantic_type: "abstract", aac_category: "content"
- "worried_man" → semantic_type: "emotion", aac_category: "emotion"
- "red" → semantic_type: "quality", aac_category: "content"

Output only a JSON array, nothing else."""

        response = self.generate_response(prompt, max_new_tokens=2000)
        
        # 提取JSON数组
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            print(f"JSON解析失败: {response[:200]}...")
        
        return []
    
    def build_ontology_for_dataset(self, 
                                    mapping_path: str, 
                                    output_path: str,
                                    batch_size: int = 10):
        """
        为整个数据集构建本体
        
        mapping_path: 原始映射文件路径
        output_path: 输出本体文件路径
        """
        # 加载原始映射
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        print(f"加载 {len(mapping_data)} 个符号")
        
        # 准备符号信息
        all_symbols = []
        for item in mapping_data:
            classifications = item.get("classifications", [])
            primary = classifications[0] if classifications else {}
            
            symbol_info = {
                "icon_id": item.get("filename", "").replace(".png", ""),
                "label": item.get("word", ""),
                "pos": primary.get("pos", "NOUN"),
                "meaning": primary.get("meaning", ""),
                "grammar_tags": primary.get("grammar_tags", []),
                "context_hints": primary.get("context_hints", [])
            }
            all_symbols.append(symbol_info)
        
        # 批量处理
        all_ontologies = []
        
        print(f"\n开始构建本体，共 {len(all_symbols)} 个符号，每批 {batch_size} 个")
        
        for i in tqdm(range(0, len(all_symbols), batch_size), desc="构建本体"):
            batch = all_symbols[i:i+batch_size]
            
            # 使用 Qwen 推断本体信息
            batch_ontologies = self.infer_ontology_batch(batch)
            
            # 合并原始信息
            for orig, onto in zip(batch, batch_ontologies):
                all_ontologies.append(onto)
        
        # 保存本体
        output_data = {
            "metadata": {
                "total_symbols": len(all_ontologies),
                "semantic_types": list(set(o.get("semantic_type", "") for o in all_ontologies)),
                "source": mapping_path
            },
            "ontology": all_ontologies
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n本体构建完成，保存到: {output_path}")
        print(f"共处理 {len(all_ontologies)} 个符号")
        
        # 统计
        self._print_statistics(all_ontologies)
        
        return all_ontologies
    
    def _print_statistics(self, ontologies: List[Dict]):
        """打印统计信息"""
        print("\n" + "="*60)
        print("本体统计")
        print("="*60)
        
        # 语义类型分布
        type_counts = {}
        for o in ontologies:
            t = o.get("semantic_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print("\n语义类型分布:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")
        
        # 语法角色分布
        role_counts = {}
        for o in ontologies:
            r = o.get("grammar_role", "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1
        
        print("\n语法角色分布:")
        for r, c in sorted(role_counts.items(), key=lambda x: -x[1]):
            print(f"  {r}: {c}")
        
        # 上层概念分布
        super_counts = {}
        for o in ontologies:
            s = o.get("super_concept", "unknown")
            super_counts[s] = super_counts.get(s, 0) + 1
        
        print("\n上层语义概念分布 (Top 15):")
        for s, c in sorted(super_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {s}: {c}")
        
        # 示例
        print("\n示例数据:")
        for o in ontologies[:3]:
            print(f"\n  {o.get('label')}:")
            print(f"    核心语义: {o.get('core_semantic')}")
            print(f"    语义类型: {o.get('semantic_type')}")
            print(f"    语法角色: {o.get('grammar_role')}")
            print(f"    可组合: {o.get('can_combine_with')}")
            print(f"    上层概念: {o.get('super_concept')}")
            if o.get('typical_objects'):
                print(f"    典型宾语: {o.get('typical_objects')}")
            if o.get('typical_modifiers'):
                print(f"    典型修饰: {o.get('typical_modifiers')}")


def main():
    """主函数"""
    mapping_path = "/home/user1/liuduanye/AACTest/AAC/data/dataset_custom.json"
    ontology_output = "../../data/processed/aac_full_ontology.json"
    
    print("="*60)
    print("AAC 图标本体构建")
    print("="*60)
    
    # 初始化构建器
    builder = OntologyBuilder()
    
    # 构建本体
    builder.build_ontology_for_dataset(
        mapping_path=mapping_path,
        output_path=ontology_output,
        batch_size=10
    )


if __name__ == '__main__':
    main()
