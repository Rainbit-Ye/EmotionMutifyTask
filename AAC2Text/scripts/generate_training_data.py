"""
基于语义约束的 AAC 训练数据生成器

流程：
1. 组合标签（主语 + 动作 + 宾语）
2. 翻译Agent生成句子 + 自评语义合理性（一次Qwen调用）
3. 量化验证器评判质量（coverage + density，纯计算零开销）
"""

import os
# 只使用 GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import math
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


class QuantitativeValidator:
    """量化验证器：纯计算指标

    评估维度及权重:
    - 模型自评 (Naturalness): 0.50  — 翻译时模型对自己句子的语义合理性打分(1-5)
    - 标签覆盖率 (Coverage):  0.35  — 每个label是否出现在sentence中（简单词形变化匹配）
    - 标签密度 (Density):     0.15  — 被覆盖标签数/句子词数，高斯核映射

    综合得分 S = Σ(w_i × score_i)，阈值:
    - S >= 0.55 → accept
    - S < 0.55  → reject

    一票否决: coverage < 0.5 直接 reject
    """

    def __init__(self):
        self.weights = {
            "naturalness": 0.50,
            "coverage": 0.35,
            "density": 0.15,
        }
        self.threshold_accept = 0.55
        self.veto_coverage = 0.5

    def validate(self, labels: List[str], sentence: str, naturalness: int = 3) -> Dict:
        """量化评估

        Args:
            labels: AAC 标签列表
            sentence: 翻译生成的句子
            naturalness: 模型自评的语义合理性分数(1-5)
        """
        # 1) 模型自评 → 归一化到 [0,1]
        norm_naturalness = max(0.0, min(1.0, (naturalness - 1) / 4.0))

        # 2) 标签覆盖率
        coverage, missing = self._coverage_score(labels, sentence)

        # 3) 标签密度
        density = self._density_score(coverage, len(labels), sentence)

        metrics = {
            "naturalness": norm_naturalness,
            "coverage": coverage,
            "density": density,
        }

        # 一票否决
        if coverage < self.veto_coverage:
            return {
                "action": "reject",
                "metrics": metrics,
                "missing_labels": missing,
                "detail": f"REJECT (veto: coverage={coverage:.2f} < {self.veto_coverage})",
            }

        score = sum(self.weights[k] * metrics[k] for k in self.weights)
        action = "accept" if score >= self.threshold_accept else "reject"

        detail = (f"Score={score:.3f} [{action}]  "
                  f"naturalness={norm_naturalness:.2f}(raw={naturalness})  "
                  f"coverage={coverage:.2f}  density={density:.2f}")

        return {
            "action": action,
            "metrics": metrics,
            "missing_labels": missing,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # 指标1: 模型自评（归一化）
    # ------------------------------------------------------------------
    # naturalness 1-5 → (n-1)/4 → [0.0, 0.25, 0.50, 0.75, 1.0]

    # ------------------------------------------------------------------
    # 指标2: 标签覆盖率（简单词形变化匹配）
    # ------------------------------------------------------------------
    def _coverage_score(self, labels: List[str], sentence: str) -> Tuple[float, List[str]]:
        """计算每个 label 在 sentence 中是否出现

        匹配策略:
        1. 原形子串匹配
        2. 简单词形变化: -s, -es, -ed, -ing, -er（规则生成）
        """
        sent_lower = sentence.lower()
        sent_words = set(re.findall(r"[a-z']+", sent_lower))
        missing = []
        hit = 0

        for label in labels:
            label_clean = label.lower().replace("_", " ").strip()
            label_words = label_clean.split()

            # 策略1: 原形子串匹配
            if label_clean in sent_lower:
                hit += 1
                continue

            # 策略2: 所有子词都出现
            if all(w in sent_words for w in label_words):
                hit += 1
                continue

            # 策略3: 任意子词 + 简单词形变化匹配
            found = False
            for lw in label_words:
                if lw in sent_words:
                    found = True
                    break
                for v in self._variants(lw):
                    if v in sent_words:
                        found = True
                        break
                if found:
                    break

            if found:
                hit += 1
            else:
                missing.append(label)

        return (hit / len(labels) if labels else 0.0), missing

    @staticmethod
    def _variants(word: str) -> List[str]:
        """简单词形变化变体"""
        vs = []
        if word.endswith("e"):
            vs += [word + "d", word + "s", word[:-1] + "ing"]
        elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            vs += [word[:-1] + "ies", word[:-1] + "ied"]
        else:
            vs += [word + "s", word + "es", word + "ed", word + "ing", word + "er"]
        return vs

    # ------------------------------------------------------------------
    # 指标3: 标签密度
    # ------------------------------------------------------------------
    @staticmethod
    def _density_score(coverage: float, n_labels: int, sentence: str) -> float:
        """标签密度高斯核映射，峰值在 density=0.3"""
        words = sentence.lower().split()
        n_words = len(words) if words else 1
        density = (coverage * n_labels) / n_words
        return math.exp(-0.5 * ((density - 0.3) / 0.25) ** 2)


class SemanticDataGenerator:
    """基于语义约束的训练数据生成器"""

    def __init__(self, ontology_path: str, model_path: str, prompts_config: PromptsConfig):
        self.prompts_config = prompts_config
        self.validator = QuantitativeValidator()

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

        # 提取关键类别（扩展更多语义类型）
        self.persons = [item for item in self._get_items_by_types(
                        ["person", "relationship"])
                        if len(item["clean_id"]) > 2 and not item["clean_id"].startswith(("features", "man_-", "woman_-"))]
        self.actions = [item for item in self._get_items_by_types(
                        ["action", "verb", "activity"])
                        if len(item["clean_id"]) > 2]
        self.objects = [item for item in self._get_items_by_types(
                        ["entity", "object", "food", "drink", "body", "body part", "body_part",
                         "animal", "tool", "clothing", "device", "material", "event", "noun"])
                        if len(item["clean_id"]) > 2
                        and not item["clean_id"].startswith(("flag_", "country_"))]
        self.emotions = [item for item in self._get_items_by_types(
                         ["emotion", "quality"])
                         if len(item["clean_id"]) > 2
                         and not re.match(r'^[a-z]_-', item["clean_id"])]
        self.places = [item for item in self._get_items_by_types(
                       ["place", "location"])
                       if len(item["clean_id"]) > 2]
        self.times = [item for item in self._get_items_by_types(["time"])
                      if len(item["clean_id"]) > 2]

        print(f"人称: {len(self.persons)}, 动作: {len(self.actions)}, 物体: {len(self.objects)}, "
              f"情绪/修饰: {len(self.emotions)}, 地点: {len(self.places)}, 时间: {len(self.times)}")

        # 加载模型
        print(f"\n加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        print("模型加载完成")

    def _get_items_by_types(self, types: List[str]) -> List[Dict]:
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
        labels = []
        combo_type = random.choice([
            "svo", "sv", "svo_emo", "sv_emo",
            "svo_place", "svo_time", "sv_time",
        ])

        subject = random.choice(self.persons) if self.persons else None
        action = random.choice(self.actions) if self.actions else None
        obj = random.choice(self.objects) if self.objects else None
        emotion = random.choice(self.emotions) if self.emotions else None
        place = random.choice(self.places) if self.places else None
        time_ = random.choice(self.times) if self.times else None

        if combo_type == "svo" and subject and action and obj:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"]]
        elif combo_type == "sv" and subject and action:
            labels = [subject["clean_id"], action["clean_id"]]
        elif combo_type == "svo_emo" and subject and action and obj and emotion:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"], emotion["clean_id"]]
        elif combo_type == "sv_emo" and subject and action and emotion:
            labels = [subject["clean_id"], action["clean_id"], emotion["clean_id"]]
        elif combo_type == "svo_place" and subject and action and obj and place:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"], place["clean_id"]]
        elif combo_type == "svo_time" and subject and action and obj and time_:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"], time_["clean_id"]]
        elif combo_type == "sv_time" and subject and action and time_:
            labels = [subject["clean_id"], action["clean_id"], time_["clean_id"]]

        return labels, combo_type

    def clean_symbol(self, symbol: str) -> str:
        symbol = re.sub(r'_\d+[a-z]?$', '', symbol)
        symbol = symbol.replace("_,_to", "").replace("_to", "")
        symbol = symbol.replace("_", " ")
        return symbol

    def translate(self, labels: List[str]) -> Tuple[bool, str, int]:
        """翻译Agent：标签 -> 句子 + 自评语义合理性

        一次 Qwen 调用，同时输出句子和自然度评分(1-5)。

        Returns:
            (success, sentence, naturalness)
        """
        clean_labels = [self.clean_symbol(l) for l in labels]

        prompt = f"""Translate these AAC symbols into ONE simple English sentence. You MUST use ALL symbols.

Symbols: {clean_labels}

If you cannot make a sensible sentence using ALL symbols, output: REJECT

After the sentence, rate how NATURALLY the symbols combine into this sentence:
1 = completely absurd/nonsensical combination
2 = very forced/unnatural
3 = somewhat acceptable but odd
4 = natural and reasonable
5 = perfectly natural and common

Output format (two lines):
Sentence: <your sentence>
Naturalness: <1-5>

Your output:"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 解析句子和评分
        sentence = ""
        naturalness = 3  # 默认中间分

        if "REJECT" in response.upper():
            return False, "", 1

        # 提取 Sentence: 行
        sent_match = re.search(r'[Ss]entence:\s*(.+?)(?:\n|$)', response)
        if sent_match:
            sentence = sent_match.group(1).strip().strip('"\'')
        else:
            # 没找到标记，取第一行
            sentence = response.strip().split('\n')[0].strip().strip('"\'')

        # 提取 Naturalness: 行
        nat_match = re.search(r'[Nn]aturalness:\s*(\d)', response)
        if nat_match:
            naturalness = int(nat_match.group(1))
            naturalness = max(1, min(5, naturalness))

        return True, sentence, naturalness

    def generate_training_data(self, output_path: str, num_combinations: int = 1000, debug: bool = False):
        """生成训练数据（翻译Agent + 量化验证）

        流程:
        1. 翻译Agent: labels → sentence + naturalness（一次 Qwen 调用）
        2. 量化验证: naturalness + coverage + density（纯计算）
        3. 只收 accept，reject 的丢弃重新生成补全
        """
        training_data = []
        stats = {"accepted": 0, "rejected_by_translate": 0, "rejected_by_quant": 0}

        pbar = tqdm(total=num_combinations, desc="生成数据")
        attempts = 0
        max_attempts = num_combinations * 20

        while len(training_data) < num_combinations and attempts < max_attempts:
            attempts += 1

            labels, combo_type = self.generate_combination()
            if len(labels) < 2:
                continue

            # Step 1: 翻译 + 自评
            success, sentence, naturalness = self.translate(labels)

            if not success or len(sentence) <= 5 or "REJECT" in sentence.upper():
                stats["rejected_by_translate"] += 1
                continue

            # Step 2: 量化验证
            result = self.validator.validate(labels, sentence, naturalness)
            action = result["action"]

            if debug and stats["accepted"] + stats["rejected_by_quant"] < 30:
                print(f"\n[DEBUG] labels={labels}, sentence={sentence}")
                print(f"  {result['detail']}")

            if action == "accept":
                training_data.append({
                    "labels": labels,
                    "sentence": sentence,
                    "type": combo_type,
                    "validation": result.get("metrics", {}),
                })
                stats["accepted"] += 1
                pbar.update(1)
            else:
                stats["rejected_by_quant"] += 1

        pbar.close()

        random.shuffle(training_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        # 统计报告
        print(f"\n{'='*60}")
        print(f"保存 {len(training_data)} 条数据到: {output_path}")
        print(f"统计: accepted={stats['accepted']}, "
              f"rejected_by_translate={stats['rejected_by_translate']}, "
              f"rejected_by_quant={stats['rejected_by_quant']}, attempts={attempts}")

        # naturalness 分布
        nat_vals = [item["validation"].get("naturalness", 0) for item in training_data]
        if nat_vals:
            raw_scores = [round(v * 4 + 1) for v in nat_vals]  # 反归一化
            from collections import Counter
            dist = Counter(raw_scores)
            print(f"\nNaturalness分布:")
            for s in range(1, 6):
                print(f"  {s}: {'█' * dist.get(s, 0)} ({dist.get(s, 0)})")

        print(f"\n示例数据:")
        for item in training_data[:10]:
            v = item.get("validation", {})
            nat = round(v.get("naturalness", 0) * 4 + 1)
            cov = v.get("coverage", 0)
            print(f"  [nat={nat} cov={cov:.1f}] {item['labels']} → {item['sentence']}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    args = parser.parse_args()

    base_dir = "/home/user1/liuduanye/EmotionClassify/AAC2Text"
    ontology_path = f"{base_dir}/data/processed/aac_full_ontology.json"
    output_path = args.output or f"{base_dir}/data/processed/training_data.json"
    prompts_path = f"{base_dir}/config/prompts.yaml"
    model_path = "/home/user1/liuduanye/qwen/Qwen2_5-1_5B-Instruct"

    print("=" * 60)
    print("AAC 训练数据生成（翻译Agent + 量化验证器）")
    print("=" * 60)

    prompts_config = PromptsConfig(prompts_path)
    generator = SemanticDataGenerator(ontology_path, model_path, prompts_config)
    generator.generate_training_data(output_path, num_combinations=args.num, debug=args.debug)


if __name__ == '__main__':
    main()
