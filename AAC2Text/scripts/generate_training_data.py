"""
基于语义约束的 AAC 训练数据生成器

流程：
1. 组合标签（主语 + 动作 + 宾语）
2. 翻译Agent生成句子 (Qwen2.5-1.5B, GPU 2)
3. CoT验证Agent：链式推理评估 (Llama-3-8B, GPU 3)
4. 量化验证器评判质量：
   - CoT自然度 (Naturalness): 0.35
   - 标签覆盖率 (Coverage):   0.30
   - 公式化程度 (Formulaicness): 0.20 — BERT回归器 (GPU 0)
"""

import os
import json
import torch
import torch.nn as nn
import random
import re
import yaml
import threading
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer, BertPreTrainedModel, BertModel
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


def find_available_gpu(min_free_gb: int) -> Optional[int]:
    """找到空闲显存 >= min_free_gb 的 GPU，返回 index，无则返回 None"""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            idx, free = line.strip().split(", ")
            if int(free) >= min_free_gb * 1024:
                return int(idx)
    except Exception:
        pass
    return None


class PromptsConfig:
    """人设配置管理"""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get_translation_prompt(self, labels: List[str]) -> str:
        template = self.config['translation_prompt']
        return template.format(labels=labels)

    def get_validation_cot_prompt(self, labels: List[str], sentence: str) -> str:
        template = self.config['validation_cot_prompt']
        return template.format(labels=labels, sentence=sentence)


# ==========================================================================
# BERT 回归模型：预测公式化程度
# 来自论文: Incorporating Formulaicness in the Automatic Evaluation of
# Naturalness (INLG 2025, Calò et al.)
# ==========================================================================
class BertRegressionModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output).squeeze(-1)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}


class QuantitativeValidator:
    """量化验证器：CoT自然度 + BERT公式化程度 + 标签覆盖率

    评估维度及权重:
    - CoT自然度 (Naturalness): 0.35
    - 标签覆盖率 (Coverage):   0.30
    - 公式化程度 (Formulaicness): 0.20 — BERT回归器预测

    综合得分 S = 0.35×naturalness + 0.30×coverage + 0.20×(1-formulaicness)
    阈值: S >= 0.55 → accept
    一票否决: coverage < 0.5 直接 reject
    """

    def __init__(self, formulaicness_model_path: Optional[str] = None):
        self.weights = {
            "naturalness": 0.40,
            "coverage": 0.25,
            "formulaicness": 0.15,
        }
        self.threshold_accept = 0.55
        self.veto_coverage = 0.5
        self.veto_naturalness = 0.5  # naturalness过低也一票否决

        # 加载 BERT 公式化程度回归器
        self.form_model = None
        self.form_tokenizer = None
        if formulaicness_model_path:
            print(f"  加载公式化程度模型: {formulaicness_model_path}")
            from transformers import BertConfig, BertTokenizer
            config = BertConfig.from_pretrained(formulaicness_model_path)
            self.form_model = BertRegressionModel(config)
            # 手动加载权重，绕过 from_pretrained 的 tied_weights 兼容问题
            from safetensors.torch import load_file
            import os
            sf_path = os.path.join(formulaicness_model_path, "model.safetensors")
            state_dict = load_file(sf_path)
            self.form_model.load_state_dict(state_dict, strict=False)
            # 先加载到CPU，后续由SemanticDataGenerator统一分配GPU
            self.form_model = self.form_model.to("cpu")
            self.form_model.eval()
            # tokenizer: 优先从本地加载，否则从bert-base-uncased
            if os.path.exists(os.path.join(formulaicness_model_path, "vocab.txt")):
                self.form_tokenizer = BertTokenizer.from_pretrained(formulaicness_model_path)
            else:
                self.form_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            print("  公式化程度模型加载完成 (GPU 0)")

    def validate(self, labels: List[str], sentence: str,
                 naturalness_scores: Optional[Dict] = None) -> Dict:
        """量化评估

        Args:
            labels: AAC 标签列表
            sentence: 翻译生成的句子
            naturalness_scores: CoT验证Agent的评估结果
                {"coherence": 1-5, "grammaticality": 1-5,
                 "integration": 1-5, "overall": 1-5,
                 "reasoning": "..."}
        """
        # 1) CoT自然度评估 → 归一化到 [0,1]
        if naturalness_scores and isinstance(naturalness_scores, dict):
            coherence = naturalness_scores.get("coherence", 3)
            grammaticality = naturalness_scores.get("grammaticality", 3)
            integration = naturalness_scores.get("integration", 3)
            overall = naturalness_scores.get("overall", 3)

            norm_coherence = max(0.0, min(1.0, (coherence - 1) / 4.0))
            norm_grammaticality = max(0.0, min(1.0, (grammaticality - 1) / 4.0))
            norm_integration = max(0.0, min(1.0, (integration - 1) / 4.0))
            norm_overall = max(0.0, min(1.0, (overall - 1) / 4.0))

            norm_naturalness = (
                0.25 * norm_coherence +
                0.30 * norm_grammaticality +
                0.20 * norm_integration +
                0.25 * norm_overall
            )
            reasoning = naturalness_scores.get("reasoning", "")
        else:
            raw_nat = naturalness_scores if isinstance(naturalness_scores, int) else 3
            norm_naturalness = max(0.0, min(1.0, (raw_nat - 1) / 4.0))
            norm_coherence = norm_grammaticality = norm_integration = norm_overall = norm_naturalness
            coherence = grammaticality = integration = overall = raw_nat
            reasoning = ""

        # 2) 标签覆盖率
        coverage, missing = self._coverage_score(labels, sentence)

        # 3) 公式化程度（BERT回归器）
        formulaicness = self._formulaicness_score(sentence)

        metrics = {
            "naturalness": norm_naturalness,
            "coverage": coverage,
            "formulaicness": formulaicness,
            "cot_detail": {
                "coherence": norm_coherence,
                "grammaticality": norm_grammaticality,
                "integration": norm_integration,
                "overall": norm_overall,
            },
            "formulaicness_detail": formulaicness,
        }

        # 一票否决
        if coverage < self.veto_coverage:
            return {
                "action": "reject",
                "metrics": metrics,
                "missing_labels": missing,
                "reasoning": reasoning,
                "detail": (f"REJECT (veto: coverage={coverage:.2f} < {self.veto_coverage})  "
                           f"formulaicness={formulaicness:.2f}"),
            }
        if norm_naturalness <= self.veto_naturalness:
            return {
                "action": "reject",
                "metrics": metrics,
                "missing_labels": missing,
                "reasoning": reasoning,
                "detail": (f"REJECT (veto: naturalness={norm_naturalness:.2f} < {self.veto_naturalness})  "
                           f"formulaicness={formulaicness:.2f}"),
            }

        # 综合得分
        score = (
            self.weights["naturalness"] * norm_naturalness +
            self.weights["coverage"] * coverage +
            self.weights["formulaicness"] * (1.0 - formulaicness)
        )
        total_weight = sum(self.weights.values())
        score = score / total_weight

        action = "accept" if score >= self.threshold_accept else "reject"

        detail = (f"Score={score:.3f} [{action}]  "
                  f"naturalness={norm_naturalness:.2f}  "
                  f"coverage={coverage:.2f}  "
                  f"formulaicness={formulaicness:.2f}(penalty={1.0-formulaicness:.2f})  "
                  f"cot=[coh={coherence} gram={grammaticality} integ={integration} ovr={overall}]")

        return {
            "action": action,
            "metrics": metrics,
            "missing_labels": missing,
            "reasoning": reasoning,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # 指标3: 公式化程度 — BERT回归器 (INLG 2025)
    # ------------------------------------------------------------------
    def _formulaicness_score(self, sentence: str) -> float:
        """公式化程度：输出文本与输入结构的相似度

        使用 Calò et al. (INLG 2025) 训练的 BERT 回归器预测。
        公式化程度越高 → 句子越接近输入符号的线性罗列 → 不自然。
        """
        if self.form_model is not None and self.form_tokenizer is not None:
            return self._formulaicness_bert(sentence)
        else:
            return 0.5  # 无模型时取中间值

    @torch.no_grad()
    def _formulaicness_bert(self, sentence: str) -> float:
        """BERT回归器预测公式化程度"""
        inputs = self.form_tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=128
        )
        inputs = {k: v.to(self.form_model.device) for k, v in inputs.items()
                  if k != "token_type_ids"}
        outputs = self.form_model(**inputs)
        score = outputs["logits"].item()
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # 指标2: 标签覆盖率（简单词形变化匹配）
    # ------------------------------------------------------------------
    def _coverage_score(self, labels: List[str], sentence: str) -> Tuple[float, List[str]]:
        """计算每个 label 在 sentence 中是否出现"""
        sent_lower = sentence.lower()
        sent_words = set(re.findall(r"[a-z']+", sent_lower))
        missing = []
        hit = 0

        for label in labels:
            label_clean = label.lower().replace("_", " ").strip()
            label_words = label_clean.split()

            if label_clean in sent_lower:
                hit += 1
                continue

            if all(w in sent_words for w in label_words):
                hit += 1
                continue

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


class SemanticDataGenerator:
    """基于语义约束的训练数据生成器

    三模型架构:
    - 翻译模型 (Qwen2.5-1.5B): 标签 → 句子，GPU 2
    - CoT验证模型 (Llama-3-8B): 链式推理评估，GPU 3
    - 公式化程度模型 (BERT): BERT回归器，GPU 0
    """

    def __init__(self, ontology_path: str, translate_model_path: str,
                 cot_model_path: str, formulaicness_model_path: str,
                 prompts_config: PromptsConfig):
        self.prompts_config = prompts_config
        self.validator = QuantitativeValidator(formulaicness_model_path)

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

        # 提取关键类别
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

        # 自动分配 GPU
        translate_gpu = find_available_gpu(min_free_gb=5)   # Qwen1.5B ~3GB
        if translate_gpu is None:
            raise RuntimeError("未找到空闲GPU（需要>=5GB）来加载翻译模型")

        # 加载翻译模型 (Qwen2.5-1.5B)
        print(f"\n加载翻译模型: {translate_model_path} → GPU {translate_gpu}")
        self.translate_tokenizer = AutoTokenizer.from_pretrained(
            translate_model_path, trust_remote_code=True)
        self.translate_model = AutoModelForCausalLM.from_pretrained(
            translate_model_path, torch_dtype=torch.float16,
            device_map={"": translate_gpu}, trust_remote_code=True
        )
        self.translate_model.eval()
        print(f"翻译模型加载完成 (GPU {translate_gpu})")

        # 加载CoT验证模型 (Llama-3-8B)，避开翻译模型占用的GPU
        cot_gpu = find_available_gpu(min_free_gb=18)   # Llama-8B ~15GB
        if cot_gpu is None:
            # 尝试和翻译模型共享GPU（如果显存够）
            print(f"  未找到独立空闲GPU，尝试与翻译模型共享 GPU {translate_gpu}...")
            cot_gpu = translate_gpu
        print(f"\n加载CoT验证模型: {cot_model_path} → GPU {cot_gpu}")
        self.cot_tokenizer = AutoTokenizer.from_pretrained(
            cot_model_path, trust_remote_code=True)
        self.cot_tokenizer.pad_token = self.cot_tokenizer.eos_token
        self.cot_model = AutoModelForCausalLM.from_pretrained(
            cot_model_path, torch_dtype=torch.float16,
            device_map={"": cot_gpu}, trust_remote_code=True
        )
        self.cot_model.eval()
        print(f"CoT验证模型加载完成 (GPU {cot_gpu})")

        # 将BERT公式化模型移到CoT所在GPU（共享显存）
        if self.validator.form_model is not None:
            self.validator.form_model = self.validator.form_model.to(f"cuda:{cot_gpu}")
            print(f"公式化程度模型加载完成 (GPU {cot_gpu})")

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

    def translate(self, labels: List[str]) -> Tuple[bool, str]:
        """翻译Agent：标签 -> 句子 (Qwen2.5-1.5B, GPU 2)"""
        clean_labels = [self.clean_symbol(l) for l in labels]

        prompt = f"""Translate these AAC symbols into ONE simple English sentence. You MUST use ALL symbols.

Symbols: {clean_labels}

CRITICAL RULES:
1. You MUST use ALL symbols in your sentence. Do NOT skip any symbol.
2. Add articles and prepositions as needed to make the sentence natural.
3. Try to RESTRUCTURE the symbols naturally - don't just list them in order.
   Good: "The patient is helped by the doctor in the hospital."
   Bad:  "Doctor help patient hospital."
4. Be creative with context to make the sentence plausible. Even unusual combinations can work with the right context.
5. Do NOT output REJECT. Always try your best to create a sentence.

Output format (one line):
Sentence: <your sentence>

Your output:"""

        messages = [{"role": "user", "content": prompt}]
        text = self.translate_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.translate_tokenizer([text], return_tensors="pt").to(self.translate_model.device)

        with torch.no_grad():
            outputs = self.translate_model.generate(**inputs, max_new_tokens=80, do_sample=False)

        response = self.translate_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if "REJECT" in response.upper():
            return False, ""

        sent_match = re.search(r'[Ss]entence:\s*(.+?)(?:\n|$)', response)
        if sent_match:
            sentence = sent_match.group(1).strip().strip('"\'')
        else:
            sentence = response.strip().split('\n')[0].strip().strip('"\'')

        return True, sentence

    def validate_cot(self, labels: List[str], sentence: str) -> Dict:
        """CoT验证Agent：链式推理评估"""
        results = self.validate_cot_batch([labels], [sentence])
        return results[0]

    def validate_cot_batch(self, labels_list: List[List[str]], sentences: List[str]) -> List[Dict]:
        """批量CoT验证 — 多个样本一起推理，显著提升GPU利用率"""
        prompts = []
        for labels, sentence in zip(labels_list, sentences):
            clean_labels = [self.clean_symbol(l) for l in labels]
            prompt = f"""You are a quality evaluator for AAC (Augmentative and Alternative Communication) text generation.
Your task is to evaluate how natural the generated sentence is, given the input symbols.

Input Symbols: {clean_labels}
Generated Sentence: {sentence}

Analyze this sentence step by step along the following dimensions:

1. **Semantic Coherence**: Do these symbols form a coherent, meaningful scenario? Can you imagine this happening in real life?
   Score 1 = completely absurd/nonsensical
   Score 2 = very forced/unnatural scenario
   Score 3 = somewhat acceptable but odd
   Score 4 = natural and reasonable scenario
   Score 5 = perfectly natural and common scenario

2. **Grammatical Naturalness**: Is the sentence grammatically natural in English? Does it feel like something a native speaker would say, or does it have a "machine-translated" feel with awkward phrasing?
   Score 1 = completely unnatural grammar
   Score 2 = very awkward phrasing
   Score 3 = acceptable but slightly odd grammar
   Score 4 = natural grammar
   Score 5 = perfectly natural, native-like phrasing

3. **Label Integration**: Are the symbols naturally integrated into the sentence? Or are they just listed/plugged in mechanically without linguistic restructuring?
   Score 1 = symbols just listed mechanically
   Score 2 = symbols barely integrated, very forced
   Score 3 = symbols present but integration feels partial
   Score 4 = symbols well integrated with some natural restructuring
   Score 5 = symbols fully integrated with natural linguistic restructuring

4. **Overall Naturalness**: Your overall holistic assessment.
   Score 1 = completely unnatural
   Score 2 = mostly unnatural
   Score 3 = somewhat acceptable
   Score 4 = natural
   Score 5 = perfectly natural

        For each dimension, give your score directly with a brief reason (one sentence).

Output format:
Semantic Coherence: <1-5>
Grammatical Naturalness: <1-5>
Label Integration: <1-5>
Overall Naturalness: <1-5>

Your analysis:"""
            prompts.append(prompt)

        # 构建batch输入
        all_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.cot_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)

        # 使用left-padding以方便批量生成
        old_padding_side = self.cot_tokenizer.padding_side
        self.cot_tokenizer.padding_side = "left"
        inputs = self.cot_tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.cot_model.device)
        self.cot_tokenizer.padding_side = old_padding_side

        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.cot_model.generate(
                **inputs, max_new_tokens=150, do_sample=False,
                pad_token_id=self.cot_tokenizer.eos_token_id
            )

        results = []
        for j in range(len(labels_list)):
            response = self.cot_tokenizer.decode(outputs[j, input_len:], skip_special_tokens=True).strip()

            result = {
                "coherence": 3, "grammaticality": 3,
                "integration": 3, "overall": 3,
                "reasoning": response,
            }

            coh_match = (re.search(r'Semantic\s+Coherence[:\s]*(\d)', response, re.IGNORECASE)
                         or re.search(r'(?:Semantic\s+Coherence)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))
            gram_match = (re.search(r'Grammatical\s+Naturalness[:\s]*(\d)', response, re.IGNORECASE)
                          or re.search(r'(?:Grammatical\s+Naturalness)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))
            integ_match = (re.search(r'Label\s+Integration[:\s]*(\d)', response, re.IGNORECASE)
                           or re.search(r'(?:Label\s+Integration)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))
            overall_match = (re.search(r'Overall\s+Naturalness[:\s]*(\d)', response, re.IGNORECASE)
                             or re.search(r'(?:Overall\s+Naturalness)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))

            if coh_match:
                result["coherence"] = max(1, min(5, int(coh_match.group(1))))
            if gram_match:
                result["grammaticality"] = max(1, min(5, int(gram_match.group(1))))
            if integ_match:
                result["integration"] = max(1, min(5, int(integ_match.group(1))))
            if overall_match:
                result["overall"] = max(1, min(5, int(overall_match.group(1))))

            results.append(result)

        return results

    def generate_training_data(self, output_path: str, num_combinations: int = 1000, debug: bool = False):
        """生成训练数据 — 流水线并行 + 批量CoT推理

        流水线架构:
        - 翻译线程 (GPU A): 持续翻译标签组合 → 放入队列
        - CoT线程 (GPU B): 批量从队列取数据 → 批量CoT推理 + 量化验证
        - 两个GPU同时工作，互不等待
        """
        # 共享状态
        training_data = []
        stats = {
            "accepted": 0,
            "rejected_by_translate": 0,
            "rejected_by_precheck": 0,
            "rejected_by_quant": 0,
            "skipped_empty_labels": 0,
        }
        stats_lock = threading.Lock()
        stop_event = threading.Event()

        # 队列: 翻译线程 → CoT线程
        translated_queue = queue.Queue(maxsize=32)
        COT_BATCH_SIZE = 4

        # ===== 翻译线程 =====
        def translate_worker():
            attempts = 0
            max_attempts = num_combinations * 80
            while attempts < max_attempts and not stop_event.is_set():
                attempts += 1
                labels, combo_type = self.generate_combination()
                if len(labels) < 2:
                    with stats_lock:
                        stats["skipped_empty_labels"] += 1
                    continue

                success, sentence = self.translate(labels)

                if not success or len(sentence) <= 5 or "REJECT" in sentence.upper():
                    with stats_lock:
                        stats["rejected_by_translate"] += 1
                    continue

                # 轻量级预筛选
                coverage, missing = self.validator._coverage_score(labels, sentence)
                if coverage < self.validator.veto_coverage:
                    with stats_lock:
                        stats["rejected_by_precheck"] += 1
                    continue
                formulaicness = self.validator._formulaicness_score(sentence)
                if formulaicness > 0.5:
                    with stats_lock:
                        stats["rejected_by_precheck"] += 1
                    continue

                translated_queue.put((labels, sentence, combo_type))

            # 发送结束信号
            translated_queue.put(None)

        # ===== CoT + 验证线程 =====
        def cot_worker():
            batch = []
            done = False
            while not done and not stop_event.is_set():
                # 收集一个batch
                batch.clear()
                try:
                    # 等待第一个元素
                    item = translated_queue.get(timeout=5.0)
                    if item is None:
                        done = True
                        break
                    batch.append(item)
                    # 非阻塞地收集更多元素
                    while len(batch) < COT_BATCH_SIZE:
                        try:
                            item = translated_queue.get_nowait()
                            if item is None:
                                done = True
                                break
                            batch.append(item)
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                if not batch:
                    continue

                # 批量CoT推理
                labels_list = [b[0] for b in batch]
                sentences = [b[1] for b in batch]
                combo_types = [b[2] for b in batch]

                try:
                    cot_results = self.validate_cot_batch(labels_list, sentences)
                except Exception as e:
                    print(f"\n[WARN] CoT batch failed: {e}, falling back to single")
                    cot_results = []
                    for lbl, sent in zip(labels_list, sentences):
                        try:
                            cot_results.append(self.validate_cot(lbl, sent))
                        except Exception:
                            cot_results.append({"coherence": 3, "grammaticality": 3,
                                                "integration": 3, "overall": 3, "reasoning": ""})

                # 逐条量化验证
                for i in range(len(batch)):
                    result = self.validator.validate(labels_list[i], sentences[i], cot_results[i])
                    action = result["action"]

                    if debug and stats["accepted"] + stats["rejected_by_quant"] < 30:
                        print(f"\n[DEBUG] labels={labels_list[i]}, sentence={sentences[i]}")
                        print(f"  CoT: coh={cot_results[i]['coherence']} gram={cot_results[i]['grammaticality']} "
                              f"integ={cot_results[i]['integration']} ovr={cot_results[i]['overall']}")
                        print(f"  {result['detail']}")

                    if action == "accept":
                        training_data.append({
                            "labels": labels_list[i],
                            "sentence": sentences[i],
                            "type": combo_types[i],
                            "validation": result.get("metrics", {}),
                            "cot_reasoning": cot_results[i].get("reasoning", ""),
                        })
                        with stats_lock:
                            stats["accepted"] += 1
                    else:
                        with stats_lock:
                            stats["rejected_by_quant"] += 1

                # 检查是否已收集足够
                if len(training_data) >= num_combinations:
                    stop_event.set()
                    break

        # 启动流水线
        pbar = tqdm(total=num_combinations, desc="生成数据")

        t_translate = threading.Thread(target=translate_worker, daemon=True)
        t_cot = threading.Thread(target=cot_worker, daemon=True)
        t_translate.start()
        t_cot.start()

        # 主线程: 监控进度
        prev_count = 0
        while not stop_event.is_set() and t_cot.is_alive():
            t_cot.join(timeout=2.0)
            curr = len(training_data)
            if curr > prev_count:
                pbar.update(curr - prev_count)
                prev_count = curr
            if curr >= num_combinations:
                stop_event.set()
                break

        pbar.close()
        stop_event.set()
        t_translate.join(timeout=5.0)
        t_cot.join(timeout=5.0)

        training_data = training_data[:num_combinations]
        random.shuffle(training_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        # 统计报告
        print(f"\n{'='*60}")
        print(f"保存 {len(training_data)} 条数据到: {output_path}")
        total_attempts = (stats["accepted"] + stats["rejected_by_translate"] +
                          stats["rejected_by_precheck"] + stats["rejected_by_quant"] +
                          stats["skipped_empty_labels"])
        print(f"统计: accepted={stats['accepted']}, "
              f"rejected_by_translate={stats['rejected_by_translate']}, "
              f"rejected_by_precheck={stats['rejected_by_precheck']}, "
              f"rejected_by_quant={stats['rejected_by_quant']}, "
              f"skipped_empty_labels={stats['skipped_empty_labels']}, "
              f"total_processed={total_attempts}")

        # CoT各维度分布
        coh_vals = [item["validation"]["cot_detail"]["coherence"] for item in training_data]
        gram_vals = [item["validation"]["cot_detail"]["grammaticality"] for item in training_data]
        integ_vals = [item["validation"]["cot_detail"]["integration"] for item in training_data]
        ovr_vals = [item["validation"]["cot_detail"]["overall"] for item in training_data]

        from collections import Counter

        def print_dist(name, vals):
            raw_scores = [round(v * 4 + 1) for v in vals]
            dist = Counter(raw_scores)
            print(f"\n{name}分布:")
            for s in range(1, 6):
                print(f"  {s}: {'█' * dist.get(s, 0)} ({dist.get(s, 0)})")

        print_dist("Semantic Coherence", coh_vals)
        print_dist("Grammatical Naturalness", gram_vals)
        print_dist("Label Integration", integ_vals)
        print_dist("Overall Naturalness", ovr_vals)

        # Formulaicness分布
        form_vals = [item["validation"].get("formulaicness", 0) for item in training_data]
        if form_vals:
            avg_form = sum(form_vals) / len(form_vals)
            print(f"\nFormulaicness 平均: {avg_form:.3f} (越低越好)")

        # 示例数据
        print(f"\n示例数据:")
        for item in training_data[:10]:
            v = item.get("validation", {})
            cot = v.get("cot_detail", {})
            ovr = round(cot.get("overall", 0) * 4 + 1)
            cov = v.get("coverage", 0)
            form = v.get("formulaicness", 0)
            print(f"  [ovr={ovr} cov={cov:.2f} form={form:.2f}] "
                  f"{item['labels']} → {item['sentence']}")


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
    translate_model_path = "/home/user1/liuduanye/qwen/Qwen2_5-1_5B-Instruct"
    cot_model_path = "/home/user1/liuduanye/Meta-Llama-3-8B-Instruct"
    formulaicness_model_path = "/home/user1/liuduanye/EmotionClassify/models/formulaicness"

    print("=" * 60)
    print("AAC 训练数据生成（翻译Agent + CoT验证 + BERT公式化 + 量化验证器）")
    print("=" * 60)

    prompts_config = PromptsConfig(prompts_path)
    generator = SemanticDataGenerator(
        ontology_path, translate_model_path, cot_model_path,
        formulaicness_model_path, prompts_config
    )
    generator.generate_training_data(output_path, num_combinations=args.num, debug=args.debug)


if __name__ == '__main__':
    main()
