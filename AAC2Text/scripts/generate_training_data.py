"""
基于语义约束的 AAC 训练数据生成器（v2：支持 I/U 主语 + 主语感知Prompt）

流程：
1. 组合标签（主语[I/U/名词] + 动作 + 宾语 + 可选[情绪/地点/时间]）
2. 翻译Agent生成句子（根据主语类型选择不同Prompt）
3. CoT验证Agent：5维度链式推理评估（含人称一致性）
4. 量化验证器评判质量：
   - CoT自然度 (Naturalness): 0.40
   - 标签覆盖率 (Coverage):   0.25
   - 公式化程度 (Formulaicness): 0.15 — BERT回归器
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


# 第三人称组合模板
THIRD_PERSON_COMBOS = ["svo", "sv", "svo_emo", "sv_emo", "svo_place", "svo_time", "sv_time"]
# 第一人称组合模板
FIRST_PERSON_COMBOS = ["i_sv", "i_svo", "i_sv_emo", "i_svo_emo", "i_svo_place", "i_svo_time", "i_sv_time"]
# 第二人称组合模板
SECOND_PERSON_COMBOS = ["u_sv", "u_svo", "u_sv_emo", "u_svo_emo", "u_svo_place", "u_svo_time", "u_sv_time"]
# 全部组合模板
ALL_COMBO_TYPES = THIRD_PERSON_COMBOS + FIRST_PERSON_COMBOS + SECOND_PERSON_COMBOS

# I/U 标签到句子词形的映射
PRONOUN_SENTENCE_FORMS = {
    "I": {"i", "me", "my", "mine", "myself"},
    "U": {"you", "your", "yours", "yourself", "yourselves", "u"},
}


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
    """人设配置管理 — 支持主语感知的英/中翻译模板"""

    SUBJECT_TYPE_TO_EN_TEMPLATE = {
        "first": "translation_prompt_first",
        "second": "translation_prompt_second",
        "third": "translation_prompt_third",
    }
    SUBJECT_TYPE_TO_ZH_TEMPLATE = {
        "first": "translation_prompt_zh_first",
        "second": "translation_prompt_zh_second",
        "third": "translation_prompt_zh_third",
    }

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get_translation_prompt(self, labels: List[str], subject_type: str = "third") -> str:
        """根据主语类型选择对应的英文翻译模板"""
        template_key = self.SUBJECT_TYPE_TO_EN_TEMPLATE.get(subject_type, "translation_prompt_third")
        template = self.config[template_key]
        return template.format(labels=labels)

    def get_translation_prompt_zh(self, labels: List[str], subject_type: str = "third") -> str:
        """根据主语类型选择对应的中文翻译模板"""
        template_key = self.SUBJECT_TYPE_TO_ZH_TEMPLATE.get(subject_type, "translation_prompt_zh_third")
        template = self.config[template_key]
        return template.format(labels=labels)

    def get_validation_cot_prompt(self, labels: List[str], sentence: str, subject_type: str = "third") -> str:
        """获取验证CoT模板"""
        template = self.config['validation_cot_prompt']
        return template.format(labels=labels, sentence=sentence, subject_type=subject_type)


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
        # 跳过 init_weights()，权重从 safetensors 手动加载

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
    - CoT自然度 (Naturalness): 0.40
    - 标签覆盖率 (Coverage):   0.25
    - 公式化程度 (Formulaicness): 0.15 — BERT回归器预测

    综合得分 S = 0.40×naturalness + 0.25×coverage + 0.15×(1-formulaicness)
    阈值: S >= 0.55 → accept
    一票否决: coverage < 0.5 或 naturalness <= 0.5
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
            print("  公式化程度模型加载完成")

    def validate(self, labels: List[str], sentence: str,
                 naturalness_scores: Optional[Dict] = None) -> Dict:
        """量化评估

        Args:
            labels: AAC 标签列表
            sentence: 翻译生成的句子
            naturalness_scores: CoT验证Agent的评估结果
                {"coherence": 1-5, "grammaticality": 1-5,
                 "integration": 1-5, "person": 1-5, "overall": 1-5,
                 "reasoning": "..."}
        """
        # 1) CoT自然度评估 → 归一化到 [0,1]
        if naturalness_scores and isinstance(naturalness_scores, dict):
            coherence = naturalness_scores.get("coherence", 3)
            grammaticality = naturalness_scores.get("grammaticality", 3)
            integration = naturalness_scores.get("integration", 3)
            person = naturalness_scores.get("person", 3)
            overall = naturalness_scores.get("overall", 3)

            norm_coherence = max(0.0, min(1.0, (coherence - 1) / 4.0))
            norm_grammaticality = max(0.0, min(1.0, (grammaticality - 1) / 4.0))
            norm_integration = max(0.0, min(1.0, (integration - 1) / 4.0))
            norm_person = max(0.0, min(1.0, (person - 1) / 4.0))
            norm_overall = max(0.0, min(1.0, (overall - 1) / 4.0))

            norm_naturalness = (
                0.20 * norm_coherence +
                0.25 * norm_grammaticality +
                0.15 * norm_integration +
                0.15 * norm_person +
                0.25 * norm_overall
            )
            reasoning = naturalness_scores.get("reasoning", "")
        else:
            raw_nat = naturalness_scores if isinstance(naturalness_scores, int) else 3
            norm_naturalness = max(0.0, min(1.0, (raw_nat - 1) / 4.0))
            norm_coherence = norm_grammaticality = norm_integration = norm_person = norm_overall = norm_naturalness
            coherence = grammaticality = integration = person = overall = raw_nat
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
                "person": norm_person,
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
                  f"cot=[coh={coherence} gram={grammaticality} integ={integration} "
                  f"person={person} ovr={overall}]")

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
        """公式化程度：输出文本与输入结构的相似度"""
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
    # 指标2: 标签覆盖率（词形变化 + 代词映射）
    # ------------------------------------------------------------------
    def _coverage_score(self, labels: List[str], sentence: str) -> Tuple[float, List[str]]:
        """计算每个 label 在 sentence 中是否出现

        特殊处理：
        - "I" → 匹配 I/me/my/mine/myself
        - "U" → 匹配 you/your/yours/u
        """
        sent_lower = sentence.lower()
        sent_words = set(re.findall(r"[a-z']+", sent_lower))
        missing = []
        hit = 0

        for label in labels:
            # 代词特殊处理
            if label in PRONOUN_SENTENCE_FORMS:
                forms = PRONOUN_SENTENCE_FORMS[label]
                if forms & sent_words:
                    hit += 1
                else:
                    missing.append(label)
                continue

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
    """基于语义约束的训练数据生成器（v3：双语并行生成）

    四模型架构:
    - 英文翻译模型 (Llama-3-8B): 标签 → 英文句子
    - 中文翻译模型 (Qwen2.5-1.5B): 标签 → 中文句子
    - CoT验证模型 (Llama-3-8B): 5维度链式推理评估，含人称一致性
    - 公式化程度模型 (BERT): BERT回归器
    """

    def __init__(self, ontology_path: str, translate_model_path: str,
                 cot_model_path: str, formulaicness_model_path: str,
                 prompts_config: PromptsConfig,
                 zh_model_path: str = None):
        self.prompts_config = prompts_config
        self.validator = QuantitativeValidator(formulaicness_model_path)

        # 加载本体
        with open(ontology_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ontology = data.get("ontology", [])

        # 构建索引 + 可读名称映射
        self.by_semantic_type = {}
        self.readable_names = {}  # clean_id → 人类可读名称
        for item in self.ontology:
            icon_id = item.get("icon_id", "")
            clean_id = re.sub(r'_\d+[a-z]?$', '', icon_id)
            item["clean_id"] = clean_id
            st = item.get("semantic_type", "")
            if st:
                if st not in self.by_semantic_type:
                    self.by_semantic_type[st] = []
                self.by_semantic_type[st].append(item)
            # 构建可读名称：优先 core_semantic，其次 label（去编号），最后 clean_id
            # 特殊处理：I/U 保持原样，_to 后缀的动词去掉 _to
            if clean_id not in self.readable_names:
                # I 和 U 是 AAC 代词符号，保持原样
                if clean_id == "I":
                    self.readable_names[clean_id] = "I"
                elif clean_id == "U":
                    self.readable_names[clean_id] = "you"
                else:
                    core = item.get("core_semantic", "").strip()
                    label = re.sub(r'_\d+[a-z]?$', '', item.get("label", "")).replace("_", " ").strip()
                    # 去掉 core_semantic 中的下划线（如 take_into_custody → take into custody）
                    core_display = core.replace("_", " ") if core else ""
                    if core_display and len(core_display.split()) >= 2:
                        self.readable_names[clean_id] = core_display
                    elif label and label != clean_id.replace("_", " ") and len(label.split()) >= 2:
                        self.readable_names[clean_id] = label
                    else:
                        # fallback：去掉变体编号和语法标记
                        name = re.sub(r'_\d+[a-z]?$', '', clean_id)
                        name = re.sub(r'_\d+(?=_)', '', name)
                        name = name.replace("_,_to", "").replace("_to", "")
                        name = name.replace("_", " ")
                        name = re.sub(r'\s+', ' ', name).strip()
                        self.readable_names[clean_id] = name

        # 提取关键类别 — 覆盖所有3295个图标，不丢弃任何有效图标
        # 仅按 semantic_type 归入对应类别，不做前缀/长度过滤
        self.persons = self._get_items_by_types(
            ["person", "relationship", "subject", "topic", "PRON"])
        self.actions = self._get_items_by_types(
            ["action", "verb", "activity", "emission"])
        self.objects = self._get_items_by_types(
            ["entity", "object", "food", "drink", "body", "body part", "body_part",
             "animal", "tool", "clothing", "device", "material", "event", "noun",
             "quantity", "symbol", "content", "geography", "art", "structure",
             "medicine", "electronics", "reference", "part_of_an_organism",
             "substance", "education", "mathematics", "character", "moderator",
             "numeral", "letter", "geometry"])
        self.emotions = self._get_items_by_types(
            ["emotion", "quality", "abstract", "adjective", "modifier",
             "concept", "posture", "shape", "emotional aid", "adverb", "adv"])
        self.places = self._get_items_by_types(["place", "location"])
        self.times = self._get_items_by_types(["time"])

        # 提取 I 和 U 代词（单独提取，不在 persons 中）
        self.i_pronoun = None
        self.u_pronoun = None
        for item in self.ontology:
            if item["clean_id"] == "I":
                self.i_pronoun = item
            elif item["clean_id"] == "U":
                self.u_pronoun = item

        # 验证覆盖率
        covered_ids = set()
        for lst in [self.persons, self.actions, self.objects,
                    self.emotions, self.places, self.times]:
            for i in lst:
                covered_ids.add(i["clean_id"])
        covered_ids.add("I")
        covered_ids.add("U")
        all_ids = set(i["clean_id"] for i in self.ontology)
        uncovered = all_ids - covered_ids
        # 对未覆盖的图标，按 cs_role 兜底归入最近类别
        if uncovered:
            for item in self.ontology:
                if item["clean_id"] in uncovered:
                    cs = item.get("cs_role", "")
                    if cs == "WHO":
                        self.persons.append(item)
                    elif cs == "WHAT_DOING":
                        self.actions.append(item)
                    elif cs in ("WHAT", "WHERE", "WHEN"):
                        self.objects.append(item)
                    elif cs == "HOW":
                        self.emotions.append(item)
                    else:
                        self.objects.append(item)  # 默认归入objects
                    covered_ids.add(item["clean_id"])

        total_covered = len(self.persons) + len(self.actions) + len(self.objects) + \
                        len(self.emotions) + len(self.places) + len(self.times) + \
                        (1 if self.i_pronoun else 0) + (1 if self.u_pronoun else 0)
        final_coverage = len(covered_ids)
        print(f"第三人称: {len(self.persons)}, I代词: {self.i_pronoun is not None}, "
              f"U代词: {self.u_pronoun is not None}, "
              f"动作: {len(self.actions)}, 物体: {len(self.objects)}, "
              f"情绪/修饰: {len(self.emotions)}, 地点: {len(self.places)}, 时间: {len(self.times)}")
        print(f"覆盖图标: {final_coverage}/{len(all_ids)} ({final_coverage/len(all_ids)*100:.1f}%)")

        # 根据翻译模型大小确定GPU需求
        # 检测模型参数量：简单通过路径名判断
        model_name = os.path.basename(translate_model_path).lower()
        if any(sz in model_name for sz in ["7b", "8b"]):
            translate_min_gb = 18  # 7B/8B ~15GB fp16
        else:
            translate_min_gb = 5

        # 自动分配 GPU（翻译模型）
        translate_gpu = find_available_gpu(min_free_gb=translate_min_gb)
        if translate_gpu is None:
            raise RuntimeError(f"未找到空闲GPU（需要>={translate_min_gb}GB）来加载翻译模型")

        # 加载翻译模型
        print(f"\n加载翻译模型: {translate_model_path} → GPU {translate_gpu}")
        self.translate_tokenizer = AutoTokenizer.from_pretrained(
            translate_model_path, trust_remote_code=True)
        self.translate_tokenizer.pad_token = self.translate_tokenizer.eos_token
        self.translate_model = AutoModelForCausalLM.from_pretrained(
            translate_model_path, torch_dtype=torch.float16,
            device_map={"": translate_gpu}, trust_remote_code=True
        )
        self.translate_model.eval()
        print(f"英文翻译模型加载完成 (GPU {translate_gpu})")

        # 加载中文翻译模型 (Qwen, 擅长中文EN→ZH翻译)
        self.zh_model = None
        self.zh_tokenizer = None
        if zh_model_path:
            zh_min_gb = 5  # Qwen1.5B ~3GB
            # 优先找和翻译模型不同的GPU
            zh_gpu = None
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                for line in result.stdout.strip().split("\n"):
                    idx_str, free_str = line.strip().split(", ")
                    idx, free = int(idx_str), int(free_str)
                    if idx != translate_gpu and free >= zh_min_gb * 1024:
                        zh_gpu = idx
                        break
            except Exception:
                pass
            if zh_gpu is None:
                zh_gpu = translate_gpu  # fallback: 和翻译模型共享

            print(f"\n加载中文翻译模型: {zh_model_path} → GPU {zh_gpu}")
            self.zh_tokenizer = AutoTokenizer.from_pretrained(
                zh_model_path, trust_remote_code=True)
            self.zh_model = AutoModelForCausalLM.from_pretrained(
                zh_model_path, torch_dtype=torch.float16,
                device_map={"": zh_gpu}, trust_remote_code=True
            )
            self.zh_model.eval()
            print(f"中文翻译模型加载完成 (GPU {zh_gpu})")

        # 从中文本体直接加载中文可读名称，无需实时翻译
        self._load_chinese_names_from_ontology()

        # 加载CoT验证模型，避开翻译模型占用的GPU
        # nvidia-smi 此时已反映翻译模型占用的显存，优先找不同GPU
        cot_min_gb = 18  # Llama-8B ~15GB
        cot_gpu = find_available_gpu(min_free_gb=cot_min_gb)
        if cot_gpu is not None and cot_gpu == translate_gpu:
            # 同一张GPU不够放两个8B，继续找
            cot_gpu = None
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                for line in result.stdout.strip().split("\n"):
                    idx_str, free_str = line.strip().split(", ")
                    idx, free = int(idx_str), int(free_str)
                    if idx != translate_gpu and free >= cot_min_gb * 1024:
                        cot_gpu = idx
                        break
            except Exception:
                pass
        if cot_gpu is None:
            # 最后尝试与翻译模型共享GPU
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

    def _load_chinese_names_from_ontology(self):
        """从中文本体 aac_full_ontology_zh.json 直接加载中文名称，无需实时翻译"""
        zh_ontology_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "processed", "aac_full_ontology_zh.json"
        )

        self.readable_names_zh = {}

        if not os.path.exists(zh_ontology_path):
            print(f"中文本体不存在: {zh_ontology_path}，中文图标名将使用英文fallback")
            return

        with open(zh_ontology_path, 'r', encoding='utf-8') as f:
            zh_data = json.load(f)

        zh_items = zh_data.get("ontology", [])
        # 建立 icon_id → 中文映射
        for item in zh_items:
            icon_id = item.get("icon_id", "")
            if not icon_id:
                continue
            clean_id = re.sub(r'_\d+[a-z]?$', '', icon_id)
            # 优先 core_semantic_zh，其次 label_zh
            zh_name = item.get("core_semantic_zh", "")
            if not zh_name or not re.search(r'[\u4e00-\u9fff]', zh_name):
                zh_name = item.get("label_zh", "")
            if zh_name and re.search(r'[\u4e00-\u9fff]', zh_name):
                if clean_id not in self.readable_names_zh:
                    self.readable_names_zh[clean_id] = zh_name

        # 特殊处理：I/U 保持原样
        self.readable_names_zh["I"] = "我"
        self.readable_names_zh["U"] = "你"

        print(f"中文图标名称: 从中文本体加载 {len(self.readable_names_zh)} 条")

    def clean_symbol_zh(self, symbol: str) -> str:
        """将 clean_id 转为中文可读名称"""
        if symbol in self.readable_names_zh:
            return self.readable_names_zh[symbol]
        # fallback to English
        return self.clean_symbol(symbol)

    def _combo_type_to_subject_type(self, combo_type: str) -> str:
        """从组合类型推断主语类型"""
        if combo_type.startswith("i_"):
            return "first"
        elif combo_type.startswith("u_"):
            return "second"
        else:
            return "third"

    def generate_combination(self) -> Tuple[List[str], str, str]:
        """生成一组标签组合

        Returns:
            (labels, combo_type, subject_type)
            subject_type: "first" / "second" / "third"
        """
        labels = []
        combo_type = random.choice(ALL_COMBO_TYPES)
        subject_type = self._combo_type_to_subject_type(combo_type)

        # 根据主语类型选择主语
        if subject_type == "first":
            subject = self.i_pronoun
        elif subject_type == "second":
            subject = self.u_pronoun
        else:
            subject = random.choice(self.persons) if self.persons else None

        action = random.choice(self.actions) if self.actions else None
        obj = random.choice(self.objects) if self.objects else None
        emotion = random.choice(self.emotions) if self.emotions else None
        place = random.choice(self.places) if self.places else None
        time_ = random.choice(self.times) if self.times else None

        # 去掉前缀得到基础模式
        base = combo_type.split("_", 1)[1] if "_" in combo_type and combo_type[0] in "iu" else combo_type

        if base == "svo" and subject and action and obj:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"]]
        elif base == "sv" and subject and action:
            labels = [subject["clean_id"], action["clean_id"]]
        elif base == "svo_emo" and subject and action and obj and emotion:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"], emotion["clean_id"]]
        elif base == "sv_emo" and subject and action and emotion:
            labels = [subject["clean_id"], action["clean_id"], emotion["clean_id"]]
        elif base == "svo_place" and subject and action and obj and place:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"], place["clean_id"]]
        elif base == "svo_time" and subject and action and obj and time_:
            labels = [subject["clean_id"], action["clean_id"], obj["clean_id"], time_["clean_id"]]
        elif base == "sv_time" and subject and action and time_:
            labels = [subject["clean_id"], action["clean_id"], time_["clean_id"]]

        return labels, combo_type, subject_type

    def clean_symbol(self, symbol: str) -> str:
        """将 clean_id 转为人类可读的名称，用于 Prompt"""
        # 优先使用预构建的可读名称映射
        if symbol in self.readable_names:
            name = self.readable_names[symbol]
        else:
            # fallback：去掉变体编号和语法标记
            name = re.sub(r'_\d+[a-z]?$', '', symbol)
            name = re.sub(r'_\d+(?=_)', '', name)
            name = name.replace("_,_to", "").replace("_to", "")
            name = name.replace("_", " ")
            name = re.sub(r'\s+', ' ', name).strip()
        return name

    def _generate_single(self, prompt: str, use_zh_model: bool = False) -> str:
        """用指定模型生成单条回复"""
        if use_zh_model and self.zh_model is not None:
            model = self.zh_model
            tokenizer = self.zh_tokenizer
        else:
            model = self.translate_model
            tokenizer = self.translate_tokenizer

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)

        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def translate(self, labels: List[str], subject_type: str = "third") -> Tuple[bool, str, str]:
        """双语翻译Agent：标签 -> 英文句子 + 中文句子

        Returns:
            (success, sentence_en, sentence_zh)
        """
        # 英文生成
        clean_labels_en = [self.clean_symbol(l) for l in labels]
        prompt_en = self.prompts_config.get_translation_prompt(clean_labels_en, subject_type)
        response_en = self._generate_single(prompt_en)

        if "REJECT" in response_en.upper():
            return False, "", ""

        sent_match = re.search(r'[Ss]entence:\s*(.+?)(?:\n|$)', response_en)
        if sent_match:
            sentence_en = sent_match.group(1).strip().strip('"\'')
        else:
            sentence_en = response_en.strip().split('\n')[0].strip().strip('"\'')

        if len(sentence_en) <= 5:
            return False, "", ""

        # 中文翻译（EN→ZH，用Qwen模型翻译英文句子，不是icon→中文直译）
        sentence_zh = self._translate_en_to_zh(sentence_en)

        return True, sentence_en, sentence_zh

    def _translate_en_to_zh(self, sentence_en: str) -> str:
        """用Qwen模型将英文句子翻译为中文（EN→ZH，比icon→ZH直译更可靠）"""
        template = self.prompts_config.config.get("translation_en_to_zh", "")
        if not template:
            return sentence_en  # fallback
        prompt = template.format(sentence_en=sentence_en)
        response_zh = self._generate_single(prompt, use_zh_model=True)
        # 清理输出
        sentence_zh = response_zh.strip().split('\n')[0].strip()
        for ch in ['"', "'", '\u201c', '\u201d']:
            sentence_zh = sentence_zh.strip(ch)
        # 去掉可能的"中文："前缀
        for prefix in ['中文：', '中文:', '翻译：', '翻译:']:
            if sentence_zh.startswith(prefix):
                sentence_zh = sentence_zh[len(prefix):].strip()
        return sentence_zh if sentence_zh else sentence_en

    def validate_cot(self, labels: List[str], sentence: str, subject_type: str = "third") -> Dict:
        """CoT验证Agent：链式推理评估"""
        results = self.validate_cot_batch([labels], [sentence], [subject_type])
        return results[0]

    def validate_cot_batch(self, labels_list: List[List[str]], sentences: List[str],
                           subject_types: List[str] = None) -> List[Dict]:
        """批量CoT验证 — 5维度评估（含人称一致性）"""
        if subject_types is None:
            subject_types = ["third"] * len(labels_list)

        prompts = []
        for labels, sentence, subject_type in zip(labels_list, sentences, subject_types):
            clean_labels = [self.clean_symbol(l) for l in labels]
            prompt = self.prompts_config.get_validation_cot_prompt(clean_labels, sentence, subject_type)
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
            all_texts, return_tensors="pt", padding=True, truncation=True, max_length=768
        ).to(self.cot_model.device)
        self.cot_tokenizer.padding_side = old_padding_side

        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.cot_model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                pad_token_id=self.cot_tokenizer.eos_token_id
            )

        results = []
        for j in range(len(labels_list)):
            response = self.cot_tokenizer.decode(outputs[j, input_len:], skip_special_tokens=True).strip()

            result = {
                "coherence": 3, "grammaticality": 3,
                "integration": 3, "person": 3, "overall": 3,
                "reasoning": response,
            }

            coh_match = (re.search(r'Semantic\s+Coherence[:\s]*(\d)', response, re.IGNORECASE)
                         or re.search(r'(?:Semantic\s+Coherence)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))
            gram_match = (re.search(r'Grammatical\s+Naturalness[:\s]*(\d)', response, re.IGNORECASE)
                          or re.search(r'(?:Grammatical\s+Naturalness)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))
            integ_match = (re.search(r'Label\s+Integration[:\s]*(\d)', response, re.IGNORECASE)
                           or re.search(r'(?:Label\s+Integration)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))
            person_match = (re.search(r'Person\s+Consistency[:\s]*(\d)', response, re.IGNORECASE)
                            or re.search(r'(?:Person\s+Consistency)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))
            overall_match = (re.search(r'Overall\s+Naturalness[:\s]*(\d)', response, re.IGNORECASE)
                             or re.search(r'(?:Overall\s+Naturalness)[\s\S]*?Score[:\s]*(\d)', response, re.IGNORECASE))

            if coh_match:
                result["coherence"] = max(1, min(5, int(coh_match.group(1))))
            if gram_match:
                result["grammaticality"] = max(1, min(5, int(gram_match.group(1))))
            if integ_match:
                result["integration"] = max(1, min(5, int(integ_match.group(1))))
            if person_match:
                result["person"] = max(1, min(5, int(person_match.group(1))))
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
                labels, combo_type, subject_type = self.generate_combination()
                if len(labels) < 2:
                    with stats_lock:
                        stats["skipped_empty_labels"] += 1
                    continue

                success, sentence_en, sentence_zh = self.translate(labels, subject_type)

                if not success or len(sentence_en) <= 5 or "REJECT" in sentence_en.upper():
                    with stats_lock:
                        stats["rejected_by_translate"] += 1
                    continue

                # 轻量级预筛选（基于英文句子）
                coverage, missing = self.validator._coverage_score(labels, sentence_en)
                if coverage < self.validator.veto_coverage:
                    with stats_lock:
                        stats["rejected_by_precheck"] += 1
                    continue
                formulaicness = self.validator._formulaicness_score(sentence_en)
                if formulaicness > 0.5:
                    with stats_lock:
                        stats["rejected_by_precheck"] += 1
                    continue

                translated_queue.put((labels, sentence_en, sentence_zh, combo_type, subject_type))

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
                sentences_en = [b[1] for b in batch]
                sentences_zh = [b[2] for b in batch]
                combo_types = [b[3] for b in batch]
                subject_types = [b[4] for b in batch]

                try:
                    cot_results = self.validate_cot_batch(labels_list, sentences_en, subject_types)
                except Exception as e:
                    print(f"\n[WARN] CoT batch failed: {e}, falling back to single")
                    cot_results = []
                    for lbl, sent, st in zip(labels_list, sentences_en, subject_types):
                        try:
                            cot_results.append(self.validate_cot(lbl, sent, st))
                        except Exception:
                            cot_results.append({"coherence": 3, "grammaticality": 3,
                                                "integration": 3, "person": 3, "overall": 3,
                                                "reasoning": ""})

                # 逐条量化验证
                for i in range(len(batch)):
                    result = self.validator.validate(labels_list[i], sentences_en[i], cot_results[i])
                    action = result["action"]

                    if debug and stats["accepted"] + stats["rejected_by_quant"] < 30:
                        print(f"\n[DEBUG] labels={labels_list[i]}, en={sentences_en[i]}, zh={sentences_zh[i]}, subject={subject_types[i]}")
                        cot_r = cot_results[i]
                        print(f"  CoT: coh={cot_r['coherence']} gram={cot_r['grammaticality']} "
                              f"integ={cot_r['integration']} person={cot_r['person']} ovr={cot_r['overall']}")
                        print(f"  {result['detail']}")

                    if action == "accept":
                        training_data.append({
                            "labels": labels_list[i],
                            "sentence_en": sentences_en[i],
                            "sentence_zh": sentences_zh[i],
                            "type": combo_types[i],
                            "subject_type": subject_types[i],
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

        # 主语类型分布
        from collections import Counter
        subj_dist = Counter(item.get("subject_type", "unknown") for item in training_data)
        print(f"\n主语类型分布:")
        for st in ["first", "second", "third"]:
            cnt = subj_dist.get(st, 0)
            print(f"  {st}: {cnt} ({cnt/len(training_data)*100:.1f}%)")

        # CoT各维度分布
        coh_vals = [item["validation"]["cot_detail"]["coherence"] for item in training_data]
        gram_vals = [item["validation"]["cot_detail"]["grammaticality"] for item in training_data]
        integ_vals = [item["validation"]["cot_detail"]["integration"] for item in training_data]
        person_vals = [item["validation"]["cot_detail"]["person"] for item in training_data]
        ovr_vals = [item["validation"]["cot_detail"]["overall"] for item in training_data]

        def print_dist(name, vals):
            raw_scores = [round(v * 4 + 1) for v in vals]
            dist = Counter(raw_scores)
            print(f"\n{name}分布:")
            for s in range(1, 6):
                print(f"  {s}: {'█' * dist.get(s, 0)} ({dist.get(s, 0)})")

        print_dist("Semantic Coherence", coh_vals)
        print_dist("Grammatical Naturalness", gram_vals)
        print_dist("Label Integration", integ_vals)
        print_dist("Person Consistency", person_vals)
        print_dist("Overall Naturalness", ovr_vals)

        # Formulaicness分布
        form_vals = [item["validation"].get("formulaicness", 0) for item in training_data]
        if form_vals:
            avg_form = sum(form_vals) / len(form_vals)
            print(f"\nFormulaicness 平均: {avg_form:.3f} (越低越好)")

        # 示例数据（按主语类型分组）
        for st in ["first", "second", "third"]:
            items = [item for item in training_data if item.get("subject_type") == st]
            if items:
                st_label = {"first": "1st", "second": "2nd", "third": "3rd"}[st]
                print(f"\n{st_label} person 示例:")
                for item in items[:3]:
                    v = item.get("validation", {})
                    cot = v.get("cot_detail", {})
                    ovr = round(cot.get("overall", 0) * 4 + 1)
                    cov = v.get("coverage", 0)
                    form = v.get("formulaicness", 0)
                    print(f"  [ovr={ovr} cov={cov:.2f} form={form:.2f}] "
                          f"{item['labels']}")
                    print(f"    EN: {item['sentence_en']}")
                    print(f"    ZH: {item['sentence_zh']}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--translate-model', type=str, default=None,
                        help='英文翻译模型路径 (默认 Llama-3-8B)')
    parser.add_argument('--zh-model', type=str, default=None,
                        help='中文翻译模型路径 (默认 Qwen2.5-1.5B)')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    args = parser.parse_args()

    base_dir = "/home/user1/liuduanye/EmotionClassify/AAC2Text"
    ontology_path = f"{base_dir}/data/processed/aac_full_ontology.json"
    output_path = args.output or f"{base_dir}/data/processed/training_data.json"
    prompts_path = f"{base_dir}/config/prompts.yaml"
    translate_model_path = args.translate_model or "/home/user1/liuduanye/Meta-Llama-3-8B-Instruct"
    zh_model_path = args.zh_model or "/home/user1/liuduanye/qwen/Qwen2_5-1_5B-Instruct"
    cot_model_path = "/home/user1/liuduanye/Meta-Llama-3-8B-Instruct"
    formulaicness_model_path = "/home/user1/liuduanye/EmotionClassify/models/formulaicness"

    print("=" * 60)
    print("AAC 训练数据生成 v3（双语：EN=Llama icon→en, ZH=Qwen en→zh翻译）")
    print(f"英文模型: {translate_model_path}")
    print(f"中文模型: {zh_model_path}")
    print("=" * 60)

    prompts_config = PromptsConfig(prompts_path)
    generator = SemanticDataGenerator(
        ontology_path, translate_model_path, cot_model_path,
        formulaicness_model_path, prompts_config,
        zh_model_path=zh_model_path
    )
    generator.generate_training_data(output_path, num_combinations=args.num, debug=args.debug)


if __name__ == '__main__':
    main()
