#!/usr/bin/env python3
"""
AAC 完整交流系统 - 象形图翻译 + 情感分析 + 下一轮预测

整合两个模块：
1. AAC2Text: AAC象形图 → 自然语言翻译
2. EmotionClassify: 自然语言 → 情感分类 + 下一轮预测

使用方式：
    python aac_emotion_pipeline.py --interactive
    python aac_emotion_pipeline.py --symbols "I" "want_to" "water"
"""

import os
import sys
import json
import re
import threading
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import deque
from typing import List, Dict, Tuple, Optional
import numpy as np

# ==================== 情感相关常量 ====================

EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LABEL2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}
ID2LABEL = {idx: emotion for emotion, idx in LABEL2ID.items()}

EMOTION_VALENCE = {
    "neutral": 0.0, "happiness": 1.0, "surprise": 0.3,
    "sadness": -0.8, "anger": -0.9, "fear": -0.7, "disgust": -0.6
}

# ==================== AAC图标预测器（语义嵌入版本）====================

class AACIconPredictor:
    """基于语义嵌入匹配预测可能的AAC图标"""

    def __init__(self, ontology_path: str = None, embedding_model: str = './Model/all-MiniLM-L6-v2'):
        self.ontology = {}
        self.icon_list = []
        self.icon_embeddings = None
        self.embedding_model = None
        self.embedding_model_name = embedding_model

        if ontology_path and os.path.exists(ontology_path):
            self._load_ontology(ontology_path)
            self._init_embeddings()
        else:
            print("[IconPredictor] Warning: No ontology loaded")

    def _load_ontology(self, ontology_path: str):
        """加载AAC语义本体"""
        with open(ontology_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        skipped = 0
        for item in data.get('ontology', []):
            icon_id = item.get('icon_id', '')
            label = item.get('label', icon_id)
            
            # 跳过空图标
            if not icon_id or not label:
                skipped += 1
                continue
                
            semantic_type = item.get('semantic_type', 'unknown')
            core_semantic = item.get('core_semantic', '')

            icon_info = {
                'icon_id': icon_id,
                'label': label,
                'semantic_type': semantic_type,
                'core_semantic': core_semantic,
                'grammar_role': item.get('grammar_role', ''),
                'can_combine_with': item.get('can_combine_with', []),
                'super_concept': item.get('super_concept', ''),
                'typical_objects': item.get('typical_objects', []),
                # 构建用于嵌入的文本
                'embed_text': f"{label}: {core_semantic} {item.get('super_concept', '')}".strip()
            }

            self.ontology[icon_id] = icon_info
            self.icon_list.append(icon_info)

        print(f"[IconPredictor] Loaded {len(self.ontology)} icons (skipped {skipped} empty)")

    def _init_embeddings(self):
        """初始化嵌入模型并预计算图标嵌入

        注意：sentence_transformers 为可选依赖。若未安装或加载失败，
        则 embedding_model / icon_embeddings 置为 None，RAG 分支自动停用，
        但不影响翻译、情感分析、SASRec 预测主流程。
        """
        try:
            print(f"[IconPredictor] Loading embedding model: {self.embedding_model_name}")
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.embedding_model_name, local_files_only=True)

            # 预计算所有图标的嵌入
            texts = [icon['embed_text'] for icon in self.icon_list]
            print(f"[IconPredictor] Computing embeddings for {len(texts)} icons...")

            self.icon_embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            print(f"[IconPredictor] Embeddings shape: {self.icon_embeddings.shape}")
        except Exception as e:
            self.embedding_model = None
            self.icon_embeddings = None
            print(f"[IconPredictor] Warning: embedding model unavailable ({type(e).__name__}: {e}). "
                  f"RAG icon suggestions disabled; SASRec + translation still work.")

    def predict_next_icons_by_context(self,
                                       conversation_context: List[str],
                                       current_emotion: str,
                                       next_emotion: str = None,
                                       used_symbols: List[str] = None,
                                       current_sentence: str = "",
                                       top_k: int = 10,
                                       lambda_balance: float = 0.3) -> Dict:
        """
        Emotional RAG: 基于预测的下一个情感生成引导词，增强语义检索
        
        公式:
        S(i) = λ·cos(E(Q_emo), E(i)) + (1-λ)·cos(E(Q_orig), E(i))
        
        其中:
        - Q_orig: 用户原始查询文本
        - Q_emo: 预测情感引导的增强查询
        - E(·): 语义嵌入模型（all-MiniLM-L6-v2）
        - i: AAC 图标
        - λ: 平衡系数
        
        流程:
        1. 用户输入 → 翻译 → 情感识别 → 预测下一个情感 E
        2. 用 E 生成情感引导词 (Emotion Prompt)
        3. 分别计算 Q_orig 和 Q_emo 的嵌入
        4. 分别计算余弦相似度，按 λ 加权融合
        5. 推荐更符合当前情感场景的图标
        """
        if self.embedding_model is None or self.icon_embeddings is None:
            return {'actions': [], 'entities': [], 'emotions': [], 'others': [], 'combinations': []}

        used_set = set(used_symbols) if used_symbols else set()
        from sentence_transformers import util

        # ============ Emotional RAG 核心实现 ============
        # 用预测的下一个情感 E 生成引导词
        target_emotion = next_emotion if next_emotion else current_emotion
        
        # 1. 获取情感引导词配置
        emotion_config = self._get_emotion_rag_config(target_emotion, current_emotion)
        
        # 2. 构建两个查询
        # Q_orig: 原始查询
        Q_orig = current_sentence if current_sentence else ""
        
        # Q_emo: 情感增强查询
        emotion_prompts = emotion_config.get('emotion_prompts', [])
        emotion_keywords = emotion_config.get('keywords', [])
        
        Q_emo = Q_orig
        if emotion_prompts:
            Q_emo = Q_emo + " " + " ".join(emotion_prompts[:2])
        if emotion_keywords:
            Q_emo = Q_emo + " " + " ".join(emotion_keywords[:3])

        # 3. 分别计算两个查询的嵌入
        E_orig = self.embedding_model.encode(
            Q_orig, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        E_emo = self.embedding_model.encode(
            Q_emo, 
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 4. 分别计算余弦相似度
        sim_orig = util.cos_sim(E_orig, self.icon_embeddings)[0]  # (N,)
        sim_emo = util.cos_sim(E_emo, self.icon_embeddings)[0]    # (N,)

        # 5. 按公式融合：S(i) = λ·sim_emo + (1-λ)·sim_orig
        lambda_weight = lambda_balance  # 情感增强的权重
        combined_sim = lambda_weight * sim_emo + (1 - lambda_weight) * sim_orig

        # 6. 综合评分
        icon_scores = []
        for idx, icon_info in enumerate(self.icon_list):
            icon_id = icon_info['icon_id']
            label = icon_info['label']
            semantic_type = icon_info['semantic_type']

            if icon_id in used_set or label in used_set:
                continue

            # 融合后的语义相似度分数
            score = combined_sim[idx].item()

            # 情感类型偏好
            prefer_types = emotion_config.get('prefer_types', [])
            if semantic_type in prefer_types:
                score += 0.1
            
            # 情感匹配图标加成
            positive_keywords = emotion_config.get('positive_keywords', [])
            if any(kw in label.lower() for kw in positive_keywords):
                score += 0.2
            
            # 负面关键词惩罚
            negative_keywords = emotion_config.get('negative_keywords', [])
            if any(kw in label.lower() for kw in negative_keywords):
                score -= 0.35

            # 情绪图标特殊处理
            if semantic_type == 'emotion':
                emotion_match = emotion_config.get('emotion_match', [])
                if any(kw in label.lower() for kw in emotion_match):
                    score += 0.4
                else:
                    score -= 0.5

            icon_scores.append((icon_id, score, combined_sim[idx].item(), 
                               sim_orig[idx].item(), sim_emo[idx].item()))

        # 7. 排序
        icon_scores.sort(key=lambda x: -x[1])

        # 8. 分类返回
        actions = []
        entities = []
        emotions = []
        others = []

        for icon_id, final_score, combined_sim_val, orig_sim, emo_sim in icon_scores[:top_k * 2]:
            if len(actions) >= 5 and len(entities) >= 5:
                break
                
            icon_info = self.ontology.get(icon_id, {})
            semantic_type = icon_info.get('semantic_type', 'unknown')
            label = icon_info.get('label', icon_id)

            item = {
                'icon_id': icon_id,
                'label': label,
                'semantic_type': semantic_type,
                'sim_combined': round(combined_sim_val, 3),
                'sim_orig': round(orig_sim, 3),
                'sim_emo': round(emo_sim, 3),
                'final_score': round(final_score, 3)
            }

            if semantic_type == 'action' and len(actions) < 5:
                actions.append(item)
            elif semantic_type in ['entity', 'object', 'noun', 'person', 'food', 'drink'] and len(entities) < 5:
                entities.append(item)
            elif semantic_type == 'emotion' and len(emotions) < 3:
                emotions.append(item)
            elif len(others) < 3:
                others.append(item)

        return {
            'actions': actions,
            'entities': entities,
            'emotions': emotions,
            'others': others,
            'combinations': self._generate_combinations(actions[:3], entities[:3]),
            # Emotional RAG 详细信息
            'emotional_rag': {
                'Q_orig': Q_orig,                           # 原始查询
                'Q_emo': Q_emo,                             # 情感增强查询
                'target_emotion': target_emotion,           # 目标情感（预测的下一个情感）
                'current_emotion': current_emotion,         # 当前情感
                'emotion_prompts': emotion_prompts,         # 情感引导词
                'lambda': lambda_weight,                    # 平衡系数
                'formula': 'S(i) = λ·cos(E(Q_emo), E(i)) + (1-λ)·cos(E(Q_orig), E(i))'
            }
        }

    def _get_emotion_rag_config(self, target_emotion: str, current_emotion: str = None) -> Dict:
        """
        Emotional RAG 配置：根据预测的下一个情感生成引导词
        
        Args:
            target_emotion: 预测的下一个情感
            current_emotion: 当前情感（用于情感转换场景）
        
        Returns:
            情感引导配置
        """
        # 情感 → 引导词映射（基于 Emotion Prompt 设计）
        emotion_rag_configs = {
            "happiness": {
                # 情感引导词：描述期望的情感状态
                "emotion_prompts": ["happy", "joyful", "excited", "celebrate"],
                # 关联关键词：相关活动/实体
                "keywords": ["fun", "play", "smile", "love", "share", "enjoy"],
                # 正向匹配图标
                "positive_keywords": ["happy", "laugh", "smile", "celebrate", "excited", "joy", "love"],
                # 负向匹配图标
                "negative_keywords": ["sad", "cry", "angry", "fear", "pain"],
                # 偏好类型
                "prefer_types": ["action", "entity"],
                # 情绪图标匹配
                "emotion_match": ["happy", "excited", "smile", "joy", "laugh"]
            },
            "sadness": {
                "emotion_prompts": ["sad", "need comfort", "support"],
                "keywords": ["help", "comfort", "friend", "family", "care", "listen"],
                "positive_keywords": ["sad", "cry", "comfort", "help", "support", "hug", "friend"],
                "negative_keywords": ["happy", "celebrate", "laugh", "excited", "fun"],
                "prefer_types": ["action", "entity"],
                "emotion_match": ["sad", "cry", "tear", "depress"]
            },
            "anger": {
                "emotion_prompts": ["frustrated", "need calm", "relax"],
                "keywords": ["calm", "relax", "breathe", "peace", "quiet"],
                "positive_keywords": ["angry", "frustrated", "calm", "relax", "peace"],
                "negative_keywords": ["happy", "celebrate", "fun", "excited"],
                "prefer_types": ["action"],
                "emotion_match": ["angry", "frustrated", "mad", "rage"]
            },
            "fear": {
                "emotion_prompts": ["scared", "need safety", "protection"],
                "keywords": ["safe", "protect", "help", "security", "comfort"],
                "positive_keywords": ["scared", "afraid", "safe", "protect", "help", "security"],
                "negative_keywords": ["happy", "celebrate", "fun"],
                "prefer_types": ["action", "entity"],
                "emotion_match": ["scared", "afraid", "worried", "fear", "anxious"]
            },
            "disgust": {
                "emotion_prompts": ["dislike", "want to avoid", "clean"],
                "keywords": ["clean", "away", "remove", "different", "change"],
                "positive_keywords": ["disgust", "clean", "away", "remove", "avoid"],
                "negative_keywords": ["love", "enjoy", "good"],
                "prefer_types": ["action"],
                "emotion_match": ["disgust", "yuck", "gross"]
            },
            "surprise": {
                "emotion_prompts": ["surprised", "curious", "wonder"],
                "keywords": ["look", "see", "find", "discover", "new", "unexpected"],
                "positive_keywords": ["surprise", "wow", "amazing", "wonder", "discover"],
                "negative_keywords": ["boring", "normal", "usual"],
                "prefer_types": ["action", "entity"],
                "emotion_match": ["surprise", "wow", "shock", "amaze"]
            },
            "neutral": {
                "emotion_prompts": [],
                "keywords": ["want", "need", "do", "get", "go"],
                "positive_keywords": [],
                "negative_keywords": [],
                "prefer_types": ["action", "entity", "object"],
                "emotion_match": []
            }
        }
        
        config = emotion_rag_configs.get(target_emotion, emotion_rag_configs["neutral"])
        
        # 情感转换场景增强
        if current_emotion and current_emotion != target_emotion:
            transition_boost = self._get_transition_boost(current_emotion, target_emotion)
            if transition_boost:
                # 合并转换场景的额外引导词
                config = {
                    **config,
                    "emotion_prompts": config["emotion_prompts"] + transition_boost.get("emotion_prompts", []),
                    "keywords": config["keywords"] + transition_boost.get("keywords", []),
                }
        
        return config
    
    def _get_transition_boost(self, from_emotion: str, to_emotion: str) -> Dict:
        """
        情感转换场景的额外引导词
        
        例如：sadness → happiness 意味着用户需要安慰/鼓励
        """
        transitions = {
            ("sadness", "happiness"): {
                "emotion_prompts": ["cheer up", "hope"],
                "keywords": ["celebrate", "enjoy", "play", "friend", "support"]
            },
            ("anger", "neutral"): {
                "emotion_prompts": ["calm down"],
                "keywords": ["relax", "breathe", "peace", "rest"]
            },
            ("fear", "neutral"): {
                "emotion_prompts": ["feel safe"],
                "keywords": ["safe", "protect", "comfort", "stay"]
            },
            ("neutral", "happiness"): {
                "emotion_prompts": ["excited"],
                "keywords": ["fun", "celebrate", "play", "enjoy"]
            },
            ("happiness", "neutral"): {
                "emotion_prompts": ["content"],
                "keywords": ["continue", "enjoy", "keep"]
            },
            ("neutral", "sadness"): {
                "emotion_prompts": ["express feelings"],
                "keywords": ["talk", "share", "comfort", "support"]
            }
        }
        
        return transitions.get((from_emotion, to_emotion), {})

    def _generate_combinations(self, actions: List[Dict], entities: List[Dict]) -> List[Dict]:
        """生成合理的图标组合建议"""
        combinations = []

        for action in actions[:2]:
            for entity in entities[:2]:
                action_info = self.ontology.get(action['icon_id'], {})
                can_combine = action_info.get('can_combine_with', [])

                entity_info = self.ontology.get(entity['icon_id'], {})
                entity_semantic = entity_info.get('semantic_type', '')

                if not can_combine or entity_semantic in can_combine:
                    combinations.append({
                        'action': action['icon_id'],
                        'entity': entity['icon_id'],
                        'label': f"{action['label']} + {entity['label']}"
                    })

        return combinations[:4]

    def search_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict]:
        """根据关键词搜索图标"""
        results = []
        keyword_lower = keyword.lower()

        for icon_info in self.icon_list:
            if keyword_lower in icon_info['embed_text'].lower():
                results.append({
                    'icon_id': icon_info['icon_id'],
                    'label': icon_info['label'],
                    'semantic_type': icon_info['semantic_type'],
                    'core_semantic': icon_info['core_semantic']
                })

        return results[:top_k]


# ==================== 情感分类模型 ====================

class MultiTaskEmotionClassifier(nn.Module):
    """多任务情绪分类模型"""
    def __init__(self, base_model_path, num_labels=7, lora_config=None):
        super().__init__()
        from transformers import RobertaModel
        from peft import get_peft_model

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
        self.next_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, return_next=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        main_hidden = hidden_states[:, 0, :]
        main_logits = self.main_classifier(main_hidden)

        if return_next:
            next_logits = self.next_classifier(main_hidden)
            return main_logits, next_logits
        return main_logits


# ==================== AAC翻译器 ====================

class AACTranslator:
    """AAC符号到自然语言翻译器"""
    def __init__(self, model_path, base_model_path, device='cuda', sft_model_path=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"[AAC2Text] Loading base model: {base_model_path}, target device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True,
            local_files_only=True
        )

        # 可选：先合并 SFT LoRA 到基座。
        # 关键：中文 DPO 权重必须叠在 SFT(aac_model_zh) 之上，不能从基座直接加载，
        # 否则输出退化（如 "A simple one!"）。配方与 AAC2Text/scripts/test_zh.py 一致。
        if (sft_model_path and os.path.exists(sft_model_path)
                and os.path.abspath(sft_model_path) != os.path.abspath(model_path)):
            print(f"[AAC2Text] Merging SFT LoRA first: {sft_model_path}")
            self.model = PeftModel.from_pretrained(self.model, sft_model_path)
            self.model = self.model.merge_and_unload()
            print("[AAC2Text] SFT merged into base")

        # 加载主 LoRA 权重（SFT 或 DPO）
        if os.path.exists(model_path):
            print(f"[AAC2Text] Loading LoRA weights: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)

        self.model.eval()

        # Warmup: 首次推理会触发 CUDA kernel 编译，非常慢，提前跑一次
        print("[AAC2Text] Warming up model (first inference)...")
        import time as _time
        _t0 = _time.time()
        with torch.no_grad():
            _dummy = self.tokenizer("Hello", return_tensors="pt").to(self.model.device)
            self.model.generate(**_dummy, max_new_tokens=1, do_sample=False)
        _warmup_time = _time.time() - _t0
        print(f"[AAC2Text] Warmup done ({_warmup_time:.1f}s)")

        print("[AAC2Text] Model loaded successfully")

    def translate(self, symbols: List[str]) -> str:
        """将AAC符号列表翻译为自然语言句子"""
        import time as _time
        _t0 = _time.time()

        # 与中文 SFT 训练（scripts/train_zh.py）保持一致：中文指令 -> 中文输出
        prompt = f"请把这些 AAC 图标序列翻译成一个简单的中文句子：{' '.join(symbols)}"
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                repetition_penalty=1.15,  # 抑制模型重复/闲扯
                stop_strings=["<|eot_id|>", "<|end_of_text|>", "\n"],
                tokenizer=self.tokenizer,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        # 只取第一句，避免模型闲扯/重复/解释图标（如 "xx表示…"）。
        # 关键：模型输出的是全角句号 "。"，原代码只匹配半角 "." 导致从不截断。
        for sep in ['\n', '。', '！', '？', '；']:
            idx = response.find(sep)
            if idx != -1:
                response = response[:idx + len(sep)]
                break
        response = response.strip()

        _infer_time = _time.time() - _t0
        print(f"[AAC2Text] Inference time: {_infer_time:.2f}s, result: {response}")
        return response


# ==================== 情感分析器 ====================

class EmotionAnalyzer:
    """情感分析和预测器"""
    def __init__(self, model_path, base_model_path, device='cuda'):
        from transformers import RobertaTokenizer
        from peft import LoraConfig, TaskType

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # LoRA配置
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )

        print(f"[EmotionClassify] Loading base model: {base_model_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = MultiTaskEmotionClassifier(
            base_model_path, num_labels=7, lora_config=lora_config
        )
        self.model.load_state_dict(
            torch.load(os.path.join(model_path, "model.pt"), map_location=self.device),
            strict=False
        )
        self.model.to(self.device)
        self.model.eval()

        self.history = deque(maxlen=10)
        print("[EmotionClassify] Model loaded successfully")

    def analyze(self, text: str, conversation_history: List[Dict] = None) -> Dict:
        """分析文本情感"""
        from collections import Counter

        # 1. 分析单句情绪
        single_inputs = self.tokenizer(
            text, return_tensors='pt', max_length=256,
            truncation=True, padding=True
        )
        single_inputs = {k: v.to(self.device) for k, v in single_inputs.items()}

        with torch.no_grad():
            single_logits, _ = self.model(**single_inputs, return_next=True)
            single_probs = F.softmax(single_logits, dim=-1)[0]
            single_pred_idx = torch.argmax(single_probs).item()

        single_emotion = EMOTION_LIST[single_pred_idx]

        # 2. 计算当前状态（最近3轮情绪的众数）
        if conversation_history and len(conversation_history) > 0:
            recent_singles = [t.get('single_emotion', 'neutral') for t in conversation_history[-3:]]
            recent_singles.append(single_emotion)
            current_emotion = Counter(recent_singles).most_common(1)[0][0]
        else:
            current_emotion = single_emotion

        # 3. 分析主题情绪和预测下一轮（使用完整对话）
        if conversation_history and len(conversation_history) > 0:
            dialog_text = ""
            for turn in conversation_history:
                role = turn.get('role', 'user')
                sentence = turn['sentence']
                if role == 'user':
                    dialog_text += f"User: {sentence}\n"
                else:
                    dialog_text += f"Assistant: {sentence}\n"
            dialog_text += f"User: {text}\n"

            dialog_inputs = self.tokenizer(
                dialog_text, return_tensors='pt', max_length=256,
                truncation=True, padding=True
            )
            dialog_inputs = {k: v.to(self.device) for k, v in dialog_inputs.items()}

            with torch.no_grad():
                theme_logits, next_logits = self.model(**dialog_inputs, return_next=True)
                theme_probs = F.softmax(theme_logits, dim=-1)[0]
                next_probs = F.softmax(next_logits, dim=-1)[0]

                theme_pred_idx = torch.argmax(theme_probs).item()
                next_pred_idx = torch.argmax(next_probs).item()

            theme_emotion = EMOTION_LIST[theme_pred_idx]
            next_emotion = EMOTION_LIST[next_pred_idx]

            theme_probabilities = {EMOTION_LIST[i]: theme_probs[i].item() for i in range(7)}
            next_probabilities = {EMOTION_LIST[i]: next_probs[i].item() for i in range(7)}

            history_emotions = [t.get('single_emotion', 'neutral') for t in conversation_history]
            all_emotions = history_emotions + [single_emotion]
            emotion_distribution = dict(Counter(all_emotions))

        else:
            theme_emotion = single_emotion
            next_emotion = single_emotion
            theme_probabilities = {EMOTION_LIST[i]: single_probs[i].item() for i in range(7)}
            next_probabilities = theme_probabilities.copy()
            emotion_distribution = {single_emotion: 1}

        self.history.append(current_emotion)

        return {
            'single_emotion': single_emotion,
            'single_confidence': single_probs[single_pred_idx].item(),
            'single_probabilities': {EMOTION_LIST[i]: single_probs[i].item() for i in range(7)},
            'theme_emotion': theme_emotion,
            'theme_confidence': theme_probabilities.get(theme_emotion, 0.5),
            'theme_probabilities': theme_probabilities,
            'current_emotion': current_emotion,
            'next_emotion': next_emotion,
            'next_confidence': next_probabilities.get(next_emotion, 0.5),
            'next_probabilities': next_probabilities,
            'emotion_distribution': emotion_distribution
        }

    def get_trend(self, conversation_history=None) -> Dict:
        """获取情感趋势

        Args:
            conversation_history: 可选，传入该会话的对话历史（每项为含 'single_emotion' 的 dict）
                以实现每会话独立趋势；为 None 时回退到 self.history（兼容旧用法）。
        """
        if conversation_history is not None:
            hist = [t.get('single_emotion', 'neutral') if isinstance(t, dict) else t
                    for t in conversation_history]
        else:
            hist = list(self.history)

        if len(hist) < 2:
            return {'trend': 'stable', 'direction': 0.0}

        valences = [EMOTION_VALENCE.get(e, 0.0) for e in hist]
        x = np.arange(len(valences))
        slope = np.polyfit(x, valences, 1)[0]

        if slope > 0.1:
            trend = 'improving'
        elif slope < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'

        return {'trend': trend, 'direction': slope}

    def reset(self):
        """重置历史"""
        self.history.clear()


# ==================== 增量预测状态管理 ====================

class IncrementalState:
    """增量预测模式的状态管理器"""
    def __init__(self, max_seq_len=50):
        self.current_sequence = []   # 当前轮次的icon序列
        self.current_cs_roles = []   # 当前轮次的CS角色
        self.turn_history = []       # 已提交的轮次历史
        self.max_seq_len = max_seq_len

    def add_icon(self, icon_id: str, cs_role: str):
        """添加一个icon到当前序列"""
        self.current_sequence.append(icon_id)
        self.current_cs_roles.append(cs_role)
        # 滑动窗口
        if len(self.current_sequence) > self.max_seq_len:
            self.current_sequence = self.current_sequence[-self.max_seq_len:]
            self.current_cs_roles = self.current_cs_roles[-self.max_seq_len:]

    def commit_turn(self, translation: str, emotion: Dict):
        """提交当前轮次"""
        self.turn_history.append({
            'sequence': self.current_sequence.copy(),
            'cs_roles': self.current_cs_roles.copy(),
            'translation': translation,
            'emotion': emotion,
        })
        self.current_sequence = []
        self.current_cs_roles = []

    def undo(self) -> bool:
        """撤销上一个icon"""
        if self.current_sequence:
            self.current_sequence.pop()
            self.current_cs_roles.pop()
            return True
        return False

    def reset(self):
        """重置当前序列"""
        self.current_sequence = []
        self.current_cs_roles = []

    def get_context_for_sasrec(self) -> Tuple[List[str], List[str]]:
        """获取SASRec输入: 最近3轮 + 当前序列"""
        context_icons = []
        context_cs = []
        for turn in self.turn_history[-3:]:
            context_icons.extend(turn['sequence'])
            context_cs.extend(turn['cs_roles'])
        context_icons.extend(self.current_sequence)
        context_cs.extend(self.current_cs_roles)
        return context_icons[-self.max_seq_len:], context_cs[-self.max_seq_len:]


# ==================== 完整Pipeline ====================

class AACEmotionPipeline:
    """AAC完整交流Pipeline：翻译 + 情感分析 + 预测 + 图标推荐"""

    def __init__(self,
                 aac_model_path: str,
                 aac_base_model_path: str,
                 emotion_model_path: str,
                 emotion_base_model_path: str,
                 ontology_path: str = None,
                 embedding_model: str = './Model/all-MiniLM-L6-v2',
                 device: str = 'cuda',
                 mode: str = 'batch',
                 log_path: str = './output/incremental_usage.jsonl',
                 aac_sft_model_path: str = None,
                 aac_translator_device: str = None):
        """初始化Pipeline

        Args:
            mode: 'batch' (整段输入后预测) 或 'incremental' (逐icon即时预测)
            aac_translator_device: 翻译器(8B Llama)单独占用的 GPU。
                与 SASRec/RoBERTa 分开可彻底消除点击时的 GPU 争用（默认与 device 相同）。
        """

        print("=" * 60)
        print(f"Initializing AAC Emotion Pipeline (mode={mode})")
        print("=" * 60)

        self.mode = mode
        self.device = device
        self.log_path = log_path
        self.session_id = None  # 由服务端按会话设置（多用户并发落盘区分）
        self._log_lock = threading.Lock()

        # 加载翻译器（8B Llama，可放独立 GPU 以不阻塞预测）
        translator_device = aac_translator_device or device
        self.translator = AACTranslator(
            aac_model_path, aac_base_model_path, translator_device,
            sft_model_path=aac_sft_model_path,
        )

        # 加载情感分析器
        self.analyzer = EmotionAnalyzer(emotion_model_path, emotion_base_model_path, device)

        # 加载图标预测器
        if ontology_path is None:
            ontology_path = "./AAC2Text/data/processed/aac_full_ontology.json"
        self.icon_predictor = AACIconPredictor(ontology_path, embedding_model=embedding_model)

        # 对话历史
        self.conversation_history = []

        # 增量模式: 加载SASRec和融合预测器
        self.incremental_state = None
        self.fused_predictor = None

        if mode == 'incremental':
            self._init_incremental_mode(ontology_path, device)

        print("\nPipeline initialized successfully!")
        print("=" * 60)

    def _init_incremental_mode(self, ontology_path: str, device: str):
        """初始化增量预测模式的SASRec模型"""
        from sequence_model.sasrec import SASRec, CS_ROLE_TO_ID, build_item_vocabulary
        from sequence_model.fusion import FusedIconPredictor

        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            sasrec_config = config.get('sasrec', {})
        else:
            sasrec_config = {}

        # 解析模型路径并优先加载 checkpoint
        model_path = sasrec_config.get('model_path', './output/sasrec/best_model.pt')
        checkpoint = None
        if os.path.exists(model_path):
            print(f"[Incremental] Loading trained SASRec from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        else:
            print(f"[Incremental] Warning: SASRec model not found at {model_path}")

        # 词表：优先使用 checkpoint 自带的 item2idx/idx2item（=训练时的 review5000 词表），
        # 否则回退到从本体构建（此时若 checkpoint 存在会与模型尺寸不匹配）。
        if checkpoint and 'item2idx' in checkpoint and 'idx2item' in checkpoint:
            self.item2idx = checkpoint['item2idx']
            self.idx2item = checkpoint['idx2item']
        else:
            self.item2idx, self.idx2item = build_item_vocabulary(self.icon_predictor.ontology)
        num_items = len(self.item2idx) - 1  # exclude padding

        # 创建SASRec模型
        saved_args = checkpoint.get('args', {}) if checkpoint else {}
        if checkpoint is not None:
            self.sasrec_model = SASRec(
                num_items=num_items,
                num_cs_roles=len(CS_ROLE_TO_ID),
                hidden_size=saved_args.get('hidden_size', 64),
                num_heads=saved_args.get('num_heads', 2),
                num_blocks=saved_args.get('num_blocks', 2),
                max_seq_len=saved_args.get('max_seq_len', 50),
                dropout=0.0,  # no dropout at inference
                cs_role_emb_dim=saved_args.get('cs_role_emb_dim', 16),
            ).to(device)
            self.sasrec_model.load_state_dict(checkpoint['model_state_dict'])
            self.sasrec_model.eval()
        else:
            print(f"[Incremental] Creating untrained SASRec (predictions will be random)")
            self.sasrec_model = SASRec(
                num_items=num_items,
                num_cs_roles=len(CS_ROLE_TO_ID),
                hidden_size=sasrec_config.get('hidden_size', 64),
                num_heads=sasrec_config.get('num_heads', 2),
                num_blocks=sasrec_config.get('num_blocks', 2),
                max_seq_len=sasrec_config.get('max_seq_len', 50),
                dropout=0.0,
                cs_role_emb_dim=sasrec_config.get('cs_role_emb_dim', 16),
            ).to(device)
            self.sasrec_model.eval()

        # 创建融合预测器
        self.fused_predictor = FusedIconPredictor(
            sasrec_model=self.sasrec_model,
            icon_predictor=self.icon_predictor,
            item2idx=self.item2idx,
            idx2item=self.idx2item,
            alpha=sasrec_config.get('fusion_alpha', 0.5),
            lambda_balance=sasrec_config.get('fusion_lambda', 0.3),
            device=device,
        )

        # 初始化增量状态
        self.incremental_state = IncrementalState(
            max_seq_len=sasrec_config.get('max_seq_len', 50)
        )

        print("[Incremental] SASRec + FusedIconPredictor loaded")

    def _log_usage(self, record: Dict):
        """将真实使用记录追加写入 jsonl（供后续用真实数据重训 / 优化）。
        多用户并发时由 _log_lock 保护，并把 session_id 写入记录以便区分。"""
        if not self.log_path:
            return
        try:
            record = {
                'ts': datetime.now().isoformat(),
                'session_id': getattr(self, 'session_id', None),
                **record
            }
            with self._log_lock:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[UsageLog] Warning: failed to write log: {e}")

    def process(self, symbols: List[str], role: str = "user") -> Dict:
        """处理AAC符号输入"""
        # Step 1: 翻译
        sentence = self.translator.translate(symbols)

        # Step 2: 情感分析
        emotion_result = self.analyzer.analyze(sentence, self.conversation_history)

        # Step 3: 记录对话历史
        self.conversation_history.append({
            'role': role,
            'symbols': symbols,
            'sentence': sentence,
            'single_emotion': emotion_result['single_emotion'],
            'theme_emotion': emotion_result['theme_emotion'],
            'current_emotion': emotion_result['current_emotion']
        })

        # Step 4: 获取趋势
        trend = self.analyzer.get_trend()

        # Step 5: 预测可能的下一个图标
        history_sentences = [t['sentence'] for t in self.conversation_history[:-1]]
        used_symbols = []
        for t in self.conversation_history:
            used_symbols.extend(t.get('symbols', []))

        icon_predictions = self.icon_predictor.predict_next_icons_by_context(
            conversation_context=history_sentences,
            current_emotion=emotion_result['single_emotion'],  # 使用单句情绪
            next_emotion=emotion_result['next_emotion'],
            used_symbols=used_symbols,
            current_sentence=sentence
        )

        return {
            'input': {
                'symbols': symbols,
                'role': role
            },
            'translation': {
                'sentence': sentence
            },
            'emotion': {
                'single': emotion_result['single_emotion'],
                'single_confidence': emotion_result['single_confidence'],
                'theme': emotion_result['theme_emotion'],
                'theme_confidence': emotion_result['theme_confidence'],
                'current': emotion_result['current_emotion'],
            },
            'prediction': {
                'next_emotion': emotion_result['next_emotion'],
                'confidence': emotion_result['next_confidence'],
                'probabilities': emotion_result['next_probabilities']
            },
            'icon_recommendations': icon_predictions,
            'trend': trend,
            'emotion_distribution': emotion_result.get('emotion_distribution', {}),
            'conversation_turn': len(self.conversation_history)
        }

    # ==================== 增量模式API ====================

    def add_icon(self, icon_id: str, state=None, history=None, session_id=None,
                 do_translate: bool = True) -> Dict:
        """增量模式: 用户点击一个icon -> 即时预测下一个icon

        Args:
            do_translate: True=每次点击都用 8B Llama 翻译（质量高但每次 ~4-5s，
                是点击卡顿的根因）；False=用图标 label 快速拼接预览（即时），
                真正的高质量翻译放到后台/提交时再做。Web 端默认 False 以保证点击即时响应。

        Args:
            icon_id: 用户选择的icon ID
            state: 可选，传入 IncrementalState 以实现每会话独立状态（多用户并发）。
                  为 None 时使用 self.incremental_state（兼容交互/批量旧用法）。
            history: 可选，传入该会话的 conversation_history list。
            session_id: 可选，写入使用日志以区分多用户。

        Returns:
            预测结果，包含下一个icon推荐
        """
        if self.mode != 'incremental':
            raise RuntimeError("add_icon() only works in incremental mode. Use process() for batch mode.")

        state = state if state is not None else self.incremental_state
        history = history if history is not None else self.conversation_history
        self.session_id = session_id

        # 1. 更新增量状态
        cs_role = self.icon_predictor.ontology.get(icon_id, {}).get('cs_role', 'WHAT')
        state.add_icon(icon_id, cs_role)

        # 2. 局部翻译（点击即时预览用）。
        #    默认 do_translate=False：直接拼最近3个图标的 label 作为预览，
        #    不跑 8B Llama，保证每次点击 <0.1s。高质量翻译由服务端后台线程异步完成。
        current_seq = state.current_sequence
        if do_translate:
            if len(current_seq) >= 2:
                partial_translation = self.translator.translate(current_seq[-3:])
            else:
                partial_translation = self.icon_predictor.ontology.get(icon_id, {}).get('label', icon_id)
        else:
            # 快速预览：最近3个图标的 label 直接拼接（如 "我 想 水"）
            _tail = current_seq[-3:] if current_seq else (
                [icon_id] if icon_id != '__undo_rerun__' else [])
            partial_translation = ''.join(
                self.icon_predictor.ontology.get(i, {}).get('label', i) for i in _tail
            )

        # 3. 情感分析 (使用局部翻译)
        emotion_result = self.analyzer.analyze(partial_translation, history)

        # 4. 获取SASRec上下文 + 融合预测
        context_icons, context_cs = state.get_context_for_sasrec()

        used_symbols = []
        for t in history:
            used_symbols.extend(t.get('symbols', []))
        used_symbols.extend(current_seq)

        predictions = self.fused_predictor.predict_next(
            current_sequence=context_icons,
            current_cs_roles=context_cs,
            current_emotion=emotion_result['single_emotion'],
            next_emotion=emotion_result.get('next_emotion'),
            current_sentence=partial_translation,
            used_symbols=used_symbols,
            conversation_context=[t['sentence'] for t in history],
        )

        # 5. 返回结果
        cs_display = [f"{r}:{i}" for i, r in zip(current_seq, state.current_cs_roles)]

        # 6. 记录真实使用数据（供后续重训/优化）；跳过 undo 内部重跑的哨兵
        if icon_id != '__undo_rerun__':
            top_items = []
            for cat in ('actions', 'entities', 'emotions', 'others'):
                for it in predictions.get(cat, []):
                    top_items.append({
                        'icon_id': it['icon_id'],
                        'label': it.get('label', ''),
                        'score': it.get('final_score', 0.0),
                    })
            top_items.sort(key=lambda x: -x['score'])
            self._log_usage({
                'event': 'icon_add',
                'prefix': current_seq[:-1] if len(current_seq) > 1 else [],
                'chosen_icon': icon_id,
                'chosen_label': self.icon_predictor.ontology.get(icon_id, {}).get('label', icon_id),
                'model_top_k': top_items[:10],
                'partial_translation': partial_translation,
                'emotion_single': emotion_result['single_emotion'],
            })

        return {
            'current_sequence': current_seq[:],
            'cs_display': cs_display,
            'partial_translation': partial_translation,
            'emotion': {
                'single': emotion_result['single_emotion'],
                'confidence': emotion_result['single_confidence'],
                'current': emotion_result['current_emotion'],
                'next': emotion_result.get('next_emotion', 'neutral'),
            },
            'next_icon_predictions': predictions,
            'sequence_length': len(current_seq),
        }

    def commit_sequence(self, state=None, history=None, session_id=None) -> Dict:
        """增量模式: 用户完成当前序列 -> 完整翻译+情感分析+提交

        Args:
            state: 可选，传入 IncrementalState 实现每会话独立状态。
            history: 可选，该会话的 conversation_history list。
            session_id: 可选，写入使用日志区分多用户。

        Returns:
            完整分析结果（类似batch模式的process()输出）
        """
        if self.mode != 'incremental':
            raise RuntimeError("commit_sequence() only works in incremental mode.")

        state = state if state is not None else self.incremental_state
        history = history if history is not None else self.conversation_history
        self.session_id = session_id

        current_seq = state.current_sequence
        if not current_seq:
            return {'error': 'No icons in current sequence'}

        # 1. 完整翻译
        full_translation = self.translator.translate(current_seq)

        # 2. 完整情感分析
        emotion_result = self.analyzer.analyze(full_translation, history)

        # 3. 记录到对话历史
        history.append({
            'role': 'user',
            'symbols': current_seq,
            'sentence': full_translation,
            'single_emotion': emotion_result['single_emotion'],
            'theme_emotion': emotion_result['theme_emotion'],
            'current_emotion': emotion_result['current_emotion'],
        })

        # 4. 提交到增量状态
        state.commit_turn(full_translation, emotion_result)

        # 4b. 记录真实使用数据（整句序列 + 翻译）
        self._log_usage({
            'event': 'commit',
            'full_sequence': current_seq,
            'full_translation': full_translation,
            'emotion_single': emotion_result['single_emotion'],
            'emotion_theme': emotion_result['theme_emotion'],
        })

        # 5. 获取趋势
        trend = self.analyzer.get_trend(history)

        # 6. 完整预测 (使用完整翻译)
        history_sentences = [t['sentence'] for t in history[:-1]]
        used_symbols = []
        for t in history:
            used_symbols.extend(t.get('symbols', []))

        icon_predictions = self.icon_predictor.predict_next_icons_by_context(
            conversation_context=history_sentences,
            current_emotion=emotion_result['single_emotion'],
            next_emotion=emotion_result['next_emotion'],
            used_symbols=used_symbols,
            current_sentence=full_translation,
        )

        return {
            'input': {'symbols': current_seq, 'role': 'user'},
            'translation': {'sentence': full_translation},
            'emotion': {
                'single': emotion_result['single_emotion'],
                'single_confidence': emotion_result['single_confidence'],
                'theme': emotion_result['theme_emotion'],
                'current': emotion_result['current_emotion'],
            },
            'prediction': {
                'next_emotion': emotion_result['next_emotion'],
                'confidence': emotion_result['next_confidence'],
            },
            'icon_recommendations': icon_predictions,
            'trend': trend,
            'conversation_turn': len(history),
        }

    def undo_icon(self, state=None, history=None) -> Dict:
        """增量模式: 撤销上一个icon

        Args:
            state: 可选，传入 IncrementalState 实现每会话独立状态。
            history: 可选，该会话的 conversation_history list。
        """
        if self.mode != 'incremental':
            raise RuntimeError("undo_icon() only works in incremental mode.")

        state = state if state is not None else self.incremental_state
        history = history if history is not None else self.conversation_history

        success = state.undo()
        current_seq = state.current_sequence

        if success and current_seq:
            # 重新预测（do_translate=False：undo 也是一次点击，保持即时）
            result = self.add_icon('__undo_rerun__', state, history, do_translate=False)
            # 恢复正确序列（add_icon会多加一个，但我们不需要）
            state.undo()  # 撤销add_icon内部添加的

            # 手动重新预测而不修改状态
            context_icons, context_cs = state.get_context_for_sasrec()
            _tail = current_seq[-3:] if len(current_seq) >= 2 else (
                [current_seq[-1]] if current_seq else [])
            partial_translation = ''.join(
                self.icon_predictor.ontology.get(i, {}).get('label', i) for i in _tail
            )
            emotion_result = self.analyzer.analyze(partial_translation, history)

            predictions = self.fused_predictor.predict_next(
                current_sequence=context_icons,
                current_cs_roles=context_cs,
                current_emotion=emotion_result['single_emotion'],
                next_emotion=emotion_result.get('next_emotion'),
                current_sentence=partial_translation,
            )

            return {
                'current_sequence': current_seq[:],
                'partial_translation': partial_translation,
                'next_icon_predictions': predictions,
                'undone': True,
            }

        return {
            'current_sequence': current_seq[:],
            'undone': False,
        }

    def process_conversation(self, conversation: List[Dict]) -> Dict:
        """处理完整对话"""
        self.reset()
        results = []

        for turn in conversation:
            symbols = turn['symbols']
            role = turn.get('role', 'user')
            result = self.process(symbols, role)
            results.append(result)

        from collections import Counter
        emotions = [r['emotion']['current'] for r in results]
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]

        return {
            'turns': results,
            'summary': {
                'total_turns': len(results),
                'emotion_distribution': dict(emotion_counts),
                'dominant_emotion': dominant_emotion,
                'final_prediction': results[-1]['prediction'] if results else None
            }
        }

    def reset(self):
        """重置对话历史"""
        self.conversation_history = []
        self.analyzer.reset()


# ==================== 交互模式 ====================

def interactive_mode(pipeline: AACEmotionPipeline):
    """交互式模式 (batch)"""
    print("\n" + "=" * 60)
    print("AAC Emotion Pipeline - Interactive Mode (Batch)")
    print("=" * 60)
    print("\nEnter AAC symbols separated by spaces (e.g., 'I want_to water')")
    print("Enter 'quit' to exit, 'reset' to clear history")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("AAC symbols> ").strip()

            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break

            if user_input.lower() == 'reset':
                pipeline.reset()
                print("History cleared.\n")
                continue

            if not user_input:
                continue

            symbols = user_input.split()
            result = pipeline.process(symbols)

            turn_num = result['conversation_turn']
            single_emo = result['emotion']['single']
            single_conf = result['emotion']['single_confidence']
            theme_emo = result['emotion']['theme']
            current_emo = result['emotion']['current']

            print(f"\n{'─' * 60}")
            print(f"📌 Turn {turn_num}: {result['translation']['sentence']}")
            print(f"   😊 Single: {single_emo} ({single_conf:.0%})")
            if turn_num > 1:
                print(f"   📍 Current State: {current_emo} (recent 3 turns)")
                print(f"   🎯 Theme: {theme_emo} (model predicted)")
            else:
                print(f"   🎯 Theme: {theme_emo} (model predicted)")
            print(f"   🔮 Next Emotion: {result['prediction']['next_emotion']} ({result['prediction']['confidence']:.0%})")
            print(f"   📈 Trend: {result['trend']['trend']}")

            # 显示图标推荐
            icons = result['icon_recommendations']
            rag_info = icons.get('emotional_rag', {})

            print(f"\n   🎯 Recommended Next Icons (Emotional RAG):")
            print(f"      λ={rag_info.get('lambda', 0.3):.1f}, Target: {rag_info.get('target_emotion', 'N/A')}")

            if icons.get('actions'):
                action_strs = [f"{a['label']}(sim:{a['sim_combined']:.2f})" for a in icons['actions'][:3]]
                print(f"      Actions: {', '.join(action_strs)}")

            if icons.get('entities'):
                entity_strs = [f"{e['label']}(sim:{e['sim_combined']:.2f})" for e in icons['entities'][:3]]
                print(f"      Entities: {', '.join(entity_strs)}")

            if icons.get('emotions'):
                emotion_strs = [f"{e['label']}" for e in icons['emotions'][:2]]
                print(f"      Emotions: {', '.join(emotion_strs)}")

            if icons.get('combinations'):
                combo_strs = [c['label'] for c in icons['combinations'][:2]]
                print(f"      Try: {', '.join(combo_strs)}")

            print(f"{'─' * 60}")

            if turn_num > 1:
                singles = [t['single_emotion'] for t in pipeline.conversation_history]
                print(f"   Single Trajectory: {' → '.join(singles)}")
                print(f"   Emotion Counts: {result['emotion_distribution']}")
                print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def incremental_mode(pipeline: AACEmotionPipeline):
    """交互式增量模式: 逐icon输入，即时预测下一个icon"""
    print("\n" + "=" * 60)
    print("AAC Emotion Pipeline - Incremental Mode (IME-style)")
    print("=" * 60)
    print("\nEnter ONE icon at a time (like an input method)")
    print("Commands: '.' = commit sequence, 'u' = undo, 'reset' = clear, 'quit' = exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("icon> ").strip()

            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break

            if user_input.lower() == 'reset':
                pipeline.incremental_state.reset()
                pipeline.conversation_history = []
                pipeline.analyzer.reset()
                print("History cleared.\n")
                continue

            if user_input.lower() in ('u', 'undo'):
                result = pipeline.undo_icon()
                if result.get('undone'):
                    seq = result.get('current_sequence', [])
                    print(f"  <- Undo. Current: {seq}")
                    # Show updated predictions
                    preds = result.get('next_icon_predictions', {})
                    _display_incremental_predictions(preds)
                else:
                    print("  Nothing to undo.")
                continue

            if user_input.lower() in ('.', 'commit'):
                result = pipeline.commit_sequence()
                if 'error' in result:
                    print(f"  {result['error']}")
                    continue

                turn = result['conversation_turn']
                trans = result['translation']['sentence']
                emo = result['emotion']
                print(f"\n  [Committed Turn {turn}] {trans}")
                print(f"  Emotion: {emo['single']} ({emo['single_confidence']:.0%}) | Next: {result['prediction']['next_emotion']}")

                # Show full icon recommendations
                icons = result.get('icon_recommendations', {})
                if icons.get('actions'):
                    action_strs = [f"{a['label']}" for a in icons['actions'][:3]]
                    print(f"  Actions: {', '.join(action_strs)}")
                if icons.get('entities'):
                    entity_strs = [f"{e['label']}" for e in icons['entities'][:3]]
                    print(f"  Entities: {', '.join(entity_strs)}")
                print()
                continue

            if not user_input:
                continue

            # Add one icon
            icon_id = user_input
            result = pipeline.add_icon(icon_id)

            # Display
            seq = result.get('current_sequence', [])
            cs_display = result.get('cs_display', [])
            partial = result.get('partial_translation', '')
            emo = result.get('emotion', {})

            print(f"  {' → '.join(cs_display)}")
            print(f"  Translation: {partial}")
            print(f"  Emotion: {emo.get('single', 'N/A')} -> Next: {emo.get('next', 'N/A')}")

            # Show predictions
            preds = result.get('next_icon_predictions', {})
            _display_incremental_predictions(preds)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def _display_incremental_predictions(preds: Dict):
    """显示增量模式下的icon预测结果"""
    fusion_info = preds.get('fusion_info', {})
    alpha = fusion_info.get('alpha', 0.5)

    if preds.get('actions'):
        action_strs = [f"{a['label']}({a['final_score']:.2f})" for a in preds['actions'][:3]]
        print(f"  Top Actions: {', '.join(action_strs)}")

    if preds.get('entities'):
        entity_strs = [f"{e['label']}({e['final_score']:.2f})" for e in preds['entities'][:3]]
        print(f"  Top Entities: {', '.join(entity_strs)}")

    if preds.get('emotions'):
        emo_strs = [f"{e['label']}" for e in preds['emotions'][:2]]
        print(f"  Emotions: {', '.join(emo_strs)}")

    print(f"  [α={alpha:.1f}: SASRec={alpha:.0%} + RAG={1-alpha:.0%}]")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='AAC Emotion Pipeline')

    # AAC2Text 模型路径
    parser.add_argument('--aac_model_path', type=str,
                        default='./AAC2Text/checkpoints/aac_model')
    parser.add_argument('--aac_base_model_path', type=str,
                        default='/home/user1/liuduanye/Meta-Llama-3-8B-Instruct')

    # EmotionClassify 模型路径
    parser.add_argument('--emotion_model_path', type=str,
                        default='./output/cls_final')
    parser.add_argument('--emotion_base_model_path', type=str,
                        default='./Model/roberta-base')

    # 运行模式
    parser.add_argument('--interactive', action='store_true', help='Interactive mode (batch)')
    parser.add_argument('--incremental', action='store_true', help='Incremental mode (IME-style)')
    parser.add_argument('--symbols', nargs='+', help='AAC symbols to process (batch mode)')
    parser.add_argument('--mode', type=str, choices=['batch', 'incremental'], default='batch',
                        help='Pipeline mode: batch (complete sequence) or incremental (one icon at a time)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_path', type=str, default='./output/incremental_usage.jsonl',
                        help='Path to append real-usage records (jsonl). Set --no_log to disable.')
    parser.add_argument('--no_log', action='store_true', help='Disable usage logging')

    args = parser.parse_args()

    # 确定模式
    mode = 'incremental' if args.incremental else args.mode

    # 初始化Pipeline
    pipeline = AACEmotionPipeline(
        aac_model_path=args.aac_model_path,
        aac_base_model_path=args.aac_base_model_path,
        emotion_model_path=args.emotion_model_path,
        emotion_base_model_path=args.emotion_base_model_path,
        device=args.device,
        mode=mode,
        log_path=None if args.no_log else args.log_path,
    )

    # 增量交互模式
    if args.incremental:
        incremental_mode(pipeline)

    # Batch交互模式
    elif args.interactive:
        interactive_mode(pipeline)

    # 命令行符号输入 (batch only)
    elif args.symbols:
        result = pipeline.process(args.symbols)

        print("\n" + "=" * 60)
        print("Result")
        print("=" * 60)
        print(f"Input Symbols: {result['input']['symbols']}")
        print(f"Translation: {result['translation']['sentence']}")
        print(f"Single Emotion: {result['emotion']['single']} ({result['emotion']['single_confidence']:.0%})")
        print(f"Theme Emotion: {result['emotion']['theme']} ({result['emotion']['theme_confidence']:.0%})")
        print(f"Current State: {result['emotion']['current']}")
        print(f"Next Prediction: {result['prediction']['next_emotion']} ({result['prediction']['confidence']:.0%})")

    # 演示模式
    else:
        print("\n" + "=" * 60)
        print("Demo Mode - Multi-turn Conversation with Icon Prediction")
        print("=" * 60)

        demo_inputs = [
            ["I", "am", "happy"],
            ["I", "want_to", "water"],
            ["I", "feel", "sad"],
            ["I", "love_to", "eat_to", "pizza"],
        ]

        for symbols in demo_inputs:
            result = pipeline.process(symbols)
            turn = result['conversation_turn']
            print(f"\nTurn {turn}: {symbols}")
            print(f"  📝 Translation: {result['translation']['sentence']}")
            print(f"  😊 Single: {result['emotion']['single']} | "
                  f"📍 Current: {result['emotion']['current']} | "
                  f"🔮 Next: {result['prediction']['next_emotion']}")

            icons = result['icon_recommendations']
            rag_info = icons.get('emotional_rag', {})
            actions = [f"{a['label']}(sim:{a['sim_combined']:.2f})" for a in icons.get('actions', [])[:3]]
            entities = [f"{e['label']}(sim:{e['sim_combined']:.2f})" for e in icons.get('entities', [])[:3]]
            print(f"  🎯 Actions: {actions}")
            print(f"  🎯 Entities: {entities}")
            print(f"  📊 RAG: λ={rag_info.get('lambda', 0.3):.1f}, next_emo={rag_info.get('target_emotion', 'N/A')}")

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        summary = pipeline.process_conversation(
            [{'symbols': s} for s in demo_inputs]
        )
        print(f"Total turns: {summary['summary']['total_turns']}")
        print(f"Dominant emotion: {summary['summary']['dominant_emotion']}")
        print(f"Emotion distribution: {summary['summary']['emotion_distribution']}")


if __name__ == '__main__':
    main()
