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

    def __init__(self, ontology_path: str = None, embedding_model: str = '/home/user1/liuduanye/EmotionClassify/Model/all-MiniLM-L6-v2'):
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
        """初始化嵌入模型并预计算图标嵌入"""
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
    def __init__(self, model_path, base_model_path, device='cuda'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"[AAC2Text] Loading base model: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # 加载LoRA权重
        if os.path.exists(model_path):
            print(f"[AAC2Text] Loading LoRA weights: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)

        self.model.eval()
        print("[AAC2Text] Model loaded successfully")

    def translate(self, symbols: List[str]) -> str:
        """将AAC符号列表翻译为自然语言句子"""
        prompt = f"Translate these AAC symbols to a sentence: {' '.join(symbols)}"
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                stop_strings=["<|im_end|>", "\n"],
                tokenizer=self.tokenizer,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        response = response.strip().split('\n')[0].strip()
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

    def get_trend(self) -> Dict:
        """获取情感趋势"""
        if len(self.history) < 2:
            return {'trend': 'stable', 'direction': 0.0}

        valences = [EMOTION_VALENCE.get(e, 0.0) for e in self.history]
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


# ==================== 完整Pipeline ====================

class AACEmotionPipeline:
    """AAC完整交流Pipeline：翻译 + 情感分析 + 预测 + 图标推荐"""

    def __init__(self,
                 aac_model_path: str,
                 aac_base_model_path: str,
                 emotion_model_path: str,
                 emotion_base_model_path: str,
                 ontology_path: str = None,
                 device: str = 'cuda'):

        print("=" * 60)
        print("Initializing AAC Emotion Pipeline")
        print("=" * 60)

        # 加载翻译器
        self.translator = AACTranslator(aac_model_path, aac_base_model_path, device)

        # 加载情感分析器
        self.analyzer = EmotionAnalyzer(emotion_model_path, emotion_base_model_path, device)

        # 加载图标预测器
        if ontology_path is None:
            ontology_path = "/home/user1/liuduanye/AgentPipeline/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json"
        self.icon_predictor = AACIconPredictor(ontology_path)

        # 对话历史
        self.conversation_history = []

        print("\nPipeline initialized successfully!")
        print("=" * 60)

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
    """交互式模式"""
    print("\n" + "=" * 60)
    print("AAC Emotion Pipeline - Interactive Mode")
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


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='AAC Emotion Pipeline')

    # AAC2Text 模型路径
    parser.add_argument('--aac_model_path', type=str,
                        default='/home/user1/liuduanye/AgentPipeline/EmotionClassify/AAC2Text/checkpoints/aac_model')
    parser.add_argument('--aac_base_model_path', type=str,
                        default='/home/user1/liuduanye/AgentPipeline/qwen/Qwen2_5-1_5B-Instruct')

    # EmotionClassify 模型路径
    parser.add_argument('--emotion_model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/cls_final')
    parser.add_argument('--emotion_base_model_path', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/Model/roberta-base')

    # 运行模式
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--symbols', nargs='+', help='AAC symbols to process')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # 初始化Pipeline
    pipeline = AACEmotionPipeline(
        aac_model_path=args.aac_model_path,
        aac_base_model_path=args.aac_base_model_path,
        emotion_model_path=args.emotion_model_path,
        emotion_base_model_path=args.emotion_base_model_path,
        device=args.device
    )

    # 交互模式
    if args.interactive:
        interactive_mode(pipeline)

    # 命令行符号输入
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
