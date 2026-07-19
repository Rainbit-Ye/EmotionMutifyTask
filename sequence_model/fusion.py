"""
FusedIconPredictor: Combines SASRec sequential prediction with Emotional RAG.

Fusion formula:
  Final(i) = alpha * P_sasrec(i|seq) + (1-alpha) * [lambda*cos(Q_emo,i) + (1-lambda)*cos(Q_orig,i)]

Where:
  - P_sasrec: SASRec softmax probability for next icon
  - cos(Q_emo, i): cosine similarity with emotion-augmented query
  - cos(Q_orig, i): cosine similarity with original query
  - alpha: fusion weight (0.5 default)
  - lambda: Emotional RAG balance (0.3 default)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from sequence_model.sasrec import SASRec, CS_ROLE_TO_ID
# Import from parent module
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FusedIconPredictor:
    """
    Fuses SASRec sequential prediction with Emotional RAG similarity.

    SASRec provides P(next_icon | icon_sequence) -- sequential context.
    Emotional RAG provides emotion-guided semantic similarity.
    Both are normalized to [0,1] and weighted by alpha.
    """

    def __init__(
        self,
        sasrec_model: SASRec,
        icon_predictor,  # AACIconPredictor from aac_emotion_pipeline.py
        item2idx: Dict[str, int],
        idx2item: Dict[int, str],
        alpha: float = 0.5,
        lambda_balance: float = 0.3,
        device: str = 'cuda',
    ):
        self.sasrec = sasrec_model
        self.rag = icon_predictor
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.alpha = alpha
        self.lambda_balance = lambda_balance
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def predict_next(
        self,
        current_sequence: List[str],
        current_cs_roles: List[str],
        current_emotion: str = 'neutral',
        next_emotion: str = None,
        current_sentence: str = '',
        used_symbols: List[str] = None,
        top_k: int = 10,
        conversation_context: List[str] = None,
    ) -> Dict:
        """
        Predict next icons using SASRec + Emotional RAG fusion.

        Args:
            current_sequence: List of icon_ids in the current sequence
            current_cs_roles: CS roles for each icon
            current_emotion: Current detected emotion
            next_emotion: Predicted next emotion (for RAG)
            current_sentence: Current partial/full translation
            used_symbols: Icons already used (to exclude)
            top_k: Number of top predictions to return
            conversation_context: Previous sentences for RAG context

        Returns:
            Dict with predictions and metadata
        """
        # 1. SASRec prediction
        sasrec_scores = self._get_sasrec_scores(current_sequence, current_cs_roles)

        # 2. Emotional RAG prediction
        rag_scores = self._get_rag_scores(
            current_emotion=current_emotion,
            next_emotion=next_emotion,
            current_sentence=current_sentence,
            used_symbols=used_symbols,
            conversation_context=conversation_context or [],
        )

        # 3. Normalize both to [0, 1]
        sasrec_norm = self._normalize_scores(sasrec_scores)
        rag_norm = self._normalize_scores(rag_scores)

        # 4. Fuse
        all_icons = set(sasrec_norm.keys()) | set(rag_norm.keys())
        fused_scores = {}
        for icon_id in all_icons:
            s_score = sasrec_norm.get(icon_id, 0.0)
            r_score = rag_norm.get(icon_id, 0.0)
            fused_scores[icon_id] = self.alpha * s_score + (1 - self.alpha) * r_score

        # 5. Sort and categorize
        sorted_icons = sorted(fused_scores.items(), key=lambda x: -x[1])

        # Categorize by semantic type
        actions = []
        entities = []
        emotions = []
        others = []

        used_set = set(used_symbols) if used_symbols else set()

        for icon_id, score in sorted_icons:
            if icon_id in used_set:
                continue

            icon_info = self.rag.ontology.get(icon_id, {})
            semantic_type = icon_info.get('semantic_type', 'unknown')
            label = icon_info.get('label', icon_id)

            item = {
                'icon_id': icon_id,
                'label': label,
                'semantic_type': semantic_type,
                'cs_role': icon_info.get('cs_role', 'WHAT'),
                'final_score': round(score, 4),
                'sasrec_score': round(sasrec_norm.get(icon_id, 0.0), 4),
                'rag_score': round(rag_norm.get(icon_id, 0.0), 4),
            }

            if semantic_type == 'action' and len(actions) < 5:
                actions.append(item)
            elif semantic_type in ['entity', 'object', 'noun', 'person', 'food', 'drink'] and len(entities) < 5:
                entities.append(item)
            elif semantic_type == 'emotion' and len(emotions) < 3:
                emotions.append(item)
            elif len(others) < 3:
                others.append(item)

            if len(actions) >= 5 and len(entities) >= 5 and len(emotions) >= 3 and len(others) >= 3:
                break

        return {
            'actions': actions,
            'entities': entities,
            'emotions': emotions,
            'others': others,
            'fusion_info': {
                'alpha': self.alpha,
                'lambda': self.lambda_balance,
                'current_emotion': current_emotion,
                'next_emotion': next_emotion,
                'sequence_length': len(current_sequence),
            }
        }

    def _get_sasrec_scores(self, sequence: List[str], cs_roles: List[str]) -> Dict[str, float]:
        """Get SASRec prediction scores for all items."""
        if not sequence:
            return {}

        self.sasrec.eval()
        with torch.no_grad():
            # Convert to indices
            item_ids = [self.item2idx.get(icon, 0) for icon in sequence]
            cs_ids = [CS_ROLE_TO_ID.get(role, 0) for role in cs_roles]

            # Sequence has no in-vocab items (all OOV -> mapped to padding 0).
            # SASRec cannot score it; skip to avoid garbage predictions / index errors.
            if not any(item_ids):
                return {}

            # Pad to max_seq_len
            max_len = self.sasrec.max_seq_len
            if len(item_ids) > max_len:
                item_ids = item_ids[-max_len:]
                cs_ids = cs_ids[-max_len:]

            # Create tensors
            item_tensor = torch.tensor([item_ids], dtype=torch.long).to(self.device)
            cs_tensor = torch.tensor([cs_ids], dtype=torch.long).to(self.device)

            # Forward
            logits = self.sasrec(item_tensor, cs_tensor)  # [1, num_items+1]
            probs = F.softmax(logits, dim=-1)[0]  # [num_items+1]

        # Convert to icon_id -> score dict
        scores = {}
        for idx in range(1, len(self.idx2item)):  # skip padding
            icon_id = self.idx2item.get(idx, '')
            if icon_id and icon_id != 'PAD':
                scores[icon_id] = probs[idx].item()

        return scores

    def _get_rag_scores(
        self,
        current_emotion: str,
        next_emotion: str = None,
        current_sentence: str = '',
        used_symbols: List[str] = None,
        conversation_context: List[str] = None,
    ) -> Dict[str, float]:
        """Get Emotional RAG scores for all items."""
        if not self.rag or not self.rag.embedding_model:
            return {}

        # Use the existing RAG predictor
        result = self.rag.predict_next_icons_by_context(
            conversation_context=conversation_context or [],
            current_emotion=current_emotion,
            next_emotion=next_emotion,
            used_symbols=used_symbols or [],
            current_sentence=current_sentence,
            top_k=len(self.rag.icon_list),  # get all scores
        )

        # Extract scores from result
        scores = {}
        for category in ['actions', 'entities', 'emotions', 'others']:
            for item in result.get(category, []):
                scores[item['icon_id']] = item.get('final_score', 0.0)

        # If RAG didn't return all icons, fill with computed similarity
        if len(scores) < len(self.rag.icon_list):
            # Use the raw combined similarity from RAG
            from sentence_transformers import util
            target_emotion = next_emotion if next_emotion else current_emotion
            emotion_config = self.rag._get_emotion_rag_config(target_emotion, current_emotion)

            Q_orig = current_sentence
            Q_emo = Q_orig
            emotion_prompts = emotion_config.get('emotion_prompts', [])
            if emotion_prompts:
                Q_emo = Q_emo + " " + " ".join(emotion_prompts[:2])

            if Q_orig or Q_emo:
                E_orig = self.rag.embedding_model.encode(Q_orig, convert_to_tensor=True, show_progress_bar=False) if Q_orig else None
                E_emo = self.rag.embedding_model.encode(Q_emo, convert_to_tensor=True, show_progress_bar=False) if Q_emo else None

                if E_orig is not None and E_emo is not None:
                    sim_orig = util.cos_sim(E_orig, self.rag.icon_embeddings)[0]
                    sim_emo = util.cos_sim(E_emo, self.rag.icon_embeddings)[0]
                    combined = self.lambda_balance * sim_emo + (1 - self.lambda_balance) * sim_orig

                    for idx, icon_info in enumerate(self.rag.icon_list):
                        icon_id = icon_info['icon_id']
                        if icon_id not in scores:
                            scores[icon_id] = combined[idx].item()

        return scores

    @staticmethod
    def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return {}
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
