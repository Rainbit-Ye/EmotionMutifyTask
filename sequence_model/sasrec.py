"""
SASRec: Self-Attentive Sequential Recommendation for AAC Icon Prediction

Architecture:
  Input: [item_emb + cs_role_emb + position_emb] -> CausalSelfAttention x N -> Linear -> logits

Key features:
  - Causal mask ensures autoregressive property
  - CS role embeddings guide slot-filling prediction
  - Lightweight (~2M params) for real-time inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple


# ==================== Model Components ====================

class CausalSelfAttentionBlock(nn.Module):
    """Causal self-attention block with residual connections."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            mask: causal mask [seq_len, seq_len], True = masked positions
        """
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = residual + self.dropout(attn_out)

        # FFN with residual
        residual = x
        x = self.ln2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation for AAC icon prediction.

    Predicts the next icon given a sequence of previously selected icons.
    Incorporates Colourful Semantics (CS) role embeddings for slot-filling guidance.
    """

    def __init__(
        self,
        num_items: int,
        num_cs_roles: int = 7,  # WHO, WHAT_DOING, WHAT, WHERE, WHEN, HOW + PAD
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        cs_role_emb_dim: int = 16,
    ):
        super().__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.cs_role_emb = nn.Embedding(num_cs_roles, cs_role_emb_dim, padding_idx=0)
        self.position_emb = nn.Embedding(max_seq_len, hidden_size)

        # Project CS role embedding to hidden_size if dimensions differ
        self.cs_project = nn.Linear(cs_role_emb_dim, hidden_size, bias=False) if cs_role_emb_dim != hidden_size else None

        # Layer norm for input
        self.input_ln = nn.LayerNorm(hidden_size)

        # Causal self-attention blocks
        self.attention_blocks = nn.ModuleList([
            CausalSelfAttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # Output layer
        self.output_ln = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_items + 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive attention.
        Returns mask where True = position should be ignored.
        Upper triangular = future positions masked out.
        """
        # PyTorch MultiheadAttention attn_mask: True = ignore
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask

    def forward(
        self,
        item_ids: torch.Tensor,
        cs_roles: torch.Tensor,
        return_all_positions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            item_ids: [batch, seq_len] item indices (0 = padding)
            cs_roles: [batch, seq_len] CS role indices (0 = padding)
            return_all_positions: if True, return logits for all positions

        Returns:
            If return_all_positions:
                logits: [batch, seq_len, num_items+1]
            Else:
                logits: [batch, num_items+1] for last non-padding position
        """
        batch_size, seq_len = item_ids.shape
        device = item_ids.device

        # Create attention mask (padding + causal)
        padding_mask = (item_ids == 0)  # [batch, seq_len]

        # Item embedding + CS role embedding + position embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        item_e = self.item_emb(item_ids)
        cs_e = self.cs_role_emb(cs_roles)
        if self.cs_project is not None:
            cs_e = self.cs_project(cs_e)
        pos_e = self.position_emb(positions)

        x = self.input_ln(item_e + cs_e + pos_e)
        x = self.dropout(x)

        # Causal self-attention
        causal_mask = self._generate_causal_mask(seq_len, device)

        for block in self.attention_blocks:
            x = block(x, mask=causal_mask)

        x = self.output_ln(x)

        if return_all_positions:
            return self.output_layer(x)

        # Get the last non-padding position for each sequence
        # Find the last non-zero position
        seq_lengths = (item_ids != 0).sum(dim=1)  # [batch]
        last_positions = seq_lengths - 1  # 0-indexed

        # Gather the hidden states at last positions
        idx = last_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_size)
        last_hidden = x.gather(1, idx).squeeze(1)  # [batch, hidden_size]

        return self.output_layer(last_hidden)  # [batch, num_items+1]

    def predict_next(
        self,
        item_ids: torch.Tensor,
        cs_roles: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k next items.

        Returns:
            top_k_indices: [batch, top_k]
            top_k_probs: [batch, top_k]
        """
        logits = self.forward(item_ids, cs_roles)  # [batch, num_items+1]

        # Mask padding token (index 0)
        logits[:, 0] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)

        return top_k_indices, top_k_probs

    def get_item_probabilities(self, item_ids: torch.Tensor, cs_roles: torch.Tensor) -> torch.Tensor:
        """Get full probability distribution over items."""
        logits = self.forward(item_ids, cs_roles)
        logits[:, 0] = float('-inf')  # mask padding
        return F.softmax(logits, dim=-1)


# ==================== Dataset ====================

# CS role to index mapping
CS_ROLE_TO_ID = {
    'PAD': 0,
    'WHO': 1,
    'WHAT_DOING': 2,
    'WHAT': 3,
    'WHERE': 4,
    'WHEN': 5,
    'HOW': 6,
}
ID_TO_CS_ROLE = {v: k for k, v in CS_ROLE_TO_ID.items()}


class SASRecDataset(Dataset):
    """Dataset for SASRec training on icon sequences."""

    def __init__(
        self,
        sequences: List[Dict],
        item2idx: Dict[str, int],
        max_seq_len: int = 50,
        is_training: bool = True,
    ):
        self.sequences = sequences
        self.item2idx = item2idx
        self.max_seq_len = max_seq_len
        self.is_training = is_training

        # Pre-process sequences
        self.processed = []
        for seq in sequences:
            icons = seq['sequence']
            cs_roles = seq.get('cs_roles', [])

            # Convert to indices
            item_ids = [item2idx.get(icon, 0) for icon in icons]
            cs_ids = [CS_ROLE_TO_ID.get(role, 0) for role in cs_roles]

            # If CS roles missing, fill with defaults
            if len(cs_ids) < len(item_ids):
                cs_ids.extend([0] * (len(item_ids) - len(cs_ids)))

            # Pad/truncate
            item_ids = item_ids[:max_seq_len]
            cs_ids = cs_ids[:max_seq_len]

            if len(item_ids) < 2:
                continue  # Need at least 2 items for next-item prediction

            self.processed.append({
                'item_ids': item_ids,
                'cs_ids': cs_ids,
                'emotion': seq.get('emotion', 'neutral'),
            })

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        item = self.processed[idx]
        item_ids = item['item_ids']
        cs_ids = item['cs_ids']
        seq_len = len(item_ids)

        # For training: input = item_ids[:-1], target = item_ids[1:]
        # This creates the next-item prediction task
        input_ids = item_ids[:-1]
        input_cs = cs_ids[:-1]
        target_ids = item_ids[1:]
        target_cs = cs_ids[1:]

        # Padding
        pad_len = self.max_seq_len - 1 - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        input_cs = input_cs + [0] * pad_len
        target_ids = target_ids + [0] * pad_len

        return {
            'item_ids': torch.tensor(input_ids, dtype=torch.long),
            'cs_roles': torch.tensor(input_cs, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'seq_lengths': torch.tensor(len(item_ids) - 1, dtype=torch.long),
        }


def build_item_vocabulary(ontology: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build item vocabulary from ontology.

    Returns:
        item2idx: icon_id -> index (1-based, 0 = padding)
        idx2item: index -> icon_id
    """
    item2idx = {'PAD': 0}
    idx2item = {0: 'PAD'}
    idx = 1

    for icon_id in sorted(ontology.keys()):
        item2idx[icon_id] = idx
        idx2item[idx] = icon_id
        idx += 1

    return item2idx, idx2item


def compute_metrics(
    model: SASRec,
    dataloader: DataLoader,
    idx2item: Dict[int, str],
    device: torch.device,
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute Hit@K, MRR, NDCG@K metrics."""
    model.eval()
    hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = {k: 0.0 for k in ks}
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            item_ids = batch['item_ids'].to(device)
            cs_roles = batch['cs_roles'].to(device)
            target_ids = batch['target_ids'].to(device)
            seq_lengths = batch['seq_lengths']

            # Get all position logits
            logits = model(item_ids, cs_roles, return_all_positions=True)
            probs = F.softmax(logits, dim=-1)

            # For each sequence, evaluate at the last non-padding position
            for i in range(len(seq_lengths)):
                last_pos = seq_lengths[i].item() - 1
                if last_pos < 0:
                    continue

                target = target_ids[i, last_pos].item()
                if target == 0:
                    continue

                item_probs = probs[i, last_pos]  # [vocab_size]
                item_probs[0] = 0  # mask padding

                # Get ranking
                sorted_indices = torch.argsort(item_probs, descending=True)
                rank = (sorted_indices == target).nonzero(as_tuple=True)[0]
                if len(rank) == 0:
                    continue
                rank = rank[0].item() + 1  # 1-based

                # Hit@K
                for k in ks:
                    if rank <= k:
                        hits[k] += 1

                # MRR
                mrr_sum += 1.0 / rank

                # NDCG@K
                for k in ks:
                    if rank <= k:
                        ndcg_sum[k] += 1.0 / math.log2(rank + 1)

                total += 1

    metrics = {}
    for k in ks:
        metrics[f'hit@{k}'] = hits[k] / max(total, 1)
        metrics[f'ndcg@{k}'] = ndcg_sum[k] / max(total, 1)
    metrics['mrr'] = mrr_sum / max(total, 1)
    metrics['total'] = total

    return metrics
