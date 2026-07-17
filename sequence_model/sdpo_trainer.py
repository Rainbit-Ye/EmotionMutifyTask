#!/usr/bin/env python3
"""
S-DPO (Simultaneous DPO) trainer for SASRec.

S-DPO extends standard DPO to handle multiple rejected items simultaneously.
Instead of a single rejected item, K-1 unchosen candidates are averaged.

Loss:
  L = -E[log sigma(beta * (log pi(chosen|seq)/pi_ref(chosen|seq)
                          - mean_j log pi(rejected_j|seq)/pi_ref(rejected_j|seq)))]

Reference: Hu et al., "Aligning Large Language Models via S-DPO", NeurIPS 2024
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence_model.sasrec import SASRec, CS_ROLE_TO_ID, build_item_vocabulary


class SDPOLoss(nn.Module):
    """Simultaneous DPO loss for multi-negative preference data."""

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        rejected_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute S-DPO loss.

        Args:
            policy_chosen_logps: [batch] log pi_policy(chosen|seq)
            policy_rejected_logps: [batch, K-1] log pi_policy(rejected_j|seq)
            ref_chosen_logps: [batch] log pi_ref(chosen|seq)
            ref_rejected_logps: [batch, K-1] log pi_ref(rejected_j|seq)

        Returns:
            Scalar loss
        """
        # Log-ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps  # [batch]
        rejected_logratios = policy_rejected_logps - ref_rejected_logps  # [batch, K-1]

        # Average over rejected items. Mask-aware: padding rows (mask=0) must
        # NOT dilute the average (otherwise rejected signal collapses to ~0).
        if rejected_mask is not None:
            m = rejected_mask.float()
            denom = m.sum(dim=-1).clamp(min=1.0)
            avg_rejected_logratio = (rejected_logratios * m).sum(dim=-1) / denom  # [batch]
        else:
            avg_rejected_logratio = rejected_logratios.mean(dim=-1)  # [batch]

        # S-DPO loss
        loss = -F.logsigmoid(
            self.beta * (chosen_logratios - avg_rejected_logratio)
        )

        return loss.mean()


class SDPODataset(Dataset):
    """Dataset for S-DPO training."""

    def __init__(
        self,
        preference_data: List[Dict],
        item2idx: Dict[str, int],
        max_seq_len: int = 50,
    ):
        self.item2idx = item2idx
        self.max_seq_len = max_seq_len
        self.data = []

        for pref in preference_data:
            prompt = pref['prompt']
            chosen = pref['chosen']
            rejected = pref.get('rejected', [])

            # Convert sequence to indices
            seq_icons = prompt.get('sequence', [])
            seq_cs = prompt.get('cs_roles', [])

            seq_item_ids = [item2idx.get(i, 0) for i in seq_icons[-max_seq_len:]]
            seq_cs_ids = [CS_ROLE_TO_ID.get(r, 0) for r in seq_cs[-max_seq_len:]]

            chosen_idx = item2idx.get(chosen, 0)
            rejected_idxs = [item2idx.get(r, 0) for r in rejected if r in item2idx]

            if chosen_idx == 0 or len(rejected_idxs) == 0:
                continue

            self.data.append({
                'seq_item_ids': seq_item_ids,
                'seq_cs_ids': seq_cs_ids,
                'chosen_idx': chosen_idx,
                'rejected_idxs': rejected_idxs,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Pad sequence
        seq_len = len(item['seq_item_ids'])
        pad_len = max(0, self.max_seq_len - seq_len)
        seq_item_ids = item['seq_item_ids'] + [0] * pad_len
        seq_cs_ids = item['seq_cs_ids'] + [0] * pad_len

        # Pad rejected to max K-1
        rejected = item['rejected_idxs']
        max_neg = max(len(d['rejected_idxs']) for d in self.data[:1000]) if self.data else 4
        rejected_padded = rejected + [0] * max(0, max_neg - len(rejected))
        rejected_mask = [1] * len(rejected) + [0] * max(0, max_neg - len(rejected))

        return {
            'seq_item_ids': torch.tensor(seq_item_ids, dtype=torch.long),
            'seq_cs_ids': torch.tensor(seq_cs_ids, dtype=torch.long),
            'seq_lengths': torch.tensor(min(seq_len, self.max_seq_len), dtype=torch.long),
            'chosen_idx': torch.tensor(item['chosen_idx'], dtype=torch.long),
            'rejected_idxs': torch.tensor(rejected_padded[:max_neg], dtype=torch.long),
            'rejected_mask': torch.tensor(rejected_mask[:max_neg], dtype=torch.float),
        }


def get_log_probability(model: SASRec, item_ids: torch.Tensor, cs_ids: torch.Tensor,
                         target_idx: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """Get log probability of target item given sequence."""
    logits = model(item_ids, cs_ids, return_all_positions=True)  # [batch, seq_len, vocab]

    # Get logits at last position
    batch_size = logits.shape[0]
    last_positions = (seq_lengths - 1).clamp(min=0)
    last_logits = logits[torch.arange(batch_size), last_positions]  # [batch, vocab]

    log_probs = F.log_softmax(last_logits, dim=-1)  # [batch, vocab]

    # Gather target log probs
    target_logps = log_probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)  # [batch]

    return target_logps


def train_sdpo(
    policy_model: SASRec,
    ref_model: SASRec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
):
    """Train SASRec with S-DPO."""
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sdpo_loss_fn = SDPOLoss(beta=args.beta)

    best_val_loss = float('inf')
    patience_counter = 0
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        # Train
        policy_model.train()
        ref_model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            seq_item_ids = batch['seq_item_ids'].to(device)
            seq_cs_ids = batch['seq_cs_ids'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            chosen_idx = batch['chosen_idx'].to(device)
            rejected_idxs = batch['rejected_idxs'].to(device)
            rejected_mask = batch['rejected_mask'].to(device)

            # Policy model log probs
            with torch.enable_grad():
                policy_chosen_logps = get_log_probability(
                    policy_model, seq_item_ids, seq_cs_ids, chosen_idx, seq_lengths
                )

                # For rejected items (multiple)
                batch_size = rejected_idxs.shape[0]
                num_neg = rejected_idxs.shape[1]

                policy_rejected_logps = []
                for j in range(num_neg):
                    rej_idx = rejected_idxs[:, j]
                    mask = rejected_mask[:, j]
                    logp = get_log_probability(
                        policy_model, seq_item_ids, seq_cs_ids, rej_idx, seq_lengths
                    )
                    policy_rejected_logps.append(logp * mask)
                policy_rejected_logps = torch.stack(policy_rejected_logps, dim=1)  # [batch, K-1]

            # Reference model log probs (no grad)
            with torch.no_grad():
                ref_chosen_logps = get_log_probability(
                    ref_model, seq_item_ids, seq_cs_ids, chosen_idx, seq_lengths
                )
                ref_rejected_logps = []
                for j in range(num_neg):
                    rej_idx = rejected_idxs[:, j]
                    mask = rejected_mask[:, j]
                    logp = get_log_probability(
                        ref_model, seq_item_ids, seq_cs_ids, rej_idx, seq_lengths
                    )
                    ref_rejected_logps.append(logp * mask)
                ref_rejected_logps = torch.stack(ref_rejected_logps, dim=1)

            # S-DPO loss
            loss = sdpo_loss_fn(policy_chosen_logps, policy_rejected_logps,
                               ref_chosen_logps, ref_rejected_logps)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            if num_batches % 100 == 0:
                pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        avg_loss = total_loss / max(num_batches, 1)

        # Validate
        policy_model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_item_ids = batch['seq_item_ids'].to(device)
                seq_cs_ids = batch['seq_cs_ids'].to(device)
                seq_lengths = batch['seq_lengths'].to(device)
                chosen_idx = batch['chosen_idx'].to(device)
                rejected_idxs = batch['rejected_idxs'].to(device)
                rejected_mask = batch['rejected_mask'].to(device)

                policy_chosen_logps = get_log_probability(
                    policy_model, seq_item_ids, seq_cs_ids, chosen_idx, seq_lengths
                )
                ref_chosen_logps = get_log_probability(
                    ref_model, seq_item_ids, seq_cs_ids, chosen_idx, seq_lengths
                )

                num_neg = rejected_idxs.shape[1]
                policy_rejected_logps = []
                ref_rejected_logps = []
                for j in range(num_neg):
                    rej_idx = rejected_idxs[:, j]
                    mask = rejected_mask[:, j]
                    p_logp = get_log_probability(
                        policy_model, seq_item_ids, seq_cs_ids, rej_idx, seq_lengths
                    ) * mask
                    r_logp = get_log_probability(
                        ref_model, seq_item_ids, seq_cs_ids, rej_idx, seq_lengths
                    ) * mask
                    policy_rejected_logps.append(p_logp)
                    ref_rejected_logps.append(r_logp)
                policy_rejected_logps = torch.stack(policy_rejected_logps, dim=1)
                ref_rejected_logps = torch.stack(ref_rejected_logps, dim=1)

                loss = sdpo_loss_fn(policy_chosen_logps, policy_rejected_logps,
                                   ref_chosen_logps, ref_rejected_logps)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_model.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(output_dir, 'best_sdpo_model.pt'))
            print(f"  -> Saved best S-DPO model (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nS-DPO training complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train SASRec with S-DPO')
    parser.add_argument('--sasrec-model', type=str,
                        default='./output/sasrec/best_model.pt')
    parser.add_argument('--dpo-train', type=str,
                        default='./data/sasrec_dpo_train.json')
    parser.add_argument('--dpo-val', type=str,
                        default='./data/sasrec_dpo_val.json')
    parser.add_argument('--output-dir', type=str,
                        default='./output/sasrec_dpo')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load trained SASRec as reference model
    print(f"Loading reference SASRec from: {args.sasrec_model}")
    checkpoint = torch.load(args.sasrec_model, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})

    # Load ontology for vocab
    ontology_path = saved_args.get('ontology', './AAC2Text/data/processed/aac_full_ontology.json')
    with open(ontology_path, 'r') as f:
        ont_data = json.load(f)
    ontology = {}
    for item in ont_data['ontology']:
        if item.get('icon_id'):
            ontology[item['icon_id']] = item

    item2idx, idx2item = build_item_vocabulary(ontology)
    num_items = len(item2idx) - 1

    # Create reference model (frozen)
    ref_model = SASRec(
        num_items=num_items,
        num_cs_roles=len(CS_ROLE_TO_ID),
        hidden_size=saved_args.get('hidden_size', 64),
        num_heads=saved_args.get('num_heads', 2),
        num_blocks=saved_args.get('num_blocks', 2),
        max_seq_len=saved_args.get('max_seq_len', 50),
        dropout=0.0,
        cs_role_emb_dim=saved_args.get('cs_role_emb_dim', 16),
    ).to(device)
    ref_model.load_state_dict(checkpoint['model_state_dict'])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Create policy model (from reference, will be trained)
    policy_model = SASRec(
        num_items=num_items,
        num_cs_roles=len(CS_ROLE_TO_ID),
        hidden_size=saved_args.get('hidden_size', 64),
        num_heads=saved_args.get('num_heads', 2),
        num_blocks=saved_args.get('num_blocks', 2),
        max_seq_len=saved_args.get('max_seq_len', 50),
        dropout=0.1,
        cs_role_emb_dim=saved_args.get('cs_role_emb_dim', 16),
    ).to(device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])

    # Load preference data
    print(f"Loading preference data...")
    with open(args.dpo_train, 'r') as f:
        train_prefs = json.load(f)
    with open(args.dpo_val, 'r') as f:
        val_prefs = json.load(f)

    train_dataset = SDPODataset(train_prefs, item2idx, max_seq_len=saved_args.get('max_seq_len', 50))
    val_dataset = SDPODataset(val_prefs, item2idx, max_seq_len=saved_args.get('max_seq_len', 50))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Train
    train_sdpo(policy_model, ref_model, train_loader, val_loader, args, device)


if __name__ == '__main__':
    main()
