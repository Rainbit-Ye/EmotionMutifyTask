#!/usr/bin/env python3
"""
Train SASRec model on icon sequences.

Usage:
    python train_sasrec.py
    python train_sasrec.py --epochs 30 --hidden-size 128
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Dict, List
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence_model.sasrec import (
    SASRec, SASRecDataset, CS_ROLE_TO_ID,
    build_item_vocabulary, compute_metrics
)


def load_ontology(ontology_path: str) -> Dict:
    """Load ontology for vocabulary building."""
    with open(ontology_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ontology = {}
    for item in data['ontology']:
        icon_id = item.get('icon_id', '')
        if icon_id:
            ontology[icon_id] = item

    return ontology


def load_sequences(path: str) -> List[Dict]:
    """Load icon sequences."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def train_epoch(
    model: SASRec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        item_ids = batch['item_ids'].to(device)
        cs_roles = batch['cs_roles'].to(device)
        target_ids = batch['target_ids'].to(device)
        seq_lengths = batch['seq_lengths']

        # Forward: get all positions
        logits = model(item_ids, cs_roles, return_all_positions=True)

        # Compute loss only at non-padding positions
        loss = torch.tensor(0.0, device=device)
        correct = 0
        samples = 0

        for i in range(len(seq_lengths)):
            last_pos = seq_lengths[i].item() - 1
            if last_pos < 0:
                continue

            target = target_ids[i, last_pos]
            if target == 0:
                continue

            pred = logits[i, last_pos]  # [vocab_size]
            loss += F.cross_entropy(pred.unsqueeze(0), target.unsqueeze(0))

            if pred.argmax().item() == target.item():
                correct += 1
            samples += 1

        if samples > 0:
            loss = loss / samples
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * samples
            total_correct += correct
            total_samples += samples

    return {
        'loss': total_loss / max(total_samples, 1),
        'accuracy': total_correct / max(total_samples, 1),
    }


def evaluate(
    model: SASRec,
    dataloader: DataLoader,
    idx2item: Dict[int, str],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    return compute_metrics(model, dataloader, idx2item, device)


def main():
    parser = argparse.ArgumentParser(description='Train SASRec')
    parser.add_argument('--ontology', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json')
    parser.add_argument('--train-data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/icon_sequences_train.json')
    parser.add_argument('--val-data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/icon_sequences_val.json')
    parser.add_argument('--test-data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/icon_sequences_test.json')
    parser.add_argument('--output-dir', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/output/sasrec')
    # Model hyperparameters
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--max-seq-len', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--cs-role-emb-dim', type=int, default=16)
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=10)
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    ontology = load_ontology(args.ontology)
    item2idx, idx2item = build_item_vocabulary(ontology)
    num_items = len(item2idx) - 1  # exclude padding
    print(f"Vocabulary size: {num_items + 1} (including padding)")

    train_sequences = load_sequences(args.train_data)
    val_sequences = load_sequences(args.val_data)
    test_sequences = load_sequences(args.test_data)
    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")

    # Create datasets
    train_dataset = SASRecDataset(train_sequences, item2idx, args.max_seq_len, is_training=True)
    val_dataset = SASRecDataset(val_sequences, item2idx, args.max_seq_len, is_training=False)
    test_dataset = SASRecDataset(test_sequences, item2idx, args.max_seq_len, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Create model
    model = SASRec(
        num_items=num_items,
        num_cs_roles=len(CS_ROLE_TO_ID),
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        cs_role_emb_dim=args.cs_role_emb_dim,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_mrr = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        train_result = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, idx2item, device)

        # Print
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | "
              f"Loss: {train_result['loss']:.4f} | "
              f"Acc: {train_result['accuracy']:.4f} | "
              f"Val Hit@1: {val_metrics['hit@1']:.4f} | "
              f"Val Hit@5: {val_metrics['hit@5']:.4f} | "
              f"Val MRR: {val_metrics['mrr']:.4f} | "
              f"Val NDCG@5: {val_metrics['ndcg@5']:.4f} | "
              f"LR: {lr:.6f}")

        # Save best model
        if val_metrics['mrr'] > best_val_mrr:
            best_val_mrr = val_metrics['mrr']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'item2idx': item2idx,
                'idx2item': idx2item,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (MRR: {best_val_mrr:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, idx2item, device)

    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nTraining complete. Best Val MRR: {best_val_mrr:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
