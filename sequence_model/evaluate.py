#!/usr/bin/env python3
"""
Evaluation metrics for SASRec and fused icon prediction.

Metrics:
  - Accuracy@K (Hit@K): Is the correct next icon in the top-K predictions?
  - MRR (Mean Reciprocal Rank)
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - CS Role Accuracy: Does the predicted icon have the correct CS role?
"""

import math
from typing import List, Dict, Tuple


def accuracy_at_k(ranked_items: List[str], target: str, k: int) -> float:
    """Hit@K: 1.0 if target is in top-K, else 0.0."""
    return 1.0 if target in ranked_items[:k] else 0.0


def reciprocal_rank(ranked_items: List[str], target: str) -> float:
    """MRR: 1/rank of target item."""
    try:
        rank = ranked_items.index(target) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(ranked_items: List[str], target: str, k: int) -> float:
    """NDCG@K: Normalized discounted cumulative gain (binary relevance)."""
    try:
        rank = ranked_items.index(target) + 1
    except ValueError:
        return 0.0

    if rank > k:
        return 0.0

    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(2)  # ideal rank = 1
    return dcg / idcg


def cs_role_accuracy(ranked_items: List[str], target: str,
                     ontology: Dict, k: int = 5) -> float:
    """CS Role Accuracy: Does any top-K predicted icon have the same CS role as the target?"""
    target_cs = ontology.get(target, {}).get('cs_role', 'UNKNOWN')

    for item_id in ranked_items[:k]:
        item_cs = ontology.get(item_id, {}).get('cs_role', 'UNKNOWN')
        if item_cs == target_cs:
            return 1.0

    return 0.0


def evaluate_predictions(
    predictions: List[Dict],
    ontology: Dict,
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Evaluate a list of predictions.

    Each prediction dict should have:
      - 'ranked_items': List[str] of ranked icon_ids
      - 'target': str of the correct next icon_id

    Returns dict of metric_name -> average value.
    """
    metrics = {}

    for k in ks:
        metrics[f'hit@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0
    metrics['mrr'] = 0.0
    metrics['cs_role_acc@5'] = 0.0

    total = 0

    for pred in predictions:
        ranked = pred.get('ranked_items', [])
        target = pred.get('target', '')

        if not target or not ranked:
            continue

        for k in ks:
            metrics[f'hit@{k}'] += accuracy_at_k(ranked, target, k)
            metrics[f'ndcg@{k}'] += ndcg_at_k(ranked, target, k)

        metrics['mrr'] += reciprocal_rank(ranked, target)
        metrics['cs_role_acc@5'] += cs_role_accuracy(ranked, target, ontology, k=5)

        total += 1

    if total > 0:
        for key in metrics:
            metrics[key] /= total

    metrics['total'] = total
    return metrics


def compare_modes(
    batch_predictions: List[Dict],
    incremental_predictions: List[Dict],
    ontology: Dict,
    ks: List[int] = [1, 3, 5, 10],
) -> Dict:
    """Compare batch vs incremental prediction modes."""
    batch_metrics = evaluate_predictions(batch_predictions, ontology, ks)
    inc_metrics = evaluate_predictions(incremental_predictions, ontology, ks)

    comparison = {
        'batch': batch_metrics,
        'incremental': inc_metrics,
        'diff': {},
    }

    for key in batch_metrics:
        if key == 'total':
            continue
        comparison['diff'][key] = inc_metrics.get(key, 0.0) - batch_metrics.get(key, 0.0)

    return comparison
