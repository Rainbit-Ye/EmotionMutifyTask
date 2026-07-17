#!/usr/bin/env python3
"""
Train a SASRec next-icon predictor on review5000_combined_full.jsonl.

Dual-track (matches "Route A"):
  Track 1 (CE / next-item):  train SASRec with CrossEntropy on POSITIVE samples
                                (sequence transitions + human_good) -> base model.
  Track 2 (S-DP-O / preference): align the base model with human good/bad labels,
                                using each prefix's pos `next` as chosen and its
                                human_bad + random_neg `next` as (multiple) rejected.

Data format (review5000_combined_full.jsonl), one JSON per line:
    {"id":int, "prefix":[icon_id,...], "next":icon_id, "label":"pos"|"neg",
     "src":"sequence"|"human_good"|"human_bad"|"random_neg", "annotator"?:str}

No `cs_roles` in the data -> derived per icon from the ontology's `cs_role` field.

Usage (run from EmotionClassify/):
    python sequence_model/train_review5000.py \
        --data /home/user1/liuduanye/review5000_combined_full.jsonl \
        --ontology /home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json \
        --output-dir /home/user1/liuduanye/EmotionClassify/output/review5000 \
        --hidden-size 128 --num-blocks 3 --num-heads 4

To skip Track 1 and use an existing CE checkpoint as the reference:
    --skip-ce --ce-checkpoint /path/to/best_model.pt
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence_model.sasrec import (
    SASRec, SASRecDataset, CS_ROLE_TO_ID, compute_metrics,
)
from sequence_model.sdpo_trainer import SDPOLoss, get_log_probability


# ============================================================
# Vocabulary & CS-role helpers
# ============================================================

def build_vocab_from_data(lines: List[dict]) -> (Dict[str, int], Dict[int, str]):
    """1-based vocab over every icon appearing in prefix/next. 0 = PAD."""
    icons = set()
    for d in lines:
        icons.update(d["prefix"])
        icons.add(d["next"])
    item2idx = {"PAD": 0}
    idx2item = {0: "PAD"}
    for ic in sorted(icons):
        item2idx[ic] = len(item2idx)
        idx2item[item2idx[ic]] = ic
    return item2idx, idx2item


def build_icon2cs(ontology_path: str) -> Dict[str, int]:
    """icon_id -> CS role INDEX, from ontology `cs_role` field (fallback WHAT=3)."""
    icon2cs = {}
    try:
        with open(ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        arr = data["ontology"] if "ontology" in data else data
        for it in arr:
            iid = it.get("icon_id")
            if iid:
                role = it.get("cs_role") or "WHAT"
                icon2cs[iid] = CS_ROLE_TO_ID.get(role, CS_ROLE_TO_ID["WHAT"])
    except Exception as e:
        print(f"[warn] cannot load ontology for cs_role: {e}")
    return icon2cs


def cs_seq(prefix: List[str], icon2cs: Dict[str, int]) -> List[int]:
    return [icon2cs.get(x, CS_ROLE_TO_ID["WHAT"]) for x in prefix]


# ============================================================
# Track 1: CE next-item dataset (wraps SASRecDataset)
# ============================================================

def make_ce_raw(pos_lines: List[dict], icon2cs: Dict[str, int]) -> List[dict]:
    """Each pos line -> a sequence = prefix + [next]; SASRecDataset does the shift."""
    raw = []
    for d in pos_lines:
        seq = list(d["prefix"]) + [d["next"]]
        if len(seq) < 2:
            continue
        raw.append({"sequence": seq, "cs_roles": cs_seq(seq, icon2cs)})
    return raw


# ============================================================
# Track 2: S-DP-O dataset (built from pos/neg grouped by prefix)
# ============================================================

class Review5000SDPODataset(Dataset):
    """
    Each instance: (prefix, chosen, [rejected...]).
    chosen   = a POS `next` for this prefix.
    rejected = same-prefix NEG `next`(s) + (if fewer than neg_k) random vocab fill.
    `prefix_cs` is a precomputed list of CS role indices (same length as prefix).
    """

    def __init__(self, prefs: List[dict], item2idx: Dict[str, int],
                 max_seq_len: int = 16, max_neg: int = 4):
        self.item2idx = item2idx
        self.max_seq_len = max_seq_len
        self.max_neg = max_neg
        self.data = []
        for p in prefs:
            pref = p["prefix"][-max_seq_len:]
            pref_cs = p["prefix_cs"][-max_seq_len:]
            seq_icons = [item2idx.get(i, 0) for i in pref]
            seq_cs = pref_cs
            chosen = item2idx.get(p["chosen"], 0)
            rejected = [item2idx.get(r, 0) for r in p["rejected"] if r in item2idx]
            if chosen == 0 or len(rejected) == 0:
                continue
            self.data.append({
                "seq_item_ids": seq_icons,
                "seq_cs_ids": seq_cs,
                "chosen_idx": chosen,
                "rejected_idxs": rejected,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq_len = len(item["seq_item_ids"])
        pad = self.max_seq_len - seq_len
        seq_item = item["seq_item_ids"] + [0] * pad
        seq_cs = item["seq_cs_ids"] + [0] * pad

        rej = item["rejected_idxs"][:self.max_neg]
        rej_mask = [1] * len(rej) + [0] * (self.max_neg - len(rej))
        rej_pad = rej + [0] * (self.max_neg - len(rej))

        return {
            "seq_item_ids": torch.tensor(seq_item, dtype=torch.long),
            "seq_cs_ids": torch.tensor(seq_cs, dtype=torch.long),
            "seq_lengths": torch.tensor(min(seq_len, self.max_seq_len), dtype=torch.long),
            "chosen_idx": torch.tensor(item["chosen_idx"], dtype=torch.long),
            "rejected_idxs": torch.tensor(rej_pad, dtype=torch.long),
            "rejected_mask": torch.tensor(rej_mask, dtype=torch.float),
        }


def build_preference_pairs(lines: List[dict], item2idx: Dict[str, int],
                           icon2cs: Dict[str, int],
                           neg_k: int = 4, seed: int = 42) -> List[dict]:
    """Group by prefix: chosen=pos next(s), rejected=neg next(s)+random fill."""
    rng = random.Random(seed)
    pos_by_pref = defaultdict(list)
    neg_by_pref = defaultdict(list)
    for d in lines:
        key = tuple(d["prefix"])
        (pos_by_pref if d["label"] == "pos" else neg_by_pref)[key].append(d["next"])

    vocab_pool = [ic for ic in item2idx.keys() if item2idx[ic] != 0]
    prefs = []
    for key, chosens in pos_by_pref.items():
        rejs = list(neg_by_pref.get(key, []))
        forbidden = set(chosens) | set(rejs)
        while len(rejs) < neg_k:
            cand = rng.choice(vocab_pool)
            if cand not in forbidden:
                rejs.append(cand)
                forbidden.add(cand)
        for ch in chosens:
            prefs.append({
                "prefix": list(key),
                "prefix_cs": cs_seq(list(key), icon2cs),
                "chosen": ch,
                "rejected": rejs,
            })
    return prefs


# ============================================================
# Track 1 training (reuses sasrec train_epoch-style loop)
# ============================================================

def train_ce(model, train_loader, val_loader, idx2item, optimizer, scheduler,
             device, args):
    best_val_mrr = 0.0
    patience = 0
    for epoch in range(1, args.ce_epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"CE Epoch {epoch}", leave=False):
            item_ids = batch["item_ids"].to(device)
            cs_roles = batch["cs_roles"].to(device)
            target_ids = batch["target_ids"].to(device)
            seq_lengths = batch["seq_lengths"]
            logits = model(item_ids, cs_roles, return_all_positions=True)
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
                pred = logits[i, last_pos]
                loss += F.cross_entropy(pred.unsqueeze(0), target.unsqueeze(0))
                if pred.argmax().item() == target.item():
                    correct += 1
                samples += 1
            if samples > 0:
                loss = loss / samples
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() * samples
                total_correct += correct
                total_samples += samples
        scheduler.step()
        val = compute_metrics(model, val_loader, idx2item, device)
        print(f"CE Epoch {epoch:3d} | Loss {total_loss/max(total_samples,1):.4f} | "
              f"Acc {total_correct/max(total_samples,1):.4f} | "
              f"Val Hit@1 {val['hit@1']:.4f} Hit@5 {val['hit@5']:.4f} "
              f"MRR {val['mrr']:.4f} NDCG@5 {val['ndcg@5']:.4f}")
        if val["mrr"] > best_val_mrr:
            best_val_mrr = val["mrr"]
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val,
                "item2idx": model._item2idx,
                "idx2item": model._idx2item,
                "args": vars(args),
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> saved best CE model (MRR {best_val_mrr:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"CE early stopping at epoch {epoch}")
                break
    return best_val_mrr


# ============================================================
# Track 2 training (S-DP-O, reuses SDPOLoss + get_log_probability)
# ============================================================

def train_sdpo(policy, ref, train_loader, val_loader, device, args):
    optimizer = torch.optim.AdamW(policy.parameters(),
                                  lr=args.lr_sdpo, weight_decay=args.weight_decay)
    sdpo_fn = SDPOLoss(beta=args.beta)
    best_val_loss = float("inf")
    patience = 0
    for epoch in range(1, args.sdpo_epochs + 1):
        policy.train(); ref.eval()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"S-DP-O Epoch {epoch}", leave=False):
            seq_item = batch["seq_item_ids"].to(device)
            seq_cs = batch["seq_cs_ids"].to(device)
            seq_len = batch["seq_lengths"].to(device)
            chosen = batch["chosen_idx"].to(device)
            rejected = batch["rejected_idxs"].to(device)
            rej_mask = batch["rejected_mask"].to(device)
            num_neg = rejected.shape[1]

            with torch.enable_grad():
                pc = get_log_probability(policy, seq_item, seq_cs, chosen, seq_len)
                pr = []
                for j in range(num_neg):
                    pr.append(get_log_probability(policy, seq_item, seq_cs,
                                                 rejected[:, j], seq_len) * rej_mask[:, j])
                pr = torch.stack(pr, dim=1)

            with torch.no_grad():
                rc = get_log_probability(ref, seq_item, seq_cs, chosen, seq_len)
                rr = []
                for j in range(num_neg):
                    rr.append(get_log_probability(ref, seq_item, seq_cs,
                                                 rejected[:, j], seq_len) * rej_mask[:, j])
                rr = torch.stack(rr, dim=1)

            loss = sdpo_fn(pc, pr, rc, rr, rej_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item(); n += 1

        # val
        policy.eval(); val_loss = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_item = batch["seq_item_ids"].to(device)
                seq_cs = batch["seq_cs_ids"].to(device)
                seq_len = batch["seq_lengths"].to(device)
                chosen = batch["chosen_idx"].to(device)
                rejected = batch["rejected_idxs"].to(device)
                rej_mask = batch["rejected_mask"].to(device)
                num_neg = rejected.shape[1]
                pc = get_log_probability(policy, seq_item, seq_cs, chosen, seq_len)
                rc = get_log_probability(ref, seq_item, seq_cs, chosen, seq_len)
                pr, rr = [], []
                for j in range(num_neg):
                    pr.append(get_log_probability(policy, seq_item, seq_cs,
                                                 rejected[:, j], seq_len) * rej_mask[:, j])
                    rr.append(get_log_probability(ref, seq_item, seq_cs,
                                                 rejected[:, j], seq_len) * rej_mask[:, j])
                pr = torch.stack(pr, dim=1); rr = torch.stack(rr, dim=1)
                val_loss += sdpo_fn(pc, pr, rc, rr, rej_mask).item(); vn += 1

        avg_val = val_loss / max(vn, 1)
        print(f"S-DP-O Epoch {epoch:3d} | Train Loss {total_loss/max(n,1):.4f} | "
              f"Val Loss {avg_val:.4f}")
        if avg_val < best_val_loss:
            best_val_loss = avg_val; patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": policy.state_dict(),
                "val_loss": avg_val,
                "item2idx": policy._item2idx,
                "idx2item": policy._idx2item,
                "args": vars(args),
            }, os.path.join(args.output_dir, "best_sdpo_model.pt"))
            print(f"  -> saved best S-DP-O model (val_loss {best_val_loss:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"S-DP-O early stopping at epoch {epoch}")
                break
    return best_val_loss


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Train SASRec on review5000 (CE + S-DP-O)")
    ap.add_argument("--data", default="/home/user1/liuduanye/review5000_combined_full.jsonl")
    ap.add_argument("--ontology",
                    default="/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json")
    ap.add_argument("--output-dir", default="/home/user1/liuduanye/EmotionClassify/output/review5000")
    # architecture
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cs-role-emb-dim", type=int, default=16)
    # CE track
    ap.add_argument("--ce-epochs", type=int, default=20)
    ap.add_argument("--ce-batch-size", type=int, default=128)
    ap.add_argument("--lr-ce", type=float, default=1e-3)
    ap.add_argument("--skip-ce", action="store_true")
    ap.add_argument("--ce-checkpoint", type=str, default="")
    # S-DP-O track
    ap.add_argument("--sdpo-epochs", type=int, default=5)
    ap.add_argument("--sdpo-batch-size", type=int, default=16)
    ap.add_argument("--lr-sdpo", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--neg-k", type=int, default=4,
                    help="rejected negatives per prefix (human_bad+random_neg, padded to this)")
    ap.add_argument("--neg-cap", type=int, default=16,
                    help="hard cap on padded negatives per instance (bounds max_neg; avoids 458-long padding)")
    # shared
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print("device:", device)

    # ---- load data ----
    lines = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    print(f"loaded {len(lines)} lines from {args.data}")
    pos_lines = [d for d in lines if d["label"] == "pos"]
    neg_lines = [d for d in lines if d["label"] == "neg"]
    print(f"  pos={len(pos_lines)} (sequence+human_good)  neg={len(neg_lines)} (human_bad+random_neg)")

    item2idx, idx2item = build_vocab_from_data(lines)
    num_items = len(item2idx) - 1
    print(f"vocab size = {num_items} icons (+PAD)")

    icon2cs = build_icon2cs(args.ontology)
    covered = sum(1 for ic in item2idx if ic in icon2cs)
    print(f"cs_role lookup: {covered}/{num_items} icons covered")

    def make_model():
        m = SASRec(
            num_items=num_items,
            num_cs_roles=len(CS_ROLE_TO_ID),
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            cs_role_emb_dim=args.cs_role_emb_dim,
        ).to(device)
        m._item2idx = item2idx
        m._idx2item = idx2item
        return m

    # ============================================================
    # Track 1: CE
    # ============================================================
    if not args.skip_ce:
        ce_raw = make_ce_raw(pos_lines, icon2cs)
        random.seed(args.seed)
        random.shuffle(ce_raw)
        n_val = max(1, int(len(ce_raw) * args.val_split))
        val_raw, train_raw = ce_raw[:n_val], ce_raw[n_val:]
        train_ds = SASRecDataset(train_raw, item2idx, args.max_seq_len, is_training=True)
        val_ds = SASRecDataset(val_raw, item2idx, args.max_seq_len, is_training=False)
        train_loader = DataLoader(train_ds, batch_size=args.ce_batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.ce_batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)
        print(f"CE dataset: train={len(train_ds)} val={len(val_ds)}")

        model = make_model()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"SASRec params: {n_params:,}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_ce,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ce_epochs)
        train_ce(model, train_loader, val_loader, idx2item, optimizer, scheduler, device, args)
        ce_ckpt = os.path.join(args.output_dir, "best_model.pt")
    else:
        ce_ckpt = args.ce_checkpoint or os.path.join(args.output_dir, "best_model.pt")
        print(f"[skip-ce] using reference checkpoint: {ce_ckpt}")

    # ============================================================
    # Track 2: S-DP-O
    # ============================================================
    prefs = build_preference_pairs(lines, item2idx, icon2cs,
                                  neg_k=args.neg_k, seed=args.seed)
    random.shuffle(prefs)
    n_val = max(1, int(len(prefs) * args.val_split))
    val_prefs, train_prefs = prefs[:n_val], prefs[n_val:]

    raw_max = max([args.neg_k] + [len(p["rejected"]) for p in prefs])
    max_neg = min(args.neg_cap, raw_max)
    print(f"  (raw max_neg={raw_max}, capped to {max_neg} via --neg-cap)")
    train_sdpo_ds = Review5000SDPODataset(train_prefs, item2idx, args.max_seq_len, max_neg)
    val_sdpo_ds = Review5000SDPODataset(val_prefs, item2idx, args.max_seq_len, max_neg)
    train_sdpo_loader = DataLoader(train_sdpo_ds, batch_size=args.sdpo_batch_size,
                                   shuffle=True, num_workers=2, pin_memory=True)
    val_sdpo_loader = DataLoader(val_sdpo_ds, batch_size=args.sdpo_batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
    print(f"S-DP-O pairs: train={len(train_sdpo_ds)} val={len(val_sdpo_ds)} (max_neg={max_neg})")

    ckpt = torch.load(ce_ckpt, map_location=device, weights_only=False)
    ref_model = make_model(); ref_model.load_state_dict(ckpt["model_state_dict"]); ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    policy_model = make_model(); policy_model.load_state_dict(ckpt["model_state_dict"])

    train_sdpo(policy_model, ref_model, train_sdpo_loader, val_sdpo_loader, device, args)
    print("done. artifacts:")
    print("  ", os.path.join(args.output_dir, "best_model.pt"), "(CE base)")
    print("  ", os.path.join(args.output_dir, "best_sdpo_model.pt"), "(S-DP-O aligned)")


if __name__ == "__main__":
    main()
