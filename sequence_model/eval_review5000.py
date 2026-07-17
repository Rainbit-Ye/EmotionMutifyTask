#!/usr/bin/env python3
"""
Diagnostic eval for review5000 SASRec checkpoints.

Answers 3 questions about why metrics are low:
  1. Did S-DP-O (Track 2) actually help vs CE-only?  -> compare on same val
  2. Popularity collapse?  -> how many UNIQUE icons does top-1 cover
  3. How concentrated is the prediction distribution?

Run from EmotionClassify/:
    python sequence_model/eval_review5000.py
"""
import os, sys, json, random
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sequence_model.train_review5000 as tr
from sequence_model.sasrec import SASRec, SASRecDataset, CS_ROLE_TO_ID, compute_metrics

DATA = "/home/user1/liuduanye/review5000_combined_full.jsonl"
ONTO = "/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json"
OUT  = "/home/user1/liuduanye/EmotionClassify/output/review5000"
SEED = 42
MAX_SEQ = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

lines = [json.loads(l) for l in open(DATA, encoding="utf-8")]
item2idx, idx2item = tr.build_vocab_from_data(lines)
icon2cs = tr.build_icon2cs(ONTO)
num_items = len(item2idx) - 1

pos = [d for d in lines if d["label"] == "pos"]
ce_raw = tr.make_ce_raw(pos, icon2cs)
random.seed(SEED); random.shuffle(ce_raw)
n_val = max(1, int(len(ce_raw) * 0.1))
val_raw = ce_raw[:n_val]
val_ds = SASRecDataset(val_raw, item2idx, MAX_SEQ, is_training=False)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
print(f"val size={len(val_ds)}  vocab={num_items}")


def load(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    m = SASRec(num_items=num_items, num_cs_roles=len(CS_ROLE_TO_ID),
                hidden_size=a["hidden_size"], num_heads=a["num_heads"],
                num_blocks=a["num_blocks"], max_seq_len=a["max_seq_len"],
                dropout=0.0, cs_role_emb_dim=a["cs_role_emb_dim"]).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def popularity(m):
    """unique top-1 icons predicted over val + entropy of top-1 dist."""
    m.eval()
    preds = []
    with torch.no_grad():
        for b in val_loader:
            logits = m(b["item_ids"].to(device), b["cs_roles"].to(device))
            logits[:, 0] = float("-inf")
            preds += logits.argmax(1).tolist()
    from collections import Counter
    c = Counter(preds)
    uniq = len(c)
    top10 = c.most_common(10)
    total = len(preds)
    # share covered by top-10 predicted icons
    top10_share = sum(v for _, v in top10) / total
    print(f"  unique top-1 predicted: {uniq}/{num_items} ({100*uniq/num_items:.1f}% of vocab)")
    print(f"  top-10 predicted icons cover {100*top10_share:.1f}% of all val predictions")
    print("  top-10 predicted icons:",
          [(idx2item.get(i, "?"), n) for i, n in top10])
    # entropy
    import math
    p = [v / total for v in c.values()]
    ent = -sum(x * math.log(x + 1e-12) for x in p)
    print(f"  top-1 prediction entropy: {ent:.2f} (max={math.log(num_items):.2f}, higher=more diverse)")


for name, path in [("CE-only", os.path.join(OUT, "best_model.pt")),
                           ("S-DP-O", os.path.join(OUT, "best_sdpo_model.pt"))]:
    print(f"\n===== {name} =====")
    m = load(path)
    print("  metrics:", {k: round(v, 4) for k, v in compute_metrics(m, val_loader, idx2item, device).items()})
    popularity(m)
