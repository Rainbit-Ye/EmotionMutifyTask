#!/usr/bin/env python3
"""
Fusion semantic-RAG experiment for review5000 next-icon.

Goal: does adding a SEMANTIC similarity term to the sequential SASRec
score relieve the popularity collapse (only 6.7% of vocab ever top-1)
and produce more USEFUL (semantically-coherent) suggestions?

Because no sentence-transformers is installed anywhere, the "query" is the
MEAN EMBEDDING of the already-chosen prefix icons (pure geometry,
no text encoder). Candidate score = cosine(query, icon_embedding) over the
full 3154-icon table (covers icons outside the 1157 SASRec vocab).

Fusion (mirrors fusion.py):
    fused(i) = alpha * norm(sasrec(i)) + (1-alpha) * norm(sem(i))

Metrics per alpha:
  EXACT (same yardstick as before): Hit@1/3/5/10, MRR
  SEMANTIC-USEFULNESS:
    sem@top1  = mean over val of cos(emb(pred_top1), emb(true_next))
    sem@top5  = mean over val of max_{k<=5} cos(emb(pred_k), emb(true_next))
    close@1   = fraction of val where cos(top1, true_next) > 0.5
  COLLAPSE: unique top-1 icons / 3154 (higher = less collapse)

Run from EmotionClassify/:
    python sequence_model/fuse_review5000_exp.py
"""
import os, sys, json, math, random
from collections import Counter
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sequence_model.train_review5000 as tr
from sequence_model.sasrec import SASRec, SASRecDataset, CS_ROLE_TO_ID

DATA   = "/home/user1/liuduanye/review5000_combined_full.jsonl"
ONTO   = "/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json"
CKPT   = "/home/user1/liuduanye/EmotionClassify/output/review5000/best_model.pt"
EMB    = "/home/user1/liuduanye/EmotionClassify/data/icon_embeddings_rich.pt"
SEED, MAX_SEQ, SPLIT = 42, 16, 0.1
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- data -> val (same split as training) ----
lines = [json.loads(l) for l in open(DATA, encoding="utf-8")]
item2idx, idx2item = tr.build_vocab_from_data(lines)
num_items = len(item2idx) - 1
icon2cs = tr.build_icon2cs(ONTO)
pos = [d for d in lines if d["label"] == "pos"]
ce_raw = tr.make_ce_raw(pos, icon2cs)
random.seed(SEED); random.shuffle(ce_raw)
val_raw = ce_raw[:max(1, int(len(ce_raw) * SPLIT))]
val_ds = SASRecDataset(val_raw, item2idx, MAX_SEQ, is_training=False)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
print(f"val={len(val_ds)}  sasrec vocab={num_items} (+PAD)")

# ---- SASRec ----
ck = torch.load(CKPT, map_location=device, weights_only=False)
a = ck["args"]
model = SASRec(num_items=num_items, num_cs_roles=len(CS_ROLE_TO_ID),
               hidden_size=a["hidden_size"], num_heads=a["num_heads"],
               num_blocks=a["num_blocks"], max_seq_len=a["max_seq_len"],
               dropout=0.0, cs_role_emb_dim=a["cs_role_emb_dim"]).to(device)
model.load_state_dict(ck["model_state_dict"]); model.eval()

# ---- embeddings ----
emb = torch.load(EMB, map_location="cpu", weights_only=False)
E = emb["embeddings"].float()            # [3154, 384]
E_ids = emb["icon_ids"]                  # list of icon_id (str)
icon2emb = {iid: E[n] for n, iid in enumerate(E_ids)}
E_norm = F.normalize(E, dim=-1)
print(f"embeddings: {E.shape[0]} icons, dim={E.shape[1]}; "
      f"overlap w/ sasrec vocab: {sum(1 for i in item2idx if i in icon2emb and i!='PAD')}/{num_items}")


def sasrec_scores(item_ids, cs_ids):
    """softmax prob over sasrec vocab (mask PAD). returns dict icon->prob."""
    with torch.no_grad():
        logits = model(item_ids, cs_ids)            # [1, num_items+1]
        logits[0, 0] = float("-inf")
        probs = F.softmax(logits[0], -1)
    return {idx2item[i]: probs[i].item() for i in range(1, num_items + 1)}


def semantic_scores(prefix_icons):
    """cosine(query_mean_emb, all 3154 embs). returns dict icon->cos."""
    q = torch.stack([icon2emb[x] for x in prefix_icons if x in icon2emb])
    if len(q) == 0:
        return {iid: 0.0 for iid in icon2emb}
    q = F.normalize(q.mean(0, keepdim=True), dim=-1)       # [1,384]
    cos = (q @ E_norm.T)[0]                                  # [3154]
    return {E_ids[n]: cos[n].item() for n in range(len(E_ids))}


def minmax(d):
    vs = list(d.values())
    lo, hi = min(vs), max(vs)
    if hi == lo:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


# precompute per-val-sample scores
samples = []
with torch.no_grad():
    for b in val_loader:
        B = b["item_ids"].shape[0]
        for i in range(B):
            seq_len = int(b["seq_lengths"][i].item())
            seq = b["item_ids"][i, :seq_len].tolist()
            cs = b["cs_roles"][i, :seq_len].tolist()
            prefix = [idx2item[t] for t in seq]
            target = idx2item[int(b["target_ids"][i, seq_len - 1].item())]
            if target == "PAD" or target not in icon2emb:
                continue
            sas = sasrec_scores(b["item_ids"][i:i+1].to(device),
                                  b["cs_roles"][i:i+1].to(device))
            sem = semantic_scores(prefix)
            samples.append((prefix, target, sas, sem))

print(f"scored samples: {len(samples)}")

# reference: pure-semantic retrieval of the true next
pure_sem_hit = sum(1 for _, t, _, sem in samples
                     if max(sem.values()) > 0 and
                     sorted(sem, key=sem.get, reverse=True)[:1] == [t]) / len(samples)
print(f"[ref] pure-semantic top-1 recovers true_next: {pure_sem_hit:.3f}")

# ---- sweep alpha ----
print(f"\n{'alpha':>6} | {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7} | "
      f"{'sem@top1':>9} {'sem@top5':>9} {'close@1':>8} | {'uniqTop1':>8}")
for alpha in ALPHAS:
    fused_rank, hit1, hit5 = [], 0, 0
    sem_top1, sem_top5, close1, uniq = [], [], 0, Counter()
    for prefix, target, sas, sem in samples:
        fs = {}
        for k in set(sas) | set(sem):
            fs[k] = alpha * sas.get(k, 0.0) + (1 - alpha) * sem.get(k, 0.0)
        ranking = sorted(fs, key=fs.get, reverse=True)
        # exact rank of true next
        r = ranking.index(target) + 1
        fused_rank.append(r)
        if r <= 1: hit1 += 1
        if r <= 5: hit5 += 1
        # semantic usefulness
        t_emb = icon2emb[target]
        preds = [icon2emb[x] for x in ranking[:5] if x in icon2emb]
        if preds:
            cs = F.cosine_similarity(torch.stack(preds),
                                      t_emb.unsqueeze(0))
            sem_top5.append(float(cs.max().item()))
            sem_top1.append(float(cs[0].item()))
            if cs[0].item() > 0.5:
                close1 += 1
        uniq[ranking[0]] += 1
    n = len(samples)
    mrr = sum(1.0 / r for r in fused_rank) / n
    print(f"{alpha:>6.1f} | {hit1/n:>7.3f} {hit5/n:>7.3f} {mrr:>7.3f} | "
          f"{sum(sem_top1)/n:>9.3f} {sum(sem_top5)/n:>9.3f} {close1/n:>8.3f} | "
          f"{len(uniq):>6}/{len(icon2emb)}")
print("\nsem@top1/5 = mean cosine of predicted top-1/5 to the TRUE next embedding")
print("close@1 = fraction where predicted top-1 is semantically near true next (cos>0.5)")
print("uniqTop1 = how many distinct icons appear as top-1 (collapse check)")
