#!/usr/bin/env python3
"""
Route D 评测：用 AI（基于 ontology 语义）替代人工，判"建议是否合理"。

为什么需要这个：review5000 的 next-icon，上下文只有 2-3 张图、候选 1157 个，
精确命中（Hit@K/MRR）天花板极低。但"用户觉得建议合理吗"才是产品真问题。
本脚本把"我（AI）当人工"编码成可复现的判分函数，跑在全部 held-out val 上。

判分标准（对每条 top 建议给 reasonable yes/no）：
  1) 槽位契合 role-fit：CS 规范语序 WHO→WHAT_DOING→WHAT→WHERE→WHEN→HOW。
     看 prefix 已填槽，建议的 cs_role 应落在"下一个待填槽"或之后需要的槽；
     若建议一个已填过的槽（如已说 WHO 又建议 WHO）则欠合理。
  2) 语义连贯 coherence：建议的 super_concept / aac_category / can_combine_with /
     typical_objects 需与 prefix 任何一方有重叠；若双方都有明确主题且主题互斥则判不连贯。
  3) 非退化 non-degenerate：建议不是 prefix 里已出现的同一图标。

运行（Emotion 环境，用 CUDA）：
    python sequence_model/eval_human_reasonable.py
"""
import os, sys, json, math, random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sequence_model.train_review5000 as tr
from sequence_model.sasrec import SASRec, SASRecDataset, CS_ROLE_TO_ID, compute_metrics

# ---------------- 配置 ----------------
DATA = "/home/user1/liuduanye/review5000_combined_full.jsonl"
ONTO = "/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json"
OUT  = "/home/user1/liuduanye/EmotionClassify/output/review5000"
SEED = 42
MAX_SEQ = 16
TOPK = 5
N_EXAMPLES = 15          # 打印多少条"判读卡"供人工审计

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------- 载入数据 / 词表 ----------------
lines = [json.loads(l) for l in open(DATA, encoding="utf-8")]
item2idx, idx2item = tr.build_vocab_from_data(lines)
icon2cs = tr.build_icon2cs(ONTO)
num_items = len(item2idx) - 1
print(f"vocab = {num_items} icons")

pos = [d for d in lines if d["label"] == "pos"]
ce_raw = tr.make_ce_raw(pos, icon2cs)
random.seed(SEED); random.shuffle(ce_raw)
n_val = max(1, int(len(ce_raw) * 0.1))
val_raw = ce_raw[:n_val]
val_ds = SASRecDataset(val_raw, item2idx, MAX_SEQ, is_training=False)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
val_samples = [{"prefix": seq["sequence"][:-1], "next": seq["sequence"][-1]} for seq in val_raw]
print(f"held-out val = {len(val_samples)} 条 (来自 {len(pos)} 条 pos, 10% split)")

# ---------------- ontology 语义字段 ----------------
def load_sem(onto_path):
    sem = {}
    with open(onto_path, encoding="utf-8") as f:
        data = json.load(f)
    arr = data["ontology"] if "ontology" in data else data
    for it in arr:
        sem[it["icon_id"]] = it
    return sem

sem = load_sem(ONTO)

def tagset(d):
    if not d:
        return set()
    s = set()
    for k in ("can_combine_with", "typical_objects", "typical_modifiers"):
        for v in (d.get(k) or []):
            s.add(str(v).lower())
    for k in ("super_concept", "aac_category", "core_semantic"):
        if d.get(k):
            s.add(str(d[k]).lower())
    return s

def label_of(ic):
    d = sem.get(ic)
    return f"{d['label']} ({ic})" if d and d.get("label") else ic

ROLE_ORDER = {"WHO":1,"WHAT_DOING":2,"WHAT":3,"WHERE":4,"WHEN":5,"HOW":6}
ROLE_NAME = {v:k for k,v in ROLE_ORDER.items()}

def expected_role(prefix):
    filled = set()
    for ic in prefix:
        r = sem.get(ic, {}).get("cs_role")
        if r in ROLE_ORDER:
            filled.add(ROLE_ORDER[r])
    for order in (1,2,3,4,5,6):
        if order not in filled:
            return order
    return 6

def coherence_ok(prefix, sug):
    s = sem.get(sug, {})
    s_tags = tagset(s)
    if not s_tags:
        return True
    p_union = set(); p_topics = []
    for ic in prefix:
        p = sem.get(ic, {})
        p_union |= tagset(p)
        if p.get("super_concept"):
            p_topics.append(p["super_concept"])
    if not p_union:
        return True
    if s_tags & p_union:
        return True
    s_topic = s.get("super_concept")
    if s_topic and p_topics and s_topic not in p_topics:
        return False
    return True

def ai_judge(prefix, sug):
    """返回 (reasonable: bool, reason: str)。即"AI 当人工"的判分。"""
    s = sem.get(sug, {})
    sr = ROLE_ORDER.get(s.get("cs_role"))
    er = expected_role(prefix)
    if sug in prefix:
        return (False, f"与上下文重复 ({label_of(sug)})")
    if sr is None:
        return (False, f"无 CS 槽位信息 ({label_of(sug)})")
    if sr < er:
        return (False, f"槽位错: 建议 {s.get('cs_role')} 但下一个待填 {ROLE_NAME[er]}")
    if not coherence_ok(prefix, sug):
        return (False, f"语义不连贯: {label_of(sug)} 与上下文主题互斥")
    return (True, f"槽位 {s.get('cs_role')} 契合且语义可搭配 ({label_of(sug)})")

# ---------------- 载入模型 ----------------
def load(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    m = SASRec(num_items=num_items, num_cs_roles=len(CS_ROLE_TO_ID),
                hidden_size=a["hidden_size"], num_heads=a["num_heads"],
                num_blocks=a["num_blocks"], max_seq_len=a["max_seq_len"],
                dropout=0.0, cs_role_emb_dim=a["cs_role_emb_dim"]).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m

models = {}
if os.path.exists(os.path.join(OUT, "best_model.pt")):
    models["CE"] = load(os.path.join(OUT, "best_model.pt"))
if os.path.exists(os.path.join(OUT, "best_sdpo_model.pt")):
    models["S-DP-O"] = load(os.path.join(OUT, "best_sdpo_model.pt"))

def predict_topk(model, prefix, k=TOPK):
    p = prefix[-MAX_SEQ:]
    item_ids = torch.tensor([[item2idx.get(i,0) for i in p]], dtype=torch.long, device=device)
    cs_ids   = torch.tensor([[icon2cs.get(i, CS_ROLE_TO_ID["WHAT"]) for i in p]], dtype=torch.long, device=device)
    with torch.no_grad():
        idx, prob = model.predict_next(item_ids, cs_ids, top_k=k)
    idx = idx[0].tolist(); prob = prob[0].tolist()
    return [(idx2item.get(i, "?"), round(p_, 4)) for i, p_ in zip(idx, prob)]

# ---------------- 基线 ----------------
def evaluate(model, samples):
    model.eval()
    hits = defaultdict(int); mrr_sum = 0.0; total = 0
    with torch.no_grad():
        for s in samples:
            p = s["prefix"][-MAX_SEQ:]
            item_ids = torch.tensor([[item2idx.get(i,0) for i in p]], dtype=torch.long, device=device)
            cs_ids   = torch.tensor([[icon2cs.get(i, CS_ROLE_TO_ID["WHAT"]) for i in p]], dtype=torch.long, device=device)
            logits = model(item_ids, cs_ids); logits[0,0] = float("-inf")
            probs = F.softmax(logits, dim=-1)[0]
            t = item2idx.get(s["next"], 0)
            if t == 0: continue
            ranked = torch.argsort(probs, descending=True)
            rank = (ranked == t).nonzero(as_tuple=True)[0]
            if len(rank) == 0: continue
            rank = rank[0].item() + 1
            total += 1
            if rank <= 1: hits[1] += 1
            if rank <= 5: hits[5] += 1
            mrr_sum += 1.0/rank
    return {"hit@1": hits[1]/max(total,1), "hit@5": hits[5]/max(total,1),
            "mrr": mrr_sum/max(total,1), "total": total}

def popularity_baseline(train_pos, samples):
    cnt = defaultdict(int)
    for d in train_pos: cnt[d["next"]] += 1
    order = sorted(cnt, key=lambda k: -cnt[k])
    rank_of = {ic: r+1 for r, ic in enumerate(order)}
    N = len(order); h1=h5=0; mrr=0.0; total=0
    for s in samples:
        r = rank_of.get(s["next"], N+1); total += 1
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        mrr += 1.0/r
    return {"hit@1": h1/max(total,1), "hit@5": h5/max(total,1), "mrr": mrr/max(total,1)}

def uniform_random_baseline(N, n):
    H = sum(1.0/r for r in range(1, N+1))
    return {"hit@1": 1.0/N, "hit@5": 5.0/N, "mrr": H/N}

# ---------------- 跑评测 ----------------
print("\n===== 三基线对照（精确命中）=====")
pop = popularity_baseline(pos, val_samples)
uni = uniform_random_baseline(num_items, len(val_samples))
print(f"{'方法':<14}{'Hit@1':>10}{'Hit@5':>10}{'MRR':>10}")
print(f"{'均匀随机':<14}{uni['hit@1']:>10.4f}{uni['hit@5']:>10.4f}{uni['mrr']:>10.4f}")
print(f"{'猜最高频':<14}{pop['hit@1']:>10.4f}{pop['hit@5']:>10.4f}{pop['mrr']:>10.4f}")
for name, m in models.items():
    r = evaluate(m, val_samples)
    print(f"{name+' 模型':<14}{r['hit@1']:>10.4f}{r['hit@5']:>10.4f}{r['mrr']:>10.4f}")

# ---------------- AI 当人工：全量判读 ----------------
print("\n===== AI 当人工：全量判'建议是否合理'（held-out val 全 " + str(len(val_samples)) + " 条）=====")
print(f"{'模型':<10}{'top1合理率':>12}{'top5合理率':>12}{'  (对照: top1精确':>16}{'top5精确)':>12}")
for name, m in models.items():
    top1_r = top5_r = n = 0
    for s in val_samples:
        top5 = predict_topk(m, s["prefix"])
        n += 1
        r1, _ = ai_judge(s["prefix"], top5[0][0])
        if r1: top1_r += 1
        any5 = any(ai_judge(s["prefix"], ic)[0] for ic, _ in top5)
        if any5: top5_r += 1
    met = evaluate(m, val_samples)
    print(f"{name:<10}{top1_r/max(n,1):>12.3f}{top5_r/max(n,1):>12.3f}{met['hit@1']:>16.3f}{met['hit@5']:>12.3f}")

# ---------------- 判读卡示例（供人工审计我的判分口径）----------------
print(f"\n===== 判读卡示例（随机 {N_EXAMPLES} 条，看 AI 判分是否靠谱）=====")
rng = random.Random(SEED)
for s in rng.sample(val_samples, min(N_EXAMPLES, len(val_samples))):
    prefix = s["prefix"]; true_next = s["next"]
    print("\n" + "-"*70)
    print("上下文 prefix:", " > ".join(label_of(i) for i in prefix))
    print("真实 next   :", label_of(true_next),
          f"(CS={sem.get(true_next,{}).get('cs_role','?')}, 在词表 idx={item2idx.get(true_next,'?')})")
    top5 = predict_topk(models.get("CE") or list(models.values())[0], prefix)
    for rank, (ic, prob) in enumerate(top5, 1):
        ok, reason = ai_judge(prefix, ic)
        mark = "✓合理" if ok else "✗不合理"
        print(f"  #{rank} {label_of(ic):<40} p={prob:<7} {mark}  {reason}")
