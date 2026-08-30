#!/usr/bin/env python3
"""
把 propicto-eval.json 映射到 AAC2Text 可用 label:
  - picto ID -> arasaac_en_keyword2ids.json 反查英文 keyword, 直接当 label (Llama 预训练认识英文)
  - 优先选已存在于 mamba_vocab 的 label (与现有模型一致); 否则取最短(最具体头词)的英文 keyword
  - 仅保留所有 pictos 都能映射到英文 label 的序列
输出 data/cleardata/propicto_eval_labels_en.json
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "public_datasets/propicto-cnrs/propicto-eval/propicto-eval.json"
EN2IDS = ROOT / "public_datasets/arasaac_en_keyword2ids.json"
VOCAB = ROOT / "propicto_zh_pipeline/aac2text_web/data/mamba_vocab.json"
OUT = ROOT / "EmotionClassify/AAC2Text/data/cleardata/propicto_eval_labels_en.json"

def norm(s):
    return s.lower().replace(" ", "_").replace("-", "_").replace("'", "").strip("_")

def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    en2ids = json.load(open(EN2IDS, encoding="utf-8"))
    item2idx = json.load(open(VOCAB, encoding="utf-8"))["item2idx"]
    known = set(item2idx.keys())

    id2en = {}
    for kw, ids in en2ids.items():
        nk = norm(kw)
        for i in ids:
            id2en.setdefault(i, []).append(nk)

    def choose_label(i):
        cands = id2en.get(i, [])
        if not cands:
            return None
        # 优先 mamba_vocab 中已有的
        in_vocab = [c for c in cands if c in known]
        if in_vocab:
            return sorted(in_vocab)[0]
        # 否则取词数最少(头词)的, 平局按字母
        return sorted(cands, key=lambda c: (c.count("_") + 1, c))[0]

    total = len(ev)
    full = []
    no_en = 0
    for idx, e in enumerate(ev):
        pictos = e.get("pictos") or []
        if not pictos:
            continue
        labels = []
        ok = True
        for p in pictos:
            lab = choose_label(p)
            if lab is None:
                ok = False
                break
            labels.append(lab)
        if ok:
            full.append({"orig_idx": idx, "labels": labels,
                         "fr": e.get("sentence"), "pictos": pictos})
        else:
            no_en += 1

    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(full, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"总条数: {total}")
    print(f"全可映射(英文 label): {len(full)} ({len(full)/total*100:.1f}%)")
    print(f"含无英文 keyword 图标的序列: {no_en}")
    print(f"写出: {OUT}")

if __name__ == "__main__":
    main()
