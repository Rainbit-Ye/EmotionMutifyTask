#!/usr/bin/env python3
"""
把 propicto_eval_labels_en.json 的 2999 条法文 sentence 翻译成中文, 作为大评测集 gold。
输出 data/cleardata/propicto_eval_zh.json: {orig_idx, labels, fr, reference_zh}
支持断点续跑 (--resume)。用 Llama-3-8B-Instruct 直接 FR->ZH。
"""
import json, os, argparse, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "Meta-Llama-3-8B-Instruct"
IN = ROOT / "EmotionClassify/AAC2Text/data/cleardata/propicto_eval_labels_en.json"
OUT = ROOT / "EmotionClassify/AAC2Text/data/cleardata/propicto_eval_zh.json"

PROMPT = "请把下面这句法文翻译成中文，只输出中文译文。\n法文：{fr}\n中文："

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data = json.load(open(IN, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="auto")
    model.eval()

    # 断点续跑
    done = {}
    if args.resume and os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            if r.get("reference_zh"):
                done[r["orig_idx"]] = r
        print(f"[resume] 已完成 {len(done)} 条")

    results = [done[i] for i in sorted(done)]
    t0 = time.time()
    n_new = 0
    for d in data:
        oi = d["orig_idx"]
        if oi in done:
            continue
        fr = d["fr"] or ""
        if not fr.strip():
            results.append({**d, "reference_zh": ""})
            continue
        text = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(fr=fr)}],
                                       tokenize=False, add_generation_prompt=True)
        inp = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=80, do_sample=False,
                                 repetition_penalty=1.2, no_repeat_ngram_size=4,
                                 pad_token_id=tok.eos_token_id)
        zh = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # 去掉可能的 "中文：" 前缀残留
        for p in ["中文：", "中文:", "翻译：", "翻译:"]:
            if zh.startswith(p):
                zh = zh[len(p):].strip()
        results.append({**d, "reference_zh": zh})
        n_new += 1
        if n_new % 100 == 0:
            json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            el = time.time() - t0
            print(f"[{n_new} new | {len(results)} total] {el:.0f}s | {fr[:40]} -> {zh[:40]}")

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] 共 {len(results)} 条, 新增 {n_new}, 用时 {(time.time()-t0)/60:.1f}min -> {OUT}")

if __name__ == "__main__":
    main()
