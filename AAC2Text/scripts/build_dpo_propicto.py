#!/usr/bin/env python3
"""
用 propicto-eval (法文 AAC) 构造 AAC2Text DPO 配对, 并顺带算出 SFT 基线指标。
  - 加载 SFT 模型 (aac_model_zh), 对 2999 条 labels 生成中文 -> rejected
  - chosen = 法文翻译的中文参考 (reference_zh)
  - 过滤 chosen==rejected (DPO 学不到)
  - 复用 test_zh.py 的生成/指标, 保证与 168 条评测可比
输出:
  data/cleardata/dpo_pairs_propicto.json  ({labels, chosen, rejected, source})
  data/cleardata/propicto_sft_preds.json  (SFT 预测, 供核对)
  checkpoints/eval_zh_propicto_sft.json   (SFT 基线 BLEU/BERTScore)
"""
import json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 复用 test_zh 的生成与指标 (完全一致)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_zh import generate_predictions, compute_bertscore, compute_bleu_zh

ROOT = Path(__file__).resolve().parents[3]
BASE = str(ROOT / "Meta-Llama-3-8B-Instruct")
SFT = str(ROOT / "EmotionClassify/AAC2Text/checkpoints/aac_model_zh")
IN = str(ROOT / "EmotionClassify/AAC2Text/data/cleardata/propicto_eval_zh.json")
OUT_PAIRS = str(ROOT / "EmotionClassify/AAC2Text/data/cleardata/dpo_pairs_propicto.json")
OUT_PREDS = str(ROOT / "EmotionClassify/AAC2Text/data/cleardata/propicto_sft_preds.json")
OUT_EVAL = str(ROOT / "EmotionClassify/AAC2Text/checkpoints/eval_zh_propicto_sft.json")

def main():
    data = json.load(open(IN, encoding="utf-8"))
    print(f"加载数据: {len(data)} 条")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, SFT)
    model.eval()

    preds = generate_predictions(model, tok, data, max_new_tokens=64, batch_size=8)
    refs = [d.get("reference_zh", "") for d in data]

    # 指标
    bert = compute_bertscore(preds, refs)
    bleu = compute_bleu_zh(preds, refs)
    print(f"SFT 基线: BERTScore-F1={bert['bertscore_f1']:.4f} BLEU={bleu['bleu']:.2f} chrF={bleu['chrf']:.2f}")

    # 构造 DPO 对
    pairs = []
    skip_eq = 0
    skip_empty = 0
    skip_foreign = 0
    import re
    foreign_re = re.compile(r"[A-Za-z]")  # 含未翻译外文词(如 Garage/insulin)的参考会教模型输出英文, 必须丢弃
    for d, p in zip(data, preds):
        chosen = d.get("reference_zh", "").strip()
        rejected = p.strip()
        if not chosen or not rejected:
            skip_empty += 1
            continue
        if foreign_re.search(chosen):
            skip_foreign += 1
            continue
        if chosen == rejected:
            skip_eq += 1
            continue
        pairs.append({
            "labels": d["labels"],
            "chosen": chosen,
            "rejected": rejected,
            "source": "propicto_fr_zh",
            "orig_idx": d.get("orig_idx"),
            "fr": d.get("fr"),
        })

    os.makedirs(os.path.dirname(OUT_PAIRS), exist_ok=True)
    json.dump(pairs, open(OUT_PAIRS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump([{**d, "sft_pred": p} for d, p in zip(data, preds)],
              open(OUT_PREDS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump([{"checkpoint": SFT, "num_samples": len(preds), "bertscore": bert, "bleu": bleu,
                "name": "SFT(propicto-eval)"}],
              open(OUT_EVAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"DPO 对: {len(pairs)} (跳过 含外文词 {skip_foreign}, chosen==rejected {skip_eq}, 空 {skip_empty})")
    print(f"写出: {OUT_PAIRS}")

if __name__ == "__main__":
    main()
