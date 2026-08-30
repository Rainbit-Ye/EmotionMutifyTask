#!/usr/bin/env python3
"""快速验证: aac_model_zh 对 propicto-eval 的英文 label 序列能否翻出合理中文。
挑若干含 unseen label(不在 mamba_vocab)的序列 + 几条全 seen 的, greedy 生成中文并打印。
"""
import json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "Meta-Llama-3-8B-Instruct"
SFT = ROOT / "EmotionClassify/AAC2Text/checkpoints/aac_model_zh"
DATA = ROOT / "EmotionClassify/AAC2Text/data/cleardata/propicto_eval_labels_en.json"
VOCAB = ROOT / "propicto_zh_pipeline/aac2text_web/data/mamba_vocab.json"

def main():
    item2idx = json.load(open(VOCAB, encoding="utf-8"))["item2idx"]
    known = set(item2idx.keys())
    data = json.load(open(DATA, encoding="utf-8"))

    print("加载 tokenizer/base/SFT-LoRA ...")
    tok = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(model, str(SFT))
    model.eval()

    # 挑样本: 含 unseen label 的优先
    unseen_samples, seen_samples = [], []
    for d in data:
        has_unseen = any(l not in known for l in d["labels"])
        (unseen_samples if has_unseen else seen_samples).append(d)
    pick = unseen_samples[:6] + seen_samples[:3]
    print(f"(unseen 序列共 {len(unseen_samples)}, seen 共 {len(seen_samples)})\n")

    for d in pick:
        labels = d["labels"]
        prompt = f"请把这些 AAC 图标序列翻译成一个简单的中文句子：{' '.join(labels)}"
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        zh = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        unseen = [l for l in labels if l not in known]
        print(f"[{'UNSEEN' if unseen else 'seen'}] labels: {labels}")
        if unseen: print(f"   unseen: {unseen}")
        print(f"   法文: {d['fr']}")
        print(f"    中文: {zh}\n")

if __name__ == "__main__":
    main()
