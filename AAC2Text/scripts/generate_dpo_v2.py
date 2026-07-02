#!/usr/bin/env python3
"""
DPO v2 偏好数据生成 — 基于 annotations_final_v2 的人工修正

与 v1 (dpo_pairs.json) 的关键区别:
- chosen: 人工手输的 zh_correction (v1 用的是 sentence_zh polished 版)
- rejected: 对【人工修改后】的 labels 重新跑 Llama(EN)+Qwen(ZH) 生成的翻译
  (v1 用的是 original_zh 改前翻译 或 sentence_zh polished 版)
- 输入一致: chosen 和 rejected 对应同一个 labels 序列, 差异只在翻译质量
  → DPO 信号纯净, 无长度捷径, 无图标序列差异

生成方法复刻 generate_training_data.py:
- 英文: Llama-3-8B, translation_prompt_{first/second/third} (icon → EN)
- 中文: Qwen2.5-1.5B, translation_en_to_zh (EN → ZH)

Usage:
    python generate_dpo_v2.py
    python generate_dpo_v2.py --gpu-en 1 --gpu-zh 2 --resume
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]  # /home/user1/liuduanye
ANNOT_PATH = ROOT / "EmotionClassify/AAC2Text/data/cleardata/annotations_final_v2_20260630_111636.json"
ONTOLOGY_PATH = ROOT / "EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json"
PROMPTS_PATH = ROOT / "EmotionClassify/AAC2Text/config/prompts.yaml"
LLAMA_PATH = str(ROOT / "Meta-Llama-3-8B-Instruct")
QWEN_PATH = str(ROOT / "qwen/Qwen2_5-1_5B-Instruct")
OUT_PATH = ROOT / "EmotionClassify/AAC2Text/data/cleardata/dpo_pairs_v2.json"

SUBJ_TO_EN_TEMPLATE = {
    "first": "translation_prompt_first",
    "second": "translation_prompt_second",
    "third": "translation_prompt_third",
}


def build_readable_names(ontology_path: Path) -> dict:
    """复刻 generate_training_data.py 的 readable_names 构建 (clean_id → 可读英文名)."""
    with open(ontology_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ontology = data.get("ontology", [])

    readable_names = {}
    for item in ontology:
        icon_id = item.get("icon_id", "")
        if not icon_id:
            continue
        clean_id = re.sub(r"_\d+[a-z]?$", "", icon_id)
        if clean_id in readable_names:
            continue
        if clean_id == "I":
            readable_names[clean_id] = "I"
        elif clean_id == "U":
            readable_names[clean_id] = "you"
        else:
            core = item.get("core_semantic", "").strip()
            label = re.sub(r"_\d+[a-z]?$", "", item.get("label", "")).replace("_", " ").strip()
            core_display = core.replace("_", " ") if core else ""
            if core_display and len(core_display.split()) >= 2:
                readable_names[clean_id] = core_display
            elif label and label != clean_id.replace("_", " ") and len(label.split()) >= 2:
                readable_names[clean_id] = label
            else:
                name = re.sub(r"_\d+[a-z]?$", "", clean_id)
                name = re.sub(r"_\d+(?=_)", "", name)
                name = name.replace("_,_to", "").replace("_to", "")
                name = name.replace("_", " ")
                name = re.sub(r"\s+", " ", name).strip()
                readable_names[clean_id] = name
    return readable_names


def clean_symbol(symbol: str, readable_names: dict) -> str:
    """复刻 clean_symbol: label (可能带数字后缀) → 可读名."""
    clean_id = re.sub(r"_\d+[a-z]?$", "", symbol)
    if clean_id in readable_names:
        return readable_names[clean_id]
    # fallback
    name = re.sub(r"_\d+[a-z]?$", "", symbol)
    name = re.sub(r"_\d+(?=_)", "", name)
    name = name.replace("_,_to", "").replace("_to", "")
    name = name.replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def gen(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    """单条生成 (greedy, 复刻 _generate_single)."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def translate_labels_to_en(labels, subject_type, readable_names, prompts, en_model, en_tok):
    """图标序列 → 英文句子 (Llama)."""
    clean_labels = [clean_symbol(l, readable_names) for l in labels]
    template_key = SUBJ_TO_EN_TEMPLATE.get(subject_type, "translation_prompt_third")
    prompt = prompts[template_key].format(labels=clean_labels)
    resp = gen(en_model, en_tok, prompt, max_new_tokens=80)

    if "REJECT" in resp.upper():
        return ""
    m = re.search(r"[Ss]entence:\s*(.+?)(?:\n|$)", resp)
    if m:
        sent = m.group(1).strip().strip("\"'")
    else:
        sent = resp.strip().split("\n")[0].strip().strip("\"'")
    return sent if len(sent) > 5 else ""


def translate_en_to_zh(sentence_en, prompts, zh_model, zh_tok):
    """英文句子 → 中文 (Qwen)."""
    template = prompts.get("translation_en_to_zh", "")
    if not template:
        return sentence_en
    prompt = template.format(sentence_en=sentence_en)
    resp = gen(zh_model, zh_tok, prompt, max_new_tokens=80)
    sent_zh = resp.strip().split("\n")[0].strip()
    for ch in ['"', "'", "\u201c", "\u201d"]:
        sent_zh = sent_zh.strip(ch)
    for prefix in ["中文：", "中文:", "翻译：", "翻译:"]:
        if sent_zh.startswith(prefix):
            sent_zh = sent_zh[len(prefix):].strip()
    return sent_zh if sent_zh else sentence_en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-en", type=str, default="1", help="Llama GPU id")
    ap.add_argument("--gpu-zh", type=str, default="2", help="Qwen GPU id")
    ap.add_argument("--resume", action="store_true", help="断点续跑")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_en},{args.gpu_zh}"
    # 注意: 设置 CUDA_VISIBLE_DEVICES 后, GPU 重新编号为 0,1
    en_device = "cuda:0"  # 对应原 gpu-en
    zh_device = "cuda:1"  # 对应原 gpu-zh

    # 加载 prompts
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 加载本体 + readable_names
    print(f"[load] ontology: {ONTOLOGY_PATH}")
    readable_names = build_readable_names(ONTOLOGY_PATH)
    print(f"[load] {len(readable_names)} readable names")

    # 加载标注
    print(f"[load] annotations: {ANNOT_PATH}")
    with open(ANNOT_PATH, "r", encoding="utf-8") as f:
        ann = json.load(f)
    print(f"[load] {len(ann)} annotations total")

    # 过滤: is_valid=1 AND 有 zh_correction
    valid = [d for d in ann if d.get("is_valid") == 1]
    has_corr = [d for d in valid if d.get("zh_correction") and str(d["zh_correction"]).strip()]
    skipped_no_corr = len(valid) - len(has_corr)
    skipped_invalid = len(ann) - len(valid)
    print(f"[filter] valid={len(valid)}, has zh_correction={len(has_corr)}")
    print(f"[filter] skipped (is_valid=0): {skipped_invalid}")
    print(f"[filter] skipped (no zh_correction): {skipped_no_corr}")
    print(f"[filter] TOTAL skipped: {skipped_invalid + skipped_no_corr}")
    print(f"[filter] will process: {len(has_corr)}")

    # 断点续跑
    existing = {}
    if args.resume and os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for r in prev:
                if "item_id" in r and r.get("rejected"):
                    existing[r["item_id"]] = r
            print(f"[resume] 已加载 {len(existing)} 条已完成结果")
        except Exception as e:
            print(f"[resume] 加载失败: {e}")

    # 加载模型
    print(f"\n[model] loading Llama EN → {en_device}")
    t0 = time.time()
    en_tok = AutoTokenizer.from_pretrained(LLAMA_PATH, trust_remote_code=True)
    en_tok.pad_token = en_tok.eos_token
    en_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, torch_dtype=torch.float16, device_map={"": en_device}, trust_remote_code=True
    )
    en_model.eval()
    print(f"[model] Llama loaded in {time.time()-t0:.1f}s")

    print(f"[model] loading Qwen ZH → {zh_device}")
    t0 = time.time()
    zh_tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    zh_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.float16, device_map={"": zh_device}, trust_remote_code=True
    )
    zh_model.eval()
    print(f"[model] Qwen loaded in {time.time()-t0:.1f}s")

    # 生成
    results = []
    skipped_resume = 0
    skipped_gen_fail = 0
    t_start = time.time()

    for i, item in enumerate(has_corr):
        item_id = item.get("item_id")

        # 断点续跑
        if args.resume and item_id in existing:
            results.append(existing[item_id])
            skipped_resume += 1
            continue

        labels = item.get("labels", [])
        subject_type = item.get("subject_type", "third")
        zh_correction = str(item["zh_correction"]).strip()
        deleted_labels = item.get("deleted_labels", [])

        t_item = time.time()

        # 生成 EN
        sent_en = translate_labels_to_en(labels, subject_type, readable_names, prompts, en_model, en_tok)
        if not sent_en:
            print(f"  [{i+1}/{len(has_corr)}] id={item_id} EN 生成失败, 跳过")
            skipped_gen_fail += 1
            continue

        # 生成 ZH
        sent_zh = translate_en_to_zh(sent_en, prompts, zh_model, zh_tok)
        if not sent_zh or sent_zh == sent_en:
            print(f"  [{i+1}/{len(has_corr)}] id={item_id} ZH 生成失败, 跳过")
            skipped_gen_fail += 1
            continue

        # 跳过 chosen==rejected (DPO 学不到东西)
        if zh_correction == sent_zh:
            print(f"  [{i+1}/{len(has_corr)}] id={item_id} chosen==rejected, 跳过")
            skipped_gen_fail += 1
            continue

        pair = {
            "labels": labels,
            "chosen": zh_correction,
            "rejected": sent_zh,
            "source": "v2_edit_derived_ai_rejected",
            "item_id": item_id,
            "deleted_labels": deleted_labels,
            # 追溯字段 (不影响 v1 schema 兼容)
            "generated_en": sent_en,
            "subject_type": subject_type,
            "original_zh": item.get("original_zh", ""),
            "sentence_zh": item.get("sentence_zh", ""),
        }
        results.append(pair)

        if (i + 1) % 50 == 0 or i == len(has_corr) - 1:
            done = len(results) - skipped_resume
            elapsed = time.time() - t_start
            rate = elapsed / max(done, 1)
            print(f"[{i+1}/{len(has_corr)}] id={item_id} | done={done} fail={skipped_gen_fail} "
                  f"| {rate:.1f}s/sample | ETA={(len(has_corr)-i-1)*rate/60:.1f}min")
            print(f"  labels: {labels}")
            print(f"  EN: {sent_en}")
            print(f"  rejected(AI): {sent_zh}")
            print(f"  chosen(人工): {zh_correction}")

        # checkpoint
        if (i + 1) % 100 == 0:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 最终保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"[done] 保存 {len(results)} 条到: {args.out}")
    print(f"[stats] 耗时 {elapsed/60:.1f}min")
    print(f"[stats] 断点续跑跳过: {skipped_resume}")
    print(f"[stats] 生成失败跳过: {skipped_gen_fail}")
    print(f"[stats] 本次新生成: {len(results) - skipped_resume}")
    print(f"[stats] 总有效输出: {len(results)}")
    print(f"[stats] 数据源 skipped (is_valid=0): {skipped_invalid}")
    print(f"[stats] 数据源 skipped (no zh_correction): {skipped_no_corr}")
    print("=" * 60)


if __name__ == "__main__":
    main()
