#!/usr/bin/env python3
"""
多模态 G-Eval 评估 — llava 看图标图片生成视觉描述 + 本地 Llama-3-8B 评估

方案3 改进版:
1. llava(多模态):看每个图标图片,生成视觉描述(10字以内)
2. Llama-3-8B(本地文本 LLM):基于图标视觉描述 + 翻译,按 5 维度打分

优势:
- llava 提供视觉语义(不依赖文本本体)
- Llama-3-8B 本地推理快(已验证可用),中文评估稳定
"""
import base64
import json
import os
import re
import requests
import time
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

API = "http://172.31.226.24:4433"
ICON_DIR = "/home/user1/liuduanye/AACTest/AAC/data/images"
TEST_DATA = "/home/user1/liuduanye/EmotionClassify/AAC2Text/data/cleardata/sft_val.json"
EXISTING_GEVAL = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/geval_results.json"
OUT_PATH = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/geval_multimodal_results.json"

BASE = "/home/user1/liuduanye/Meta-Llama-3-8B-Instruct"
SFT = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_model_zh"
DPO_MID = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_dpo_zh_v2_mid"


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_icon_description(label, icon_dir):
    """用 llava 看图标图片,生成中文视觉描述"""
    candidates = [
        os.path.join(icon_dir, f"{label}.png"),
        os.path.join(icon_dir, f"{label}.jpg"),
    ]
    img_path = None
    for c in candidates:
        if os.path.exists(c):
            img_path = c
            break
    if not img_path:
        return f"(图标 {label} 图片未找到)"

    img_b64 = encode_image(img_path)
    prompt = f"这个 AAC 辅助沟通图标叫 '{label}',请用一句简短的中文描述这个图标画的什么(10字以内)。只输出描述,不要其他内容。"
    try:
        resp = requests.post(f"{API}/api/generate", json={
            "model": "llava:latest",
            "prompt": prompt,
            "stream": False,
            "images": [img_b64],
        }, timeout=60)
        desc = resp.json().get("response", "").strip()
        desc = desc.split("\n")[0].strip().strip("\"'。")
        return desc if len(desc) < 50 else desc[:50]
    except Exception as e:
        return f"(描述失败: {e})"


def build_multimodal_hints(labels, icon_dir, cache):
    """为每个图标生成视觉描述(带缓存)"""
    hints = []
    for i, lab in enumerate(labels, 1):
        if lab not in cache:
            cache[lab] = get_icon_description(lab, icon_dir)
            time.sleep(0.2)  # 避免 API 过载
        hints.append(f"  {i}. label={lab} | 图标画面: {cache[lab]}")
    return "\n".join(hints)


GEVAL_PROMPT = """你是 AAC(辅助沟通)翻译质量评审。给定一组 AAC 图标序列(含每个图标的画面描述)和一段候选中文翻译,请按 5 个维度评分(1-5 分)。

【图标序列】
{icon_hints}

【候选翻译】
{candidate}

【评分维度与标准】
1. 图标覆盖(1-5):每个图标的含义是否在翻译中被表达
   - 1分:大量图标含义遗漏
   - 3分:主要图标覆盖,部分遗漏
   - 5分:所有图标含义均被表达

2. 语义准确(1-5):翻译是否正确理解了每个图标的真实含义(不是字面直译)
   - 1分:严重误译
   - 3分:基本准确,个别偏差
   - 5分:完全准确

3. 自然度(1-5):翻译是否像中文母语者会说的话
   - 1分:严重翻译腔
   - 3分:基本通顺,略有翻译腔
   - 5分:地道自然

4. 无幻觉(1-5):翻译是否添加了图标中没有的信息
   - 1分:大量凭空添加
   - 3分:少量合理推断
   - 5分:无凭空添加

5. 整体质量(1-5):综合评价
   - 1分:差
   - 3分:可用但有改进空间
   - 5分:完美

【输出格式】请严格按以下格式输出,每行一个维度,先给理由再给分数:
1. 图标覆盖: <理由> | 分数: <1-5>
2. 语义准确: <理由> | 分数: <1-5>
3. 自然度: <理由> | 分数: <1-5>
4. 无幻觉: <理由> | 分数: <1-5>
5. 整体质量: <理由> | 分数: <1-5>

【重要】不要看任何参考翻译,只根据图标画面描述评判。只输出上述 5 行,不要其他内容。"""


def parse_scores(text):
    scores = {}
    reasons = {}
    dims = ["图标覆盖", "语义准确", "自然度", "无幻觉", "整体质量"]
    for dim in dims:
        patterns = [
            rf"{dim}[^|]*?[:：]\s*(.*?)\s*\|\s*分数[:：]\s*(\d)",
            rf"{dim}.*?分数[:：]\s*(\d)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                if m.lastindex == 2:
                    reasons[dim] = m.group(1).strip()
                    scores[dim] = int(m.group(2))
                else:
                    scores[dim] = int(m.group(1))
                break
    return scores, reasons


def geval_llama(judge_model, judge_tok, labels, candidate, icon_hints):
    """用本地 Llama-3-8B 评估"""
    prompt = GEVAL_PROMPT.format(icon_hints=icon_hints, candidate=candidate)
    messages = [{"role": "user", "content": prompt}]
    text = judge_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tok([text], return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )
    response = judge_tok.decode(out[0][inputs.input_ids[0].shape[0]:], skip_special_tokens=True).strip()
    scores, reasons = parse_scores(response)
    return scores, reasons, response


def main():
    # 复用已有翻译
    with open(EXISTING_GEVAL) as f:
        existing = json.load(f)
    print(f"复用已有翻译: {len(existing['results'])} 条")

    with open(TEST_DATA) as f:
        test_data = json.load(f)

    # Step 1: 用 llava 生成所有图标的视觉描述(带缓存)
    icon_cache_path = "/tmp/icon_descriptions_cache.json"
    if os.path.exists(icon_cache_path):
        with open(icon_cache_path) as f:
            icon_cache = json.load(f)
        print(f"加载图标描述缓存: {len(icon_cache)} 个")
    else:
        icon_cache = {}

    # 收集所有需要的图标
    all_labels = set()
    for item in test_data:
        all_labels.update(item["labels"])
    print(f"需要描述的图标: {len(all_labels)} 个")

    # 用 llava 生成缺失的描述(多线程加速)
    missing = [l for l in all_labels if l not in icon_cache]
    if missing:
        print(f"用 llava 生成 {len(missing)} 个图标描述...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def fetch_desc(label):
            return label, get_icon_description(label, ICON_DIR)
        with ThreadPoolExecutor(max_workers=4) as exe:
            futures = [exe.submit(fetch_desc, l) for l in missing]
            for f in tqdm(as_completed(futures), total=len(futures), desc="llava看图"):
                lab, desc = f.result()
                icon_cache[lab] = desc
        # 保存缓存
        with open(icon_cache_path, "w", encoding="utf-8") as f:
            json.dump(icon_cache, f, ensure_ascii=False, indent=2)
        print(f"图标描述缓存保存到: {icon_cache_path}")

    # Step 2: 加载本地 Llama-3-8B judge
    print("\n加载 Llama-3-8B judge...")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    judge_model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    judge_model.eval()

    # Step 3: G-Eval 评估
    DIMS = ["图标覆盖", "语义准确", "自然度", "无幻觉", "整体质量"]
    results = []
    sft_sum = {d: 0 for d in DIMS}
    dpo_sum = {d: 0 for d in DIMS}
    sft_cnt = {d: 0 for d in DIMS}
    dpo_cnt = {d: 0 for d in DIMS}

    print("\n开始多模态 G-Eval 评估(llava 视觉描述 + Llama-3-8B 评估)...")

    for i, (item, ex) in enumerate(tqdm(zip(test_data, existing["results"]), total=len(test_data), desc="G-Eval")):
        labels = item["labels"]
        sft_pred = ex["sft_pred"]
        dpo_pred = ex["dpo_pred"]

        icon_hints = build_multimodal_hints(labels, ICON_DIR, icon_cache)

        # 评估 SFT
        sft_scores, sft_reasons, _ = geval_llama(judge_model, tokenizer, labels, sft_pred, icon_hints)
        # 评估 DPO
        dpo_scores, dpo_reasons, _ = geval_llama(judge_model, tokenizer, labels, dpo_pred, icon_hints)

        for d in DIMS:
            if d in sft_scores:
                sft_sum[d] += sft_scores[d]
                sft_cnt[d] += 1
            if d in dpo_scores:
                dpo_sum[d] += dpo_scores[d]
                dpo_cnt[d] += 1

        results.append({
            "i": i,
            "labels": " ".join(labels),
            "ref": item["target_zh"],
            "sft_pred": sft_pred,
            "dpo_pred": dpo_pred,
            "icon_descriptions": {lab: icon_cache.get(lab, "") for lab in labels},
            "sft_scores": sft_scores,
            "dpo_scores": dpo_scores,
            "sft_reasons": sft_reasons,
            "dpo_reasons": dpo_reasons,
        })

        if (i + 1) % 20 == 0:
            print(f"\n[{i+1}/{len(test_data)}]")
            print(f"  labels: {' '.join(labels)}")
            print(f"  图标: {icon_hints}")
            print(f"  SFT: {sft_pred} → {sft_scores}")
            print(f"  DPO: {dpo_pred} → {dpo_scores}")

        # 每 20 条保存
        if (i + 1) % 20 == 0:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    # 汇总
    print("\n" + "=" * 70)
    print("多模态 G-Eval 评估结果(llava 视觉描述 + Llama-3-8B 评估)")
    print("=" * 70)
    print(f"\n{'维度':<12} {'SFT':<10} {'DPO-mid':<10} {'差异':<10} {'胜出':<10}")
    print("-" * 55)

    summary = {}
    for d in DIMS:
        sft_avg = sft_sum[d] / max(sft_cnt[d], 1)
        dpo_avg = dpo_sum[d] / max(dpo_cnt[d], 1)
        diff = dpo_avg - sft_avg
        winner = "DPO ↑" if diff > 0.05 else ("SFT ↓" if diff < -0.05 else "持平")
        print(f"{d:<12} {sft_avg:<10.2f} {dpo_avg:<10.2f} {diff:<+10.2f} {winner:<10}")
        summary[d] = {"sft_avg": sft_avg, "dpo_avg": dpo_avg, "diff": diff}

    output = {
        "summary": summary,
        "num_samples": len(test_data),
        "results": results,
        "icon_cache": icon_cache,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果保存到: {OUT_PATH}")

    # DPO 胜出样例
    print("\n=== DPO 整体质量胜出样例(前 5 条)===")
    dpo_wins = [r for r in results if r["dpo_scores"].get("整体质量", 0) > r["sft_scores"].get("整体质量", 0)]
    for r in dpo_wins[:5]:
        print(f"\n[{r['i']}] labels: {r['labels']}")
        print(f"  图标: {r['icon_descriptions']}")
        print(f"  SFT: {r['sft_pred']}  分数: {r['sft_scores']}")
        print(f"  DPO: {r['dpo_pred']}  分数: {r['dpo_scores']}")

    print("\n=== SFT 整体质量胜出样例(前 5 条)===")
    sft_wins = [r for r in results if r["sft_scores"].get("整体质量", 0) > r["dpo_scores"].get("整体质量", 0)]
    for r in sft_wins[:5]:
        print(f"\n[{r['i']}] labels: {r['labels']}")
        print(f"  图标: {r['icon_descriptions']}")
        print(f"  SFT: {r['sft_pred']}  分数: {r['sft_scores']}")
        print(f"  DPO: {r['dpo_pred']}  分数: {r['dpo_scores']}")


if __name__ == "__main__":
    main()
