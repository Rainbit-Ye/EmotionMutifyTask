#!/usr/bin/env python3
"""
G-Eval 风格 LLM 评估脚本 — Reference-free 多维度语义评分

基于 Liu et al., NeurIPS 2023 (G-Eval) 思路:
- 不给 judge 看参考翻译(reference-free)
- 注入图标含义(core_semantic_zh + cs_role)
- 思维链(CoT):先解释再打分
- 5 个维度各打 1-5 分

评估维度:
1. 图标覆盖:每个图标含义是否被表达
2. 语义准确:翻译是否正确理解图标
3. 自然度:是否像中文母语者说的话
4. 无幻觉:是否添加图标中没有的信息
5. 整体质量:综合评分

Usage:
    CUDA_VISIBLE_DEVICES=1 python geval.py
    CUDA_VISIBLE_DEVICES=1 python geval.py --num-samples 50  # 只跑 50 条
"""
import argparse
import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/home/user1/liuduanye/Meta-Llama-3-8B-Instruct"
SFT = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_model_zh"
DPO_MID = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_dpo_zh_v2_mid"
TEST_DATA = "/home/user1/liuduanye/EmotionClassify/AAC2Text/data/cleardata/sft_val.json"
ONTOLOGY = "/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology_zh.json"
OUT_PATH = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/geval_results.json"

# === G-Eval Prompt (5 维度,reference-free,思维链) ===

GEVAL_PROMPT = """你是 AAC(辅助沟通)翻译质量评审。给定一组 AAC 图标序列(含每个图标的中文含义和语法角色)和一段候选中文翻译,请按 5 个维度评分(1-5 分)。

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
   - 1分:严重误译(如把"iron_to"译成"铁子")
   - 3分:基本准确,个别图标理解偏差
   - 5分:所有图标含义理解准确

3. 自然度(1-5):翻译是否像中文母语者会说的话(无翻译腔)
   - 1分:严重翻译腔("进行""使用""做出"等)
   - 3分:基本通顺,略有翻译腔
   - 5分:地道自然,像日常说话

4. 无幻觉(1-5):翻译是否添加了图标中没有的信息
   - 1分:大量凭空添加(人名、地名、物品等)
   - 3分:添加了少量合理推断(如量词、介词)
   - 5分:无任何凭空添加

5. 整体质量(1-5):综合上述维度的整体评价
   - 1分:差,需要重译
   - 3分:可用,但有改进空间
   - 5分:完美,无需修改

【输出格式】请严格按以下格式输出,每行一个维度,先给理由再给分数:
1. 图标覆盖: <理由> | 分数: <1-5>
2. 语义准确: <理由> | 分数: <1-5>
3. 自然度: <理由> | 分数: <1-5>
4. 无幻觉: <理由> | 分数: <1-5>
5. 整体质量: <理由> | 分数: <1-5>

【重要】不要看任何参考翻译,只根据图标含义评判。只输出上述 5 行,不要其他内容。"""


def load_ontology(path):
    """加载本体,按 icon_id(label) 索引"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("ontology", [])
    ont = {}
    for item in items:
        icon_id = item.get("icon_id", "")
        if icon_id:
            ont[icon_id] = item
    return ont


def build_icon_hints(labels, ontology):
    """构造图标提示串:每个 icon 的 label / 中文语义 / CS角色"""
    lines = []
    for i, lab in enumerate(labels, 1):
        info = ontology.get(lab, {})
        label_zh = info.get("label_zh", lab)
        core_zh = info.get("core_semantic_zh", "")
        cs_role = info.get("cs_role", "?")
        lines.append(f"  {i}. label={lab} | 中文含义={label_zh}/{core_zh} | CS角色={cs_role}")
    return "\n".join(lines)


def gen_translation(model, tokenizer, labels):
    """生成中文翻译"""
    prompt_text = f"请把这些 AAC 图标序列翻译成一个简单的中文句子：{' '.join(labels)}"
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    r = tokenizer.decode(out[0][inputs.input_ids[0].shape[0]:], skip_special_tokens=True).strip()
    # 取第一个句末标点前的内容
    for p in ["。", "？", "！", "\n"]:
        i = r.find(p)
        if i != -1:
            r = r[: i + len(p)]
            break
    return r


def parse_geval_scores(text):
    """解析 G-Eval 输出,提取 5 个维度的分数"""
    scores = {}
    dims = ["图标覆盖", "语义准确", "自然度", "无幻觉", "整体质量"]
    reasons = {}

    for dim in dims:
        # 匹配 "维度: 理由 | 分数: X" 或 "维度:分数:X"
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


def geval_one(judge_model, judge_tok, labels, candidate, ontology):
    """对单条翻译跑 G-Eval"""
    icon_hints = build_icon_hints(labels, ontology)
    prompt = GEVAL_PROMPT.format(icon_hints=icon_hints, candidate=candidate)
    messages = [{"role": "user", "content": prompt}]
    text = judge_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tok([text], return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=300,  # CoT 需要更长
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )
    response = judge_tok.decode(out[0][inputs.input_ids[0].shape[0]:], skip_special_tokens=True).strip()
    scores, reasons = parse_geval_scores(response)
    return scores, reasons, response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=None, help="评估样本数(默认全部 168)")
    ap.add_argument("--out", type=str, default=OUT_PATH)
    args = ap.parse_args()

    # 加载数据
    with open(TEST_DATA) as f:
        test_data = json.load(f)
    if args.num_samples:
        test_data = test_data[: args.num_samples]
    print(f"测试样本: {len(test_data)}")

    ontology = load_ontology(ONTOLOGY)
    print(f"本体图标: {len(ontology)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 SFT (合并)
    print("加载 base + SFT...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    base_model = PeftModel.from_pretrained(base_model, SFT).merge_and_unload()
    base_model.eval()

    # 包装 DPO
    print("加载 DPO-mid adapter...")
    model_dpo = PeftModel.from_pretrained(base_model, DPO_MID)
    model_dpo.eval()

    # 生成 SFT 和 DPO 翻译
    print("\n生成翻译...")
    translations = []
    for item in tqdm(test_data, desc="翻译"):
        labels = item["labels"]
        with model_dpo.disable_adapter():
            sft_pred = gen_translation(model_dpo, tokenizer, labels)
        dpo_pred = gen_translation(model_dpo, tokenizer, labels)
        translations.append({"labels": labels, "sft": sft_pred, "dpo": dpo_pred})

    # 释放生成模型,加载 judge
    del model_dpo, base_model
    torch.cuda.empty_cache()

    print("加载 judge 模型 (Llama-3-8B)...")
    judge_model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    judge_model.eval()
    judge_tok = tokenizer

    # G-Eval 评估
    print("\n开始 G-Eval 评估...")
    DIMS = ["图标覆盖", "语义准确", "自然度", "无幻觉", "整体质量"]
    results = []

    # 累计分数
    sft_scores_sum = {d: 0 for d in DIMS}
    dpo_scores_sum = {d: 0 for d in DIMS}
    sft_count = {d: 0 for d in DIMS}
    dpo_count = {d: 0 for d in DIMS}

    for i, (item, trans) in enumerate(tqdm(zip(test_data, translations), total=len(test_data), desc="G-Eval")):
        labels = item["labels"]

        # 评估 SFT 翻译
        sft_scores, sft_reasons, sft_raw = geval_one(
            judge_model, judge_tok, labels, trans["sft"], ontology
        )
        # 评估 DPO 翻译
        dpo_scores, dpo_reasons, dpo_raw = geval_one(
            judge_model, judge_tok, labels, trans["dpo"], ontology
        )

        # 累计
        for d in DIMS:
            if d in sft_scores:
                sft_scores_sum[d] += sft_scores[d]
                sft_count[d] += 1
            if d in dpo_scores:
                dpo_scores_sum[d] += dpo_scores[d]
                dpo_count[d] += 1

        results.append({
            "i": i,
            "labels": " ".join(labels),
            "ref": item["target_zh"],
            "sft_pred": trans["sft"],
            "dpo_pred": trans["dpo"],
            "sft_scores": sft_scores,
            "dpo_scores": dpo_scores,
            "sft_reasons": sft_reasons,
            "dpo_reasons": dpo_reasons,
        })

        if (i + 1) % 20 == 0:
            print(f"\n[{i+1}/{len(test_data)}]")
            print(f"  labels: {' '.join(labels)}")
            print(f"  SFT: {trans['sft']}")
            print(f"  DPO: {trans['dpo']}")
            print(f"  SFT 分数: {sft_scores}")
            print(f"  DPO 分数: {dpo_scores}")

    # 计算平均分
    print("\n" + "=" * 70)
    print("G-Eval 评估结果(Reference-free,5 维度平均分)")
    print("=" * 70)
    print(f"\n{'维度':<12} {'SFT':<10} {'DPO-mid':<10} {'差异':<10} {'DPO 优势':<10}")
    print("-" * 55)

    summary = {}
    for d in DIMS:
        sft_avg = sft_scores_sum[d] / max(sft_count[d], 1)
        dpo_avg = dpo_scores_sum[d] / max(dpo_count[d], 1)
        diff = dpo_avg - sft_avg
        winner = "DPO ↑" if diff > 0.05 else ("SFT ↓" if diff < -0.05 else "持平")
        print(f"{d:<12} {sft_avg:<10.2f} {dpo_avg:<10.2f} {diff:<+10.2f} {winner:<10}")
        summary[d] = {"sft_avg": sft_avg, "dpo_avg": dpo_avg, "diff": diff}

    # 整体对比
    sft_overall = summary["整体质量"]["sft_avg"]
    dpo_overall = summary["整体质量"]["dpo_avg"]
    print(f"\n{'整体质量对比':<20} SFT={sft_overall:.2f}  DPO={dpo_overall:.2f}  差异={dpo_overall-sft_overall:+.2f}")

    # 保存
    output = {
        "summary": summary,
        "num_samples": len(test_data),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果保存到: {args.out}")

    # 打印部分样例(每个维度 DPO 胜出的)
    print("\n=== DPO 整体质量胜出的样例(前 5 条)===")
    dpo_wins = [r for r in results if r["dpo_scores"].get("整体质量", 0) > r["sft_scores"].get("整体质量", 0)]
    for r in dpo_wins[:5]:
        print(f"\n[{r['i']}] labels: {r['labels']}")
        print(f"  ref:    {r['ref']}")
        print(f"  SFT:    {r['sft_pred']}  分数: {r['sft_scores']}")
        print(f"  DPO:    {r['dpo_pred']}  分数: {r['dpo_scores']}")

    print("\n=== SFT 整体质量胜出的样例(前 5 条)===")
    sft_wins = [r for r in results if r["sft_scores"].get("整体质量", 0) > r["dpo_scores"].get("整体质量", 0)]
    for r in sft_wins[:5]:
        print(f"\n[{r['i']}] labels: {r['labels']}")
        print(f"  ref:    {r['ref']}")
        print(f"  SFT:    {r['sft_pred']}  分数: {r['sft_scores']}")
        print(f"  DPO:    {r['dpo_pred']}  分数: {r['dpo_scores']}")


if __name__ == "__main__":
    main()
