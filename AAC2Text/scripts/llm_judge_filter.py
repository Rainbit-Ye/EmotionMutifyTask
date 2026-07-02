"""
LLM-as-judge 过滤润色后样本
============================
基于 Cherry Filtering (Li et al., NAACL 2024, arxiv 2308.12032) 思路改进:
- 不只打分, 还要结构化诊断: 幻觉/遗漏/翻译腔/人称漂移/句式漂移
- LLM 自审 LLM 输出, 给出 pass/fail + 原因
- pass 的进入 SFT 高质量子集; fail 的进入 invalid_examples

输入: polish_baseline_full.json (润色后)
输出:
  - polish_judged.json (含 judge 字段)
  - polish_passed.json (通过质检的子集)
  - polish_failed.json (被剔除的子集, 含原因)
  - judge_report.md (统计报告)

Usage:
    python llm_judge_filter.py --gpu 1 --in polish_baseline_full.json
    python llm_judge_filter.py --gpu 1 --resume
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
IN_PATH = ROOT / "EmotionClassify/AAC2Text/data/cleardata/polish_baseline_full.json"
ONTOLOGY_ZH = ROOT / "EmotionClassify/AAC2Text/data/processed/aac_full_ontology_zh.json"
LLAMA_PATH = str(ROOT / "Meta-Llama-3-8B-Instruct")
OUT_DIR = ROOT / "EmotionClassify/AAC2Text/data/cleardata"


JUDGE_PROMPT = """You are an AAC data quality inspector. Given an icon sequence and polished Chinese/English sentences, judge whether the polished version is acceptable.

[Icon Sequence]
{icon_hints}

[Original ZH] {original_zh}
[Polished ZH] {polished_zh}
[Polished EN] {polished_en}
[Subject Type] {subject_type}

[Quality Dimensions] Check each:
1. **Hallucination**: Does the polished sentence add info not in the icons (names/places/objects/numbers)?
2. **Omission**: Does it drop any icon's core semantics?
3. **Translation calque**: Does it still contain stilted words like 使用/进行/制造/做出/享用/装置?
4. **Icon misread**: Is any icon literally mistranslated (e.g. iron_to → "铁子", Halloween → "博士节")?
5. **Subject drift**: Does the subject match subject_type? (first=我/I, second=你/you, third=he/she or specific person)
6. **Form drift**: If original is a question, is the polished still a question? If negative, still negative?
7. **Country fabrication**: Does it add a country name not in the labels?
8. **Echo leak**: Does the polished output contain prompt echo (e.g. "根据给定的", "Based on the icon sequence")?

[Scoring]
- 5 = perfect, natural and faithful to icons
- 4 = good, minor issues
- 3 = acceptable, has obvious issues
- 2 = poor, has serious hallucination/mistranslation/calque
- 1 = totally wrong
- 0 = output is prompt echo / empty / corrupted

[Output Format] Output ONLY a single JSON object on one line, nothing else. No preamble, no explanation, no code fence. Example:
{{"score": 5, "pass": true, "reasons": ""}}
{{"score": 2, "pass": false, "reasons": "Hallucination:added 'Tuvalu';Calque:'使用' not fixed"}}

Now output the JSON for this sample:"""


def load_ontology(path):
    ont = json.load(open(path))
    items = ont["ontology"] if isinstance(ont, dict) and "ontology" in ont else ont
    return {it["label"]: it for it in items}


def icon_hints_str(labels, ont_by_label):
    lines = []
    for i, lab in enumerate(labels, 1):
        info = ont_by_label.get(lab, {})
        label_zh = info.get("label_zh", lab)
        core_zh = info.get("core_semantic_zh", "")
        cs_role = info.get("cs_role", "?")
        lines.append(f"  {i}. label={lab} | 中文语义={label_zh}/{core_zh} | CS角色={cs_role}")
    return "\n".join(lines)


def gen(model, tokenizer, prompt, max_new_tokens=200):
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


def parse_judge_output(text):
    """解析 LLM judge 输出, 容错"""
    if not text:
        return {"score": 0, "pass": False, "reasons": "空输出"}
    # 1) 去掉 code fence
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # 2) 找最后一个 JSON 对象 (LLM 可能先输出解释再输出 JSON)
    matches = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
    if matches:
        # 优先找含 "score" 的
        for m in reversed(matches):
            if '"score"' in m:
                try:
                    obj = json.loads(m)
                    obj["score"] = int(obj.get("score", 0))
                    obj["pass"] = bool(obj.get("pass", obj["score"] >= 4))
                    obj["reasons"] = str(obj.get("reasons", ""))
                    return obj
                except Exception:
                    continue
    # 3) 尝试宽松匹配 score
    sm = re.search(r'"score"\s*:\s*(\d+)', text)
    pm = re.search(r'"pass"\s*:\s*(true|false)', text, re.IGNORECASE)
    rm = re.search(r'"reasons"\s*:\s*"([^"]*)"', text)
    if sm:
        return {
            "score": int(sm.group(1)),
            "pass": (pm.group(1).lower() == "true") if pm else int(sm.group(1)) >= 4,
            "reasons": rm.group(1) if rm else "",
        }
    return {"score": 0, "pass": False, "reasons": f"解析失败: {text[:120]}"}


def judge_item(item, ont, model, tokenizer):
    prompt = JUDGE_PROMPT.format(
        icon_hints=icon_hints_str(item["labels"], ont),
        original_zh=item.get("original_zh", ""),
        polished_zh=item.get("baseline_zh", ""),
        polished_en=item.get("baseline_en", ""),
        subject_type=item.get("subject_type", ""),
    )
    out = gen(model, tokenizer, prompt, max_new_tokens=200)
    return parse_judge_output(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--in", dest="in_path", default=str(IN_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--model", default=LLAMA_PATH)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threshold", type=int, default=4, help="pass 阈值 score>=")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    in_path = args.in_path
    out_dir = Path(args.out_dir)
    judged_path = out_dir / "polish_judged.json"
    passed_path = out_dir / "polish_passed.json"
    failed_path = out_dir / "polish_failed.json"
    report_path = out_dir / "judge_report.md"

    print(f"[load] {in_path}")
    data = json.load(open(in_path))
    print(f"[load] {len(data)} samples")

    print(f"[load] ontology from {ONTOLOGY_ZH}")
    ont = load_ontology(ONTOLOGY_ZH)
    print(f"[load] {len(ont)} icons")

    # 断点续跑
    existing = {}
    if args.resume and judged_path.exists():
        try:
            prev = json.load(open(judged_path))
            for r in prev:
                if "item_id" in r and "judge" in r:
                    existing[r["item_id"]] = r
            print(f"[resume] 加载 {len(existing)} 条已判样本")
        except Exception as e:
            print(f"[resume] 失败: {e}")

    print(f"[model] loading {args.model} on GPU {args.gpu}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    print(f"[model] loaded in {time.time()-t0:.1f}s")

    results = []
    t_start = time.time()
    for i, item in enumerate(data):
        # 断点续跑
        if args.resume and item.get("item_id") in existing:
            results.append(existing[item["item_id"]])
            continue

        t_item = time.time()
        judge = judge_item(item, ont, model, tokenizer)
        out = dict(item)
        out["judge"] = judge
        results.append(out)

        if (i + 1) % 50 == 0:
            passed = sum(1 for r in results if r.get("judge", {}).get("pass"))
            print(f"[{i+1}/{len(data)}] pass={passed}/{len(results)} ({passed/max(len(results),1)*100:.1f}%)")
        if (i + 1) % 100 == 0:
            with open(judged_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[done] {len(results)} judged in {elapsed:.1f}s")

    # 保存
    with open(judged_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[save] {judged_path}")

    # 拆分 pass / fail
    passed = [r for r in results if r.get("judge", {}).get("pass")]
    failed = [r for r in results if not r.get("judge", {}).get("pass")]
    with open(passed_path, "w") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)
    with open(failed_path, "w") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)
    print(f"[save] {passed_path} ({len(passed)} 条)")
    print(f"[save] {failed_path} ({len(failed)} 条)")

    # 报告
    from collections import Counter
    score_dist = Counter(r.get("judge", {}).get("score", 0) for r in results)
    pass_rate = len(passed) / max(len(results), 1) * 100

    # 失败原因聚类
    reason_keywords = Counter()
    for r in failed:
        reasons = r.get("judge", {}).get("reasons", "")
        for kw in ["幻觉", "遗漏", "翻译腔", "误译", "人称漂移", "句式漂移", "国家名", "字面直译"]:
            if kw in reasons:
                reason_keywords[kw] += 1

    with open(report_path, "w") as f:
        f.write("# LLM-as-Judge 质检报告\n\n")
        f.write(f"**输入**: {in_path}\n")
        f.write(f"**总样本**: {len(results)}\n")
        f.write(f"**通过 (score>={args.threshold})**: {len(passed)} ({pass_rate:.1f}%)\n")
        f.write(f"**剔除**: {len(failed)} ({100-pass_rate:.1f}%)\n\n")
        f.write("## 分数分布\n\n")
        f.write("| score | 数量 | 占比 |\n|-------|------|------|\n")
        for s in sorted(score_dist.keys(), reverse=True):
            c = score_dist[s]
            f.write(f"| {s} | {c} | {c/len(results)*100:.1f}% |\n")
        f.write("\n## 失败原因聚类\n\n")
        f.write("| 原因 | 数量 |\n|------|------|\n")
        for kw, c in reason_keywords.most_common():
            f.write(f"| {kw} | {c} |\n")
    print(f"[save] {report_path}")
    print(f"\n[summary] pass_rate={pass_rate:.1f}%, score_dist={dict(score_dist)}")


if __name__ == "__main__":
    main()
