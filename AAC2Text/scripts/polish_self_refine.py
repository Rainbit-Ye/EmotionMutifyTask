"""
Self-Refine 润色验证脚本
========================
基于 Madaan et al., Self-Refine, NeurIPS 2023 (arxiv 2303.17651)

对照实验:
- Baseline: 一次性润色（仅注入 icon 语义表）
- Self-Refine: 初始润色 → 自我批判 → 改写 (K=2 轮)

关键设计:
- 必须注入 icon 语义表 (core_semantic_zh + cs_role)，否则 LLM 继续把 sad_man 当人
- 双语润色: EN + ZH 分别跑（中文是主要痛点）
- 50 条随机抽样，固定 seed 保证可复现

Usage:
    python polish_self_refine.py --n 50 --K 2 --gpu 1
    python polish_self_refine.py --n 50 --K 2 --gpu 1 --out /tmp/polish_demo.json
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === 路径 ===
ROOT = Path(__file__).resolve().parents[3]  # /home/user1/liuduanye
VALID_DATA = ROOT / "EmotionClassify/AAC2Text/data/cleardata/firstchoose/valid_data.json"
ONTOLOGY_ZH = ROOT / "EmotionClassify/AAC2Text/data/processed/aac_full_ontology_zh.json"
LLAMA_PATH = str(ROOT / "Meta-Llama-3-8B-Instruct")

# === Prompt 模板 ===

# 人称代词映射 - 用于显式锁定人称
SUBJ_PRONOUN_EN = {
    "first": "I",
    "second": "you",
    "third": "he/she/they/it",
}
SUBJ_PRONOUN_ZH = {
    "first": "我",
    "second": "你",
    "third": "他/她/它",
}


def detect_subject_hint(item: dict, ont_by_label: dict) -> str:
    """检测主语提示：如果 labels 第一个是具体人物/角色，提示 LLM 用具体名而非'他/她'"""
    if item.get("subject_type") != "third":
        return ""
    if not item.get("labels"):
        return ""
    first_label = item["labels"][0]
    info = ont_by_label.get(first_label, {})
    cs_role = info.get("cs_role", "")
    sem_type = info.get("semantic_type", "")
    label_zh = info.get("label_zh", "")
    # WHO 角色或 person/relationship 类型 → 用具体名
    if cs_role == "WHO" or sem_type in ("person", "relationship"):
        if label_zh and label_zh != first_label:
            return f"（注：第一个图标是具体人物 '{label_zh}'，作主语时直接用该人物名，不要用'他/她'）"
    return ""

# 中文润色 - 初始生成
INITIAL_PROMPT_ZH = """你是 AAC（辅助沟通）语句润色专家。给定一组 AAC 图标序列和原始生成的中文句子，请润色成自然流畅的中文。

【图标序列】
{icon_hints}

【原始中文】{original_zh}
【原始英文】{original_en}
【句型模板】{type}
【人称】{subject_type}（必须用"{subj_zh}"作主语）{subj_hint}
【句式提示】{form_hint}

【润色规则】
1. 每个 icon 的【中文语义】列出了它的真实含义，必须按这个含义理解，不要字面直译 label
2. 【CS角色】标注该 icon 在句中的功能：WHO=主语/WHAT_DOING=动作/WHAT=宾语/WHERE=地点/WHEN=时间/HOW=方式或情绪
3. 情绪类 icon（CS角色=HOW）表达情绪状态，不是人物（如 sad_man → "难过/低落"，不是"可怜的人"）
4. flag_X / country_X 开头的 icon 表示地点/国家，不是手持国旗（如 flag_Tuvalu → "在图瓦卢"）；如果句中已有 country_X 则用该国家，**禁止凭空添加 labels 中没有的国家名**
5. 时间类 icon（CS角色=WHEN）作时间状语，不要直译（如 dinner_time → "晚饭时"，不是"一顿饭"）
6. 保持 icon 序列的核心语义不变，**不要添加 icon 中没有的信息（禁止凭空添加人名、地名、物品名、数字）**
7. 中文要像日常说话，可用省略、语气词（吧/呢/啊），避免翻译腔
8. **【人称锁定】** 句子主语必须用"{subj_zh}"，不得改成其他代词（如不得把"你"改成"他"）
9. **【句式保留】** 必须保留原句的句式：疑问句必须仍是疑问句（含"吗/呢/？"），否定句必须仍含否定词，陈述句不得改成疑问句
10. 避免翻译腔词汇：不要用"使用/进行/制造/做出/享用/装置/设备"等书面化词，改用口语化的"用/做/做出来/吃/东西"
11. **【禁止字面直译】** icon label 是英文标识符，不要按字面翻译（如 iron_to 是"熨烫"不是"铁子"，lead 是"铅"或"领导"看上下文）

【输出格式】仅输出一行润色后的中文句子。禁止输出解释、前缀、引号、"以下是"、"润色后"等任何元话语。"""

# 中文 - 反馈
FEEDBACK_PROMPT_ZH = """你是 AAC 语句质量评审。请对下面的中文润色句给出具体改进意见。

【图标序列】
{icon_hints}

【原始中文】{original_zh}
【润色后中文】{polished_zh}

【评审重点】
1. icon 语义是否被正确表达（对照【中文语义】和【CS角色】）
2. 是否有翻译腔、生硬直译
3. 是否添加了 icon 中没有的信息（幻觉）
4. 是否遗漏了某个 icon 的语义
5. 句式是否自然

【输出格式】列出 1-3 条具体问题，每条一行，不要泛泛而谈。如果没有问题，输出"无问题"。"""

# 中文 - 改写
REFINE_PROMPT_ZH = """根据反馈改进中文句子。

【图标序列】
{icon_hints}

【原始中文】{original_zh}
【当前润色】{polished_zh}
【反馈意见】{feedback}

【规则】保持 icon 核心语义，参考反馈改进，输出更自然的中文。

【输出格式】只输出一行改进后的中文句子，不要其他内容。"""

# 英文同理
INITIAL_PROMPT_EN = """You are an AAC (Augmentative Communication) sentence polishing expert. Given an AAC icon sequence and the original generated English sentence, polish it into natural English.

[Icon Sequence]
{icon_hints}

[Original EN] {original_en}
[Original ZH] {original_zh}
[Template] {type}
[Subject] {subject_type} (MUST use "{subj_en}" as the subject){subj_hint}
[Form Hint] {form_hint}

[Rules]
1. Each icon's [core_semantic] gives its true meaning; do not literally translate the label
2. [CS role]: WHO=subject / WHAT_DOING=verb / WHAT=object / WHERE=location / WHEN=time / HOW=manner or emotion
3. Emotion icons (CS role=HOW) express emotional state, not a person (e.g. sad_man → "feeling sad", not "the sad man")
4. flag_X / country_X icons indicate location/country, not holding a flag (e.g. flag_Tuvalu → "in Tuvalu"); if country_X is in labels, use that country; **do NOT add country names not in the labels**
5. Time icons (CS role=WHEN) act as adverbials (e.g. dinner_time → "at dinner", not "a meal time")
6. Keep core semantics of all icons; **do not add information not in the icons (no invented names, places, objects, numbers)**
7. Output natural English, like everyday speech
8. **[Subject Lock]** The subject MUST be "{subj_en}"; do not change to any other pronoun
9. **[Form Preservation]** Preserve the sentence form: questions must remain questions (with ?), negations must keep negation words, statements must not become questions

[Output Format] Output ONLY the polished English sentence. No explanations, no prefixes, no quotes, no meta language like "Here is", "Polished:". Just the sentence itself."""

FEEDBACK_PROMPT_EN = """You are an AAC sentence quality reviewer. Give specific improvement suggestions for the polished English sentence.

[Icon Sequence]
{icon_hints}

[Original EN] {original_en}
[Polished EN] {polished_en}

[Review Focus]
1. Whether icon semantics are correctly expressed (compare with [core_semantic] and [CS role])
2. Whether the sentence is fluent English, not translated-sounding
3. Whether information not in the icons was added (hallucination)
4. Whether any icon's semantics was omitted
5. Whether the sentence structure is natural

[Output Format] List 1-3 specific issues, one per line. If no issues, output "No issue"."""

REFINE_PROMPT_EN = """Refine the English sentence based on feedback.

[Icon Sequence]
{icon_hints}

[Original EN] {original_en}
[Current Polish] {polished_en}
[Feedback] {feedback}

[Rules] Keep core icon semantics, improve based on feedback, output more natural English.

[Output Format] Output only one refined English sentence, nothing else."""


def detect_form_hint(item: dict) -> str:
    """检测原句句式：疑问/否定/陈述，给出 prompt 提示"""
    en = item.get("sentence_en", "")
    zh = item.get("sentence_zh", "")
    # 检测 what/where/when/who/why/how 疑问词 或 中文的什么/哪里/谁/吗/？
    is_question_en = bool(re.search(r"\b(what|where|when|who|why|how)\b", en, re.IGNORECASE)) or en.rstrip().endswith("?")
    is_question_zh = ("？" in zh) or ("吗" in zh) or ("什么" in zh) or ("哪" in zh) or ("谁" in zh)
    is_question = is_question_en or is_question_zh
    # 检测否定
    is_neg_en = bool(re.search(r"\b(not|no|never|without|n't)\b", en, re.IGNORECASE))
    is_neg_zh = ("不" in zh) or ("没" in zh) or ("无" in zh) or ("非" in zh)
    is_neg = is_neg_en or is_neg_zh
    # 检测 type 中的 _emo
    is_emo = "_emo" in item.get("type", "")

    hints = []
    if is_question:
        hints.append("原句是疑问句，必须保留疑问形式（含疑问词和问号）")
    if is_neg:
        hints.append("原句含否定，必须保留否定词")
    if is_emo:
        hints.append("原句含情绪 icon，情绪应作为状态描述而非人物")
    if not hints:
        return "陈述句，正常润色"
    return "；".join(hints)


# === 后处理规则 ===
# 中文翻译腔替换规则（保守，只替换明确的翻译腔）
ZH_CALQUE_RULES = [
    # 使用 → 用
    (re.compile(r"使用"), "用"),
    # 进行 + 动词 → 动词直接
    (re.compile(r"进行(研究|学习|训练|讨论|分析|调查|检查|治疗|操作|工作|活动|扫描|比赛)"), r"\1"),
    (re.compile(r"正在(进行|从事)一场"), "正在进行"),
    # 制造 + 名词 → 具体动词
    (re.compile(r"制造噪音"), "发出噪音"),
    (re.compile(r"制造嘈杂的噪音"), "发出噪音"),
    (re.compile(r"制造(声响|声音)"), r"发出\1"),
    (re.compile(r"制造气泡"), "吹泡泡"),
    (re.compile(r"创造气泡"), "吹泡泡"),
    # 做出 + 名词 → 删除"做出"（保守：只删明显翻译腔，保留"做出决定/贡献"）
    (re.compile(r"做出(反应|回应|表现|手势|游戏|动作)"), r"\1"),
    # 享用 → 吃/喝
    (re.compile(r"享用(饮料|饮品|茶|咖啡|酒)"), r"喝\1"),
    (re.compile(r"享用(餐|饭|食物|早餐|午餐|晚餐|大餐|烤肉|烤肉大餐)"), r"吃\1"),
    # 前往 → 去
    (re.compile(r"前往"), "去"),
    # 手持 → 拿着
    (re.compile(r"手持"), "拿着"),
    # 装置 → 设备
    (re.compile(r"装置"), "设备"),
    # 辅助设备 → 辅助工具
    (re.compile(r"辅助设备"), "辅助工具"),
    # "一个" 在动词+宾语前多余
    (re.compile(r"(参加|吃|喝|看|做|玩|用|举办|进行)了一个"), r"\1了"),
    # "社会聚会" → "聚会"
    (re.compile(r"社会聚会"), "聚会"),
    # "时间段" → "时候"
    (re.compile(r"一个时间段内"), "时候"),
    (re.compile(r"某个时间段内"), "那时候"),
    (re.compile(r"未来的时间段"), "未来"),
    (re.compile(r"未来的时间段里"), "未来"),
    # "交通工具" → "车"
    (re.compile(r"交通工具"), "车"),
    # "博士节日/博士节" → "万圣节"（本体已修，但润色输出可能残留）
    (re.compile(r"博士节日"), "万圣节"),
    (re.compile(r"博士节"), "万圣节"),
    # "铁子" 误译 → "熨斗"
    (re.compile(r"铁子"), "熨斗"),
    # "用铅做出" → "用铅"（lead 词义消歧由 LLM 处理，后处理兜底）
    (re.compile(r"用铅做出一场游戏"), "在领先"),
]


def postprocess_zh(s: str) -> str:
    """中文翻译腔后处理"""
    if not s:
        return s
    for pat, rep in ZH_CALQUE_RULES:
        s = pat.sub(rep, s)
    return s


def postprocess_en(s: str) -> str:
    """英文后处理：去多余空格、修标点"""
    if not s:
        return s
    # 多空格
    s = re.sub(r"\s+", " ", s).strip()
    # 末尾加句号（如果没有标点）
    if s and not s[-1] in ".!?":
        s = s + "."
    return s


def load_ontology(path: Path) -> dict:
    """加载本体，按 label 索引"""
    ont = json.load(open(path))
    items = ont["ontology"] if isinstance(ont, dict) and "ontology" in ont else ont
    return {it["label"]: it for it in items}


def icon_hints_str(labels, ont_by_label):
    """构造 icon 提示串：每个 icon 的 label / 中文语义 / CS角色"""
    lines = []
    for i, lab in enumerate(labels, 1):
        info = ont_by_label.get(lab, {})
        label_zh = info.get("label_zh", lab)
        core_zh = info.get("core_semantic_zh", "")
        cs_role = info.get("cs_role", "?")
        sem_type = info.get("semantic_type", "")
        lines.append(
            f"  {i}. label={lab} | 中文语义={label_zh}/{core_zh} | CS角色={cs_role} | 类型={sem_type}"
        )
    return "\n".join(lines)


def gen(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# 检测元话语泄漏的标志（用于从输出中剥离 prompt 泄漏）
_META_PREFIXES_EN = [
    "here is", "here's", "polished:", "polished sentence", "refined:", "output:",
    "based on the icon", "i'm happy to help", "i polish", "i'd be happy",
    "sure,", "certainly,", "of course,", "the polished",
    "sentence:", "refined sentence", "polished sentence:",
]
_META_PREFIXES_ZH = [
    "润色后：", "润色后:", "润色：", "润色:", "改进后：", "改进后:",
    "修改后：", "修改后:", "句子：", "句子:", "以下是", "这是润色",
    "根据图标", "基于图标", "输出：", "输出:", "答：", "答:",
]


def clean_sentence(text: str) -> str:
    """提取第一行非空、去引号、剥离元话语泄漏。"""
    if not text:
        return ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""

    # 1) 找到第一个不含元话语的行
    s = ""
    for line in lines:
        low = line.lower()
        is_meta = False
        for p in _META_PREFIXES_EN + _META_PREFIXES_ZH:
            if low.startswith(p.lower()):
                is_meta = True
                break
        # 含冒号且长度 >30 的行多为解释
        if is_meta or ((":" in line) and len(line) > 30 and not _looks_like_sentence(line)):
            continue
        s = line
        break

    if not s:
        s = lines[0]

    # 2) 去引号
    s = s.strip().strip("\"'`""''")

    # 3) 去常见前缀
    s = re.sub(r"^(润色后[:：]|Polished[:：]|Refined[:：]|句子[:：]|Sentence[:：])\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(Here is|Here's|Polished|Refined|Output|Sure|Certainly)[^:]*:\s*", "", s, flags=re.IGNORECASE)

    # 4) 长度过滤 - 太长的输出基本是泄漏
    if len(s) > 120:
        # 试着截到第一个句号
        m = re.search(r"[.。!?！？]", s)
        if m and m.start() < 80:
            s = s[: m.start() + 1]
        else:
            # 完全泄漏，返回空
            return ""

    return s.strip()


def _looks_like_sentence(s: str) -> bool:
    """简单判断：句子通常以句末标点结尾且不太长。"""
    return bool(re.search(r"[.。!?！？]$", s.strip())) and len(s) < 100


def baseline_polish(item, ont, model, tokenizer):
    """Baseline: 一次性润色，无反馈循环（带人称锁定 + 句式保留 + 后处理）"""
    st = item.get("subject_type", "third")
    common = dict(
        icon_hints=icon_hints_str(item["labels"], ont),
        original_zh=item.get("sentence_zh", ""),
        original_en=item.get("sentence_en", ""),
        type=item.get("type", ""),
        subject_type=st,
        subj_en=SUBJ_PRONOUN_EN.get(st, "he/she/they/it"),
        subj_zh=SUBJ_PRONOUN_ZH.get(st, "他/她/它"),
        form_hint=detect_form_hint(item),
        subj_hint=detect_subject_hint(item, ont),
    )
    en_out = gen(model, tokenizer, INITIAL_PROMPT_EN.format(**common), max_new_tokens=100)
    zh_out = gen(model, tokenizer, INITIAL_PROMPT_ZH.format(**common), max_new_tokens=100)
    zh = clean_sentence(zh_out)
    en = clean_sentence(en_out)
    # 后处理
    zh = postprocess_zh(zh)
    en = postprocess_en(en)
    return {
        "baseline_en": en,
        "baseline_zh": zh,
    }


def self_refine_polish(item, ont, model, tokenizer, K: int = 2):
    """Self-Refine: 初始润色 → (反馈 → 改写) × K（带人称锁定 + 句式保留 + 后处理）"""
    st = item.get("subject_type", "third")
    common = dict(
        icon_hints=icon_hints_str(item["labels"], ont),
        original_zh=item.get("sentence_zh", ""),
        original_en=item.get("sentence_en", ""),
        type=item.get("type", ""),
        subject_type=st,
        subj_en=SUBJ_PRONOUN_EN.get(st, "he/she/they/it"),
        subj_zh=SUBJ_PRONOUN_ZH.get(st, "他/她/它"),
        form_hint=detect_form_hint(item),
        subj_hint=detect_subject_hint(item, ont),
    )

    # --- English ---
    en_cur = clean_sentence(gen(model, tokenizer, INITIAL_PROMPT_EN.format(**common), max_new_tokens=100))
    en_trace = [{"step": "init", "out": en_cur}]
    for k in range(K):
        fb = gen(
            model,
            tokenizer,
            FEEDBACK_PROMPT_EN.format(
                icon_hints=common["icon_hints"],
                original_en=item.get("sentence_en", ""),
                polished_en=en_cur,
            ),
            max_new_tokens=200,
        )
        refined = clean_sentence(
            gen(
                model,
                tokenizer,
                REFINE_PROMPT_EN.format(
                    icon_hints=common["icon_hints"],
                    original_en=item.get("sentence_en", ""),
                    polished_en=en_cur,
                    feedback=fb,
                ),
                max_new_tokens=100,
            )
        )
        en_trace.append({"step": f"feedback_{k+1}", "out": fb.strip()})
        if refined:
            en_cur = refined
            en_trace.append({"step": f"refine_{k+1}", "out": en_cur})

    # --- Chinese ---
    zh_cur = clean_sentence(gen(model, tokenizer, INITIAL_PROMPT_ZH.format(**common), max_new_tokens=100))
    zh_trace = [{"step": "init", "out": zh_cur}]
    for k in range(K):
        fb = gen(
            model,
            tokenizer,
            FEEDBACK_PROMPT_ZH.format(
                icon_hints=common["icon_hints"],
                original_zh=item.get("sentence_zh", ""),
                polished_zh=zh_cur,
            ),
            max_new_tokens=200,
        )
        refined = clean_sentence(
            gen(
                model,
                tokenizer,
                REFINE_PROMPT_ZH.format(
                    icon_hints=common["icon_hints"],
                    original_zh=item.get("sentence_zh", ""),
                    polished_zh=zh_cur,
                    feedback=fb,
                ),
                max_new_tokens=100,
            )
        )
        zh_trace.append({"step": f"feedback_{k+1}", "out": fb.strip()})
        if refined:
            zh_cur = refined
            zh_trace.append({"step": f"refine_{k+1}", "out": zh_cur})

    # 后处理
    en_cur = postprocess_en(en_cur)
    zh_cur = postprocess_zh(zh_cur)

    return {
        "selfrefine_en": en_cur,
        "selfrefine_en_trace": en_trace,
        "selfrefine_zh": zh_cur,
        "selfrefine_zh_trace": zh_trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="样本数（-1 表示全量）")
    ap.add_argument("--K", type=int, default=2, help="Self-Refine 迭代轮数")
    ap.add_argument("--gpu", type=str, default="1", help="使用的 GPU id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None, help="输出 JSON 路径")
    ap.add_argument("--model", type=str, default=LLAMA_PATH)
    ap.add_argument("--skip-baseline", action="store_true", help="只跑 Self-Refine，跳过 baseline")
    ap.add_argument("--baseline-only", action="store_true", help="只跑 Baseline，跳过 Self-Refine")
    ap.add_argument("--resume", action="store_true", help="断点续跑：从已有输出文件继续")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)

    out_path = args.out or str(ROOT / "EmotionClassify/AAC2Text/data/cleardata/polish_demo_50.json")

    print(f"[load] valid_data from {VALID_DATA}")
    valid = json.load(open(VALID_DATA))
    print(f"[load] {len(valid)} valid samples")

    print(f"[load] ontology from {ONTOLOGY_ZH}")
    ont = load_ontology(ONTOLOGY_ZH)
    print(f"[load] {len(ont)} icons in ontology")

    # 抽样
    if args.n == -1:
        samples = list(valid)
        print(f"[sample] 全量 {len(samples)} 条")
    else:
        random.seed(args.seed)
        samples = random.sample(valid, min(args.n, len(valid)))
        print(f"[sample] {len(samples)} samples (seed={args.seed})")

    # 断点续跑
    existing_results = {}
    if args.resume and os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            for r in prev:
                if "item_id" in r:
                    existing_results[r["item_id"]] = r
            print(f"[resume] 从已有结果加载 {len(existing_results)} 条")
        except Exception as e:
            print(f"[resume] 加载失败: {e}")

    # 加载模型
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
    skipped = 0
    t_start = time.time()
    for i, item in enumerate(samples):
        # 断点续跑
        if args.resume and item.get("item_id") in existing_results:
            results.append(existing_results[item["item_id"]])
            skipped += 1
            continue

        t_item = time.time()
        print(f"\n[{i+1}/{len(samples)}] id={item.get('id')} labels={item['labels']}")
        print(f"  原 ZH: {item.get('sentence_zh','')}")
        print(f"  原 EN: {item.get('sentence_en','')}")

        out = {
            "id": item.get("id"),
            "item_id": item.get("item_id"),
            "labels": item["labels"],
            "type": item.get("type"),
            "subject_type": item.get("subject_type"),
            "original_en": item.get("sentence_en", ""),
            "original_zh": item.get("sentence_zh", ""),
        }

        if not args.skip_baseline:
            b = baseline_polish(item, ont, model, tokenizer)
            out.update(b)
            print(f"  Base ZH: {b['baseline_zh']}")
            print(f"  Base EN: {b['baseline_en']}")

        if not args.baseline_only:
            sr = self_refine_polish(item, ont, model, tokenizer, K=args.K)
            out.update(sr)
            print(f"  SR  ZH: {sr['selfrefine_zh']}")
            print(f"  SR  EN: {sr['selfrefine_en']}")
        print(f"  ({time.time()-t_item:.1f}s)")

        results.append(out)

        # 每 100 条保存一次（断点保护）
        if (i + 1) % 100 == 0:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            elapsed = time.time() - t_start
            done = len(results) - skipped
            print(f"[checkpoint] 已处理 {done} 条 (跳过 {skipped}), {elapsed:.0f}s, {elapsed/max(done,1):.1f}s/sample")

    elapsed = time.time() - t_start
    print(f"\n[done] {len(results)} samples (skipped {skipped}), {elapsed:.1f}s total")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")

    # 简单统计改动率
    if not args.skip_baseline and not args.baseline_only:
        zh_changed = sum(1 for r in results if r.get("baseline_zh") != r.get("selfrefine_zh"))
        en_changed = sum(1 for r in results if r.get("baseline_en") != r.get("selfrefine_en"))
        print(f"[stats] Self-Refine vs Baseline 改动率: ZH {zh_changed}/{len(results)}, EN {en_changed}/{len(results)}")
    elif args.baseline_only:
        # Baseline-only 统计
        from collections import Counter as _C
        def has_leak(s):
            if not s: return True
            leak = ['based on the icon', "i'm happy to help", 'here is', "here's", 'polished:', '【', 'output format']
            sl = s.lower()
            return any(p in sl for p in leak) or len(s) > 120
        zh_leak = sum(1 for r in results if has_leak(r.get("baseline_zh","")))
        en_leak = sum(1 for r in results if has_leak(r.get("baseline_en","")))
        zh_empty = sum(1 for r in results if not r.get("baseline_zh","").strip())
        en_empty = sum(1 for r in results if not r.get("baseline_en","").strip())
        # 人称锁定检查
        pronoun_mismatch = 0
        for r in results:
            st = r.get("subject_type","")
            zh = r.get("baseline_zh","")
            if st == "first" and "我" not in zh: pronoun_mismatch += 1
            elif st == "second" and "你" not in zh: pronoun_mismatch += 1
        # 翻译腔残留
        calque = ['一个','那个','这个','使用','进行','做出','前往','手持','制造','享用','装置','设备','区域']
        zh_calque = sum(1 for r in results if any(w in r.get("baseline_zh","") for w in calque))
        print(f"[stats] Baseline-only 50条:")
        print(f"  Prompt 泄漏: EN {en_leak}/50, ZH {zh_leak}/50")
        print(f"  空输出: EN {en_empty}/50, ZH {zh_empty}/50")
        print(f"  人称锁定失败: ZH {pronoun_mismatch}/50")
        print(f"  翻译腔残留: ZH {zh_calque}/50")


if __name__ == "__main__":
    main()
