"""
修复中文本体 aac_full_ontology_zh.json 的质量问题（v2）
策略：
1. 对所有含格式污染的字段，用更鲁棒的解析提取实际内容
2. 解析失败或仍为英文的字段，逐字段单独翻译（避免多字段混合输出问题）
3. super_concept_zh 大量未翻译是因为许多super_concept本身就是常见英文词
"""

import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
from tqdm import tqdm


def _is_english(text: str) -> bool:
    if not text:
        return True
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return False
    return True


def _is_letter_or_abbreviation(text: str) -> bool:
    if not text:
        return True
    return bool(re.match(r'^[A-Z0-9]+$', text))


def _has_format_artifact(text: str) -> bool:
    """检测格式污染：包含 '核心语义', '标签', '上位概念' 等结构化前缀"""
    if not text:
        return False
    # 含冒号/短横线且含中文关键词
    markers = ['核心语义', '标签', '上位概念']
    for m in markers:
        if m in text:
            return True
    return False


def _robust_extract(text: str, target_field: str) -> Optional[str]:
    """
    从格式污染的文本中提取目标字段的值。
    支持格式：
    - "核心语义: xxx | 标签: yyy | 上位概念: zzz"
    - "核心语义: xxx - 标签: yyy - 上位概念: zzz"
    - "xxx - 标签: yyy - 上位概念: zzz"
    - "标签: yyy" (单字段)
    """
    if not text:
        return None

    field_map = {'核心语义': 'core_semantic_zh', '标签': 'label_zh', '上位概念': 'super_concept_zh'}
    target_cn = target_field
    if target_field in field_map:
        target_cn = target_field
    else:
        # reverse lookup
        for cn, en in field_map.items():
            if en == target_field:
                target_cn = cn
                break

    # 尝试按分隔符分割：| 或 换行
    # 先统一分隔符
    normalized = text.replace('|', '\n').replace(' - ', '\n').replace(' -\n', '\n')
    lines = [l.strip() for l in normalized.split('\n') if l.strip()]

    # 解析每行，提取 字段名→值
    parsed = {}
    for line in lines:
        # 尝试 "字段名：值" 或 "字段名:值" 格式
        for marker in ['核心语义', '标签', '上位概念']:
            for sep in ['：', ':']:
                prefix = marker + sep
                if line.startswith(prefix):
                    parsed[marker] = line[len(prefix):].strip()
                    break
            if any(line.startswith(m + s) for s in ['：', ':'] for m in ['核心语义', '标签', '上位概念']):
                break

    if target_cn in parsed:
        return parsed[target_cn]

    # 如果只有1行且没有格式标记，直接返回（本身就是干净内容）
    if len(lines) == 1 and not any(m in lines[0] for m in ['核心语义', '标签', '上位概念']):
        return lines[0]

    # 如果有多行但没有成功解析，取不含格式标记的行
    clean_lines = [l for l in lines if not any(m in l for m in ['核心语义', '标签', '上位概念'])]
    if clean_lines:
        return clean_lines[0]

    return None


def find_available_gpu(min_free_gb: int = 5):
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            idx, free = line.strip().split(", ")
            if int(free) >= min_free_gb * 1024:
                return int(idx)
    except Exception:
        pass
    return None


def translate_field(model, tokenizer, english_text: str, field_desc: str) -> str:
    """
    单字段翻译，更可靠。返回中文翻译。
    """
    prompt = (
        f"将以下英文{field_desc}翻译为简洁自然的中文。只输出翻译结果，不加解释、标点、序号。\n\n"
        f"英文：{english_text}\n\n中文："
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # 清理可能的格式污染
    for marker in ['核心语义：', '核心语义:', '标签：', '标签:', '上位概念：', '上位概念:']:
        if response.startswith(marker):
            response = response[len(marker):].strip()
    # 去掉换行后的多余内容
    response = response.split('\n')[0].strip()
    # 去掉引号
    for ch in ['"', "'", '\u201c', '\u201d']:
        response = response.strip(ch)

    return response


def main():
    base_dir = "/home/user1/liuduanye/EmotionClassify/AAC2Text"
    zh_ontology_path = f"{base_dir}/data/processed/aac_full_ontology_zh.json"
    output_path = f"{base_dir}/data/processed/aac_full_ontology_zh.json"  # 直接覆盖原文件

    # 加载原始中文本体
    with open(zh_ontology_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data["ontology"]
    print(f"中文本体: {len(items)} 条")

    zh_char = re.compile(r'[\u4e00-\u9fff]')

    # === Step 1: 用鲁棒解析修复所有格式污染 ===
    print(f"\n=== Step 1: 鲁棒解析修复格式污染 ===")
    fixed_by_parse = 0
    still_broken = []

    for item in items:
        icon_id = item.get("icon_id", "N/A")
        for field_zh in ["core_semantic_zh", "label_zh", "super_concept_zh"]:
            val = item.get(field_zh, "")
            if _has_format_artifact(val):
                # 尝试从格式化输出中提取实际内容
                field_cn_map = {"core_semantic_zh": "核心语义", "label_zh": "标签", "super_concept_zh": "上位概念"}
                extracted = _robust_extract(val, field_cn_map[field_zh])
                if extracted and not _has_format_artifact(extracted):
                    item[field_zh] = extracted
                    fixed_by_parse += 1
                else:
                    # 解析失败，标记需要重新翻译
                    still_broken.append((item, field_zh))

    print(f"通过解析修复: {fixed_by_parse} 处")
    print(f"仍需重新翻译: {len(still_broken)} 处")

    # === Step 2: 收集所有需要翻译的字段 ===
    print(f"\n=== Step 2: 收集待翻译字段 ===")
    translate_tasks = []  # (item, field_zh, english_text, field_desc)

    # 从still_broken中收集
    for item, field_zh in still_broken:
        en_field = field_zh.replace("_zh", "")
        en_text = item.get(en_field, "").replace("_", " ")
        desc_map = {"core_semantic_zh": "核心语义", "label_zh": "标签", "super_concept_zh": "上位概念"}
        if en_text:
            translate_tasks.append((item, field_zh, en_text, desc_map[field_zh]))

    # 收集仍为英文的字段
    for item in items:
        icon_id = item.get("icon_id", "")

        # core_semantic_zh
        if _is_english(item.get("core_semantic_zh", "")):
            en = item.get("core_semantic", "").replace("_", " ")
            if en:
                translate_tasks.append((item, "core_semantic_zh", en, "核心语义"))

        # label_zh (跳过字母/缩写)
        if _is_english(item.get("label_zh", "")) and not _is_letter_or_abbreviation(item.get("label", "")):
            en = item.get("label", "").replace("_", " ")
            if en:
                translate_tasks.append((item, "label_zh", en, "标签"))

        # super_concept_zh
        if _is_english(item.get("super_concept_zh", "")):
            en = item.get("super_conantic", "") if "super_conantic" in item else item.get("super_concept", "").replace("_", " ")
            if en:
                translate_tasks.append((item, "super_concept_zh", en, "上位概念"))

    # 去重（同一item同一field可能被添加两次）
    seen = set()
    unique_tasks = []
    for task in translate_tasks:
        key = (id(task[0]), task[1])
        if key not in seen:
            seen.add(key)
            unique_tasks.append(task)
    translate_tasks = unique_tasks

    print(f"需要翻译: {len(translate_tasks)} 个字段")

    # === Step 3: 用Qwen逐字段翻译 ===
    if translate_tasks:
        print(f"\n=== Step 3: 逐字段翻译 ===")
        zh_model_path = "/home/user1/liuduanye/qwen/Qwen2_5-1_5B-Instruct"
        gpu = find_available_gpu(min_free_gb=5)
        if gpu is None:
            raise RuntimeError("未找到空闲GPU")

        print(f"加载 Qwen: {zh_model_path} → GPU {gpu}")
        tokenizer = AutoTokenizer.from_pretrained(zh_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            zh_model_path, torch_dtype=torch.float16,
            device_map={"": gpu}, trust_remote_code=True
        )
        model.eval()

        success = 0
        for item, field_zh, en_text, desc in tqdm(translate_tasks, desc="翻译字段"):
            zh = translate_field(model, tokenizer, en_text, desc)
            if zh and not _is_english(zh) and not _has_format_artifact(zh):
                item[field_zh] = zh
                success += 1

        print(f"翻译成功: {success}/{len(translate_tasks)}")

    # === Step 4: 保存 ===
    print(f"\n=== Step 4: 保存 ===")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"ontology": items}, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path}")

    # === Step 5: 验证 ===
    artifact_after = 0
    untranslated_cs = 0
    untranslated_sc = 0
    untranslated_label = 0

    for item in items:
        if _has_format_artifact(item.get("label_zh", "")):
            artifact_after += 1
        if not zh_char.search(item.get("core_semantic_zh", "")):
            untranslated_cs += 1
        if not zh_char.search(item.get("super_concept_zh", "")):
            untranslated_sc += 1
        if not zh_char.search(item.get("label_zh", "")) and not _is_letter_or_abbreviation(item.get("label", "")):
            untranslated_label += 1

    cs_zh_count = sum(1 for item in items if zh_char.search(item.get("core_semantic_zh", "")))
    label_zh_count = sum(1 for item in items if zh_char.search(item.get("label_zh", "")))
    sc_zh_count = sum(1 for item in items if zh_char.search(item.get("super_concept_zh", "")))

    print(f"\n=== 修复后质量 ===")
    print(f"总条目: {len(items)}")
    print(f"core_semantic_zh 中文: {cs_zh_count} ({cs_zh_count/len(items)*100:.1f}%)")
    print(f"label_zh 中文: {label_zh_count} ({label_zh_count/len(items)*100:.1f}%)")
    print(f"super_concept_zh 中文: {sc_zh_count} ({sc_zh_count/len(items)*100:.1f}%)")
    print(f"label_zh 格式污染: {artifact_after}")
    print(f"core_semantic_zh 未翻译: {untranslated_cs}")
    print(f"super_concept_zh 未翻译: {untranslated_sc}")
    print(f"label_zh 未翻译(非字母): {untranslated_label}")

    if untranslated_cs > 0:
        print(f"\n仍未翻译的 core_semantic_zh:")
        count = 0
        for item in items:
            if not zh_char.search(item.get("core_semantic_zh", "")):
                print(f"  {item.get('icon_id')}: {item.get('core_semantic_zh')}")
                count += 1
                if count >= 15:
                    break

    if untranslated_sc > 0:
        print(f"\n仍未翻译的 super_concept_zh 示例:")
        count = 0
        for item in items:
            if not zh_char.search(item.get("super_concept_zh", "")):
                print(f"  {item.get('icon_id')}: {item.get('super_concept_zh')} (EN: {item.get('super_concept')})")
                count += 1
                if count >= 15:
                    break


if __name__ == '__main__':
    main()
