"""
构建中文图标本体 aac_full_ontology_zh.json

基于英文本体 aac_full_ontology.json，批量翻译所有文本字段为中文：
- core_semantic_zh: 核心语义（中文）
- label_zh: 可读标签（中文）
- super_concept_zh: 上位概念（中文）
- typical_objects_zh: 典型宾语（中文列表）
- typical_modifiers_zh: 典型修饰语（中文列表）

结构性字段（semantic_type, grammar_role, cs_role, can_combine_with, aac_category）
保持英文，因为用于程序逻辑而非人类阅读。

增量写入，可中断恢复。
"""

import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from tqdm import tqdm
import unicodedata


def _is_english(text: str) -> bool:
    """检查文本是否主要是英文字符（无中文字符）"""
    if not text:
        return True
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return False
    return True


def _save(zh_ontology_map, path):
    save_ontology = [zh_ontology_map[oid] for oid in sorted(zh_ontology_map.keys())]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"ontology": save_ontology}, f, ensure_ascii=False, indent=2)


def find_available_gpu(min_free_gb: int):
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


def batch_translate(model, tokenizer, items: List[str], desc: str = "") -> List[str]:
    """批量翻译字符串列表，返回对应的中文翻译"""
    if not items:
        return []

    batch_size = 40
    results = [None] * len(items)

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        prompt = (
            "将以下英文短语翻译为简短自然的中文。每行一个，保持顺序一致。\n"
            "要求：简洁，像词典释义，不加序号标点。\n\n"
        )
        for j, item in enumerate(batch):
            prompt += f"{j+1}. {item}\n"
        prompt += "\n请直接输出中文，每行一个："

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=len(batch) * 10, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        zh_lines = [l.strip().lstrip('0123456789.、)） ') for l in response.split('\n') if l.strip()]

        for j in range(len(batch)):
            if j < len(zh_lines) and zh_lines[j]:
                results[i+j] = zh_lines[j]
            else:
                results[i+j] = batch[j]  # fallback to English

    return results


def build_vocabulary_map(model, tokenizer, ontology: List[Dict]) -> Dict[str, str]:
    """翻译所有unique词汇（typical_objects, typical_modifiers 等），建立 EN→ZH 映射"""
    vocab = set()
    for item in ontology:
        for o in item.get("typical_objects", []):
            if o:
                vocab.add(o)
        for m in item.get("typical_modifiers", []):
            if m:
                vocab.add(m)

    vocab_list = sorted(vocab)
    print(f"词汇表大小: {len(vocab_list)}")

    en_to_zh = {}
    # 增量：如果已有部分翻译，跳过
    for i in range(0, len(vocab_list), batch_size := 40):
        batch = vocab_list[i:i+batch_size]
        untranslated = [w for w in batch if w not in en_to_zh]
        if not untranslated:
            continue

        zh = batch_translate(model, tokenizer, untranslated, "vocab")
        for en, zh_word in zip(untranslated, zh):
            en_to_zh[en] = zh_word

        if (i // 40) % 20 == 0:
            print(f"  词汇翻译进度: {min(i+40, len(vocab_list))}/{len(vocab_list)}")

    return en_to_zh


def main():
    base_dir = "/home/user1/liuduanye/EmotionClassify/AAC2Text"
    en_ontology_path = f"{base_dir}/data/processed/aac_full_ontology.json"
    zh_ontology_path = f"{base_dir}/data/processed/aac_full_ontology_zh.json"
    vocab_cache_path = f"{base_dir}/data/processed/vocab_en_zh.json"

    # 加载英文本体
    with open(en_ontology_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ontology = data["ontology"]
    print(f"英文本体: {len(ontology)} 条")

    # 检查是否已有完整的中文本体
    if os.path.exists(zh_ontology_path):
        with open(zh_ontology_path, 'r', encoding='utf-8') as f:
            zh_data = json.load(f)
        if len(zh_data.get("ontology", [])) >= len(ontology):
            # 检查每条是否都有中文
            all_have_zh = all(
                item.get("core_semantic_zh") for item in zh_data["ontology"]
            )
            if all_have_zh:
                print(f"中文本体已存在且完整: {zh_ontology_path}")
                return

    # 加载 Qwen 模型（擅长中文翻译）
    zh_model_path = "/home/user1/liuduanye/qwen/Qwen2_5-1_5B-Instruct"
    gpu = find_available_gpu(min_free_gb=5)
    if gpu is None:
        raise RuntimeError("未找到空闲GPU")

    print(f"加载 Qwen 模型: {zh_model_path} → GPU {gpu}")
    tokenizer = AutoTokenizer.from_pretrained(zh_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        zh_model_path, torch_dtype=torch.float16,
        device_map={"": gpu}, trust_remote_code=True
    )
    model.eval()

    # Step 1: 翻译词汇表（typical_objects, typical_modifiers）
    print("\n=== Step 1: 翻译词汇表 ===")
    en_to_zh_vocab = {}
    if os.path.exists(vocab_cache_path):
        with open(vocab_cache_path, 'r', encoding='utf-8') as f:
            en_to_zh_vocab = json.load(f)
        print(f"  从缓存加载词汇: {len(en_to_zh_vocab)} 条")

    # 收集所有需要翻译的词汇
    all_vocab = set()
    for item in ontology:
        for o in item.get("typical_objects", []):
            if o and o not in en_to_zh_vocab:
                all_vocab.add(o)
        for m in item.get("typical_modifiers", []):
            if m and m not in en_to_zh_vocab:
                all_vocab.add(m)

    if all_vocab:
        print(f"  需翻译新词汇: {len(all_vocab)} 条")
        new_vocab_list = sorted(all_vocab)
        new_zh = batch_translate(model, tokenizer, new_vocab_list, "vocab")
        for en, zh in zip(new_vocab_list, new_zh):
            en_to_zh_vocab[en] = zh
        # 保存词汇缓存
        with open(vocab_cache_path, 'w', encoding='utf-8') as f:
            json.dump(en_to_zh_vocab, f, ensure_ascii=False, indent=2)
        print(f"  词汇缓存已保存: {len(en_to_zh_vocab)} 条")
    else:
        print("  词汇表已完整，无需翻译")

    # Step 2: 逐条构建中文本体（单条翻译，避免批量错位）
    print("\n=== Step 2: 构建中文本体 ===")

    # 加载已有进度
    zh_ontology = []
    existing_ids = set()
    if os.path.exists(zh_ontology_path):
        with open(zh_ontology_path, 'r', encoding='utf-8') as f:
            zh_ontology = json.load(f).get("ontology", [])
        for item in zh_ontology:
            if item.get("core_semantic_zh") and not _is_english(item.get("core_semantic_zh", "")):
                existing_ids.add(item.get("icon_id", ""))
        print(f"  已有进度: {len(existing_ids)} 条")

    zh_ontology_map = {item["icon_id"]: item for item in zh_ontology if item.get("icon_id")}

    items_to_translate = [item for item in ontology if item.get("icon_id") and item["icon_id"] not in existing_ids]
    print(f"  待翻译: {len(items_to_translate)} 条")

    def translate_single_item(item):
        """单条翻译一个本体条目"""
        core = item.get("core_semantic", "").replace("_", " ")
        label = item.get("label", "").replace("_", " ")
        super_c = item.get("super_concept", "").replace("_", " ")

        prompt = (
            f"将以下AAC象形图符号的英文字段翻译为中文，保持简洁自然：\n\n"
            f"核心语义: {core}\n"
            f"标签: {label}\n"
            f"上位概念: {super_c}\n\n"
            f"请按格式输出：\n"
            f"核心语义 | 标签 | 上位概念\n\n"
            f"输出："
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 解析
        zh_item = dict(item)
        # 尝试按 | 分割
        clean = response.lstrip('0123456789.、)） ')
        parts = [p.strip() for p in clean.split('|')]
        prefixes = ['核心语义：', '核心语义:', '标签：', '标签:', '上位概念：', '上位概念:']
        for pi, p in enumerate(parts):
            for prefix in prefixes:
                parts[pi] = parts[pi].replace(prefix, "").strip()

        if len(parts) >= 3:
            zh_item["core_semantic_zh"] = parts[0]
            zh_item["label_zh"] = parts[1]
            zh_item["super_concept_zh"] = parts[2]
        elif len(parts) == 2:
            zh_item["core_semantic_zh"] = parts[0]
            zh_item["label_zh"] = parts[1]
            zh_item["super_concept_zh"] = super_c
        else:
            # fallback: 按换行解析
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            if lines:
                zh_item["core_semantic_zh"] = lines[0]
                zh_item["label_zh"] = lines[0]
                zh_item["super_concept_zh"] = super_c
            else:
                zh_item["core_semantic_zh"] = core
                zh_item["label_zh"] = label
                zh_item["super_concept_zh"] = super_c

        # 翻译 typical_objects / typical_modifiers（用词汇表映射）
        zh_item["typical_objects_zh"] = [en_to_zh_vocab.get(o, o) for o in item.get("typical_objects", [])]
        zh_item["typical_modifiers_zh"] = [en_to_zh_vocab.get(m, m) for m in item.get("typical_modifiers", [])]

        return zh_item

    # 逐条翻译，每50条保存一次
    save_interval = 50
    for idx, item in enumerate(tqdm(items_to_translate, desc="翻译本体")):
        zh_item = translate_single_item(item)
        zh_ontology_map[item["icon_id"]] = zh_item

        if (idx + 1) % save_interval == 0:
            _save(zh_ontology_map, zh_ontology_path)

    # 最终保存
    _save(zh_ontology_map, zh_ontology_path)

    print(f"\n中文本体已保存: {zh_ontology_path}")
    print(f"总计: {len(zh_ontology_map)} 条")


if __name__ == '__main__':
    main()
