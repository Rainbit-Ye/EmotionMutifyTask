#!/usr/bin/env python3
"""
AAC2Text 中文评估脚本 — 对比英文 SFT / 中文 SFT / 中文 SFT+DPO

指标:
  - BERTScore (bert-base-chinese, 主指标)
  - BLEU-4 (jieba 分词, 参考指标)
  - 翻译腔判别 (人工抽检辅助)

Usage:
    python test_zh.py --checkpoint /path/to/lora --base-model /path/to/base
    python test_zh.py --checkpoint /path/to/lora --compare  # 对比多个 checkpoint
"""

import os
import sys
import json
import torch
import argparse
import jieba
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_test_data(test_path: str):
    """加载测试数据 (cleardata/sft_train.json 的验证集划分)"""
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_predictions(model, tokenizer, data, max_new_tokens=64, batch_size=8):
    """批量生成翻译 — 带句号强约束停止, 避免重复生成"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    preds = []
    model.eval()

    # 中文句末标点作为停止符, 避免模型重复生成
    stop_puncts = ["。", ".", "？", "?", "！", "!", "\n"]

    for i in tqdm(range(0, len(data), batch_size), desc="生成翻译"):
        batch = data[i:i + batch_size]
        prompts = []
        for item in batch:
            labels = item["labels"]
            # 中文指令 (英文版本注释保留)
            # prompt_text = f"Translate these AAC symbols into ONE simple Chinese sentence: {' '.join(labels)}"
            prompt_text = f"请把这些 AAC 图标序列翻译成一个简单的中文句子：{' '.join(labels)}"
            messages = [{"role": "user", "content": prompt_text}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(input_text)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # 在中文句末标点停止生成, 避免重复
                stop_strings=["。", "？", "！", "<|eot_id|>"],
                tokenizer=tokenizer,
            )

        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            response = response.strip()

            # 取第一个句末标点前的内容 (强约束单句翻译)
            for punct in stop_puncts:
                pos = response.find(punct)
                if pos != -1:
                    response = response[:pos + len(punct)]
                    break
            # 去除换行后的内容
            response = response.split('\n')[0].strip()
            preds.append(response)

    return preds


def compute_bertscore(preds, refs):
    """计算 BERTScore (用本地 roberta-base, 相对对比指标)
    注: 无 bert-base-chinese 本地缓存, 用 roberta-base 做相对对比
    """
    from bert_score import score as bert_score_fn
    print("计算 BERTScore (本地 roberta-base)...")
    bertscore_model_path = "/home/user1/liuduanye/EmotionClassify/Model/roberta-base"
    P, R, F1 = bert_score_fn(
        preds, refs,
        model_type=bertscore_model_path,
        num_layers=12,
        verbose=False,
        lang="en",  # 用英文 BERTScore 仍能捕捉字符级相似度, 作相对对比
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def compute_bleu_zh(preds, refs):
    """计算 BLEU-4 (jieba 分词) — 参考指标"""
    import sacrebleu
    print("计算 BLEU (jieba 分词)...")

    # jieba 分词后用空格连接, sacrebleu 按空格切分
    preds_seg = [" ".join(jieba.cut(p)) for p in preds]
    refs_seg = [" ".join(jieba.cut(r)) for r in refs]

    bleu = sacrebleu.corpus_bleu(preds_seg, [refs_seg])
    chrf = sacrebleu.corpus_chrf(preds_seg, [refs_seg])

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
    }


def evaluate_checkpoint(checkpoint_path, base_model, test_data, sft_checkpoint=None, device="cuda"):
    """评估单个 checkpoint
    Args:
        checkpoint_path: 待评估的 LoRA checkpoint
        base_model: 基模路径
        sft_checkpoint: 若提供, 先合并 SFT LoRA 到 base, 再加载待评估 LoRA (用于 DPO 评估)
    """
    print(f"\n{'='*60}")
    print(f"评估 checkpoint: {checkpoint_path}")
    if sft_checkpoint:
        print(f"base 先合并 SFT: {sft_checkpoint}")
    print(f"{'='*60}")

    # 加载 tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 若提供 SFT checkpoint, 先合并 SFT LoRA 到 base (用于 DPO 评估)
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        print(f"加载并合并 SFT LoRA: {sft_checkpoint}")
        model = PeftModel.from_pretrained(model, sft_checkpoint)
        model = model.merge_and_unload()
        print("SFT LoRA 已合并到 base")

    # 加载待评估 LoRA
    if os.path.exists(checkpoint_path):
        print(f"加载 LoRA: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        print(f"警告: checkpoint 不存在: {checkpoint_path}")
        return None

    model.eval()

    # 生成翻译
    preds = generate_predictions(model, tokenizer, test_data)
    refs = [item["target_zh"] for item in test_data]

    # 计算指标
    bertscore = compute_bertscore(preds, refs)
    bleu = compute_bleu_zh(preds, refs)

    # 打印样本
    print("\n样本预览 (前 10 条):")
    print("-" * 60)
    for i in range(min(10, len(preds))):
        print(f"  labels:    {' '.join(test_data[i]['labels'])}")
        print(f"  ref:       {refs[i]}")
        print(f"  pred:      {preds[i]}")
        print()

    return {
        "checkpoint": checkpoint_path,
        "num_samples": len(preds),
        "bertscore": bertscore,
        "bleu": bleu,
        "predictions": preds,
        "references": refs,
    }


def main():
    parser = argparse.ArgumentParser(description='AAC2Text 中文评估')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='单个 checkpoint 路径')
    parser.add_argument('--compare', action='store_true',
                        help='对比模式: 评估英文 SFT / 中文 SFT / 中文 SFT+DPO 三个 checkpoint')
    parser.add_argument('--base-model', type=str,
                        default='/home/user1/liuduanye/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--test-data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/data/cleardata/sft_val.json',
                        help='测试数据 (中文 SFT 验证集)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='评估样本数 (默认全部)')
    parser.add_argument('--sft-checkpoint', type=str, default=None,
                        help='SFT checkpoint 路径 (评估 DPO 时需指定, 作为 base 合并)')
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出 JSON 路径')
    args = parser.parse_args()

    # 加载测试数据
    test_data = load_test_data(args.test_data)
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    print(f"测试数据: {len(test_data)} 条")

    # 待评估的 checkpoint 列表 (DPO checkpoint 需要指定 sft_checkpoint 作为 base)
    sft_zh_path = "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_model_zh"
    if args.compare:
        checkpoints = [
            (sft_zh_path, "中文SFT", None),
            ("/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_dpo_zh", "SFT+DPOv1", sft_zh_path),
            ("/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/aac_dpo_zh_v2", "SFT+DPOv2", sft_zh_path),
        ]
    else:
        if not args.checkpoint:
            print("错误: 单 checkpoint 模式需指定 --checkpoint")
            return
        checkpoints = [(args.checkpoint, os.path.basename(args.checkpoint), args.sft_checkpoint)]

    # 逐个评估
    all_results = []
    for ckpt_path, ckpt_name, sft_ckpt in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"\n跳过不存在的 checkpoint: {ckpt_name} ({ckpt_path})")
            continue
        print(f"\n评估: {ckpt_name}")
        result = evaluate_checkpoint(ckpt_path, args.base_model, test_data, sft_checkpoint=sft_ckpt)
        if result:
            result["name"] = ckpt_name
            all_results.append(result)

    # 汇总对比
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("汇总对比")
        print("=" * 60)
        print(f"{'Checkpoint':<25} {'BERTScore-F1':<15} {'BLEU':<10} {'chrF':<10}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['name']:<25} "
                  f"{r['bertscore']['bertscore_f1']:.4f}        "
                  f"{r['bleu']['bleu']:.2f}     "
                  f"{r['bleu']['chrf']:.2f}")

    # 保存结果
    output_path = args.output or "/home/user1/liuduanye/EmotionClassify/AAC2Text/checkpoints/eval_zh_results.json"
    # 不保存 predictions/references 到汇总
    save_results = [{k: v for k, v in r.items() if k not in ["predictions", "references"]} for r in all_results]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
