"""
AAC 模型测试脚本

测试训练好的模型，计算多种评估指标：
- BLEU: n-gram 重叠
- METEOR: 对齐+同义词+词干匹配
- ROUGE-L: 最长公共子序列
- BERTScore: 语义相似度
- chrF: 字符级 n-gram F-score
- 语义完整性: 输入符号是否都被翻译
- Exact Match: 预测与参考完全相同
- Partial Match: 至少共享一个词
"""

import os
import sys
import json
import torch
import random
import argparse
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

# 添加本地模块路径
sys.path.insert(0, os.path.dirname(__file__))
nltk.data.path.insert(0, os.path.join(os.path.dirname(__file__), "nltk_data"))
from bleu.bleu import Bleu


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_all_metrics(preds, refs):
    """计算所有评估指标

    Args:
        preds: 预测句子列表 ["I want water.", ...]
        refs: 参考句子列表（每条可以是字符串或列表） ["I want water.", ...] 或 [["I want water."], ...]
    """
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    from nltk import word_tokenize
    import sacrebleu

    # 统一 refs 格式为字符串列表
    ref_strs = []
    for r in refs:
        if isinstance(r, list):
            ref_strs.append(r[0] if len(r) == 1 else r)
        else:
            ref_strs.append(r)

    results = {}

    # ---- BLEU ----
    print("计算 BLEU...")
    bleu_refs = [[r] if isinstance(r, str) else r for r in refs]
    bleu_metric = Bleu()
    bleu_result = bleu_metric.compute(predictions=preds, references=bleu_refs)
    results["bleu"] = bleu_result["bleu"]

    # ---- chrF (sacrebleu) ----
    print("计算 chrF...")
    chrf = sacrebleu.corpus_chrf(preds, [[r] for r in ref_strs])
    results["chrf"] = chrf.score

    # ---- ROUGE-L ----
    print("计算 ROUGE-L...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_f1_scores = []
    for pred, ref in zip(preds, ref_strs):
        s = scorer.score(ref, pred)
        rouge_l_f1_scores.append(s["rougeL"].fmeasure)
    results["rouge_l"] = sum(rouge_l_f1_scores) / len(rouge_l_f1_scores)

    # ---- METEOR ----
    print("计算 METEOR...")
    meteor_scores = []
    for pred, ref in zip(preds, ref_strs):
        pred_tokens = word_tokenize(pred)
        ref_tokens = word_tokenize(ref)
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
    results["meteor"] = sum(meteor_scores) / len(meteor_scores)

    # ---- BERTScore ----
    print("计算 BERTScore...")
    from bert_score import score as bert_score_fn
    bertscore_model_path = "/home/user1/liuduanye/AgentPipeline/bertscore_model"
    P, R, F1 = bert_score_fn(preds, ref_strs, lang="en", model_type=bertscore_model_path, num_layers=17, verbose=False)
    results["bertscore_precision"] = P.mean().item()
    results["bertscore_recall"] = R.mean().item()
    results["bertscore_f1"] = F1.mean().item()

    # ---- Exact Match ----
    exact_match = sum(1 for p, r in zip(preds, ref_strs) if p.strip() == r.strip())
    results["exact_match"] = exact_match / len(preds)

    # ---- Partial Match ----
    partial_match = 0
    for pred, ref in zip(preds, ref_strs):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if pred_words & ref_words:
            partial_match += 1
    results["partial_match"] = partial_match / len(preds)

    return results


def test_model(config: dict, num_samples: int = 50):
    """测试模型"""

    model_config = config["model"]
    test_config = config.get("test", {})

    print("="*60)
    print("AAC 模型测试")
    print("="*60)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True
    )

    # 加载基础模型
    print(f"\n加载基础模型: {model_config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载 LoRA 权重
    output_dir = model_config["output_dir"]
    if os.path.exists(output_dir):
        print(f"加载 LoRA 权重: {output_dir}")
        model = PeftModel.from_pretrained(model, output_dir)
    else:
        print(f"警告: LoRA 权重目录不存在: {output_dir}")

    model.eval()
    print("模型加载完成\n")

    # 测试样例
    test_cases = test_config.get("test_samples", [
        ["I", "want_to", "water"],
        ["I", "am", "happy"],
        ["I", "eat_to", "apple"],
    ])

    print("-"*60)
    print("推理测试:")
    print("-"*60)

    for labels in test_cases:
        prompt = f"Translate these AAC symbols to a sentence: {' '.join(labels)}"
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                stop_strings=["<|im_end|>", "\n"],
                tokenizer=tokenizer,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().split('\n')[0].strip()
        print(f"Labels: {labels}")
        print(f"Output: {response}")
        print()

    # 加载验证集计算指标
    val_path = config["data"].get("val_data")
    if val_path and os.path.exists(val_path):
        print("-"*60)
        print(f"计算评估指标 (采样 {num_samples} 条)...")
        print("-"*60)

        with open(val_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # 随机采样
        random.seed(42)
        test_data = random.sample(test_data, min(num_samples, len(test_data)))

        preds = []
        refs = []

        print(f"\n开始推理 {len(test_data)} 条数据...\n")

        for i, item in enumerate(test_data):
            labels = item["labels"]
            sentence = item["sentence"].strip('"').strip("'").strip()

            prompt = f"Translate these AAC symbols to a sentence: {' '.join(labels)}"
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    stop_strings=["<|im_end|>", "\n"],
                    tokenizer=tokenizer,
                )

            pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = pred.strip().split('\n')[0].strip()

            preds.append(pred)
            refs.append([sentence])

            # 每 10 条打印进度
            if (i + 1) % 10 == 0:
                print(f"已推理: {i+1}/{len(test_data)} 条")

        # 计算所有指标
        print(f"\n推理完成，开始计算评估指标...\n")
        metrics = compute_all_metrics(preds, refs)

        # 打印结果
        print(f"\n{'='*60}")
        print("评估结果:")
        print(f"{'='*60}")
        print(f"  BLEU:                  {metrics['bleu']:.4f}")
        print(f"  chrF:                  {metrics['chrf']:.2f}")
        print(f"  ROUGE-L:               {metrics['rouge_l']:.4f}")
        print(f"  METEOR:                {metrics['meteor']:.4f}")
        print(f"  BERTScore-Precision:   {metrics['bertscore_precision']:.4f}")
        print(f"  BERTScore-Recall:      {metrics['bertscore_recall']:.4f}")
        print(f"  BERTScore-F1:          {metrics['bertscore_f1']:.4f}")
        print(f"  Exact Match:           {metrics['exact_match']:.4f}")
        print(f"  Partial Match:         {metrics['partial_match']:.4f}")
        print(f"{'='*60}")

        # 显示一些预测示例
        print("\n预测示例:")
        for i in range(min(5, len(preds))):
            print(f"\n标签: {test_data[i]['labels']}")
            print(f"参考: {refs[i][0]}")
            print(f"预测: {preds[i]}")

        return metrics

    return None


def main():
    parser = argparse.ArgumentParser(description='AAC 模型测试')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--num', type=int, default=50, help='测试样本数量')
    args = parser.parse_args()

    # 加载配置
    config_path = args.config or "../../config.yaml"
    config = load_config(config_path)

    test_model(config, args.num)


if __name__ == "__main__":
    main()
