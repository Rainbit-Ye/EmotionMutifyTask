"""
基线对比实验 — 一键运行所有基线训练
按顺序执行：IA3 → PrefixTuning → LoRA r=16 → LoRA r=32 → Context Window
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from baseline_ia3 import run_ia3_training
from baseline_prefix import run_prefix_training
from baseline_lora_ablation import run_lora_ablation
from baseline_context import run_context_training


def run_all_baselines(config):
    """运行所有基线实验"""
    results = {}

    baselines = [
        ("IA3", lambda: run_ia3_training(config)),
        ("PrefixTuning", lambda: run_prefix_training(config)),
        ("LoRA-r16", lambda: run_lora_ablation(config, rank=16)),
        ("LoRA-r32", lambda: run_lora_ablation(config, rank=32)),
        ("Context-LoRA", lambda: run_context_training(config)),
    ]

    for name, fn in baselines:
        print(f"\n{'='*60}")
        print(f"开始训练: {name}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            model, best_acc = fn()
            elapsed = time.time() - start_time
            results[name] = {
                "best_val_acc": best_acc,
                "time_seconds": elapsed,
                "status": "success"
            }
            print(f"\n[{name}] 完成! 最佳验证 acc: {best_acc:.4f}, 耗时: {elapsed/60:.1f} 分钟")
        except Exception as e:
            elapsed = time.time() - start_time
            results[name] = {
                "best_val_acc": None,
                "time_seconds": elapsed,
                "status": f"failed: {str(e)}"
            }
            print(f"\n[{name}] 失败: {e}")

        # 清理 GPU 缓存
        import torch
        torch.cuda.empty_cache()

    # 保存结果摘要
    output_dir = config['model']['output_dir']
    summary_path = os.path.join(output_dir, "baseline_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印对比表
    print(f"\n{'='*60}")
    print("基线实验结果汇总")
    print(f"{'='*60}")
    print(f"{'方法':<20} {'最佳验证Acc':<15} {'耗时(分钟)':<12} {'状态'}")
    print("-" * 60)
    for name, r in results.items():
        acc = f"{r['best_val_acc']:.4f}" if r['best_val_acc'] is not None else "N/A"
        t = f"{r['time_seconds']/60:.1f}" if r['time_seconds'] else "N/A"
        print(f"{name:<20} {acc:<15} {t:<12} {r['status']}")
    print(f"\n结果已保存到: {summary_path}")

    return results


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    run_all_baselines(config)
