#!/bin/bash
set -e
cd /home/user1/liuduanye/EmotionClassify/AAC2Text

BASE=/home/user1/liuduanye/Meta-Llama-3-8B-Instruct
SFT=checkpoints/aac_model_zh
DPO=checkpoints/aac_dpo_zh_propicto
TD=data/cleardata/propicto_eval_test.json
PY=.venv_dpo/bin/python

export CUDA_VISIBLE_DEVICES=3

echo ">>> [$(date)] SFT 基线评测开始"
"$PY" scripts/test_zh.py \
  --checkpoint "$SFT" \
  --base-model "$BASE" \
  --test-data "$TD" \
  --output checkpoints/eval_sft_propicto.json \
  > checkpoints/eval_sft.log 2>&1
echo ">>> [$(date)] SFT 基线评测完成"

echo ">>> [$(date)] 新 DPO 评测开始"
"$PY" scripts/test_zh.py \
  --checkpoint "$DPO" \
  --sft-checkpoint "$SFT" \
  --base-model "$BASE" \
  --test-data "$TD" \
  --output checkpoints/eval_dpo_propicto.json \
  > checkpoints/eval_dpo.log 2>&1
echo ">>> [$(date)] 新 DPO 评测完成"

echo "ALL DONE"
