#!/usr/bin/env bash
# 启动 AAC Web 服务：先构建前端，再起 FastAPI 后端。
set -e

# 切到仓库根目录（保证 ./AAC2Text/... 等相对路径 & sys.path 正确）
cd "$(dirname "$0")/.."

# 1) 构建前端（React + Vite -> frontend/dist）
echo "===== building frontend ====="
cd frontend
npm install
npm run build
cd ..

# 2) 起后端（pin GPU；CUDA_LAUNCH_BLOCKING=1 让异步 CUDA 错误同步抛出）。
#    cuda:0 = 物理 GPU3（SASRec/RoBERTa 快速模型），cuda:1 = 物理 GPU2（空闲，专给 8B Llama 翻译器），
#    两者分开彻底消除点击预测时的 GPU 争用。port 8001：本机 8000 已被 icon_game 后端占用。
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_LAUNCH_BLOCKING=1
PORT=8001
echo "===== starting server on 0.0.0.0:${PORT} (GPU ${CUDA_VISIBLE_DEVICES}) ====="
cd web
exec /home/user1/miniconda3/envs/Emotion/bin/python -m uvicorn server:app --host 0.0.0.0 --port ${PORT}
