"""
server.py —— AAC 选词预测 + 自然语言翻译 的 Web 后端（FastAPI + uvicorn）。

启动（由 web/run.sh 调用，CWD=EmotionClassify 根目录）：
    python -m uvicorn web.server:app --host 0.0.0.0 --port 8000

要点：
- 单 pipeline 实例（SASRec + Llama-3-8B LoRA + RoBERTa），仅加载一次。
- 每用户按 X-Session-Id 头隔离状态（IncrementalState + 对话历史）。
- 全局 INFER_LOCK 串行化所有模型推理（GPU 不可并发 forward），保证并发正确。
- 真实使用数据继续落盘 output/incremental_usage.jsonl（带 session_id）。
"""
import os
import sys
import json
import time
import threading
from pathlib import Path
import torch

# 让仓库根目录可被 import（aac_emotion_pipeline 内部用相对 import sequence_model.*）
REPO = "/home/user1/liuduanye/EmotionClassify"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aac_emotion_pipeline import AACEmotionPipeline, IncrementalState
from web.icon_map import build_icon_map

IMAGES_DIR = "/home/user1/liuduanye/AACTest/AAC/data/images"
# 默认加载【中文 DPO 中配置（beta=0.2, lr=1e-6, 4ep，评测最佳）】，
# 但 DPO 必须叠在 SFT(aac_model_zh) 之上（test_zh.py 的配方），
# 故先用 AAC_SFT_MODEL 合并 SFT 再加载 DPO；缺省 SFT 直接加载 DPO 会退化。
# 切换：
#   AAC_MODEL=.../AAC2Text/checkpoints/aac_dpo_zh_v2_mid   (DPO 中配置，默认)
#   AAC_MODEL=.../AAC2Text/checkpoints/aac_model_zh            (纯中文 SFT)
AAC_MODEL_PATH = os.environ.get(
    "AAC_MODEL", REPO + "/AAC2Text/checkpoints/aac_dpo_zh_v2_mid"
)
# DPO 的 SFT 底座；与 AAC_MODEL 相同时跳过合并。
AAC_SFT_MODEL_PATH = os.environ.get(
    "AAC_SFT_MODEL", REPO + "/AAC2Text/checkpoints/aac_model_zh"
)
AAC_BASE_PATH = "/home/user1/liuduanye/Meta-Llama-3-8B-Instruct"
EMOTION_MODEL_PATH = REPO + "/output/cls_final"
EMOTION_BASE_PATH = REPO + "/Model/roberta-base"
LOG_PATH = REPO + "/output/incremental_usage.jsonl"
DIST_DIR = REPO + "/frontend/dist"

MAX_SEQ_LEN = 16
MAX_SESSIONS = 200

# ---- 全局单例 / 共享状态 ----
PIPELINE = None
ICON_MAP = {}
CATALOG = None
SESSIONS = {}                 # sid -> {"state": IncrementalState, "history": []}
_SESSION_ORDER = []           # 用于 LRU 淘汰
SESSIONS_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()   # 串行化 GPU 推理（8B 翻译等重推理）

# 后台异步翻译（防抖/debounce）：每次点击只跑轻量预测（SASRec+RAG+RoBERTa，<0.1s），
# 8B Llama 翻译放到"每会话一个常驻线程"，等用户停手(~0.7s)后才翻译最新序列。
# 好处：① 点击永远即时（生成期间用户没在点，不争 GIL）；
#       ② 不浪费——连续点击只翻译最终那一串，不会每点一次都跑 4-5s。
# TRANS_PENDING[sid] = 当前待翻译的序列串（最新）；
# TRANS_CACHE[sid]   = 该会话最近一次高质量【中文】翻译结果；
# TRANS_VER[sid]     = 版本号（自增），用于丢弃过期生成；
# TRANS_EVENT[sid]   = 通知常驻线程"有新点击"的 Event；
# TRANS_WORKERS[sid]= 该会话常驻翻译线程。
TRANS_PENDING = {}
TRANS_CACHE = {}
TRANS_VER = {}
TRANS_EVENT = {}
TRANS_WORKERS = {}
TRANS_LAST = {}   # sid -> 上次点击时刻（防抖：自该时刻起满 0.7s 才翻译）
TRANS_LOCK = threading.Lock()


def _get_session(sid: str) -> dict:
    """取/建会话，并做 LRU 淘汰。返回该会话的 state/history 字典。"""
    with SESSIONS_LOCK:
        if sid not in SESSIONS:
            if len(SESSIONS) >= MAX_SESSIONS and _SESSION_ORDER:
                old = _SESSION_ORDER.pop(0)
                SESSIONS.pop(old, None)
            SESSIONS[sid] = {
                "state": IncrementalState(max_seq_len=MAX_SEQ_LEN),
                "history": [],
            }
            _SESSION_ORDER.append(sid)
        else:
            # 刷新 LRU 顺序
            _SESSION_ORDER.remove(sid)
            _SESSION_ORDER.append(sid)
        return SESSIONS[sid]


def _build_catalog():
    """从全量本体构建图标目录（一次缓存）。"""
    global CATALOG
    onto = PIPELINE.icon_predictor.ontology
    vocab = set(PIPELINE.item2idx.keys())
    out = []
    for icon_id, info in onto.items():
        out.append({
            "icon_id": icon_id,
            "label": info.get("label", icon_id),
            "semantic_type": info.get("semantic_type", "unknown"),
            "cs_role": info.get("cs_role", "WHAT"),
            "in_vocab": icon_id in vocab,
            "has_image": icon_id in ICON_MAP,
        })
    CATALOG = out


def _startup():
    global PIPELINE, ICON_MAP, CATALOG
    print("=" * 60)
    print("Loading AAC pipeline (this may take 1-2 min)...")
    print("=" * 60)
    ICON_MAP, _ = build_icon_map()
    PIPELINE = AACEmotionPipeline(
        aac_model_path=AAC_MODEL_PATH,
        aac_base_model_path=AAC_BASE_PATH,
        aac_sft_model_path=AAC_SFT_MODEL_PATH,
        emotion_model_path=EMOTION_MODEL_PATH,
        emotion_base_model_path=EMOTION_BASE_PATH,
            # CUDA_VISIBLE_DEVICES=2,3 时：cuda:0=物理 GPU2（空闲），cuda:1=物理 GPU3。
            # 把最重的 8B Llama 翻译器放 cuda:0（GPU2，独占不与他人争用），
            # 轻量快速模型（SASRec/RoBERTa）放 cuda:1（GPU3），点击预测彻底不被翻译拖慢。
            device="cuda:1",
            aac_translator_device="cuda:0",
        mode="incremental",
        log_path=LOG_PATH,
    )
    PIPELINE.session_id = None
    _build_catalog()
    print("=" * 60)
    print(f"SERVER READY. catalog={len(CATALOG)} icons, images={len(ICON_MAP)}")
    print("=" * 60)


# 模块导入时执行一次（uvicorn 加载 web.server:app 即触发）
_startup()


app = FastAPI(title="AAC 选词预测 + 自然语言翻译")


def _sid_from_request(request: Request) -> str:
    sid = request.headers.get("X-Session-Id") or request.query_params.get("sid")
    return sid or "default"


@app.get("/api/catalog")
def api_catalog():
    """全部图标元数据（前端本地搜索/过滤）。"""
    return CATALOG or []


@app.get("/api/icon/{icon_id}")
def api_icon(icon_id: str):
    """返回图标真实 PNG（404 则前端回退文字）。"""
    fn = ICON_MAP.get(icon_id)
    if not fn:
        return Response(status_code=404)
    path = os.path.join(IMAGES_DIR, fn)
    if not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(path)


@app.post("/api/add")
def api_add(request: Request, body: dict):
    sid = _sid_from_request(request)
    icon_id = (body or {}).get("icon_id")
    if not icon_id:
        return JSONResponse({"error": "missing icon_id"}, status_code=400)
    sess = _get_session(sid)
    # 轻量预测：不跑 8B 翻译（do_translate=False），点击即时返回。
    try:
        res = PIPELINE.add_icon(icon_id, sess["state"], sess["history"],
                               session_id=sid, do_translate=False)
    except Exception as e:  # 单次推理偶发 CUDA 抖动不让服务挂掉
        return JSONResponse({"error": f"inference failed: {e}"}, status_code=500)

    # 触发后台异步翻译（仅保留该会话最新一次，旧请求自动合并）。
    # 单图标（len<2）无意义，跳过，避免无谓的 4-5s 翻译卡住后续点击。
    seq = sess["state"].current_sequence[-3:]
    seq_key = " ".join(seq)
    if len(seq) >= 2:
        with TRANS_LOCK:
            # 确保该会话有常驻翻译线程；否则建一个（defer 到用户停手才生成）
            if sid not in TRANS_WORKERS or not TRANS_WORKERS[sid].is_alive():
                TRANS_EVENT.setdefault(sid, threading.Event())
                t = threading.Thread(target=_translate_loop, args=(sid,), daemon=True)
                TRANS_WORKERS[sid] = t
                t.start()
            TRANS_VER[sid] = TRANS_VER.get(sid, 0) + 1
            TRANS_PENDING[sid] = seq_key
            TRANS_LAST[sid] = time.time()
            TRANS_EVENT[sid].set()
        res["translation_pending"] = True
    else:
        res["translation_pending"] = False
    # 优先返回已缓存的高质量【中文】翻译；没有（新句子的首两次点击）
    # 才留空，绝不用英文图标 id 当预览（那会被误认成"翻译变英文"）。
    res["partial_translation"] = TRANS_CACHE.get(sid, "")
    return res


# 8B 翻译专用 CUDA stream 在 _startup() 里按翻译器所在 GPU 创建
# （cuda:1 / 物理 GPU2），与快速预测（cuda:0，默认 stream）并发互不排队，
# 保证点击预测不被后台翻译拖慢。占位声明见上方全局变量。


# 空闲多久（秒）无新点击后，该会话的常驻翻译线程自动退出，避免线程无限堆积。
# 用户再次点击时 api_add 会按需重建（见 api_add 里的 is_alive 判断）。
TRANS_IDLE_EXIT = 600


def _translate_loop(sid: str):
    """每会话一个常驻后台线程：用户停手(~0.7s)后才翻译最新序列（防抖）。

    - 连续点击期间本线程只在 ev 上休眠，绝不跑生成 → 点击永远即时、不争 GIL。
    - 自"最后一次点击"起满 0.7s 才翻译【最新那一串】，中途新点击会重置计时；
      因此连续点击只翻译最终串，不会每点一次都跑 4-5s 的生成（无浪费）。
    - 关键：翻译过某一版本号(version)后，不再对"同一版本"重复生成——否则线程会
      一直空转重翻同一句话，霸占 INFER_LOCK，把新会话的翻译卡到轮询超时之外，
      前端就永远停在"翻译中…"。新点击会令 version 自增，从而触发一次新翻译。
    - 若生成期间用户又点了新图标（version 变大），本次结果丢弃，下一轮循环翻译新串。
    """
    ev = TRANS_EVENT.get(sid)
    if ev is None:
        return
    last_done_ver = -1        # 本线程已成功翻译过的版本号
    while True:
        try:
            ev.wait(timeout=1.0)      # 等首个点击或空闲
            ev.clear()
            # 防抖：自"最后一次点击"起等满 0.7s；期间有新点击则 ev 唤醒并重算。
            while True:
                last = TRANS_LAST.get(sid, 0)
                remain = 0.7 - (time.time() - last)
                if remain > 0:
                    ev.wait(timeout=remain)   # 新点击 set ev 会提前唤醒 → 重算计时
                    ev.clear()
                    continue
                break
            with TRANS_LOCK:
                seq_key = TRANS_PENDING.get(sid)
                ver = TRANS_VER.get(sid, 0)
            if not seq_key:
                continue
            # 没有新版本（用户停手后已翻译过这串）→ 跳过，绝不空转重翻。
            # 同时判断是否空闲太久，是则退出线程，释放资源（用户下次点击会重建）。
            if ver == last_done_ver:
                if time.time() - TRANS_LAST.get(sid, 0) > TRANS_IDLE_EXIT:
                    break
                continue
            try:
                with INFER_LOCK:
                    PIPELINE.session_id = sid
                    text = PIPELINE.translator.translate(seq_key.split())
            except Exception as e:
                print(f"[TranslateWorker] sid={sid} failed: {e}")
                text = None
            with TRANS_LOCK:
                # 不写空翻译，避免覆盖已有的正确中文；且只写"仍是同一串"的结果。
                if text and text.strip() and TRANS_VER.get(sid, 0) == ver:
                    TRANS_CACHE[sid] = text
                    last_done_ver = ver   # 标记本版本已翻译，避免后续空转重翻
                    print(f"[TranslateWorker] sid={sid} ver={ver} -> {text}")
        except Exception as e:
            print(f"[TranslateLoop] sid={sid} error: {e}")
            continue


@app.get("/api/translation")
def api_translation(request: Request):
    """前端轮询：取该会话最新高质量翻译（后台线程异步产出）。"""
    sid = _sid_from_request(request)
    return {"partial_translation": TRANS_CACHE.get(sid, "")}


@app.post("/api/commit")
def api_commit(request: Request):
    sid = _sid_from_request(request)
    sess = _get_session(sid)
    with INFER_LOCK:
        PIPELINE.session_id = sid
        try:
            res = PIPELINE.commit_sequence(sess["state"], sess["history"], session_id=sid)
        except Exception as e:
            return JSONResponse({"error": f"inference failed: {e}"}, status_code=500)
    # 提交后本轮结束，清掉翻译缓存，下一句从干净的 label 预览开始。
    with TRANS_LOCK:
        TRANS_CACHE.pop(sid, None)
        TRANS_PENDING.pop(sid, None)
        TRANS_VER.pop(sid, None)
    return res


@app.post("/api/undo")
def api_undo(request: Request):
    sid = _sid_from_request(request)
    sess = _get_session(sid)
    with INFER_LOCK:
        try:
            res = PIPELINE.undo_icon(sess["state"], sess["history"])
        except Exception as e:
            return JSONResponse({"error": f"inference failed: {e}"}, status_code=500)
    # undo 后序列变了，清掉旧翻译缓存，并触发常驻线程刷新显示。
    with TRANS_LOCK:
        TRANS_CACHE.pop(sid, None)
        seq = sess["state"].current_sequence[-3:]
        seq_key = " ".join(seq)
        if len(seq) >= 2:
            if sid not in TRANS_WORKERS or not TRANS_WORKERS[sid].is_alive():
                TRANS_EVENT.setdefault(sid, threading.Event())
                t = threading.Thread(target=_translate_loop, args=(sid,), daemon=True)
                TRANS_WORKERS[sid] = t
                t.start()
            TRANS_VER[sid] = TRANS_VER.get(sid, 0) + 1
            TRANS_PENDING[sid] = seq_key
            TRANS_LAST[sid] = time.time()
            TRANS_EVENT[sid].set()
    return res


@app.post("/api/reset")
def api_reset(request: Request):
    sid = _sid_from_request(request)
    with SESSIONS_LOCK:
        SESSIONS[sid] = {
            "state": IncrementalState(max_seq_len=MAX_SEQ_LEN),
            "history": [],
        }
        if sid in _SESSION_ORDER:
            _SESSION_ORDER.remove(sid)
        _SESSION_ORDER.append(sid)
    # 清掉该会话的翻译缓存/待翻译，避免下一句误显示上一句的翻译。
    with TRANS_LOCK:
        TRANS_CACHE.pop(sid, None)
        TRANS_PENDING.pop(sid, None)
        TRANS_VER.pop(sid, None)
    return {"ok": True, "session_id": sid}


# ---- 托管前端静态文件（构建后的 dist）----
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")
else:
    @app.get("/")
    def api_root():
        return {"status": "ok", "note": "frontend/dist not built yet; API is up."}
