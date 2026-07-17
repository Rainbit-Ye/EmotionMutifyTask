#!/usr/bin/env python3
"""
Full calibration of the ollama LLM-as-judge (qwen3.6:27b, AAC-functional prompt)
against the 1806 clean/reaL dataset.

- 200 known-GOOD sequences (from 1806) -> judge should rate 合理/high.
- 50 synthetically corrupted -> judge should rate 不合理/low.
Runs in parallel, saves incrementally to calibration_judge.jsonl.
"""
import json, ast, re, random, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA = "http://172.31.226.24:4433/v1/chat/completions"
MODEL = "qwen3.6:27b"
SRC = "/home/user1/liuduanye/AAC2TextAnnotator/data/sft_high_for_review.json"
OUT = "sequence_model/calibration_judge.jsonl"
N_GOOD = 200
N_BAD = 50
WORKERS = 4

PROMPT = """你是一名辅助沟通(AAC)语料评审员。AAC 使用者常有沟通障碍,表达常出现非常规、跨类别的图标组合;只要"一个沟通障碍者可能想表达这个意思"就视为合理,不要按文学连贯性苛求。

图标序列: {seq}
参考英文表达: {sent}

请只输出 JSON: {{"label":"合理"或"不合理","score":1到5,"reason":"一句话"}}。
5=非常合理可表达;3=可接受的非常规表达;1=完全无意义。"""

def parse_labels(s):
    if isinstance(s, list):
        return [str(x) for x in s]
    if not isinstance(s, str):
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return re.findall(r"[A-Za-z0-9_]+", s)

def call_judge(seq, sent):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(seq=", ".join(seq), sent=sent or "")}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode()
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=400) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            return json.loads(txt)
        except Exception as e:
            last = e
    return {"label": "ERROR", "score": -1, "reason": str(last)[:200]}

def corrupt(seq):
    if len(seq) >= 3 and random.random() < 0.5:
        s = seq[:]; random.shuffle(s); return s
    pool = ["apple", "car", "sleep_to", "school", "angry", "computer", "birthday", "swim_to"]
    s = seq[:]; i = random.randrange(1, len(s)); s[i] = random.choice(pool); return s

def main():
    data = json.load(open(SRC))
    jobs = []
    for e in data[:N_GOOD]:
        seq = parse_labels(e.get("labels"))
        if seq:
            jobs.append(("good", seq, e.get("sentence_en", ""), e.get("id")))
    random.seed(0)
    for e in data[N_GOOD:N_GOOD + N_BAD]:
        seq = parse_labels(e.get("labels"))
        if seq:
            jobs.append(("bad", corrupt(seq), "", e.get("id")))

    fout = open(OUT, "w")
    print(f"judging {len(jobs)} seqs | model={MODEL} | workers={WORKERS}")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call_judge, seq, sent): (kind, seq, sent, eid)
                for kind, seq, sent, eid in jobs}
        for f in as_completed(futs):
            kind, seq, sent, eid = futs[f]
            r = f.result()
            rec = {"set": kind, "id": eid, "sequence": seq, "sentence_en": sent, **r}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}")
    fout.close()
    print(f"done -> {OUT}")

if __name__ == "__main__":
    main()
