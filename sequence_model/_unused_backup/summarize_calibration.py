#!/usr/bin/env python3
"""Summarize calibration_judge.jsonl: score distribution + samples for human review."""
import json
from collections import Counter

rows = [json.loads(l) for l in open("sequence_model/calibration_judge.jsonl")]
good = [r for r in rows if r["set"] == "good"]
bad = [r for r in rows if r["set"] == "bad"]

def dist(rs):
    c = Counter(r.get("label") for r in rs)
    avg = sum(r.get("score", 0) for r in rs) / max(len(rs), 1)
    return dict(c), round(avg, 2)

gd, ga = dist(good)
bd, ba = dist(bad)
print(f"=== CALIBRATION SUMMARY (n_good={len(good)}, n_bad={len(bad)}) ===")
print(f"GOOD (known-clean 1806): label={gd}  avg_score={ga}")
print(f"BAD  (corrupted)      : label={bd}  avg_score={ba}")
print(f"\n--- 10 GOOD samples ---")
for r in good[:10]:
    print(f"  [{r.get('score')}] {r.get('label')} | {r['sequence']} | {r.get('reason','')[:50]}")
print(f"\n--- 10 BAD samples ---")
for r in bad[:10]:
    print(f"  [{r.get('score')}] {r.get('label')} | {r['sequence']} | {r.get('reason','')[:50]}")
