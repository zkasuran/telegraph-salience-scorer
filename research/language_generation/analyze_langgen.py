#!/usr/bin/env python3
"""Analyse the candidate's ranking of the LANGUAGE_GENERATION corpus against the
hypothesised champion axis (answer length / detail) and against the two axes the
node already proved the champion does NOT use (precision word-overlap to ground
truth). The real champion is not downloadable, so length is the named PROXY; the
node read is the only true test.
"""
import json, os, subprocess, sys, math

ROOT = "/tmp/flag-LANGUAGE_GENERATION"
CORPUS = os.path.join(ROOT, "bench", "traffic-langgen.json")
WASM = sys.argv[1] if len(sys.argv) > 1 else "/tmp/langgen-w45.wasm"

rows = json.load(open(CORPUS))["rows"]

def probe(q, gt, a):
    out = subprocess.run([os.path.join(ROOT, "harness", "harness"),
                          os.path.join(ROOT, "bench", "benchmark.json"),
                          os.path.join(ROOT, "bench", "attacks.json"), WASM],
                         env={**os.environ, "PROBE": f"{q}|{gt}|{a}"},
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts:
            try:
                return float(parts[-1])
            except ValueError:
                continue
    return None

def word_overlap(a, gt):
    aw = a.split(); gw = set(w.lower() for w in gt.split())
    if not aw: return 0.0
    return sum(1 for w in aw if w.lower() in gw) / len(aw)

def gt_recall(a, gt):
    gw = gt.split(); aw = set(w.lower() for w in a.split())
    if not gw: return 0.0
    return sum(1 for w in gw if w.lower() in aw) / len(gw)

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[order[j+1]] == v[order[i]]: j += 1
        avg = (i+j)/2.0 + 1
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r

def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x); mx = sum(rx)/n; my = sum(ry)/n
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx = math.sqrt(sum((a-mx)**2 for a in rx)); dy = math.sqrt(sum((b-my)**2 for b in ry))
    return num/(dx*dy) if dx and dy else 0.0

cand, length, overlap, recall = [], [], [], []
for r in rows:
    s = probe(r["q"], r["gt"], r["a"])
    if s is None: continue
    cand.append(s)
    length.append(float(len(r["a"].split())))
    overlap.append(word_overlap(r["a"], r["gt"]))
    recall.append(gt_recall(r["a"], r["gt"]))

res = {
    "n": len(cand),
    "wasm": os.path.basename(WASM),
    "spearman_candidate_vs_length": round(spearman(cand, length), 4),
    "spearman_candidate_vs_word_overlap_precision": round(spearman(cand, overlap), 4),
    "spearman_candidate_vs_gt_recall": round(spearman(cand, recall), 4),
    "spearman_length_vs_overlap": round(spearman(length, overlap), 4),
    "cand_score_min": round(min(cand), 4), "cand_score_max": round(max(cand), 4),
    "cand_score_stddev": round((sum((x-sum(cand)/len(cand))**2 for x in cand)/len(cand))**0.5, 4),
}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "bench", "langgen-agreement.json"), "w"), indent=1)
