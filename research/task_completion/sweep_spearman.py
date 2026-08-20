#!/usr/bin/env python3
"""Score every module over the TASK_COMPLETION corpus once, cache the per-row scores,
then compute Spearman of each candidate against the proxy champion (good.wasm) and the
lexical proxy (rust word-overlap). good.wasm is a NAMED PROXY: it is the CHAT_COMPLETION
champion (a MiniLM sentence transformer) used as a topical stand-in, NOT the real
TASK_COMPLETION champion, which is not downloadable.
"""
import json, os, subprocess, sys
ROOT = "/tmp/flag-TASK_COMPLETION"
H = os.path.join(ROOT, "harness", "harness")
CORPUS = os.path.join(ROOT, "bench", "traffic-task-sub.json")
BMIN = os.path.join(ROOT, "bench", "benchmark-min.json")
AMIN = os.path.join(ROOT, "bench", "attacks-min.json")
RUST = os.path.join(ROOT, "reference/rust-module/target/wasm32-unknown-unknown/release/scoring_module.wasm")
OUT = os.path.join(ROOT, "research/task_completion/scores")
os.makedirs(OUT, exist_ok=True)

MODS = {
    "good_proxy": os.path.join(ROOT, "reference/champion-good.wasm"),
    "rust_lexical": RUST,
    "anchor_old_node0.5454": os.path.join(ROOT, "dist/telegraph-salience-scorer-task_completion.wasm"),
    "V_BASE": os.path.join(ROOT, ".scratch/variants/tc_base.wasm"),
    "V_CHAT": os.path.join(ROOT, ".scratch/variants/tc_chat.wasm"),
    "V_Q": os.path.join(ROOT, ".scratch/variants/tc_q.wasm"),
}
_EXTRA = {
    "V_LEX": os.path.join(ROOT, ".scratch/variants/tc_lex.wasm"),
    "V_QLEX": os.path.join(ROOT, ".scratch/variants/tc_qlex.wasm"),
    "V_QHEAVY": os.path.join(ROOT, ".scratch/variants/tc_qheavy.wasm"),
}

def dump(name, path):
    outp = os.path.join(OUT, name + ".json")
    if os.path.exists(outp):
        return json.load(open(outp))
    env = dict(os.environ, CORPUS=CORPUS, DUMP_SCORES=outp)
    # rust as cheap mods[0]; module of interest LAST so DUMP_SCORES captures it.
    subprocess.run([H, BMIN, AMIN, RUST, path], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(outp))

def rankvec(x):
    idx = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0]*len(x); i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and x[idx[j+1]] == x[idx[i]]: j += 1
        avg = (i+j)/2.0 + 1
        for k in range(i, j+1): r[idx[k]] = avg
        i = j+1
    return r

def spearman(a, b):
    ra, rb = rankvec(a), rankvec(b)
    n = len(a); ma = sum(ra)/n; mb = sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da = sum((ra[i]-ma)**2 for i in range(n))**0.5
    db = sum((rb[i]-mb)**2 for i in range(n))**0.5
    return num/(da*db) if da*db else 0.0

def main():
    scores = {}
    for name, path in MODS.items():
        print("scoring", name, flush=True)
        scores[name] = dump(name, path)
        print("  ", len(scores[name]), "rows", flush=True)
    n = len(json.load(open(CORPUS))["rows"])
    good = scores["good_proxy"]; rust = scores["rust_lexical"]
    res = {}
    for name, s in scores.items():
        if len(s) != n:
            res[name] = {"rows": len(s), "error": "row count mismatch"}; continue
        res[name] = {"vs_good_proxy": round(spearman(s, good), 4),
                     "vs_rust_lexical": round(spearman(s, rust), 4)}
    json.dump({"corpus": os.path.basename(CORPUS), "rows": n,
               "proxy_note": "good_proxy = CHAT_COMPLETION champion good.wasm (MiniLM transformer), a TOPICAL PROXY, not the real TASK_COMPLETION champion",
               "spearman": res}, open(os.path.join(ROOT, "research/task_completion/spearman.json"), "w"), indent=1)
    for name, r in res.items():
        print(f"{name:26s} {r}", flush=True)

if __name__ == "__main__":
    main()
