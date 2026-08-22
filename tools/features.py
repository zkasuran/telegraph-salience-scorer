#!/usr/bin/env python3
"""Which of our signals orders answers the way the champion does, inside one request?

The agreement gate ranks all of an intent's rows together, but the rows are a handful of
LLM miners answering the same request, so the ranking that decides the gate is the ranking
inside each request. This takes every feature we can compute (each one a dumped build) and
reports its mean per-request Spearman against the champion, per intent, on whichever traffic
corpus is passed. That says what the champion is actually discriminating on when every
answer in front of it is fluent and roughly right.

usage: features.py <corpus-prefix e.g. v> <traffic-file-prefix e.g. bench/traffic-verbose->
"""
import glob, json, os, sys
import numpy as np

INTENTS = {"agent_task": "agent1", "language_generation": "langgen1",
           "task_completion": "taskcomp1", "web_search": "web2",
           "weather_forecast": "weather1"}


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    def rk(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = rk(x), rk(y); a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return None if d == 0 else float((a * b).sum() / d)


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "v"
    tpre = sys.argv[2] if len(sys.argv) > 2 else "bench/traffic-verbose-"
    rows = {it: json.load(open(f"{tpre}{it}.json"))["rows"] for it in INTENTS}
    champ = {}
    for it, c in INTENTS.items():
        p = f"/tmp/{pre}-champ-{c}.json"
        if os.path.exists(p):
            champ[it] = json.load(open(p))["scores"]
    feats = {}
    for p in sorted(glob.glob(f"/tmp/{pre}-raw*.json")):
        feats[os.path.basename(p)[len(pre) + 1:-5]] = json.load(open(p))["scores"]
    # a length feature costs nothing and is the thing verbosity moves most
    print(f"{'feature':12} " + " ".join(f"{i[:10]:>12}" for i in INTENTS))
    for name in list(feats) + ["len", "-len"]:
        cells = []
        for it in INTENTS:
            if it not in champ:
                cells.append(f"{'--':>12}"); continue
            byq = {}
            for i, r in enumerate(rows[it]):
                byq.setdefault(r["q"], []).append(f"{it}|{i:03d}")
            vals = []
            for ks in byq.values():
                if name == "len":
                    x = [len(rows[it][int(k.split('|')[1])]["a"]) for k in ks]
                elif name == "-len":
                    x = [-len(rows[it][int(k.split('|')[1])]["a"]) for k in ks]
                else:
                    if any(k not in feats[name] for k in ks):
                        x = None
                    else:
                        x = [feats[name][k] for k in ks]
                if x is None:
                    continue
                v = sp(x, [champ[it][k] for k in ks])
                if v is not None:
                    vals.append(v)
            cells.append(f"{np.mean(vals):12.3f}" if vals else f"{'--':>12}")
        print(f"{name:12} " + " ".join(cells))


if __name__ == "__main__":
    main()
