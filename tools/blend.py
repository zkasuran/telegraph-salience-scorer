#!/usr/bin/env python3
"""Grid-search the blend weights offline, against the champion, per intent.

Every term in our score is dumped separately (cb the transformer cosine, ca the shallow
one, cq the answer-to-question cosine, lex the lexical F-beta, r the ground-truth recall),
so any linear blend of them can be scored without another build: the module computes
clamp01 of exactly this sum. The objective is the mean per-request Spearman against the
champion, which is what the node's agreement gate measures.

usage: blend.py [prefix]        prefix v = the verbose corpus, a = the concise one
"""
import itertools, json, os, sys
import numpy as np

INTENTS = {"agent_task": "agent1", "language_generation": "langgen1",
           "task_completion": "taskcomp1", "web_search": "web2",
           "weather_forecast": "weather1"}
FEATS = {"cb": "rawB", "ca": "rawA", "cq": "rawQ100", "lex": "rawLex", "r": "rawRecall",
         "c2": "rawL2", "c4": "rawL4"}


def rk(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def sp(x, y):
    if len(x) < 3:
        return None
    a, b = rk(np.asarray(x, float)), rk(np.asarray(y, float))
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return None if d == 0 else float((a * b).sum() / d)


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "v"
    tpre = "bench/traffic-verbose-" if pre == "v" else "bench/traffic-"
    rows = {it: json.load(open(f"{tpre}{it}.json"))["rows"] for it in INTENTS}
    F = {}
    for k, mod in FEATS.items():
        p = f"/tmp/{pre}-{mod}.json"
        if os.path.exists(p):
            F[k] = json.load(open(p))["scores"]
    print("features present:", sorted(F))
    champ = {it: json.load(open(f"/tmp/{pre}-champ-{c}.json"))["scores"]
             for it, c in INTENTS.items() if os.path.exists(f"/tmp/{pre}-champ-{c}.json")}
    keys = {it: sorted(k for k in champ[it] if k.startswith(it + "|")) for it in champ}
    groups = {}
    for it in champ:
        g = {}
        for k in keys[it]:
            g.setdefault(rows[it][int(k.split("|")[1])]["q"], []).append(k)
        groups[it] = list(g.values())

    def score(w):
        out = {}
        for it in champ:
            vals = []
            for grp in groups[it]:
                x = [min(1.0, sum(w.get(f, 0.0) * F[f][k] for f in w if f in F)) for k in grp]
                v = sp(x, [champ[it][k] for k in grp])
                if v is not None:
                    vals.append(v)
            out[it] = float(np.mean(vals)) if vals else float("nan")
        return out

    grid = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
    best = {it: (-2, None) for it in champ}
    rows_out = []
    for cb in grid:
        for cq in grid:
            for lex in grid:
                for r in (0.0, 0.15, 0.3):
                    if cb + cq + lex + r == 0:
                        continue
                    w = {"cb": cb, "cq": cq, "lex": lex, "r": r}
                    s = score(w)
                    rows_out.append((w, s))
                    for it in champ:
                        if s[it] > best[it][0]:
                            best[it] = (s[it], dict(w))
    print(f"\n{'intent':22} {'best per-request rho':>20}  weights")
    for it in champ:
        v, w = best[it]
        print(f"{it:22} {v:20.3f}  " + " ".join(f"{a}={b}" for a, b in w.items() if b))
    # a single blend that is good everywhere, since one build should serve several intents
    def mean_of(s):
        v = [s[it] for it in champ if not np.isnan(s[it])]
        return float(np.mean(v))
    rows_out.sort(key=lambda t: -mean_of(t[1]))
    print("\ntop blends by mean over intents:")
    for w, s in rows_out[:8]:
        print("  " + " ".join(f"{a}={b}" for a, b in w.items() if b)
              + "   " + " ".join(f"{it[:8]}={s[it]:.3f}" for it in champ))


if __name__ == "__main__":
    main()
