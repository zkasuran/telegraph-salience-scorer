#!/usr/bin/env python3
"""Within-cluster agreement: the part of the node's traffic gate that actually bites.

The node ranks all of an intent's real rows together, and those rows are a handful of
LLM miners answering the same request. Pooled over many requests the ranking is easy,
because a right answer to request A and a wasted answer to request B are far apart for
any scorer. Inside one request they are not: five fluent answers to the same question
sit within a few hundredths of each other, and whoever orders those differently from the
champion loses the gate. So this reports the pooled Spearman and the mean per-request
Spearman side by side, and the second one is the number to optimise.

usage: cluster.py /tmp/a-rawB.json [...]
"""
import glob, json, os, sys
import numpy as np

INTENTS = {"agent_task": "agent1", "language_generation": "langgen1",
           "task_completion": "taskcomp1", "web_search": "web2",
           "weather_forecast": "weather1"}


def load(p):
    return json.load(open(p))["scores"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    def rank(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return None if d == 0 else float((rx * ry).sum() / d)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("/tmp/a-raw*.json"))
    rows = {it: json.load(open(f"bench/traffic-{it}.json"))["rows"] for it in INTENTS}
    champ = {it: load(f"/tmp/i-champ-{c}.json") for it, c in INTENTS.items()
             if os.path.exists(f"/tmp/i-champ-{c}.json")}
    print(f"{'variant':10} " + " ".join(f"{i[:8]:>17}" for i in INTENTS))
    print(f"{'':10} " + " ".join(f"{'pooled/percluster':>17}" for _ in INTENTS))
    for p in paths:
        sc = load(p)
        cells = []
        for it in INTENTS:
            keys = sorted(k for k in sc if k.startswith(it + "|"))
            ours = [sc[k] for k in keys]
            theirs = [champ[it][k] for k in keys]
            pooled = spearman(ours, theirs) or 0.0
            byq = {}
            for k in keys:
                i = int(k.split("|")[1])
                byq.setdefault(rows[it][i]["q"], []).append(k)
            per = [spearman([sc[k] for k in ks], [champ[it][k] for k in ks]) for ks in byq.values()]
            per = [x for x in per if x is not None]
            cells.append(f"{pooled:7.3f}/{np.mean(per) if per else float('nan'):8.3f}")
        print(f"{os.path.basename(p)[2:-5]:10} " + " ".join(cells))


if __name__ == "__main__":
    main()
