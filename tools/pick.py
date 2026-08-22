#!/usr/bin/env python3
"""Pick the raw ranking to ship, from local dumps.

Two numbers decide a build and both come off a raw-score dump, because the threshold
calibration on top is monotone: separation is the ROC of the raw score on the fixtures
(a hard step at the best threshold), and agreement is the raw score's Spearman against
the champion on that intent's traffic. So this reads one dump per variant and prints
both, per intent, without another build or another registration.

usage: pick.py /tmp/a-rawB.json /tmp/a-rawP.json ...
"""
import glob, json, os, sys
import numpy as np

F = np.float32
INTENTS = {"agent_task": "agent1", "language_generation": "langgen1",
           "task_completion": "taskcomp1", "web_search": "web2",
           "weather_forecast": "weather1"}


def load(p):
    return json.load(open(p))["scores"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
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
    return 0.0 if d == 0 else float((rx * ry).sum() / d)


def roc(good, bad):
    """Best hard-step separation and the threshold that gets it. This is the ceiling any
    monotone transform can reach, and the step reaches it."""
    good, bad = np.asarray(good, np.float32), np.asarray(bad, np.float32)
    cuts = np.unique(np.concatenate([good, bad]))
    best = (0.0, 0.0)
    for t in cuts:
        m = float((good >= t).mean() - (bad >= t).mean())
        if m > best[1]:
            best = (float(t), m)
    return best


def fixture(scores, prefix="tp:"):
    g = [v for k, v in sorted(scores.items()) if k.startswith(prefix) and k.endswith(":good")]
    b = [v for k, v in sorted(scores.items()) if k.startswith(prefix) and k.endswith(":bad")]
    return np.array(g, np.float32), np.array(b, np.float32)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("/tmp/a-raw*.json"))
    champ_traffic = {}
    champ_fix = {}
    for it, c in INTENTS.items():
        p = f"/tmp/i-champ-{c}.json"
        if os.path.exists(p):
            champ_traffic[it] = load(p)
        p2 = f"/tmp/tp-champ-{c}.json"
        if os.path.exists(p2):
            champ_fix[it] = load(p2)

    for it, sc in champ_fix.items():
        g, b = fixture(sc)
        t, m = roc(g, b)
        print(f"champion {it:20} topical fixtures: raw margin {g.mean()-b.mean():.4f} "
              f"good {g.mean():.3f} bad {b.mean():.3f} wins {(g>b).sum()}/{len(g)} "
              f"| best step t={t:.3f} margin {m:.4f}")
    print()
    hdr = f"{'variant':10} {'fix.raw':>8} {'step':>7} {'t*':>6} " + " ".join(f"{i[:9]:>10}" for i in INTENTS)
    print(hdr)
    for p in paths:
        sc = load(p)
        name = os.path.basename(p)[2:-5]
        g, b = fixture(sc)
        t, m = roc(g, b) if len(g) else (0, 0)
        cells = []
        for it in INTENTS:
            if it not in champ_traffic:
                cells.append("     --   "); continue
            keys = sorted(k for k in sc if k.startswith(it + "|"))
            ours = [sc[k] for k in keys]
            theirs = [champ_traffic[it][k] for k in keys]
            cells.append(f"{spearman(ours, theirs):10.4f}")
        print(f"{name:10} {float(g.mean()-b.mean()):8.4f} {m:7.4f} {t:6.3f} " + " ".join(cells))
    print("\nagreement floor 0.60; node champion separation to beat 0.78587 "
          "(step margin here is on our topical proxy, not the node's fixtures)")


if __name__ == "__main__":
    main()
