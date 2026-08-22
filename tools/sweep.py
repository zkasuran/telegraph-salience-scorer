#!/usr/bin/env python3
"""Offline sweep over monotone post-transforms of a dumped raw score.

The node measures two things: separation on its hidden fixtures (mean_good - mean_bad)
and rank agreement with the champion on real traffic (Spearman, floor 0.60). Both are
functions of the raw score, so one dump per binary is enough to evaluate any
post-transform without rebuilding: separation changes with the transform, agreement
does not (any strictly increasing map has the same ranking) as long as f32 keeps
distinct scores distinct. This script measures both, in float32, exactly the way the
module would compute them.

usage: sweep.py <candidate-dump.json> <champion-dump.json> [--rows rows.json]
"""
import json, sys
import numpy as np

F = np.float32


def load(path):
    d = json.load(open(path))
    return d["scores"]


def split(scores):
    good, bad, self_, traffic, ids = [], [], [], [], []
    for k, v in scores.items():
        if k.startswith("b:") and k.endswith(":good"):
            good.append((k, v))
        elif k.startswith("b:") and k.endswith(":bad"):
            bad.append((k, v))
        elif k.startswith("b:") and k.endswith(":self"):
            self_.append((k, v))
        elif k.startswith("t:"):
            traffic.append((k, v))
    good.sort(); bad.sort(); self_.sort(); traffic.sort()
    return good, bad, self_, traffic


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return 0.0 if d == 0 else float((rx * ry).sum() / d)


def step_tb(x, t, b):
    """The module's transform, in f32: hard step at t plus b of the raw score, so the
    ranking survives (strictly increasing) while the two clusters land at the ends."""
    x = np.asarray(x, dtype=np.float32)
    h = (x >= F(t)).astype(np.float32)
    return (F(1.0) - F(b)) * h + F(b) * x


def smoothstep_iters(x, n, frac=0.0):
    x = np.asarray(x, dtype=np.float32)
    for _ in range(n):
        x = x * x * (F(3.0) - F(2.0) * x)
    if frac:
        s = x * x * (F(3.0) - F(2.0) * x)
        x = x + F(frac) * (s - x)
    return x


def main():
    cand = load(sys.argv[1])
    champ = load(sys.argv[2])
    cg, cb, cs, ct = split(cand)
    hg, hb, hs, ht = split(champ)
    g = np.array([v for _, v in cg], dtype=np.float32)
    b = np.array([v for _, v in cb], dtype=np.float32)
    tr = np.array([v for _, v in ct], dtype=np.float32)
    htr = np.array([v for _, v in ht], dtype=np.float32)
    hgg = np.array([v for _, v in hg], dtype=np.float32)
    hbb = np.array([v for _, v in hb], dtype=np.float32)

    print(f"cases {len(g)}  traffic rows {len(tr)}")
    print(f"CANDIDATE raw: margin {g.mean()-b.mean():.4f} | good {g.min():.3f}..{g.max():.3f} "
          f"mean {g.mean():.3f} | bad {b.min():.3f}..{b.max():.3f} mean {b.mean():.3f} "
          f"| per-case wins {(g>b).sum()}/{len(g)}")
    print(f"CHAMPION  raw: margin {hgg.mean()-hbb.mean():.4f} | good {hgg.min():.3f}..{hgg.max():.3f} "
          f"mean {hgg.mean():.3f} | bad {hbb.min():.3f}..{hbb.max():.3f} mean {hbb.mean():.3f} "
          f"| per-case wins {(hgg>hbb).sum()}/{len(hgg)}")
    base_rho = spearman(tr, htr)
    print(f"base traffic agreement (raw vs champion): {base_rho:.4f}")

    # ROC: the margin a hard step at t would earn, and what the champion's own score
    # would earn under the same treatment.
    cuts = np.unique(np.concatenate([g, b, hgg, hbb]))
    print("\n=== hard-step ROC on the fixtures (candidate) ===")
    rows = []
    for t in np.unique(np.round(np.arange(0.02, 1.0, 0.02), 3)):
        m = float((g >= F(t)).mean() - (b >= F(t)).mean())
        rows.append((t, m))
    best = max(rows, key=lambda r: r[1])
    for t, m in rows:
        if m > 0:
            bar = "#" * int(m * 40)
            print(f"  t={t:4.2f}  step margin {m:.4f} {bar}")
    print(f"  best hard step: t={best[0]:.2f} margin {best[1]:.4f}  (champion to beat 0.78587)")

    print("\n=== champion's own score under a hard step (upper bound if we could rank like it) ===")
    hrows = [(t, float((hgg >= F(t)).mean() - (hbb >= F(t)).mean()))
             for t in np.unique(np.round(np.arange(0.02, 1.0, 0.02), 3))]
    hbest = max(hrows, key=lambda r: r[1])
    print(f"  best t={hbest[0]:.2f} margin {hbest[1]:.4f}")

    print("\n=== step + tie-break: margin and preserved agreement ===")
    for t in [best[0] - 0.04, best[0] - 0.02, best[0], best[0] + 0.02]:
        for bb_ in [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]:
            mg = step_tb(g, t, bb_)
            mb = step_tb(b, t, bb_)
            mtr = step_tb(tr, t, bb_)
            rho = spearman(mtr, htr)
            ties = len(mtr) - len(np.unique(mtr))
            print(f"  t={t:4.2f} B={bb_:4.2f}  margin {float(mg.mean()-mb.mean()):.4f} "
                  f"agreement {rho:.4f} (base {base_rho:.4f}) f32 ties {ties}")

    print("\n=== for reference: iterated smoothstep (what the last round shipped) ===")
    for n in [3, 5, 6, 7]:
        mg, mb, mtr = smoothstep_iters(g, n), smoothstep_iters(b, n), smoothstep_iters(tr, n)
        rho = spearman(mtr, htr)
        ties = len(mtr) - len(np.unique(mtr))
        print(f"  iters={n}  margin {float(mg.mean()-mb.mean()):.4f} agreement {rho:.4f} f32 ties {ties}")


if __name__ == "__main__":
    main()
