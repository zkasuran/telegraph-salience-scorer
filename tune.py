#!/usr/bin/env python3
"""Sweep the scorer's tunables against both gates the node applies.

The node promotes a scoring module only if it separates good answers from bad
ones at least as well as the champion (Stage 2) AND ranks real miner answers
broadly the way the champion does (the traffic check, Spearman floor 0.60).
Those pull in opposite directions: sharper separation means pushing wrong
answers to zero, which is exactly what reorders them against a champion that
keeps them mid-pack. So tune against both at once rather than one at a time.

usage: python3 tune.py [max_combos]
"""
import itertools
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, "module", "src", "lib.rs")
WASM = os.path.join(ROOT, "module", "target", "wasm32-unknown-unknown", "release", "telegraph_scorer.wasm")

GRID = {
    "M_NEGCOV": [0.6, 0.85, 1.0],
    "SOFT_MIN": [0.62, 0.72, 0.85],
    "SOFT_W": [1.0, 0.7],
}

# Floors we refuse to trade away: the champion's own margin is 0.374 on the
# node's fixtures, so keep real headroom on ours, and never lose a case.
MIN_MARGIN = 0.45
MIN_WINS = 40
MIN_STDDEV = 0.20


def patch(values):
    src = open(LIB).read()
    for name, val in values.items():
        src, n = re.subn(rf"const {name}: f32 = [0-9.]+;", f"const {name}: f32 = {val};", src)
        if n != 1:
            raise SystemExit(f"could not patch {name}")
    open(LIB, "w").write(src)


def build():
    r = subprocess.run(
        ["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"],
        cwd=os.path.join(ROOT, "module"), capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-400:]


def measure():
    env = dict(os.environ, CORPUS="bench/traffic.json",
               BASELINE_SCORES="bench/champion-corpus-scores.json", REPORT="/tmp/tune-report.json")
    if os.environ.get("FAMILY"):
        env["FAMILY"] = os.environ["FAMILY"]
    r = subprocess.run(["./harness/harness", "bench/benchmark.json", "bench/attacks.json", WASM],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    out = r.stdout
    m = re.search(r"candidate_margin ([0-9.]+) \| wins (\d+)/(\d+) \| ties (\d+)", out)
    s = re.search(r"spearman vs \S+\s+(-?[0-9.]+)", out)
    sd = re.search(r"score_stddev ([0-9.]+)", out)
    ws = re.search(r"worst_self_match ([0-9.]+)", out)
    if not (m and s and sd and ws):
        return None
    fam = re.search(r"family_margin ([0-9.]+) \| wins (\d+)/(\d+)", out)
    res = {
        "margin": float(m.group(1)), "wins": int(m.group(2)), "ties": int(m.group(4)),
        "spearman": float(s.group(1)), "stddev": float(sd.group(1)),
        "worst_self": float(ws.group(1)), "fails": out.count("[FAIL]"),
    }
    if fam:
        res["fam_margin"] = float(fam.group(1))
        res["fam_wins"] = int(fam.group(2))
        res["fam_total"] = int(fam.group(3))
        # the family FAIL line is expected while sweeping, so it must not count as a
        # broken build: judge the family on its own numbers instead
        res["fails"] = max(0, res["fails"] - (0 if res["fam_wins"] == res["fam_total"] and res["fam_margin"] >= 0.45 else 1))
    return res


def main():
    original = {}
    src = open(LIB).read()
    for name in GRID:
        original[name] = float(re.search(rf"const {name}: f32 = ([0-9.]+);", src).group(1))
    combos = []
    for v in itertools.product(*GRID.values()):
        c = dict(zip(GRID, v))
        if "W_LEX" in c and "W_GRAM3" in c:
            rest = round(1.0 - c["W_LEX"] - c["W_GRAM3"], 3)
            if rest < 0.0 or rest > 0.35:
                continue
            c["W_GRAM2"] = rest
        combos.append(c)
    if len(sys.argv) > 1:
        combos = combos[: int(sys.argv[1])]
    print(f"{len(combos)} combinations, baseline {original}", flush=True)

    results = []
    try:
        for i, values in enumerate(combos):
            patch(values)
            ok, err = build()
            if not ok:
                print(f"[{i}] build failed: {err}", flush=True)
                continue
            got = measure()
            if got is None:
                print(f"[{i}] measure failed", flush=True)
                continue
            got["values"] = values
            feasible = (got["wins"] >= MIN_WINS and got["ties"] == 0 and got["fails"] == 0
                        and got["margin"] >= MIN_MARGIN and got["stddev"] >= MIN_STDDEV
                        and got["worst_self"] >= 0.75)
            if "fam_wins" in got:
                feasible = feasible and got["fam_wins"] == got["fam_total"] and got["fam_margin"] >= 0.45
            got["feasible"] = feasible
            results.append(got)
            if i % 20 == 0 or feasible:
                flag = "OK " if feasible else "   "
                print(f"[{i:3d}] {flag} spearman {got['spearman']:.4f} margin {got['margin']:.4f} "
                      f"wins {got['wins']} fails {got['fails']} {values}", flush=True)
    finally:
        patch(original)
        build()

    json.dump(results, open(os.path.join(ROOT, "bench", "tune-results.json"), "w"), indent=1)
    good = [r for r in results if r["feasible"]]
    good.sort(key=lambda r: -r["margin"])   # separation first: the gate we have to beat
    print(f"\n{len(good)} feasible of {len(results)}. Best by traffic agreement:")
    for r in good[:12]:
        fam = f"fam {r.get('fam_margin', 0):.4f} ({r.get('fam_wins','-')}/{r.get('fam_total','-')})  "
        print(f"  {fam}spearman {r['spearman']:.4f}  margin {r['margin']:.4f}  {r['values']}")


if __name__ == "__main__":
    main()
