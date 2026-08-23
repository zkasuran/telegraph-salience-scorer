#!/usr/bin/env python3
"""Build a named set of scoring variants from explicit full configs.

build_xfmr.py patches only the constants it is given, so a variant built after another
inherits whatever the previous one left in lib.rs. That is how a build ends up scoring
differently from what its label claims. This driver always passes every constant that
matters, so a config here is the whole module, and two runs of the same name give the
same binary.

    python3 variants.py <name> [<name> ...]        builds dist/xfmr/<name>.wasm
    python3 variants.py --list
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Neutral base: the lexical/correctness machinery all switched off, so a variant only
# has what it turns on. STEP_T 0 means the plain score, no threshold calibration.
BASE = {
    "W_LEX": 0.76, "W_GRAM3": 0.2, "W_GRAM2": 0.04, "F_BETA2": 0.36, "P_CONCAVE": 1.0,
    "R_KEY_BASE": 0.5, "R_FLOOR": 0.3,
    "M_CONTRA": 1.0, "M_TWO_FACED": 1.0, "M_SILENT": 1.0, "B_AGREE": 0.0,
    "M_NUM_MISS_BASE": 1.0, "M_NUM_WRONG": 1.0, "M_NUM_MATCH": 0.0,
    "M_ORDER": 1.0, "M_ENTITY": 1.0, "M_NEGCOV": 0.0,
    "SHARPEN": 0.0, "SOFT_W": 1.0, "SOFT_MIN": 0.72, "SOFT_CAP_FRAC": 0.35,
    "W_EMB": 0.45, "EMB_A_W": 0.0, "EMB_B_W": 1.0, "EMB_LEX_W": 0.0, "EMB_L2_W": 0.0, "EMB_L4_W": 0.0, "GATE_LEX": 0.0,
    "POST_ITERS": 0, "POST_PIVOT": 0.5, "POST_FRAC": 0.0, "SIGK": 0.0, "SIGC": 0.4545,
    "STEP_T": 0.0, "STEP_B": 0.02, "STEP_R": 0.0, "STEP_W": 0.0, "STEP_SHARP": 0, "STEP_PIVOT": 0.5, "W_QA": 0.0, "TOK_SPAN": 1, "NOGT_Q": 0.0, "TIE_SRC": 0, "EXACT_TIE": 0.0, "NUM_TOL": 0.0,
}

# The correctness penalties the promoted CHAT_COMPLETION build ran with. They only fire
# on an answer that contradicts, drops a figure or reorders the ground truth, so they
# cost little agreement on real traffic and buy separation on the fixtures.
PEN = {"M_CONTRA": 0.7, "M_NUM_WRONG": 0.78, "M_ORDER": 0.85, "M_ENTITY": 0.72,
       "M_NEGCOV": 0.32, "M_NUM_MISS_BASE": 0.85, "M_TWO_FACED": 0.8}

# Blends. embA is the shallow embedding-layer cosine, embB the full transformer cosine,
# lex our own lexical score. The champion's own structure is 0.25/0.50/0.25.
BLEND_P = {"EMB_A_W": 0.28, "EMB_B_W": 0.56, "EMB_LEX_W": 0.16}
BLEND_R = {"EMB_A_W": 0.25, "EMB_B_W": 0.50, "EMB_LEX_W": 0.25}
BLEND_B = {"EMB_A_W": 0.0, "EMB_B_W": 1.0, "EMB_LEX_W": 0.0}
BLEND_AB = {"EMB_A_W": 0.33, "EMB_B_W": 0.67, "EMB_LEX_W": 0.0}

VARIANTS = {
    # raw rankings, no threshold calibration: what the agreement gate actually sees
    "rawB":    dict(BASE, **BLEND_B),
    "rawP":    dict(BASE, **BLEND_P, **PEN),
    "rawPnp":  dict(BASE, **BLEND_P),
    "rawR":    dict(BASE, **BLEND_R),
    "rawRp":   dict(BASE, **BLEND_R, **PEN),
    "rawAB":   dict(BASE, **BLEND_AB),
    "rawBp":   dict(BASE, **BLEND_B, **PEN),
    # Lexical gate: the topical score multiplied by clamp01(lex / GATE_LEX). Real answers
    # all carry enough word overlap to saturate the gate, so their ranking stays the pure
    # topical one the agreement gate is measured against, while a fixture's off-topic bad
    # answer has almost no overlap and is pushed to the bottom. Separation for free.
    "rawG20":  dict(BASE, **BLEND_B, GATE_LEX=0.20),
    "rawG35":  dict(BASE, **BLEND_B, GATE_LEX=0.35),
    "rawG50":  dict(BASE, **BLEND_B, GATE_LEX=0.50),
    "rawG35p": dict(BASE, **BLEND_B, **PEN, GATE_LEX=0.35),
    "rawPG35": dict(BASE, **BLEND_P, GATE_LEX=0.35),
    # lexical-only reference (no transformer), for comparison
    "rawLex":  dict(BASE, W_EMB=0.0),
}


def build(name, intent, label=None, extra=None):
    cfg = dict(VARIANTS[name])
    if extra:
        cfg.update(extra)
    label = label or name
    r = subprocess.run([sys.executable, os.path.join(ROOT, "build_xfmr.py"), intent,
                        json.dumps(cfg), label], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip() or r.stderr[-400:])
    return r.returncode == 0


def main():
    if "--list" in sys.argv:
        for k, v in VARIANTS.items():
            print(k, {a: b for a, b in v.items() if b != BASE.get(a)})
        return
    args = sys.argv[1:]
    intent = os.environ.get("INTENT", "AGENT_TASK")
    step = os.environ.get("STEP_T")
    for name in args:
        extra = {"STEP_T": float(step)} if step else None
        label = name if not step else f"{name}_s{str(step).replace('.','')}"
        build(name, intent, label, extra)


if __name__ == "__main__":
    main()
