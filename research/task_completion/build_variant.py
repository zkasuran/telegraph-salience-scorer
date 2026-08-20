#!/usr/bin/env python3
"""Build one TASK_COMPLETION scoring-module variant.

Patches the transformer-blend constants + intent marker in module/src/lib.rs, builds
--features minilm at opt-level 3 (matching the CHAT_COMPLETION champion clone), and
copies the wasm to an output path. Restores lib.rs afterward so variants don't stack.

usage: build_variant.py <name> <out.wasm> A B LEX Q [--lex]
  A B LEX Q : EMB_A_W EMB_B_W EMB_LEX_W EMB_Q_W (floats). --lex builds lexical (no minilm).
"""
import os, re, shutil, subprocess, sys
from Crypto.Hash import keccak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "module", "src", "lib.rs")
WASM = os.path.join(ROOT, "module", "target", "wasm32-unknown-unknown", "release", "telegraph_scorer.wasm")

def patch(a, b, lex, q, lexical, soften):
    src = open(LIB).read()
    orig = src
    wemb = "0.0" if lexical else "0.1"
    patches = [("W_EMB", wemb), ("EMB_A_W", a), ("EMB_B_W", b), ("EMB_LEX_W", lex), ("EMB_Q_W", q)]
    if soften:
        # The CHAT_COMPLETION champion is topical: it keeps a confidently-wrong but
        # on-topic answer mid-pack. To rank like it (the traffic gate), soften the
        # correctness penalties to the set that won CHAT_COMPLETION (BUILD-chat.md).
        patches += [("M_CONTRA", "0.7"), ("M_NUM_WRONG", "0.78"), ("M_ORDER", "0.85"),
                    ("M_ENTITY", "0.72"), ("M_NEGCOV", "0.32"), ("M_NUM_MISS_BASE", "0.85"),
                    ("M_TWO_FACED", "0.8")]
    for name, val in patches:
        src, n = re.subn(rf"const {name}: f32 = [0-9.]+;", f"const {name}: f32 = {val};", src)
        assert n == 1, f"patch {name} matched {n}"
    src, n = re.subn(r'pub static TELEGRAPH_INTENT: \[u8; 32\] = \*b"[^"]{32}";',
                     'pub static TELEGRAPH_INTENT: [u8; 32] = *b"TASK_COMPLETION                 ";', src)
    assert n == 1, "marker patch"
    open(LIB, "w").write(src)
    return orig

def main():
    name, out = sys.argv[1], sys.argv[2]
    a, b, lex, q = sys.argv[3:7]
    lexical = "--lex" in sys.argv
    soften = "--soft" in sys.argv
    orig = patch(a, b, lex, q, lexical, soften)
    try:
        cmd = ["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"]
        if not lexical:
            cmd += ["--features", "minilm"]
        env = dict(os.environ, CARGO_PROFILE_RELEASE_OPT_LEVEL="3")
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, "module"), env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:]); sys.exit(1)
    finally:
        open(LIB, "w").write(orig)
    shutil.copy(WASM, out)
    d = open(out, "rb").read()
    h = keccak.new(digest_bits=256); h.update(d)
    print(f"{name}: {out} {len(d)} bytes keccak 0x{h.hexdigest()}")

if __name__ == "__main__":
    main()
