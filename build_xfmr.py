#!/usr/bin/env python3
"""Build one transformer scoring variant (minilm feature, opt-level 3).

Patches the tunables in lib.rs, builds with the minilm feature, copies the wasm to
dist/xfmr/<label>.wasm and prints its keccak + size. Does NOT register (deploy/reclaim do
that). Config is a JSON dict of const overrides, e.g.:

  python3 build_xfmr.py AGENT_TASK '{"W_EMB":0.45,"EMB_A_W":0.28,"EMB_B_W":0.56,
    "EMB_LEX_W":0.16,"POST_ITERS":3,"M_CONTRA":0.7,"M_NUM_WRONG":0.78,"M_ORDER":0.85,
    "M_ENTITY":0.72,"M_NEGCOV":0.32,"M_NUM_MISS_BASE":0.85,"M_TWO_FACED":0.8}' agent_p3
"""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, "module", "src", "lib.rs")
WASM = os.path.join(ROOT, "module", "target", "wasm32-unknown-unknown", "release", "telegraph_scorer.wasm")
OUT = os.path.join(ROOT, "dist", "xfmr")


def patch(intent, values):
    src = open(LIB).read()
    for name, val in values.items():
        if name in ("POST_ITERS", "TIE_SRC", "STEP_SHARP"):
            src, n = re.subn(rf"const {name}: u32 = \d+;", f"const {name}: u32 = {int(val)};", src)
        elif name in ("TOK_SPAN", "MAXTOK"):
            # these live in minilm.rs and are usize
            mp = os.path.join(ROOT, "module", "src", "minilm.rs")
            msrc = open(mp).read()
            msrc, n = re.subn(rf"const {name}: usize = \d+;", f"const {name}: usize = {int(val)};", msrc)
            if n == 1:
                open(mp, "w").write(msrc)
            continue
        else:
            src, n = re.subn(rf"const {name}: f32 = [0-9.]+;", f"const {name}: f32 = {val};", src)
        if n != 1:
            raise SystemExit(f"could not patch {name} (matched {n})")
    padded = intent.ljust(32)
    src, n = re.subn(r'pub static TELEGRAPH_INTENT: \[u8; 32\] = \*b"[^"]{32}";',
                     f'pub static TELEGRAPH_INTENT: [u8; 32] = *b"{padded}";', src)
    if n != 1:
        raise SystemExit("could not patch the intent marker")
    open(LIB, "w").write(src)


def keccak(path):
    from Crypto.Hash import keccak as _k
    data = open(path, "rb").read()
    h = _k.new(digest_bits=256); h.update(data)
    return "0x" + h.hexdigest(), len(data)


def main():
    intent = sys.argv[1]
    values = json.loads(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else intent.lower()
    lexical = "--lexical" in sys.argv
    patch(intent, values)
    env = dict(os.environ, CARGO_PROFILE_RELEASE_OPT_LEVEL="3")
    cmd = ["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"]
    if not lexical:
        cmd += ["--features", "minilm"]
    r = subprocess.run(cmd, cwd=os.path.join(ROOT, "module"), env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:]); raise SystemExit("build failed")
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{label}.wasm")
    import shutil; shutil.copy(WASM, dst)
    h, size = keccak(dst)
    print(f"BUILT {dst} size={size} keccak={h}")


if __name__ == "__main__":
    main()
