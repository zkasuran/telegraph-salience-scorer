#!/usr/bin/env python3
"""Build, gate and register one scoring-module variant per intent.

Every registration is a separate build: the intent it serves is baked into the
binary (TELEGRAPH_INTENT) and the tunables are set for the shape of answer that
intent returns. Nothing is registered that has not passed the local harness
first, which is the same set of gates the node applies.

usage: python3 deploy.py <intent> [<intent> ...]      (dry run without --send)
       python3 deploy.py --send <intent> [<intent> ...]
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, "module", "src", "lib.rs")
WASM = os.path.join(ROOT, "module", "target", "wasm32-unknown-unknown", "release", "telegraph_scorer.wasm")
DIAMOND = "0x5a2324aA18613FAD4e44bDF0d6c73Ec1f6D87ff8"
RPC = "https://sepolia.base.org"
WALLET_ENV = os.path.join(ROOT, "..", ".wallet.env")
COOKIE = "/tmp/tg/cookie.txt"
LEDGER = os.path.join(ROOT, "bench", "registrations.json")

# The base configuration is the one swept in tune.py: best rank agreement with the
# live champion among configurations that still win every benchmark case.
BASE = {
    "F_BETA2": 0.36, "P_CONCAVE": 1.0, "W_LEX": 0.76, "W_GRAM3": 0.20, "W_GRAM2": 0.04,
    "M_CONTRA": 0.3, "M_SILENT": 0.95, "B_AGREE": 0.35, "R_KEY_BASE": 0.5, "R_FLOOR": 0.3,
    "SHARPEN": 0.82, "M_NUM_MISS_BASE": 0.62, "M_NUM_WRONG": 0.45, "M_TWO_FACED": 0.5,
    "M_ORDER": 0.55, "SOFT_MIN": 0.55, "SOFT_W": 0.85, "SOFT_CAP_FRAC": 0.5,
}

# Per-shape overrides. A verdict intent lives or dies on polarity, so contradicting
# the ground truth costs more and agreeing with it earns more. A reference intent
# turns on naming the right entity, so coverage of the ground truth counts for more
# than brevity. A numeric intent is decided by the figure.
PROFILES = {
    "verdict": {"M_CONTRA": 0.15, "B_AGREE": 0.45},
    "reference": {"F_BETA2": 0.6, "R_KEY_BASE": 0.6},
    "numeric": {"M_NUM_WRONG": 0.3, "M_NUM_MISS_BASE": 0.5},
    "text": {},
}

TARGETS = {
    "AI_TEXT_DETECTION": "verdict",
    "FACT_CHECK": "verdict",
    "URL_SCAN": "verdict",
    "DEEPFAKE_DETECTION": "verdict",
    "SSL_VERIFICATION": "verdict",
    "SENTIMENT_ANALYSIS": "verdict",
    "CVE_LOOKUP": "reference",
    "ACADEMIC_SEARCH": "reference",
    "CHAT_COMPLETION": "text",
}


def patch(intent, values):
    src = open(LIB).read()
    for name, val in values.items():
        src, n = re.subn(rf"const {name}: f32 = [0-9.]+;", f"const {name}: f32 = {val};", src)
        if n != 1:
            raise SystemExit(f"could not patch {name}")
    padded = intent.ljust(32)
    if len(padded) != 32:
        raise SystemExit(f"intent name too long for the marker: {intent}")
    src, n = re.subn(r'pub static TELEGRAPH_INTENT: \[u8; 32\] = \*b"[^"]{32}";',
                     f'pub static TELEGRAPH_INTENT: [u8; 32] = *b"{padded}";', src)
    if n != 1:
        raise SystemExit("could not patch the intent marker")
    open(LIB, "w").write(src)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build_and_gate():
    r = run(["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"],
            cwd=os.path.join(ROOT, "module"))
    if r.returncode != 0:
        return None, r.stderr[-500:]
    env = dict(os.environ, CORPUS="bench/traffic.json",
               BASELINE_SCORES="bench/champion-corpus-scores.json", REPORT="/tmp/deploy-report.json")
    g = run(["./harness/harness", "bench/benchmark.json", "bench/attacks.json", WASM], cwd=ROOT, env=env)
    if g.returncode != 0:
        return None, "harness gates failed:\n" + g.stdout[-1200:]
    m = re.search(r"candidate_margin ([0-9.]+) \| wins (\d+)/(\d+)", g.stdout)
    s = re.search(r"spearman vs \S+\s+(-?[0-9.]+)", g.stdout)
    return {"margin": float(m.group(1)), "wins": f"{m.group(2)}/{m.group(3)}",
            "spearman": float(s.group(1)) if s else None}, None


def keccak(path):
    """keccak256 of the raw bytes. Done in process because a 2 MB binary as a hex
    argument overflows the exec argument limit, and this is the hash the registry
    stores (miner YAML uses sha256, WASM uses keccak256)."""
    from Crypto.Hash import keccak as _k
    data = open(path, "rb").read()
    h = _k.new(digest_bits=256)
    h.update(data)
    return "0x" + h.hexdigest(), len(data)


def pin(path, name):
    cookie = open(COOKIE).read().strip()
    r = run(["curl", "-sS", "--max-time", "120", "https://integrate.telegraphprotocol.com/api/upload-wasm",
             "-X", "POST", "-b", cookie, "-F", f"file=@{path};type=application/wasm", "-F", f"name={name}"])
    return json.loads(r.stdout)["gateway"]


def wait_for_gateway(url, size):
    for _ in range(10):
        r = run(["curl", "-sSL", "--max-time", "15", "-o", "/tmp/deploy-fetch.wasm", "-w", "%{http_code}", url])
        if r.stdout.strip() == "200" and os.path.getsize("/tmp/deploy-fetch.wasm") == size:
            return True
        subprocess.run(["sleep", "8"])
    return False


def register(wasm_hash, url, intent):
    key = [l.split("=", 1)[1].strip() for l in open(WALLET_ENV) if l.startswith("TELEGRAPH_PRIVATE_KEY")][0]
    r = run(["cast", "send", DIAMOND, "registerWasm(bytes32,string,string)", wasm_hash, url, intent,
             "--rpc-url", RPC, "--private-key", key, "--json"])
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)[-400:]
    return json.loads(r.stdout)["transactionHash"], None


def main():
    args = [a for a in sys.argv[1:] if a != "--send"]
    send = "--send" in sys.argv
    if not args:
        raise SystemExit(__doc__)
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    for intent in args:
        profile = TARGETS.get(intent)
        if profile is None:
            print(f"{intent}: no profile, skipping")
            continue
        values = dict(BASE, **PROFILES[profile])
        print(f"\n=== {intent} ({profile} profile) ===", flush=True)
        patch(intent, values)
        metrics, err = build_and_gate()
        if err:
            print(f"  gate failed, not registering:\n{err}")
            continue
        h, size = keccak(WASM)
        print(f"  gates passed: margin {metrics['margin']:.4f} wins {metrics['wins']} "
              f"spearman {metrics['spearman']}", flush=True)
        print(f"  binary {size} bytes, keccak {h}")
        if not send:
            print("  dry run, not pinning or registering")
            continue
        url = pin(WASM, f"telegraph-salience-scorer-{intent.lower()}")
        print(f"  pinned {url}", flush=True)
        if not wait_for_gateway(url, size):
            print("  gateway did not serve the binary, not registering")
            continue
        tx, err = register(h, url, intent)
        if err:
            print(f"  registerWasm reverted: {err}")
            continue
        print(f"  registered, tx {tx}")
        ledger.append({"intent": intent, "profile": profile, "hash": h, "url": url,
                       "size": size, "tx": tx, "metrics": metrics, "values": values})
        json.dump(ledger, open(LEDGER, "w"), indent=1)
    # Leave the tree on the CHAT_COMPLETION build so a plain `cargo build` is
    # reproducible against dist/.
    patch("CHAT_COMPLETION", dict(BASE, **PROFILES["text"]))
    build_and_gate()


if __name__ == "__main__":
    main()
