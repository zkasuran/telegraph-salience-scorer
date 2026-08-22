#!/usr/bin/env python3
"""Host a (large) transformer wasm on a GitHub raw permalink and register it for an intent.

usage: python3 reg_xfmr.py <INTENT> <wasm-path> [--send]
Without --send it only hosts+prints the keccak (dry). With --send it registers via cast.
Reuses deploy.py's keccak/register and a GitHub-raw host (Pinata caps at ~900KB).
"""
import os, subprocess, sys, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = "zkasuran/telegraph-salience-scorer"
_spec = importlib.util.spec_from_file_location("deploy", os.path.join(ROOT, "deploy.py"))
dep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dep)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kw)


def host(path):
    rel = os.path.relpath(path, ROOT)
    run(["git", "add", "-f", rel])
    run(["git", "commit", "-q", "-m", f"host {os.path.basename(path)}"])
    run(["git", "pull", "--rebase", "-q", "origin", "master"])
    p = run(["git", "push", "origin", "master"])
    if p.returncode != 0:
        print("push failed:", p.stderr[-300:]); raise SystemExit(1)
    sha = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    return f"https://raw.githubusercontent.com/{REPO}/{sha}/{rel}"


def fetch_ok(url, size):
    for _ in range(12):
        r = run(["curl", "-sSL", "--max-time", "30", "-o", "/tmp/reg-verify.wasm", "-w", "%{http_code}", url])
        if r.stdout.strip() == "200" and os.path.getsize("/tmp/reg-verify.wasm") == size:
            return True
        subprocess.run(["sleep", "6"])
    return False


def main():
    intent = sys.argv[1]
    wasm = sys.argv[2]
    send = "--send" in sys.argv
    h, size = dep.keccak(wasm)
    print(f"{intent}: {wasm} size={size} keccak={h}")
    if not send:
        print("  dry run"); return
    url = host(wasm)
    print(f"  hosted {url}", flush=True)
    if not fetch_ok(url, size):
        print("  URL did not serve exact bytes; aborting"); raise SystemExit(1)
    vh, _ = dep.keccak("/tmp/reg-verify.wasm")
    if vh != h:
        print("  hosted hash mismatch; aborting"); raise SystemExit(1)
    tx, err = dep.register(h, url, intent)
    if err:
        print(f"  registerWasm reverted: {err}"); raise SystemExit(1)
    print(f"  REGISTERED {intent} tx {tx}")


if __name__ == "__main__":
    main()
