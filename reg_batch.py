#!/usr/bin/env python3
"""Host a batch of wasm builds on one commit and register each for its intent.

reg_xfmr.py does one file per commit and one push per file, which costs a push per
24 MB build. This does the whole round in one commit and one push, then verifies each
raw URL serves the exact bytes before it sends a registration for it.

usage: python3 reg_batch.py INTENT=path [INTENT=path ...] [--send]
"""
import os, subprocess, sys, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = "zkasuran/telegraph-salience-scorer"
_spec = importlib.util.spec_from_file_location("deploy", os.path.join(ROOT, "deploy.py"))
dep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dep)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kw)


def push(paths):
    rels = [os.path.relpath(p, ROOT) for p in paths]
    run(["git", "add", "-f"] + rels)
    run(["git", "commit", "-q", "-m", "host " + " ".join(os.path.basename(p) for p in paths)])
    run(["git", "pull", "--rebase", "-q", "origin", "master"])
    p = run(["git", "push", "origin", "master"])
    if p.returncode != 0:
        print("push failed:", p.stderr[-400:]); raise SystemExit(1)
    sha = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    return sha, rels


def fetch_ok(url, size, tries=12):
    for _ in range(tries):
        r = run(["curl", "-sSL", "--max-time", "60", "-o", "/tmp/regb.wasm", "-w", "%{http_code}", url])
        if r.stdout.strip() == "200" and os.path.getsize("/tmp/regb.wasm") == size:
            return True
        subprocess.run(["sleep", "6"])
    return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    send = "--send" in sys.argv
    jobs = []
    for a in args:
        intent, path = a.split("=", 1)
        path = os.path.abspath(path)
        h, size = dep.keccak(path)
        jobs.append((intent, path, h, size))
        print(f"{intent:22} {os.path.basename(path)} size={size} keccak={h}")
    if not send:
        print("dry run"); return
    sha, rels = push([j[1] for j in jobs])
    print(f"pushed {sha}", flush=True)
    for (intent, path, h, size), rel in zip(jobs, rels):
        url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{rel}"
        if not fetch_ok(url, size):
            print(f"{intent}: url did not serve exact bytes, skipping\n  {url}"); continue
        vh, _ = dep.keccak("/tmp/regb.wasm")
        if vh != h:
            print(f"{intent}: hosted hash mismatch, skipping"); continue
        tx, err = dep.register(h, url, intent)
        if err:
            print(f"{intent}: registerWasm reverted: {err}")
        else:
            print(f"{intent}: REGISTERED tx {tx}", flush=True)


if __name__ == "__main__":
    main()
