#!/usr/bin/env python3
"""Poll the node and log every registration result as it lands.

The node evaluates one registration at a time and a margin-passing one takes about 17
minutes, so a round of probes reports over hours. This watches in the background and writes
one line per newly resolved registration, so a session can read the whole history in one
look instead of polling by hand.

    python3 tools/watch.py [seconds between polls]     appends to .scratch/watch.log
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALLET = "0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287"
URL = f"https://devnode.telegraphprotocol.com/engine/validator/v1/addresses/{WALLET}"
LOG = os.path.join(ROOT, ".scratch", "watch.log")


def poll():
    r = subprocess.run(["curl", "-s", "--max-time", "40", URL], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def main():
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    seen = {}
    if os.path.exists(LOG):
        for line in open(LOG):
            p = line.split()
            if len(p) > 2 and p[1].isdigit():
                seen[int(p[1])] = p[3]
    while True:
        d = poll()
        if d:
            out = []
            for x in d.get("wasm", []):
                rid = x.get("RegistrationID") or 0
                st = x.get("ActivationStatus")
                if rid < 256 or seen.get(rid) == st or st == "pending":
                    continue
                seen[rid] = st
                ed = x.get("EvalDetails") or ""
                try:
                    ed = json.loads(ed)
                except Exception:
                    ed = {}
                m = ed.get("candidate_margin")
                sp = ed.get("spearman") or {}
                spv = list(sp.values())[0] if sp else None
                k = (m - 0.010046) / 0.98 * 32 if m else None
                out.append(f"{time.strftime('%H:%M:%S')} {rid} {x.get('IntentID')} {st} "
                           f"margin={m} k~={round(k,2) if k else None} "
                           f"rows={ed.get('historical_rows_evaluated')} "
                           f"spearman={round(spv,4) if spv is not None else None} "
                           f"url={(x.get('WasmURL') or '')[-40:]}")
            if out:
                with open(LOG, "a") as f:
                    f.write("\n".join(out) + "\n")
        time.sleep(every)


if __name__ == "__main__":
    main()
