#!/usr/bin/env python3
"""Rebake the SCORER-DASHBOARD.html snapshot from the live node.

The page can refresh itself from the node when it is served over http, but opened from
file:// the browser blocks the cross-origin fetch, so the baked snapshot is what a reader
sees. This rewrites that snapshot in place.

    python3 tools/bake_dashboard.py
"""
import json, os, re, subprocess, sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "..", "SCORER-DASHBOARD.html")
WALLET = "0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287"
NODE = "https://devnode.telegraphprotocol.com"


def fetch():
    r = subprocess.run(["curl", "-s", "--max-time", "60",
                        f"{NODE}/engine/validator/v1/addresses/{WALLET}"],
                       capture_output=True, text=True)
    return json.loads(r.stdout)


def main():
    d = fetch()
    best = {}
    for w in d.get("wasm", []):
        it = w.get("IntentID")
        cur = best.get(it)
        if (cur is None or w.get("ActivationStatus") == "active"
                or (cur.get("ActivationStatus") != "active"
                    and (w.get("RegistrationID") or 0) > (cur.get("RegistrationID") or 0))):
            best[it] = w
    page = open(PAGE).read()
    snap_now = json.loads(re.search(r"let SNAP = (\{.*?\});", page, re.S).group(1))
    intents = [s["intent"] for s in snap_now["scorers"]]
    scorers = []
    for it in intents:
        w = best.get(it, {})
        scorers.append({"intent": it, "status": w.get("ActivationStatus") or "unregistered",
                        "score": round(w.get("EvalScore") or 0, 4)})
    mbest = {}
    for m in d.get("miners", []):
        s = m.get("MinerSlug") or m.get("Slug") or m.get("slug")
        if s not in mbest or m.get("ActivationStatus") == "active":
            mbest[s] = m
    miners = [{"slug": s, "status": m.get("ActivationStatus"), "reg": m.get("RegistrationID")}
              for s, m in mbest.items()]
    snap = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "active": sum(1 for s in scorers if s["status"] == "active"),
            "total": len(scorers), "scorers": scorers, "miners": miners}
    page = re.sub(r"let SNAP = \{.*?\};", "let SNAP = " + json.dumps(snap) + ";", page, count=1,
                  flags=re.S)
    open(PAGE, "w").write(page)
    print(f"baked {snap['active']}/{snap['total']} active at {snap['generated']}")
    lost = [s["intent"] for s in scorers if s["status"] != "active"]
    print("not held:", lost)


if __name__ == "__main__":
    main()
