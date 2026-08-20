import json, os
from Crypto.Hash import keccak
ROOT = "/tmp/flag-TASK_COMPLETION"
D = os.path.join(ROOT, "research/task_completion/scores")

def rank(x):
    idx = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0]*len(x); i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and x[idx[j+1]] == x[idx[i]]: j += 1
        avg = (i+j)/2.0 + 1
        for k in range(i, j+1): r[idx[k]] = avg
        i = j+1
    return r

def sp(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra)/n; mb = sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da = sum((ra[i]-ma)**2 for i in range(n))**.5; db = sum((rb[i]-mb)**2 for i in range(n))**.5
    return num/(da*db) if da*db else 0.0

s = {}
for f in os.listdir(D):
    if f.endswith(".json"):
        s[f[:-5]] = json.load(open(os.path.join(D, f)))
g = s["good_proxy"]; r = s["rust_lexical"]
res = {k: {"vs_good_proxy": round(sp(s[k], g), 4), "vs_rust_lexical": round(sp(s[k], r), 4)} for k in sorted(s)}

kc = {}
for name, p in [("winner_V_softq", "dist/track2-v2/task_completion.wasm"),
                ("anchor_old_node0.5454", "dist/telegraph-salience-scorer-task_completion.wasm")]:
    d = open(os.path.join(ROOT, p), "rb").read()
    h = keccak.new(digest_bits=256); h.update(d)
    kc[name] = {"bytes": len(d), "keccak": "0x" + h.hexdigest()}

out = {
 "corpus": "bench/traffic-task-sub.json (15 multi-step-task questions x 5 varied-quality gateway answers = 75 rows)",
 "proxy_note": "good_proxy = the CHAT_COMPLETION champion good.wasm (all-MiniLM-L6-v2 sentence transformer). TOPICAL PROXY for the real TASK_COMPLETION champion, which is NOT downloadable (devnode /wasm/TASK_COMPLETION.wasm => 404). Local proxy agreement OVERSTATES the node.",
 "calibration": "anchor_old (dist/telegraph-salience-scorer-task_completion.wasm, keccak 0x9387...) scored 0.5454 on the REAL node and 0.6074 vs the good.wasm proxy here => the proxy overstates node Spearman by ~0.062 for this build.",
 "spearman_vs_proxies": res,
 "winner": {
   "name": "V_softq",
   "config": "features=minilm, W_EMB on; blend 0.25*embA + 0.45*embB + 0.15*lexical + 0.15*embB(question,answer); softened penalties M_CONTRA 0.7 / M_NUM_WRONG 0.78 / M_ORDER 0.85 / M_ENTITY 0.72 / M_NEGCOV 0.32 / M_NUM_MISS_BASE 0.85 / M_TWO_FACED 0.8; marker TASK_COMPLETION",
   "proxy_spearman": res["V_softq"]["vs_good_proxy"],
   "est_node_spearman_after_calibration": round(res["V_softq"]["vs_good_proxy"] - 0.062, 4),
 },
 "keccak": kc,
}
json.dump(out, open(os.path.join(ROOT, "research/task_completion/spearman.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
