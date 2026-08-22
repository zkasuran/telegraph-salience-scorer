#!/usr/bin/env python3
"""Reference all-MiniLM-L6-v2 in numpy, f32, no ML framework.

The point is to have a ground truth for the wasm port. Two questions it answers: how far
our int8 port drifts from the real model, and how far the live champion is from the real
model. If the champion tracks the real model and we do not, the gap is our port and worth
fixing; if neither does, the champion is a different model and no amount of port work closes
it.

    python3 tools/ref_minilm.py <triples.json> <out.json>
      triples.json: [{"id":..,"gt":..,"a":..}, ...]  -> cosine of mean-pooled embeddings
"""
import json, struct, sys
import numpy as np

MODEL = "/home/asuran/.semantic_search/models/all-MiniLM-L6-v2/model.safetensors"
TOKJSON = "/home/asuran/.semantic_search/models/all-MiniLM-L6-v2/tokenizer.json"
H, LAYERS, HEADS, MAXTOK = 384, 6, 12, 256


def load_weights():
    f = open(MODEL, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    blob = f.read()
    W = {}
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        dt = np.float32 if v["dtype"] == "F32" else np.int64
        W[k] = np.frombuffer(blob[a:b], dtype=dt).reshape(v["shape"])
    return W


def load_vocab():
    return json.load(open(TOKJSON))["model"]["vocab"]


def tokenize(text, vocab):
    """BertNormalizer(lowercase) + BertPreTokenizer + WordPiece, the standard path."""
    text = text.lower()
    words, cur = [], []
    for ch in text:
        if ch.isspace():
            if cur:
                words.append("".join(cur)); cur = []
        elif ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                words.append("".join(cur)); cur = []
            words.append(ch)
    if cur:
        words.append("".join(cur))
    ids = [vocab["[CLS]"]]
    for w in words:
        if len(ids) >= MAXTOK - 1:
            break
        if w in vocab:
            ids.append(vocab[w]); continue
        start, pieces, ok = 0, [], True
        while start < len(w):
            end, found = len(w), None
            while end > start:
                piece = w[start:end] if start == 0 else "##" + w[start:end]
                if piece in vocab:
                    found = vocab[piece]; break
                end -= 1
            if found is None:
                ok = False; break
            pieces.append(found); start = end
        ids.extend(pieces if ok else [vocab["[UNK]"]])
    ids = ids[:MAXTOK - 1] + [vocab["[SEP]"]]
    return np.array(ids, dtype=np.int64)


def layernorm(x, g, b, eps=1e-12):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def encode(ids, W):
    n = len(ids)
    x = (W["embeddings.word_embeddings.weight"][ids]
         + W["embeddings.position_embeddings.weight"][:n]
         + W["embeddings.token_type_embeddings.weight"][0])
    x = layernorm(x, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"])
    dh = H // HEADS
    for li in range(LAYERS):
        p = f"encoder.layer.{li}."
        q = x @ W[p + "attention.self.query.weight"].T + W[p + "attention.self.query.bias"]
        k = x @ W[p + "attention.self.key.weight"].T + W[p + "attention.self.key.bias"]
        v = x @ W[p + "attention.self.value.weight"].T + W[p + "attention.self.value.bias"]
        q = q.reshape(n, HEADS, dh).transpose(1, 0, 2)
        k = k.reshape(n, HEADS, dh).transpose(1, 0, 2)
        v = v.reshape(n, HEADS, dh).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s); a = e / e.sum(-1, keepdims=True)
        c = (a @ v).transpose(1, 0, 2).reshape(n, H)
        c = c @ W[p + "attention.output.dense.weight"].T + W[p + "attention.output.dense.bias"]
        x = layernorm(c + x, W[p + "attention.output.LayerNorm.weight"],
                      W[p + "attention.output.LayerNorm.bias"])
        h = x @ W[p + "intermediate.dense.weight"].T + W[p + "intermediate.dense.bias"]
        h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
        o = h @ W[p + "output.dense.weight"].T + W[p + "output.dense.bias"]
        x = layernorm(o + x, W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"])
    e = x.mean(0)
    return e / np.linalg.norm(e)


def main():
    triples = json.load(open(sys.argv[1]))
    W = load_weights(); vocab = load_vocab()
    cache = {}
    def emb(t):
        if t not in cache:
            cache[t] = encode(tokenize(t, vocab), W)
        return cache[t]
    out = {}
    for i, t in enumerate(triples):
        gt, a = t.get("gt", ""), t.get("a", "")
        if not gt or not a:
            out[t["id"]] = 0.0; continue
        out[t["id"]] = float(np.dot(emb(gt), emb(a)))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(triples)}", flush=True)
    json.dump({"module": "ref-minilm-f32", "scores": out}, open(sys.argv[2], "w"))
    print(f"wrote {len(out)} reference cosines to {sys.argv[2]}")


if __name__ == "__main__":
    main()
