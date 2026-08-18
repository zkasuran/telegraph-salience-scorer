#!/usr/bin/env python3
"""Pack all-MiniLM-L6-v2 into the blob the no_std module reads for embB.

Big matrices (attention/FFN weights, the word-embedding table) are int8 with one
f32 scale each; small vectors (biases, LayerNorm gains) stay f32. Tensors are laid
out in a FIXED order the Rust side reads sequentially, so no names travel in the blob.
A vocab section maps FNV-1a hashes of the wordpiece tokens to their row in the word
table, sorted for binary search.

Validated first in numpy: int8 keeps champion agreement at 0.92 (0.99 rank match to
f32), so the quantisation is safe.

    python3 pack_minilm.py model.safetensors vocab.txt module/src/minilm.bin
"""
import json, struct, sys
import numpy as np

H, LAYERS, HEADS, INTER, VOCAB, MAXPOS = 384, 6, 12, 1536, 30522, 512


def load_safetensors(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); hdr.pop("__metadata__", None); blob = f.read()
    DT = {"F32": np.float32, "I64": np.int64}
    W = {}
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        W[k] = np.frombuffer(blob[a:b], dtype=DT[v["dtype"]]).reshape(v["shape"]).astype(
            np.float32 if v["dtype"] == "F32" else np.int64)
    return W


def fnv1a(s: bytes) -> int:
    h = 0x811C9DC5
    for b in s:
        h ^= b; h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def tensor_order(W):
    """The fixed sequence the Rust reads. (name, quantise?)"""
    seq = [("embeddings.word_embeddings.weight", True),
           ("embeddings.position_embeddings.weight", True),
           ("embeddings.token_type_embeddings.weight", True),
           ("embeddings.LayerNorm.weight", False),
           ("embeddings.LayerNorm.bias", False)]
    for i in range(LAYERS):
        p = f"encoder.layer.{i}."
        for nm, q in [("attention.self.query.weight", True), ("attention.self.query.bias", False),
                      ("attention.self.key.weight", True), ("attention.self.key.bias", False),
                      ("attention.self.value.weight", True), ("attention.self.value.bias", False),
                      ("attention.output.dense.weight", True), ("attention.output.dense.bias", False),
                      ("attention.output.LayerNorm.weight", False), ("attention.output.LayerNorm.bias", False),
                      ("intermediate.dense.weight", True), ("intermediate.dense.bias", False),
                      ("output.dense.weight", True), ("output.dense.bias", False),
                      ("output.LayerNorm.weight", False), ("output.LayerNorm.bias", False)]:
            seq.append((p + nm, q))
    return seq


def main():
    st, vocab_txt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    W = load_safetensors(st)
    buf = bytearray()
    buf += b"MLM1"

    # vocab section: sorted (fnv hash, row id)
    toks = [line.rstrip("\n") for line in open(vocab_txt, encoding="utf-8")]
    assert len(toks) == VOCAB, len(toks)
    pairs = sorted((fnv1a(t.encode("utf-8")), i) for i, t in enumerate(toks))
    buf += struct.pack("<I", len(pairs))
    for h, i in pairs:
        buf += struct.pack("<II", h, i)

    # tensors in fixed order
    for name, do_q in tensor_order(W):
        a = W[name].astype(np.float32).ravel()
        if do_q:
            s = float(np.abs(a).max() / 127.0) or 1.0
            q = np.round(a / s).clip(-127, 127).astype(np.int8)
            buf += bytes([0]); buf += struct.pack("<f", s); buf += struct.pack("<I", q.size)
            buf += q.tobytes()
        else:
            buf += bytes([1]); buf += struct.pack("<f", 0.0); buf += struct.pack("<I", a.size)
            buf += a.astype("<f4").tobytes()

    open(out, "wb").write(buf)
    print(f"wrote {out}: {len(buf)} bytes ({len(buf)/1e6:.1f} MB), vocab {len(pairs)}")


if __name__ == "__main__":
    main()
