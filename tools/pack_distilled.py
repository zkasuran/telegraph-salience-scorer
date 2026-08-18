#!/usr/bin/env python3
"""Pack the champion-distilled word vectors into the module's fixed blob.

Unlike pack_vectors.py (generic GloVe), these vectors are distilled from the
CHAT_COMPLETION champion's own sentence encoder: each word was run through its
embed() export, so a mean-pool of them approximates the champion's 384d sentence
embedding. Validation Spearman against the champion's true ranking: 0.90 once the
vocabulary common component is removed, versus ~0.31 for generic GloVe.

Steps that matter:
  1. read the distilled word -> 384 float rows,
  2. subtract the vocabulary mean (the SIF common component that otherwise makes
     every sentence cosine ~0.87 and flattens the ranking),
  3. L2-normalise and quantise to int8, one row per word,
  4. FNV-1a hash the words (same hash the module computes at runtime), sort, emit
     the TGV blob the module binary-searches.

usage: python3 tools/pack_distilled.py <distilled.txt> module/src/vectors.bin [dim]
"""
import struct
import sys

import numpy as np


def fnv1a(word: str) -> int:
    h = 0x811C9DC5
    for b in word.encode("utf-8"):
        if b == 0x2C:
            continue
        if 0x41 <= b <= 0x5A:
            b += 32
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def main():
    src, out = sys.argv[1], sys.argv[2]
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 384

    words, vecs = [], []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            x = line.rstrip("\n").split(" ")
            if len(x) < dim + 1:
                continue
            w = " ".join(x[:-dim])
            if not (w.isalnum() and w.isascii() and w.islower()):
                continue
            v = np.array([float(t) for t in x[-dim:]], dtype=np.float64)
            words.append(w)
            vecs.append(v)
    V = np.array(vecs)
    print(f"loaded {len(words)} distilled vectors, dim {V.shape[1]}")

    # SIF common-component removal: subtract the vocabulary mean direction. This is
    # what lifted validation Spearman from 0.72 to 0.90.
    mean = V.mean(axis=0)
    V = V - mean

    rows = {}
    for w, v in zip(words, V):
        n = np.linalg.norm(v)
        if n == 0:
            continue
        key = fnv1a(w)
        if key in rows:
            continue
        q = np.clip(np.round(v / n * 127), -127, 127).astype(np.int8)
        rows[key] = q.tobytes()

    keys = sorted(rows)
    buf = bytearray()
    buf += b"TGV"
    buf += bytes([1])
    buf += struct.pack("<I", len(keys))
    buf += struct.pack("<I", dim)
    for k in keys:
        buf += struct.pack("<I", k)
    for k in keys:
        buf += rows[k]
    with open(out, "wb") as fh:
        fh.write(buf)
    print(f"wrote {out}: {len(keys)} words, dim {dim}, {len(buf)} bytes")


if __name__ == "__main__":
    main()
