#!/usr/bin/env python3
"""Pack a GloVe text file into the fixed blob the scoring module embeds.

The module runs with no network, no filesystem and no allocator, so semantic
similarity has to be compiled in. This turns the top N GloVe vectors into a table
the module can binary search: FNV-1a hashes of the words (the same hash the module
computes at runtime), then one L2-normalised int8 row per word, so a cosine is an
integer dot product.

Vectors: GloVe (Pennington, Socher and Manning 2014), released under the Open Data
Commons Public Domain Dedication and License v1.0.

usage: python3 tools/pack_vectors.py glove.6B.50d.txt module/src/vectors.bin [words]
"""
import struct
import sys


def fnv1a(word: str) -> int:
    h = 0x811C9DC5
    for b in word.encode("utf-8"):
        if b == 0x2C:  # ',' is skipped by the module's hash so 1,000 == 1000
            continue
        if 0x41 <= b <= 0x5A:
            b += 32
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 40000

    rows = {}
    dim = None
    skipped_shape = 0
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word, rest = parts[0], parts[1:]
            if dim is None:
                dim = len(rest)
            elif len(rest) != dim:
                skipped_shape += 1
                continue
            # The module only ever tokenises runs of letters and digits, so a vector
            # for "," or "n't" can never be looked up.
            if not word.isalnum() or not word.isascii():
                continue
            key = fnv1a(word)
            if key in rows:  # keep the more frequent word on a hash collision
                continue
            vals = [float(x) for x in rest]
            norm = sum(v * v for v in vals) ** 0.5
            if norm == 0:
                continue
            rows[key] = [max(-127, min(127, int(round(v / norm * 127)))) for v in vals]
            if len(rows) >= want:
                break

    keys = sorted(rows)
    with open(out, "wb") as fh:
        fh.write(b"TGV1")
        fh.write(struct.pack("<II", len(keys), dim))
        for k in keys:
            fh.write(struct.pack("<I", k))
        for k in keys:
            fh.write(struct.pack(f"<{dim}b", *rows[k]))
    size = 12 + 4 * len(keys) + len(keys) * dim
    print(f"{len(keys)} words, dim {dim}, {size} bytes -> {out}")
    if skipped_shape:
        print(f"  ({skipped_shape} malformed lines skipped)")


if __name__ == "__main__":
    main()
