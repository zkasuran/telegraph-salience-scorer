#!/usr/bin/env python3
"""Minimal core-wasm section walker + f32 constant reader for champion post-maps."""
import struct, sys, hashlib

def u32(b, i):
    v = 0; s = 0
    while True:
        x = b[i]; i += 1
        v |= (x & 0x7f) << s
        if not (x & 0x80): return v, i
        s += 7

def sections(b):
    assert b[:8] == b"\x00asm\x01\x00\x00\x00", "not wasm v1"
    i = 8; out = []
    while i < len(b):
        sid = b[i]; i += 1
        size, i = u32(b, i)
        out.append((sid, i, size))
        i += size
    return out

def code_bodies(b):
    for sid, off, size in sections(b):
        if sid == 10:
            n, i = u32(b, off)
            bodies = []
            for _ in range(n):
                bsz, j = u32(b, i)
                bodies.append((j, bsz))
                i = j + bsz
            return bodies
    return []

def f32consts(b, off, size):
    """every f32.const in a byte range, with its offset"""
    out = []
    i = off; end = off + size
    while i < end:
        if b[i] == 0x43 and i + 5 <= end:
            out.append((i + 1, struct.unpack("<f", b[i+1:i+5])[0]))
            i += 5
        else:
            i += 1
    return out

def exports(b):
    for sid, off, size in sections(b):
        if sid == 7:
            n, i = u32(b, off); out = []
            for _ in range(n):
                ln, i = u32(b, i)
                name = b[i:i+ln].decode("utf8", "replace"); i += ln
                kind = b[i]; i += 1
                idx, i = u32(b, i)
                out.append((name, kind, idx))
            return out
    return []

if __name__ == "__main__":
    for p in sys.argv[1:]:
        b = open(p, "rb").read()
        bodies = code_bodies(b)
        print(f"== {p}  {len(b)} bytes  sha256={hashlib.sha256(b).hexdigest()[:16]}  bodies={len(bodies)}")
        ex = [e for e in exports(b) if e[1] == 0]
        print("   func exports:", [(n, i) for n, k, i in ex][:10])
        for label, (off, size) in (("last", bodies[-1]), ("2nd-last", bodies[-2])):
            raw = b[off:off+size]
            print(f"   {label} body size={size} hex={raw[:12].hex()}...{raw[-8:].hex()}")
            print(f"      f32: {[(hex(o), v) for o, v in f32consts(b, off, size)]}")
