#!/usr/bin/env python3
"""Stamp a licence and provenance notice into a wasm binary, in the bytes.

The problem this fixes: a scoring module is fetched as a bare `.wasm` from a raw URL.
Whoever holds that file has no `LICENSE`, no `NOTICE` and no README. The repository's
terms are one click away and easy to never look at, which is exactly how a binary ends
up rescaled and re-registered by someone who never formed a view about permission.

So the notice goes in the module. A custom wasm section named `license` is inert: the
node's runtime ignores every custom section, the module's behaviour and its exports are
untouched, and `rank_answer` returns the same f32 for the same input. But the terms now
travel with the bytes and `strings module.wasm | head` shows them.

    python3 stamp.py dist/xfmr/foo.wasm --intent CVE_LOOKUP
    python3 stamp.py dist/**/*.wasm --check        # report, change nothing

The section is added once. Re-stamping replaces it rather than appending, so a build
pipeline can call this unconditionally.
"""
import argparse
import glob
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wasmx

SECTION = b"license"

HOLDER = "zkasuran"
YEAR = "2026"
REPO = "https://github.com/zkasuran/telegraph-salience-scorer"

# Kept short on purpose. A wasm custom section is not the place for the full text, and a
# reader who needs the full text needs the URL more than they need the clauses.
NOTICE = """\
Telegraph scoring module. Copyright (c) {year} {holder}. All rights reserved.
SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

Licence: {repo}/blob/master/LICENSE
Notices: {repo}/blob/master/NOTICE
Lineage: {repo}/blob/master/PROVENANCE.json

You may run this module for any purpose, including commercially, and you may read,
disassemble, measure and benchmark it and publish what you find.

You may NOT redistribute it, publish a modified copy, or register this module or any
work based on it as your own scoring module. Appending a function that rescales the
output of this one and re-registering the result is a modified copy.

Third-party components inside this binary, if any, keep their own licences and those
licences override this notice for those parts. NOTICE names every one.
"""


def leb(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def read_section(blob, name):
    """Return the payload of a named custom section, or None."""
    for sid, off, size in wasmx.sections(blob):
        if sid != 0:
            continue
        n, i = wasmx.u32(blob, off)
        if blob[i:i + n] == name:
            return blob[i + n:off + size]
    return None


def strip_section(blob, name):
    """Every section except the named custom one, in order."""
    out = bytearray(blob[:8])
    for sid, off, size in wasmx.sections(blob):
        if sid == 0:
            n, i = wasmx.u32(blob, off)
            if blob[i:i + n] == name:
                continue
        out += bytes([sid]) + leb(size) + blob[off:off + size]
    return bytes(out)


def notice_for(intent):
    text = NOTICE.format(year=YEAR, holder=HOLDER, repo=REPO)
    if intent:
        text = f"Intent: {intent}\n" + text
    return text.encode()


def stamp(path, intent, dry=False):
    blob = open(path, "rb").read()
    before = hashlib.sha256(blob).hexdigest()
    base = strip_section(blob, SECTION)
    payload = notice_for(intent)
    body = leb(len(SECTION)) + SECTION + payload
    out = base + bytes([0]) + leb(len(body)) + body
    if not dry:
        open(path, "wb").write(out)
    from Crypto.Hash import keccak
    k = keccak.new(digest_bits=256)
    k.update(out)
    return {
        "path": path,
        "was_stamped": read_section(blob, SECTION) is not None,
        "bytes_before": len(blob),
        "bytes_after": len(out),
        "sha256_before": before,
        "sha256_after": hashlib.sha256(out).hexdigest(),
        "keccak256_after": k.hexdigest(),
    }


def check(paths):
    missing = []
    for p in paths:
        blob = open(p, "rb").read()
        if read_section(blob, SECTION) is None:
            missing.append(p)
    print(f"{len(paths)} binaries, {len(paths) - len(missing)} carry a licence section")
    for p in missing[:20]:
        print(f"  UNSTAMPED  {p}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    return 1 if missing else 0


# lib.rs declares the marker as `[u8; 32] = *b"NAME<spaces to 32>"`, so the intent name is
# the only 32-byte run in the data section that is an uppercase identifier right-padded with
# spaces. Matching on that shape rather than on "an uppercase word somewhere" avoids picking
# up an unrelated token out of the quantised weight tables.
INTENT_RE = re.compile(rb"(?<![A-Z0-9_])([A-Z][A-Z0-9_]{3,31}?)( +)(?![ A-Z0-9_])")


def intent_of(path):
    """Read the intent name our builds bake in, or None for a module that has none.

    Only used to label the notice, so a miss costs one line of text and nothing else.
    """
    blob = open(path, "rb").read()
    if not any(n == "TELEGRAPH_INTENT" for n, _, _ in wasmx.exports(blob)):
        return None
    for sid, off, size in wasmx.sections(blob):
        if sid != 11:
            continue
        for m in INTENT_RE.finditer(blob[off:off + size]):
            name, pad = m.group(1), m.group(2)
            if len(name) + len(pad) == 32 and b"_" in name:
                return name.decode()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--intent", default=None,
                    help="intent name to record; read from the module when omitted")
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    paths = []
    for p in a.paths:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    paths = [p for p in paths if p.endswith(".wasm") and os.path.isfile(p)]
    if not paths:
        raise SystemExit("no .wasm files matched")

    if a.check:
        return check(paths)

    for p in paths:
        intent = a.intent or intent_of(p)
        r = stamp(p, intent, dry=a.dry)
        verb = "re-stamped" if r["was_stamped"] else "stamped"
        print(f"{verb} {p}  intent={intent or '-'}  "
              f"{r['bytes_before']} -> {r['bytes_after']}B  keccak={r['keccak256_after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
