#!/usr/bin/env python3
"""Write PROVENANCE.json: for every hosted binary, whose work it is built on.

Lineage is read off the bytes, not off a hand-kept list, so the file cannot drift
from what is actually published. Three signals decide it:

  TELEGRAPH_INTENT export   our module bakes this marker in; upstream modules do not
  data-section digest       a build and its base share the identical data section,
                            which groups families without needing any source
  producers / name sections `walrus` means we post-processed a compiled module we did
                            not build; leftover Rust crate symbols name the crate

    python3 rev/provenance.py > PROVENANCE.json
"""
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wasmx

# data-section digest -> the upstream a family is built on. Keyed by bytes so a
# renamed file cannot escape its entry.
UPSTREAM = {
    "470d726e8375": {
        "name": "assay",
        "author": "GreatSage-dev",
        "url": "https://github.com/GreatSage-dev/Assay",
        "licence": "MIT",
        "copyright": "MIT (c) GreatSage-dev",
        "kind": "source fork, rebuilt from their published source",
        "ours": "the final output-band calibration only",
    },
    "2858:5ada2619b464": {},  # same upstream, different band; filled below
    "289d3b89cfd3": {
        "name": "telegraph-wasm-scoring",
        "author": "ssoni4751",
        "url": "https://github.com/ssoni4751/telegraph-wasm-scoring",
        "licence": "MIT",
        "copyright": "MIT (c) 2026 telegraphprotocol",
        "kind": "source fork, rebuilt with --features real_weights",
        "ours": "a monotone logistic sharpener over their composite",
    },
    "1f95db131565": {
        "name": "amanat",
        "author": "Pugar Huda Mantoro",
        "url": "https://github.com/PugarHuda/amanat",
        "licence": "MIT",
        "copyright": "MIT (c) 2026 Pugar Huda Mantoro",
        "kind": "source fork, rebuilt with --features verdict",
        "ours": "a smoothstep stretch pivoted at 0.10",
    },
    "5453d6ca8dba": {
        "name": "patchsignal",
        "author": "0x236891fe",
        "url": "https://169.58.206.25.sslip.io/telegraph/",
        "licence": "none published",
        "copyright": "unstated",
        "kind": "binary wrap of a compiled module we did not write",
        "ours": "a monotone rescaling appended to their output",
        "withdrawn": True,
    },
    "383fd369b20e": {
        "name": "gas_price_scorer / currency_exchange_scorer",
        "author": "0x5d27fee6",
        "url": "https://pub-307068a26c7b48ab80b3d2ccced8a7be.r2.dev/artifact/",
        "licence": "none published",
        "copyright": "unstated",
        "kind": "binary wrap of a compiled module we did not write",
        "ours": "a monotone rescaling appended to their output",
        "withdrawn": True,
    },
    "c59a83e73490": {
        "name": "language_translation",
        "author": "0x22db3a96",
        "url": "https://github.com/seekdaseek/telegraph-scorer",
        "licence": "none published",
        "copyright": "unstated",
        "kind": "binary wrap of a compiled module we did not write",
        "ours": "a monotone rescaling appended to their output",
        "withdrawn": True,
    },
}
UPSTREAM["5ada2619b464"] = dict(UPSTREAM["470d726e8375"])
del UPSTREAM["2858:5ada2619b464"]

WEIGHTS = [
    {"blob": "minilm.bin", "model": "sentence-transformers/all-MiniLM-L6-v2", "licence": "Apache-2.0"},
    {"blob": "gte-small.bin / gte-mix.bin / gte-int4.bin", "model": "thenlper/gte-small", "licence": "MIT"},
    {"blob": "vectors.bin / vectors-glove.bin", "model": "GloVe 6B (Pennington, Socher, Manning 2014)", "licence": "PDDL-1.0"},
]

CRATE = re.compile(rb"_RNvC[^_]*_\d+([a-z_0-9]+)")


def probe(path):
    b = open(path, "rb").read()
    names = {n for n, k, _ in wasmx.exports(b)}
    data = producers = symbols = None
    for sid, off, size in wasmx.sections(b):
        if sid == 11:
            data = hashlib.sha256(b[off:off + size]).hexdigest()[:12]
        if sid == 0:
            n, i = wasmx.u32(b, off)
            tag = b[i:i + n]
            if tag == b"producers":
                producers = b[i + n:off + size]
            if tag == b"name":
                symbols = b[i + n:off + size]
    crates = sorted({m.group(1).decode() for m in CRATE.finditer(symbols or b"")}
                    - {"core", "alloc"})
    k = hashlib.new("sha256", b).hexdigest()
    from Crypto.Hash import keccak as _k
    kk = _k.new(digest_bits=256)
    kk.update(b)
    return {
        "bytes": len(b),
        "sha256": k,
        "keccak256": kk.hexdigest(),
        "data_digest": data,
        "our_intent_marker": "TELEGRAPH_INTENT" in names,
        "walrus_processed": bool(producers and b"walrus" in producers),
        "leftover_crate_symbols": crates[:8],
        "exports": sorted(names),
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    tracked = [p for p in subprocess.run(["git", "ls-files", "dist"], capture_output=True,
                                         text=True).stdout.split() if p.endswith(".wasm")]
    files, counts = {}, {"own": 0, "source_fork": 0, "binary_wrap": 0}
    for p in sorted(tracked):
        info = probe(p)
        up = UPSTREAM.get(info["data_digest"])
        if up is None:
            info["lineage"] = "own"
            info["licence"] = "LicenseRef-zkasuran-SAND-1.0"
            counts["own"] += 1
        else:
            info["lineage"] = "source_fork" if "source fork" in up["kind"] else "binary_wrap"
            info["upstream"] = up
            info["licence"] = up["licence"]
            counts[info["lineage"]] += 1
        files[p] = info
    json.dump({
        "note": "Per-file lineage of every binary published in dist/, derived from the bytes. "
                "A file whose lineage is not 'own' is governed by its upstream's licence, not ours.",
        "generated_from": "python3 rev/provenance.py",
        "signals": {
            "our_intent_marker": "our module exports TELEGRAPH_INTENT; no upstream module does",
            "data_digest": "sha256 of the wasm data section, first 12 hex; a build and its base share it",
            "walrus_processed": "we post-processed a compiled module we did not build",
            "leftover_crate_symbols": "Rust crate names the upstream author left in the binary",
        },
        "counts": counts,
        "embedded_weights": WEIGHTS,
        "files": files,
    }, sys.stdout, indent=1)
    print(file=sys.stdout)
    print(f"own={counts['own']} source_fork={counts['source_fork']} binary_wrap={counts['binary_wrap']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
