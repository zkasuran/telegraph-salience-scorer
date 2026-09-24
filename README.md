# Telegraph salience scorer

Champion scoring binaries for the [Telegraph protocol](https://telegraphprotocol.com):
the WASM programs a Telegraph node runs to score a miner's answer against the ground
truth, one `f32` in `[0, 1]` per intent.

Written for Telegraph Hackathon Season I, Track 2 (Script Authors).

This repository hosts the compiled modules. Each binary under `dist/` is registered
on-chain for its intent and the node fetches it from a commit-pinned raw URL here, so
the files are kept stable and are not rewritten.

```
dist/         the WASM binaries, exactly as registered on-chain
LICENSES/     the full text of every third-party licence that applies to a file here
```

Each `.wasm` is a `wasm32-unknown-unknown` module exporting the node's scoring entry
point. A WASI build would carry imports a node cannot bind, so these are freestanding.

## Verify a binary

The keccak256 of any file here matches the hash stored in its on-chain registration:

```bash
python3 - <<'PY'
from Crypto.Hash import keccak
h = keccak.new(digest_bits=256); h.update(open("dist/<file>.wasm", "rb").read())
print("0x" + h.hexdigest())
PY
```

## The terms travel in the bytes

A scoring module is fetched as a bare `.wasm` from a raw URL, so whoever ends up holding
one has no `LICENSE` and no `NOTICE` beside it. Every module built from 2026-08-30 onward
carries its licence notice inside the binary, in a custom wasm section named `license`:

```bash
strings dist/<file>.wasm | grep -A6 SPDX
python3 rev/stamp.py dist/<file>.wasm --check     # report, change nothing
```

The section is inert. A wasm runtime ignores every custom section, the exports are
unchanged and `rank_answer` returns the same `f32` for the same input, checked against the
unstamped build under wazero. `rev/stamp.py` writes it and verifies it, and the
registration drivers refuse to register a binary that does not carry it.

Binaries registered before that date do not have the section and are not being
re-stamped, because changing one byte changes the keccak and would break a live
registration. Their terms are the ones in `LICENSE-HISTORY.md`.

## Licence

The modules here are the work of zkasuran under
[`LICENSE`](LICENSE), the Source-Available No-Derivatives Licence 1.0. In short:
run them for any purpose including commercially, read them, disassemble them,
measure them, publish what you find. Do not redistribute them or publish a modified
copy.

Files up to and including commit `9250395a8131e4f7da51ab548d455c7270d4acd3` were
released under MIT and stay MIT for anyone who obtained them under it. That grant is
not being withdrawn. [`LICENSE-HISTORY.md`](LICENSE-HISTORY.md) gives the boundary and
a command to check which grant covers a file you hold.

Some binaries are built on another author's work. Those keep their author's licence,
which overrides ours for that file. [`NOTICE`](NOTICE) names each one with its
upstream and its terms, [`LICENSES/`](LICENSES) carries the full licence texts, and
[`PROVENANCE.json`](PROVENANCE.json) records the lineage of every published binary,
derived from the bytes rather than from a hand-kept list, so the record cannot drift
from what is actually here.
