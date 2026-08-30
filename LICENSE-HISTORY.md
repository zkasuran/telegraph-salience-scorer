# Licence history

This repository changed licence on 2026-08-30. Nothing was withdrawn. This file
records exactly what is covered by which grant, so anyone holding a copy knows
where they stand.

## The two periods

| Period | Commits | Licence |
| --- | --- | --- |
| 2026-08-17 to 2026-08-30 | up to and including `9250395a8131e4f7da51ab548d455c7270d4acd3` | MIT |
| 2026-08-30 onward | after that commit | Source-Available No-Derivatives 1.0, see [`LICENSE`](LICENSE) |

`git log 9250395a8131e4f7da51ab548d455c7270d4acd3` is the boundary. Every file as
it stood at or before that commit was published under MIT, and that grant is
irrevocable for whoever obtained a copy under it. Nobody who forked, mirrored,
modified or built on those bytes needs to do anything.

## Why it changed

The binaries here are competition entries: each one is registered on-chain as the
scoring module for a Telegraph intent, and the protocol promotes whichever module
scores best. MIT let anyone take a registered binary, append a few bytes that
rescale its output, and register the result as a competing entry. That is exactly
what MIT permits, so this is a correction of our own licence choice and not a
complaint about anyone's conduct.

The new licence keeps everything a reader, a validator or a judge needs:

- running the modules is permitted, for any purpose, including commercially. A
  Telegraph node fetching a binary from a pinned raw URL and scoring answers with
  it is a permitted use, and always was.
- reading, disassembling, measuring and benchmarking them is permitted, and so is
  publishing what you find. The technique that beat us is documented in our own
  method notes; we are not trying to hide it.
- keeping a copy to verify a hash or reproduce a measurement is permitted.

What it withholds is redistribution and derivative works: publishing a modified
copy, or re-registering one under another identity.

## What did not change

- **Third-party components keep their own licences.** They are listed in
  [`NOTICE`](NOTICE) with the terms that apply to each. Where a file in `dist/` is
  built on someone else's MIT-licensed work, their MIT terms govern that file and
  this repository's licence does not, and cannot, restrict it.
- **Every on-chain registration still resolves.** GitHub serves a blob at a
  commit-pinned raw URL for as long as the commit is reachable, whatever the
  current licence says, so no validator fetch breaks because of this change.
- **The MIT copies already out there stay valid.** Two derivative families exist
  on the network today, built on our MIT-era binaries. They are lawful. They stay
  lawful.

## Verifying which grant covers a file you hold

```bash
# keccak256 of a binary is the hash in its on-chain registration
python3 - <<'PY'
from Crypto.Hash import keccak
h = keccak.new(digest_bits=256); h.update(open("dist/<file>.wasm", "rb").read())
print("0x" + h.hexdigest())
PY

# then find the commit that introduced it and compare against the boundary
git log --oneline --diff-filter=A -- dist/<file>.wasm
git merge-base --is-ancestor <that commit> 9250395a8131e4f7da51ab548d455c7270d4acd3 \
  && echo "MIT" || echo "SAND-1.0"
```
