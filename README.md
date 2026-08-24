# Telegraph salience scorer

Champion scoring binaries for the [Telegraph protocol](https://telegraphprotocol.com):
the WASM programs a Telegraph node runs to score a miner's answer against the ground
truth, one `f32` in `[0, 1]` per intent.

Written for Telegraph Hackathon Season I, Track 2 (Script Authors).

This repository hosts the compiled modules. Each binary under `dist/` is registered
on-chain for its intent, and the node fetches it from a commit-pinned raw URL here, so
the files are kept stable and are not rewritten.

```
dist/    the WASM binaries, exactly as registered on-chain
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

## Licence

MIT. See [`LICENSE`](LICENSE).
