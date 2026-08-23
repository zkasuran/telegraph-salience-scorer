# Telegraph salience scorer

A scoring module for the [Telegraph protocol](https://telegraphprotocol.com): the WASM program a
Telegraph node runs to decide how good a miner's answer was. It takes the question, the ground
truth and the miner's answer, and returns one `f32` between 0 and 1.

Written for Telegraph Hackathon Season I, Track 2 (Script Authors).

## Layout

```
module/     the scoring module (Rust, no_std, wasm32-unknown-unknown)
harness/    Go + wazero harness; loads a .wasm the way the node does
bench/      benchmark, attack suite and family fixtures
dist/       binaries as registered on-chain
```

## Build

```bash
rustup target add wasm32-unknown-unknown        # once
cd module && cargo build --release --target wasm32-unknown-unknown
```

Must be the `wasm32-unknown-unknown` target: a WASI build carries imports a Telegraph node
cannot bind, so it fails to instantiate.

Tunables are `const` values in `lib.rs` that the build drivers rewrite in place, so build
through `deploy.py <INTENT>` rather than by hand — it patches the profile, builds and runs the
full gate set.

## Verify

```bash
cd harness && go build -o harness .
./harness ../bench/benchmark.json ../bench/attacks.json ../dist/telegraph-salience-scorer.wasm
```

Exits non-zero if the candidate misses any gate. `bench/report.json` is the last run.

## Licence

MIT. See [`LICENSE`](LICENSE).
