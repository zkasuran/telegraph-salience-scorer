<p align="center">
  <img src="https://img.shields.io/badge/licence-source--available%20no--derivatives-blue?style=for-the-badge" alt="Licence: source-available no-derivatives"/>
  <img src="https://img.shields.io/badge/target-wasm32--unknown--unknown-orange?style=for-the-badge" alt="Target: wasm32-unknown-unknown"/>
  <img src="https://img.shields.io/badge/runtime-no__std%20·%20no%20net%20·%20no%20fs-informational?style=for-the-badge" alt="no_std, no network, no filesystem"/>
  <img src="https://img.shields.io/badge/wallet-0x8b22...5287-blueviolet?style=for-the-badge" alt="Wallet"/>
</p>

<h1 align="center">⚡ Telegraph salience scorer</h1>

<p align="center">
  <strong>The scoring modules a Telegraph node runs, one <code>f32</code> in <code>[0, 1]</code> per intent.</strong><br/>
  This repository is the exact bytes registered on-chain. The method that built them lives in the lab.
</p>

---

## What this is

Telegraph runs a permissionless contest for every one of its canonical intents. A scoring module is the program a validator node runs to decide how good a miner's answer was. It reads three strings, the question, the ground truth and the miner's answer, then returns a single `f32` between 0 and 1. The node promotes whichever registered module scores best on that intent's hidden fixtures and its real traffic. Hold the slot and you set how that intent is graded for the whole network.

Each file under `dist/` is a `wasm32-unknown-unknown` module that exports exactly what the node binds:

| Export | Signature | Purpose |
| --- | --- | --- |
| `alloc` | `(size: i32) -> i32` | give the node memory to write the input strings into |
| `dealloc` | `(ptr: i32, size: i32)` | free it |
| `rank_answer` | `(q, q_len, gt, gt_len, ans, ans_len) -> f32` | score the answer |

The module runs in a bare sandbox. No std, no allocator beyond a fixed arena, no network, no filesystem, no host imports. Everything it needs is compiled in. A WASI build would carry imports a node cannot bind, so these are freestanding.

This repository hosts the compiled modules and their terms. The node fetches each one from a commit-pinned raw URL here, so a file is never rewritten in place: one changed byte changes its keccak and breaks a live registration. The source, the drivers, the verification harness and the worklogs live in the sibling lab, [`telegraph-scorer-lab`](https://github.com/zkasuran/telegraph-scorer-lab).

## Where it stands

Read it live, because the board is contested in real time and every number below is true for its date and nothing more.

As of 2026-09-25, 43 of the 45 canonical intents run one of these modules as their live champion, all under the wallet `0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287`. The two exceptions are held by a single rival author. Forty-nine intents carry wasm registrations in total. Forty-five of them have an active champion. The other four are unnamed test intents with no promotion.

`dist/` holds 1,461 published binaries: every build that was ever hosted for a registration, winners next to rejected attempts, kept stable because each is pinned by a live on-chain fetch. Their lineage is read off the bytes, not a hand-kept list:

| Lineage | Count | Licence |
| --- | --- | --- |
| `own` (our source, our build) | 1,429 | SAND-1.0 |
| `source_fork` (built from another author's source under their licence) | 32 | theirs, named in `NOTICE`, plus ours on our addition |
| `binary_wrap` (another author's compiled binary plus our bytes) | 0 | not publishable, all withdrawn |

`binary_wrap` reads zero on purpose. Forty-four such files were withdrawn on 2026-08-30 and stay withdrawn. See the licensing section.

## How the node grades and how we learned it

The scorer was reverse-engineered from the node's own grading. The protocol ships a default module that scores word overlap, the fraction of the answer's words that also appear in the ground truth. That module has two failure modes that decide leaderboards. It pays out for an answer that shares vocabulary with the ground truth while asserting the opposite ("the certificate has not expired"). It pays nothing for a correct answer in different words ("prices tend to rise" against a ground truth of "prices usually increase"). On a 40 case benchmark it scores wrong answers higher than right ones on average, a margin of -0.117.

Every registration returns a labelled measurement, `EvalDetails`. Reading those on accept and on reject decoded exactly what the node rewards. A challenger has to clear three gates in order to replace the incumbent:

| Gate | Rule | What it takes |
| --- | --- | --- |
| Ordering | `candidate_wins >= champion_wins` | Rank good above bad on at least as many fixture pairs. The hardest gate, because no calibration fixes a wins loss. |
| Separation | `candidate_margin > champion_margin` (strict) | `margin = mean(good) - mean(bad)`. A tie loses. This gate rewards contrast. |
| Agreement | `spearman(you, champion) >= 0.60` | Binds only when the intent has real traffic. Your ranking of live answers must track the incumbent's. |

Two behaviours are in no doc and cost a registration each to learn. At the ceiling the separation gate stops behaving like a plain `>`: against a champion at 0.999999 a candidate at 0.99999994 was rejected. Only an exact 1.0 was taken. And a Spearman of 0.0000 is undefined rather than low, which is what a bare hard step produces when it collapses every live answer to one score. The most useful thing the numbers gave up: a step build's margin decodes to a fixture count, `margin ≈ 0.010 + 0.98 · (k / N)`, where `k` is the fixture pairs one threshold cleanly splits. A rejection is not a wall. It is a readout of how many pairs you still miss.

## What the module measures

Word overlap is one signal of several. It is the one the others override. The design rests on a single observation: most of the meaning sits in a few words, so weight those and score precision and recall over the weights.

- **Salience weighting.** A corpus-free stand-in for IDF, since a module gets no corpus and no network. Numbers weigh most, proper nouns next, ordinary words by length, function words and boilerplate almost nothing.
- **Precision and recall on those weights.** Precision is concave, so supporting context is free while a shotgun list of every candidate answer collapses. Recall is measured against the part of the ground truth the question did not already contain, because that part is the answer and the rest is the prompt coming back.
- **Character trigrams and bigrams**, taking the better of Dice and containment. Containment is why a correct answer padded with boilerplate stays correct.
- **Correctness penalties.** A contradiction, a right figure on the wrong entity, right words with no shared adjacency ("France is the capital of Paris"), a transposed literal, a missing or wrong number: each cuts a lexically perfect answer to a fraction of its score. These are the levers that fix ordering. They are the only non-monotone part of the pipeline.
- **Polarity on three axes**, verdict, authenticity and direction, negation aware and clause aware. "No, written by a human" is negative on verdict and positive on authenticity at once, which is why one polarity table would misread it as self-contradiction.
- **Numeric agreement.** Figures parse to values so format falls away ("$4.31", "4.31 USD", "4,310", "1.2M"), then match by relative error. A contradicting figure costs most of the score.
- **Final calibration.** One path per build: a smoothstep, a logistic curve, a hard step with a tie-break, a three-band step with exact rails.

The lexical core is `no_std`, every buffer a fixed static and every loop bounded, around 9 KB of wasm. Figure intents (prices, balances, scores) get a separate freestanding numeric scorer, about 5 KB, where a wrong number is close to fatal and text overlap only breaks ties.

## The from-scratch MiniLM blend

Some champions rank on sentence-embedding similarity rather than lexical overlap, CHAT_COMPLETION most of all. To compete there with no runtime and no network, we ported MiniLM-L6-v2 into the `no_std` module from scratch: a WordPiece tokeniser, six transformer layers at 384 hidden dimensions with 12 heads, GELU, post-LayerNorm, weights quantised to int8 (about 23 MB). It runs inside the same sandbox, with a small LRU memo so the transformer is not recomputed for every answer scored against one question.

The blend reads cosine similarity at several depths, a shallow embedding tap, mid-layer taps and the full six-layer output, plus an answer-to-question cosine for relevance. Fine-tuning moves the last layers most, so a shallower tap can track a fine-tuned champion better than our own final layer does. A compact GloVe table (int8, 50 dimensions) supplies topicality for the lexical path, never correctness: distributional vectors put "rise" and "fall" at cosine 0.88, so direction stays with the polarity axes.

## The master key and the defence

A strictly monotone transform of the final score cannot reorder any two answers. That one fact is the lever the whole lane runs on. It leaves ordering untouched, so wins are preserved. It leaves rank correlation untouched, so agreement is preserved. It pushes the margin up freely. So when a champion is open source, the winning move is to rebuild its exact scorer, confirm it reproduces bit for bit, then wrap its output in a monotone sharpener. The ranking stays the champion's, the agreement comes for free, the margin rises past theirs. The corollary is the trap: a monotone transform cannot fix a bad ranking. Fix the ordering first with penalties, then buy the margin.

That same fact is the attack, which is the whole reason for the licence below. A wrap of a scorer can never report a margin higher than `f32(k/N)`, the fraction of fixture pairs one threshold cleanly splits. So a slot whose margin already reads exactly `f32(k/N)` is wrap-proof, whatever that number is. CONTENT_MODERATION sat at 0.800000012 and was as safe as a slot pinned at exactly 1.0. On 2026-08-31 the board stood at 40 of 45 held, 13 slots on that ROC ceiling, 9 of them sealed bit-exact at 1.0 and therefore beyond any wrap.

## Verified offline first

Nothing is registered on a hope. The lab carries a Go harness built on wazero, the same class of wasm runtime the node uses. It runs the node's own promotion logic locally: the structural checks (empty input scores 0, a self-match clears the 0.75 floor, self beats cross-match, oversized and adversarial inputs do not trap), the separation metrics, a 12 case attack suite (question echo, verdict flip, direction flip, number swap, stopword spam and more), then rank agreement against the live champion's cached scores. A round trip to the node takes about 17 minutes and evaluates out of order, so the local harness is where a build is proven or thrown out before it costs a registration.

## The arc

Built from nothing to all 45 slots, intent by intent, reading each rejection until the module held every one. A strong field then took fifteen slots with one good reused scorer and we fell to 27. The climb back was five different problems, each wrong for its own reason, from a 29 MB binary silently dropped for exceeding the node's fetch limit to a penalty of ours firing on a good answer. Then better builds arrived, several of them open source. The reverse-engineer-then-out-build loop took them back one at a time. The last phase was a wrap war: an author was taking slots by appending a two-band rescaling to our own registered binaries, which MIT expressly allowed. That produced the two things this repository now rests on, the licence and the ROC-ceiling defence. As of 2026-09-25 the live count is 43 of 45.

## Licensing, in the bytes

This repository moved off MIT on 2026-08-30. The reason is written into the story above. Each binary here is a competition entry. MIT let anyone take a registered binary, append a few bytes that rescale its output, then register the result as a rival entry. Twelve slots were taken from us that way, every one decoding to our own base with a two-band map on top. That was lawful, so this is a correction of our own licence choice rather than a complaint about anyone.

Entries now ship under the Source-Available No-Derivatives Licence 1.0, SPDX `LicenseRef-zkasuran-SAND-1.0`, in [`LICENSE`](LICENSE). It grants what a reader, a validator or a judge needs: run the module for any purpose including commercially, read it, disassemble it, benchmark it, publish what you find. It withholds one thing, redistribution and derivative works, because that is the thing that was used against us. Files up to and including commit `9250395a8131e4f7da51ab548d455c7270d4acd3` were released under MIT and stay MIT for anyone who obtained them then. That grant is not being clawed back. [`LICENSE-HISTORY.md`](LICENSE-HISTORY.md) marks the boundary and gives a command to check which grant covers a file you hold.

### The terms travel with the module

A scoring module is fetched as a bare `.wasm`, so whoever ends up holding one has no `LICENSE` and no `NOTICE` beside it. So the notice goes inside the binary, in an inert custom wasm section named `license`. A runtime ignores every custom section, the exports are unchanged and `rank_answer` returns the same `f32` for the same input, checked against the unstamped build under wazero. But `strings module.wasm` now shows the terms. `rev/stamp.py` writes the section and verifies it. The registration drivers refuse to register a module that does not carry it, so this is enforced by the pipeline rather than trusted to memory. Modules registered before the 2026-08-30 cutover do not carry the section and are not being re-stamped, because changing one byte changes the keccak and breaks a live registration.

### When the work is mostly someone else's

Some builds are forks of another author's source, reused under their licence. Their terms govern those portions and ours cover only our addition, so their notice goes first, both in `NOTICE` and inside the binary. Three MIT upstreams are used this way and named in full: `assay` by GreatSage-dev (FACT_CHECK), `telegraph-wasm-scoring` by ssoni4751 (CHAT_COMPLETION) and `amanat` by Pugar Huda Mantoro (GAME_RESULT). The embedded model weights keep their own licences too, all-MiniLM-L6-v2 under Apache-2.0, gte-small under MIT, GloVe under PDDL-1.0, with full texts in [`LICENSES/`](LICENSES).

A third lineage is not allowed to ship. A `binary_wrap` is a rescaling of another author's compiled binary that we did not have the source for. Forty-four such files, built on three upstreams that published no licence at all, were withdrawn on 2026-08-30. No licence means no permission to redistribute a modified copy, so they came out of the tree whatever they scored. [`PROVENANCE.json`](PROVENANCE.json) is generated from the bytes by `rev/provenance.py`. It groups each build with its base by a data-section fingerprint and reports the three counts. `binary_wrap` must read zero. It does. [`WITHDRAWN.json`](WITHDRAWN.json) keeps the full record of the 44, with hashes, because removing a file from a git tree does not unpublish the bytes a pinned URL still serves.

## Verify any of it

Every claim here is checkable without our help.

The keccak256 of a file matches the `WasmHash` in its on-chain registration. Worked against the live CHAT_COMPLETION champion, registration 2810:

```bash
# what the node says: the author, the commit-pinned URL, the stored hash
curl -s https://devnode.telegraphprotocol.com/engine/validator/v1/wasm/2810
# -> AuthorAddress 0x8b224783fe5b3c52b7db0cb9b1754f8812b75287
#    WasmHash     8db9d289440d870dba24c5f74aeed9771e688cfeaaa308743e76e1a59c6e851e

# the bytes it points at, hashed, equal that WasmHash
curl -sL "<WasmURL from above>" \
  | python3 -c "import sys; from Crypto.Hash import keccak; h=keccak.new(digest_bits=256); h.update(sys.stdin.buffer.read()); print(h.hexdigest())"
# -> 8db9d289440d870dba24c5f74aeed9771e688cfeaaa308743e76e1a59c6e851e
```

Who authors a live champion, straight off the node. An intent id is the keccak256 of its name:

```bash
IID=$(python3 -c "from Crypto.Hash import keccak; h=keccak.new(digest_bits=256); h.update(b'CHAT_COMPLETION'); print(h.hexdigest())")
curl -s "https://devnode.telegraphprotocol.com/engine/validator/v1/intents/$IID" \
  | python3 -c "import sys, json
for w in json.load(sys.stdin)['wasm']:
    if w['status'] == 'active': print(w['registration_id'], w['address'])"
# -> 2810 0x8b224783fe5b3c52b7db0cb9b1754f8812b75287
```

The terms are in the bytes. The lineage sums the way it should:

```bash
strings dist/reclaim5/ccr_t050top002.wasm | grep -A6 SPDX      # the SAND-1.0 notice
python3 rev/stamp.py dist/reclaim5/ccr_t050top002.wasm --check  # report, change nothing
python3 rev/provenance.py > PROVENANCE.json                     # regenerate lineage from dist/
python3 -c "import json; print(json.load(open('PROVENANCE.json'))['counts'])"
# -> {'own': 1429, 'source_fork': 32, 'binary_wrap': 0}
```

## Layout

```
dist/                the compiled modules, exactly as registered on-chain
LICENSE              Source-Available No-Derivatives Licence 1.0 (our own builds)
LICENSE-HISTORY.md   the MIT to SAND boundary, dated, with a per-file check
NOTICE               every third-party component and the licence it carries
LICENSES/            full licence texts, so they travel with the code
PROVENANCE.json      per-file lineage, generated from the bytes
WITHDRAWN.json       the 44 withdrawn binary_wrap files, with hashes
rev/                 stamp.py (licence into the bytes) and provenance.py (lineage)
```

The source, the harness, the drivers and the worklogs are in the sibling lab, [`telegraph-scorer-lab`](https://github.com/zkasuran/telegraph-scorer-lab).

## Licence

The modules here are the work of zkasuran under [`LICENSE`](LICENSE), the Source-Available No-Derivatives Licence 1.0. Run them for any purpose including commercially, read them, disassemble them, measure them, publish what you find. Do not redistribute them, publish a modified copy or register a build of one as your own scoring module. Where a file is built on another author's work their licence governs that file, named in [`NOTICE`](NOTICE) with full texts in [`LICENSES/`](LICENSES) and full lineage in [`PROVENANCE.json`](PROVENANCE.json).

<p align="center">
  <sub>Every byte earned on-chain.</sub>
</p>
