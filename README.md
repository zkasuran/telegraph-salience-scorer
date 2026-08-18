# Telegraph salience scorer

A scoring module for the [Telegraph protocol](https://telegraphprotocol.com): the
WASM program a Telegraph node runs to decide how good a miner's answer was. It
takes the question, the ground truth and the miner's answer and returns one
`f32` between 0 and 1.

Written for Telegraph Hackathon Season I, Track 2 (Script Authors). The Track 1
miner that goes with it is at
[telegraph-gaswire-miner](https://github.com/zkasuran/telegraph-gaswire-miner).

The starting point that ships with the protocol scores word overlap: what
fraction of the answer's words also appear in the ground truth. That is a
reasonable floor and it has two failure modes that matter in production. It pays
out for answers that share vocabulary with the ground truth while asserting the
opposite ("the certificate has **not** expired") and it pays nothing for correct
answers phrased in different words ("bond prices tend to **increase**" for a
ground truth of "bond prices usually **rise**"). Both wreck a leaderboard: the
first lets a miner farm score without answering, the second demotes miners who
answer well.

This module is built around that gap. Words are weighted by how much they
actually pin down the answer, coverage is measured against what the question did
not already give away and the handful of features that flip an answer from right
to wrong (numbers, negation, polarity) are checked before overlap is allowed to
count for anything.

## Interface

Three exports, the ones the node calls:

| Export | Signature | Purpose |
| --- | --- | --- |
| `alloc` | `(i32) -> i32` | hand the node memory to write the three strings into |
| `dealloc` | `(i32, i32)` | no-op, every call gets fresh memory |
| `rank_answer` | `(i32,i32,i32,i32,i32,i32) -> f32` | `(q_ptr,q_len, gt_ptr,gt_len, ma_ptr,ma_len)` |

`no_std`, no allocator, no imports at all, so it instantiates in the node's
sandbox with nothing bound. Every buffer is a fixed static and every loop is
bounded, so a 76 KB answer costs a predictable amount of work rather than an
unpredictable amount of memory. All parsing is byte level: the input is whatever
a miner sent, so emoji, CJK, right to left script and invalid UTF-8 all have to
score without trapping.

Compiled size: 8.4 KB.

## How it scores

1. **Exact answer, exact 1.0.** Compared over word bytes only, so case, spacing
   and punctuation cannot cost a perfect answer its perfect score.
2. **Blank is exactly 0.0.** Empty, whitespace, punctuation only.
3. **Salience weighting.** Numbers weigh most, proper nouns next, ordinary words
   by length, function words and assistant boilerplate ("sure", "happy to help",
   "in conclusion") almost nothing. It is a corpus-free stand-in for IDF and it
   is what separates "same topic" from "same answer".
4. **Precision and recall on those weights.** Precision is concave, so a correct
   answer that adds supporting context keeps its score while a shotgun list of
   candidates collapses. Recall is measured first against the part of the ground
   truth the question did not already contain, because that part is the answer
   and the rest is the prompt coming back.
5. **Character trigrams,** taking the better of symmetric Dice and how much of the
   ground truth's structure is present in the answer. This is what recognises a
   reworded answer and the asymmetric half is why boilerplate padding does not
   read as a wrong answer.
6. **The facts that flip an answer.** Numbers in the ground truth have to appear,
   and a contradicting number costs most of the score. Polarity is read on three
   independent axes (verdict, authenticity, direction), negation aware, so "not
   valid" is negative and "No, written by a human" is negative on the verdict and
   positive on authenticity at the same time. Agreeing on an axis earns credit
   even when the wording differs; contradicting one costs 85% of the score.
7. **Adjacency, narrowly.** An answer carrying exactly the ground truth's content
   words, nothing missing and nothing added, that shares no content-word
   adjacency with it, is "France is the capital of Paris". A bag of words cannot
   tell that from the truth. This can.
8. **Contrast.** A smoothstep, so confident matches move up and near misses move
   down, without flattening the middle: a module whose scores do not vary is
   rejected by the node and one that is all or nothing cannot rank the answers
   in between.

## What it will not reward

Every row is a test in `bench/attacks.json`, run by the harness.

| Attempt | Result |
| --- | --- |
| Echo the question back | 0.10 against 0.89 for the real answer |
| List every candidate answer | 0.18 against 0.89 |
| Reuse the ground truth's words, insert a negation | 0.20 against 0.39 |
| Reuse the ground truth's words, flip the verdict | 0.05 against 0.92 |
| Same claim, opposite direction | 0.01 against 0.31 |
| Right shape, wrong figure | 0.14 against 0.46 |
| Same words, reordered into a different claim | 0.55 against 0.99 |
| Function-word padding | 0.01 against 0.94 |
| Right answer buried in a keyword dump | 0.02 against 0.96 |
| Punctuation only | 0.00 |
| A correct answer wrapped in assistant boilerplate | 0.98, still correct |
| A correct answer with emoji and mixed scripts | 0.71, still correct |

The last two matter as much as the rest. A scorer that punishes noise punishes
real miners, who pad and emoji constantly.

## Measured

`harness/` loads a `.wasm` the way the node does (wazero, no host imports,
strings written through the module's own `alloc`) and reports the same numbers the
node records on a registration, against a 40 case benchmark in `bench/`
(question, ground truth, a good answer worded differently, a plausible wrong
answer) spanning 20 canonical intents.

| | this module | reference word-overlap module |
| --- | --- | --- |
| `candidate_margin` (mean good minus bad) | **0.678** | -0.117 |
| good ranked above bad | **40 / 40** | 13 / 40 |
| mean score, good answers | 0.754 | 0.232 |
| mean score, wrong answers | 0.076 | 0.349 |
| `worst_self_match` (node floor is 0.75) | 1.000 | 1.000 |
| `score_stddev` | 0.391 | 0.286 |
| structural gates | 8 / 8 | 8 / 8 |
| gaming and robustness suite | 12 / 12 | 3 / 12 |

The reference module scores wrong answers higher than right ones on this
benchmark (a negative margin) because a plausible wrong answer usually reuses the
question's words and word overlap cannot see the difference.

## Build

```bash
rustup target add wasm32-unknown-unknown        # once
cd module && cargo build --release --target wasm32-unknown-unknown
# -> target/wasm32-unknown-unknown/release/telegraph_scorer.wasm
```

Must be the `wasm32-unknown-unknown` target. A `wasm32-wasip1` build carries WASI
imports (`fd_write`, `proc_exit`) and a Telegraph node runs modules with no WASI
and no OS, so a WASI build fails to instantiate.

## Verify

```bash
cd harness && go build -o harness .
./harness ../bench/benchmark.json ../bench/attacks.json \
  ../module/target/wasm32-unknown-unknown/release/telegraph_scorer.wasm \
  [any other .wasm to compare against]
```

It exits non-zero if the candidate misses any gate. What it checks, in the node's
own terms: the module loads with no imports and exports `alloc`, `dealloc`,
`rank_answer` plus linear memory; a blank answer scores exactly 0; a perfect
answer scores at least 0.75 everywhere; a correct answer beats an unrelated one on
every case; a 76 KB answer, an oversized ground truth and emoji / CJK / RTL /
invalid UTF-8 input neither trap nor leave [0,1]. Then the benchmark and the
gaming suite above.

Two optional inputs, both used by `deploy.py` on every build it registers:

```bash
FAMILY=../bench/family-numeric.json          # a second benchmark for one answer shape
CORPUS=../bench/traffic.json \
BASELINE_SCORES=../bench/champion-corpus-scores.json   # rank agreement with the champion
PROBE="question|ground truth|answer"         # score one triple and exit
```

`bench/report.json` is the last run, per case, checked in so the numbers in this
README can be diffed rather than trusted.

## Family benchmarks

A general benchmark cannot tell you whether a build is right for the intent it is
registered against, so each family of intents has its own fixture set and a build
has to clear it as well as the general 40:

| family | file | intents | what the cases turn on |
| --- | --- | --- | --- |
| numeric | `bench/family-numeric.json` | CRYPTO_PRICE, CURRENCY_EXCHANGE, STOCK_PRICE, TVL_LOOKUP | the figure, its unit, its magnitude, its direction, and which entity it is attached to |
| authenticity | `bench/family-authenticity.json` | IMAGE_VERIFICATION, VIDEO_VERIFICATION, MEDIA_AUTHENTICITY_CHECK, CONTENT_VERIFICATION | a verdict about whether something is genuine, where the wrong answer shares almost every word |
| reference | `bench/family-reference.json` | IP_GEOLOCATION, NEWS_HEADLINES | naming the right entity, with wrong answers that are plausible neighbours of the right one |

The gate is: every case won bar at most one, family margin at least 0.40, perfect
answers still at 1.000. One documented miss is allowed because the families
deliberately include cases past what a lexical scorer can reach, and deleting those
would be the dishonest way to a clean sheet. The current miss is
`ref-ip-hosting`: the ground truth says AWS, the good answer says Amazon, and no
amount of character overlap gets you from one to the other. That needs an entity
alias table this module does not ship.

Writing those families paid for itself immediately. `auth-img-real` failed, and the
cause was not the scoring at all: `bnd`, the flag marking a clause boundary, was the
one per-token field `tokenize` did not write on every push, so a previous call's
boundary survived into the next one. "no" in "Authentic, no sign of manipulation"
was then read as a standalone verdict and flipped a correct answer into a
contradiction. The score depended on how many calls had come before it, which is the
one thing a scorer must never do. Fixed, and the fix is why every build registered
after 2026-08-17 evening carries a different binary from the eight before it.


## Deployed

Live on Base Sepolia (84532), registry `0x5a2324aA18613FAD4e44bDF0d6c73Ec1f6D87ff8`,
via `registerWasm(bytes32 wasmHash, string wasmUrl, string intent)`. The hash is
keccak256 of the `.wasm` bytes; a miner YAML uses sha256, a scoring module uses
keccak256.

This module is the **active scoring module for eight canonical intents**, which is
to say it is the program that decides how the miners serving them are ranked:

| Reg | Intent | node margin | champion | fixture wins |
| --- | --- | --- | --- | --- |
| 26 | AI_TEXT_DETECTION | 0.793 | 0.374 | 32/32 |
| 27 | FACT_CHECK | 0.789 | 0.374 | 32/32 |
| 28 | URL_SCAN | 0.789 | 0.374 | 32/32 |
| 29 | DEEPFAKE_DETECTION | 0.789 | 0.374 | 32/32 |
| 30 | SSL_VERIFICATION | 0.789 | 0.374 | 32/32 |
| 31 | SENTIMENT_ANALYSIS | 0.789 | 0.374 | 32/32 |
| 32 | CVE_LOOKUP | 0.808 | 0.374 | 32/32 |
| 33 | ACADEMIC_SEARCH | 0.808 | 0.374 | 32/32 |

```bash
curl -s https://devnode.telegraphprotocol.com/engine/validator/v1/addresses/0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287
```

One build per intent: the intent is baked into the binary as `TELEGRAPH_INTENT` and
the tunables are set for the shape of answer that intent returns (`deploy.py`).
`dist/` holds the binaries as registered. Rebuilding from `module/` reproduces them
byte for byte with the same toolchain.

## CHAT_COMPLETION: rejected four times and why that is interesting

CHAT_COMPLETION is the busiest intent and the one place this module is not live. It
clears the first two gates comfortably and fails the third:

| build | our margin | champion | fixture wins | agreement with champion on 66 real answers |
| --- | --- | --- | --- | --- |
| lexical (reg 24) | 0.711 | 0.374 | 31/32 | gate not reached |
| lexical (reg 25) | 0.818 | 0.374 | 32/32 | 0.308, floor 0.60 |
| 50d vectors (reg 38) | 0.770 | 0.374 | 32/32 | 0.391 |
| 300d vectors (reg 39) | 0.798 | 0.374 | 32/32 | 0.385 |

The incumbent is a 24 MB module that is almost entirely an embedded table and it
rates a confidently wrong but on-topic answer around 0.6 where this module rates it
near zero. On a benchmark of good against bad answers that strictness wins, nearly
two to one. On a *ranking* of real answers it reorders the middle of the pack, and
the protocol will not hot-swap a scorer that moves live rankings that far.

Adding semantic capability moved the agreement from 0.308 to 0.391 and then stopped:
300 dimension vectors, which separate synonymy from mere topicality far better than
50 (rise/increase 0.67 against rise/fall 0.63, where at 50d the pair was inverted),
scored 0.385. So the remaining gap is not vector quality. It is that the two modules
are ranking on different things and closing it means being less strict about
correctness on purpose. That trade is available and it is not one worth making, so
this is recorded rather than papered over.

## How this was built

Written for the hackathon by [zkasuran](https://github.com/zkasuran) with AI
assistance (Claude, Anthropic). Every number in this README comes from the
harness in this repo run against the checked-in binary, not from an estimate. The
benchmark and the attack suite are original to this repo: Telegraph's own Stage 2
benchmark is not public, so this is a proxy for it, built from the behaviour the
protocol documents.

## Licence

MIT. See `LICENSE`.
