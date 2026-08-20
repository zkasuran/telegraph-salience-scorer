# WEB_SEARCH scorer: champion hypothesis, candidate, and winnability

Target: win the Track-2 champion slot for the **WEB_SEARCH** intent on the Telegraph
devnet node (Base Sepolia). 7 live miners. This file is the durable record: the
hypothesis, the evidence, every source, the local numbers, and an honest call.

## TL;DR

- WEB_SEARCH is a **Tier B "LLM-Judge"** intent (node docs), the *same tier* as
  CHAT_COMPLETION, which we already won on the node with a ported MiniLM transformer
  (reg 77, node Spearman 0.6266).
- The champion is **not downloadable** (`/wasm/WEB_SEARCH.wasm` -> 404; only
  `/wasm/good.wasm` -> 200). So there is no local check against the *real* judge. Our
  proxy is **`good.wasm`, the downloadable Tier-B MiniLM sentence-transformer** that is
  CHAT_COMPLETION's champion. Every "agreement" number below is against that proxy, not
  the real WEB_SEARCH champion. The node is the only real judge.
- Best candidate: the **same transformer build that won CHAT_COMPLETION**, re-marked for
  WEB_SEARCH (MiniLM embA/embB blended with a softened lexical/correctness score).
- Local gates all clear against the proxy (ordering, margin, agreement). Attacks show the
  documented topical-blindness cost. Honest confidence it clears the node's 0.60 traffic
  gate: **medium** (the identical architecture cleared it on CHAT_COMPLETION at 0.6266,
  with a ~0.24 local-to-node gap on that intent).

## 1. What the champion ranks on (hypothesis)

The WEB_SEARCH champion is a **sentence-embedding topical scorer of the MiniLM class**:
it scores a miner answer by semantic similarity to an LLM-supplied reference answer, with
at most a small word-overlap (BM25) component. It is *not* a lexical/word-overlap scorer,
and it is *not* an out-of-band LLM call (the scorer itself is a deterministic in-sandbox
WASM module; the "LLM" in Tier B supplies the `ground_truth` context, which is then handed
to the WASM as `gt`).

Why this shape and not the alternatives:

- **Docs classify WEB_SEARCH as Tier B — LLM-Judge**, defined as "the answer is
  open-ended; a language model supplies context and the WASM module scores quality
  against it." CHAT_COMPLETION, LANGUAGE_GENERATION, TASK_COMPLETION, NEWS_SEARCH etc. are
  all in the same tier. The node ships one downloadable Tier-B champion, `good.wasm`, a
  384-dim MiniLM sentence transformer whose score is
  `0.25*embA + 0.50*embB + 0.15*bm25(gt,answer) + 0.10*bm25(question,answer)` (dissected
  earlier for the CHAT_COMPLETION build). The most parsimonious hypothesis is that
  WEB_SEARCH runs the same default Tier-B judge, or a close sibling.
- **Word overlap is anti-correlated with the champion on the node (-0.1343).** A pure
  word-overlap lexical scorer registered for WEB_SEARCH scored a *negative* traffic
  agreement. That rules out a lexical champion and points to a semantic one: two competent
  LLM web-search syntheses of the same query share facts but not phrasing, so raw word
  reuse is close to noise (slightly negative), while embedding similarity to the reference
  tracks answer quality. This is a stronger anti-lexical signal than CHAT_COMPLETION
  showed (+0.308 lexical there), consistent with WEB_SEARCH answers being longer,
  source-citing syntheses where phrasing diverges even more.
- **The live per-miner spread is wide and discriminating**, which fits a semantic judge
  that both rewards good syntheses and zeroes out off-topic miners (see evidence).

The task's suggested directions (conciseness, freshness, source-naming) are *properties an
embedding scorer already rewards indirectly* (a concise, fact-dense, source-naming answer
sits closer to a good reference than a padded or parroting one), rather than separate
signals the champion measures explicitly. We did not find evidence for a dedicated
length/citation term, so we did not bet the build on one.

## 2. Evidence

### 2.1 Node tier classification (primary)

`docs/using/intents.md`:

```
| WEB_SEARCH | B — LLM-Judge | General research agents, scrapers |
```

Tier B is "LLM context + WASM. A language model supplies context and the WASM module
scores quality against it." CHAT_COMPLETION, NEWS_SEARCH, RESEARCH_QUERY, LANGUAGE_GENERATION,
TASK_COMPLETION, AGENT_TASK are all Tier B. Tier A (deterministic, exact match) is the
prices/weather/on-chain family. WEB_SEARCH is squarely in the semantic-judge tier.

### 2.2 Champion is not downloadable

```
WEB_SEARCH.wasm: 404
web_search.wasm: 404
good.wasm:       200   (24 MB, the Tier-B MiniLM champion)
```

So the real WEB_SEARCH champion cannot be dissected or scored locally. `good.wasm` is used
as the **named proxy** throughout.

### 2.3 Live leaderboard (epoch 231), the champion's own ranking of miners

```
telegraph-chatbot            0.7404723  rank 1   (LLM, web-grounded synthesis)
litellm                      0.7065477  rank 2   (LLM proxy, Nova web search)
bedrock-nova-2-lite          0.6657981  rank 3   (LLM, web-grounded)
tavily                       0          rank 4
cryptopulse-multisource      0          rank 5
fourcast-sports-intelligence 0          rank 6
sentinel-risk-oracle         0          rank 7
```

Reading: the three general LLM web-search answerers cluster tightly at **0.66-0.74**; the
four specialised/None-matching miners (a price worker, a sports feed, a DeFi-risk oracle,
a raw Tavily API) are floored at **0**. A wide 0-to-0.74 spread means the champion is
*discriminating*, not near-flat, and it separates "good open-ended synthesis" from
"off-topic or non-answer" sharply. That is embedding-topical behaviour, and it means a
candidate has real ranking room to match rather than a hard-flat target.

### 2.4 Known node agreement (given, not re-tested)

- word-overlap lexical: **-0.1343** (negative -> champion is not lexical).
- topical: not yet tested on the node -> this build is the topical test.

## 3. The candidate

Same architecture as the promoted CHAT_COMPLETION build, re-marked for WEB_SEARCH:

- `module/src/minilm.rs`: full `all-MiniLM-L6-v2` forward pass in `no_std` behind the
  `minilm` cargo feature (wordpiece, 6 layers, mean-pool, L2). Reproduces the champion's
  own `embed()` at cosine ~0.97. Blob `minilm.bin` (22.9 MB int8/f32).
- Blend in `lib.rs` (line ~1322), `W_EMB` gate on:
  `raw = 0.28*embA + 0.56*embB + 0.16*lexical`, then the correctness penalties multiply.
- Softened penalties (so strict correctness does not fight the topical ranking and sink
  the agreement gate): `M_CONTRA 0.7, M_NUM_WRONG 0.78, M_ORDER 0.85, M_ENTITY 0.72,
  M_NEGCOV 0.32, M_NUM_MISS_BASE 0.85, M_TWO_FACED 0.8`.
- Intent marker `WEB_SEARCH` (padded to 32 bytes).
- Built `--features minilm` at `opt-level = 3`, target `wasm32-unknown-unknown`.
- Size **23,959,595 bytes** (< 32 MB node cap), **0 imports** (no WASI).
- keccak256 **0x1b34e24999b632a8561481e874b4d6d91e56c1113291e323ee66433550b0cf0c**

Build type: **transformer clone (MiniLM embA/embB blend + softened lexical), NOT lexical.**

## 4. Local numbers (all real harness output)

Harness = the wazero CLI that loads a `.wasm` exactly as the node does. Proxy champion =
`reference/champion-good.wasm`. Default baseline = `reference/rust-module` (the docs'
word-overlap example, the intent's original scorer class).

### 4.1 Ordering + separation (node gates 1 and 2), on the 40-case benchmark

| module | candidate_margin | wins | worst_self_match | score_stddev |
|---|---|---|---|---|
| **our candidate** | **0.3014** | **34/40** | 1.0000 | 0.2270 |
| default word-overlap baseline | -0.1169 | 13/40 | 1.0000 | 0.2859 |
| topical proxy champion (good.wasm) | 0.0133 | 20/40 | 0.5935 | 0.1354 |

Head-to-head: candidate beats both on margin AND wins.
- vs default baseline: `margin 0.3014 vs -0.1169 | wins 34 vs 13` -> [ok]
- vs topical proxy champion: `margin 0.3014 vs 0.0133 | wins 34 vs 20` -> [ok]

This mirrors CHAT_COMPLETION exactly: the topical judge scores ~20/40 at margin ~0.01, and
our penalty-carrying transformer beats it comfortably on both counts. So gates 1 and 2 are
clear against the champion *class*.

### 4.2 Traffic agreement (node gate 3), Spearman vs the proxy champion

- **CHAT-shaped corpus** (`bench/traffic-real.json`, 125 rows, cached good.wasm scores):
  **Spearman 0.8668** (floor 0.60). [ok]
- **WEB_SEARCH-shaped corpus** (`research/web_search/traffic-websearch.json`, generated
  via the house gateway: 35 research questions x 6 varied-quality answers = 210 rows, plus
  a 90-row lean subset): the local agreement run against `good.wasm` was **CPU-starved by
  several concurrent sibling agents** running their own 24 MB transformer harnesses on the
  same host (four `champion-good.wasm` harnesses at 100% CPU simultaneously) and did not
  finish in the time budget. The corpus and lean subset are saved
  (`research/web_search/traffic-websearch.json`, `bench/traffic-ws-lean.json`) for a later
  uncontended re-run: `CORPUS=research/web_search/traffic-websearch.json ./harness/harness
  bench/mini.json bench/miniatk.json <candidate.wasm> reference/champion-good.wasm`.

Both agreement corpora use the same **good.wasm proxy** and the same Tier-B methodology
that the CHAT_COMPLETION win used, so the 0.8668 CHAT-shaped figure is the load-bearing
proxy-agreement number here. It is against the **proxy**, not the real champion. On CHAT_COMPLETION this same build
measured ~0.82-0.87 locally and landed **0.6266** on the node's real 66-answer gate, a
~0.24 local-to-node gap. Applying that gap here, a local ~0.85 predicts a node value near
the 0.60 floor: winnable, but not with wide room.

### 4.3 Attacks (our own gaming suite, 12 cases)

7/12 pass. 5 fail: `question-echo`, `verdict-flip`, `direction-flip`, `number-swap`,
`word-order-swap`. These are the **documented cost of tracking a topical champion**: to
agree with an embedding judge you must be lenient where it is lenient (a fluent on-topic
answer that flips a verdict or swaps a number still sits close to the reference in embedding
space). The real champion, being topical, fails these the same way, so they do not cost us
the head-to-head. They are recorded here honestly rather than hidden.

## 5. Winnability

- Gates 1 and 2 (ordering, margin): **clear** against both the default baseline and the
  topical proxy champion. High confidence these transfer, because the champion class scores
  ~20/40 / margin ~0.01 and we score 34/40 / margin 0.30.
- Gate 3 (traffic agreement >= 0.60): **the binding risk.** We can only measure it against
  the proxy, where we sit ~0.85. The same architecture cleared the real node gate on
  CHAT_COMPLETION at 0.6266. If the WEB_SEARCH champion is the same/adjacent MiniLM judge,
  this build should clear 0.60; if the real WEB_SEARCH judge diverges from good.wasm (a
  different reference-answer style, a citation/freshness term we could not observe), the
  node number could fall below 0.60 the way lexical did (-0.1343).

**Honest confidence: medium.** Recommend registering: the cost is gas only, and the
registration is the *only* way to read the real node agreement (it is not observable
offline). A rejection returns the exact candidate-vs-champion Spearman, which is worth more
than any further local tuning against a proxy. This is the same play that took
CHAT_COMPLETION after four measured rejections.

## 6. Sources

- Node docs, scoring gates: `docs/scoring/build-a-scoring-module.md` (three-gate promotion:
  ordering >= champion, margin >= champion + floor, worst_self_match >= 0.75, stddev floor,
  plus traffic agreement when the intent has history).
- Node docs, tiers: `docs/using/intents.md` (WEB_SEARCH = Tier B LLM-Judge).
- `https://devnode.telegraphprotocol.com/engine/v1/intents/WEB_SEARCH` (7-miner roster).
- `https://devnode.telegraphprotocol.com/leaderboard/miners?intent=WEB_SEARCH` (epoch 231
  per-miner champion scores).
- `https://devnode.telegraphprotocol.com/wasm/{WEB_SEARCH.wasm -> 404, good.wasm -> 200}`.
- `https://telegraphprotocol.com/` ("Validators score results using evaluation scripts
  written by Script Authors, then route traffic only to the best performer").
- Memory: `telegraph-wasm-promotion-gates`, `telegraph-lane-live-state` (CHAT_COMPLETION
  reg 77 promoted at node Spearman 0.6266, progression 0.308 lexical -> 0.6266 transformer).

## 7. Reproduce

```bash
cp -a work/telegraph/scorer /tmp/flag-WEB_SEARCH && cd /tmp/flag-WEB_SEARCH
# patch lib.rs: W_EMB=0.45, blend 0.28*ca+0.56*cb+0.16*raw, softened penalties, marker WEB_SEARCH
# Cargo.toml: opt-level = 3
cd module && cargo build --release --target wasm32-unknown-unknown --features minilm
# ordering/margin/attacks:
./harness/harness bench/benchmark.json bench/attacks.json \
  module/target/wasm32-unknown-unknown/release/telegraph_scorer.wasm \
  reference/champion-good.wasm
# agreement vs proxy on the generated WEB_SEARCH corpus:
CORPUS=research/web_search/traffic-websearch.json \
  ./harness/harness bench/benchmark.json bench/attacks.json \
  <candidate.wasm> reference/champion-good.wasm
```

Corpus generator: `research/web_search/gen_corpus.py` (house gateway, 35 web-search
questions x 6 quality-varied answers, concurrent, incremental checkpoint).
