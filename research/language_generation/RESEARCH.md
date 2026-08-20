# LANGUAGE_GENERATION champion: research and candidate

Intent: `LANGUAGE_GENERATION` (Telegraph Track 2, Base Sepolia). 10 live miners,
8 with real scores. This is one of the traffic-gated intents we do not yet hold.
Goal: find what the real champion ranks on and build a scoring module that can
clear the three node gates (ordering, margin, traffic agreement >= 0.60 Spearman).

Date: 2026-08-20. Author lane wallet `0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287`.

## TL;DR

- The champion for LANGUAGE_GENERATION is the node's **built-in default scorer**,
  not a downloadable WASM. All five WASM registrations for this intent (regs
  5, 6, 8, 11, 13 from rival authors) are `rejected`, and
  `https://devnode.telegraphprotocol.com/wasm/LANGUAGE_GENERATION.wasm` is 404.
  Only `/wasm/good.wasm` exists, and that is a different module (see below).
- The default scorer is **deterministic** (WASM sandbox, no network, no LLM
  judge is possible), it scores exact self-matches at 1.0, and it posts an
  identical benchmark margin of **0.37360683** on ~40 different intents. So it is
  one fixed function used everywhere an author has not won the slot.
- On LANGUAGE_GENERATION **real traffic** its ranking has essentially zero
  rank-correlation with the two content-similarity axes we already tried on the
  node: precision word-overlap to ground truth **0.0053** and MiniLM
  sentence-embedding similarity **-0.0034**. Both are dead.
- Its per-miner ordering is the **near-inverse** of the topical CHAT_COMPLETION
  champion's, and it puts the verbose/grounded miners on top and the terse
  instant miner last. Best surviving hypothesis: it ranks on an **answer
  detail/length (verbosity / ground-truth coverage) axis**, orthogonal to
  content-similarity. That is reproducible in no_std WASM.
- Built a lexical candidate that blends a monotone length/coverage reward
  (`W_LEN = 0.45`) into the correctness score. It clears ordering and margin
  locally and every attack. Traffic agreement can NOT be verified locally: the
  champion is not downloadable and the node exposes no per-answer champion score,
  so the length axis is a named PROXY and the node read is the only real test.

## The three gates (from the docs, verified)

Source: https://docs.telegraphprotocol.com/docs/scoring/build-a-scoring-module
(fetched 2026-08-20, follows a 307 to the docs SPA). The module gets
(question, ground_truth, miner_answer) and returns f32 in [0,1]; blank -> 0.
Stage 1 structural checks, Stage 2 must match-or-beat the champion on a hidden
32-case benchmark (ordering wins, mean good-minus-bad margin, absolute floor,
worst_self_match >= 0.75, score_stddev floor). Then, only where an intent has
traffic history, a Spearman >= 0.60 agreement gate against the champion's
ranking of real answers.

## Evidence

### 1. The champion is the universal default scorer, deterministic, not an LLM judge

Read from our own registrations' `EvalDetails` at
`/engine/validator/v1/addresses/0x8b224783...`. Every intent we have ever been
scored against reports the same `champion_margin` value, `0.37360683`, on the
benchmark:

    IP_GEOLOCATION, CHAT_COMPLETION, FACT_CHECK, SENTIMENT_ANALYSIS, CRYPTO_PRICE,
    LANGUAGE_GENERATION, TASK_COMPLETION, AGENT_TASK, WEB_SEARCH, TEXT_GENERATION,
    ... (~40 intents), all champion_margin = 0.37360683

(The only exceptions are the two intents where a WASM was actually promoted:
AI_TEXT_DETECTION 0.7930425 and FINANCIAL_DATA 0.50372154.) One fixed number
across 40 unrelated intents means one fixed default scorer. The scoring sandbox
has **no network access** (docs, verified), so the default cannot be a live
LLM-as-judge. It is a deterministic function. `worst_self_match = 1` on our
LANGUAGE_GENERATION registrations shows it scores an exact match at 1.0 (an
exact-match short-circuit like the docs' reference example).

`good.wasm` is NOT this default. Run through our harness on the 40-case
benchmark it posts candidate_margin **0.0133** and worst_self_match **0.5935**,
nowhere near 0.3736 / 1.0. `good.wasm` is a 24 MB MiniLM sentence transformer,
the CHAT_COMPLETION champion, a separate thing.

### 2. Content-similarity is dead on this intent (both axes, on the node)

Our two real LANGUAGE_GENERATION registrations and the Spearman the node
recorded on real traffic:

| reg | build | candidate_margin | wins | historical rows | spearman vs champion |
|-----|-------|------------------|------|-----------------|----------------------|
| 79  | MiniLM transformer blend (W_EMB) | 0.5723 | 32/32 | 99  | **-0.0034** |
| 103 | lexical correctness (W_EMB 0) | 0.7293 | 32/32 | 104 | **0.0053** |

Both pass ordering and margin with room to spare and both fail only gate 3. A
scorer that ranks by similarity-to-ground-truth (whether lexical precision or
transformer topicality) has zero correlation with what the champion does here.

### 3. The champion's ordering is the inverse of the topical champion's

Live leaderboards (`/leaderboard/miners?intent=...`, epoch 231):

    LANGUAGE_GENERATION (default champion)     CHAT_COMPLETION (good.wasm, topical)
    1 chatbot        0.6758                     1 qwen           0.8187
    2 nova-2-lite    0.6730                     2 groq           0.8148
    3 deepseek       0.6518                     3 voxtral        0.8147
    4 qwen           0.6199                     4 litellm        0.8145
    5 litellm        0.6019                     5 nova-2-lite    0.7824
    6 kimi           0.5949                     6 kimi           0.7172
    7 voxtral        0.5888                     7 deepseek       0.5510
    8 groq           0.5702                     8 chatbot        0.5025

`telegraph-chatbot` is last on the topical champion and first on the
LANGUAGE_GENERATION champion; `groq-llama-3.1-instant` is second on the topical
one and last here. The two orderings are close to reversed. The miners the
LANGUAGE_GENERATION champion ranks highest are the thorough/grounded ones
(a knowledge assistant, a web-grounded model with citations, a reasoning model);
the lowest is the fast instant model. That is a detail/length ordering, not a
topicality one, and it is why MiniLM scored -0.0034: it ranks the opposite way.

Confirmed one point directly: the public groq miner
(`https://telegraph-miner-node.onrender.com/chat`) returns compact ~4-sentence
answers and sits last. The 127.0.0.1 miners are not reachable off-node, so a full
per-miner length regression against the 8 leaderboard scores is not possible
without paying the x402 ask gate (a real-money step we do not take).

### 4. Why length can separate good from bad on the benchmark too

On the hidden fixture set the "good" answer covers the ground-truth content and
the "bad" one does not, so a coverage/length-aware score still orders good above
bad (our candidate wins 40/40 locally with margin 0.383). The axis only *looks*
orthogonal on real traffic, where every miner answer is on-topic and
content-similarity saturates, leaving the length/detail difference as the thing
that still varies. This is the same shape as the CHAT_COMPLETION story (match the
champion's axis for the traffic gate, keep correctness for the fixture gate),
except the axis here is length/coverage instead of embeddings.

## Hypotheses considered and ruled out

- **LLM-as-judge quality score.** Ruled out: the scoring sandbox has no network,
  and one identical benchmark margin across 40 intents means a fixed deterministic
  function, not a per-answer model call.
- **Precision word-overlap to ground truth.** Ruled out on the node: 0.0053.
- **Sentence-embedding / topical similarity (MiniLM).** Ruled out on the node:
  -0.0034, and its per-miner order is inverted.
- **Near-flat / pure-noise ranking (unwinnable).** Not ruled out. The exactly-zero
  correlations are also consistent with the champion scoring every reasonable
  answer almost identically, so that its ranking is dominated by tie-break noise
  no deterministic scorer can track. Against this: the per-miner aggregate spread
  is a stable 0.10 wide and orders sensibly by detail, which pure noise would not
  produce. This is the main reason confidence is not higher.
- **Answer length / verbosity / ground-truth coverage (kept).** Reproducible,
  fits the inverted ordering and the two dead content axes. This is what the
  candidate builds to.

## The candidate

`module/src/lib.rs` gains a monotone length reward `length_reward(ma) =
n/(n+LEN_HALF)` (n = whitespace token count, `LEN_HALF = 70`, discriminating
across the ~20-400 word range real generations occupy) and a weight `W_LEN`.
The final score is `(1 - W_LEN)*correctness + W_LEN*length_reward`. Every other
intent keeps `W_LEN = 0.0` and is untouched. This build sets `W_LEN = 0.45`,
intent marker `LANGUAGE_GENERATION`, lexical (no minilm feature, ~1.04 MB).

Rationale for the blend rather than a pure length scorer: a pure length scorer
fails Stage 1 (a long unrelated answer would beat a short correct one) and the
ordering gate. The blend keeps the correctness term that carries the fixture
separation while letting length drive the ranking of real, all-on-topic answers.

### Local numbers (harness, real tool output)

Benchmark (`bench/benchmark.json`, 40 cases) + attacks (`bench/attacks.json`,
12) with the default word-overlap reference as the local baseline:

- ordering: **40/40** wins, 0 ties (baseline word-overlap 13/40)
- candidate_margin: **0.3832** (> node champion 0.37360683)
- worst_self_match: **1.0** (floor 0.75)
- score_stddev: **0.2204** (above floor)
- attacks: **12/12 passed**, all gates passed

Note the local baseline is the docs' word-overlap example, not the real default
champion (which is not downloadable). The real champion's benchmark margin is
0.37360683; our 0.3832 clears it, and our two earlier builds already showed 32/32
ordering wins on the node's own benchmark, so Stage 1 + Stage 2 are not the risk.

### Traffic agreement: PROXY only, cannot be verified locally

The real champion is not downloadable and the node publishes no per-answer
champion score, so there is no way to compute the real Spearman offline. The
named PROXY is **answer length / detail**. On a generated corpus of open-ended
LANGUAGE_GENERATION answers at four detail levels per prompt
(`bench/traffic-langgen.json`, house-gateway generated, terse to detailed, n=13
scored; generation is slow on the gateway's reasoning model, rerun
`tools/gen_langgen.py` to extend and `tools/analyze_langgen.py` to re-measure):

- Spearman(candidate, answer length): **0.618**
- Spearman(candidate, ground-truth recall/coverage): **0.645**
- Spearman(candidate, word-overlap precision): **-0.429**
- Spearman(answer length, word-overlap precision): **-0.778**
  (longer answers carry more words absent from the short reference, so precision
  and length pull against each other; the candidate follows length, not precision)

The candidate's ranking follows the length/coverage axis and runs *against* the
precision axis the node already proved dead (0.0053). This positions it on the
one untested axis. It does NOT prove agreement with the real champion; only the
node read does that, which the orchestrator performs. The earlier n=8 slice gave
0.69 / 0.90 / -0.21, same direction, so the sign and rough magnitude are stable.

## Winnability

**Confidence: low-to-medium.** The signal (length/coverage) is coherent,
reproducible in WASM and consistent with every piece of real evidence, and the
candidate clears the two gates we can measure. What keeps this from higher: the
exactly-zero content correlations also fit a near-flat champion that nothing can
track, and even if length is the axis, a 0.45 length blend may land below the
0.60 floor and need a stronger blend (which trades against the margin gate). No
local number can settle it because the champion is not observable offline.

**Recommend register: yes.** Length is the one untested axis (word-overlap and
MiniLM are both spent), the candidate passes the local gates so it will not be
rejected before the traffic gate runs, and a single gas registration buys the
real Spearman on the length axis, which is the only way to resolve the
near-flat-vs-length question. If the node returns a Spearman well above 0.0053
but below 0.60, raise `W_LEN` (and soften penalties) and re-register; if it comes
back near zero again, the champion is effectively unrankable and we stop.

## Sources

- Docs, scoring module: https://docs.telegraphprotocol.com/docs/scoring/build-a-scoring-module
- Docs, engine ask (x402 gate on traffic): https://docs.telegraphprotocol.com/docs/using/engine-ask
- Protocol overview (validators run author evaluation scripts): https://telegraphprotocol.com/
- Node intent detail: https://devnode.telegraphprotocol.com/engine/v1/intents/LANGUAGE_GENERATION
- Node leaderboards: https://devnode.telegraphprotocol.com/leaderboard/miners?intent=LANGUAGE_GENERATION and ?intent=CHAT_COMPLETION
- Our registrations + EvalDetails: https://devnode.telegraphprotocol.com/engine/validator/v1/addresses/0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287
- good.wasm (CHAT_COMPLETION champion, different module): https://devnode.telegraphprotocol.com/wasm/good.wasm
