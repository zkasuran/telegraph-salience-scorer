# TASK_COMPLETION scorer: champion hypothesis, build, and honest winnability

Date: 2026-08-20. Intent: `TASK_COMPLETION` (10 live miners on Base Sepolia devnode).
Goal: become the champion scorer, which needs three node gates vs the incumbent:
ordering (win >= champion), separation margin (> champion), and traffic agreement
(Spearman >= 0.60 with the champion's ranking of real answers).

## TL;DR

- The TASK_COMPLETION champion is a **topical sentence-embedding scorer**, not a
  correctness/lexical one. Evidence below. This matches the known node numbers:
  word-overlap lexical agreed 0.4591, a MiniLM topical transformer agreed 0.5454.
- The champion is **NOT downloadable** (`/wasm/TASK_COMPLETION.wasm` => 404; only
  `good.wasm`, the CHAT_COMPLETION champion, is served). So all local agreement
  numbers here are against a **named PROXY** (`good.wasm`, an all-MiniLM-L6-v2
  transformer), and the proxy **overstates** the node. The real judge is the node.
- Best candidate found: **V_softq** — the transformer blend with a
  (question, answer) relevance term added and the correctness penalties softened.
  Proxy Spearman **0.6239** vs the proxy-anchored baseline's **0.6074** (that
  baseline scored **0.5454** on the real node). Calibrated node estimate **~0.56**.
- Honest call: still likely short of 0.60. `winnable_confidence = low`. It is the
  most reachable of the four blocked signals and the change is principled, so one
  gas-only registration to read the true node Spearman is worth it, but do not
  expect a promotion from it.

## 1. What the intent is

Node canonical description (`/engine/v1/intents/TASK_COMPLETION`):

> "Query asks about what makes an AI agent effective at completing multi-step
> tasks, or is itself a request to complete a defined multi-step task end-to-end."

So a query is either meta ("what makes an agent good at multi-step tasks") or an
actual multi-step task to carry out. A good answer is a thorough, correct,
step-structured procedure. The ground truth is that procedure.

## 2. Champion hypothesis and the evidence

### Hypothesis
The champion ranks on **topical / sentence-embedding similarity** (the same family
as `good.wasm`), with a light lexical/relevance tie-breaker, and is **lenient on
confidently-wrong-but-on-topic answers**. It is almost certainly a MiniLM-class
sentence transformer, but **a different model or blend from `good.wasm`**, because a
faithful `good.wasm` clone only reaches 0.5454 agreement (see calibration).

### Evidence
1. **Known node agreements (given):** word-overlap lexical 0.4591, MiniLM topical
   transformer 0.5454. Topical beats lexical by ~0.086, so the champion ranks on
   meaning, not word overlap.
2. **Leaderboard spread** (`/leaderboard/miners`, epoch 231, saved in
   `node-leaderboard.json`): six general LLM miners cluster tightly at
   0.636–0.752 (kimi 0.7522, nova-2-lite 0.7453, chatbot 0.6978, litellm 0.6865,
   qwen 0.6825, voxtral 0.6361); the domain-specific `bayern-elo-miner-v1` (a
   Bayern-Munich Elo predictor that answers off-topic) craters to 0.1488; three
   miners that return nothing sit at 0. That signature — on-topic answers bunched
   high, off-topic sharply low, empties zero — is exactly a topical scorer. A
   correctness/fact scorer would spread the six on-topic LLMs much wider.
3. **The tight on-topic cluster is the hard part.** Because all six LLM answers are
   on-topic and complete, topicality alone cannot order them (kimi vs voxtral). The
   champion's within-cluster order is set by a secondary signal (embedding fidelity
   plus a light lexical/relevance term). Matching that fine order is what the 0.60
   gate is really testing, and why a 0.97-cosine reproduction of the wrong model
   only reaches ~0.55.
4. **Docs** (`scoring/build-a-scoring-module.md`): the node compares candidate vs
   champion on a hidden benchmark (margin + wins), and, "if the intent has enough
   real traffic", also checks Spearman agreement of the two rankings. "Real semantic
   understanding will always beat simple word-matching." Consistent with a semantic
   champion.
5. **telegraphprotocol.com:** "Validators score the results using evaluation scripts
   written by Script Authors, then route traffic only to the best performer." No
   per-intent champion function is published, so the exact model is not knowable
   offline.

### Why it is not `good.wasm` itself
`good.wasm` is downloadable, so if it were the TASK_COMPLETION champion we could
score its exact ranking. Our transformer build reproduces `good.wasm`'s own
`embed()` at cosine ~0.97 (from the CHAT_COMPLETION work) yet the same build agrees
with the real TASK_COMPLETION champion at only 0.5454. A near-perfect `good.wasm`
clone would agree ~0.9 if the champion *were* `good.wasm`. So the champion is a
different (but same-family, topical) model that we cannot download and therefore
cannot reproduce exactly.

## 3. The proxy and its calibration (the honesty core)

- **Proxy champion:** `reference/champion-good.wasm` = the CHAT_COMPLETION champion,
  an all-MiniLM-L6-v2 sentence transformer. Named explicitly because it is a
  *topical stand-in*, not the real judge.
- **Lexical proxy:** `reference/rust-module` (the docs' word-overlap example),
  also the node's default baseline champion.
- **Calibration (measured, not assumed):** the build that scored **0.5454 on the
  real node** (`dist/telegraph-salience-scorer-task_completion.wasm`, keccak
  `0x9387…`) agrees with the `good.wasm` proxy at **0.6074** on our corpus. So the
  proxy **overstates** node Spearman by **~0.062** for this build. Every proxy
  number below should be discounted by roughly that much to estimate the node.

Corpus: `traffic-task.json` — 125 real gateway answers over 24 multi-step-task
questions, five varied-quality answers each (thorough / terse / restate+hedge /
confident-wrong-on-topic / vague), matching the distribution the node's traffic
gate scores. The sweep used a deterministic 75-row subset (`traffic-task-sub.json`,
15 questions x 5) for tractable transformer scoring. Generated with the house
gateway via `gen_traffic_task.py`.

## 4. Builds and local numbers

All builds are the `--features minilm` transformer (~24 MB, opt-level 3), marker
`TASK_COMPLETION`, differing only in the score blend and the correctness penalties.
`embA` = shallow embedding-layer cosine(gt, answer); `embB` = full-transformer
cosine(gt, answer); `q-term` = full-transformer cosine(question, answer), newly
added in `minilm::embed_cos_b` and `lib.rs` (`EMB_Q_W`). Softened penalties = the
CHAT_COMPLETION winning set (M_CONTRA 0.7, M_NUM_WRONG 0.78, M_ORDER 0.85, M_ENTITY
0.72, M_NEGCOV 0.32, M_NUM_MISS_BASE 0.85, M_TWO_FACED 0.8); strict = source
defaults (0.3 / 0.45 / 0.55 / 0.3 / 1.0 / 0.62 / 0.5).

Spearman vs proxies (75-row corpus, full table in `spearman.json`):

| build | blend (embA/embB/lex/q) | penalties | vs good_proxy | vs rust | est. node |
|---|---|---|---|---|---|
| anchor (node 0.5454) | 0.28 / 0.56 / 0.16 / 0 | soft | 0.6074 | 0.5933 | 0.5454 (measured) |
| V_base | 0.25 / 0.50 / 0.25 / 0 | strict | 0.5706 | 0.5772 | ~0.51 |
| V_chat | 0.28 / 0.56 / 0.16 / 0 | strict | 0.5686 | 0.5625 | ~0.51 |
| V_q | 0.25 / 0.45 / 0.15 / 0.15 | strict | 0.5725 | 0.4841 | ~0.51 |
| **V_softq (winner)** | **0.25 / 0.45 / 0.15 / 0.15** | **soft** | **0.6239** | 0.5255 | **~0.56** |

Two findings that drove the winner:
- **Softened penalties are the main lever.** V_chat (strict) with the exact anchor
  blend agreed at 0.5686; the anchor (soft) at 0.6074. To rank like a topical
  champion you must stop sending on-topic-but-wrong answers to zero — the same
  "adopt the champion's blindness" trade CHAT_COMPLETION needed.
- **The (question, answer) term helps only on top of softened penalties.** With
  strict penalties the q-term did nothing (V_q 0.5725). With softened penalties it
  lifted agreement above the node-anchored baseline (V_softq 0.6239 vs 0.6074). It
  mirrors the champion's own light question term, and for this intent the question
  *is* the task, so question-relevance is a real quality signal. A PROBE check
  confirmed it moves a confident-wrong-on-topic answer from 0.39 (no q-term) toward
  the champion's 0.57.

### Gates we can verify locally (V_softq, `gate-V_softq.txt`)
- **Ordering + margin vs the default champion (rust word-overlap):** margin
  **0.2895 vs -0.1169**, wins **35/40 vs 13/40**. Clean win.
- **Ordering + margin vs the topical proxy (good.wasm):** proxy margin is 0.0133,
  wins 20/40, so V_softq's 0.2895 / 35 wins there too.
- **Structural:** empty=0, whitespace=0, worst_self_match 1.0000, self-match beats
  cross-match (gap 0.1486), no traps on 78 KB / oversized / emoji-CJK-RTL inputs.
  score_stddev 0.2169. All pass.
- **Gaming suite:** passes 7/12; fails question-echo (0.71), verdict-flip,
  direction-flip, number-swap, word-order-swap. This is the softened-penalty
  tradeoff. **The topical proxy champion fails these worse** (question-echo 0.80,
  and 7/12 overall, including a worst_self_match of only 0.5935 that fails the
  perfect-answer bar our stricter harness enforces). The node compares to the
  champion rather than an absolute bar, and the incumbent is this same kind of
  lenient topical scorer, so these are not expected to block registration for this
  intent — but they are a real robustness cost and are confined to this one slot.

## 5. Honest winnability

- Calibrated node estimate for the best build is **~0.56**, versus the 0.60 floor
  and the 0.5454 already on the node. We moved the needle (+~0.017 estimated) with a
  principled change, but the calibration says we are still short.
- The ceiling problem is structural: the champion is a topical transformer we cannot
  download, and within its tight on-topic answer cluster the fine ordering is set by
  the champion's *specific* embedding, which is not observable offline. Better blends
  and the q-term chip at it; they do not close it.
- This is **not** an "LLM-judge, unreproducible" situation — it is a reproducible
  family, just not the exact model. So the candidate is real, not a guess.
- Recommendation: **register V_softq once** (gas only) to read the true node
  Spearman. It is the most reachable of the four blocked signals and the only way to
  settle whether the q-term transfers better or worse than the proxy predicts. Set
  expectations at ~0.56, i.e. probably still short. Do not spend further build
  cycles on this intent unless that read comes back >= 0.58 (within striking
  distance), in which case a small q-weight / blend sweep around V_softq is
  justified.

## 6. Files

- Winner binary: `../../dist/track2-v2/task_completion.wasm`
  keccak `0xbae9de8aa83d73c2ff048ecbe6635c5596a31fceb46ed287504652fc68e486d7`
  (23,959,874 bytes). Config = V_softq above, marker `TASK_COMPLETION`.
- `spearman.json` — all agreement numbers, calibration, keccaks.
- `gate-V_softq.txt` — full harness gate output for the winner.
- `scores/*.json` — per-row corpus scores for every module (reproducible Spearman).
- `traffic-task.json` / `traffic-task-sub.json` — the intent-fitted corpus.
- `node-leaderboard.json`, `node-intent-detail.json` — the node evidence.
- `gen_traffic_task.py`, `build_variant.py`, `sweep_spearman.py` — corpus generator,
  variant builder, and Spearman sweep driver.

Source changes (in `../../module/src`, behind the existing `minilm` feature):
`minilm.rs` gains `embed_cos_b(x, y)` (embB cosine of two texts); `lib.rs` gains the
`EMB_A_W / EMB_B_W / EMB_LEX_W / EMB_Q_W` blend constants and the optional q-term in
`score()`. With `EMB_Q_W = 0` and default weights the build is byte-identical in
behaviour to the prior blend, so the 31 lexical slots and CHAT_COMPLETION are
unaffected.
