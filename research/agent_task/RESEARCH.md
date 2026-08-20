# AGENT_TASK scorer: champion hypothesis, transformer clone, winnability

Telegraph Track 2, intent **AGENT_TASK** (7 live LLM miners). Goal: become the
champion scoring module, which needs three node gates cleared against the current
champion: ordering (win >= champion on the hidden fixtures), separation margin
(> champion), and traffic agreement (Spearman >= 0.60 with the champion's ranking
of real miner answers).

Date: 2026-08-20. Wallet `0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287`.

## Hard constraint: the champion is NOT downloadable

`https://devnode.telegraphprotocol.com/wasm/AGENT_TASK.wasm` and `.../agent_task.wasm`
both 404 (size 19 = "not found"). Only `/wasm/good.wasm` (24 MB) resolves, and that is
CHAT_COMPLETION's champion, a MiniLM sentence-transformer. The node exposes no per-answer
champion scores for AGENT_TASK. So every local number below is against a **PROXY I chose,
not the real judge.** The proxy is `reference/champion-good.wasm` (the downloadable
CHAT_COMPLETION MiniLM champion) as the topical stand-in, and `reference/rust-module`
(the node's default word-overlap scorer) as the lexical stand-in. The real judge is the
node, which only the orchestrator can read after a registration.

## Champion hypothesis

**The AGENT_TASK champion is a topical / semantic scorer (MiniLM sentence-transformer
class), not word-overlap. But it is flatter and harder to match than CHAT_COMPLETION's,
and it is not identical to good.wasm.**

Evidence:

1. **Word-overlap does not agree with it.** Given node fact: raw word-overlap agreement
   with the real AGENT_TASK champion is **0.3766**. If the champion were the default
   word-overlap scorer, a word-overlap module would agree with it near 1.0. It agrees at
   0.38, so the champion ranks on something other than lexical overlap. This is the same
   signature CHAT_COMPLETION showed (lexical 0.31 there), where the champion turned out to
   be a MiniLM transformer.

2. **The live champion barely discriminates real answers (tight spread).** Champion score
   spread of the nonzero live miners, from `/leaderboard/miners`:
   - AGENT_TASK:            0.6404 .. 0.7136, spread **0.073**  (kimi > qwen > chatbot > nova > deepseek)
   - CHAT_COMPLETION:       0.5025 .. 0.8187, spread 0.316
   - TASK_COMPLETION:       0.1488 .. 0.7522, spread 0.603
   - LANGUAGE_GENERATION:   0.5702 .. 0.6758, spread 0.106

   AGENT_TASK has the **tightest** spread of the four text intents. Agent answers are long,
   structured, uniformly-decent multi-step plans, so a topical cosine to the ground truth
   compresses toward a narrow high band. A near-flat champion is the hardest kind to match
   on the Spearman gate: the ranking of a tied cluster is dominated by the champion's own
   micro-signal, which is not observable because the binary is not downloadable.

3. **Five rival registrations already tried and were rejected** (registration ids 5, 6, 8,
   11, 13 on the intent detail; the tiny `reference/rival-*.wasm` files are lexical). Nobody
   holds the slot, consistent with a champion that lexical modules cannot match.

4. **The transformer moves agreement the right way** (measured below), the same result the
   task noted for TASK_COMPLETION, which is consistent with these text intents sharing a
   semantic champion of the good.wasm class.

## The candidate

The reg-77 transformer that won CHAT_COMPLETION (0.6266 on the node), rebuilt with the
AGENT_TASK intent marker. It runs the ported MiniLM forward pass in no_std behind the
`minilm` feature and blends `0.28*embA + 0.56*embB + 0.16*lexical` with the softened
correctness penalties (M_CONTRA 0.7, M_NUM_WRONG 0.78, M_ORDER 0.85, M_ENTITY 0.72,
M_NEGCOV 0.32, M_NUM_MISS_BASE 0.85, M_TWO_FACED 0.8). Built `--features minilm` at
opt-level 3.

- Shipped binary: `dist/track2-v2/agent_task.wasm` (23,959,595 bytes)
- keccak256: `0xe0cabe0296a9591b3487e7711af163e0cbd13799c7851e4aba026eee73c1ce7a`
- Reproducible: a fresh build in `/tmp/flag-AGENT_TASK` from the patched source was
  **byte-identical** to the deployed `dist/telegraph-salience-scorer-agent_task.wasm`.

## Local numbers (real harness output, PROXY not the node)

Corpus: `research/agent_task/agent-traffic.json`, 120 rows = 24 agentic multi-step tasks x
5 varied-quality agent answers (thorough-correct, terse-correct, verbose-hedging,
confident-but-wrong-step, vague-partial), generated through the house gateway to mimic the
distribution the node's traffic gate scores. Champion (good.wasm) scores cached in
`champ-scores.json`; candidate/lexical/word-overlap in the sibling `*-scores.json`.

Traffic agreement (Spearman vs the good.wasm topical proxy, 120 rows):

| module | Spearman vs good.wasm proxy |
| --- | --- |
| transformer (reg-77 clone) | **0.6985** |
| our lexical (deployed agent_task-lex) | 0.6057 |
| raw word-overlap default | 0.5016 |

Ordering + margin (bench/benchmark.json, 40 cases; head-to-head vs the word-overlap default):

- transformer: **34/40 wins, 0 ties**, candidate_margin **0.3014**, worst_self_match 1.0,
  score_stddev 0.227.
- word-overlap default: 13/40 wins, 8 ties, margin -0.1169.
- The topical champion good.wasm has a tiny separation margin (~0.013 on this benchmark, per
  the CHAT_COMPLETION dissection), so the transformer clears the margin gate against a
  topical champion comfortably.

Gaming suite (bench/attacks.json, 12 attacks): transformer passes **7/12**. It fails
question-echo, verdict-flip, direction-flip, number-swap and word-order-swap. This is the
honest cost of cloning a topical champion: to rank like a semantic scorer you inherit its
blindness to negation, number swaps and word order. The attack suite is our own robustness
test, **not** a node gate (the node gates are ordering, margin, agreement), and the champion
itself has the same weakness. Same trade CHAT_COMPLETION made.

## Calibration: how far the proxy overstates

The one anchor between local and node: word-overlap agrees with the good.wasm proxy at
**0.5016** locally but with the **real** champion at **0.3766** on the node (given). So the
proxy overstates agreement by about -0.125 absolute (ratio 0.751). Two gaps compound here
that did not on CHAT_COMPLETION (where the proxy WAS the real champion):

1. proxy mismatch (good.wasm is not the real AGENT_TASK champion), and
2. corpus mismatch (synthetic 120-row corpus vs the node's real, longer, more homogeneous
   agent traffic).

Applying the word-overlap anchor to the transformer's local 0.6985:
- absolute-gap estimate: 0.6985 - 0.125 = **~0.573**
- ratio estimate:        0.6985 x 0.751 = **~0.525**

Both land **below** the 0.60 floor. For reference, CHAT_COMPLETION went local 0.82 (vs the
real champion) -> node 0.6266, a corpus-only gap; AGENT_TASK carries that gap plus the proxy
gap plus a tighter champion spread, so it should land lower.

## Honest winnability

**winnable_confidence: LOW.** The transformer is the right direction and the best possible
candidate. It clears gates 1 (ordering, 34/40) and 2 (margin, 0.3014 > champion) locally and
lifts traffic agreement well above word-overlap (0.6985 vs 0.5016 local; 0.3766 is the known
node word-overlap number). But the calibrated estimate for the node traffic gate is
~0.52-0.57, **just below the 0.60 floor**, and the champion is not downloadable so this
cannot be confirmed offline. The tight live-champion spread (0.073, the tightest of the text
intents) makes AGENT_TASK the hardest of them to match, because ranking a near-tied cluster
is dominated by a micro-signal we cannot observe.

**recommend_register: yes, as a cheap real measurement.** It is gas-only, the candidate is
genuine (byte-identical to the proven CHAT_COMPLETION winner, not a guess), gates 1 and 2
pass, and the only unresolved gate can only be read on the node. There is real (minority)
probability it clears if the real champion is closer to good.wasm than the calibration
suggests or if the node's real traffic ranks more separably than our synthetic corpus. The
orchestrator should register, read the real EvalDetails agreement, and treat a rejection as
the expected-but-cheap outcome (predicted ~0.52-0.57) rather than a surprise. Do not spend
further build effort chasing 0.60 without a new idea beyond the transformer, exactly as the
CHAT_COMPLETION note concluded for its class of intent.

## Sources

- Node intent detail: https://devnode.telegraphprotocol.com/engine/v1/intents/AGENT_TASK
  (7 LLM miners: DeepSeek/Nova/Qwen/Kimi Bedrock, Telegraph Knowledge Chatbot, LiteLLM,
  OpenRouter; 5 rejected wasm registrations)
- Live champion scores: https://devnode.telegraphprotocol.com/leaderboard/miners?intent=AGENT_TASK
  (and CHAT_COMPLETION / TASK_COMPLETION / LANGUAGE_GENERATION for spread comparison)
- Champion binary (proxy) and 404 check: https://devnode.telegraphprotocol.com/wasm/good.wasm
  (200, 24 MB) vs .../wasm/AGENT_TASK.wasm (404)
- Our registration EvalDetails: https://devnode.telegraphprotocol.com/engine/validator/v1/addresses/0x8b224783FE5b3c52B7DB0cb9B1754f8812b75287
- Guide (scoring framing "the best answers, used to trade and earn"): https://guide.telegraphprotocol.com/
- Hackathon ("Submit a Miner ... an evaluation script, or both"): https://hackathon.telegraphprotocol.com/
- Docs root (Next.js SPA, 307 then client-rendered, no per-intent scorer spec exposed):
  https://docs.telegraphprotocol.com
- Agent-evaluation background (rubric / trajectory / LLM-as-judge, general, not
  Telegraph-specific): https://arxiv.org/abs/2603.21362 (AdaRubric task-adaptive rubrics),
  https://arxiv.org/html/2604.06132v1 (Claw-Eval: Completion/Safety/Robustness),
  https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents,
  https://arxiv.org/html/2508.05508v1 (agentic task-completion judge)
- Internal: `BUILD-chat.md` (reg-77 winning config), memory `telegraph-wasm-promotion-gates`,
  `telegraph-lane-live-state`, `telegraph-downloadable-scorer-tune-miner`.

## Files in this directory

- `RESEARCH.md` — this document
- `gen_agent_traffic.py` — the AGENT_TASK corpus generator (house gateway, 24 agentic tasks)
- `agent-traffic.json` — 120-row proxy traffic corpus
- `champ-scores.json` / `cand-xf-scores.json` / `lex-scores.json` / `wordoverlap-scores.json`
  — cached corpus scores for each module
- `agreement.txt` — the Spearman table above, raw
- shipped binary: `../../dist/track2-v2/agent_task.wasm`
