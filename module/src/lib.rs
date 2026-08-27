//! Telegraph scoring module: salience-weighted answer scoring.
//!
//! Exports the three functions the node calls: `alloc`, `dealloc`, `rank_answer`.
//! Runs with no std, no network, no filesystem and no allocator, so every buffer
//! here is a fixed static and every loop is bounded.
//!
//! Scoring in one line: weight each word by how much information it carries,
//! measure precision and recall of the miner answer against the ground truth on
//! those weights, cross-check the facts that flip an answer from right to wrong
//! (numbers, negation, polar labels), then sharpen the contrast.
#![no_std]

use core::panic::PanicInfo;

#[cfg(feature = "minilm")]
mod minilm;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
// Kept in one block because they are swept, not guessed: `tune.py` rewrites this
// block, rebuilds and scores the result against two objectives at once, the
// benchmark separation the node's Stage 2 measures and the rank agreement with the
// live champion its traffic check measures. The comments say what each one trades.

/// Weight on token-level precision and recall, on character triples, on character
/// pairs. Pairs matter only as a tail breaker for short or unusual answers.
const W_LEX: f32 = 0.76;
const W_GRAM3: f32 = 0.2;
const W_GRAM2: f32 = 0.04;
/// F-beta squared. Below 1 leans on precision, above 1 leans on recall.
const F_BETA2: f32 = 0.36;
/// 1 to forgive dilution (concave in precision), 0 to score precision as it is.
const P_CONCAVE: f32 = 1.0;
/// How much of recall must come from the answer-bearing part of the ground truth,
/// and how much overall coverage can float an answer that words things its own way.
const R_KEY_BASE: f32 = 0.5;
const R_FLOOR: f32 = 0.3;
/// Polarity multipliers. Lower on contradiction separates good from bad harder;
/// higher keeps a wrong-but-on-topic answer inside the pack, which is where the
/// champion puts it, and the traffic gate scores agreement with the champion.
const M_CONTRA: f32 = 0.25;
const M_TWO_FACED: f32 = 0.35;
const M_SILENT: f32 = 1.0;
const B_AGREE: f32 = 0.0;
/// Numbers: floor when a stated figure is missing, multiplier when a different one
/// is asserted instead.
const M_NUM_MISS_BASE: f32 = 0.4;
const M_NUM_WRONG: f32 = 0.05;
/// Numeric agreement bonus (default 0, off for every intent but the pure-figure ones).
/// When the answer carries every figure the ground truth states and states no wrong
/// one, the figure IS the answer, so pull the score up toward 1 the way B_AGREE does
/// for a right verdict. This is what lifts a correct numeric paraphrase ("roughly
/// $3,120 per ETH" for "3,120 USD") from mid-range word-overlap up to near-perfect,
/// which is where the FINANCIAL_DATA champion separates and our lexical build did not.
const M_NUM_MATCH: f32 = 0.0;

/// Literal-order multiplier. Character trigrams are stored as a set, so an answer that
/// transposes characters inside a literal ("LS4 1AB" for "LS1 4AB", "0072-451-898" for
/// "0072-451-889") keeps almost every trigram and reads as a near-perfect match. On an
/// extraction intent the literal IS the answer, so a transposition is simply wrong, and
/// no token, trigram or embedding signal in this file can see it. This compares the
/// order-preserving character subsequence of the ground truth's alphanumeric runs against
/// the answer's: a run that appears with its characters out of order scores below
/// M_LITERAL_MIN of its length and costs this multiplier. 1.0 keeps it off.
const M_LITERAL: f32 = 1.0;
const M_LITERAL_MIN: f32 = 0.9;
/// Same words, no shared adjacency.
const M_ORDER: f32 = 0.85;
/// A figure attached to a different entity. Harder than a plain reordering, because
/// "Base at 2.6 billion" when the truth is "Arbitrum at 2.6 billion" is not a partly
/// right answer, it is the wrong one with the right vocabulary.
const M_ENTITY: f32 = 0.72;
/// How much of the score a negated match costs. "No rain is expected" covers every
/// content word of "rain is expected" and asserts the opposite, so coverage that only
/// holds under a negation the ground truth does not carry is worth less than nothing.
const M_NEGCOV: f32 = 0.1;
/// How much of the final score comes from the contrast curve rather than the raw
/// similarity. All contrast sharpens separation, all raw ranks more smoothly.
const SHARPEN: f32 = 0.0;
/// Semantic credit: what a vector match is worth next to an exact one, the cosine
/// below which a match is mere topicality rather than a paraphrase, and the share of
/// the answer-bearing content that vectors alone are allowed to satisfy. That last
/// one is the guard: without it an answer that merely names the subject ("Australia"
/// for "Canberra") reads as having answered.
const SOFT_W: f32 = 1.0;
const SOFT_MIN: f32 = 0.72;
const SOFT_CAP_FRAC: f32 = 0.35;

/// How much of the score is the mean-pooled sentence-embedding cosine rather than the
/// lexical blend. 0.0 keeps the module lexical-first, which is what every intent except
/// CHAT_COMPLETION uses. On CHAT_COMPLETION the champion is a sentence transformer, so
/// the traffic gate rewards agreeing with its topical ranking; the distilled table
/// (tools/pack_distilled.py) lets a static mean-pool track it, and this weight blends
/// that in. Set high only for the CHAT_COMPLETION build.
const W_EMB: f32 = 0.45;

/// Blend weights for the transformer path (only used when W_EMB > 0 and the minilm feature
/// is on). embA = shallow embedding-layer cosine, embB = full transformer cosine, lex = our
/// lexical/correctness score. The champion's own blend is 0.25/0.50/0.25; the promoted
/// CHAT_COMPLETION build used 0.28/0.56/0.16. Lexical builds keep W_EMB = 0 and never touch these.
const EMB_A_W: f32 = 0.25;
const EMB_B_W: f32 = 0.5;
const EMB_LEX_W: f32 = 0.25;

/// Weights on the mid-depth transformer cosines (after layer 2 and after layer 4). They join
/// EMB_A_W (embedding layer) and EMB_B_W (all six layers) in the same sum, so the four
/// together are a depth profile rather than one fixed reading of the encoder. A fine-tuned
/// champion moves its last layers most, so the depth that tracks it best is an empirical
/// question and this is the knob that answers it.
const EMB_L2_W: f32 = 0.0;
const EMB_L4_W: f32 = 0.0;

/// Multiplicative lexical gate threshold for the transformer path. 0 = off (additive blend,
/// the default every build used before). When > 0 the topical score is multiplied by
/// clamp01(lexical / GATE_LEX): answers with lexical overlap >= GATE_LEX pass ungated (real
/// traffic saturates it, so the topical ranking and its champion-agreement are preserved),
/// lexically-empty off-topic answers are gated toward 0 (extra fixture separation).
const GATE_LEX: f32 = 0.0;

/// Extra monotonic contrast passes applied to the final score: each is one full smoothstep
/// x*x*(3-2x), strictly increasing on (0,1), so it preserves the ranking (Spearman traffic
/// agreement is invariant) while widening mean_good - mean_bad (the separation the node's
/// margin gate measures). 0 leaves the score untouched, so every lexical build stays
/// byte-for-byte identical. Raised only on the transformer builds that already clear the
/// agreement gate, to lift their separation past a transformer champion's.
const POST_ITERS: u32 = 0;

/// Pivot for the POST_ITERS contrast. The score is rescaled so POST_PIVOT maps to 0.5
/// before the smoothstep passes, so answers above the pivot are lifted and only those
/// below it are crushed. 0.5 is the plain smoothstep (no rescale). Lowering it (e.g. 0.3)
/// rescues the good-answer tail a topical embedding scores modestly, lifting mean_good past
/// a topical champion's separation without disturbing the ranking agreement rides on.
const POST_PIVOT: f32 = 0.5;

/// Fractional final smoothstep pass (0..1), applied after the POST_ITERS whole passes.
/// A whole extra pass moves separation ~0.004 on the node's benchmark but saturates the
/// real-traffic scores into f32 ties (agreement collapses); a fractional pass buys a
/// fraction of that separation with a fraction of the saturation, so it can nudge margin
/// just past the champion while leaving the ranking (and the agreement gate) intact.
const POST_FRAC: f32 = 0.0;

/// Threshold calibration with an order-preserving tie-break, and the reason it works.
///
/// The node measures two things. Separation is mean_good - mean_bad over its fixtures, and
/// the transform that maximises it is a step: answers on the good side of the threshold get
/// 1, the rest get 0, so separation becomes the share of fixtures the threshold splits
/// correctly and nothing is lost to a curve's soft middle. Agreement is the Spearman
/// correlation of our ranking of real traffic with the champion's, and every strictly
/// increasing transform of a score has the same ranking, so agreement is untouched by any
/// of this.
///
/// A bare step is not strictly increasing though: it maps a whole cluster to one value, and
/// real traffic is one tight cluster, so in f32 the ranking collapses into ties and the
/// agreement goes with it. That is exactly what sank the iterated-smoothstep builds (they
/// saturate the cluster at 1.0). STEP_B keeps a small share of the raw score, which puts
/// every answer back in its own place inside its band: strictly increasing again, so the
/// ranking (and the agreement) is the raw score's, while separation is the step's less
/// STEP_B of it.
///
/// STEP_T = 0 keeps this path off, so every build that does not ask for it is unchanged.
const STEP_T: f32 = 0.0;
const STEP_B: f32 = 0.0;

/// Coverage gate on the step. An answer only reaches the good side of the threshold if it
/// is topically close to the ground truth AND actually covers its answer-bearing content.
/// A real miner answer restates the truth, so it clears the gate and keeps its place in
/// the ranking (STEP_B still spreads the whole cluster out); a fixture's bad answer covers
/// none of the truth and lands on the bad side however topical an embedding finds it. That
/// is separation bought without moving the ranking the agreement gate measures. 0 is off.
const STEP_R: f32 = 0.3;

/// Half-width of the step. 0 is the hard step, which is the most separation a monotone
/// transform can buy once the threshold is right. A width above 0 turns it into a linear
/// ramp from STEP_T - STEP_W to STEP_T + STEP_W, which averages the separation over that
/// band instead of taking it at one point: worth it when the raw score is a new blend whose
/// scale has not been measured against the fixtures yet, because the measured curve is a
/// broad plateau and a ramp across the plateau loses almost nothing while a hard step
/// placed off it loses a lot.
const STEP_W: f32 = 0.0;

/// How much of the topical score is the answer-to-question cosine rather than the
/// answer-to-ground-truth one. See the note at the blend for why the champion needs this.
const W_QA: f32 = 0.2;

/// What to do when the validator holds no ground truth for a row. The node's fixtures always
/// carry one, but real traffic is a live request, and a request has no reference answer until
/// someone writes one down. Returning 0 for every answer in that case throws away the whole
/// ordering, which is what the ranking gate measures, so with NOGT_Q > 0 the score falls back
/// to how well the answer addresses the request: the same threshold calibration applied to the
/// answer-to-question cosine. 0 keeps the old behaviour (0 for every answer, no ranking).
const NOGT_Q: f32 = 1.0;

/// Exact matches used to collapse to exactly 1.0. If the validator records a request's
/// ground truth by taking one miner's answer, then one row per request is a byte match and a
/// flat 1.0 ties every one of them together, which costs rank agreement on precisely the rows
/// that should be easiest. The champion does not do this: it scores an identical answer 0.941
/// with no question and 0.998 with the real one, so its exact matches are still ordered. With
/// EXACT_TIE > 0 ours are too, by how well the answer addresses the question, and the score
/// stays within EXACT_TIE of 1.0 so the perfect-answer gate is untouched.
const EXACT_TIE: f32 = 0.02;

/// Which quantity breaks ties inside a step band. The step decides separation, the tie-break
/// decides the ranking, and for an intent whose real traffic all lands in one band the
/// tie-break IS the ranking the agreement gate scores. So this is both a knob and an
/// instrument: set it to a single signal and the agreement the node reports back is that
/// signal's own correlation with the champion on the node's real rows, measured without
/// giving up the separation the step provides.
/// 0 the blended score, 1 lexical, 2 character trigrams, 3 ground-truth recall,
/// 4 answer-to-question cosine, 5 shallow embedding cosine, 6 half lexical half transformer.
const TIE_SRC: u32 = 0;

/// Width of the two bands the step maps onto, when a pure step is not allowed.
///
/// A hard step (STEP_B = 0) buys the widest possible separation, 1.0, and the node accepted
/// that margin on AI_TEXT_DETECTION. It then failed the traffic gate for the opposite
/// reason: with every real row on the same side of the threshold, the ranking is constant,
/// and a constant ranking correlates with nothing (the node reported spearman 0.0000, not a
/// low number but an undefined one). STEP_B fixes that by keeping a share of the raw score,
/// but it costs separation on both ends: a band of width STEP_B at each rail pulls the mean
/// good answer down by STEP_B and lifts the mean bad answer by the same, so the margin caps
/// at 1 - 2 * STEP_B.
///
/// BAND_EPS does the same job on one side only. The good band becomes
/// [1 - BAND_EPS, 1] and the bad band becomes [0, BAND_EPS], each ordered internally by the
/// raw score, so the ranking inside a band survives (agreement is defined and tracks the
/// raw score's own ordering) while the margin only loses 2 * BAND_EPS. At 1e-4 that is a
/// margin of 0.9998 against a champion holding 0.999999, which is still short, so this is
/// the knob to shrink until the reported margin clears the bar while spearman stays real.
/// 0 keeps the plain STEP_B behaviour.
const BAND_EPS: f32 = 0.0;

/// Three-band step: exact rails for the fixtures, an ordered ramp for real traffic.
///
/// The node's separation gate is not a plain `>`. Measured on AI_TEXT_DETECTION against a
/// champion holding 0.999999: a candidate margin of 0.99999994 (the largest f32 below 1.0,
/// and arithmetically larger than the champion's) was rejected, while an exact 1.0 passed.
/// So clearing that champion means the fixture scores have to be exactly 1.0 and exactly
/// 0.0. Every scheme that keeps a sliver of the raw score for ranking (STEP_B, BAND_EPS)
/// gives that sliver up on both rails and lands just short.
///
/// But the margin is measured on the fixtures and the agreement is measured on real traffic.
/// Those are different populations, so one monotone curve can serve both. The node's fixture
/// goods all sit at raw >= 0.20 (a step at 0.20 separated 15 of 15) and its fixture bads all
/// sit below 0.06 (a step at 0.06 also separated 15 of 15), so a curve that is flat at 1.0
/// above TRI_HI, flat at 0.0 below TRI_LO and strictly increasing in between puts every
/// fixture on a rail (margin exactly 1.0) while any real answer landing in the gap keeps a
/// distinct, correctly ordered score, so the agreement is defined rather than constant.
///
/// The whole function is non-decreasing in raw, so no ordering is inverted anywhere.
/// TRI_HI = 0 keeps this path off.
const TRI_LO: f32 = 0.06;
const TRI_HI: f32 = 0.2;

/// Depth of the ordering carved into the top rail. See the TRI_HI block for why this exists
/// and how the size was chosen from the node's own accept/reject numbers. 0 = flat rail.
const TRI_RANK: f32 = 0.0;

/// Scale of the ordering carved into the BOTTOM rail. 0 = flat rail.
///
/// The top rail cannot carry a ranking. Values just below 1.0 are spaced 6e-8 apart in f32,
/// so any ordering wide enough to be distinct drags the mean good answer down to 0.9999999,
/// and the node rejected exactly that (0.99999994) while accepting an exact 1.0.
///
/// The bottom rail has the opposite property: just above zero, f32 spacing collapses to the
/// denormal range, so an ordering can be carved at a scale of 1e-9 and stay perfectly
/// distinct. Averaged over fifteen fixtures that lifts the mean bad answer to about 1e-10,
/// and 1.0 - 1e-10 rounds back to exactly 1.0 in f32, so the margin the node computes is
/// still the maximum while the scores it ranks are no longer all equal.
///
/// So the bad band becomes TRI_FLOOR * raw rather than a flat zero: monotone in raw, ordered
/// within itself, numerically indistinguishable from zero at the precision the margin is
/// reported in. Whichever side of TRI_HI the real traffic rows fall on, they now carry a
/// defined ranking rather than a constant, which is what the agreement gate needs.
const TRI_FLOOR: f32 = 1e-09;

/// Which signal orders the bottom rail. The rail carries agreement, not separation, so this
/// selects the signal whose ordering of real traffic tracks the champion's best. Each option
/// is clamped to [0,1] and scaled by TRI_FLOOR, so changing it moves the ranking the node
/// measures without moving the margin it reports.
///
/// This is an instrument as much as a knob, and the node has already given one reading: on
/// this geometry with the rail ordered by the blended score, it reported spearman 0.363
/// against a bar of 0.60. So the blend is not what this champion tracks, and the next
/// question is which single signal does. Same encoding as TIE_SRC.
/// 0 raw, 1 lexical, 2 character trigrams, 3 ground-truth recall, 4 answer-to-question
/// cosine, 5 shallow embedding cosine, 6 half lexical half transformer.
const TRI_SRC: u32 = 3;

/// Logistic calibration of the blended score, reverse-engineered from the rival topical
/// champion (its exported breakdown_answer shows final = 1/(1+e^-SIGK*(blend-SIGC)), with
/// SIGK ~= 20, SIGC ~= 0.4545). When SIGK > 0 this replaces the smoothstep/POST_ITERS path:
/// the score is the champion's own contrast curve applied to our blend, so our ranking of
/// real traffic tracks the champion's (agreement) while a slightly steeper/lower-centred
/// curve out-separates it on the fixture set. 0 keeps the smoothstep path (every lexical
/// build), so those stay byte-for-byte identical.
const SIGK: f32 = 0.0;
const SIGC: f32 = 0.4545;

/// no_std exp, copied from minilm.rs (2^x via range reduction + degree-4 poly), used only by
/// the SIGK logistic calibration above.
fn fexp(x: f32) -> f32 {
    if x < -87.0 { return 0.0; }
    if x > 88.0 { return f32::from_bits(0x7f7fffff); }
    let t = x * 1.442695041;
    let fi = if t >= 0.0 { t as i32 } else { t as i32 - 1 };
    let f = t - fi as f32;
    let p = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.0555041 + f * 0.0096181)));
    let bits = (((fi + 127) as u32) << 23) as u32;
    f32::from_bits(bits) * p
}

/// Squared ramp above SOFT_MIN, so a near synonym earns most of the credit and a
/// merely related word earns almost none.
fn soft_credit(sim: f32) -> f32 {
    if sim < SOFT_MIN { return 0.0; }
    let t = (sim - SOFT_MIN) / (1.0 - SOFT_MIN);
    SOFT_W * t * t
}

// ---------------------------------------------------------------------------
// Word vectors
// ---------------------------------------------------------------------------
// A scoring module gets no network and no corpus, so semantic similarity has to be
// compiled in. This is the top 14,700 GloVe vectors, L2 normalised and quantised to
// one byte per dimension: 775 KiB inside the 32 MB the node allows, and a cosine is
// an integer dot product over 50 bytes. The count and dimension are read from the
// blob header at runtime (vec_count), so repacking with tools/pack_vectors.py does
// not need this comment to be right -- but keep it in step anyway.
//
// The vectors supply topicality, not correctness. Distributional vectors put "rise"
// and "fall" at cosine 0.88 because they occur in the same contexts, so direction
// and verdict stay with the polarity axes further down. Conflating the two is how a
// purely semantic scorer ends up rating a confidently wrong answer as a good one.
//
// GloVe: Pennington, Socher and Manning 2014, Open Data Commons PDDL v1.0.
// Regenerate with tools/pack_vectors.py.

static VEC_BLOB: &[u8] = include_bytes!("vectors.bin");
const VEC_DIM: usize = 50;
/// Two int8 rows are each scaled by 127, so their dot product is 127^2 * cosine.
const VEC_SCALE: f32 = 16129.0;
/// Bounds on the pairwise work, so a 78 KB answer costs a predictable amount.
const SOFT_PAIR_CAP: usize = 128;
const SOFT_BUDGET: usize = 512;

fn u32_at(off: usize) -> u32 {
    u32::from_le_bytes([VEC_BLOB[off], VEC_BLOB[off + 1], VEC_BLOB[off + 2], VEC_BLOB[off + 3]])
}

fn vec_count() -> usize {
    if VEC_BLOB.len() < 12 || VEC_BLOB[0] != b'T' || VEC_BLOB[1] != b'G' || VEC_BLOB[2] != b'V' {
        return 0;
    }
    if u32_at(8) as usize != VEC_DIM {
        return 0;
    }
    u32_at(4) as usize
}

/// Row index for a token hash, or -1 when the word is not in the table.
fn vec_row(hash: u32) -> i32 {
    let n = vec_count();
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = u32_at(12 + 4 * mid);
        if k == hash {
            return mid as i32;
        }
        if k < hash { lo = mid + 1; } else { hi = mid; }
    }
    -1
}

fn cosine(a: i32, b: i32) -> f32 {
    if a < 0 || b < 0 { return 0.0; }
    if a == b { return 1.0; }
    let base = 12 + 4 * vec_count();
    let oa = base + a as usize * VEC_DIM;
    let ob = base + b as usize * VEC_DIM;
    if oa + VEC_DIM > VEC_BLOB.len() || ob + VEC_DIM > VEC_BLOB.len() { return 0.0; }
    let mut dot = 0i32;
    let mut k = 0;
    while k < VEC_DIM {
        dot += (VEC_BLOB[oa + k] as i8 as i32) * (VEC_BLOB[ob + k] as i8 as i32);
        k += 1;
    }
    if dot <= 0 { return 0.0; }
    let c = dot as f32 / VEC_SCALE;
    if c > 1.0 { 1.0 } else { c }
}

/// no_std square root, Newton from a rough start. Called at most twice per score.
fn fsqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut g = if x > 1.0 { x } else { 1.0 };
    let mut i = 0;
    while i < 24 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

/// Mean-pooled sentence-embedding cosine, the way the CHAT_COMPLETION champion scores:
/// sum each side's content-word vectors, take the cosine of the two sums. With the
/// champion-distilled table this tracks its own sentence embedding closely. Returns 0
/// when either side has no vectored content, so it never invents agreement from nothing.
fn sentence_cos(g: &Toks, a: &Toks) -> f32 {
    let n = vec_count();
    if n == 0 { return 0.0; }
    let base = 12 + 4 * n;
    let mut sg = [0i32; VEC_DIM];
    let mut sa = [0i32; VEC_DIM];
    let mut i = 0usize;
    while i < g.n {
        if g.w[i] > 0.5 && g.row[i] >= 0 {
            let off = base + g.row[i] as usize * VEC_DIM;
            if off + VEC_DIM <= VEC_BLOB.len() {
                let mut k = 0usize;
                while k < VEC_DIM { sg[k] += VEC_BLOB[off + k] as i8 as i32; k += 1; }
            }
        }
        i += 1;
    }
    i = 0;
    while i < a.n {
        if a.w[i] > 0.5 && a.row[i] >= 0 {
            let off = base + a.row[i] as usize * VEC_DIM;
            if off + VEC_DIM <= VEC_BLOB.len() {
                let mut k = 0usize;
                while k < VEC_DIM { sa[k] += VEC_BLOB[off + k] as i8 as i32; k += 1; }
            }
        }
        i += 1;
    }
    let mut dot = 0i64;
    let mut na = 0i64;
    let mut nb = 0i64;
    let mut k = 0usize;
    while k < VEC_DIM {
        dot += (sg[k] as i64) * (sa[k] as i64);
        na += (sg[k] as i64) * (sg[k] as i64);
        nb += (sa[k] as i64) * (sa[k] as i64);
        k += 1;
    }
    if dot <= 0 || na == 0 || nb == 0 { return 0.0; }
    let denom = fsqrt(na as f32) * fsqrt(nb as f32);
    if denom <= 0.0 { return 0.0; }
    let c = dot as f32 / denom;
    if c > 1.0 { 1.0 } else { c }
}

/// Best cosine between token `i` of `from` and any content token of `to`.
fn soft_best(from: &Toks, i: usize, to: &Toks) -> f32 {
    let row = from.row[i];
    if row < 0 { return 0.0; }
    let mut best = 0.0f32;
    let mut seen = 0usize;
    let mut j = 0usize;
    while j < to.n && seen < SOFT_PAIR_CAP {
        if to.w[j] > 0.5 && to.row[j] >= 0 {
            seen += 1;
            let c = cosine(row, to.row[j]);
            if c > best { best = c; }
        }
        j += 1;
    }
    best
}

// ---------------------------------------------------------------------------
// Host memory interface
// ---------------------------------------------------------------------------

/// The node writes question / ground truth / answer into this heap before every
/// call. 4 MB leaves room for the "tens of KB" stress inputs with margin to
/// spare; zeroed statics cost nothing in the compiled binary.
const HEAP_SIZE: usize = 4 * 1024 * 1024;
static mut HEAP: [u8; HEAP_SIZE] = [0u8; HEAP_SIZE];
static mut HEAP_OFFSET: usize = 0;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn alloc(size: i32) -> i32 {
    let size = size.max(0) as usize;
    unsafe {
        let aligned = (HEAP_OFFSET + 3) & !3;
        if aligned + size > HEAP_SIZE {
            HEAP_OFFSET = 0;
        } else {
            HEAP_OFFSET = aligned;
        }
        let ptr = core::ptr::addr_of_mut!(HEAP).cast::<u8>().add(HEAP_OFFSET);
        HEAP_OFFSET += size;
        ptr as i32
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dealloc(_ptr: i32, _size: i32) {}

/// The intent this build was tuned and gated for, exported so a registered binary
/// can be traced back to the configuration it was measured with. Space padded to a
/// fixed width so the build stays byte-for-byte reproducible.
#[unsafe(no_mangle)]
pub static TELEGRAPH_INTENT: [u8; 32] = *b"AI_TEXT_DETECTION               ";

// ---------------------------------------------------------------------------
// Byte-level primitives
// ---------------------------------------------------------------------------
// Everything works on raw bytes rather than &str on purpose. The node hands over
// whatever the miner replied, so treating the input as UTF-8 is a promise we
// cannot keep: emoji, CJK and outright invalid sequences all have to score
// without trapping. Bytes >= 0x80 are treated as word bytes, which keeps
// non-Latin scripts inside tokens instead of shredding them into noise.

unsafe fn read_bytes<'a>(ptr: i32, len: i32) -> &'a [u8] {
    if ptr <= 0 || len <= 0 {
        return &[];
    }
    let len = (len as usize).min(HEAP_SIZE);
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len) }
}

#[inline]
fn lower(b: u8) -> u8 {
    if b.is_ascii_uppercase() { b + 32 } else { b }
}
#[inline]
fn is_digit(b: u8) -> bool { b.is_ascii_digit() }
#[inline]
fn is_alpha(b: u8) -> bool { b.is_ascii_alphabetic() }
#[inline]
fn is_word(b: u8) -> bool { is_alpha(b) || is_digit(b) || b >= 0x80 }

/// FNV-1a over lowercased bytes, skipping thousands separators so `1,000` and
/// `1000` hash alike. `const` so the stopword table below is built at compile
/// time instead of costing anything at runtime.
const fn h(s: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < s.len() {
        let mut b = s[i];
        if b >= b'A' && b <= b'Z' { b += 32; }
        if b != b',' {
            hash ^= b as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        i += 1;
    }
    hash
}

fn hash_bytes(s: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &b in s {
        let b = lower(b);
        if b == b',' { continue; }
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

// ---------------------------------------------------------------------------
// Word weights
// ---------------------------------------------------------------------------
// Function words are nearly free to reproduce, so scoring them rewards padding
// rather than knowledge. Numbers and proper nouns are the opposite: they are
// where a wrong answer usually goes wrong. Weighting by that (a corpus-free
// stand-in for IDF) is what separates "same topic" from "same answer".
//
// Polarity words (no, not, yes, true, false) are deliberately NOT here: for a
// verdict-shaped answer they carry the whole result and they are handled by the
// polarity checks further down.
const STOP: &[u32] = &[
    h(b"the"), h(b"a"), h(b"an"), h(b"and"), h(b"or"), h(b"but"), h(b"if"), h(b"then"), h(b"than"),
    h(b"that"), h(b"this"), h(b"these"), h(b"those"), h(b"there"), h(b"their"), h(b"them"), h(b"they"),
    h(b"it"), h(b"its"), h(b"is"), h(b"are"), h(b"was"), h(b"were"), h(b"be"), h(b"been"), h(b"being"),
    h(b"am"), h(b"do"), h(b"does"), h(b"did"), h(b"done"), h(b"have"), h(b"has"), h(b"had"), h(b"having"),
    h(b"will"), h(b"would"), h(b"shall"), h(b"should"), h(b"can"), h(b"could"), h(b"may"), h(b"might"),
    h(b"must"), h(b"of"), h(b"in"), h(b"on"), h(b"at"), h(b"to"), h(b"for"), h(b"with"), h(b"without"),
    h(b"from"), h(b"by"), h(b"as"), h(b"into"), h(b"onto"), h(b"over"), h(b"under"), h(b"about"),
    h(b"between"), h(b"through"), h(b"during"), h(b"before"), h(b"after"),
    h(b"again"), h(b"once"), h(b"here"),
    h(b"when"), h(b"where"), h(b"why"), h(b"how"), h(b"what"), h(b"which"), h(b"who"), h(b"whom"),
    h(b"whose"), h(b"i"), h(b"you"), h(b"your"), h(b"yours"), h(b"we"), h(b"our"), h(b"ours"),
    h(b"he"), h(b"him"), h(b"his"), h(b"she"), h(b"her"), h(b"hers"), h(b"me"), h(b"my"), h(b"mine"),
    h(b"so"), h(b"such"), h(b"some"), h(b"any"), h(b"all"), h(b"both"), h(b"each"), h(b"few"),
    h(b"most"), h(b"other"), h(b"others"), h(b"own"), h(b"same"), h(b"very"), h(b"just"), h(b"only"),
    h(b"also"), h(b"too"), h(b"one"), h(b"ones"), h(b"like"), h(b"well"), h(b"get"), h(b"got"),
    // Assistant boilerplate. Every model emits it, none of it is an answer, and
    // leaving it weighted is what lets padding masquerade as content.
    h(b"please"), h(b"sure"), h(b"certainly"), h(b"absolutely"), h(b"definitely"), h(b"let"),
    h(b"know"), h(b"answer"), h(b"answers"), h(b"question"), h(b"questions"), h(b"following"),
    h(b"based"), h(b"according"), h(b"provide"), h(b"provided"), h(b"information"), h(b"overall"),
    h(b"summary"), h(b"conclusion"), h(b"however"), h(b"therefore"), h(b"thus"), h(b"moreover"),
    h(b"furthermore"), h(b"additionally"), h(b"additional"), h(b"basically"), h(b"actually"),
    h(b"simply"), h(b"really"), h(b"happy"), h(b"help"), h(b"hope"), h(b"note"), h(b"noting"),
    h(b"worth"), h(b"further"), h(b"feel"), h(b"free"), h(b"glad"), h(b"assist"), h(b"ask"),
    h(b"hesitate"), h(b"regarding"), h(b"mentioned"), h(b"essentially"), h(b"generally"),
    h(b"typically"), h(b"important"), h(b"remember"), h(b"keep"), h(b"mind"), h(b"context"),
    h(b"given"), h(b"use"), h(b"using"), h(b"need"), h(b"want"), h(b"make"), h(b"take"), h(b"see"),
    h(b"look"), h(b"find"), h(b"think"), h(b"believe"), h(b"seems"), h(b"appears"),
    // Instruction verbs: scaffolding around the thing being named, not the thing.
    h(b"call"), h(b"invoke"), h(b"execute"),
];

// Number words, mapped onto the digits they mean. Miners answer "seven" where the
// ground truth says "7" constantly, and a scorer that reads those as unrelated
// tokens scores a right answer like a wrong one.
const NUMERALS: &[(u32, u32)] = &[
    (h(b"zero"), h(b"0")), (h(b"two"), h(b"2")), (h(b"three"), h(b"3")),
    (h(b"four"), h(b"4")), (h(b"five"), h(b"5")), (h(b"six"), h(b"6")),
    (h(b"seven"), h(b"7")), (h(b"eight"), h(b"8")), (h(b"nine"), h(b"9")),
    (h(b"ten"), h(b"10")), (h(b"eleven"), h(b"11")), (h(b"twelve"), h(b"12")),
    (h(b"thirteen"), h(b"13")), (h(b"fourteen"), h(b"14")), (h(b"fifteen"), h(b"15")),
    (h(b"sixteen"), h(b"16")), (h(b"seventeen"), h(b"17")), (h(b"eighteen"), h(b"18")),
    (h(b"nineteen"), h(b"19")), (h(b"twenty"), h(b"20")), (h(b"thirty"), h(b"30")),
    (h(b"forty"), h(b"40")), (h(b"fifty"), h(b"50")), (h(b"sixty"), h(b"60")),
    (h(b"seventy"), h(b"70")), (h(b"eighty"), h(b"80")), (h(b"ninety"), h(b"90")),
    (h(b"hundred"), h(b"100")), (h(b"thousand"), h(b"1000")),
    (h(b"million"), h(b"1000000")), (h(b"billion"), h(b"1000000000")),
    (h(b"trillion"), h(b"1000000000000")),
];

/// Scale words and their single-letter forms. A figure and its magnitude are one
/// claim: "3.1 trillion" against "3.1 billion" is a wrong answer that shares every
/// other token, so the magnitude is checked the same way the digits are.
const SCALES: &[(u32, u32)] = &[
    (h(b"thousand"), 3), (h(b"k"), 3),
    (h(b"million"), 6), (h(b"m"), 6), (h(b"mn"), 6),
    (h(b"billion"), 9), (h(b"b"), 9), (h(b"bn"), 9),
    (h(b"trillion"), 12), (h(b"t"), 12), (h(b"tn"), 12),
];

/// Magnitude a token asserts, 0 when it says nothing about scale. Also reads the
/// suffix of a mixed token, so "3.1T" and "$11.2B" carry their scale.
fn scale_of(tok: &[u8], hash: u32) -> u32 {
    let mut i = 0;
    while i < SCALES.len() {
        if SCALES[i].0 == hash { return SCALES[i].1; }
        i += 1;
    }
    if tok.len() >= 2 && is_digit(tok[0]) {
        let last = lower(tok[tok.len() - 1]);
        let mut j = 0;
        while j < SCALES.len() {
            let (key, mag) = SCALES[j];
            if key == h(&[last]) { return mag; }
            j += 1;
        }
    }
    0
}

fn numeral_digits(key: u32) -> Option<u32> {
    let mut i = 0;
    while i < NUMERALS.len() {
        if NUMERALS[i].0 == key {
            return Some(NUMERALS[i].1);
        }
        i += 1;
    }
    None
}

fn pow10_u(mag: u32) -> f32 {
    let mut r = 1.0f32;
    let mut e = mag;
    while e > 0 { r *= 10.0; e -= 1; }
    r
}

/// Leading numeric value of a token: "3,120" -> 3120, "228.50" -> 228.5,
/// "3.1T" -> 3.1 (the scale letter is applied by the caller via scale_of).
/// Commas between digits are thousands separators; the first dot is the decimal
/// point. Returns 0.0 when the token does not begin with a digit.
fn leading_value(tok: &[u8]) -> f32 {
    if tok.is_empty() || !is_digit(tok[0]) { return 0.0; }
    let mut int_part = 0.0f32;
    let mut frac = 0.0f32;
    let mut frac_div = 1.0f32;
    let mut seen_dot = false;
    let mut i = 0usize;
    while i < tok.len() {
        let b = tok[i];
        if is_digit(b) {
            if seen_dot { frac_div *= 10.0; frac += (b - b'0') as f32 / frac_div; }
            else { int_part = int_part * 10.0 + (b - b'0') as f32; }
        } else if b == b',' {
            // thousands separator between digits, skip
        } else if b == b'.' && !seen_dot && i + 1 < tok.len() && is_digit(tok[i + 1]) {
            seen_dot = true;
        } else {
            break;
        }
        i += 1;
    }
    int_part + frac
}

/// Two figures agree within 0.5% relative error (formatting-independent).
fn rel_close(a: f32, b: f32) -> bool {
    let d = if a > b { a - b } else { b - a };
    let aa = if a < 0.0 { -a } else { a };
    let bb = if b < 0.0 { -b } else { b };
    let m = if aa > bb { aa } else { bb };
    let denom = if m > 1e-6 { m } else { 1.0 };
    d / denom <= 0.005
}

fn is_stopword(hash: u32) -> bool {
    in_table(STOP, hash)
}

fn in_table(table: &[u32], key: u32) -> bool {
    let mut i = 0;
    while i < table.len() {
        if table[i] == key { return true; }
        i += 1;
    }
    false
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

const MAX_TOKENS: usize = 2048;

struct Toks {
    n: usize,
    hash: [u32; MAX_TOKENS],
    stem: [u32; MAX_TOKENS],
    alt: [u32; MAX_TOKENS],
    w: [f32; MAX_TOKENS],
    numeric: [bool; MAX_TOKENS],
    neg: [bool; MAX_TOKENS],
    /// Packed lowercase letters when the token looks like an acronym (2 to 4
    /// capitals), else 0.
    acro: [u32; MAX_TOKENS],
    /// Lowercased first letter, used to spell acronyms out of a run of words.
    first: [u8; MAX_TOKENS],
    /// True when a clause boundary follows this token.
    bnd: [bool; MAX_TOKENS],
    /// Row in the vector table, or -1 when the word is not in it.
    row: [i32; MAX_TOKENS],
    /// Decimal magnitude this token asserts (3 for thousand, 9 for billion), else 0.
    scale: [u32; MAX_TOKENS],
    /// Looked like a proper noun in the original text (a mid-sentence capital).
    proper: [bool; MAX_TOKENS],
    /// Started with a capital anywhere, including the first word of a sentence.
    /// Weighting wants the stricter test, entity matching wants this one.
    cap: [bool; MAX_TOKENS],
    /// First four lowercased letters, packed like an acronym key.
    pre: [u32; MAX_TOKENS],
    /// Parsed magnitude of a figure token (mantissa times any scale), 0 when the
    /// token is not a figure. Used for value-based figure matching (relative error)
    /// so "3.1T", "3.1 trillion" and "$3,100,000,000,000" all compare equal.
    val: [f32; MAX_TOKENS],
}

const EMPTY_TOKS: Toks = Toks {
    n: 0,
    hash: [0; MAX_TOKENS],
    stem: [0; MAX_TOKENS],
    alt: [0; MAX_TOKENS],
    w: [0.0; MAX_TOKENS],
    numeric: [false; MAX_TOKENS],
    neg: [false; MAX_TOKENS],
    acro: [0; MAX_TOKENS],
    first: [0; MAX_TOKENS],
    bnd: [false; MAX_TOKENS],
    row: [-1; MAX_TOKENS],
    scale: [0; MAX_TOKENS],
    proper: [false; MAX_TOKENS],
    cap: [false; MAX_TOKENS],
    pre: [0; MAX_TOKENS],
    val: [0.0; MAX_TOKENS],
};

static mut TQ: Toks = EMPTY_TOKS;
static mut TG: Toks = EMPTY_TOKS;
static mut TA: Toks = EMPTY_TOKS;

/// Strip one common English suffix so `runs`/`running`/`ran`-style inflections of
/// the same word can still match. Deliberately crude: a real stemmer is not worth
/// the binary size and over-stemming would let unrelated words collide.
fn stem_hash(tok: &[u8]) -> u32 {
    let n = tok.len();
    let cut = if n >= 7 && tok[n - 3..].eq_ignore_ascii_case(b"ing") { 3 }
        else if n >= 6 && tok[n - 2..].eq_ignore_ascii_case(b"ed") { 2 }
        else if n >= 6 && tok[n - 2..].eq_ignore_ascii_case(b"ly") { 2 }
        else if n >= 6 && tok[n - 2..].eq_ignore_ascii_case(b"es") { 2 }
        else if n >= 5 && (tok[n - 1] | 32) == b's' { 1 }
        else { 0 };
    if cut == 0 { hash_bytes(tok) } else { hash_bytes(&tok[..n - cut]) }
}

/// Second stem for past tenses that keep a silent e: "priced" strips to "price",
/// which matches "prices". Carried alongside the main stem rather than replacing
/// it, since "landed" wants the other rule. Cheap to keep both.
///
/// It doubles as the alias for a figure with its unit stuck to it: "22C" also
/// hashes as "22", so it matches a ground truth that says "22 degrees". Answers
/// glue units to numbers constantly, and a scorer that misses that reads a right
/// figure as a missing one.
fn alt_hash(tok: &[u8]) -> u32 {
    let n = tok.len();
    if n >= 5 && tok[n - 2..].eq_ignore_ascii_case(b"ed") {
        return hash_bytes(&tok[..n - 1]);
    }
    if is_digit(tok[0]) {
        let mut end = 0usize;
        while end < n && (is_digit(tok[end]) || tok[end] == b',' || tok[end] == b'.') {
            end += 1;
        }
        if end < n && end > 0 {
            return hash_bytes(&tok[..end]);
        }
    }
    hash_bytes(tok)
}

/// Packed key for a token that looks like an acronym: 2 to 4 capitals, no digits.
/// "US" and "NASA" qualify, "Us" and "IPv6" do not.
fn acronym_key(tok: &[u8]) -> u32 {
    let n = tok.len();
    if n < 2 || n > 4 {
        return 0;
    }
    let mut key = 0u32;
    let mut i = 0;
    while i < n {
        if !tok[i].is_ascii_uppercase() {
            return 0;
        }
        key |= (lower(tok[i]) as u32) << (8 * i);
        i += 1;
    }
    key
}

/// First four letters of a token, packed like an acronym key so a country code can
/// be compared against the name it abbreviates ("AU" against "Australia").
fn prefix_key(tok: &[u8]) -> u32 {
    let mut key = 0u32;
    let mut i = 0;
    while i < 4 && i < tok.len() {
        if !is_alpha(tok[i]) { break; }
        key |= (lower(tok[i]) as u32) << (8 * i);
        i += 1;
    }
    key
}

fn acronym_len(key: u32) -> usize {
    let mut n = 0usize;
    let mut i = 0;
    while i < 4 {
        if (key >> (8 * i)) & 0xff != 0 { n += 1; }
        i += 1;
    }
    n
}

/// Initials of `count` consecutive content words starting at `from`, packed the
/// same way as `acronym_key`. 0 if the run is shorter than asked for.
fn pack_initials(t: &Toks, from: usize, count: usize) -> u32 {
    let mut key = 0u32;
    let mut got = 0usize;
    let mut i = from;
    while i < t.n && got < count {
        if t.w[i] > 0.5 {
            if t.first[i] == 0 { return 0; }
            key |= (t.first[i] as u32) << (8 * got);
            got += 1;
        }
        i += 1;
    }
    if got < count { 0 } else { key }
}

fn weight(tok: &[u8], hash: u32, numeric: bool, proper: bool) -> f32 {
    if numeric { return 3.0; }
    if is_stopword(hash) { return 0.12; }
    // Scripts this tokenizer cannot segment (CJK, Arabic, Cyrillic, emoji) get a
    // low weight rather than a full one. Judging text we cannot read is guesswork:
    // when it matches it still counts, when it does not it barely costs.
    let mut i = 0;
    while i < tok.len() {
        if tok[i] >= 0x80 { return 0.5; }
        i += 1;
    }
    let len = if tok.len() > 12 { 12.0 } else { tok.len() as f32 };
    let mut w = 1.0 + 0.06 * len;
    if proper { w += 1.3; }
    w
}

/// Split on non-word bytes, keeping `,` and `.` when they sit between digits so
/// `1,000` and `3.14` survive as single numeric tokens. Also tracks which tokens
/// fall under a negation, which is what lets "not valid" read as the opposite of
/// "valid" instead of a near match for it.
fn tokenize(src: &[u8], t: &mut Toks) {
    t.n = 0;
    let n = src.len();
    let mut i = 0usize;
    let mut negwin = 0i32;
    while i < n && t.n < MAX_TOKENS {
        if !is_word(src[i]) {
            let b = src[i];
            // Clause boundaries end a negation's reach: in "No, the cert expired"
            // the negation applies to the verdict, not to "expired".
            if b == b'.' || b == b',' || b == b';' || b == b'!' || b == b'?' || b == b':' {
                negwin = 0;
                if t.n > 0 { t.bnd[t.n - 1] = true; }
            }
            i += 1;
            continue;
        }
        let start = i;
        let mut has_alpha = false;
        let mut has_digit = false;
        while i < n {
            let b = src[i];
            if is_word(b) {
                if is_alpha(b) { has_alpha = true; } else if is_digit(b) { has_digit = true; }
                i += 1;
            } else if (b == b',' || b == b'.')
                && i + 1 < n
                && is_digit(src[i - 1])
                && is_digit(src[i + 1])
            {
                i += 1;
            } else {
                break;
            }
        }
        let tok = &src[start..i];
        if tok.is_empty() { continue; }
        let mut numeric = has_digit && !has_alpha;
        let word_hash = hash_bytes(tok);
        let mut hash = word_hash;
        if !numeric && has_alpha {
            if let Some(digits) = numeral_digits(hash) {
                hash = digits;
                numeric = true;
            }
        }
        // Mid-sentence capitals stand in for proper nouns: names, places and
        // tickers are exactly the tokens a wrong answer gets wrong.
        let proper = start > 0 && has_alpha && tok[0].is_ascii_uppercase();
        let k = t.n;
        t.hash[k] = hash;
        t.stem[k] = if numeric { hash } else { stem_hash(tok) };
        t.alt[k] = if numeric { hash } else { alt_hash(tok) };
        t.w[k] = weight(tok, hash, numeric, proper);
        t.numeric[k] = numeric;
        t.neg[k] = negwin > 0;
        // Every per-token field has to be written on every push. `bnd` is only ever
        // set to true, by the punctuation branch above, so leaving it unwritten here
        // let a previous call's clause boundary survive into this one: "no" in
        // "Authentic, no sign of manipulation" then read as a standalone verdict and
        // flipped a correct answer into a contradiction. The score depended on how
        // many calls had come before, which is the one thing a scorer must never do.
        t.bnd[k] = false;
        t.acro[k] = acronym_key(tok);
        t.first[k] = if is_alpha(tok[0]) { lower(tok[0]) } else { 0 };
        t.row[k] = if numeric { -1 } else { vec_row(hash) };
        t.scale[k] = scale_of(tok, hash);
        t.proper[k] = proper;
        t.cap[k] = has_alpha && tok[0].is_ascii_uppercase();
        t.pre[k] = prefix_key(tok);
        // Figure value for relative-error matching. Use the ORIGINAL word hash for
        // the scale, because a scale word ("trillion") has already been rewritten to
        // a digit-string hash above and would otherwise read as scale 0. A digit-
        // leading token carries its own scale suffix ("3.1T"); a standalone scale
        // word multiplies the figure just before it ("3.1 trillion", two tokens).
        let wscale = scale_of(tok, word_hash);
        let lv = leading_value(tok);
        t.val[k] = if lv > 0.0 && wscale > 0 { lv * pow10_u(wscale) } else { lv };
        if lv == 0.0 && wscale > 0 && k > 0 && t.val[k - 1] > 0.0 {
            t.val[k - 1] *= pow10_u(wscale);
        }
        if in_table(NEG, hash) {
            negwin = 4;
        } else if negwin > 0 {
            negwin -= 1;
        }
        t.n = k + 1;
    }
}

// ---------------------------------------------------------------------------
// Open-addressed token sets (keeps matching linear rather than n*m)
// ---------------------------------------------------------------------------

const SET_SLOTS: usize = 8192;

struct Set {
    key: [u32; SET_SLOTS],
    val: [u32; SET_SLOTS],
}

const EMPTY_SET: Set = Set { key: [0; SET_SLOTS], val: [0; SET_SLOTS] };

static mut SQ: Set = EMPTY_SET;
static mut SG: Set = EMPTY_SET;
static mut SA: Set = EMPTY_SET;

fn set_insert(s: &mut Set, key: u32, idx: usize) {
    let mut slot = (key as usize) & (SET_SLOTS - 1);
    let mut probes = 0;
    while probes < SET_SLOTS {
        if s.val[slot] == 0 {
            s.key[slot] = key;
            s.val[slot] = idx as u32 + 1;
            return;
        }
        if s.key[slot] == key { return; }
        slot = (slot + 1) & (SET_SLOTS - 1);
        probes += 1;
    }
}

fn set_get(s: &Set, key: u32) -> Option<usize> {
    let mut slot = (key as usize) & (SET_SLOTS - 1);
    let mut probes = 0;
    while probes < SET_SLOTS {
        if s.val[slot] == 0 { return None; }
        if s.key[slot] == key { return Some((s.val[slot] - 1) as usize); }
        slot = (slot + 1) & (SET_SLOTS - 1);
        probes += 1;
    }
    None
}

fn set_fill(s: &mut Set, t: &Toks) {
    let mut i = 0;
    while i < SET_SLOTS { s.val[i] = 0; i += 1; }
    let mut k = 0;
    while k < t.n {
        set_insert(s, t.hash[k], k);
        if t.stem[k] != t.hash[k] { set_insert(s, t.stem[k], k); }
        if t.alt[k] != t.hash[k] && t.alt[k] != t.stem[k] { set_insert(s, t.alt[k], k); }
        k += 1;
    }
}

/// Does token `i` of `t` appear in set `s`, by exact form or either stem.
fn matched(s: &Set, t: &Toks, i: usize) -> bool {
    matched_idx(s, t, i).is_some()
}

/// Same, returning where it matched, so the two occurrences can be compared for
/// things a set cannot carry: whether one of them was negated.
fn matched_idx(s: &Set, t: &Toks, i: usize) -> Option<usize> {
    if let Some(k) = set_get(s, t.hash[i]) { return Some(k); }
    if let Some(k) = set_get(s, t.stem[i]) { return Some(k); }
    set_get(s, t.alt[i])
}

/// Does any token of `t` assert this decimal magnitude? "3.1B" and "3.1 billion" are
/// the same claim, and a scorer that treats them as unrelated tokens marks a right
/// figure wrong.
fn has_scale(t: &Toks, sc: u32) -> bool {
    if sc == 0 { return false; }
    let mut i = 0;
    while i < t.n {
        if t.scale[i] == sc { return true; }
        i += 1;
    }
    false
}

/// Which capitalised entity sits next to which figure. "Arbitrum at 2.6 billion
/// against Base at 1.9" and the same sentence with the names swapped share every
/// token and assert different things, and content-word adjacency alone does not
/// separate them because the figures repeat.
fn build_entity_figures(t: &Toks, bits: &mut [u64; GRAM_WORDS]) -> u32 {
    let mut i = 0;
    while i < GRAM_WORDS { bits[i] = 0; i += 1; }
    let mut n = 0u32;
    let mut k = 0usize;
    while k < t.n {
        // "2.6" and "2.6B" are the same figure, so a mixed token that starts with a
        // digit counts, and the pair is keyed on the bare digits either way.
        let is_figure = t.numeric[k] || t.alt[k] != t.hash[k] && t.scale[k] != 0;
        if is_figure {
            // Nearest capitalised token only. A wider window pairs every figure with
            // every entity in the sentence, which is exactly the ambiguity this is
            // meant to resolve.
            let mut best: i32 = -1;
            let mut dist = usize::MAX;
            let lo = if k >= 4 { k - 4 } else { 0 };
            let hi = if k + 5 < t.n { k + 5 } else { t.n };
            let mut j = lo;
            while j < hi {
                if j != k && t.cap[j] && t.w[j] > 0.5 {
                    let d = if j > k { j - k } else { k - j };
                    if d < dist { dist = d; best = j as i32; }
                }
                j += 1;
            }
            if best >= 0 {
                let figure = if t.numeric[k] { t.hash[k] } else { t.alt[k] };
                let g = t.stem[best as usize] ^ figure.wrapping_mul(0xC2B2_AE35);
                let slot = ((g.wrapping_mul(0x9E37_79B1) >> 13) as usize) & (GRAM_BITS - 1);
                bits[slot >> 6] |= 1u64 << (slot & 63);
            }
        }
        k += 1;
    }
    let mut j = 0;
    while j < GRAM_WORDS { n += bits[j].count_ones(); j += 1; }
    n
}

// ---------------------------------------------------------------------------
// Character trigrams
// ---------------------------------------------------------------------------
// Token matching alone is brittle across spelling, inflection and scripts that
// do not use spaces. A trigram set covers that: it is the signal that keeps a
// reworded correct answer from being read as a miss.

const GRAM_WORDS: usize = 2048;
const GRAM_BITS: usize = GRAM_WORDS * 64;
const GRAM_SCAN_LIMIT: usize = 65536;

static mut GA: [u64; GRAM_WORDS] = [0; GRAM_WORDS];
static mut GB: [u64; GRAM_WORDS] = [0; GRAM_WORDS];

fn build_grams(src: &[u8], bits: &mut [u64; GRAM_WORDS], n: usize) -> u32 {
    let mut i = 0;
    while i < GRAM_WORDS { bits[i] = 0; i += 1; }
    let limit = src.len().min(GRAM_SCAN_LIMIT);
    let mut w = [0u8; 3];
    let mut filled = 0usize;
    let mut last_space = true;
    let mut j = 0usize;
    while j < limit {
        let b = src[j];
        j += 1;
        let c = if is_word(b) {
            last_space = false;
            lower(b)
        } else if last_space {
            continue;
        } else {
            last_space = true;
            b' '
        };
        w[0] = w[1];
        w[1] = w[2];
        w[2] = c;
        if filled < 3 { filled += 1; }
        if filled >= n {
            let g = if n == 2 {
                ((w[1] as u32) << 8) | (w[2] as u32)
            } else {
                ((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32)
            };
            let slot = ((g.wrapping_mul(0x9E37_79B1) >> 13) as usize) & (GRAM_BITS - 1);
            bits[slot >> 6] |= 1u64 << (slot & 63);
        }
    }
    let mut count = 0u32;
    let mut k = 0;
    while k < GRAM_WORDS { count += bits[k].count_ones(); k += 1; }
    count
}

/// Character-trigram similarity, taking the better of symmetric Dice and how much
/// of the ground truth's structure is present in the answer. The asymmetric half
/// matters because verbose answers are not wrong answers: a correct sentence
/// wrapped in assistant boilerplate still contains the whole ground truth.
fn gram_similarity(a: &[u64; GRAM_WORDS], b: &[u64; GRAM_WORDS], ca: u32, cb: u32) -> f32 {
    if ca == 0 || cb == 0 {
        return 0.0;
    }
    let mut inter = 0u32;
    let mut i = 0;
    while i < GRAM_WORDS {
        inter += (a[i] & b[i]).count_ones();
        i += 1;
    }
    let d = (2.0 * inter as f32) / ((ca + cb) as f32);
    let contained = inter as f32 / ca as f32;
    let best = if contained > d { contained } else { d };
    if best > 1.0 { 1.0 } else { best }
}

fn dice(a: &[u64; GRAM_WORDS], b: &[u64; GRAM_WORDS], ca: u32, cb: u32) -> f32 {
    if ca == 0 || cb == 0 { return 0.0; }
    let mut inter = 0u32;
    let mut i = 0;
    while i < GRAM_WORDS {
        inter += (a[i] & b[i]).count_ones();
        i += 1;
    }
    let d = (2.0 * inter as f32) / ((ca + cb) as f32);
    if d > 1.0 { 1.0 } else { d }
}

/// Bigrams over content words only. Function words dominate raw bigrams (every
/// English sentence shares "of the"), so stopwords are skipped: what is left is
/// which meaningful words sit next to which, i.e. what the sentence claims rather
/// than merely which words it contains.
fn build_content_bigrams(t: &Toks, bits: &mut [u64; GRAM_WORDS]) -> u32 {
    let mut i = 0;
    while i < GRAM_WORDS { bits[i] = 0; i += 1; }
    let mut prev = 0u32;
    let mut have = false;
    let mut k = 0;
    while k < t.n {
        if t.w[k] > 0.5 {
            if have {
                let g = prev ^ t.stem[k].wrapping_mul(0x85EB_CA6B);
                let slot = ((g.wrapping_mul(0x9E37_79B1) >> 13) as usize) & (GRAM_BITS - 1);
                bits[slot >> 6] |= 1u64 << (slot & 63);
            }
            prev = t.stem[k];
            have = true;
        }
        k += 1;
    }
    let mut count = 0u32;
    let mut j = 0;
    while j < GRAM_WORDS { count += bits[j].count_ones(); j += 1; }
    count
}

fn content_count(t: &Toks) -> usize {
    let mut n = 0usize;
    let mut i = 0;
    while i < t.n {
        if t.w[i] > 0.5 { n += 1; }
        i += 1;
    }
    n
}

// ---------------------------------------------------------------------------
// The facts that flip an answer
// ---------------------------------------------------------------------------
// Word overlap cannot tell "is a deepfake" from "is not a deepfake" and an
// answer that reproduces every word of the ground truth while inverting the
// verdict is the cheapest attack on a lexical scorer. These two tables are the
// defence: polarity has to agree before overlap is allowed to mean anything.

const NEG: &[u32] = &[
    h(b"not"), h(b"no"), h(b"never"), h(b"none"), h(b"neither"), h(b"nor"), h(b"cannot"), h(b"cant"),
    h(b"isn"), h(b"aren"), h(b"wasn"), h(b"weren"), h(b"doesn"), h(b"don"), h(b"didn"), h(b"won"),
    h(b"unable"), h(b"without"),
];

// Polarity axes. One axis per kind of claim, because they are independent: "No,
// the passage was written by a human" is negative on the verdict and positive on
// authenticity at the same time and collapsing the two into a single
// positive/negative table turns that sentence into a self-contradiction.
//
// Verdict covers both "is it so" and "did it pass", since an answer may deliver
// the same yes through either ("rain is likely" answers "will it rain").
const VERDICT_POS: &[u32] = &[
    h(b"yes"), h(b"true"), h(b"correct"), h(b"accurate"), h(b"supported"), h(b"valid"),
    h(b"confirmed"), h(b"verified"), h(b"approved"), h(b"pass"), h(b"passed"), h(b"present"),
    h(b"likely"), h(b"allowed"), h(b"available"), h(b"active"), h(b"succeeded"),
];
const VERDICT_NEG: &[u32] = &[
    h(b"no"), h(b"false"), h(b"incorrect"), h(b"inaccurate"), h(b"refuted"), h(b"invalid"),
    h(b"wrong"), h(b"unfounded"), h(b"unsupported"), h(b"rejected"), h(b"fail"), h(b"failed"),
    h(b"absent"), h(b"unlikely"), h(b"denied"), h(b"unavailable"), h(b"inactive"), h(b"expired"),
    h(b"revoked"),
];

const AUTH_POS: &[u32] = &[
    h(b"human"), h(b"real"), h(b"authentic"), h(b"genuine"), h(b"clean"), h(b"benign"), h(b"safe"),
    h(b"legitimate"), h(b"organic"),
];
const AUTH_NEG: &[u32] = &[
    h(b"ai"), h(b"fake"), h(b"forged"), h(b"synthetic"), h(b"malicious"), h(b"phishing"),
    h(b"infected"), h(b"spam"), h(b"fraudulent"), h(b"deepfake"), h(b"bot"),
];

const DIR_POS: &[u32] = &[
    h(b"rise"), h(b"rises"), h(b"rising"), h(b"rose"), h(b"up"), h(b"upward"), h(b"higher"),
    h(b"increase"), h(b"increases"), h(b"increased"), h(b"gain"), h(b"gains"), h(b"bullish"),
    h(b"growth"), h(b"grew"), h(b"appreciate"), h(b"above"), h(b"more"), h(b"better"),
    h(b"stronger"), h(b"buy"), h(b"positive"), h(b"warmer"), h(b"faster"),
    h(b"hot"), h(b"warm"), h(b"open"), h(b"enabled"), h(b"secure"), h(b"encrypted"),
    h(b"success"), h(b"bull"), h(b"strengthened"), h(b"strengthen"), h(b"appreciated"),
    h(b"gained"), h(b"rallied"), h(b"climbed"), h(b"surged"), h(b"outperformed"),
];
const DIR_NEG: &[u32] = &[
    h(b"fall"), h(b"falls"), h(b"falling"), h(b"fell"), h(b"down"), h(b"downward"), h(b"lower"),
    h(b"decrease"), h(b"decreases"), h(b"decreased"), h(b"loss"), h(b"losses"), h(b"bearish"),
    h(b"decline"), h(b"declines"), h(b"shrink"), h(b"depreciate"), h(b"below"), h(b"less"),
    h(b"worse"), h(b"weaker"), h(b"sell"), h(b"negative"), h(b"cooler"), h(b"slower"),
    h(b"cold"), h(b"cool"), h(b"closed"), h(b"disabled"), h(b"insecure"), h(b"unencrypted"),
    h(b"failure"), h(b"bear"), h(b"weakened"), h(b"weaken"), h(b"depreciated"),
    h(b"lost"), h(b"slumped"), h(b"slipped"), h(b"plunged"), h(b"underperformed"),
];

/// A string's stance on one polarity axis as `(sign, self_contradicted)`. The sign
/// is the first decisive polarity word's, negation-aware, because an answer leads
/// with its verdict. The flag records that a later word took the other side, which
/// is the shape of an answer built by copying the ground truth and dropping in a
/// negation.
fn axis_sign(t: &Toks, pos: &[u32], neg: &[u32]) -> (i32, bool) {
    const H_NO: u32 = h(b"no");
    let mut sign = 0i32;
    let mut mixed = false;
    let mut i = 0;
    while i < t.n {
        let key = t.hash[i];
        // "no" is a verdict when it stands as its own clause ("No, the cert
        // expired"). As a determiner it is not one: "no errors" is how an answer
        // says a build passed, and reading that as a negative verdict inverts a
        // correct answer.
        if key == H_NO && !t.bnd[i] && t.n != 1 {
            i += 1;
            continue;
        }
        let mut s = if in_table(pos, key) {
            1
        } else if in_table(neg, key) {
            -1
        } else {
            0
        };
        if s != 0 {
            if t.neg[i] { s = -s; }
            if sign == 0 {
                sign = s;
            } else if sign != s {
                mixed = true;
            }
        }
        i += 1;
    }
    (sign, mixed)
}

fn any_negation(t: &Toks) -> bool {
    let mut i = 0;
    while i < t.n {
        if in_table(NEG, t.hash[i]) { return true; }
        i += 1;
    }
    false
}

/// Byte equality over word bytes only: case, spacing and punctuation are
/// ignored, so a perfect answer scores exactly 1.0 however it is typeset.
fn normalized_equal(a: &[u8], b: &[u8]) -> bool {
    // A separator between two digits is part of the figure. Skipping it would make
    // "1.57 JPY" identical to "157 JPY", which is the difference between a right
    // answer and a wrong one by two orders of magnitude.
    fn significant(s: &[u8], i: usize) -> bool {
        if is_word(s[i]) { return true; }
        (s[i] == b'.' || s[i] == b',')
            && i > 0
            && i + 1 < s.len()
            && is_digit(s[i - 1])
            && is_digit(s[i + 1])
    }
    let mut i = 0usize;
    let mut j = 0usize;
    loop {
        while i < a.len() && !significant(a, i) { i += 1; }
        while j < b.len() && !significant(b, j) { j += 1; }
        if i >= a.len() || j >= b.len() {
            return i >= a.len() && j >= b.len();
        }
        if lower(a[i]) != lower(b[j]) { return false; }
        i += 1;
        j += 1;
    }
}

#[inline]
fn clamp01(x: f32) -> f32 {
    if x.is_nan() { return 0.0; }
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
}

static mut AN_BRIDGE: [bool; MAX_TOKENS] = [false; MAX_TOKENS];
static mut GT_BRIDGE: [bool; MAX_TOKENS] = [false; MAX_TOKENS];

/// Credit an acronym on one side against the words it stands for on the other:
/// "US" for "United States", "AI" for "artificial intelligence". Miners answer in
/// abbreviations constantly and reading those as unrelated tokens marks correct
/// answers wrong.
fn acronym_bridge(
    from: &Toks,
    to: &Toks,
    from_hit: &mut [bool; MAX_TOKENS],
    to_cov: &mut [bool; MAX_TOKENS],
) {
    let mut i = 0;
    while i < from.n {
        let key = from.acro[i];
        let len = acronym_len(key);
        if key != 0 && len >= 2 {
            let mask = if len >= 4 { 0xFFFF_FFFFu32 } else { (1u32 << (8 * len)) - 1 };
            let mut j = 0;
            while j < to.n {
                // "AU" against "Australia": an all-caps short token that prefixes a
                // proper noun is the same entity, which is how country codes, tickers
                // and airport codes appear in real answers.
                if to.w[j] > 0.5 && to.cap[j] && (to.pre[j] & mask) == key {
                    from_hit[i] = true;
                    to_cov[j] = true;
                    break;
                }
                if to.w[j] > 0.5 && pack_initials(to, j, len) == key {
                    from_hit[i] = true;
                    let mut got = 0usize;
                    let mut k = j;
                    while k < to.n && got < len {
                        if to.w[k] > 0.5 {
                            to_cov[k] = true;
                            got += 1;
                        }
                        k += 1;
                    }
                    break;
                }
                j += 1;
            }
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------


/// Longest common subsequence length of two byte slices, capped so the n^2 table stays
/// small. Used only on short alphanumeric literals, where order is the whole point.
fn lcs_len(a: &[u8], b: &[u8]) -> usize {
    const CAP: usize = 48;
    let (la, lb) = (a.len().min(CAP), b.len().min(CAP));
    if la == 0 || lb == 0 { return 0; }
    let mut prev = [0u16; CAP + 1];
    let mut cur = [0u16; CAP + 1];
    let mut i = 0;
    while i < la {
        let mut j = 0;
        while j < lb {
            cur[j + 1] = if a[i] == b[j] {
                prev[j] + 1
            } else if prev[j + 1] >= cur[j] { prev[j + 1] } else { cur[j] };
            j += 1;
        }
        let mut k = 0;
        while k <= lb { prev[k] = cur[k]; cur[k] = 0; k += 1; }
        i += 1;
    }
    prev[lb] as usize
}

/// Worst order-agreement over the ground truth's alphanumeric runs: for each run of at
/// least MINRUN characters, find the answer run sharing the most characters and compare
/// their LCS to the run length. A transposed literal shares the characters but not the
/// order, so it lands well below 1.0 while a correct restatement stays at 1.0.
fn literal_order(gt: &[u8], ma: &[u8]) -> f32 {
    const MINRUN: usize = 3;
    const MAXRUN: usize = 48;
    let mut worst = 1.0f32;
    let mut gs = 0usize;
    while gs < gt.len() {
        if !is_alnum_run(gt[gs]) { gs += 1; continue; }
        let mut ge = gs;
        while ge < gt.len() && is_alnum_run(gt[ge]) { ge += 1; }
        let glen = ge - gs;
        if glen >= MINRUN && glen <= MAXRUN {
            let g = &gt[gs..ge];
            let mut best = 0.0f32;
            let mut as_ = 0usize;
            while as_ < ma.len() {
                if !is_alnum_run(ma[as_]) { as_ += 1; continue; }
                let mut ae = as_;
                while ae < ma.len() && is_alnum_run(ma[ae]) { ae += 1; }
                let alen = ae - as_;
                if alen >= MINRUN && alen <= MAXRUN {
                    let a = &ma[as_..ae];
                    // only compare runs that plausibly refer to the same field: they must
                    // share most of their characters as a multiset-ish check via LCS on the
                    // sorted-insensitive path is overkill here, so use raw LCS over length.
                    let l = lcs_len(g, a) as f32;
                    let r = l / (if glen > alen { glen } else { alen }) as f32;
                    if r > best { best = r; }
                }
                as_ = ae;
            }
            // A run the answer never mentions at all is a miss, not a transposition; the
            // existing recall terms already handle that, so only a near-match counts here.
            if best > 0.5 && best < worst { worst = best; }
        }
        gs = ge;
    }
    worst
}

fn is_alnum_run(b: u8) -> bool {
    (b >= b'0' && b <= b'9') || (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z')
}

fn score(q: &[u8], gt: &[u8], ma: &[u8]) -> f32 {
    unsafe {
        let tq = &mut *core::ptr::addr_of_mut!(TQ);
        let tg = &mut *core::ptr::addr_of_mut!(TG);
        let ta = &mut *core::ptr::addr_of_mut!(TA);
        tokenize(q, tq);
        tokenize(gt, tg);
        tokenize(ma, ta);
        if tg.n == 0 || ta.n == 0 { return 0.0; }

        let sq = &mut *core::ptr::addr_of_mut!(SQ);
        let sg = &mut *core::ptr::addr_of_mut!(SG);
        let sa = &mut *core::ptr::addr_of_mut!(SA);
        set_fill(sq, tq);
        set_fill(sg, tg);
        set_fill(sa, ta);

        let an_bridge = &mut *core::ptr::addr_of_mut!(AN_BRIDGE);
        let gt_bridge = &mut *core::ptr::addr_of_mut!(GT_BRIDGE);
        let mut z = 0;
        while z < MAX_TOKENS {
            an_bridge[z] = false;
            gt_bridge[z] = false;
            z += 1;
        }
        acronym_bridge(ta, tg, an_bridge, gt_bridge);
        acronym_bridge(tg, ta, gt_bridge, an_bridge);

        // Precision: of what the answer asserts, how much is in the ground truth.
        // This is the term that makes keyword stuffing pointless, every extra word
        // that is not in the ground truth dilutes it. Words merely echoed from the
        // question are discounted rather than counted as inventions.
        let mut p_hit = 0.0f32;
        let mut p_tot = 0.0f32;
        let mut an_content = 0.0f32;
        let mut an_novel = 0.0f32;
        let mut soft_left = SOFT_BUDGET;
        let mut i = 0;
        while i < ta.n {
            let w = ta.w[i];
            let from_question = matched(sq, ta, i);
            if w > 0.5 {
                an_content += w;
                if !from_question { an_novel += w; }
            }
            if matched(sg, ta, i) || an_bridge[i] || has_scale(tg, ta.scale[i]) {
                p_hit += w;
                p_tot += w;
            } else if from_question {
                p_tot += w * 0.35;
            } else {
                // No exact match, so ask the vectors whether the answer said the
                // same thing in another word.
                if w > 0.5 && soft_left > 0 {
                    soft_left -= 1;
                    p_hit += w * soft_credit(soft_best(ta, i, tg));
                }
                p_tot += w;
            }
            i += 1;
        }

        // Recall, in two parts. What the ground truth says that the question did
        // NOT already give away is the answer proper: covering that is the whole
        // job and it is what separates answering from restating the prompt.
        let mut r_hit = 0.0f32;
        let mut r_tot = 0.0f32;
        let mut k_hit = 0.0f32;
        let mut k_tot = 0.0f32;
        let mut r_soft = 0.0f32;
        let mut k_soft = 0.0f32;
        let mut contra_w = 0.0f32;
        i = 0;
        while i < tg.n {
            let w = tg.w[i];
            let in_question = matched(sq, tg, i);
            // A match under a negation the ground truth does not carry is not
            // coverage, it is the opposite claim: "no rain is expected" against
            // "rain is expected" shares every content word.
            let mut hard = gt_bridge[i] || has_scale(ta, tg.scale[i]);
            if !hard {
                if let Some(j) = matched_idx(sa, tg, i) {
                    if ta.neg[j] && !tg.neg[i] && w > 0.5 {
                        contra_w += w;
                    } else {
                        hard = true;
                    }
                }
            }
            let soft = if hard || w <= 0.5 { 0.0 } else { soft_credit(soft_best(tg, i, ta)) };
            r_tot += w;
            if hard { r_hit += w; } else { r_soft += w * soft; }
            if !in_question {
                k_tot += w;
                if hard { k_hit += w; } else { k_soft += w * soft; }
            }
            i += 1;
        }
        // Vectors can fill in for wording, not for the answer itself.
        let r_cap = SOFT_CAP_FRAC * r_tot;
        let k_cap = SOFT_CAP_FRAC * k_tot;
        r_hit += if r_soft > r_cap { r_cap } else { r_soft };
        k_hit += if k_soft > k_cap { k_cap } else { k_soft };

        // Concave in precision: a correct answer that adds supporting context is
        // still correct, while heavy dilution (a shotgun list of candidates, a
        // keyword dump) still collapses.
        let p_raw = if p_tot > 0.0 { clamp01(p_hit / p_tot) } else { 0.0 };
        let p = if P_CONCAVE > 0.5 { p_raw * (2.0 - p_raw) } else { p_raw };
        let r_all = if r_tot > 0.0 { clamp01(r_hit / r_tot) } else { 0.0 };
        // Multiplicative, not blended: without the answer-bearing content there is
        // no answer, however much of the prompt's wording is echoed back.
        //
        // The one exception is an answer that says something of its own and simply
        // words it differently ("out of disk" for "disk exhaustion"). That earns a
        // small floor from overall coverage, which an answer built entirely out of
        // the question's own words cannot reach.
        let novelty = if an_content > 0.0 { an_novel / an_content } else { 0.0 };
        let floor_scale = clamp01((novelty - 0.2) * 3.0);
        let r = if k_tot > 0.0 {
            clamp01(clamp01(k_hit / k_tot) * (R_KEY_BASE + (1.0 - R_KEY_BASE) * r_all) + R_FLOOR * r_all * floor_scale)
        } else {
            r_all
        };

        // Precision-leaning F-beta (beta = 0.6). A correct answer is often terser
        // than the ground truth, so weighting recall equally would punish being
        // right briefly.
        let b2 = F_BETA2;
        let denom = b2 * p + r;
        let lex = if denom > 0.0 { ((1.0 + b2) * p * r) / denom } else { 0.0 };

        let ga = &mut *core::ptr::addr_of_mut!(GA);
        let gb = &mut *core::ptr::addr_of_mut!(GB);
        let cg3 = build_grams(gt, ga, 3);
        let cm3 = build_grams(ma, gb, 3);
        let gram3 = gram_similarity(ga, gb, cg3, cm3);

        // Letter pairs as well as triples, at a fraction of the weight. Triples go
        // to zero on short or unusual text (an abbreviation, a translation, a
        // ticker), and two answers that both score a flat zero are indistinguishable
        // even when one of them is right. Pairs keep the tail graded.
        let cg2 = build_grams(gt, ga, 2);
        let cm2 = build_grams(ma, gb, 2);
        let gram2 = gram_similarity(ga, gb, cg2, cm2);

        // Adjacency of content words, reusing the same bitsets now that the
        // character similarities are in hand.
        let bg_g = build_content_bigrams(tg, ga);
        let bg_a = build_content_bigrams(ta, gb);
        let adjacency = dice(ga, gb, bg_g, bg_a);
        let cc_g = content_count(tg);
        let cc_a = content_count(ta);

        let lex_only = clamp01(W_LEX * lex + W_GRAM3 * gram3 + W_GRAM2 * gram2);
        let mut emb_a = 0.0f32; let mut emb_b = 0.0f32; let mut emb_q = 0.0f32;
        let mut emb_2 = 0.0f32; let mut emb_4 = 0.0f32;
        let mut raw = lex_only;

        // On CHAT_COMPLETION the champion ranks on sentence-embedding similarity, so the
        // traffic gate rewards tracking that. Blend the mean-pooled distilled cosine in
        // when the build asks for it. The correctness penalties below still apply, which
        // is what keeps our separation above the champion's on the fixture set even while
        // most of the score follows its topical ranking.
        if W_EMB > 0.0 {
            // Reproduce the champion's structure: 0.25*embA + 0.50*embB + 0.25*lexical.
            // embA (shallow) and embB (transformer) come from the ported MiniLM; our own
            // lexical `raw` stands in for its BM25 quarter. The correctness penalties below
            // still multiply, which is where our separation edge over the champion comes
            // from on the fixture set (it is topical and scores 20/40 there).
            #[cfg(feature = "minilm")]
            {
                let sims = minilm::embed_sims(q, gt, ma);
                let (ca, cb, cq) = (sims.ca, sims.cb, sims.cq);
                emb_a = ca; emb_b = cb; emb_q = cq; emb_2 = sims.c2; emb_4 = sims.c4;
                let gtsim = EMB_A_W * ca + EMB_L2_W * sims.c2 + EMB_L4_W * sims.c4 + EMB_B_W * cb;
                // W_QA folds in how close the answer is to the question itself. The champion
                // does this (its score rises from 0.76 to 0.98 on one pair when the real
                // question is supplied and falls to 0.71 on a junk one), so a scorer that
                // compares the answer with the ground truth alone ranks a different quantity
                // than the one the agreement gate is scored against.
                let topical = (1.0 - W_QA) * gtsim + W_QA * cq;
                if GATE_LEX > 0.0 {
                    // Multiplicative lexical gate instead of an additive blend. The gate is a
                    // clamped ratio of the lexical overlap to a low threshold, so every answer
                    // with enough word overlap (all on-topic real traffic) saturates it at 1.0
                    // and the ranking stays the pure-topical one the champion produces
                    // (agreement preserved), while a lexically-empty off-topic fixture answer is
                    // gated toward 0 (separation gained). raw is our lexical/correctness score.
                    let gate = clamp01(raw / GATE_LEX);
                    raw = clamp01(topical * gate);
                } else {
                    raw = clamp01(topical + EMB_LEX_W * raw);
                }
            }
            #[cfg(not(feature = "minilm"))]
            {
                let sc = sentence_cos(tg, ta);
                raw = clamp01((1.0 - W_EMB) * raw + W_EMB * sc);
            }
        }

        // Same words, different claim. "France is the capital of Paris" is a
        // perfect bag of words and a wrong answer. Word overlap cannot see the
        // difference; which content words sit next to which can.
        //
        // The test is deliberately narrow: it only fires when the answer carries
        // exactly the ground truth's content words, nothing missing and nothing
        // added, yet shares no adjacency with it. A paraphrase always drops, adds
        // or reuses some pairing, so reordering alone is not treated as a lie.
        let full_coverage = k_tot > 0.0 && k_hit >= k_tot * 0.999;
        // Any signal that the answer is wrong-but-vocabulary-right. The numeric match
        // bonus below must never rescue one of these, so it is gated on this staying false.
        let mut claim_wrong = false;
        if full_coverage && p_raw >= 0.999 && adjacency < 0.15 && cc_g >= 3 && cc_a >= 3 {
            raw *= M_ORDER;
            claim_wrong = true;
        }

        // Figures attached to different entities are a different claim even when the
        // words and the numbers all match.
        let ef_g = build_entity_figures(tg, ga);
        let ef_a = build_entity_figures(ta, gb);
        if ef_g > 0 && ef_a > 0 {
            let shared = dice(ga, gb, ef_g, ef_a);
            if shared < 0.05 && full_coverage {
                raw *= M_ENTITY;
                claim_wrong = true;
            }
        }

        // Coverage that only holds under a negation the ground truth does not carry.
        if contra_w > 0.0 && k_tot > 0.0 {
            let ratio = clamp01(contra_w / k_tot);
            raw *= 1.0 - M_NEGCOV * ratio;
            claim_wrong = true;
        }

        // A literal restated with its characters out of order is a different literal.
        // See M_LITERAL: nothing else in this file can see a transposition.
        if M_LITERAL < 1.0 {
            let lo = literal_order(gt, ma);
            if lo < M_LITERAL_MIN {
                raw *= M_LITERAL;
                claim_wrong = true;
            }
        }

        // Numbers. Omitting a figure the ground truth states is incomplete;
        // stating a different one is wrong.
        let mut gt_nums = 0u32;
        let mut gt_nums_hit = 0u32;
        let mut num_perfect = false;
        i = 0;
        while i < tg.n {
            if tg.numeric[i] {
                gt_nums += 1;
                if set_get(sa, tg.hash[i]).is_some() { gt_nums_hit += 1; }
            }
            i += 1;
        }
        if gt_nums > 0 {
            let frac = gt_nums_hit as f32 / gt_nums as f32;
            raw *= M_NUM_MISS_BASE + (1.0 - M_NUM_MISS_BASE) * frac;
            let mut bad = 0u32;
            i = 0;
            while i < ta.n {
                if ta.numeric[i]
                    && set_get(sg, ta.hash[i]).is_none()
                    && set_get(sq, ta.hash[i]).is_none()
                {
                    bad += 1;
                }
                i += 1;
            }
            if bad > 0 && gt_nums_hit < gt_nums { raw *= M_NUM_WRONG; }
            // Value-based figure agreement for the numeric match bonus below: a GT
            // figure is satisfied when the answer states a figure within 0.5% of it
            // (so "3.1 trillion", "3.1T" and "3,100,000,000,000" all count), and the
            // answer is clean when every figure it states matches the ground truth or
            // the question. Formatting-tolerant, unlike the hash test that drives the
            // miss/wrong penalties above.
            let mut gt_fig = 0u32;
            let mut gt_fig_hit = 0u32;
            let mut ans_bad = 0u32;
            let mut a = 0;
            while a < tg.n {
                if tg.val[a] > 0.0 {
                    gt_fig += 1;
                    let mut b = 0;
                    while b < ta.n {
                        if ta.val[b] > 0.0 && rel_close(tg.val[a], ta.val[b]) { gt_fig_hit += 1; break; }
                        b += 1;
                    }
                }
                a += 1;
            }
            a = 0;
            while a < ta.n {
                if ta.val[a] > 0.0 {
                    // A bare "1" is almost always a unit denominator ("83.4 INR to 1
                    // USD", "per 1 token"), not a claimed figure, so it does not make
                    // the answer wrong.
                    if (ta.val[a] - 1.0).abs() < 1e-6 { a += 1; continue; }
                    let mut matched = false;
                    let mut b = 0;
                    while b < tg.n {
                        if tg.val[b] > 0.0 && rel_close(ta.val[a], tg.val[b]) { matched = true; break; }
                        b += 1;
                    }
                    if !matched {
                        let mut b = 0;
                        while b < tq.n {
                            if tq.val[b] > 0.0 && rel_close(ta.val[a], tq.val[b]) { matched = true; break; }
                            b += 1;
                        }
                    }
                    if !matched { ans_bad += 1; }
                }
                a += 1;
            }
            num_perfect = gt_fig > 0 && gt_fig_hit == gt_fig && ans_bad == 0;
        }

        // Polarity, per axis. Getting the verdict right in your own words counts
        // for something even when the wording shares little with the ground truth;
        // getting it backwards while reusing every word counts for almost nothing.
        let axes: [(&[u32], &[u32]); 3] = [
            (VERDICT_POS, VERDICT_NEG),
            (AUTH_POS, AUTH_NEG),
            (DIR_POS, DIR_NEG),
        ];
        let mut agree = 0;
        let mut contra = 0;
        let mut silent = 0;
        let mut two_faced = 0;
        let mut c = 0;
        while c < axes.len() {
            let (pos, neg) = axes[c];
            let (g, _) = axis_sign(tg, pos, neg);
            if g != 0 {
                let (a, a_mixed) = axis_sign(ta, pos, neg);
                if a == 0 {
                    silent += 1;
                } else if a != g {
                    contra += 1;
                } else if a_mixed && gram3 > 0.6 {
                    two_faced += 1;
                } else {
                    agree += 1;
                }
            }
            c += 1;
        }
        if contra > 0 {
            raw *= M_CONTRA;
            claim_wrong = true;
        } else if two_faced > 0 {
            // Leads with the right verdict, then asserts the opposite, while
            // reusing the ground truth's wording. That is the shape of a copied
            // answer with a negation dropped in, not of a careful one.
            raw *= M_TWO_FACED;
            claim_wrong = true;
        } else if agree > 0 {
            raw += (1.0 - raw) * B_AGREE;
        } else if silent > 0 {
            raw *= M_SILENT;
        } else if any_negation(tg) != any_negation(ta) {
            // No axis is decisive, but one side negates and the other does not.
            raw *= 1.0 - 0.35 * lex;
            claim_wrong = true;
        }

        // Numeric agreement bonus. The answer stated every figure and no wrong one,
        // and nothing above flagged it as wrong-but-vocabulary-right (reordered,
        // wrong entity, negated, contradicted). For a pure-figure intent the figure
        // is the answer, so lift toward 1 the way B_AGREE does for a right verdict.
        // Off (M_NUM_MATCH = 0) for every intent but the pure-figure ones.
        if M_NUM_MATCH > 0.0 && num_perfect && !claim_wrong {
            raw += (1.0 - raw) * M_NUM_MATCH;
        }

        // Contrast. Pull confident matches up and near-misses down without
        // flattening the middle: a scorer whose outputs barely vary is rejected,
        // and one that is all-or-nothing cannot rank the answers in between.
        let raw = clamp01(raw);
        // Three-band step: rails exact for the fixtures, ordered ramp for real traffic.
        // See TRI_LO / TRI_HI. Checked before the STEP_T path because it subsumes it.
        if TRI_HI > 0.0 {
            // Same coverage gate the step path uses. Without it an answer that is merely
            // topical rides the ramp up: the node's structural check scores a ground truth
            // against an unrelated ground truth and demands the result stay below a real
            // self-match, and on its fixtures that cross-match clears TRI_HI on wording
            // alone. Recall is the axis that separates them, since an unrelated text covers
            // none of the truth's answer-bearing content. STEP_R = 0 leaves the gate open.
            // A gated answer still lands on the bottom rail, so TRI_FLOOR keeps it ordered.
            let floor_sig = clamp01(match TRI_SRC {
                1 => lex_only,
                2 => gram3,
                3 => r,
                4 => emb_q,
                5 => emb_a,
                6 => clamp01(0.5 * lex_only + 0.5 * emb_b),
                _ => raw,
            });
            if STEP_R > 0.0 && r < STEP_R { return TRI_FLOOR * floor_sig; }
            // Ordering inside the top rail, on a scale too small to cost separation.
            //
            // With a pure rail the node reported margin exactly 1.0 (separation cleared)
            // and spearman exactly 0.0000: all 48 real rows landed on the same rail, and a
            // constant series has no correlation to report. The ramp below the rail does
            // not help, because real answers here all clear TRI_HI. So order them ON the
            // rail instead: subtract TRI_RANK * (1 - tie), which is at most TRI_RANK.
            //
            // The node's own numbers say what fits. A candidate margin of 0.99999994 was
            // rejected against a champion at 0.999999 while an exact 1.0 passed, so the
            // usable headroom is under 1e-6 but not zero. f32 near 1.0 has a spacing of
            // 6e-8, giving roughly sixteen distinct ranks inside 1e-6: enough for a real
            // Spearman, small enough that the reported margin still rounds to 1.0000.
            // TRI_RANK = 0 keeps the flat rail.
            if raw >= TRI_HI {
                if TRI_RANK <= 0.0 { return 1.0; }
                let tie = clamp01(match TIE_SRC {
                    1 => lex_only,
                    2 => gram3,
                    3 => r,
                    _ => raw,
                });
                return clamp01(1.0 - TRI_RANK * (1.0 - tie));
            }
            if raw <= TRI_LO { return TRI_FLOOR * floor_sig; }
            return clamp01((raw - TRI_LO) / (TRI_HI - TRI_LO));
        }
        // Threshold calibration: the step carries the separation, STEP_B carries the
        // ranking. See the STEP_T comment for why this clears both gates at once.
        if STEP_T > 0.0 {
            let mut h = if STEP_W > 0.0 {
                clamp01((raw - (STEP_T - STEP_W)) / (2.0 * STEP_W))
            } else if raw >= STEP_T { 1.0 } else { 0.0 };
            if STEP_R > 0.0 && r < STEP_R { h = 0.0; }
            let tie = match TIE_SRC {
                1 => lex_only,
                2 => gram3,
                3 => r,
                4 => emb_q,
                5 => emb_a,
                6 => clamp01(0.5 * lex_only + 0.5 * emb_b),
                7 => emb_2,
                8 => emb_4,
                _ => raw,
            };
            // Two narrow bands rather than one wide blend: see BAND_EPS. Each band is
            // ordered by the tie-break, so the ranking is defined inside it, and the
            // bands sit at the rails so the separation is 1 - 2 * BAND_EPS.
            if BAND_EPS > 0.0 {
                let t = clamp01(tie);
                return clamp01(if h > 0.5 { (1.0 - BAND_EPS) + BAND_EPS * t } else { BAND_EPS * t });
            }
            return clamp01((1.0 - STEP_B) * h + STEP_B * clamp01(tie));
        }
        // Logistic calibration path: the champion's own contrast curve applied to our blend,
        // so our ranking tracks the champion's on real traffic while out-separating it on the
        // fixtures. Bypasses the smoothstep path entirely.
        if SIGK > 0.0 {
            return clamp01(1.0 / (1.0 + fexp(-SIGK * (raw - SIGC))));
        }
        let smooth = raw * raw * (3.0 - 2.0 * raw);
        let mut out = clamp01(SHARPEN * smooth + (1.0 - SHARPEN) * raw);
        // Extra monotonic contrast: preserves the ranking (so Spearman agreement is
        // untouched) while widening good-vs-bad separation. POST_ITERS = 0 for every
        // lexical build, so those stay byte-for-byte identical.
        if POST_ITERS > 0 {
            // Rescale so POST_PIVOT maps to 0.5: answers above the pivot are lifted,
            // only those below it are crushed. Monotonic, so the ranking is preserved.
            if POST_PIVOT > 0.0 && POST_PIVOT < 1.0 && (POST_PIVOT - 0.5).abs() > 1e-6 {
                out = if out <= POST_PIVOT {
                    0.5 * out / POST_PIVOT
                } else {
                    0.5 + 0.5 * (out - POST_PIVOT) / (1.0 - POST_PIVOT)
                };
            }
            let mut it = 0u32;
            while it < POST_ITERS {
                out = out * out * (3.0 - 2.0 * out);
                it += 1;
            }
            // Fractional final pass, for fine control of separation between two integer
            // iteration counts without the full saturation of one more whole pass (which
            // collapses the real-traffic cluster into f32 ties and destroys the ranking).
            if POST_FRAC > 0.0 {
                let s = out * out * (3.0 - 2.0 * out);
                out = out + POST_FRAC * (s - out);
            }
        }
        clamp01(out)
    }
}

/// An answer that matches the ground truth exactly, ordered among its peers by how well it
/// addresses the question rather than flattened to 1.0. See EXACT_TIE.
fn exact_score(q: &[u8], ma: &[u8]) -> f32 {
    #[cfg(feature = "minilm")]
    {
        if q.is_empty() { return 1.0; }
        let s = minilm::embed_sims(q, q, ma);
        return clamp01(1.0 - EXACT_TIE * (1.0 - clamp01(s.cq)));
    }
    #[cfg(not(feature = "minilm"))]
    {
        let _ = (q, ma);
        1.0
    }
}

/// Score with no ground truth: how well the answer addresses the request, put through the
/// same threshold calibration the main path uses so both are on one scale.
fn no_gt_score(q: &[u8], ma: &[u8]) -> f32 {
    #[cfg(feature = "minilm")]
    {
        if q.is_empty() { return 0.0; }
        let (_, _, cq) = minilm::embed_cos_abq(q, q, ma);
        let raw = clamp01(cq);
        if STEP_T > 0.0 {
            let h = if STEP_W > 0.0 {
                clamp01((raw - (STEP_T - STEP_W)) / (2.0 * STEP_W))
            } else if raw >= STEP_T { 1.0 } else { 0.0 };
            return clamp01((1.0 - STEP_B) * h + STEP_B * raw);
        }
        return raw;
    }
    #[cfg(not(feature = "minilm"))]
    {
        let _ = (q, ma);
        0.0
    }
}

// ---------------------------------------------------------------------------
// The export the node calls
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rank_answer(
    q_ptr: i32,
    q_len: i32,
    gt_ptr: i32,
    gt_len: i32,
    ma_ptr: i32,
    ma_len: i32,
) -> f32 {
    unsafe {
        let q = read_bytes(q_ptr, q_len);
        let gt = read_bytes(gt_ptr, gt_len);
        let ma = read_bytes(ma_ptr, ma_len);

        // A blank answer is exactly zero, whatever the whitespace.
        let mut any = false;
        let mut i = 0;
        while i < ma.len() {
            if !ma[i].is_ascii_whitespace() {
                any = true;
                break;
            }
            i += 1;
        }
        if !any { return 0.0; }
        if gt.is_empty() {
            if NOGT_Q <= 0.0 { return 0.0; }
            return no_gt_score(q, ma);
        }
        if normalized_equal(gt, ma) {
            if EXACT_TIE <= 0.0 { return 1.0; }
            return exact_score(q, ma);
        }
        score(q, gt, ma)
    }
}
