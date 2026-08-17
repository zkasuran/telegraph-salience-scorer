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

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
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
fn alt_hash(tok: &[u8]) -> u32 {
    let n = tok.len();
    if n >= 5 && tok[n - 2..].eq_ignore_ascii_case(b"ed") {
        hash_bytes(&tok[..n - 1])
    } else {
        hash_bytes(tok)
    }
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
        let numeric = has_digit && !has_alpha;
        // Mid-sentence capitals stand in for proper nouns: names, places and
        // tickers are exactly the tokens a wrong answer gets wrong.
        let proper = start > 0 && has_alpha && tok[0].is_ascii_uppercase();
        let hash = hash_bytes(tok);
        let k = t.n;
        t.hash[k] = hash;
        t.stem[k] = if numeric { hash } else { stem_hash(tok) };
        t.alt[k] = if numeric { hash } else { alt_hash(tok) };
        t.w[k] = weight(tok, hash, numeric, proper);
        t.numeric[k] = numeric;
        t.neg[k] = negwin > 0;
        t.acro[k] = acronym_key(tok);
        t.first[k] = if is_alpha(tok[0]) { lower(tok[0]) } else { 0 };
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
    set_get(s, t.hash[i]).is_some()
        || set_get(s, t.stem[i]).is_some()
        || set_get(s, t.alt[i]).is_some()
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

fn build_grams(src: &[u8], bits: &mut [u64; GRAM_WORDS]) -> u32 {
    let mut i = 0;
    while i < GRAM_WORDS { bits[i] = 0; i += 1; }
    let n = src.len().min(GRAM_SCAN_LIMIT);
    let mut w = [0u8; 3];
    let mut filled = 0usize;
    let mut last_space = true;
    let mut j = 0usize;
    while j < n {
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
        if filled == 3 {
            let g = ((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32);
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
];
const DIR_NEG: &[u32] = &[
    h(b"fall"), h(b"falls"), h(b"falling"), h(b"fell"), h(b"down"), h(b"downward"), h(b"lower"),
    h(b"decrease"), h(b"decreases"), h(b"decreased"), h(b"loss"), h(b"losses"), h(b"bearish"),
    h(b"decline"), h(b"declines"), h(b"shrink"), h(b"depreciate"), h(b"below"), h(b"less"),
    h(b"worse"), h(b"weaker"), h(b"sell"), h(b"negative"), h(b"cooler"), h(b"slower"),
];

/// A string's stance on one polarity axis: -1, 0 or +1 or CONFLICT when it
/// asserts both sides. Negation-aware, so "not valid" is negative.
const CONFLICT: i32 = 2;

fn axis_sign(t: &Toks, pos: &[u32], neg: &[u32]) -> i32 {
    let mut sign = 0i32;
    let mut i = 0;
    while i < t.n {
        let key = t.hash[i];
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
                return CONFLICT;
            }
        }
        i += 1;
    }
    sign
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
    let mut i = 0usize;
    let mut j = 0usize;
    loop {
        while i < a.len() && !is_word(a[i]) { i += 1; }
        while j < b.len() && !is_word(b[j]) { j += 1; }
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
            let mut j = 0;
            while j < to.n {
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
        let mut i = 0;
        while i < ta.n {
            let w = ta.w[i];
            let from_question = matched(sq, ta, i);
            if w > 0.5 {
                an_content += w;
                if !from_question { an_novel += w; }
            }
            if matched(sg, ta, i) || an_bridge[i] {
                p_hit += w;
                p_tot += w;
            } else if from_question {
                p_tot += w * 0.35;
            } else {
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
        i = 0;
        while i < tg.n {
            let w = tg.w[i];
            let in_question = matched(sq, tg, i);
            let covered = matched(sa, tg, i) || gt_bridge[i];
            r_tot += w;
            if covered { r_hit += w; }
            if !in_question {
                k_tot += w;
                if covered { k_hit += w; }
            }
            i += 1;
        }

        // Concave in precision: a correct answer that adds supporting context is
        // still correct, while heavy dilution (a shotgun list of candidates, a
        // keyword dump) still collapses.
        let p_raw = if p_tot > 0.0 { clamp01(p_hit / p_tot) } else { 0.0 };
        let p = p_raw * (2.0 - p_raw);
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
            clamp01(clamp01(k_hit / k_tot) * (0.7 + 0.3 * r_all) + 0.15 * r_all * floor_scale)
        } else {
            r_all
        };

        // Precision-leaning F-beta (beta = 0.6). A correct answer is often terser
        // than the ground truth, so weighting recall equally would punish being
        // right briefly.
        let b2 = 0.36f32;
        let denom = b2 * p + r;
        let lex = if denom > 0.0 { ((1.0 + b2) * p * r) / denom } else { 0.0 };

        let ga = &mut *core::ptr::addr_of_mut!(GA);
        let gb = &mut *core::ptr::addr_of_mut!(GB);
        let cg = build_grams(gt, ga);
        let cm = build_grams(ma, gb);
        let gram = gram_similarity(ga, gb, cg, cm);

        // Adjacency of content words, reusing the same bitsets now that the
        // character similarity is in hand.
        let bg_g = build_content_bigrams(tg, ga);
        let bg_a = build_content_bigrams(ta, gb);
        let adjacency = dice(ga, gb, bg_g, bg_a);
        let cc_g = content_count(tg);
        let cc_a = content_count(ta);

        let mut raw = clamp01(0.78 * lex + 0.22 * gram);

        // Same words, different claim. "France is the capital of Paris" is a
        // perfect bag of words and a wrong answer. Word overlap cannot see the
        // difference; which content words sit next to which can.
        //
        // The test is deliberately narrow: it only fires when the answer carries
        // exactly the ground truth's content words, nothing missing and nothing
        // added, yet shares no adjacency with it. A paraphrase always drops, adds
        // or reuses some pairing, so reordering alone is not treated as a lie.
        let full_coverage = k_tot > 0.0 && k_hit >= k_tot * 0.999;
        if full_coverage && p_raw >= 0.999 && adjacency < 0.15 && cc_g >= 3 && cc_a >= 3 {
            raw *= 0.55;
        }

        // Numbers. Omitting a figure the ground truth states is incomplete;
        // stating a different one is wrong.
        let mut gt_nums = 0u32;
        let mut gt_nums_hit = 0u32;
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
            raw *= 0.62 + 0.38 * frac;
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
            if bad > 0 && gt_nums_hit < gt_nums { raw *= 0.45; }
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
        let mut hedged = 0;
        let mut c = 0;
        while c < axes.len() {
            let (pos, neg) = axes[c];
            let g = axis_sign(tg, pos, neg);
            if g == 1 || g == -1 {
                let a = axis_sign(ta, pos, neg);
                if a == CONFLICT {
                    hedged += 1;
                } else if a == 0 {
                    silent += 1;
                } else if a == g {
                    agree += 1;
                } else {
                    contra += 1;
                }
            }
            c += 1;
        }
        if contra > 0 {
            raw *= 0.15;
        } else if hedged > 0 {
            // Asserting a thing and its opposite is not an answer.
            raw *= 0.55;
        } else if agree > 0 {
            raw += (1.0 - raw) * 0.35;
        } else if silent > 0 {
            raw *= 0.85;
        } else if any_negation(tg) != any_negation(ta) {
            // No axis is decisive, but one side negates and the other does not.
            raw *= 1.0 - 0.35 * lex;
        }

        // Contrast. Pull confident matches up and near-misses down without
        // flattening the middle: a scorer whose outputs barely vary is rejected,
        // and one that is all-or-nothing cannot rank the answers in between.
        let raw = clamp01(raw);
        let smooth = raw * raw * (3.0 - 2.0 * raw);
        clamp01(0.82 * smooth + 0.18 * raw)
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
        if !any || gt.is_empty() { return 0.0; }
        if normalized_equal(gt, ma) { return 1.0; }
        score(q, gt, ma)
    }
}
