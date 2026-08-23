//! all-MiniLM-L6-v2 inference in no_std, for the CHAT_COMPLETION build only.
//!
//! The champion scorer ranks on this model's sentence embedding (its embB term, half its
//! score). Static word vectors cannot reproduce that, so this reruns the transformer:
//! wordpiece tokenise, embeddings + LayerNorm, six attention/feed-forward layers, mean-pool,
//! L2 normalise. Weights are int8 (big matrices) and f32 (biases, LayerNorm) in `minilm.bin`,
//! laid out in a fixed order this file reads. Validated to reproduce the champion's own
//! embed() at cosine ~0.97 and to agree with its ranking at Spearman ~0.91.
//!
//! Gated behind the `minilm` cargo feature so the 31 lexical builds never compile or embed it.

use core::f32;

const H: usize = 384;
const LAYERS: usize = 6;
const HEADS: usize = 12;
const HDIM: usize = H / HEADS; // 32
const INTER: usize = 1536;
const VOCAB: usize = 30522;
const MAXTOK: usize = 128; // wordpiece tokens per text, truncated; bounds the n^2 attention

/// How much text the encoder may look at before it subsamples. 1 is the original behaviour:
/// the first MAXTOK wordpieces and nothing else. Above 1 the tokeniser reads up to
/// MAXTOK * TOK_SPAN wordpieces and then keeps every k-th one, so a long answer is
/// represented by a sample spread across the whole of it instead of by its opening
/// paragraph. The sentence vector is a mean over token vectors, so a spread sample
/// estimates the vector the full text would give, and the cost stays MAXTOK^2. This matters
/// because the live champion clearly reads past 128 wordpieces (its score keeps moving as an
/// answer grows beyond that) and real miner answers run to hundreds of words, so a
/// prefix-only encoder is ranking something else.
const TOK_SPAN: usize = 1;
const TOKCAP: usize = 1024;

static MLM: &[u8] = include_bytes!("minilm.bin");

// ---- little-endian readers ----
fn u32le(o: usize) -> u32 {
    u32::from_le_bytes([MLM[o], MLM[o + 1], MLM[o + 2], MLM[o + 3]])
}
fn f32le(o: usize) -> f32 {
    f32::from_le_bytes([MLM[o], MLM[o + 1], MLM[o + 2], MLM[o + 3]])
}

// ---- no_std math ----
// exp via 2^x: e^x = 2^(x*log2 e). Range-reduce to integer + fraction, 2^int by exponent
// bits, 2^frac by a degree-3 minimax polynomial. Accurate enough for softmax and gelu.
fn fexp(x: f32) -> f32 {
    if x < -87.0 { return 0.0; }
    if x > 88.0 { return f32::from_bits(0x7f7fffff); }
    let t = x * 1.442695041; // log2(e)
    let fi = if t >= 0.0 { t as i32 } else { t as i32 - 1 };
    let f = t - fi as f32;
    // 2^f, f in [0,1)
    let p = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.0555041 + f * 0.0096181)));
    let bits = (((fi + 127) as u32) << 23) as u32;
    f32::from_bits(bits) * p
}

fn fsqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut g = if x > 1.0 { x } else { 1.0 };
    let mut i = 0;
    while i < 20 { g = 0.5 * (g + x / g); i += 1; }
    g
}

// bert gelu (erf form) via the tanh approximation, which tracks it to <1e-3.
fn gelu(x: f32) -> f32 {
    let c = 0.7978845608 * (x + 0.044715 * x * x * x);
    let e = fexp(2.0 * c);
    let tanh = (e - 1.0) / (e + 1.0);
    0.5 * x * (1.0 + tanh)
}

// ---- blob sections ----
fn vocab_count() -> usize { u32le(4) as usize }
fn tensor_base() -> usize { 8 + vocab_count() * 8 }

fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811C9DC5;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u32;
        h = h.wrapping_mul(0x01000193);
        i += 1;
    }
    h
}

fn vocab_lookup(hash: u32) -> i32 {
    let n = vocab_count();
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = u32le(8 + mid * 8);
        if k == hash { return u32le(8 + mid * 8 + 4) as i32; }
        if k < hash { lo = mid + 1; } else { hi = mid; }
    }
    -1
}

#[derive(Copy, Clone)]
struct Tn { off: usize, kind: u8, scale: f32, len: usize, soff: usize, cols: usize }

// Walk the 101 tensor headers (data skipped by arithmetic) into a fixed table.
const NTENS: usize = 5 + 16 * LAYERS;

fn tensors() -> [Tn; NTENS] {
    let mut t = [Tn { off: 0, kind: 1, scale: 0.0, len: 0, soff: 0, cols: 0 }; NTENS];
    let mut o = tensor_base();
    let mut i = 0;
    while i < NTENS {
        let kind = MLM[o]; o += 1;
        if kind == 3 {
            let rows = u32le(o) as usize; o += 4;
            let cols = u32le(o) as usize; o += 4;
            let soff = o; o += rows * 4;
            t[i] = Tn { off: o, kind, scale: 0.0, len: rows * cols, soff, cols };
            o += rows * (cols / 2);
        } else if kind == 2 {
            let rows = u32le(o) as usize; o += 4;
            let soff = o; o += rows * 4;
            let len = u32le(o) as usize; o += 4;
            t[i] = Tn { off: o, kind, scale: 0.0, len, soff, cols: 0 };
            o += len;
        } else {
            let scale = f32le(o); o += 4;
            let len = u32le(o) as usize; o += 4;
            t[i] = Tn { off: o, kind, scale, len, soff: 0, cols: 0 };
            o += if kind == 0 { len } else { len * 4 };
        }
        i += 1;
    }
    t
}

// weight element e of tensor t as f32 (per-tensor int8 dequant or raw f32)
#[inline]
fn wel(t: &Tn, e: usize) -> f32 {
    if t.kind == 0 { (MLM[t.off + e] as i8 as f32) * t.scale } else { f32le(t.off + e * 4) }
}

// weight element e (in row `row`) of tensor t. For per-row int8 (kind 2) the scale is
// looked up per row; otherwise falls back to wel.
#[inline]
fn wel_r(t: &Tn, e: usize, row: usize) -> f32 {
    if t.kind == 3 {
        let col = e - row * t.cols;
        let byte = MLM[t.off + row * (t.cols / 2) + col / 2];
        let nib = if col & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        let v = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
        (v as f32) * f32le(t.soff + row * 4)
    } else if t.kind == 2 {
        (MLM[t.off + e] as i8 as f32) * f32le(t.soff + row * 4)
    } else { wel(t, e) }
}

// y[r][o] = sum_k x[r][k]*W[o,k] + b[o], W is [out,in] row-major, over `n` rows.
fn linear(x: &[f32], n: usize, inn: usize, out: usize, w: &Tn, b: &Tn, y: &mut [f32]) {
    let mut r = 0;
    while r < n {
        let xr = r * inn;
        let mut oi = 0;
        while oi < out {
            let wb = oi * inn;
            let mut acc = if b.len > 0 { wel(b, oi) } else { 0.0 };
            let mut k = 0;
            while k < inn {
                acc += x[xr + k] * wel_r(w, wb + k, oi);
                k += 1;
            }
            y[r * out + oi] = acc;
            oi += 1;
        }
        r += 1;
    }
}

fn layernorm(x: &mut [f32], n: usize, g: &Tn, b: &Tn) {
    let mut r = 0;
    while r < n {
        let base = r * H;
        let mut mean = 0.0f32;
        let mut k = 0; while k < H { mean += x[base + k]; k += 1; }
        mean /= H as f32;
        let mut var = 0.0f32;
        k = 0; while k < H { let d = x[base + k] - mean; var += d * d; k += 1; }
        var /= H as f32;
        let inv = 1.0 / fsqrt(var + 1e-12);
        k = 0;
        while k < H {
            x[base + k] = (x[base + k] - mean) * inv * wel(g, k) + wel(b, k);
            k += 1;
        }
        r += 1;
    }
}

// ---- wordpiece tokeniser ----
fn special(tok: &[u8]) -> i32 { vocab_lookup(fnv1a(tok)) }

// Push wordpiece ids for one basic token. BERT rule: if any piece misses, the whole token
// becomes [UNK].
fn wordpiece(tok: &[u8], ids: &mut [u32], n: &mut usize, unk: i32, limit: usize) {
    if *n >= limit { return; }
    let full = vocab_lookup(fnv1a(tok));
    if full >= 0 { ids[*n] = full as u32; *n += 1; return; }
    let mut scratch = [0u8; 66];
    let mut start = 0usize;
    while start < tok.len() {
        let mut end = tok.len();
        let mut found: i32 = -1;
        while end > start {
            let mut m = 0usize;
            if start > 0 { scratch[0] = b'#'; scratch[1] = b'#'; m = 2; }
            let mut j = start;
            while j < end && m < scratch.len() { scratch[m] = tok[j]; m += 1; j += 1; }
            let id = vocab_lookup(fnv1a(&scratch[..m]));
            if id >= 0 { found = id; break; }
            end -= 1;
        }
        if found < 0 { ids[*n] = unk as u32; *n += 1; return; }
        if *n >= limit { return; }
        ids[*n] = found as u32; *n += 1;
        start = end;
    }
}

fn tokenize(text: &[u8], ids: &mut [u32], limit: usize) -> usize {
    let cls = special(b"[CLS]");
    let sep = special(b"[SEP]");
    let unk = special(b"[UNK]");
    let mut n = 0usize;
    if cls >= 0 { ids[n] = cls as u32; n += 1; }
    let mut word = [0u8; 64];
    let mut wl = 0usize;
    let flush = |word: &[u8], n: &mut usize, ids: &mut [u32]| {
        if !word.is_empty() { wordpiece(word, ids, n, unk, limit); }
    };
    let mut i = 0usize;
    while i < text.len() && n < limit - 1 {
        let mut c = text[i];
        if c >= b'A' && c <= b'Z' { c += 32; }
        let alnum = (c >= b'a' && c <= b'z') || (c >= b'0' && c <= b'9');
        if alnum {
            if wl < word.len() { word[wl] = c; wl += 1; }
        } else {
            if wl > 0 { flush(&word[..wl], &mut n, ids); wl = 0; }
            // punctuation (printable, non-space) is its own token
            if c > b' ' && c < 127 && n < limit - 1 {
                let p = [c];
                wordpiece(&p, ids, &mut n, unk, limit);
            }
        }
        i += 1;
    }
    if wl > 0 && n < limit - 1 { flush(&word[..wl], &mut n, ids); }
    if sep >= 0 && n < limit { ids[n] = sep as u32; n += 1; }
    n
}

// ---- forward pass, fixed static buffers ----
static mut IDS: [u32; MAXTOK] = [0; MAXTOK];
static mut ALLIDS: [u32; TOKCAP] = [0; TOKCAP];
static mut X: [f32; MAXTOK * H] = [0.0; MAXTOK * H];
static mut QB: [f32; MAXTOK * H] = [0.0; MAXTOK * H];
static mut KB: [f32; MAXTOK * H] = [0.0; MAXTOK * H];
static mut VB: [f32; MAXTOK * H] = [0.0; MAXTOK * H];
static mut CTX: [f32; MAXTOK * H] = [0.0; MAXTOK * H];
static mut TMP: [f32; MAXTOK * H] = [0.0; MAXTOK * H];
static mut FF: [f32; MAXTOK * INTER] = [0.0; MAXTOK * INTER];


/// Mean-pool the token rows of `x` and L2 normalise into `out`. Used for the sentence vector
/// at whichever depth is being tapped.
fn pool_norm(x: &[f32], n: usize, out: &mut [f32; H]) {
    let mut k = 0;
    while k < H {
        let mut s = 0.0f32;
        let mut r = 0; while r < n { s += x[r * H + k]; r += 1; }
        out[k] = s / n as f32;
        k += 1;
    }
    let mut nrm = 0.0f32;
    k = 0; while k < H { nrm += out[k] * out[k]; k += 1; }
    nrm = fsqrt(nrm);
    if nrm > 0.0 { k = 0; while k < H { out[k] /= nrm; k += 1; } }
}

/// Encode one text into four sentence vectors: the embedding layer (out_a), after layer 2
/// (out_2), after layer 4 (out_4) and the full six layers (out_b). The mid-depth taps exist
/// because the rival champion is a fine-tune of this architecture, and fine-tuning moves the
/// last layers most, so an earlier layer can track its final layer better than ours does.
fn encode(text: &[u8], out_a: &mut [f32; H], out_2: &mut [f32; H], out_4: &mut [f32; H],
          out_b: &mut [f32; H]) {
    unsafe {
        let ids = &mut *core::ptr::addr_of_mut!(IDS);
        let n = if TOK_SPAN <= 1 {
            tokenize(text, ids, MAXTOK)
        } else {
            let all = &mut *core::ptr::addr_of_mut!(ALLIDS);
            let mut limit = MAXTOK * TOK_SPAN;
            if limit > TOKCAP { limit = TOKCAP; }
            let m = tokenize(text, all, limit);
            if m <= MAXTOK {
                let mut i = 0; while i < m { ids[i] = all[i]; i += 1; }
                m
            } else {
                // [CLS], then MAXTOK-2 tokens spread evenly over the interior, then [SEP].
                ids[0] = all[0];
                let inner = MAXTOK - 2;
                let span = m - 2;
                let mut j = 0;
                while j < inner {
                    ids[1 + j] = all[1 + (j * span) / inner];
                    j += 1;
                }
                ids[MAXTOK - 1] = all[m - 1];
                MAXTOK
            }
        };
        if n == 0 { let mut k = 0; while k < H { out_a[k] = 0.0; out_2[k] = 0.0; out_4[k] = 0.0; out_b[k] = 0.0; k += 1; } return; }
        let t = tensors();
        let x = &mut *core::ptr::addr_of_mut!(X);
        // embeddings: word + position + token_type(0), then LayerNorm
        let mut r = 0;
        while r < n {
            let wid = ids[r] as usize;
            let mut k = 0;
            while k < H {
                x[r * H + k] = wel_r(&t[0], wid * H + k, wid) + wel_r(&t[1], r * H + k, r) + wel_r(&t[2], k, 0);
                k += 1;
            }
            r += 1;
        }
        layernorm(x, n, &t[3], &t[4]);
        // embA: mean-pool of the embedding layer (pre-transformer), the champion's shallow
        // 0.25-weight term. Captured here before the six layers run.
        pool_norm(x, n, out_a);

        let q = &mut *core::ptr::addr_of_mut!(QB);
        let k_ = &mut *core::ptr::addr_of_mut!(KB);
        let v = &mut *core::ptr::addr_of_mut!(VB);
        let ctx = &mut *core::ptr::addr_of_mut!(CTX);
        let tmp = &mut *core::ptr::addr_of_mut!(TMP);
        let ff = &mut *core::ptr::addr_of_mut!(FF);

        let mut li = 0;
        while li < LAYERS {
            let b = 5 + li * 16;
            linear(x, n, H, H, &t[b], &t[b + 1], q);
            linear(x, n, H, H, &t[b + 2], &t[b + 3], k_);
            linear(x, n, H, H, &t[b + 4], &t[b + 5], v);
            // multi-head attention -> ctx
            let scale = 1.0 / fsqrt(HDIM as f32);
            let mut h = 0;
            while h < HEADS {
                let hb = h * HDIM;
                let mut i = 0;
                while i < n {
                    // scores over keys, softmax, weighted sum of V
                    let mut mx = -1e30f32;
                    let mut sc = [0.0f32; MAXTOK];
                    let mut j = 0;
                    while j < n {
                        let mut d = 0.0f32;
                        let mut c = 0;
                        while c < HDIM { d += q[i * H + hb + c] * k_[j * H + hb + c]; c += 1; }
                        d *= scale;
                        sc[j] = d;
                        if d > mx { mx = d; }
                        j += 1;
                    }
                    let mut sum = 0.0f32;
                    j = 0; while j < n { let e = fexp(sc[j] - mx); sc[j] = e; sum += e; j += 1; }
                    let inv = 1.0 / sum;
                    let mut c = 0;
                    while c < HDIM {
                        let mut acc = 0.0f32;
                        j = 0; while j < n { acc += sc[j] * v[j * H + hb + c]; j += 1; }
                        ctx[i * H + hb + c] = acc * inv;
                        c += 1;
                    }
                    i += 1;
                }
                h += 1;
            }
            linear(ctx, n, H, H, &t[b + 6], &t[b + 7], tmp);
            let mut z = 0; while z < n * H { tmp[z] += x[z]; z += 1; }
            layernorm(tmp, n, &t[b + 8], &t[b + 9]);
            z = 0; while z < n * H { x[z] = tmp[z]; z += 1; }
            // FFN
            linear(x, n, H, INTER, &t[b + 10], &t[b + 11], ff);
            z = 0; while z < n * INTER { ff[z] = gelu(ff[z]); z += 1; }
            linear(ff, n, INTER, H, &t[b + 12], &t[b + 13], tmp);
            z = 0; while z < n * H { tmp[z] += x[z]; z += 1; }
            layernorm(tmp, n, &t[b + 14], &t[b + 15]);
            z = 0; while z < n * H { x[z] = tmp[z]; z += 1; }
            if li == 1 { pool_norm(x, n, out_2); }
            if li == 3 { pool_norm(x, n, out_4); }
            li += 1;
        }
        // mean-pool over tokens, L2 normalise -> embB (full transformer output)
        pool_norm(x, n, out_b);
    }
}

fn zero_out() -> [f32; H] { [0.0f32; H] }

fn dotc(g: &[f32; H], a: &[f32; H]) -> f32 {
    let mut d = 0.0f32;
    let mut k = 0;
    while k < H { d += g[k] * a[k]; k += 1; }
    if d < 0.0 { 0.0 } else if d > 1.0 { 1.0 } else { d }
}

// ---- one-entry-per-text memo for the four sentence vectors ----
// The node scores a whole batch of answers against the same question and the same ground
// truth, one call per answer, so the encoder is asked for the identical text over and over.
// Memoising the last few texts turns three transformer passes per call into one, which is
// most of the module's cost. Pure memoisation: same text, same vectors, no behaviour change.
const CACHE_N: usize = 4;
static mut CK: [u64; CACHE_N] = [0; CACHE_N];
static mut CV: [[[f32; H]; 4]; CACHE_N] = [[[0.0; H]; 4]; CACHE_N];
static mut CNEXT: usize = 0;

fn text_key(text: &[u8]) -> u64 {
    // FNV-1a over the bytes, with the length mixed in so a prefix cannot collide with a
    // longer text that shares it.
    let mut h: u64 = 0xcbf29ce484222325;
    let mut i = 0;
    while i < text.len() {
        h ^= text[i] as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += 1;
    }
    h ^= (text.len() as u64).wrapping_mul(0x9E3779B97F4A7C15);
    if h == 0 { 1 } else { h }
}

fn encode_cached(text: &[u8], a: &mut [f32; H], c2: &mut [f32; H], c4: &mut [f32; H],
                 b: &mut [f32; H]) {
    unsafe {
        let key = text_key(text);
        let ck = &mut *core::ptr::addr_of_mut!(CK);
        let cv = &mut *core::ptr::addr_of_mut!(CV);
        let mut i = 0;
        while i < CACHE_N {
            if ck[i] == key {
                let mut k = 0;
                while k < H {
                    a[k] = cv[i][0][k]; c2[k] = cv[i][1][k]; c4[k] = cv[i][2][k]; b[k] = cv[i][3][k];
                    k += 1;
                }
                return;
            }
            i += 1;
        }
        encode(text, a, c2, c4, b);
        let slot = *core::ptr::addr_of!(CNEXT) % CACHE_N;
        ck[slot] = key;
        let mut k = 0;
        while k < H {
            cv[slot][0][k] = a[k]; cv[slot][1][k] = c2[k]; cv[slot][2][k] = c4[k]; cv[slot][3][k] = b[k];
            k += 1;
        }
        *core::ptr::addr_of_mut!(CNEXT) = slot + 1;
    }
}

/// Every similarity the module can compute from the transformer, from one pass per text:
/// the embedding-layer cosine (shallow), the layer-2 and layer-4 cosines (mid depth), the
/// full six-layer cosine, and the answer-to-question cosine at full depth. The mid-depth
/// pair is here because the champion is a fine-tune of this architecture and fine-tuning
/// moves the last layers most, so a shallower tap can track its output better than our own
/// last layer does. Costs nothing extra: the pooling is over rows we already computed.
pub struct Sims { pub ca: f32, pub c2: f32, pub c4: f32, pub cb: f32, pub cq: f32 }

pub fn embed_sims(q: &[u8], gt: &[u8], ma: &[u8]) -> Sims {
    let mut ga = zero_out(); let mut g2 = zero_out(); let mut g4 = zero_out(); let mut gb = zero_out();
    let mut aa = zero_out(); let mut a2 = zero_out(); let mut a4 = zero_out(); let mut ab = zero_out();
    encode_cached(gt, &mut ga, &mut g2, &mut g4, &mut gb);
    encode_cached(ma, &mut aa, &mut a2, &mut a4, &mut ab);
    let ca = dotc(&ga, &aa);
    let c2 = dotc(&g2, &a2);
    let c4 = dotc(&g4, &a4);
    let cb = dotc(&gb, &ab);
    if q.is_empty() {
        return Sims { ca, c2, c4, cb, cq: cb };
    }
    let mut qa = zero_out(); let mut q2 = zero_out(); let mut q4 = zero_out(); let mut qb = zero_out();
    encode_cached(q, &mut qa, &mut q2, &mut q4, &mut qb);
    Sims { ca, c2, c4, cb, cq: dotc(&qb, &ab) }
}

/// The three-cosine form lib.rs used before the mid-depth taps existed.
pub fn embed_cos_abq(q: &[u8], gt: &[u8], ma: &[u8]) -> (f32, f32, f32) {
    let s = embed_sims(q, gt, ma);
    (s.ca, s.cb, s.cq)
}

pub fn embed_cos_ab(gt: &[u8], ma: &[u8]) -> (f32, f32) {
    let s = embed_sims(&[], gt, ma);
    (s.ca, s.cb)
}
