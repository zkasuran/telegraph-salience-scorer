#!/usr/bin/env python3
"""Generate a WEB_SEARCH-shaped traffic corpus for the local agreement gate.

Each row is {q, gt, a}: a web-search question, an LLM-supplied ground-truth
answer (the shape the Tier-B pipeline feeds the WASM scorer), and one miner
answer of some quality. We spread answer quality on purpose (a good synthesis,
a concise correct one, a verbose one, a source-naming one, a question-parroting
one, an off-topic one, a stale/wrong one) so the ranking has something to
disagree about. Spearman only needs a spread of qualities, not labels.

Runs against the house OpenAI-compatible gateway (.gateway.env). One call per
question returns the gt plus the answer variants as JSON.
"""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

client = OpenAI(timeout=90)  # reads OPENAI_API_KEY / OPENAI_BASE_URL from env
MODEL = os.environ["OPENAI_MODEL"]
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic-websearch.json")
_lock = threading.Lock()
_rows = []

def checkpoint():
    with _lock:
        json.dump({"rows": list(_rows)}, open(DST, "w"), indent=1)

QUESTIONS = [
    "What is retrieval-augmented generation and why do agents use it?",
    "Who won the 2022 FIFA World Cup and how?",
    "What are the main causes of inflation?",
    "How does the Lightning Network work on Bitcoin?",
    "What is the James Webb Space Telescope and what has it discovered?",
    "What are the health benefits and risks of intermittent fasting?",
    "What is the difference between HTTP/2 and HTTP/3?",
    "What caused the 2008 financial crisis?",
    "How do mRNA vaccines work?",
    "What is the current state of nuclear fusion energy research?",
    "What is the Ethereum Merge and what did it change?",
    "What are large language model context windows and why do they matter?",
    "What is the greenhouse effect and how does it drive climate change?",
    "How does the DNS system resolve a domain name to an IP address?",
    "What is quantum entanglement in simple terms?",
    "What are the key provisions of the EU AI Act?",
    "How does Solana achieve high transaction throughput?",
    "What is CRISPR gene editing and what is it used for?",
    "What are the differences between TCP and UDP?",
    "What is the significance of the Higgs boson discovery?",
    "How does a transformer neural network attention mechanism work?",
    "What is zero-knowledge proof technology used for in blockchains?",
    "What are the main renewable energy sources and their tradeoffs?",
    "What is the role of the Federal Reserve in the US economy?",
    "How do self-driving cars perceive their environment?",
    "What is the difference between machine learning and deep learning?",
    "What are stablecoins and how do they maintain their peg?",
    "What is the current understanding of dark matter?",
    "How does end-to-end encryption protect messages?",
    "What is the Cambrian explosion in evolutionary history?",
    "What are the main features of the Rust programming language?",
    "How does photosynthesis convert sunlight into energy?",
    "What is a rollup in Ethereum scaling and how does it work?",
    "What are the causes and effects of ocean acidification?",
    "What is the difference between symmetric and asymmetric encryption?",
]

SYS = (
    "You produce evaluation data for a web-search answer-quality scorer. "
    "Return STRICT JSON only, no prose, no markdown fences."
)

def prompt(q):
    return f"""For this web-search question, produce a ground-truth answer and several miner answers of DIFFERENT quality.

Question: {q}

Return JSON with this exact shape:
{{
  "gt": "<a good, accurate, concise 2-4 sentence answer that a strong web-search agent would give, naming a source or specific facts>",
  "answers": [
    "<A: an excellent, accurate synthesized answer, phrased DIFFERENTLY from gt (low word overlap), that names concrete facts or sources>",
    "<B: a correct but very concise one-sentence answer>",
    "<C: a correct answer that is verbose and padded with tangential detail>",
    "<D: an answer that mostly restates/parrots the question with little new information>",
    "<E: a plausible-sounding but factually WRONG or outdated answer on the same topic>",
    "<F: an off-topic answer about a different subject entirely>"
  ]
}}
Every answer must be plain text. Do not label them with letters inside the text."""

def one(i, q):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt(q)}],
                temperature=0.7,
            )
            txt = r.choices[0].message.content.strip()
            if txt.startswith("```"):
                txt = txt.split("```", 2)[1]
                if txt.startswith("json"):
                    txt = txt[4:]
            obj = json.loads(txt)
            gt = obj["gt"].strip()
            local = [{"q": q, "gt": gt, "a": a.strip()} for a in obj["answers"] if a.strip()]
            with _lock:
                _rows.extend(local)
            checkpoint()
            print(f"[{i+1}/{len(QUESTIONS)}] {q[:46]} -> +{len(local)}", file=sys.stderr, flush=True)
            return
        except Exception as e:
            print(f"  retry {attempt} {q[:36]}: {e}", file=sys.stderr, flush=True)
            time.sleep(2)

def main():
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, i, q) for i, q in enumerate(QUESTIONS)]
        for _ in as_completed(futs):
            pass
    checkpoint()
    print(f"wrote {len(_rows)} rows to {DST}")

if __name__ == "__main__":
    main()
