#!/usr/bin/env python3
"""Generate a realistic LANGUAGE_GENERATION traffic corpus.

LANGUAGE_GENERATION is open-ended: "explain", "describe", "write about". Real miner
answers span a wide length/detail range, from a terse instant model (groq, ranked
last by the champion) to a verbose grounded assistant (telegraph-chatbot, ranked
first). This asks the house gateway for that spread: for each prompt, five answers
at increasing detail. Used to check whether the candidate's ranking tracks the
length/detail axis the champion appears to rank on.

Output: bench/traffic-langgen.json  {"note","rows":[{"q","gt","a"}]}.
"""
import json, os, urllib.request, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GW = "/home/asuran/Downloads/hackathon-hq/.gateway.env"

def env():
    d = {}
    for line in open(GW):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); d[k] = v
    return d

E = env()
BASE, KEY, MODEL = E["OPENAI_BASE_URL"], E["OPENAI_API_KEY"], E.get("OPENAI_MODEL", "gpt-4o-mini")

PROMPTS = [
    ("Explain what a blockchain is.", "A blockchain is a distributed, append-only ledger of transactions grouped into cryptographically linked blocks and agreed on by a network through a consensus mechanism, giving a tamper-resistant record without a central authority."),
    ("Describe how photosynthesis works.", "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to produce glucose and oxygen, capturing light energy in chloroplasts and converting it to chemical energy."),
    ("What causes the seasons on Earth?", "The seasons are caused by the tilt of the Earth's axis relative to its orbit, which changes how directly sunlight strikes each hemisphere through the year."),
    ("Explain the difference between TCP and UDP.", "TCP is connection-oriented and reliable, guaranteeing ordered delivery with acknowledgements, while UDP is connectionless and unreliable but lower-latency, sending datagrams without handshakes or retransmission."),
    ("Describe the water cycle.", "The water cycle moves water through evaporation, condensation, precipitation and collection, cycling it between the oceans, atmosphere and land."),
    ("Why is the sky blue?", "The sky is blue because air molecules scatter shorter blue wavelengths of sunlight more than longer red ones, a process called Rayleigh scattering."),
    ("Explain what machine learning is.", "Machine learning is a field of AI in which systems learn patterns from data to make predictions or decisions without being explicitly programmed for each case."),
    ("How does a vaccine work?", "A vaccine trains the immune system by exposing it to a harmless piece or form of a pathogen, so the body can recognise and respond quickly to the real infection later."),
    ("Describe how an internal combustion engine works.", "An internal combustion engine burns a fuel-air mixture inside cylinders; the expanding gases drive pistons whose motion is converted by a crankshaft into rotational power."),
    ("Explain the theory of supply and demand.", "Supply and demand describes how the price of a good settles where the quantity buyers want equals the quantity sellers offer, with prices rising when demand outstrips supply and falling when it does not."),
    ("What is climate change?", "Climate change is the long-term shift in global temperatures and weather patterns, driven largely by human greenhouse-gas emissions that trap heat in the atmosphere."),
    ("Describe the structure of an atom.", "An atom has a dense nucleus of positively charged protons and neutral neutrons, surrounded by negatively charged electrons occupying regions called orbitals."),
    ("Explain how the internet routes data.", "The internet splits data into packets that are forwarded hop by hop between routers using IP addresses, each router choosing a next hop toward the destination until the packets are reassembled."),
    ("What is inflation in economics?", "Inflation is a sustained rise in the general price level of goods and services, which reduces the purchasing power of a unit of currency over time."),
    ("Describe how DNA stores genetic information.", "DNA stores information in the sequence of four bases (A, T, C, G) along a double helix; triplets of bases code for amino acids, which are assembled into the proteins that run the cell."),
]

LEVELS = [
    "in a single short sentence, terse",
    "in two plain sentences",
    "in one clear paragraph of about four sentences",
    "in two well-developed paragraphs with examples",
    "in a thorough, detailed multi-paragraph explanation with examples and context",
]

def call(prompt, style):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question " + style + ". Do not use markdown headers."},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last = e
            import time; time.sleep(2 * (attempt + 1))
    raise last

rows = []
for q, gt in PROMPTS:
    for style in LEVELS:
        try:
            a = call(q, style)
            if a:
                rows.append({"q": q, "gt": gt, "a": a})
                print(f"  [{len(a.split()):>3}w] {q[:40]}", file=sys.stderr)
        except Exception as e:
            print("  ERR", q[:30], e, file=sys.stderr)

out = os.path.join(ROOT, "bench", "traffic-langgen.json")
json.dump({"note": "Realistic LANGUAGE_GENERATION traffic: open-ended answers at five detail levels per prompt, house-gateway generated, to mimic the terse-to-verbose spread the champion ranks.", "rows": rows}, open(out, "w"), indent=1)
print(f"wrote {len(rows)} rows to {out}")
