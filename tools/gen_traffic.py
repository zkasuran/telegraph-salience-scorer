#!/usr/bin/env python3
"""Generate a realistic CHAT_COMPLETION traffic corpus for the agreement gate.

The old proxy (bench/traffic.json) does not transfer to the node (0.63 local, 0.31
node). Real miner answers are what ten live LLMs return: mostly decent, varied in
completeness, phrasing and the occasional confident error. This asks the house
gateway to produce that spread for a set of questions, so the tuning corpus looks
like the distribution the node actually scores.

Output: bench/traffic-real.json, {"note","rows":[{"q","gt","a"}]}.
"""
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GW = "/home/asuran/Downloads/hackathon-hq/.gateway.env"


def env():
    d = {}
    for line in open(GW):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


E = env()
BASE, KEY, MODEL = E["OPENAI_BASE_URL"], E["OPENAI_API_KEY"], E.get("OPENAI_MODEL", "gpt-4o-mini")

QA = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote the novel 1984?", "George Orwell wrote 1984."),
    ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at sea level."),
    ("How many continents are there on Earth?", "There are seven continents on Earth."),
    ("What gas do plants absorb during photosynthesis?", "Plants absorb carbon dioxide during photosynthesis."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in the solar system."),
    ("In what year did World War II end?", "World War II ended in 1945."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("What language has the most native speakers worldwide?", "Mandarin Chinese has the most native speakers."),
    ("What is the speed of light in a vacuum, approximately?", "The speed of light is about 299,792 kilometres per second."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is the smallest prime number?", "The smallest prime number is 2."),
    ("What organ pumps blood through the human body?", "The heart pumps blood through the body."),
    ("What is the currency of Japan?", "The currency of Japan is the yen."),
    ("What is the tallest mountain above sea level?", "Mount Everest is the tallest mountain above sea level."),
    ("How many sides does a hexagon have?", "A hexagon has six sides."),
    ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32 degrees Fahrenheit."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    ("What is the powerhouse of the cell?", "The mitochondria is the powerhouse of the cell."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who developed the theory of general relativity?", "Albert Einstein developed general relativity."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean."),
    ("What is the square root of 144?", "The square root of 144 is 12."),
    ("What element does 'O' represent on the periodic table?", "O represents oxygen."),
    ("What is the national language of Brazil?", "The national language of Brazil is Portuguese."),
]

STYLE_PROMPT = (
    "You are simulating five different AI assistants answering the same question, the way "
    "a pool of chatbot miners would. Question: {q}\nThe correct answer is: {gt}\n"
    "Return a JSON array of exactly five strings, each a different assistant's answer:\n"
    "1. a thorough, correct answer in two or three sentences,\n"
    "2. a correct one-line answer,\n"
    "3. a correct answer that first restates the question and hedges before answering,\n"
    "4. a fluent, confident answer that is factually wrong but on the same topic,\n"
    "5. a vague partial answer that talks around the topic without clearly answering.\n"
    "Return only the JSON array, no prose."
)


def call(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.8, "max_tokens": 700}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_arr(txt):
    s = txt.find("["); e = txt.rfind("]")
    if s < 0 or e < 0:
        return []
    try:
        arr = json.loads(txt[s:e + 1])
        return [a for a in arr if isinstance(a, str) and a.strip()]
    except Exception:
        return []


def main():
    rows = []
    for i, (q, gt) in enumerate(QA):
        try:
            arr = parse_arr(call(STYLE_PROMPT.format(q=q, gt=gt)))
        except Exception as ex:
            print("call failed", i, ex)
            continue
        for a in arr:
            rows.append({"q": q, "gt": gt, "a": a.strip()})
        print(f"{i+1}/{len(QA)} {q[:40]!r} -> {len(arr)} answers")
    out = {"note": "Realistic CHAT_COMPLETION traffic: real LLM answers of varied quality per question, "
                   "generated to mimic the distribution the node's traffic gate scores.",
           "rows": rows}
    p = os.path.join(ROOT, "bench", "traffic-real.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}: {len(rows)} rows")


if __name__ == "__main__":
    main()
