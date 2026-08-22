#!/usr/bin/env python3
"""Build a per-intent traffic corpus that looks like the rows the node's agreement
gate scores.

The node's miners on the five contested intents are LLM proxies (Nova, DeepSeek,
Qwen, Kimi, a chatbot), so the rows it ranks are several fluent answers to the same
request, all roughly right, plus the occasional miner that returns nothing. Our old
proxy (bench/traffic-real.json) is one broad CHAT_COMPLETION set with wide quality
spread, which is a much easier ranking problem than the real one: agreement measured
on it is optimistic. This asks several different models the same request, per intent,
so the local corpus has the same tight clusters the real gate does.

    python3 tools/gen_intent_traffic.py bench/traffic-<intent>.json AGENT_TASK

Writes {"note":..., "rows":[{"q","gt","a","model"}...]}.
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = ["gpt-4o-mini", "gemini-2.5-flash", "deepseek-v4-flash",
          "qwen3.6-27b", "kimi-k2.5", "glm-5.1"]

# A dead miner returns nothing or an apology, which is why several miners sit at
# score 0.0000 on the node's leaderboard. Those rows are part of the ranking.
DEAD = ["", "   ", "I'm sorry, I can't help with that.", "Error: upstream timeout",
        "As an AI language model, I do not have access to that information."]

TASKS = {
    "AGENT_TASK": [
        ("Plan a three step rollout for a new feature flag across staging, canary and production.",
         "Enable the flag in staging and run the integration suite, then turn it on for a 5% canary in production while watching error rate and latency, then ramp to 100% once the canary is clean for an hour."),
        ("Draft a two sentence status update for a project that slipped one week after a failed database migration.",
         "The release moved out by a week because the database migration failed on the production replica and had to be rolled back. The migration has been fixed and retested in staging, and the new ship date is next Friday."),
        ("Turn this requirement into three acceptance criteria: users must be able to reset a password by email.",
         "A user who submits a registered email address receives a reset link within one minute. The link expires after 30 minutes and can only be used once. After a successful reset the user is signed out of all other sessions."),
        ("A customer reports the checkout page is slow. List the first four things you would check, in order.",
         "Check the browser timing waterfall for the slowest request, then the checkout API's own latency and error rate, then the database queries that endpoint runs, then any third party payment or fraud call it blocks on."),
        ("Write a shell one liner that finds every file over 100MB under the current directory, largest first.",
         "find . -type f -size +100M -printf '%s\\t%p\\n' | sort -rn"),
        ("Summarise what to do when a deploy fails its health check, in three steps.",
         "Roll back to the previous revision so traffic is served, keep the failed pods and their logs for diagnosis, then reproduce the health check locally against the new image before redeploying."),
        ("Break down 'add rate limiting to the public API' into four tasks with an owner role for each.",
         "Pick the limit and window and document them (product), add a token bucket keyed on API key in the gateway (backend), return 429 with a Retry-After header and cover it in tests (backend), then add a dashboard and an alert for rejected requests (platform)."),
        ("Given a failing test named test_invoice_total_rounds_half_up, what is the most likely bug?",
         "The invoice total is being rounded with banker's rounding rather than half up, so a value like 2.5 becomes 2 instead of 3."),
        ("Write a two line commit message for a change that fixes a null pointer when a user has no avatar.",
         "Fix crash when a user has no avatar\n\nThe profile card dereferenced the avatar URL without checking for null, so any account that never uploaded a picture crashed the page."),
        ("Schedule three tasks that each take a day across two people, with one dependency: B needs A.",
         "Day one, person one does A while person two does C. Day two, person one does B, which needed A. Everything is finished by the end of day two."),
    ],
    "LANGUAGE_GENERATION": [
        ("Write a one line tagline for a bike lock that unlocks with a phone.",
         "Your phone is the key."),
        ("Rewrite this in plain English: 'Utilisation of the aforementioned methodology facilitates optimisation.'",
         "Using that method makes things work better."),
        ("Write two sentences of release notes for a search feature that now handles typos.",
         "Search now finds the right result even when a word is misspelled. Queries with one or two wrong letters return the same results as the correct spelling."),
        ("Write a polite one paragraph reply declining a meeting invitation for Thursday.",
         "Thanks for the invitation. I have a conflict on Thursday and will not be able to join, but I am happy to read the notes afterwards or meet later in the week if that helps."),
        ("Write a haiku about a slow database query.",
         "One row at a time\nthe index sits unused, cold\nthe timeout arrives"),
        ("Write a two sentence product description for a stainless steel water bottle that keeps drinks cold for 24 hours.",
         "A stainless steel bottle with a vacuum wall that keeps drinks cold for a full day. It fits a standard cup holder and the lid seals tight enough to go in a bag."),
        ("Write an email subject line for a password expiry warning in under eight words.",
         "Your password expires in three days"),
        ("Turn these notes into one sentence: 'server down 3am, disk full, cleared logs, back up 3:20am'.",
         "The server went down at 3am because the disk was full, and clearing old logs brought it back at 3:20am."),
        ("Write a friendly one line push notification for an abandoned shopping cart.",
         "Still thinking it over? Your cart is waiting."),
        ("Describe the colour of a sunset to someone who has never seen one, in two sentences.",
         "The sky turns the warm orange of a lit lamp near the horizon and fades to deep blue overhead. The colours change slowly, so the bright part looks like it is spreading and then cooling."),
    ],
    "TASK_COMPLETION": [
        ("Sort this list ascending and return only the list: [3, 1, 2, 10, 7].",
         "[1, 2, 3, 7, 10]"),
        ("Extract the total from this invoice text: 'Subtotal 90.00, tax 10.00, total 100.00 USD'.",
         "100.00 USD"),
        ("Convert 25 degrees Celsius to Fahrenheit and give only the number.",
         "77"),
        ("From this list of emails pick the one on the example.com domain: a@test.io, b@example.com, c@mail.net.",
         "b@example.com"),
        ("Count the words in this sentence and answer with a number only: 'the quick brown fox jumps'.",
         "5"),
        ("Turn this date into ISO 8601: March 4th 2026.",
         "2026-03-04"),
        ("Given prices 12.50, 3.25 and 8.00, return the sum with two decimals.",
         "23.75"),
        ("Write a regular expression that matches a UK postcode like SW1A 1AA.",
         "^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$"),
        ("Return this JSON with the name field uppercased: {\"name\": \"ada\", \"age\": 36}.",
         "{\"name\": \"ADA\", \"age\": 36}"),
        ("Deduplicate this list, keeping order: [a, b, a, c, b].",
         "[a, b, c]"),
    ],
    "WEB_SEARCH": [
        ("What is the current stable major release of PostgreSQL?",
         "PostgreSQL 18 is the current stable major release."),
        ("What is the official documentation URL for the Rust standard library?",
         "https://doc.rust-lang.org/std/"),
        ("Who maintains the wazero WebAssembly runtime and under what licence is it released?",
         "wazero is maintained by Tetrate and released under the Apache 2.0 licence."),
        ("What does the HTTP status code 429 mean?",
         "429 means Too Many Requests: the client has sent too many requests in a given amount of time and should back off, usually after the Retry-After header."),
        ("Which company develops the Base layer 2 network?",
         "Base is developed by Coinbase, built on the OP Stack."),
        ("What is the maximum file size GitHub allows in a repository?",
         "GitHub blocks pushes containing files larger than 100 MB, and warns above 50 MB."),
        ("What licence does the all-MiniLM-L6-v2 sentence transformer use?",
         "all-MiniLM-L6-v2 is released under the Apache 2.0 licence."),
        ("What is the default port for a PostgreSQL server?",
         "5432"),
        ("What is the difference between HTTP GET and HEAD?",
         "HEAD is identical to GET except the server returns only headers and no response body, so it is used to check whether a resource exists or has changed."),
        ("What is the current version of the Ethereum execution API specification called?",
         "The Ethereum execution API is specified in the execution-apis repository, published as the JSON-RPC specification."),
    ],
    "WEATHER_FORECAST": [
        ("What is the weather forecast for Berlin tomorrow?",
         "Tomorrow in Berlin: partly cloudy with a high near 24C and a low of 14C, a 20% chance of rain and light winds around 15 km/h."),
        ("Give the three day forecast for Tokyo.",
         "Tokyo: tomorrow 27C and humid with scattered showers, the day after 29C and mostly sunny, then 26C with a 60% chance of rain."),
        ("Will it rain in London this weekend?",
         "Yes, rain is likely in London on Saturday afternoon, around 70% chance, with Sunday drier and cloudy at 18C."),
        ("What is the forecast high and low for Denver on Friday?",
         "Denver on Friday: high near 31C, low near 15C, sunny with afternoon thunderstorms possible."),
        ("Is a storm expected in Miami in the next 48 hours?",
         "No named storm is expected in Miami in the next 48 hours, though scattered thunderstorms are likely each afternoon."),
        ("What is the wind forecast for Cape Town tomorrow morning?",
         "Cape Town tomorrow morning: south easterly wind at 35 km/h gusting to 55, easing by midday."),
        ("How much snow is forecast for Oslo over the next two days?",
         "Oslo is forecast 8 to 12 cm of snow over the next two days, most of it falling tomorrow night."),
        ("What is the forecast for Sydney on the weekend?",
         "Sydney: Saturday 23C and mostly sunny, Sunday 21C with morning showers clearing in the afternoon."),
        ("Will temperatures drop below freezing in Chicago this week?",
         "Yes, Chicago drops below freezing on Wednesday night, with a low near minus 3C."),
        ("What is the rain chance for Mumbai tomorrow?",
         "Mumbai has an 85% chance of rain tomorrow with heavy showers likely in the afternoon and around 40 mm of rainfall."),
    ],
}

SYS = ("You are a miner in an open answer network. Answer the request directly. "
       "No preamble, no markdown headers, no bullet lists unless the request asks for a list.")

# The node's miners are hosted LLMs answering with whatever their default style is, which
# is long: headers, bullets, caveats, a summary at the end. VERBOSE=1 generates that shape,
# because a 400 word answer and a 20 word answer are not the same ranking problem for a
# scorer that only reads its first 128 wordpieces.
if os.environ.get("VERBOSE"):
    SYS = ("You are a helpful assistant answering a user request. Be thorough: explain your "
           "reasoning, use markdown headings and bullet lists, cover edge cases, and finish "
           "with a short summary. Aim for 300 to 500 words.")


def ask(model, q):
    import urllib.request
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ["OPENAI_BASE_URL"]
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": SYS}, {"role": "user", "content": q}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        return d["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"__ERR__ {e}"


def main():
    out_path, intent = sys.argv[1], sys.argv[2]
    tasks = TASKS[intent]
    jobs = [(q, gt, m) for q, gt in tasks for m in MODELS]
    rows = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        answers = list(ex.map(lambda j: ask(j[2], j[0]), jobs))
    for (q, gt, m), a in zip(jobs, answers):
        if a.startswith("__ERR__"):
            print("  err", m, a[:120]); continue
        rows.append({"q": q, "gt": gt, "a": a, "model": m})
    # a few dead miners per intent, the way the leaderboard shows them
    for i, (q, gt) in enumerate(tasks[:len(DEAD)]):
        rows.append({"q": q, "gt": gt, "a": DEAD[i], "model": "dead"})
    json.dump({"note": f"{intent} traffic proxy: {len(MODELS)} models answering the same "
                       f"requests, plus dead miners, to match the tight per-request clusters "
                       f"the node's agreement gate ranks.", "rows": rows},
              open(out_path, "w"), indent=1)
    print(f"{intent}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
