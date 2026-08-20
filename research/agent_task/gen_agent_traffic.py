#!/usr/bin/env python3
"""Generate a realistic AGENT_TASK traffic corpus for the agreement gate.

AGENT_TASK on Telegraph is served by LLM miners (DeepSeek, Nova, Qwen, Kimi, a
Telegraph knowledge chatbot). Their answers to an agentic prompt are long,
structured, multi-step plans of varied quality. The traffic gate ranks those
answers against the champion's ranking, so the tuning corpus has to look like
the distribution the node scores: mostly on-topic multi-step plans that differ
in completeness, ordering, correctness and verbosity, not short factual QA.

This asks the house gateway to produce that spread for a set of agentic tasks.

Output: agent-traffic.json, {"note","rows":[{"q","gt","a"}]}.
"""
import json
import os
import urllib.request

ROOT = "/home/asuran/Downloads/hackathon-hq"
GW = os.path.join(ROOT, ".gateway.env")


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

# Agentic, multi-step tasks. The ground truth is a compact correct plan naming
# the steps that matter, the shape a strong agent answer takes.
QA = [
    ("Plan the steps for an autonomous agent to deploy a Node.js web app to a fresh Ubuntu server with a PostgreSQL database and HTTPS.",
     "Provision the Ubuntu server and open ports 80/443; install Node.js, npm and PostgreSQL; create the database, user and schema; pull the app code and install dependencies; set environment variables and run migrations; start the app under a process manager like pm2 or systemd; put nginx in front as a reverse proxy; obtain a TLS certificate with certbot and enable HTTPS; verify the endpoint responds."),
    ("An agent must automate collecting daily sales CSV attachments from an email inbox and loading them into a warehouse table. Describe the workflow.",
     "Connect to the inbox over IMAP and search for unread messages matching the sender or subject; download the CSV attachments; validate the header and row schema; transform and clean the rows; load them into the warehouse table idempotently, deduplicating on a date or id key; mark the emails read or move them to a processed folder; log the row counts and alert on failures; schedule the job to run daily."),
    ("Outline how an agent should research a topic on the web and produce a cited summary.",
     "Break the topic into sub-questions; issue web searches for each; fetch and read the most relevant, reputable sources; extract the key claims with their source URLs; cross-check facts across at least two sources; synthesize a structured summary; attach inline citations to each claim; note gaps or conflicting evidence."),
    ("Describe the steps an agent takes to triage and fix a failing unit test in a Python repository.",
     "Read the failing test output and the stack trace; locate the test and the code under test; reproduce the failure locally; form a hypothesis about the root cause; inspect recent changes with git blame or the diff; write or adjust the fix; run the single test, then the full suite and the linter; confirm all green; commit with a message describing the fix."),
    ("Plan an agent workflow to book the cheapest one-way flight from London to Tokyo next month and add it to the user's calendar.",
     "Clarify the exact date window and any airline or time constraints; query a flight search API across the date range; filter by price, duration and stops; pick the cheapest acceptable option; confirm the fare and baggage terms with the user before purchase; complete the booking and capture the confirmation; create a calendar event with the flight times, terminal and confirmation code; send the user a summary."),
    ("How should an agent monitor a website for changes and notify the user when a product comes back in stock?",
     "Record the target URL and the CSS selector or text that marks stock status; poll the page on an interval with backoff; parse the current status; compare against the last known state; on a change from out-of-stock to in-stock, send a notification through the chosen channel; store the new state; handle request failures, rate limits and layout changes gracefully; stop when the user cancels."),
    ("Describe how an agent migrates a database schema in production with zero downtime.",
     "Take a backup and verify it restores; write a backward-compatible migration that adds new columns or tables without dropping the old ones; deploy the migration; deploy application code that writes to both old and new shapes; backfill existing rows in batches; switch reads to the new shape; verify data integrity; remove the old columns and dual writes in a later release; keep a rollback path at each step."),
    ("Plan the steps for an agent to summarize a 40-page PDF report and extract every dollar figure into a table.",
     "Load and parse the PDF into text per page; chunk the text to fit the context window; summarize each chunk, then combine into an overall summary; run a numeric extraction pass that captures every dollar figure with its surrounding label and page number; deduplicate and normalize the figures; assemble them into a table; validate the totals against the document; return the summary and the table."),
    ("An agent is asked to set up CI for a new GitHub repository. What steps does it take?",
     "Detect the language and build tool from the repo; add a CI config for the platform (a GitHub Actions workflow); define jobs to install dependencies, run the build, run tests and run the linter on push and pull request; cache dependencies to speed runs; add a status badge; open a pull request with the workflow; confirm the first run passes and fix any failures."),
    ("Describe how an agent should handle a multi-step customer refund request end to end.",
     "Authenticate the customer and load the order; verify the item is within the refund window and eligible; check the payment method and original charge; calculate the refund amount including or excluding shipping per policy; issue the refund through the payment processor; update the order and inventory records; email the customer a confirmation with the refund reference and timeline; log the action for audit."),
    ("Plan an agent workflow to scrape job listings from several sites, deduplicate them and email a daily digest.",
     "For each source, fetch the listings pages and parse title, company, location, link and date; normalize the fields into one schema; deduplicate across sources by title plus company or a canonical URL; filter by the user's keywords and location; rank or sort by date; render a digest email; send it once per day; persist seen listings so only new ones appear next time; handle site failures without aborting the whole run."),
    ("How does an agent provision and configure a Redis cache for an existing API, then verify it works?",
     "Provision a Redis instance and secure it with a password and network rules; add the connection settings to the API's config as environment variables; add a cache client and wrap the hot read paths with get-or-set logic and sensible TTLs; add cache invalidation on writes; deploy the change; measure latency and hit rate; confirm correctness by comparing cached and uncached responses; document the keys and TTLs."),
    ("Describe the steps for an agent to convert a folder of images to WebP, resize them and upload to object storage.",
     "Enumerate the image files in the folder; for each, read it, resize to the target dimensions preserving aspect ratio, and encode to WebP at a chosen quality; write the output with a consistent naming scheme; upload each to the object storage bucket with the right content type and cache headers; verify the upload and record the public URL; skip files already processed; report a summary of counts and any errors."),
    ("Plan how an agent should debug an intermittent 500 error in a production web service.",
     "Reproduce or capture the error from logs and error tracking; identify the affected endpoint and the request pattern that triggers it; correlate with recent deploys, traffic spikes or dependency outages; inspect stack traces and add targeted logging if needed; form and test a hypothesis in staging; ship a fix or a mitigation like a timeout or retry; monitor the error rate after deploy; write a short postmortem."),
    ("An agent must onboard a new employee: create accounts, assign hardware and schedule training. Outline the plan.",
     "Gather the new hire's role, start date and manager; create accounts in the identity provider, email and the required SaaS tools with role-based access; order or assign a laptop and peripherals and track shipping; add them to the right groups, calendars and channels; schedule orientation and role-specific training sessions; assign a buddy; send a welcome packet; verify access on day one and follow up on anything pending."),
    ("Describe how an agent generates a weekly performance report from an analytics API and posts it to Slack.",
     "Authenticate to the analytics API; query the metrics for the reporting window and the prior window for comparison; compute the deltas and any KPIs; detect notable changes or anomalies; render a concise report with the numbers and a short narrative; format it for Slack with sections and highlights; post it to the target channel; handle API errors and retries; schedule it to run weekly."),
    ("Plan an agent workflow to translate a website's content into three languages and keep translations in sync.",
     "Extract the source strings into a resource file with stable keys; send each string for machine translation into the three target languages; have the translations reviewed or spot-checked; store them keyed by locale; wire the app to serve the right locale; on any source string change, mark the affected translations stale and re-translate only those; test each locale renders correctly; track coverage per language."),
    ("How should an agent set up automated database backups with offsite storage and restore testing?",
     "Choose a backup schedule and retention policy; script a consistent dump of the database; compress and encrypt the dump; upload it to offsite object storage in a separate account or region; verify each upload and prune backups past retention; on a regular cadence, restore the latest backup into a scratch environment and run integrity checks; alert on any failed backup or failed restore; document the runbook."),
    ("Describe the steps an agent follows to A/B test a new checkout button and decide a winner.",
     "Define the hypothesis and the primary metric such as conversion rate; split traffic randomly and evenly between the control and the variant; implement the two versions behind a feature flag; run the experiment until it reaches a pre-set sample size; collect the metric per variant; test the difference for statistical significance; check guardrail metrics for regressions; roll out the winner to everyone or keep the control; document the result."),
    ("Plan how an agent should ingest a user's meeting recording, transcribe it and produce action items.",
     "Accept the audio or video file and validate the format; transcribe it to text with timestamps and speaker labels; clean and segment the transcript; summarize the discussion; extract action items with owners and due dates where stated; extract decisions and open questions; format the summary and the action items; deliver them to the user and optionally create tasks in a tracker; keep the raw transcript available."),
    ("An agent must diagnose why a scheduled cron job stopped running and restore it. Outline the steps.",
     "Check whether the scheduler is running and the job is still registered; read the job's own logs and the system scheduler logs for the last successful and first failed run; verify the script path, permissions and interpreter still exist; check for environment or credential changes and disk or quota limits; reproduce the job manually; fix the root cause; re-enable the job; confirm the next scheduled run succeeds and add alerting on failure."),
    ("Describe how an agent should set up rate limiting for a public API and verify it under load.",
     "Choose the limit and the window per client key and the algorithm such as token bucket; implement it in middleware backed by a shared store like Redis so limits hold across instances; return the standard 429 status with retry-after and rate-limit headers; exempt or raise limits for trusted keys; deploy behind a flag; run a load test that exceeds the limit and confirm requests are throttled and recover; monitor rejection rates."),
    ("Plan an agent workflow to reconcile two lists of transactions and report the discrepancies.",
     "Load both transaction lists and normalize the fields (date, amount, id, description) into one schema; match records on a stable key or a fuzzy combination of amount and date; classify each as matched, only-in-A, only-in-B or amount-mismatched; total each category; produce a report listing the discrepancies with enough detail to investigate; flag likely causes such as timing or fees; export the reconciled result."),
    ("How does an agent take a plain-language feature request and turn it into a merged pull request?",
     "Clarify the requirement and acceptance criteria; locate the relevant code and read it; design the change and note the files to touch; write the implementation following the repo's conventions; add or update tests; run the build, tests and linter until green; write a focused commit and a pull request describing what changed and why; respond to review feedback and update the branch; merge once approved and checks pass."),
]

STYLE_PROMPT = (
    "You are simulating five different AI agent miners answering the same agentic task, "
    "the way a pool of autonomous-agent models would. Each returns a multi-step plan or "
    "execution description, not a one-liner.\n"
    "Task: {q}\nA strong reference answer is: {gt}\n"
    "Return a JSON array of exactly five strings, each a different agent's answer:\n"
    "1. a thorough, correct multi-step plan that covers the key steps in a sensible order,\n"
    "2. a correct but terse plan that lists the main steps briefly,\n"
    "3. a correct plan that is verbose and restates the task and hedges before the steps,\n"
    "4. a fluent, confident plan that is on-topic but has a wrong or unsafe step or a bad ordering,\n"
    "5. a vague partial answer that gestures at the task without giving concrete steps.\n"
    "Return only the JSON array, no prose."
)


def call(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.8, "max_tokens": 1000}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
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


def one(item):
    i, (q, gt) = item
    try:
        arr = parse_arr(call(STYLE_PROMPT.format(q=q, gt=gt)))
    except Exception as ex:
        print("call failed", i, ex, flush=True)
        return []
    print(f"{i+1}/{len(QA)} {q[:48]!r} -> {len(arr)} answers", flush=True)
    return [{"q": q, "gt": gt, "a": a.strip()} for a in arr]


def main():
    from concurrent.futures import ThreadPoolExecutor
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-traffic.json")
    rows = []
    have = set()
    if os.path.exists(p):
        prev = json.load(open(p)).get("rows", [])
        rows.extend(prev)
        have = {r["q"] for r in prev}
    todo = [it for it in enumerate(QA) if it[1][0] not in have]
    print(f"resuming: {len(have)} tasks done, {len(todo)} to do", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one, todo):
            rows.extend(r)
    out = {"note": "Realistic AGENT_TASK traffic: real LLM agent answers of varied quality per "
                   "multi-step task, generated to mimic the distribution the node's traffic gate scores.",
           "rows": rows}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-traffic.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}: {len(rows)} rows")


if __name__ == "__main__":
    main()
