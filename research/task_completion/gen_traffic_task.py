#!/usr/bin/env python3
"""Generate an intent-fitted TASK_COMPLETION traffic corpus for the agreement gate.

TASK_COMPLETION intent (node canonical description): "Query asks about what makes an
AI agent effective at completing multi-step tasks, or is itself a request to complete
a defined multi-step task end-to-end." So the questions are either meta ("what makes
an agent good at X") or an actual multi-step task to carry out. Ground truth is a
thorough, correct, step-structured answer. The five per-question variants reproduce
the spread of real miner answers the node's traffic gate scores.

Output: bench/traffic-task.json, {"note","rows":[{"q","gt","a"}]}.
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
    ("What makes an AI agent effective at completing multi-step tasks end to end?",
     "An effective multi-step agent decomposes the goal into ordered subtasks, tracks state and intermediate results between steps, selects and calls the right tools, checks each step's output before continuing, recovers from errors by retrying or replanning, and stops with a verified final result."),
    ("Outline the steps to migrate a PostgreSQL database to MySQL.",
     "First audit the schema and data types, then map PostgreSQL types to MySQL equivalents, export the schema and convert incompatible constructs, migrate the data in batches, port stored procedures and sequences, update application connection strings and SQL dialect, then validate row counts and run the app's test suite against the new database."),
    ("Describe how to set up a CI pipeline that builds, tests and deploys a web app.",
     "Add a config file for the CI service, define a build stage that installs dependencies and compiles, a test stage that runs unit and integration tests and fails on any error, a package stage that produces an artifact or image, and a deploy stage gated on the tests passing that pushes to the target environment with secrets injected from the CI store."),
    ("What steps should an agent take to book a multi-city flight itinerary for a user?",
     "Collect the cities, dates and constraints, search each leg for candidate flights, filter by budget and layover limits, assemble a consistent itinerary that connects in time, confirm price and seat availability, present the option for approval, then complete the booking and return the confirmation numbers."),
    ("Complete this task: plan a data pipeline that ingests CSV files, cleans them and loads them into a warehouse.",
     "Land the raw CSVs in object storage, validate the schema and reject malformed rows to a dead-letter path, clean and normalise types and deduplicate, transform to the warehouse model, load incrementally with idempotent upserts, then run data-quality checks and record load metadata for lineage."),
    ("How does an agent effectively break a large coding task into smaller steps?",
     "It clarifies the requirement, sketches the target design, splits the work into independent units with clear interfaces, orders them so each builds on tested foundations, implements one unit at a time with tests, integrates and runs the whole suite, then reviews the diff against the original requirement."),
    ("Walk through the steps to deploy a machine learning model as a REST API.",
     "Serialise the trained model, wrap inference in a service with input validation, containerise it with pinned dependencies, add health and prediction endpoints, load-test and set resource limits, deploy behind a load balancer with autoscaling, then add monitoring for latency, errors and prediction drift."),
    ("What steps are involved in responding to a production outage?",
     "Acknowledge the alert and declare an incident, assess scope and impact, stabilise by rolling back or failing over, communicate status to stakeholders, identify the root cause, apply and verify a fix, confirm recovery, then write a blameless postmortem with action items."),
    ("Complete the task: organise a two-day technical conference from scratch.",
     "Set the goal, budget and dates, secure a venue and catering, build the call for talks and select a program, open registration and market it, arrange AV, signage and volunteers, run the event with a schedule and a green room, then gather feedback and reconcile the budget afterward."),
    ("Describe the end-to-end steps to onboard a new engineer effectively.",
     "Provision accounts and hardware before day one, assign a buddy and a starter task, walk through the codebase and deployment flow, set clear 30-60-90 day goals, schedule regular check-ins, give early code review, then confirm they can ship a change independently by the end of the first month."),
    ("What makes an agent reliable when a step in a task fails?",
     "It detects the failure from the step's output rather than assuming success, distinguishes transient from permanent errors, retries transient ones with backoff, replans or picks an alternative tool for permanent ones, avoids repeating a failing action in a loop, and surfaces the problem clearly if it cannot recover."),
    ("Outline how to conduct a security audit of a web application.",
     "Scope the assets and threat model, review authentication and session handling, test input validation for injection and XSS, check access control and privilege escalation, review dependencies for known CVEs, inspect secrets management and transport security, then report findings ranked by severity with remediation steps."),
    ("Complete this multi-step task: turn a rough research idea into a published blog post.",
     "Clarify the core claim and audience, gather and verify sources, draft an outline, write the sections, edit for clarity and cut filler, add examples and visuals, proofread, then publish and share with the intended channels."),
    ("How should an agent plan and execute a web-scraping task end to end?",
     "Confirm the target is allowed by robots and terms, identify the pages and the data fields, fetch with rate limiting and retries, parse the structure and extract fields, handle pagination and missing values, store the cleaned records, then validate coverage against a sample and schedule re-runs."),
    ("Describe the steps to refactor a large legacy function safely.",
     "Cover the existing behaviour with characterisation tests, identify seams and extract small pieces one at a time, keep the tests green after each extraction, rename for clarity, remove duplication, simplify the control flow, then confirm behaviour is unchanged and the suite still passes."),
    ("What steps make an agent effective at a long-running research task?",
     "It frames a precise question, plans the sub-questions, searches and reads primary sources, takes structured notes tied to citations, synthesises findings while tracking uncertainty, checks claims against multiple sources, then produces a sourced answer and flags what remains unknown."),
    ("Complete the task: set up automated daily backups for a database with restore testing.",
     "Choose full plus incremental backups, script a dump on a daily schedule, encrypt and copy to off-site storage with retention limits, log success and alert on failure, periodically restore to a scratch instance to prove the backup works, then document the restore procedure."),
    ("Outline how to launch a small e-commerce store end to end.",
     "Pick the products and pricing, choose a storefront platform, set up the catalog and payments, configure shipping and tax, add clear product pages and checkout, test the full purchase flow, launch with basic marketing, then monitor orders and customer feedback."),
    ("How does an agent decide which tool to call at each step of a task?",
     "It reads the current subgoal, matches it to a tool whose described capability fits, checks the required inputs are available, prefers the cheapest tool that can do the job, calls it with validated arguments, inspects the result, then either advances or picks a different tool if the result is unusable."),
    ("Describe the steps to internationalise an existing application.",
     "Extract hard-coded strings into resource files, wrap them in a translation function, externalise date, number and currency formatting, support locale selection and fallback, translate the resources, handle text expansion and right-to-left layouts, then test each locale end to end."),
    ("Complete this task: diagnose and fix a slow database query.",
     "Reproduce the slow query and capture its plan, find the expensive operations like full scans, add or adjust indexes, rewrite the query or schema where needed, re-measure against the plan, verify results are unchanged, then confirm the improvement under realistic load."),
    ("What steps should an agent follow to summarise a long document accurately?",
     "Read the whole document, identify its structure and main claims, extract the key points per section, condense without adding or distorting facts, preserve important numbers and caveats, order the summary logically, then check it against the source for fidelity."),
    ("Outline the end-to-end process of training and evaluating a classifier.",
     "Define the target and gather labelled data, split into train, validation and test, clean and engineer features, train candidate models, tune on validation, evaluate the best on the held-out test set with appropriate metrics, then check for leakage and bias before shipping."),
    ("How should an agent handle a task whose requirements are ambiguous?",
     "It identifies the specific ambiguities, states its assumptions, asks a focused clarifying question when the cost of guessing is high, otherwise proceeds with the most reasonable interpretation, keeps the work easy to revise, then confirms the result matches what the user meant."),
    ("Complete the task: set up monitoring and alerting for a microservice.",
     "Instrument the service for metrics, logs and traces, export them to a collector, define dashboards for latency, error rate and saturation, set alert thresholds tied to user impact, route alerts to on-call with runbooks, then test an alert fires and resolves correctly."),
]

STYLE_PROMPT = (
    "You are simulating five different AI assistants answering the same task-completion "
    "request, the way a pool of agent/chatbot miners would. Request: {q}\n"
    "A strong reference answer is: {gt}\n"
    "Return a JSON array of exactly five strings, each a different assistant's answer:\n"
    "1. a thorough, correct, well-structured multi-step answer,\n"
    "2. a correct but terse answer that lists only a few steps,\n"
    "3. a correct answer that restates the request and adds caveats before the steps,\n"
    "4. a fluent, confident answer that is on-topic but gives a wrong or muddled procedure,\n"
    "5. a vague answer that talks about the topic in general without giving usable steps.\n"
    "Return only the JSON array, no prose."
)


def call(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.8, "max_tokens": 900}).encode()
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


def main():
    from concurrent.futures import ThreadPoolExecutor
    rows = []
    def work(item):
        i, (q, gt) = item
        try:
            arr = parse_arr(call(STYLE_PROMPT.format(q=q, gt=gt)))
        except Exception as ex:
            print("call failed", i, ex, flush=True); return q, gt, []
        print(f"{i+1}/{len(QA)} {q[:44]!r} -> {len(arr)} answers", flush=True)
        return q, gt, arr
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(work, list(enumerate(QA))))
    for q, gt, arr in results:
        for a in arr:
            rows.append({"q": q, "gt": gt, "a": a.strip()})
    out = {"note": "Intent-fitted TASK_COMPLETION traffic: real gateway LLM answers of varied "
                   "quality per multi-step task request, mimicking the distribution the node's "
                   "traffic gate scores.",
           "rows": rows}
    p = os.path.join(ROOT, "bench", "traffic-task.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}: {len(rows)} rows")


if __name__ == "__main__":
    main()
