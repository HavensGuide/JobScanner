#!/usr/bin/env python3
"""
JevK5 on the jobscan bridge workload — v2, after the first run showed the
ranking was right but the spread was broken.

What changed and why:

  * The old `score` bands were conjunctions ("no public contact AND hires
    immediately AND commutable AND pay stated AND liveable"). Almost no
    posting states pay, so the top two bands were nearly unreachable and
    everything fell into band 2. The warehouse night shift scored 66 when it
    should have been 85.
    -> Bands are now a single axis: how much public contact, nothing else.

  * The old `choice` mixed two dimensions — three contact levels and four
    barriers — into one seven-way pick. Mass split across unrelated axes and
    confidence came back at 0.27-0.37 against a 0.14 baseline. "Shift work"
    won twice by being vague enough to fit anything.
    -> Replaced by two yes/no questions, each about a single fact.

  * Pay and shift are regex, not judgment. "$21.50/hr" and "10pm-6:30am" are
    sitting in the text. Asking a model to weigh them burned a band and cost
    spread.
    -> Extracted deterministically and composed into the reason line after.

Run the server first:

  llama-server.exe -m "...\\jevk5-4b-v0.3-Q4_K_M.gguf" -c 8192 -ngl 99 --port 8090

(or let the profile's server: block start it). Then:

  python bench_jev.py
  python bench_jev.py --n 30
  python bench_jev.py --model qwen        # any profile under models:

The same eight postings on every backend, so the spread and timing numbers
are directly comparable between JevK5 and anything else.
"""
import argparse
import statistics
import sys
import time

# Single source of truth. The questions, the extraction and the composition
# all live in jev.py and are read from config.yaml there, so tuning a band or
# a threshold changes the benchmark and the real run together. Two copies
# drifted apart once already.
try:
    import jev
except ImportError:
    sys.exit("jev.py must be alongside this script.")

def load_cfg(path="config.yaml"):
    """jev.merged() over the config.yaml jev block, if there is one."""
    try:
        import yaml
        raw = (yaml.safe_load(open(path, encoding="utf-8")) or {}).get("jev")
    except Exception:
        raw = None
    cfg = jev.merged(raw)
    problems = jev.validate(cfg)
    if problems:
        for p in problems:
            print(f"  ! jev config: {p}")
        print("  ! using built-in defaults instead\n")
        cfg = jev.merged(None)
    return cfg


POSTINGS = [
    ("warehouse night",
     "Warehouse Returns Associate - Night Shift at Quince, Carneys Point, New "
     "Jersey. Process customer returns through our reverse logistics facility. "
     "Inspect, sort, and restock returned merchandise. Scan items into WMS and "
     "route to appropriate disposition. Operate pallet jacks and material "
     "handling equipment. Maintain accurate inventory counts. Night shift, "
     "10pm-6:30am, Sunday through Thursday. Must be able to lift up to 50 lbs "
     "repeatedly and stand for the duration of an 8-hour shift. No prior "
     "warehouse experience necessary; paid training provided during the first "
     "two weeks. Benefits eligible after 60 days."),

    ("mailroom clerk",
     "Client Service Associate - Mailroom at SPS North America, New York, NY. "
     "Sort and distribute incoming mail and packages across floors. Operate "
     "postage meters and scanning equipment. Log accountable deliveries into "
     "the tracking system. Prepare outgoing shipments. Maintain the mail room "
     "and supply areas. Must be able to lift up to 55 lbs. Monday through "
     "Friday, 8:30am-5:30pm. High school diploma or equivalent required."),

    ("barista counter",
     "Barista | Brooklyn at Blank Street. Prepare espresso drinks, cold brew, "
     "and matcha to spec using automated equipment. Take orders at the counter "
     "and through the mobile app queue. Maintain station cleanliness and "
     "restock throughout shift. Opening and closing duties on a rotating "
     "basis. Standing for up to 8 hours, occasional lifting up to 25 lbs. "
     "Shifts available 6am-2pm and 2pm-10pm. No tipping; all staff paid a flat "
     "hourly rate of $18.50/hr with a quarterly bonus. Prior coffee experience "
     "preferred but not required, full training provided."),

    ("health info front desk",
     "Health Information Specialist I at Datavant, Jamaica, NY. Process "
     "release-of-information requests at a hospital site. Greet patients and "
     "visitors at the records window, verify identification, and explain "
     "request procedures. Answer incoming calls from patients, physicians and "
     "attorneys regarding record status. Log requests, scan charts, and "
     "maintain HIPAA compliance. $17.00-$20.00 per hour. Overtime required "
     "during peak season. No prior experience required."),

    ("ops w/ hidden phones",
     "Operations Associate at a growing logistics startup, New York, NY. Own "
     "the day-to-day relationship with our merchant partners. Field inbound "
     "questions via Zendesk and phone, troubleshoot delivery exceptions, and "
     "escalate to the fulfillment team. You will be the first point of contact "
     "when something goes wrong and will be measured on partner satisfaction "
     "scores and first-response time. 2+ years in a customer-facing operations "
     "role required. Comfortable with ambiguity and shifting priorities."),

    ("probable scam",
     "Data Entry Operator (Full time, Work from Home) at Skillerszone, Jersey "
     "City, NJ. $50,300 - $70,600 per year. We are seeking motivated "
     "individuals to join our remote team. Responsibilities include entering "
     "data into our database, reviewing records for accuracy, and organizing "
     "files. No experience necessary. Flexible hours, work from anywhere. "
     "Apply today and start earning immediately. Equipment provided after the "
     "onboarding fee is processed."),
]


def pct(vals, p):
    vals = sorted(vals)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12,
                    help="postings for the timing phase (cycles the samples)")
    ap.add_argument("--endpoint", default=None,
                    help="JevK5 server, overriding the profile's endpoint")
    ap.add_argument("--model", metavar="PROFILE",
                    help="model profile to benchmark (default: ai.jev for the "
                         "bridge track). Compare backends by running this "
                         "once per profile")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    import llm
    import yaml
    try:
        root = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    except FileNotFoundError:
        root = {}
    try:
        prof = (llm._lookup(root, args.model, "jev") if args.model
                else llm.resolve(root, "jev", "bridge"))
    except llm.LLMError as e:
        sys.exit(str(e))
    if args.endpoint and prof["kind"] == "jevk5":
        prof = {**prof, "endpoint": args.endpoint}
    print(f"model: {llm.describe(prof)}")
    if not llm.ensure_ready(prof)[0]:
        sys.exit("model unavailable")

    cfg = load_cfg(args.config)
    contact_q, gate_q, fee_q, temporal_q = jev.questions(cfg)
    src = "config.yaml" if (cfg.get("contact") != jev.DEFAULTS["contact"]
                            or cfg.get("thresholds") != jev.DEFAULTS["thresholds"]) \
          else "built-in defaults"
    print(f"{len(cfg['contact']['bands'])} contact bands from {src}\n")
    try:
        m = llm.decider(prof)
    except (ImportError, llm.LLMError) as e:
        sys.exit(f"cannot use {prof['name']}: {e}")

    print("warming up...", end=" ", flush=True)
    t0 = time.perf_counter()
    m.decide(POSTINGS[0][1], contact_q)
    print(f"{time.perf_counter()-t0:.2f}s (discarded)\n")

    print("=" * 74)
    print("SPREAD AND RANKING")
    print("=" * 74)
    scored = []
    for label, text in POSTINGS:
        c = m.decide(text, contact_q)
        g = m.decide(text, gate_q)
        p = m.decide(text, fee_q)
        tm = m.decide(text, temporal_q)
        score, review, reason = jev.compose(c, g, p, text, cfg, 'bridge', tm)
        scored.append((score, label, review, reason, c, g, p))

    for score, label, review, reason, c, g, p in sorted(scored, key=lambda r: -r[0]):
        flag = " [REVIEW]" if review else ""
        print(f"\n  {score:>3}{flag}  {label}")
        print(f"        {reason}")
        probs = "  ".join(f"{k}:{float(v):.2f}" for k, v in
                          sorted((c.get("probabilities") or {}).items()))
        print(f"        bands  {probs}   conf {float(c.get('confidence',0)):.2f}")
        print(f"        gate {float(g.get('noul',0)):.2f}   "
              f"fee {float(p.get('noul',0)):.2f}")

    vals = [s[0] for s in scored]
    print("\n" + "-" * 74)
    print(f"  range {min(vals)}-{max(vals)}   spread {max(vals)-min(vals)}   "
          f"stdev {statistics.pstdev(vals):.1f}")
    print("  The old rubric put everything between 45 and 67. Under ~40 points")
    print("  of spread here means the bands are still too conjunctive.")

    # ---- timing ----------------------------------------------------------
    jobs = [POSTINGS[i % len(POSTINGS)][1] for i in range(args.n)]
    print("\n" + "=" * 74)
    print(f"TIMING  ({args.n} postings, 4 questions each, serial)")
    print("=" * 74)
    lat = []
    t0 = time.perf_counter()
    for text in jobs:
        t1 = time.perf_counter()
        m.decide(text, contact_q)
        m.decide(text, gate_q)
        m.decide(text, fee_q)
        m.decide(text, temporal_q)
        lat.append(time.perf_counter() - t1)
    total = time.perf_counter() - t0
    rate = args.n / total
    print(f"  per posting: p50 {pct(lat,50):.2f}s  p95 {pct(lat,95):.2f}s")
    print(f"  {rate:.2f} postings/s")
    print("\n  extrapolated:")
    for n in (250, 600, 1273):
        print(f"    {n:>5} postings -> {n/rate/60:>6.1f} min")


if __name__ == "__main__":
    main()
