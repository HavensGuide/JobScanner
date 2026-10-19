#!/usr/bin/env python3
"""
Why is everything scoring 99?

Pulls specific career postings out of seen.db and prints the RAW three-way
result for each: the full band distribution, the seniority probability and the
credential-gate probability. That says which of the three questions is
failing, instead of us guessing from a composed number.

Pairs that should separate clearly:
  NORC Statistician I        vs  SNAP Eligibility Interviewer
  Recidiviz Data Analyst     vs  Misdemeanor Case Coordinator

If the bands are the problem, the distributions for both members of a pair
will look the same. If seniority or the gate is the problem, the bands will
differ but the composed scores will not.

    python diag_career.py
    python diag_career.py --uid 801656 --uid 5903247996
"""
import argparse
import sqlite3
import sys

import jev

# Titles from the tailor list, chosen to span what SHOULD be a wide range.
WANTED = [
    "Statistician I",
    "Data Analyst (Entry-Level)",
    "Financial Economist",
    "SNAP Eligibility Interviewer",
    "Misdemeanor Case Coordinator",
    "Community Engagement Coordinator",
    "FDNY Criminal Court",
    "Assistant Negotiator",
    "Law Extern",
    "Marine Corps Network and Policy SME",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="seen.db")
    ap.add_argument("--uid", action="append", default=[])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--facts", default="resume.yaml")
    ap.add_argument("--model", metavar="PROFILE",
                    help="model profile to diagnose (default: ai.jev for "
                         "the career track)")
    args = ap.parse_args()

    import llm
    try:
        import yaml
        root = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    except Exception:
        root = {}
    cfg = jev.merged(root.get("jev"))
    try:
        profile = (llm._lookup(root, args.model, "jev") if args.model
                   else llm.resolve(root, "jev", "career"))
    except llm.LLMError as e:
        sys.exit(str(e))
    print(f"model: {llm.describe(profile)}\n")

    conn = sqlite3.connect(args.db)
    rows = []
    if args.uid:
        for u in args.uid:
            rows += list(conn.execute(
                "SELECT uid, title, company, location, description, "
                "desc_quality, score, reason FROM jobs WHERE uid LIKE ?",
                (f"%{u}%",)))
    else:
        for t in WANTED:
            r = conn.execute(
                "SELECT uid, title, company, location, description, "
                "desc_quality, score, reason FROM jobs "
                "WHERE track='career' AND title LIKE ? "
                "ORDER BY first_seen DESC LIMIT 1", (f"%{t}%",)).fetchone()
            if r:
                rows.append(r)
    if not rows:
        sys.exit("no matching postings in seen.db")

    ok, _ = jev.ensure_server(cfg, profile)
    if not ok:
        sys.exit("jev model unavailable")
    try:
        model = jev._model(cfg, profile)
    except (ImportError, llm.LLMError) as e:
        sys.exit(f"cannot use {profile['name']}: {e}")

    score_q, level_q, gate_q, temporal_q = jev.questions(cfg, "career")
    print("no candidate summary in the state: the band order carries the "
          "preference\n")
    prefix = ""

    print(f"{'score':>5} {'band distribution':<34} {'lvl':>5} {'gate':>5} "
          f"{'open':>5} {'desc':>6}  title")
    print("-" * 104)
    for uid, title, company, loc, desc, quality, stored, reason in rows:
        body = jev._focus_description_shim(desc, 700) \
            if hasattr(jev, "_focus_description_shim") else (desc or "")[:700]
        # Mirror the pipeline's guard. Without this the diagnostic prints a
        # score for a posting score_track would never send to the model at
        # all, which is how a 0-character row showed up at 84.
        if len((body or "").strip()) < cfg.get("min_desc_chars", 120):
            print(f"{'SKIP':>5} {'(not sent to the model)':<34} "
                  f"{'':>5} {'':>5} {'':>5} {len(desc or ''):>6}  {title[:44]}")
            if stored is not None:
                print(f"{'':>5} stored {stored}: {(reason or '')[:76]}")
            continue
        text = f"{prefix}{title} at {company}, {loc}. {body}"
        c = model.decide(text, score_q)
        lv = model.decide(text, level_q)
        gt = model.decide(text, gate_q)
        tm = model.decide(text, temporal_q)
        s, review, _ = jev.compose(c, lv, gt, text, cfg, "career", tm)
        probs = c.get("probabilities") or {}
        dist = " ".join(f"{k}:{float(v):.2f}" for k, v in sorted(probs.items()))
        print(f"{s:>5} {dist:<34} {float(lv.get('noul',0)):>5.2f} "
              f"{float(gt.get('noul',0)):>5.2f} "
              f"{float(tm.get('noul',1)):>5.2f} "
              f"{(len(desc or '')):>6}  {title[:44]}")
        if stored is not None:
            print(f"{'':>5} stored {stored}: {(reason or '')[:76]}")

    print("\nband 4 is the top (direct research match), band 0 the bottom.")
    print("lvl  = P(open to a recent grad). LOW is bad.")
    print("gate = P(needs a credential you lack). HIGH is bad.")
    print("open = P(you can apply now). LOW is bad.")
    print("\nIf the casework jobs show the same distribution as the "
          "statistician jobs,\nthe bands are the problem, not the gates.")


if __name__ == "__main__":
    main()
