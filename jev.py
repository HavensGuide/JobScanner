#!/usr/bin/env python3
"""
jev.py — band / yes-no scoring, both tracks, on any model.

Originally JevK5 on llama.cpp only. The questions and composition below are
model-agnostic; which model answers them is `ai.jev` in config.yaml, and
llm.py reads the answer as probabilities from JevK5, any local server, or a
hosted API (see the readout notes there). With no `ai:` block this still runs
JevK5 exactly as before.

JevK5 is Qwen3.5-4B with a merged distilled LoRA, read as a softmax over the
answer letters' next-token logits. It generates no tokens: every call returns
a calibrated probability over the options supplied. That rules out a free-text
reason line, and rules out the failure this replaces — a small model asked for
0-100 integers clusters everything at 70.

Three questions per posting, one axis each. The shape was arrived at by
measurement; two earlier versions failed:

  v1  A five-condition score band ("no public contact AND hires immediately
      AND commutable AND pay stated AND liveable"). Most postings omit pay,
      so the top bands were unreachable and six samples landed between 45 and
      67. A warehouse night shift scored 66.

  v2  Split the conjunction out of the score but left a gestalt "does this
      look legitimate" question. That needs a US wage table the model does
      not have, so it guessed: a $70,600 remote data-entry posting came back
      legit=0.92 and ranked FIRST at 97.

  v3  Ask only what is in the text. The same posting says "Equipment provided
      after the onboarding fee is processed"; asking about that directly
      returns fee=0.96. Implausible pay became arithmetic. Spread on the same
      six samples: 10 to 86.

EVERYTHING BELOW IS A DEFAULT, NOT A RULE. The bands, the questions, the
labels and every threshold can be overridden from the `jev:` block in
config.yaml, because they encode preferences rather than facts, and
preferences change — "barista is not as bad as waiter" was a one-line config
edit, and the equivalent judgment here should be too. Anything the config
does not mention falls back to the values here, so an empty `jev:` block
behaves exactly as this file does.

Measured on an RTX 3060 Laptop, Q4_K_M, all layers offloaded: 0.74s p50 per
posting for all three questions, ~3.2 min for 250 postings. Concurrency
measured at 1.08x and is not worth the complexity, so this runs serial.
"""
import copy
from datetime import date
import re
import time

import fit

DEFAULTS = {
    # ---- runtime -------------------------------------------------------
    "endpoint": "http://127.0.0.1:8090",
    # Calibration temperatures live in the model's jevk5_config.json. A wrong
    # value does not error; it silently miscalibrates every probability,
    # which defeats the entire reason for using this model.
    "temperature": 1.22,
    "knockout_temperature": 0.93,
    "server_exe": "",
    "model_path": "",
    "ctx": 8192,
    "ngl": 99,
    "startup_timeout": 180,

    # ---- the contact axis ----------------------------------------------
    # Ordered lowest (worst) to highest (best) — the order the runtime
    # expects, and the order that encodes "less public contact is better".
    # Reverse the list and you have reversed the preference.
    #
    # Each band must describe ONE thing. The v1 failure above was a band that
    # required five conditions at once; almost nothing satisfied it and the
    # scale collapsed. If you edit these, keep them single-clause.
    "contact": {
        "instructions":
            "Read the duties and decide how much contact with customers, "
            "clients, patients or the general public this job actually "
            "involves. Judge the described work, not the job title. Ignore "
            "pay, location, schedule and required experience entirely.",
        "bands": [
            "Selling to people is the job: cold calling, door-to-door, a "
            "commissioned sales floor, or a quota attached to the worker's "
            "name.",

            "The worker owns someone's experience for an extended stretch and "
            "is the person they complain to: waiting tables with a section, a "
            "support or ticket queue, a front desk, or a named satisfaction "
            "metric.",

            "The worker deals with the public in short, repeated, scripted "
            "exchanges that end quickly: a counter, a service window, a "
            "pickup desk. Nobody's ongoing experience is the worker's "
            "responsibility.",

            "The worker occasionally encounters someone from outside the "
            "organisation, but it is incidental and not what the job is for.",

            "The worker deals only with coworkers. A whole shift can pass "
            "without speaking to a customer at all.",
        ],
        # One short phrase per band, same order, used to open the reason line.
        "labels": [
            "commissioned sales",
            "relational service",
            "transactional counter",
            "incidental contact",
            "no public contact",
        ],
    },

    # ---- yes/no checks --------------------------------------------------
    "gate": {
        "instructions":
            "Does this posting require something a first-time applicant would "
            "not already have — more than about two years of specific prior "
            "experience, a professional licence, a certification, a security "
            "clearance, or a commercial driving licence?",
        "true": "Yes, there is a stated requirement that would screen out a "
                "candidate with no experience in this field.",
        "false": "No, the posting is open to someone without prior experience "
                 "in this line of work, or any preference stated is soft.",
    },
    "fee": {
        "instructions":
            "Does this posting ask the applicant to pay a fee, buy or pay for "
            "equipment or training, provide bank account or card details, or "
            "send money at any stage of hiring or onboarding?",
        "true": "Yes, the applicant is asked for money, payment details, or "
                "to cover a cost before or during onboarding.",
        "false": "No, the employer pays the worker and asks for nothing "
                 "financial from the applicant.",
    },

    # ---- the career axis ------------------------------------------------
    # Career fit is a harder judgment than contact level: it is relative to a
    # person, not a property of the posting. So the candidate profile goes
    # into the state, and these bands rank how close the work is to it.
    #
    # Ordered lowest (worst) to highest (best), same as contact.
    "career_fit": {
        # No candidate summary in the state. The first version put one there
        # and every posting came back 0.97-0.99 on the top band, because the
        # model was matching the band text against the resume rather than
        # against the job. Preference lives in the band ORDER, exactly as it
        # does on the contact axis.
        "instructions":
            "Read the duties and decide what kind of work this job actually "
            "is. Judge the described work, not the job title, and not the "
            "employer. Ignore seniority, required years and start dates - "
            "those are asked separately.",
        # Single-clause, lowest first. The earlier top band listed domains
        # ("...in social policy, economics, criminal justice, government"),
        # and a municipal caseworker matched on the domain alone.
        # Seven bands now, lowest first. The order IS the preference: move a
        # line and you have changed what the digest ranks highest. Still
        # single-clause - the version that listed domains ("...in social
        # policy, economics, criminal justice, government") let a municipal
        # caseworker match on the domain alone and score 99.
        "bands": [
            "A different profession: sales, recruiting, teaching, nursing, "
            "trades, retail, or software engineering with no analysis.",

            "Casework, coordination or administration: processing "
            "applications, interviewing clients or claimants, scheduling, "
            "representing an agency to the public, or running a programme "
            "day to day.",

            "Business reporting: building dashboards, maintaining "
            "spreadsheets, and pulling metrics for other teams.",

            "Explaining research to people outside it: science writing, "
            "research communication, policy briefs, data journalism, "
            "dissemination, or public engagement about technical findings.",

            "Studying the social or ethical effects of technology: AI "
            "ethics, fairness, algorithmic accountability, responsible AI, "
            "AI governance or technology policy.",

            "Applied quantitative work: forecasting, experimentation, "
            "A/B testing, data science or machine learning on operational "
            "data.",

            "Original research that produces findings: econometrics, survey "
            "or administrative microdata analysis, program evaluation, or "
            "machine learning and AI research.",
        ],
        "labels": [
            "different profession",
            "casework or coordination",
            "reporting, not research",
            "research communication",
            "AI ethics or governance",
            "applied quantitative",
            "research",
        ],
    },

    # Seniority is the single most common reason a good-looking research job
    # is unreachable for a recent graduate, and it is stated in the text.
    "career_level": {
        "instructions":
            "Is this job open to a recent graduate with under two years of "
            "full-time professional experience?",
        "true": "Yes, it is entry-level, early-career, a fellowship, an "
                "analyst or assistant role, or states no experience minimum.",
        "false": "No, it is senior, lead, principal, managerial, or requires "
                 "more than about two years of relevant full-time experience.",
    },

    # Credentials, separately from years, because they are absolute.
    "career_gate": {
        "instructions":
            "Does this posting require a credential beyond a bachelor's "
            "degree - a master's, PhD, law degree, professional licence, "
            "security clearance - or require the applicant to be CURRENTLY "
            "ENROLLED as a student, which a 2026 graduate is not?",
        "true": "Yes. A graduate degree, licence, clearance, or current "
                "student enrolment is required. Internships, externships, "
                "co-ops and campus programmes count as requiring enrolment.",
        "false": "No, a bachelor's degree or equivalent experience is "
                 "sufficient.",
    },

    # The hosted rubrics carried "Today is {today}. A stated start date or
    # application deadline in the past scores 0." jev had no equivalent, so a
    # Spring 2027 law externship scored 99. main.py's regex gate catches the
    # obvious cases; this catches the ones phrased in prose.
    "temporal": {
        "instructions":
            "Today is {today}. Can someone apply to this job NOW and start "
            "within the next few months?",
        "true": "Yes, it is open for applications now and the work starts "
                "soon or immediately.",
        "false": "No. The application deadline has passed, the role is for a "
                 "future term or season more than a few months away, the "
                 "posting is for a programme that has already started, or it "
                 "names a year later than the current one.",
    },

    # Screens text pulled off a job URL. Fetching a posting page gets you a
    # sign-in wall, a cookie banner, a "position no longer available" notice
    # or a board index about as often as it gets the job. All of those are
    # long enough to clear min_desc_chars and would be scored as the posting.
    "posting_text": {
        "instructions":
            "Is this text the body of a single job posting - does it describe "
            "the duties, responsibilities or requirements of one specific "
            "role?",
        "true": "Yes, it describes what one particular job involves or "
                "requires.",
        "false": "No. It is a sign-in or access-denied page, an error or "
                 "'position no longer available' notice, a cookie or consent "
                 "banner, a list of many different jobs, or a company page "
                 "with no role described.",
    },

    # A posting with no text cannot be judged. The hosted rubric handled this
    # with [TITLE ONLY] and [EXCERPT ONLY] tags; here the posting simply does
    # not go to the model at all. Below this many characters of description,
    # the row is flagged for manual review and never scored high.
    "min_desc_chars": 120,

    # How much of resume.yaml goes into the state for career questions. The
    # full profile is ~12,000 characters and most of it is irrelevant to
    # "is this the same kind of work"; sending it would triple the prefill
    # for no gain.
    "profile_chars": 1200,

    # ---- composition ----------------------------------------------------
    # Every number here was set against six sample postings, which is enough
    # to catch a sign error and nowhere near enough to be right. Expect to
    # move them after a real run; that is why they are config and not code.
    "thresholds": {
        # A gate is a ceiling, not a subtraction: a job you cannot be hired
        # for is not a good job with a deduction, however pleasant the work.
        "gate_hard": 0.65,
        "gate_hard_cap": 40,
        "gate_soft": 0.45,
        "gate_soft_penalty": 10,

        "fee": 0.50,
        "fee_cap": 10,

        # A posting wanting nobody with experience while paying like a career
        # job is the shape of a recruitment scam. A real payroll accountant
        # at $120k does not trip this: its gate probability is high.
        "implausible_hourly": 30.0,
        "implausible_gate_below": 0.25,
        "implausible_cap": 25,

        # Review on uncertainty, not on a low score. Low and confident is a
        # decision; middling and unsure is a question. Top-band rather than
        # reported confidence, so mass smeared 0.13/0.34/0.48 across three
        # bands gets flagged. Raise toward 0.45 if too much is flagged.
        "review_top_band": 0.55,
        "review_gate_lo": 0.35,
        "review_gate_hi": 0.65,
        "review_fee_lo": 0.30,
        "review_fee_hi": 0.50,

        # Annual-to-hourly divisor for the implausible-pay test.
        "hours_per_year": 2080,

        # Career track. Seniority and credentials are ceilings for the same
        # reason a bridge gate is: a job you cannot be hired for is not a
        # good job with a deduction.
        "career_level_below": 0.35,      # P(open to a recent grad) under this
        "career_level_cap": 30,
        "career_level_soft": 0.60,
        "career_level_penalty": 10,
        "career_gate_above": 0.65,       # P(needs a credential) over this
        "career_gate_cap": 25,
        "career_review_top_band": 0.50,

        # Temporal, both tracks. Below this, the posting is closed or
        # future-dated and scores zero, matching the hosted rubrics.
        "temporal_open_below": 0.35,
        "temporal_soft": 0.60,
        "temporal_penalty": 15,

        # Score given to a posting with no usable description. It is not a
        # judgement, it is "go look at this yourself", so it sits at the
        # review floor rather than at zero.
        "no_desc_score": 45,
    },
}


def merged(cfg=None):
    """
    DEFAULTS deep-merged with a user block.

    Deep, so overriding one threshold does not silently delete the other
    twelve. A dict in the user's config updates the matching default; a list
    (the bands, the labels) replaces it wholesale, because a half-replaced
    ordered scale would be worse than either version.
    """
    out = copy.deepcopy(DEFAULTS)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        elif v is not None:
            out[k] = v
    return out


def validate(cfg, max_bands=None):
    """
    Return a list of complaints. Empty means usable.

    Called by preflight, and again at scoring time — a bad band list falls
    back to the built-ins with a warning rather than failing the run, so a
    typo in config.yaml costs you accuracy for one run, not the run itself.
    """
    problems = []
    for axis in ("contact", "career_fit"):
        c = cfg.get(axis) or {}
        bands, labels = c.get("bands") or [], c.get("labels") or []
        if len(bands) < 2:
            problems.append(f"jev.{axis}.bands needs at least 2 levels")
        if len(labels) != len(bands):
            problems.append(f"jev.{axis}.labels has {len(labels)} entries but "
                            f"bands has {len(bands)}; they must match 1:1")
        if not (c.get("instructions") or "").strip():
            problems.append(f"jev.{axis}.instructions is empty")
        if max_bands and len(bands) > max_bands:
            # JevK5 reads any number (knockout); other models are read as
            # lettered options A-P, so they stop at 16.
            problems.append(f"jev.{axis}.bands has {len(bands)} levels; "
                            f"only JevK5 reads more than {max_bands}")
    for name in ("gate", "fee", "career_level", "career_gate", "temporal",
                 "posting_text"):
        q = cfg.get(name) or {}
        for key in ("instructions", "true", "false"):
            if not (q.get(key) or "").strip():
                problems.append(f"jev.{name}.{key} is empty")
    problems += fit.validate(cfg)
    t = cfg.get("thresholds") or {}
    if t.get("gate_soft", 0) >= t.get("gate_hard", 1):
        problems.append("jev.thresholds.gate_soft must be below gate_hard")
    if t.get("review_gate_lo", 0) >= t.get("review_gate_hi", 1):
        problems.append("jev.thresholds.review_gate_lo must be below "
                        "review_gate_hi")
    if not 0 < t.get("hours_per_year", 2080):
        problems.append("jev.thresholds.hours_per_year must be positive")
    return problems


def questions(cfg, track="bridge", today=None):
    """
    (score_q, noul_a, noul_b, temporal_q) for a track.

    Both tracks use the same shape — one ordered score plus three yes/no
    checks — so timing and composition are identical. Only the axes differ.
    The temporal check is shared: a closed posting is closed on either track.
    """
    if track == "career":
        c, a, b = cfg["career_fit"], cfg["career_level"], cfg["career_gate"]
    else:
        c, a, b = cfg["contact"], cfg["gate"], cfg["fee"]
    t = cfg["temporal"]
    today = today or date.today().isoformat()
    noul = lambda q, text=None: {
        "type": "noul",
        "instructions": (text or q["instructions"]).replace("{today}", today),
        "criteria": {"true": q["true"], "false": q["false"]}}
    return (
        {"type": "score", "instructions": c["instructions"],
         "criteria": list(c["bands"])},
        noul(a), noul(b), noul(t),
    )


def profile(facts_path="resume.yaml", limit=1200):
    """
    A short candidate summary for the career state.

    resume.yaml is ~12,000 characters and most of it — publication titles,
    voice rules, bullet phrasing — says nothing about whether a posting is
    the same KIND of work. Degree, fields, methods and employers do. Returns
    "" if the file is missing, which downgrades career scoring rather than
    breaking it.
    """
    try:
        import yaml
        f = yaml.safe_load(open(facts_path, encoding="utf-8")) or {}
    except Exception:
        return ""
    ed = f.get("education") or {}
    sk = f.get("skills") or {}
    def flat(key, n):
        v = sk.get(key) or []
        return "; ".join(str(x) for x in v[:n])
    roles = []
    for e in (f.get("experience") or [])[:6]:
        roles.append(f"{e.get('role','')} at {e.get('org','')}")
    parts = [
        f"CANDIDATE: {ed.get('degree','')}, {ed.get('school','')}, "
        f"graduated {ed.get('graduated','')}.",
        f"Roles held: {'; '.join(roles)}.",
        f"Technical: {flat('technical', 10)}.",
        f"Research: {flat('research', 6)}.",
    ]
    return " ".join(p for p in parts if p.strip(" .")).strip()[:limit]


# ==========================================================================
# Deterministic extraction. No model, no tokens, no chance of invention.
# ==========================================================================
_PAY = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d{2})?\s*(?:(?:-|–|to)\s*\$?\s?\d[\d,]*(?:\.\d{2})?)?"
    r"\s*(?:/\s*|per\s+)?(?:hr|hour|hourly|yr|year|annually|k)\b",
    re.I)
_SHIFT = re.compile(
    r"\b(?:\d{1,2}(?::\d{2})?\s?(?:am|pm)\s?(?:-|–|to)\s?\d{1,2}(?::\d{2})?\s?(?:am|pm)"
    r"|overnight|night shift|third shift|graveyard|weekends?|part[- ]time)\b",
    re.I)


def extract_pay(text):
    m = _PAY.search(text or "")
    return " ".join(m.group(0).split()) if m else None


def extract_shift(text):
    m = _SHIFT.search(text or "")
    return " ".join(m.group(0).split()) if m else None


def hourly_rate(pay_str, hours_per_year=2080):
    """
    Top of the advertised range, normalised to dollars per hour.

    Arithmetic, not inference. The model cannot know what data entry pays in
    New Jersey; this can.
    """
    if not pay_str:
        return None
    nums = [float(n.replace(",", ""))
            for n in re.findall(r"\d[\d,]*(?:\.\d{2})?", pay_str)]
    if not nums:
        return None
    top = max(nums)
    if re.search(r"(yr|year|annual)", pay_str, re.I):
        return top / hours_per_year
    # "$85K" has no word boundary between digit and K, so \bk\b misses it and
    # the value falls through as an hourly rate of $85.
    if re.search(r"\dk\b|\bk\b", pay_str, re.I):
        return top * 1000.0 / hours_per_year
    return top


# ==========================================================================
# Composition
# ==========================================================================
def compose(score_res, a_res, b_res, text, cfg=None, track="bridge",
            t_res=None, explain=None):
    """
    (score 0-100, review bool, reason string) from three probability results.

    The band score is an expected value across all bands rather than the
    argmax, so a posting sitting between two bands lands between them instead
    of snapping to one. That continuous read is the payoff of a calibrated
    readout, and it works for any number of bands.

    Bridge: a=gate (needs something you lack), b=fee (asks you for money).
    Career: a=level (open to a recent grad), b=gate (needs a credential).
    Note the polarity flip on `a` — for bridge, high is bad; for career,
    high is good.
    """
    cfg = cfg if cfg and "thresholds" in cfg else merged(cfg)
    t = cfg["thresholds"]
    axis = cfg["career_fit"] if track == "career" else cfg["contact"]
    labels = axis["labels"]
    top_index = max(1, len(axis["bands"]) - 1)

    probs = score_res.get("probabilities") or {}
    ev = sum(float(p) * int(k) for k, p in probs.items())
    score = ev / top_index * 100.0
    a_p = float(a_res.get("noul", 0.0))
    b_p = float(b_res.get("noul", 0.0))
    notes = []
    steps = []
    raw_ev = score

    if track == "career":
        # Seniority is the most common reason a good research job is
        # unreachable for a recent graduate, and it is stated in the text.
        if a_p < t["career_level_below"]:
            score = min(score, float(t["career_level_cap"]))
            notes.append("senior or 2+ years required")
            steps.append(f"capped at {t['career_level_cap']}: open-to-a-"
                         f"recent-grad only {a_p:.2f}")
        elif a_p < t["career_level_soft"]:
            score -= float(t["career_level_penalty"])
            notes.append("may be above entry level")
            steps.append(f"-{t['career_level_penalty']}: open-to-a-recent-"
                         f"grad only {a_p:.2f}")
        if b_p > t["career_gate_above"]:
            score = min(score, float(t["career_gate_cap"]))
            notes.append("PhD, licence or clearance required")
            steps.append(f"capped at {t['career_gate_cap']}: credential gate "
                         f"{b_p:.2f}")
        review_floor = t["career_review_top_band"]
        review = (max((float(v) for v in probs.values()), default=1.0) < review_floor
                  or 0.30 < a_p < 0.55
                  or 0.35 < b_p < 0.65)
    else:
        if a_p > t["gate_hard"]:
            score = min(score, float(t["gate_hard_cap"]))
            notes.append("experience or licence gate")
            steps.append(f"capped at {t['gate_hard_cap']}: experience or "
                         f"licence gate {a_p:.2f}")
        elif a_p > t["gate_soft"]:
            score -= float(t["gate_soft_penalty"])
            notes.append("possible experience gate")
            steps.append(f"-{t['gate_soft_penalty']}: possible experience "
                         f"gate {a_p:.2f}")
        if b_p > t["fee"]:
            score = min(score, float(t["fee_cap"]))
            notes.append("asks applicant for money")
            steps.append(f"capped at {t['fee_cap']}: asks for money "
                         f"{b_p:.2f}")
        rate = hourly_rate(extract_pay(text), t["hours_per_year"])
        if (rate and rate > t["implausible_hourly"]
                and a_p < t["implausible_gate_below"]):
            score = min(score, float(t["implausible_cap"]))
            notes.append(f"${rate:.0f}/hr with no experience required")
            steps.append(f"capped at {t['implausible_cap']}: ${rate:.0f}/hr "
                         f"with no experience asked")
        review = (max((float(v) for v in probs.values()), default=1.0)
                  < t["review_top_band"]
                  or t["review_gate_lo"] < a_p < t["review_gate_hi"]
                  or t["review_fee_lo"] < b_p < t["review_fee_hi"])

    # Temporal, both tracks. A posting you cannot apply to is worth nothing
    # regardless of how good the work is, which is what the hosted rubrics
    # said: "a stated start date or application deadline in the past scores 0".
    if t_res is not None:
        open_p = float(t_res.get("noul", 1.0))
        if open_p < t["temporal_open_below"]:
            score = 0.0
            notes.append("closed or future-dated")
            steps.append(f"zeroed: still-open only {open_p:.2f}")
        elif open_p < t["temporal_soft"]:
            score -= float(t["temporal_penalty"])
            notes.append("check the dates")
            steps.append(f"-{t['temporal_penalty']}: still-open {open_p:.2f}")
        review = review or (t["temporal_open_below"] <= open_p
                            < t["temporal_soft"])

    score = max(0.0, min(100.0, score))
    band = int(max(probs, key=lambda k: float(probs[k]))) if probs else top_index
    bits = [labels[min(band, len(labels) - 1)]]
    pay = extract_pay(text)
    if track == "bridge":
        bits.append(pay or "no pay stated")
        shift = extract_shift(text)
        if shift:
            bits.append(shift)
    elif pay:
        bits.append(pay)
    bits += notes

    if explain is not None:
        ranked = sorted(((int(k), float(v)) for k, v in probs.items()),
                        key=lambda kv: -kv[1])
        explain.update({
            "track": track,
            "band": labels[min(band, len(labels) - 1)],
            "bands": [(labels[min(i, len(labels) - 1)], round(p, 3))
                      for i, p in ranked],
            "expected_value": round(raw_ev, 1),
            "final": round(score),
            "questions": (
                {"open to a recent grad": round(a_p, 3),
                 "credential required": round(b_p, 3)}
                if track == "career" else
                {"experience or licence gate": round(a_p, 3),
                 "asks for money": round(b_p, 3)}),
            "adjustments": steps,
            "review": bool(review),
            "pay": pay or None,
        })
        if t_res is not None:
            explain["questions"]["still open"] = round(
                float(t_res.get("noul", 1.0)), 3)
        explain["text"] = explain_text(explain)
    return round(score), review, " · ".join(bits)


def explain_text(d):
    """
    The working, as a few lines a person can read.

    Ordered the way the score is actually built: which band the posting
    landed in and how confidently, the expected value that came out of the
    whole distribution, then every cap or penalty applied to it, then the
    answer to each yes/no question. A score with no adjustments says so,
    which is itself the explanation.
    """
    lines = []
    top = d.get("bands") or []
    if top:
        first = ", ".join(f"{name} {p:.2f}" for name, p in top[:3])
        lines.append(f"Band: {first}")
    lines.append(f"Expected value across the bands: {d.get('expected_value')}")
    for s in d.get("adjustments") or []:
        lines.append(f"  {s}")
    if not d.get("adjustments"):
        lines.append("  no caps or penalties applied")
    q = d.get("questions") or {}
    if q:
        lines.append("Questions: "
                     + ", ".join(f"{k} {v:.2f}" for k, v in q.items()))
    lines.append(f"Final: {d.get('final')}"
                 + (" (flagged for review)" if d.get("review") else ""))
    return "\n".join(lines)


# ==========================================================================
# Backend. Which model answers the questions is a config choice now (ai.jev
# in config.yaml); llm.py owns the server lifecycle and every readout. These
# two wrappers keep diag_career.py and bench_jev.py working unchanged.
# ==========================================================================
def profile_for(cfg=None, profile=None, track="bridge",
                config_path="config.yaml"):
    """
    The llm profile to use: the one passed in, else whatever config.yaml
    assigns jev for this track, which by default is your local JevK5. Only
    with no readable config.yaml does it fall back to this block's keys.
    """
    import llm
    if profile is not None:
        return profile
    try:
        import yaml
        root = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    except (OSError, ImportError):
        root = {}
    if root:
        return llm.resolve(root, "jev", track)
    return llm.finalize(llm._legacy_jev_profile({"jev": cfg or {}}), "jevk5")


def ensure_server(cfg, profile=None):
    import llm
    return llm.ensure_ready(profile_for(cfg, profile))


def _model(cfg, profile=None):
    import llm
    return llm.decider(profile_for(cfg, profile))


def screen_texts(texts, cfg=None, profile=None, threshold=0.5):
    """
    [(is_a_posting, probability)] per text, or None if no model is available.

    enrich.py calls this on text scraped off a posting URL. Returns None
    rather than guessing when the model is down, so the caller decides
    whether to keep unverified text or drop it.
    """
    import llm
    cfg = merged(cfg)
    try:
        prof = profile_for(cfg, profile, "bridge")
    except llm.LLMError:
        return None
    if llm.check(prof) or not llm.ensure_ready(prof, quiet=True)[0]:
        return None
    try:
        model = llm.decider(prof)
    except (ImportError, llm.LLMError):
        return None
    q = cfg["posting_text"]
    question = {"type": "noul", "instructions": q["instructions"],
                "criteria": {"true": q["true"], "false": q["false"]}}
    out = []
    for t in texts:
        try:
            r = model.decide((t or "")[:4000], question)
        except Exception:
            return None
        p = float(r.get("noul", 0.0))
        out.append((p >= threshold, p))
    return out


def prescan(text, cfg=None, profile=None, track="career", today=None):
    """
    Score one FULL posting. Returns (score, review, reason).

    main.py scores a 700-character focus slice; tailor.py has the whole
    posting, so re-running here sees requirements the slice cut off. Same
    questions, same bands - no new judgment to calibrate.

    (None, ...) when no model is available, so the caller falls through to
    its existing behaviour rather than gating on a guess.
    """
    import llm
    cfg = merged(cfg)
    if len((text or "").strip()) < cfg.get("min_desc_chars", 120):
        return None, True, "manual review: posting text too short to judge"
    try:
        prof = profile_for(cfg, profile, track)
    except llm.LLMError as e:
        return None, False, str(e)
    problem = llm.check(prof)
    if problem:
        return None, False, problem
    if not llm.ensure_ready(prof, quiet=True)[0]:
        return None, False, "model unavailable"
    try:
        model = llm.decider(prof)
    except (ImportError, llm.LLMError) as e:
        return None, False, f"{type(e).__name__}: {e}"
    if cfg.get("mode", "resume") == "resume" and fit.candidate(cfg):
        # Same scale as the digest, on the full posting.
        try:
            r = fit.score_one(model, {"description": text}, text, cfg,
                              track, fit.candidate(cfg), today)
        except Exception as e:
            return None, False, f"{type(e).__name__}: {e}"
        return r["score"], r["review"], r["reason"]
    score_q, a_q, b_q, t_q = questions(cfg, track, today)
    try:
        c = model.decide(text, score_q)
        a = model.decide(text, a_q)
        b = model.decide(text, b_q)
        t = model.decide(text, t_q)
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"
    return compose(c, a, b, text, cfg, track, t)


# ==========================================================================
# Entry point used by score.py
# ==========================================================================
def score_track(jobs, cfg=None, focus=None, track="bridge",
                facts_path="resume.yaml", progress_every=25, profile=None):
    """
    Score one track's postings in place. Returns (n_scored, failure_reason).

    On any failure the jobs are left untouched so the caller can fall back to
    the hosted scorer or to keyword scoring — this never half-marks a list.

    `focus` is score._focus_description, passed in so the text this sees is
    byte-identical to what the hosted path sees. Without it the two providers
    would be reading different inputs and --compare would mean nothing.

    `profile` is the llm.py model profile that answers the questions (JevK5
    on llama.cpp, any local model, or a hosted API). None means the server
    described by this `jev:` block, which is what it always was.
    """
    import llm
    cfg = merged(cfg)
    if not jobs:
        return 0, None

    try:
        prof = profile_for(cfg, profile, track)
    except llm.LLMError as e:
        return 0, str(e)
    problems = validate(cfg, None if prof["kind"] == "jevk5" else 16)
    if problems:
        # A typo in config.yaml costs accuracy for one run, not the run.
        for p in problems:
            print(f"  ! jev config: {p}")
        print("  ! jev: falling back to the built-in questions for this run")
        cfg = merged(None)

    problem = llm.check(prof)
    if problem:
        return 0, problem
    ok, started = llm.ensure_ready(prof)
    if not ok:
        return 0, "server unavailable"
    try:
        model = llm.decider(prof)
    except (ImportError, llm.LLMError) as e:
        return 0, f"{type(e).__name__}: {e}"
    workers = max(1, int(prof.get("concurrency", 1)))
    print(f"  - jev ({track}): {llm.describe(prof)}"
          + (f", {workers} parallel" if workers > 1 else ""))

    score_q, a_q, b_q, t_q = questions(cfg, track)
    # resume mode: the candidate record goes above every posting and the
    # score is the product of the model's answers (fit.py).
    resume_mode = cfg.get("mode", "resume") == "resume"
    cand = fit.candidate(cfg, facts_path) if resume_mode else ""
    if resume_mode and not cand:
        print(f"  ! jev: {facts_path} unreadable; using the band scorer")
        resume_mode = False
    if resume_mode and track == "career":
        # New applications in your tracker? Refit before scoring.
        fit.maybe_refit(cfg, cand, model)

    # No candidate summary in the state, on either track. Putting one there
    # made every career posting score 0.97-0.99 on the top band: the model
    # matched the band text against the resume instead of against the job,
    # and a posting with a ZERO-character description still came back 0.98.
    text_of = focus or (lambda j: j.get("description") or "")
    min_desc = cfg.get("min_desc_chars", 120)
    floor = cfg["thresholds"]["no_desc_score"]
    t0 = time.perf_counter()

    def one(j):
        """(_jev result, skipped?) for one posting. Raises on model failure."""
        body = text_of(j) or ""
        if len(body.strip()) < min_desc:
            # Nothing to read. Do not ask the model to judge a title: it will
            # answer, confidently, from the title alone. Flag for review and
            # spend no time on it.
            return {"score": floor, "review": True,
                    "reason": f"manual review: no description retrieved "
                              f"({len(body.strip())} chars)"}, True
        if resume_mode:
            return fit.score_one(model, j, body, cfg, track, cand), False
        text = (f"{j.get('title','')} at {j.get('company','')}, "
                f"{j.get('location','')}. {body}")
        c = model.decide(text, score_q)
        f_ = model.decide(text, a_q)
        g_ = model.decide(text, b_q)
        tm = model.decide(text, t_q)
        why = {}
        score, review, reason = compose(c, f_, g_, text, cfg, track, tm,
                                        explain=why)
        return {"score": score, "review": review, "reason": reason,
                "detail": why.get("text", ""), "explain": why}, False

    results = {}
    skipped = 0
    try:
        if workers == 1:
            for i, j in enumerate(jobs):
                results[i] = one(j)
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  jev {i + 1}/{len(jobs)}")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(workers) as pool:
                futs = {pool.submit(one, j): i for i, j in enumerate(jobs)}
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
                    if progress_every and len(results) % progress_every == 0:
                        print(f"  jev {len(results)}/{len(jobs)}")
    except Exception as e:
        # Partial results are worse than none: the caller cannot tell a
        # scored row from an unscored one afterward. Nothing is written.
        return 0, f"{type(e).__name__}: {e}"

    for i, j in enumerate(jobs):
        res, was_skipped = results[i]
        j["_jev"] = res
        skipped += was_skipped
    done = len(jobs)
    if getattr(model, "missing", 0):
        print(f"  - jev: an option letter fell outside the returned "
              f"logprobs on {model.missing} question(s); scored as unlikely")

    dt = time.perf_counter() - t0
    print(f"  - jev: {done} {track} postings in {dt:.0f}s "
          f"({done/max(dt, 1e-6):.1f}/s)"
          + (f", {skipped} skipped with no description" if skipped else "")
          + ("  [server left running]" if started else ""))
    return done, None


def score_bridge(jobs, cfg=None, focus=None, progress_every=25):
    """Back-compatible alias for the bridge track."""
    return score_track(jobs, cfg, focus, track="bridge",
                       progress_every=progress_every)


if __name__ == "__main__":
    # Smoke test without touching the database or the config.
    import json
    cfg = merged(None)
    print("config validates:", validate(cfg) or "yes")
    print(f"bands: {len(cfg['contact']['bands'])}, "
          f"labels: {len(cfg['contact']['labels'])}")
    sample = {"title": "Warehouse Associate - Night Shift", "company": "Quince",
              "location": "Kearny, NJ",
              "description": "Pick, pack and sort returned merchandise. Night "
                             "shift 10pm-6:30am. Lift up to 50 lbs. No prior "
                             "warehouse experience necessary."}
    n, err = score_bridge([sample])
    print(err or json.dumps(sample["_jev"], indent=2))


# ==========================================================================
# Voice: the rules a regex cannot see
# ==========================================================================
# voicecheck.py catches the mechanical breaches perfectly and for free: an
# em dash is a character, "I am excited to apply" is a string. What it
# cannot see is whether a paragraph is a resume bullet turned into prose,
# or whether a sentence claims experience the fact base does not support.
# Those are judgments, which is what jev is for - and it is local, free,
# and emits no tokens, so asking it five questions about six paragraphs
# costs nothing but a moment of GPU.
#
# Running PER PARAGRAPH is the point. A probability for a whole letter
# tells you something is wrong somewhere; a probability per paragraph tells
# you which one to rewrite, and the revision prompt can then quote it.

VOICE_QUESTIONS = {
    "opens_with_filler": {
        "instructions":
            "Does this paragraph lead with enthusiasm, admiration or "
            "interest, instead of leading with something the writer did?",
        "true": "Yes. It opens by saying how keen the writer is, or how "
                "good the employer or the role is.",
        "false": "No. It opens with a task, a method, a decision, a "
                 "finding or a piece of the writer's own experience.",
    },
    "argues_against_self": {
        "instructions":
            "Does this paragraph suggest the writer may not be qualified, "
            "or apologise for a gap?",
        "true": "Yes. It concedes a shortfall, hedges about fit, or "
                "explains away something the writer lacks.",
        "false": "No. It states what the writer has done and leaves it "
                 "there.",
    },
    "resume_in_prose": {
        "instructions":
            "Is this paragraph a list of accomplishments rewritten as "
            "sentences, with no decision, surprise, tradeoff, mistake or "
            "reason the work mattered?",
        "true": "Yes. It recites what was done and the methods used, and "
                "nothing more.",
        "false": "No. It turns on a judgment, a problem found, a choice "
                 "made, or why the work mattered.",
    },
    "praises_employer": {
        "instructions":
            "Does this paragraph compliment the employer, their reputation, "
            "their mission or their work?",
        "true": "Yes, it contains praise directed at the organisation.",
        "false": "No, it does not comment on how good the organisation is.",
    },
    # The gap the iSpot letter found. It opened "iSpot measures how
    # television and streaming advertising actually performs", and nothing
    # caught it: that is not enthusiasm, so opens_with_filler stayed
    # silent, and it is not a compliment, so praises_employer did too. It
    # is simply a paragraph about them rather than about the writer, which
    # breaks two rules at once - open on your own work, and never write at
    # the posting.
    "about_them_not_you": {
        "instructions":
            "Does this paragraph begin by describing the employer, their "
            "product, their customers or their industry, instead of "
            "beginning with something the writer did or decided?",
        "true": "Yes. The opening sentence is about the company or its "
                "field, not about the writer's own work.",
        "false": "No. It opens on the writer's own task, method, decision "
                 "or finding.",
    },
    "vague_claim": {
        "instructions":
            "Does this paragraph make a claim about the writer's ability "
            "with no specific work, number or method behind it?",
        "true": "Yes. It asserts a strength, a passion or a fit in the "
                "abstract.",
        "false": "No. Every claim it makes points at particular work.",
    },
}

VOICE_DEFAULT_THRESHOLD = 0.72


def _paragraphs(text):
    """Body paragraphs with their position, skipping letter furniture."""
    import voicecheck
    body = voicecheck.body(text or "")
    out = []
    for i, p in enumerate(re.split(r"\n\s*\n", body)):
        s = " ".join(p.split())
        if len(s) >= 40:                 # a salutation or a stray line
            out.append((i, s))
    return out


def voice_scan(text, cfg=None, profile=None, threshold=None, track="career"):
    """
    [(rule, paragraph_index, probability, excerpt)] for one draft.

    Returns None when no local model is available, so the caller can say so
    rather than quietly reporting a clean letter it never checked.
    """
    import llm
    cfg = merged(cfg)
    thr = (threshold if threshold is not None
           else float(cfg.get("voice_threshold", VOICE_DEFAULT_THRESHOLD)))
    questions = {**VOICE_QUESTIONS, **(cfg.get("voice") or {})}
    try:
        prof = profile_for(cfg, profile, track)
    except llm.LLMError:
        return None
    if llm.check(prof) or not llm.ensure_ready(prof, quiet=True)[0]:
        return None
    try:
        model = llm.decider(prof)
    except (ImportError, llm.LLMError):
        return None

    # Per-question thresholds, when voicecal.py has fitted them. One
    # cutoff for six different questions assumes they are equally
    # separable, which they are not: "does this praise the employer" is a
    # much sharper judgment than "is this the resume in prose", and the
    # probability at which each becomes a real finding differs.
    per_q = cfg.get("voice_thresholds") or {}

    found = []
    for idx, para in _paragraphs(text):
        for rid, q in questions.items():
            question = {"type": "noul", "instructions": q["instructions"],
                        "criteria": {"true": q["true"], "false": q["false"]}}
            try:
                r = model.decide(para[:3000], question)
            except Exception:
                return None
            p = float(r.get("noul", 0.0))
            if p >= float(per_q.get(rid, thr)):
                found.append((rid, idx, p, para[:150]))
    found.sort(key=lambda f: -f[2])
    return found


def voice_probabilities(text, cfg=None, profile=None, track="career"):
    """
    Every question's probability for every paragraph, unthresholded.

    voicecal.py uses this to fit the cutoffs from labelled drafts. Returns
    None when no model answers.
    """
    import llm
    cfg = merged(cfg)
    questions = {**VOICE_QUESTIONS, **(cfg.get("voice") or {})}
    try:
        prof = profile_for(cfg, profile, track)
    except llm.LLMError:
        return None
    if llm.check(prof) or not llm.ensure_ready(prof, quiet=True)[0]:
        return None
    try:
        model = llm.decider(prof)
    except (ImportError, llm.LLMError):
        return None

    rows = []
    for idx, para in _paragraphs(text):
        scores = {}
        for rid, q in questions.items():
            question = {"type": "noul", "instructions": q["instructions"],
                        "criteria": {"true": q["true"], "false": q["false"]}}
            try:
                scores[rid] = float(
                    model.decide(para[:3000], question).get("noul", 0.0))
            except Exception:
                return None
        rows.append((idx, para, scores))
    return rows


VOICE_FIX = {
    "opens_with_filler": "open on something you did, not on how you feel "
                         "about the role",
    "argues_against_self": "cut the concession; state what is true and stop",
    "resume_in_prose": "this is the resume in sentences. Give it a decision, "
                       "a surprise, a tradeoff or a reason the work mattered",
    "praises_employer": "remove the praise",
    "about_them_not_you": "open on what you did. Their business is not news "
                          "to them",
    "vague_claim": "attach this to specific work, or cut it",
}
