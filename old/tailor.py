#!/usr/bin/env python3
"""
tailor - draft a cover letter and get resume advice for a specific posting.

    python tailor.py list                 # show jobs the scanner found
    python tailor.py 787428               # draft letter for that job id
    python tailor.py 787428 --resume      # resume advice instead of a letter
    python tailor.py 787428 --both        # letter + resume advice
    python tailor.py batch                # work through every unreviewed job
    python tailor.py batch --dry-run      # ...preview it first, bill nothing
    python tailor.py --file posting.txt   # paste-in mode, no scanner needed

Output lands in out/. Read it before you send it.

Requires: pip install anthropic pyyaml
Set ANTHROPIC_API_KEY (billed separately from a Claude Pro subscription).
"""

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import date, datetime, time as _time, timedelta, timezone
from pathlib import Path

import yaml

import env
env.load()

MODEL = "claude-sonnet-4-6"
TODAY = date.today().strftime("%B %-d, %Y" if os.name != "nt" else "%B %d, %Y")
OUT_DIR = Path("out")
DEFAULT_BATCH_MIN_SCORE = 70


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

ANALYZE = """Assess whether this candidate should actually spend time applying to this job.

TODAY'S DATE IS {today}. Compute every date claim against it. A graduation
date earlier than today means the degree IS conferred and the candidate IS a
graduate. Read experience requirements against dates that have already passed.

This is an APPLICATION-VIABILITY decision, not a theoretical possibility test.
The question is: given the candidate's actual record and the posting's stated
requirements, would you recommend spending time submitting this application?

Return ONLY valid JSON, no fences, no preamble:
{{
  "company": "employer",
  "role": "exact title",
  "seniority_fit": "under|good|over",
  "duties": ["5-7 concrete responsibilities in the posting's own words"],
  "hard_requirements": ["degrees, years, certifications, named software, eligibility rules"],
  "keywords": ["terms worth mirroring in application materials"],
  "strongest_matches": [
    {{"requirement": "...", "candidate_fact": "verbatim fact that matches", "why": "one line"}}
  ],
  "gaps": [
    {{"requirement": "...", "severity": "blocking|significant|minor",
      "honest_read": "what the candidate actually has, or nothing"}}
  ],
  "verdict": "one blunt paragraph explaining the practical application decision",
  "recommendation": "apply|manual_review|do_not_apply"
}}

Use the recommendations strictly:

  "apply"
    You would genuinely recommend spending time on the application. The
    candidate meets the role's central duties and most stated requirements. A
    small number of learnable or preferred-skill gaps is fine when the core fit
    is strong.

  "manual_review"
    The posting is incomplete, ambiguous, or genuinely close enough that a
    human should decide. Use this when one unresolved fact could materially
    change the answer. This is NOT a euphemism for a weak application.

  "do_not_apply"
    The expected return on the candidate's time is poor. This does NOT require
    literal impossibility. Use it for structural ineligibility, a substantial
    level mismatch, multiple missing core requirements, or a central required
    competency the candidate does not have. It is acceptable that the person
    could theoretically apply; the question is whether they SHOULD.

IMPORTANT GAP RULES:
- A "blocking" gap is literal eligibility/ineligibility: required license,
  required graduate degree, current-student status, civil-service status,
  expired cohort, impossible location, etc.
- A gap can justify "do_not_apply" without being blocking. For example, if a
  job explicitly requires advanced SQL as a central daily duty and the factual
  record contains no SQL, that can be a decisive significant gap.
- Missing one peripheral or readily learnable tool does not automatically
  reject an otherwise strong early-career fit.
- Multiple significant gaps across the role's core function should normally
  produce "do_not_apply", not "manual_review".
- Do not rescue a weak fit merely because a cover letter could theoretically
  explain adjacent experience.

POSTING COMPLETENESS: {quality}. If this is not "full", default to
"manual_review" unless a visible fact already makes the candidate clearly
ineligible. Never invent missing requirements.

POSTING:
<posting>
{posting}
</posting>

CANDIDATE FACTS (complete — nothing outside this exists):
<facts>
{facts}
</facts>"""

LETTER = """Write a cover letter.

HARD CONSTRAINTS. Breaking any of these makes the output unusable:
1. Every factual claim must trace to the candidate facts below. Do not invent
   employers, tools, systems, metrics, dates, or domain experience.
2. Do not claim familiarity with software the candidate has not listed, even
   if the posting requires it.
3. Do not paper over gaps. Address one honestly or leave it alone. Never imply
   experience that isn't there.
3a. But do not talk the candidate out of the job. This is an application, not
   a self-assessment. Never open by conceding the candidate is underqualified,
   never say the match is weak, never suggest a more junior role, and never
   add meta-commentary like "I am not overstating this" or "I will not
   represent otherwise". State what is true and stop. At most ONE sentence in
   the whole letter may address a gap, and only if it is genuinely
   disqualifying-if-hidden. Silence about a gap is honest; apology is not
   required.
4. Follow the voice rules exactly.
5. Output the letter only, date line through signature. No commentary, no
   explanation, no code fences.

ANALYSIS:
{analysis}

CANDIDATE FACTS:
<facts>
{facts}
</facts>

VOICE RULES:
{voice}

Today is {today}. Address to "Hiring Manager" unless the posting names someone."""

RESUME = """Give the candidate concrete advice for tailoring their resume to
this specific posting.

TODAY'S DATE IS {today}. Any graduation or end date before today has already
happened. Never list a completed degree under "Cannot claim".

Output Markdown with these sections:

## Verdict
Two sentences. Worth applying? What's the single biggest obstacle?

## Bullets to lead with
The 4-6 existing bullets that matter most here, in priority order. For each,
quote the bullet as currently written, then give a rewritten version that
mirrors the posting's language WITHOUT adding any new claim. Show both so the
candidate can see exactly what changed.

## Bullets to cut or bury
What's taking up space that this employer won't care about.

## Keywords currently missing
Terms from the posting that are absent from the resume but that the candidate
could legitimately add based on their real experience. For each, say which
existing experience justifies it. If a keyword can't be justified, list it
under "Cannot claim" instead.

## Cannot claim
Requirements the candidate genuinely lacks. No spin. If one is disqualifying,
say so plainly.

## Skills section rewrite
The skills block as it should read for this application, drawn only from facts
already listed.

RULES: Never invent experience. Never suggest wording that implies a tool or
domain the candidate hasn't used. Rewriting for emphasis is fine; inventing is
not. Quote bullets verbatim when showing the "before".

ANALYSIS:
{analysis}

CANDIDATE FACTS:
<facts>
{facts}
</facts>"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def graduation_status(resume):
    """
    Turn the resume's graduation date into a flat, computed statement.

    Models reliably get date arithmetic wrong when the date is near the current
    year: "May 2026" reads as upcoming even when today is August 2026. So we do
    the comparison here and hand over a conclusion, not a date to reason about.
    """
    raw = str((resume.get("education") or {}).get("graduated", "")).strip()
    m = re.match(r"([A-Za-z]+)\s+(\d{4})", raw)
    if not m:
        return f"Graduation per resume: {raw or 'not stated'}."

    mon, year = MONTHS.get(m.group(1).lower()), int(m.group(2))
    if not mon:
        return f"Graduation per resume: {raw}."

    today = date.today()
    grad = date(year, mon, 1)
    if grad > today:
        months = (grad.year - today.year) * 12 + (grad.month - today.month)
        return (f"STUDENT. Degree not yet conferred; graduates {m.group(1)} "
                f"{year}, about {months} month(s) from now.")

    months = (today.year - year) * 12 + (today.month - mon)
    if months < 1:
        since = "this month"
    elif months < 12:
        since = f"{months} month(s) ago"
    else:
        since = f"{months // 12} year(s), {months % 12} month(s) ago"
    return (f"GRADUATE. The degree was CONFERRED in {m.group(1)} {year} "
            f"({since}). The candidate has completed their degree and holds "
            f"it now. Do NOT describe them as a student, as pre-graduation, "
            f"as graduating soon, or as having an unconferred degree. Do NOT "
            f"list the degree as a gap or as something they cannot claim.")


def die(msg):
    sys.exit(f"error: {msg}")


def load_facts(path):
    if not Path(path).exists():
        die(f"{path} not found. Run from the jobscan folder.")
    data = yaml.safe_load(open(path, encoding="utf-8"))
    voice = "\n".join(f"- {v}" for v in data.pop("voice", []))
    status = graduation_status(data)
    facts = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    # Prepend the computed status so it is the first thing read, and repeat
    # today's date inline.
    facts = (f"CANDIDATE STATUS AS OF {TODAY} (computed, authoritative "
             f"— overrides any inference you would make from the dates below):\n"
             f"  {status}\n\n{facts}")
    return facts, voice


def client():
    try:
        from anthropic import Anthropic
    except ImportError:
        die("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        die("Set ANTHROPIC_API_KEY. This is separate from a Claude Pro plan.")
    return Anthropic()


def ask(cl, prompt, max_tokens=2500):
    r = cl.messages.create(model=MODEL, max_tokens=max_tokens,
                           messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")


def parse_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if not m:
            die("model did not return usable JSON. Try again.")
        return json.loads(m.group(0))


def normalize_recommendation(value):
    """Normalize old analysis files and new viability decisions."""
    rec = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "apply": "apply",
        "strong": "apply",
        "manual_review": "manual_review",
        "review": "manual_review",
        "stretch": "manual_review",   # old prompt's permissive middle state
        "maybe": "manual_review",
        "do_not_apply": "do_not_apply",
        "dont_apply": "do_not_apply",
        "skip": "do_not_apply",
    }.get(rec, rec)


def effective_desc_quality(description, reported="full"):
    """Detect legacy rows that were labeled full but sliced at a fixed cap."""
    q = str(reported or "unknown").strip().lower()
    d = (description or "").strip()
    # Older sources.py/ats.py versions sliced otherwise-full descriptions at
    # exactly 2k/4k/8k chars. Those rows must not masquerade as full postings
    # during the expensive viability stage.
    # 2k/4k/8k are the old fixed slices; 40k is sources.MAX_DESC, which bounds
    # what goes into the committed DB. Text sitting exactly on any of them was
    # cut mid-posting and must not masquerade as complete.
    if q == "full" and len(d) in (2000, 4000, 8000, 40000):
        return "excerpt"
    if q == "full" and (len(d) < 400 or d.endswith(("...", "…"))):
        return "excerpt"
    return q


def _candidate_grad_month(facts):
    months = {m.lower(): i + 1 for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"])}
    m = re.search(r"\bgraduated:\s*([A-Za-z]+)\s+(20\d{2})\b", facts or "", re.I)
    if not m:
        m = re.search(r"\bconferred\s+(?:in\s+)?([A-Za-z]+)\s+(20\d{2})\b",
                      facts or "", re.I)
    if not m or m.group(1).lower() not in months:
        return None
    return int(m.group(2)), months[m.group(1).lower()]


def _graduation_cohort_exclusion(text, facts):
    """High-precision graduate-cohort mismatch, e.g. Dec 2026/Summer 2027."""
    cand = _candidate_grad_month(facts)
    if not cand:
        return ""
    low = re.sub(r"\s+", " ", text or "").lower()
    month_num = {m.lower(): i + 1 for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"])}
    season_num = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}
    date_re = re.compile(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december|spring|summer|fall|autumn|winter)\s+(20\d{2})\b")

    # Only inspect text explicitly defining the candidate graduation cohort.
    for m in re.finditer(r"\bgraduat(?:ing|ion\s+date|es)\b", low):
        seg = low[max(0, m.start() - 45):m.end() + 150]
        dates = []
        for dm in date_re.finditer(seg):
            mon = month_num.get(dm.group(1), season_num.get(dm.group(1)))
            dates.append((int(dm.group(2)), mon))
        if dates:
            lo, hi = min(dates), max(dates)
            if cand < lo or cand > hi:
                return (f"posting targets a graduation cohort from {lo[0]}-{lo[1]:02d} "
                        f"to {hi[0]}-{hi[1]:02d}; candidate graduated "
                        f"{cand[0]}-{cand[1]:02d}")
        else:
            ym = re.search(r"\b(20\d{2})\b", seg)
            if ym and cand[0] != int(ym.group(1)):
                return (f"posting targets {ym.group(1)} graduates; candidate graduated "
                        f"{cand[0]}")
    return ""


def deterministic_gate(posting, facts, desc_quality="full", posted=""):
    """
    Free, high-precision application gates that run before any Sonnet call.

    Return (decision, reason), where decision is manual_review/do_not_apply,
    or (None, "") when the full viability model should decide. Keep this
    intentionally conservative: false negatives here are invisible to Sonnet.
    """
    quality = str(desc_quality or "").strip().lower()
    text = re.sub(r"\s+", " ", posting or "").strip()
    low = text.lower()
    graduate = "GRADUATE." in facts

    # Reuse the deterministic temporal parser so old rows already sitting in
    # seen.db cannot reach Sonnet just because they predate the scanner gate.
    #
    # Split the title off the RAW posting, not off `text`. `text` has already
    # had its newlines collapsed, so text.splitlines()[0] is the entire
    # posting — and _temporal_scan treats a title as high-signal, skipping the
    # season-context test that keeps "DOT Summer Streets" and "pay will fall
    # within" from reading as expired internships. Handing it the whole
    # document as a title bypassed every guard in score.py.
    raw = posting or ""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    try:
        import score as scoring
        temporal = scoring.temporal_exclusion_reason(
            {"title": lines[0] if lines else "",
             "description": raw, "posted": posted or ""})
    except Exception:
        temporal = ""
    if temporal:
        return "do_not_apply", temporal

    if graduate:
        student_only = [
            r"\bmust\s+be\s+(?:a\s+)?currently\s+enrolled\b",
            r"\brequired\s+to\s+be\s+(?:a\s+)?currently\s+enrolled\b",
            r"\bmust\s+currently\s+be\s+enrolled\b",
            r"\bmust\s+be\s+currently\s+pursuing\s+(?:an?\s+)?(?:bachelor|master|doctoral|degree)",
            r"\bcurrently\s+pursuing\s+(?:an?\s+)?(?:bachelor'?s|master'?s|doctoral)\s+degree\s+is\s+required\b",
            r"\bcurrent(?:ly)?\s+enrolled\s+(?:undergraduate|graduate|college|university)\s+student\b",
            r"\bmust\s+be\s+(?:an?\s+)?(?:undergraduate|graduate)\s+student\b",
            r"\breturn(?:ing)?\s+to\s+(?:school|college|university)\s+after\s+(?:the\s+)?(?:internship|program)\b",
        ]
        # Common internship wording states "Enrolled in ..." under a
        # Qualifications/Requirements heading without repeating "must".
        if re.search(r"\b(?:qualifications|requirements|eligibility)\b.{0,350}\benrolled\s+in\s+(?:an?\s+)?(?:top[- ]tier\s+)?(?:college|university|undergraduate|graduate)", low):
            return ("do_not_apply",
                    "posting's qualifications require enrollment in a degree program")
        if any(re.search(p, low) for p in student_only):
            return ("do_not_apply",
                    "posting explicitly requires current student/enrollment status")

        cohort = _graduation_cohort_exclusion(text, facts)
        if cohort:
            return "do_not_apply", cohort

    # Explicit advanced-degree requirements. "preferred" alone is deliberately
    # not caught; Sonnet handles softer education language.
    grad_degree_patterns = [
        r"\b(?:jd|j\.d\.|juris doctor)\b.{0,45}\b(?:required|must|required qualification)\b",
        r"\b(?:required|must have|must possess)\b.{0,45}\b(?:jd|j\.d\.|juris doctor)\b",
        r"\b(?:phd|ph\.d\.|doctorate|doctoral degree)\b.{0,45}\b(?:required|must)\b",
        r"\b(?:required|must have|must possess)\b.{0,45}\b(?:phd|ph\.d\.|doctorate|doctoral degree)\b",
        r"\bmaster'?s degree\b.{0,45}\b(?:required|must)\b",
        r"\b(?:required|must have|must possess)\b.{0,45}\bmaster'?s degree\b",
    ]
    if any(re.search(p, low) for p in grad_degree_patterns):
        return ("do_not_apply", "posting explicitly requires a graduate/professional degree")

    license_patterns = [
        r"\bmust\s+be\s+(?:currently\s+)?admitted\s+to\s+(?:the\s+)?(?:[a-z ]+\s+)?bar\b",
        r"\bactive\s+(?:law\s+)?license\s+(?:is\s+)?required\b",
        r"\b(?:cpa|certified public accountant)\b.{0,35}\b(?:required|must)\b",
        r"\b(?:required|must have|must possess)\b.{0,35}\b(?:cpa|certified public accountant)\b",
        r"\bprofessional engineer(?:ing)?\s+(?:license|licensure)\b.{0,25}\brequired\b",
    ]
    if any(re.search(p, low) for p in license_patterns):
        return ("do_not_apply", "posting explicitly requires a professional license not in the candidate record")

    civil_service = [
        r"\bmust\s+be\s+permanent\s+in\s+(?:the\s+)?(?:civil service\s+)?title\b",
        r"\bmust\s+be\s+reachable\s+on\s+(?:the\s+)?(?:civil service\s+)?list\b",
        r"\bonly\s+permanent\s+(?:city|agency|civil service)\s+employees\b",
    ]
    if any(re.search(p, low) for p in civil_service):
        return ("do_not_apply", "posting explicitly requires incumbent civil-service eligibility")

    if quality in ("excerpt", "none", "title", "title_only"):
        return ("manual_review",
                f"stored posting is {quality}; full requirements are not available")

    return None, ""


def gate_reasons(analysis, mode):
    """
    Return reasons to stop before drafting. The default gate is deliberately
    precision-oriented: only APPLY gets automatic materials.
    """
    if mode == "off":
        return []
    reasons = []
    rec = normalize_recommendation(analysis.get("recommendation"))

    if mode == "both" and rec != "apply":
        reasons.append(f'recommendation is "{rec or "unknown"}"')
    elif mode == "skip" and rec == "do_not_apply":
        reasons.append('recommendation is "do_not_apply"')

    if mode in ("blocking", "both"):
        for g in analysis.get("gaps", []):
            if isinstance(g, dict) and \
                    str(g.get("severity", "")).lower() == "blocking":
                reasons.append(f"blocking gap: {g.get('requirement', '?')}")
    return reasons


def _analysis_state(analysis):
    rec = normalize_recommendation((analysis or {}).get("recommendation"))
    if rec == "manual_review":
        return REVIEW
    if rec == "do_not_apply":
        return SKIPPED
    return DONE


def _default_batch_min_score(config_path="config.yaml"):
    """Strong-score threshold is the default handoff from discovery to Sonnet."""
    try:
        cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
        return int((cfg.get("tailoring") or {}).get(
            "batch_min_score",
            (cfg.get("scoring") or {}).get("strong_at", DEFAULT_BATCH_MIN_SCORE)))
    except Exception:
        return DEFAULT_BATCH_MIN_SCORE


def out_dir():
    """Today's output folder. Keeps out/ navigable after a few dozen drafts."""
    d = OUT_DIR / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def existing_materials(stem, ext="md"):
    """Files already written for this job, across all date folders."""
    if not OUT_DIR.exists():
        return []
    return (sorted(OUT_DIR.glob(f"*/{stem}__*.{ext}"))
            + sorted(OUT_DIR.glob(f"{stem}__*.{ext}")))


# The four states a job can be in. Manual review is deliberately distinct from
# a rejected application: both stop drafting, but they mean different things.
NEW, REVIEW, SKIPPED, DONE = "new", "review", "skipped", "done"


def review_status(stem):
    """Return (state, label) for a job across all prior tailor runs."""
    drafts = existing_materials(stem, "md")
    if drafts:
        kinds = {f.name.rsplit("__", 1)[-1].replace(".md", "") for f in drafts}
        have = "".join(k for k, n in (("L", "letter"), ("R", "resume"))
                       if n in kinds)
        return DONE, f"done {have}" if have else "done"

    analyses = existing_materials(stem, "json")
    if analyses:
        try:
            analysis = json.loads(analyses[-1].read_text(encoding="utf-8"))
            rec = normalize_recommendation(analysis.get("recommendation"))
            if rec == "manual_review":
                return REVIEW, "review"
            if rec == "do_not_apply":
                return SKIPPED, "skipped"
        except Exception:
            pass
        # Old analysis files without a recognized recommendation were reviewed,
        # but should not silently become draft candidates again.
        return REVIEW, "review"
    return NEW, ""


def short_id(uid):
    """
    A copy-pasteable handle for a job.

    Workday uids carry the whole URL path, e.g.
    "workday:rb:/job/New-York-NY/Regulatory-Data-Analyst_R-0000032890-1".
    The requisition number at the end is the part worth showing.
    """
    tail = uid.split(":")[-1]
    m = re.search(r"(R-\d[\w-]*)$", tail)
    if m:
        return m.group(1)
    # A 14-char tail is unique enough across a few thousand jobs and keeps the
    # column aligned. store.get_job resolves it by suffix.
    return tail[-14:] if len(tail) > 14 else tail


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "job").lower()).strip("_")[:40]


# --------------------------------------------------------------------------
# Date filtering for `list`
#
# store.record() writes first_seen as UTC. You are not in UTC, and the cron
# job runs twice a day, so the evening run lands on the NEXT UTC date — ask
# for "yesterday" naively and you get half of one run and half of another.
# So: parse a LOCAL calendar day, convert to a half-open UTC interval, and
# compare lexicographically against the stored ISO strings.
# --------------------------------------------------------------------------
def parse_day(s):
    """'today' | 'yesterday' | '-2' | 'YYYY-MM-DD' -> a local date."""
    s = (s or "").strip().lower()
    today = date.today()
    if s in ("today", "t"):
        return today
    if s in ("yesterday", "y"):
        return today - timedelta(days=1)
    if re.fullmatch(r"-\d+", s):
        return today - timedelta(days=int(s[1:]))
    try:
        return date.fromisoformat(s)
    except ValueError:
        die(f"could not read {s!r} as a date. Use YYYY-MM-DD, 'today', "
            f"'yesterday', or '-3' for three days ago.")


def _utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def day_bounds(first, last=None):
    """Local day (or span) -> (since, until) UTC ISO, half-open."""
    last = last or first
    start = datetime.combine(first, _time.min).astimezone()
    end = datetime.combine(last, _time.min).astimezone() + timedelta(days=1)
    return _utc_iso(start), _utc_iso(end)


def local_day(ts):
    """Stored UTC ISO timestamp -> local 'YYYY-MM-DD', or '' if unparseable."""
    try:
        return datetime.fromisoformat(ts).astimezone().date().isoformat()
    except (TypeError, ValueError):
        return ""


def cmd_list(args):
    import store
    conn = store.connect()

    if getattr(args, "days_seen", False):
        have = store.day_counts(conn, local_day, limit=60)
        if not have:
            die("no jobs stored yet. Run `python main.py` first.")
        print(f"{'DAY':<12} {'JOBS':>6}")
        for d, n in have.items():
            print(f"{d:<12} {n:>6}")
        print(f"\n  python tailor.py list --on {next(iter(have))}")
        return

    # --on / --since / --days -> UTC bounds on first_seen.
    since = until = None
    span = ""
    if getattr(args, "days", None):
        first = date.today() - timedelta(days=args.days - 1)
        since, until = day_bounds(first, date.today())
        span = f"last {args.days} day(s), {first} to {date.today()}"
    elif getattr(args, "on", None):
        d = parse_day(args.on)
        since, until = day_bounds(d)
        span = f"found on {d}"
    elif getattr(args, "since", None):
        d = parse_day(args.since)
        since, until = day_bounds(d, date.today())
        span = f"found since {d}"

    want = args.status or ("reviewed" if args.drafted else None)
    # Filtering by state needs the whole table, not just the top slice.
    scan = 10000 if want else args.limit
    rows = store.recent(conn, limit=scan, min_score=args.min_score,
                        order="first_seen" if args.newest else "score",
                        since=since, until=until)
    if span:
        print(f"({span}; {len(rows)} row(s))")
    if not rows and span:
        have = store.day_counts(conn, local_day)
        if have:
            print("Days with stored jobs (most recent first):")
            for d, n in have.items():
                print(f"  {d}  {n:>5}")
        die("nothing found in that window.")
    if want:
        def match(r):
            st = review_status(f"{slug(r['company'])}__{slug(r['title'])}")[0]
            if want == "reviewed":
                return st in (REVIEW, SKIPPED, DONE)
            return st == want
        rows = [r for r in rows if match(r)][:args.limit]
        if not rows:
            die(f"no jobs with status {want!r}.")
    if not rows:
        die("no jobs stored yet. Run `python main.py` (without --dry-run) first.")
    # The date column only appears when it is the thing you asked about;
    # default output stays the width people already have in their terminal.
    show_date = bool(span) or args.newest
    dcol = f"{'FOUND':<12}" if show_date else ""
    print(f"{'FIT':<5} {'STATUS':<10} {dcol}{'ID':<16} {'TITLE':<38} COMPANY")
    print("-" * (96 + (12 if show_date else 0)))
    indent = 29 + (12 if show_date else 0)
    if getattr(args, "reviewed", False) or getattr(args, "todo", False):
        want_reviewed = bool(args.reviewed)
        def legacy_match(r):
            st = review_status(f"{slug(r['company'])}__{slug(r['title'])}")[0]
            return (st is not NEW) if want_reviewed else (st is NEW)
        rows = [r for r in rows if legacy_match(r)]
        if not rows:
            print("(nothing matches that filter)")
            return

    undrafted = None
    for r in rows:
        jid = r["uid"] if getattr(args, "uids", False) else short_id(r["uid"])
        sc = r.get("score")
        fit = f"{sc:>3}  " if sc is not None else "  -  "
        stem = f"{slug(r['company'])}__{slug(r['title'])}"
        state, status = review_status(stem)
        if state is NEW and not r.get("description"):
            status = "no desc"
        # Only an untouched job with text to read is a candidate for "next".
        # A skipped job has already been judged; do not send them back to it.
        elif state is NEW and undrafted is None:
            undrafted = jid
        found = f"{local_day(r.get('first_seen')) or '?':<12}" if show_date else ""
        print(f"{fit}{status:<10} {found}{jid:<16} "
              f"{r['title'][:36]:<38} {r['company'][:18]}")
        if (args.drafted or getattr(args, "urls", False)) and r.get("url"):
            print(f"{'':>{indent}}Apply: {r['url']}")
        if args.drafted:
            for f in (existing_materials(stem, "md")
                      or existing_materials(stem, "json")):
                print(f"{'':>{indent}}{f}")

    states = [review_status(f"{slug(r['company'])}__{slug(r['title'])}")[0]
              for r in rows]
    n_done = states.count(DONE)
    n_review = states.count(REVIEW)
    n_skip = states.count(SKIPPED)
    print(f"\n{n_done} drafted, {n_review} manual-review, {n_skip} rejected, "
          f"{states.count(NEW)} not reviewed  ({OUT_DIR}/)")
    if undrafted:
        print(f"Next undrafted:  python tailor.py {undrafted} --both")


# --------------------------------------------------------------------------

def process_job(posting, company, role, args, facts, voice, cl,
                url="", quiet=False, desc_quality="full", posted=""):
    """
    Run a free deterministic gate, then one full-posting Sonnet viability call.
    Only an APPLY decision gets letter/resume generation unless --force/--no-gate.
    """
    def say(*a):
        if not quiet:
            print(*a)

    bypass = bool(args.force or args.no_gate or args.gate == "off")
    deterministic = (None, "") if bypass else deterministic_gate(
        posting, facts, desc_quality, posted=posted)

    if deterministic[0]:
        decision, reason = deterministic
        company = company or ""
        role = role or ""
        analysis = {
            "company": company,
            "role": role,
            "seniority_fit": "unknown",
            "duties": [],
            "hard_requirements": [],
            "keywords": [],
            "strongest_matches": [],
            "gaps": [{"requirement": reason,
                      "severity": "blocking" if decision == "do_not_apply" else "significant",
                      "honest_read": "deterministic pre-model gate"}],
            "verdict": reason,
            "recommendation": decision,
            "decision_source": "deterministic_pre_model_gate",
        }
        day = out_dir()
        stem = f"{slug(company)}__{slug(role)}"
        analysis_path = day / f"{stem}__analysis.json"
        if url:
            analysis["posting_url"] = url
        if posted:
            analysis["posted"] = posted
        analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        state = REVIEW if decision == "manual_review" else SKIPPED
        return state, analysis_path, [], [reason]

    say("Analyzing full posting for application viability...")
    analysis = parse_json(ask(
        cl, ANALYZE.replace("{posting}", posting).replace("{facts}", facts)
             .replace("{today}", TODAY)
             .replace("{quality}", str(desc_quality or "unknown")), 2600))

    # Normalize any stale model vocabulary so the gate is fail-safe.
    normalized = normalize_recommendation(analysis.get("recommendation"))
    if normalized:
        analysis["recommendation"] = normalized

    company = company or analysis.get("company", "")
    role = role or analysis.get("role", "")

    say(f"\n  Level fit: {analysis.get('seniority_fit', '?')}")
    say(f"  Decision:  {analysis.get('recommendation', '?')}")
    for g in analysis.get("gaps", []):
        if isinstance(g, dict):
            say(f"  [{g.get('severity', '?'):<11}] {g.get('requirement', '')}")
    if analysis.get("verdict") and not quiet:
        print("\n" + textwrap.fill(analysis["verdict"], 76,
                                   initial_indent="  ", subsequent_indent="  "))

    day = out_dir()
    stem = f"{slug(company)}__{slug(role)}"
    analysis_path = day / f"{stem}__analysis.json"
    if url and not analysis.get("posting_url"):
        analysis["posting_url"] = url
    if posted and not analysis.get("posted"):
        analysis["posted"] = posted
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    mode = "off" if bypass else args.gate
    stop = gate_reasons(analysis, mode)
    if stop:
        return _analysis_state(analysis), analysis_path, [], stop

    written = []
    payload = dict(analysis=json.dumps(analysis, indent=2), facts=facts,
                   today=TODAY)
    if not args.resume:
        say("\nDrafting letter...")
        letter = ask(cl, LETTER.format(voice=voice, **payload), 2000)
        f = day / f"{stem}__letter.md"
        f.write_text(letter.strip() + "\n", encoding="utf-8")
        written.append(f)
    if args.resume or args.both:
        say("Writing resume advice...")
        advice = ask(cl, RESUME.format(**payload), 3000)
        f = day / f"{stem}__resume.md"
        f.write_text(advice.strip() + "\n", encoding="utf-8")
        written.append(f)
    return DONE, analysis_path, written, []


def cmd_batch(args):
    """
    Precision-oriented bulk tailoring.

    Discovery remains generous in score.py/digest.py, but batch tailoring only
    hands strong-scoring jobs to Sonnet by default. One viability call decides
    APPLY / MANUAL REVIEW / DO NOT APPLY. Only APPLY gets drafting calls.
    """
    import store
    conn = store.connect()

    effective_min = args.min_score
    if effective_min is None:
        effective_min = _default_batch_min_score(args.config)
        print(f"batch precision gate: score >= {effective_min} "
              f"(override with --min-score)")

    rows = store.recent(conn, limit=10000, min_score=effective_min,
                        order="first_seen" if args.newest else "score")

    todo = []
    for r in rows:
        stem = f"{slug(r['company'])}__{slug(r['title'])}"
        if review_status(stem)[0] is not NEW:
            continue
        if not (r.get("description") or "").strip():
            continue
        todo.append(r)
    todo = todo[:args.limit]

    if not todo:
        die(f"nothing to do — no unreviewed jobs with descriptions at score >= {effective_min}.")

    lo = min(r.get("score") or 0 for r in todo)
    hi = max(r.get("score") or 0 for r in todo)
    print(f"{len(todo)} unreviewed job(s), scores {lo}-{hi}.")
    print("Cost model: up to 1 Sonnet viability call per full posting; "
          "only APPLY jobs get letter/resume calls. Incomplete postings and "
          "obvious eligibility failures can stop before Sonnet.")
    if args.dry_run:
        for r in todo:
            q = effective_desc_quality(r.get("description") or "",
                                       r.get("desc_quality") or "?")
            print(f"  {(r.get('score') or 0):>3}  {short_id(r['uid']):<16} "
                  f"{r['title'][:42]:<44} {r['company'][:18]:<20} {q}")
        print("\n--dry-run: nothing analyzed, nothing billed.")
        return
    if not args.yes:
        try:
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                return print("Cancelled.")
        except EOFError:
            return print("Cancelled (no tty; pass --yes to run unattended).")

    facts, voice = load_facts(args.facts)
    cl = client()
    done = review = skipped = failed = 0

    for i, r in enumerate(todo, 1):
        head = f"[{i}/{len(todo)}] {(r.get('score') or 0):>3}  {r['title'][:52]}"
        print(f"\n{head}\n      {r['company'][:40]}")
        posting = (f"{r['title']}\n{r['company']}\n{r['location']}\n\n"
                   f"{r['description']}")
        try:
            state, apath, written, stop = process_job(
                posting, r["company"], r["title"], args, facts, voice, cl,
                url=r.get("url", ""), quiet=True,
                desc_quality=effective_desc_quality(
                    r.get("description") or "", r.get("desc_quality") or "unknown"),
                posted=r.get("posted") or "")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Everything written so far is kept.")
            break
        except Exception as e:
            failed += 1
            print(f"      ! failed: {type(e).__name__}: {e}")
            continue
        if state is REVIEW:
            review += 1
            print(f"      manual review — {stop[0] if stop else 'analysis saved; no drafts'}")
        elif state is SKIPPED:
            skipped += 1
            print(f"      do not apply — {stop[0] if stop else 'gated'}")
        else:
            done += 1
            kinds = ', '.join(f.name.rsplit('__', 1)[-1][:-3] for f in written)
            print(f"      APPLY — drafted {kinds}")

    print(f"\n{'=' * 60}")
    print(f"{done} drafted, {review} manual-review, {skipped} rejected, {failed} failed.")
    print(f"Files in {OUT_DIR}/. Review before sending:")
    print("  python tailor.py list --status done")
    if review:
        print("Manual-review queue:")
        print("  python tailor.py list --status review --urls")


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("job", nargs="?",
                    help="job id, or 'list', or 'batch' to work through every "
                         "unreviewed job")
    ap.add_argument("--file", help="read posting from a text file instead")
    ap.add_argument("--resume", action="store_true", help="resume advice, no letter")
    ap.add_argument("--both", action="store_true", help="letter AND resume advice")
    ap.add_argument("--gate", choices=("both", "skip", "blocking", "off"),
                    default="both",
                    help="draft gate. both (default) = automatic materials ONLY "
                         "for APPLY; manual_review/do_not_apply stop after analysis. "
                         "skip = only do_not_apply stops; blocking = only literal "
                         "blocking gaps stop; off = never abort")
    ap.add_argument("--no-gate", action="store_true", help="same as --gate off")
    ap.add_argument("--force", action="store_true", help="same as --gate off")
    ap.add_argument("--reviewed", action="store_true",
                    help="with `list`: show only jobs already analyzed/drafted")
    ap.add_argument("--todo", action="store_true",
                    help="with `list`: show only jobs not yet reviewed")
    ap.add_argument("--uids", action="store_true",
                    help="with `list`: print full uids instead of short ids")
    ap.add_argument("--status", choices=("new", "review", "skipped", "done", "reviewed"),
                    help="with `list`: new = untouched; review = analysis says "
                         "manual review; skipped = do not apply; done = has drafts; "
                         "reviewed = any analyzed/drafted state")
    ap.add_argument("--drafted", action="store_true",
                    help="same as --status reviewed, and prints links + file paths")
    ap.add_argument("--newest", action="store_true",
                    help="with `list`: order by date found instead of fit")
    ap.add_argument("--on", metavar="DAY",
                    help="with `list`: only jobs first seen on this LOCAL day. "
                         "YYYY-MM-DD, 'today', 'yesterday', or '-3'")
    ap.add_argument("--since", metavar="DAY",
                    help="with `list`: only jobs first seen on or after this day")
    ap.add_argument("--days", type=int, metavar="N",
                    help="with `list`: only jobs from the last N days")
    ap.add_argument("--days-seen", action="store_true",
                    help="with `list`: print how many jobs were stored per day "
                         "and exit")
    ap.add_argument("--urls", action="store_true",
                    help="with `list`: print each posting's application URL")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--min-score", type=int,
                    help="minimum cheap-scorer fit. For batch only, default is "
                         "tailoring.batch_min_score (or scoring.strong_at, usually 70)")
    ap.add_argument("--config", default="config.yaml",
                    help="config used to resolve the default batch score gate")
    ap.add_argument("--facts", default="resume.yaml")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="with `batch`: skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true",
                    help="with `batch`: list what would be processed, bill nothing")
    args = ap.parse_args()

    if args.job == "list":
        return cmd_list(args)
    if args.job == "batch":
        if "--limit" not in sys.argv:
            args.limit = 25          # a sane default spend; override with --limit
        return cmd_batch(args)

    # ---- get the posting text -------------------------------------------
    company = role = ""
    if args.file:
        posting = Path(args.file).read_text(encoding="utf-8")
    elif args.job:
        import store
        conn = store.connect()
        job = store.get_job(conn, args.job)
        if not job:
            die(f"no job {args.job} in seen.db. Try `python tailor.py list`.")
        if not (job.get("description") or "").strip():
            die(f"no description stored for {args.job}.\n"
                f"       Open {job['url']}\n"
                f"       and save the text, then: python tailor.py --file posting.txt")
        posting = f"{job['title']}\n{job['company']}\n{job['location']}\n\n{job['description']}"
        company, role = job["company"], job["title"]
        print(f"{job['title']} — {job['company']}")
        if job.get("url"):
            print(f"Apply: {job['url']}")
    else:
        ap.print_help()
        return

    if len(posting.strip()) < 150:
        die("posting text is too short to analyze.")

    facts, voice = load_facts(args.facts)
    cl = client()

    # ---- analyze + draft (shared with `batch`) ---------------------------
    state, analysis_path, written, stop = process_job(
        posting, company, role, args, facts, voice, cl,
        url=(job or {}).get("url", "") if args.job else "",
        desc_quality=effective_desc_quality(
            (job or {}).get("description") or "",
            (job or {}).get("desc_quality") or "full") if args.job else "full",
        posted=((job or {}).get("posted") or "") if args.job else "")

    # REVIEW used to fall through to the "Wrote:" success path, so a gated job
    # printed an analysis path, "check every claim before sending", and exit 0
    # — with no letter and no hint that drafting had been stopped.
    if state in (SKIPPED, REVIEW):
        print("\n" + "=" * 70)
        label = "MANUAL REVIEW" if state is REVIEW else "DO NOT APPLY"
        print(f"{label}: stopped before drafting. Reasons:")
        for r in stop:
            print(f"  - {r}")
        print(f"\nAnalysis saved: {analysis_path}")
        print("No letter or resume advice was generated.")
        if state is REVIEW:
            print("Read the analysis and the posting; this one needs your call.")
        print("\nDisagree? Draft it anyway:")
        target = args.job or f"--file {args.file}"
        extra = "--both" if args.both else ("--resume" if args.resume else "")
        print(f"  python tailor.py {target} {extra} --force".replace("  ", " ", 1))
        print("=" * 70)
        # 3 = rejected, 4 = needs a human. Distinct so a wrapper script can
        # tell "not worth applying" from "I could not decide".
        sys.exit(3 if state is SKIPPED else 4)

    print("\nWrote:")
    for f in [analysis_path] + written:
        print(f"  {f}")
    print("\nCheck every claim against your actual resume before sending.")


if __name__ == "__main__":
    main()
