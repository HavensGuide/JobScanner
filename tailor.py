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

    python tailor.py 787428 --model qwen  # one-off: use this model profile

Output lands in out/. Read it before you send it.

Which model does the work is config.yaml's `ai:` block: `tailor` for all
three steps, or tailor_analyze / tailor_letter / tailor_resume separately
(a cheap model for the viability call, a strong one for the letter). Any
provider in llm.py works, local or hosted. With no `ai:` block this uses
Anthropic's Sonnet as before, which needs ANTHROPIC_API_KEY (billed
separately from a Claude Pro subscription).

Requires: pip install requests pyyaml
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

import llm

STEPS = ("analyze", "letter", "resume")
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

KEEP IT SHORT. Every "honest_read" is at most two sentences, and there are
at most four gaps. A long reply gets cut off at the token ceiling and the
whole analysis is lost.

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

The decision to apply has already been made. Do not re-litigate it, do not
open by assessing the fit, and do not tell the candidate whether to apply.
Write the letter.

POSTING:
<posting>
{posting}
</posting>

CANDIDATE FACTS:
<facts>
{facts}
</facts>

{analysis}

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

REWRITE = """Rewrite the document below, following the instruction from the
candidate.

Their instruction is the point of this pass. Do what they asked. Do not
also take the opportunity to restructure, re-order or re-argue anything
they did not mention: they have read this draft and want one thing changed.

Every factual claim must still trace to the candidate facts. If what they
ask for cannot be done without inventing something, do as much of it as the
facts allow and leave the rest alone.

Output the rewritten document only, nothing else. No commentary, no summary
of what you changed, no code fences.

WHAT THEY WANT CHANGED:
{notes}

THE CURRENT DRAFT:
{draft}

CANDIDATE FACTS (complete — nothing outside this exists):
<facts>
{facts}
</facts>

VOICE RULES:
{voice}"""

REVISE = """Below is a cover letter you wrote, and a list of specific places
where it breaks the candidate's voice rules.

Fix exactly those. Do not rewrite anything else: not the structure, not the
examples, not the claims, not the paragraphs that were not flagged. Every
fact must stay as it is.

Each item names the rule and quotes the text. Where a sentence has to be
rephrased to remove a banned construction, keep what it was saying and
change how it says it. Do not solve a problem by deleting the content.

Output the corrected letter only, date line through signature. No
commentary, no list of what you changed, no code fences.

WHAT TO FIX:
{problems}

THE LETTER:
{draft}

VOICE RULES, for reference:
{voice}"""

# The advice prompt above tells the candidate what to change. This one makes
# the change: a complete resume, ordered and worded for one posting, that
# docs.py renders straight to .docx. Everything in it has to already exist in
# the fact base, so this reorders, selects and rephrases. It never adds.
TAILORED = """Produce a complete, tailored one-page resume for this posting.

You are not giving advice. You are writing the finished document.

HARD CONSTRAINTS. Breaking any of these makes the output unusable:
1. Every claim must trace to the candidate facts below. Do not invent
   employers, roles, tools, systems, metrics, dates, or domains.
2. Do not list a tool, language or method the facts do not contain, however
   central it is to the posting. A missing skill is simply absent.
3. Rewording an existing bullet to match the posting's vocabulary is correct.
   Upgrading what it describes is not. Descriptive work does not become
   predictive modelling, data preparation does not become model estimation,
   and support does not become ownership.
4. Numbers are copied exactly as the facts state them. Never round, inflate
   or invent a figure.
5. Follow the voice rules exactly.
6. Output the resume only. No commentary, no explanation, no code fences.

WHAT TO TAILOR:
- Pick the 3 to 5 roles that matter for THIS posting and drop the rest.
  A short resume aimed at the job beats a complete one.
- Within each role, keep the 2 to 4 bullets that speak to the posting's
  duties, in priority order. Drop the others.
- Rewrite the tagline and summary for this specific role.
- Group the skills so the ones the posting asks for come first.
- Include a project only when it demonstrates something the posting wants.

OUTPUT FORMAT. Use exactly this Markdown structure, because it is parsed:

# {name}

A one-line tagline in title case, pipe-separated, aimed at this role

## Summary

Two sentences. What the candidate builds or analyses, and the evidence.

## Core Skills

**Group name**
Item | Item | Item

**Group name**
Item | Item | Item

## Experience

### Role, Organization
Start date - End date

- Bullet.
- Bullet.

## Projects

### Project name
Stack, comma separated

- Bullet.

## Education

Degree, school
Graduation date

Do not write a contact line. The renderer adds it from the fact base.

POSTING:
<posting>
{posting}
</posting>

CANDIDATE FACTS (complete — nothing outside this exists):
<facts>
{facts}
</facts>

{analysis}

VOICE RULES:
{voice}

Today is {today}."""


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


def client(args=None):
    """
    {step: model profile} for analyze / letter / resume.

    --model PROFILE pins all three for one run. Anything unusable stops here,
    before a viability call is spent on a job whose letter cannot be written.
    """
    try:
        root = yaml.safe_load(open(getattr(args, "config", None)
                                   or "config.yaml", encoding="utf-8")) or {}
    except FileNotFoundError:
        root = {}
    pinned = getattr(args, "model", None)
    out = {}
    if pinned and pinned not in (root.get("models") or {}):
        die(f"--model {pinned}: no such profile under models: in config.yaml "
            f"(defined: {', '.join(root.get('models') or {}) or 'none'})")
    for step in STEPS:
        try:
            out[step] = (llm._lookup(root, pinned, f"tailor_{step}") if pinned
                         else llm.resolve(root, f"tailor_{step}"))
        except llm.LLMError as e:
            die(str(e))
    for step, p in out.items():
        problem = llm.check(p)
        if problem:
            die(f"{step} model {llm.describe(p)}: {problem}")
    for p in {id(p): p for p in out.values()}.values():
        ok, _ = llm.ensure_ready(p, quiet=True)
        if not ok:
            die(f"{llm.describe(p)} is not answering. Start it, or pick "
                f"another profile with --model.")
    names = {s: p["name"] for s, p in out.items()}
    if len(set(names.values())) == 1:
        print(f"model: {llm.describe(out['analyze'])}")
    else:
        print("models: " + ", ".join(f"{s}={n}" for s, n in names.items()))
    return out


def ask(cl, prompt, max_tokens=2500, step="analyze"):
    p = cl[step]
    return llm.complete(p, "", prompt,
                        int(p.get("max_tokens") or max_tokens))


def parse_json(text, what="analysis"):
    """
    The model's JSON, recovered where possible.

    Three things go wrong in practice. The model wraps the object in prose
    or fences. It emits a trailing comma. Or the response is cut off at the
    token ceiling part-way through a string, which is what a long gaps
    table does: the old code then threw a JSONDecodeError traceback at the
    user, having already paid for the call.

    Truncation is recoverable. The content up to the cut is real, so the
    repair closes whatever was left open and keeps it, rather than throwing
    away a paid response because its last field is incomplete.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        _json_failed(cleaned, what, "contained no JSON object")
    body = cleaned[start:]

    # A balanced scan beats a greedy regex: it stops at the matching brace
    # instead of the last one in the file, so trailing prose is dropped.
    end = _balanced_end(body)
    if end is not None:
        candidate = body[:end]
        for attempt in (candidate, re.sub(r",(\s*[}\]])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    repaired = _close_truncated(body)
    if repaired is not None:
        try:
            out = json.loads(repaired)
        except json.JSONDecodeError:
            out = None
        # A reply cut off in its first field repairs to {} or near enough.
        # Returning that would hand the pipeline an analysis with no
        # decision in it, which is worse than admitting the call failed.
        if isinstance(out, dict) and (out.get("recommendation")
                                      or out.get("verdict") or out.get("gaps")):
            missing = [k for k in ("recommendation", "verdict", "gaps",
                                   "strongest_matches") if not out.get(k)]
            print("  ! the model's reply was cut off; recovered what arrived."
                  + (f" Missing: {', '.join(missing)}." if missing else ""))
            return out
    _json_failed(cleaned, what, "could not be parsed, even after repairing a cut-off reply")


def _scan(s):
    """Walk JSON tracking string state. Yields (index, char, depth, in_str)."""
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            yield i, ch, depth, True
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        yield i, ch, depth, False


def _balanced_end(s):
    """Index just past the brace that closes the first one, or None."""
    for i, ch, depth, in_str in _scan(s):
        if not in_str and ch in "}]" and depth == 0:
            return i + 1
    return None


def _close_truncated(s):
    """
    Close a JSON object that stops mid-way, discarding the partial tail.

    Cuts back to the last point where a value was complete, then closes
    every structure still open. What survives is everything the model had
    finished saying.
    """
    stack, cut, in_str = [], None, False
    for i, ch, _d, instr in _scan(s):
        in_str = instr
        if instr:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            cut = i + 1
        elif ch in "}]":
            if stack:
                stack.pop()
            cut = i + 1
        elif ch == ",":
            cut = i            # a comma means the value before it finished
    if not stack or cut is None:
        return None
    head = s[:cut].rstrip().rstrip(",")
    return head + "".join(reversed(stack))


def _json_failed(raw, what, why):
    path = out_dir() / f"_failed_{what}.txt"
    try:
        path.write_text(raw, encoding="utf-8")
        where = f"\nThe raw reply is in {path} if you want to look."
    except OSError:
        where = ""
    die(f"the model's {what} {why}. The call was made and billed, so try "
        f"again or use a different model with --model.{where}")


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
    Free, high-precision application gates that run before any model call.

    Return (decision, reason), where decision is manual_review/do_not_apply,
    or (None, "") when the full viability model should decide. Keep this
    intentionally conservative: false negatives here are invisible to the model.
    """
    quality = str(desc_quality or "").strip().lower()
    text = re.sub(r"\s+", " ", posting or "").strip()
    low = text.lower()
    graduate = "GRADUATE." in facts

    # Reuse the deterministic temporal parser so old rows already sitting in
    # seen.db cannot reach the model just because they predate the scanner gate.
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
    # not caught; the model handles softer education language.
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
    """Strong-score threshold is the default handoff from discovery to the tailor model."""
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
    track = None if args.track == "all" else args.track
    rows = store.recent(conn, limit=scan, min_score=args.min_score,
                        order="first_seen" if args.newest else "score",
                        since=since, until=until, track=track)
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
    Run a free deterministic gate, then one full-posting viability call.
    Only an APPLY decision gets letter/resume generation unless --force/--no-gate.
    """
    def say(*a):
        if not quiet:
            print(*a)

    # Repair a thin posting before judging it.
    #
    # Adzuna truncates descriptions, so the gate below would stop with
    # "stored posting is excerpt" on a job that is perfectly draftable once
    # the real text is fetched. Stopping and telling the user to go run
    # enrich.py themselves is not a decision, it is homework. Fetch it here,
    # and only give up if the fetch fails too.
    #
    # Costs one HTTP request and no model tokens. --no-fetch skips it.
    if (url and desc_quality != "full"
            and not getattr(args, "no_fetch", False)):
        try:
            import enrich as _enrich
            job = {"uid": getattr(args, "job", "") or "", "url": url,
                   "description": posting, "desc_quality": desc_quality}
            if _enrich.needs_text(job, thin=True):
                say("  posting is incomplete; fetching the full text...")
                try:
                    _root = yaml.safe_load(
                        open(args.config, encoding="utf-8")) or {}
                except FileNotFoundError:
                    _root = {}
                _enrich.enrich([job], workers=2, jev_cfg=_root.get("jev"),
                               quiet=True)
                if (job.get("description") or "") != posting:
                    posting = job["description"]
                    desc_quality = job.get("desc_quality") or "full"
                    say(f"  fetched {len(posting)} characters "
                        f"({desc_quality}); saving it")
                    _save_description(getattr(args, "job", ""), posting,
                                      desc_quality, say,
                                      job.get("resolved_url"))
                else:
                    say("  CONTENT INACCESSIBLE: the posting could not be "
                        "read from that link.")
                    if job.get("resolved_url"):
                        say(f"  It resolves to {job['resolved_url'][:88]}")
                    say("  Open it, copy the description, and either paste "
                        "it into the posting in the UI")
                    say("  (Paste it in, on the posting) or save it to a "
                        "file and run:")
                    say(f"    python tailor.py --file posting.txt")
        except ImportError:
            pass
        except Exception as e:
            say(f"  ! fetch skipped: {type(e).__name__}: {e}")

    # A second opinion, asked for deliberately.
    #
    # jev makes the routine decision, which is the point: it is free and it
    # is calibrated on these bands. But a calibrated classifier returning a
    # number is not an argument, and sometimes you want one - on a posting
    # it rejected that you think it misread, or one it passed that looks
    # too good. This runs the hosted analysis on its own, writes it, and
    # drafts nothing. Gates are skipped because you have already decided
    # you want this posting looked at.
    if getattr(args, "review", False):
        return _review_only(posting, facts, desc_quality, company, role,
                            cl, url, posted, say)

    bypass = bool(args.force or args.no_gate or args.gate == "off")
    deterministic = (None, "") if bypass else deterministic_gate(
        posting, facts, desc_quality, posted=posted)
    prescan_score, prescan_reason = None, ""

    # Free local pre-gate, between the deterministic rules and the paid call.
    #
    # jev cannot replace ANALYZE: that returns duties, hard_requirements,
    # keywords and strongest_matches, which are extraction and prose, and the
    # drafting step consumes all of them. It CAN decide whether to pay.
    #
    # It re-scores the FULL posting, where main.py only saw a 700-character
    # focus slice, so it reads requirements the slice cut off. Same questions
    # and bands as the main run, nothing new to calibrate. When no jev model
    # is available it returns None and this falls through untouched.
    if (not bypass and not deterministic[0]
            and not getattr(args, "no_prescan", False)):
        try:
            import jev as _jev
            try:
                _root = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
            except FileNotFoundError:
                _root = {}
            _prof = None
            if getattr(args, "model", None):
                _prof = llm._lookup(_root, args.model, "jev")
            pscore, preview, preason = _jev.prescan(
                posting, _root.get("jev"), _prof, track="career")
            prescan_score, prescan_reason = pscore, preason
            if pscore is not None:
                say(f"  pre-scan (local, free): {pscore}  {preason}")
                floor = getattr(args, "prescan_min", 45)
                # The two errors here are not worth the same. A false
                # positive costs one cheap viability call; a false negative
                # costs a job you never hear about again, and it is
                # invisible because the run stops before the analysis is
                # written. So only a clear reject is a reject. Anything
                # within 15 points of the line is handed to you instead.
                hard = floor - getattr(args, "prescan_margin", 15)
                if pscore < hard:
                    deterministic = ("do_not_apply",
                                     f"local pre-scan {pscore}: {preason}")
                elif pscore < floor:
                    deterministic = ("manual_review",
                                     f"local pre-scan {pscore} is near the "
                                     f"line ({floor}): {preason}")
                elif preview:
                    deterministic = ("manual_review",
                                     f"local pre-scan flagged review: {preason}")
        except ImportError:
            pass
        except Exception as e:
            say(f"  ! pre-scan skipped: {type(e).__name__}: {e}")

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

    # WHO DECIDES, AND WHO WRITES.
    #
    # jev decides whether to apply. It is local, free, and calibrated on
    # this candidate's own bands, and it has already run above: reaching
    # this line means it did not stop the posting. Paying a hosted model to
    # second-guess that is two bills for one judgment, and the hosted model
    # has no calibration against the bands at all — it just produces
    # confident prose either way.
    #
    # So by default there is no viability call. The analysis file is built
    # from what jev already returned, and the only paid calls are the ones
    # that write. --analyze restores the old behaviour when you want the
    # gaps table as prose.
    if getattr(args, "analyze", False):
        say("Analyzing full posting for application viability...")
        analysis = parse_json(ask(
            cl, ANALYZE.replace("{posting}", posting).replace("{facts}", facts)
                 .replace("{today}", TODAY)
                 .replace("{quality}", str(desc_quality or "unknown")), 4000,
            "analyze"))
        normalized = normalize_recommendation(analysis.get("recommendation"))
        if normalized:
            analysis["recommendation"] = normalized
    else:
        analysis = _local_analysis(company, role, prescan_score,
                                   prescan_reason, desc_quality)

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
    # The posting goes to the writing prompts directly. The analysis block
    # is included only when there is something in it worth reading: a
    # locally-built record is mostly empty by design, and pasting empty
    # lists into a prompt invites the model to treat them as findings.
    rich = (analysis.get("decision_source") != "jev_local"
            and (analysis.get("duties") or analysis.get("keywords")
                 or analysis.get("gaps")))
    payload = dict(
        posting=posting,
        analysis=("ANALYSIS:\n" + json.dumps(analysis, indent=2)
                  if rich else ""),
        facts=facts, today=TODAY)
    want_resume = args.resume or args.both
    want_advice = getattr(args, "advice", False)
    want_letter = not args.resume or args.both

    if want_letter:
        say("\nDrafting letter...")
        letter = ask(cl, LETTER.format(voice=voice, **payload), 2000, "letter")
        letter = _voice_pass(letter, voice, cl, args, say)
        f = day / f"{stem}__letter.md"
        f.write_text(letter.strip() + "\n", encoding="utf-8")
        written.append(f)
    if want_resume:
        say("Tailoring resume...")
        name = _candidate_name(args.facts)
        body = ask(cl, TAILORED.format(voice=voice, name=name, **payload),
                   3000, "resume")
        f = day / f"{stem}__resume.md"
        f.write_text(_strip_fences(body).strip() + "\n", encoding="utf-8")
        written.append(f)
    if want_advice:
        say("Writing resume advice...")
        advice = ask(cl, RESUME.format(**payload), 3000, "resume")
        f = day / f"{stem}__advice.md"
        f.write_text(advice.strip() + "\n", encoding="utf-8")
        written.append(f)

    # The .md is the editable source; the .docx is what gets sent. Advice is
    # notes to yourself, so it is never rendered.
    if not getattr(args, "no_docx", False):
        role_line = " ".join(x for x in (role, company) if x)
        for f in list(written):
            if f.stem.endswith("__advice"):
                continue
            try:
                import docs
                made, pdf = docs.render(
                    f, pdf=getattr(args, "pdf", False),
                    facts_path=args.facts, quiet=True,
                    re_line=role_line or None)
                written.append(made)
                if pdf:
                    written.append(pdf)
            except ImportError:
                say("  ! docs.py not found; wrote Markdown only")
                break
            except Exception as e:
                say(f"  ! could not render {f.name}: {type(e).__name__}: {e}")
    return DONE, analysis_path, written, []


def _review_only(posting, facts, desc_quality, company, role, cl, url,
                 posted, say):
    """
    Run the hosted viability analysis and stop. Writes nothing but the
    analysis file, drafts nothing, and costs one call.

    Kept separate from the drafting path so that asking for an opinion can
    never quietly turn into paying for a letter.
    """
    say("Reviewing the posting with the model...")
    analysis = parse_json(ask(
        cl, ANALYZE.replace("{posting}", posting).replace("{facts}", facts)
             .replace("{today}", TODAY)
             .replace("{quality}", str(desc_quality or "unknown")), 4000,
        "analyze"))
    normalized = normalize_recommendation(analysis.get("recommendation"))
    if normalized:
        analysis["recommendation"] = normalized
    analysis["decision_source"] = "model_review"
    analysis["reviewed_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")

    company = company or analysis.get("company", "")
    role = role or analysis.get("role", "")
    if url:
        analysis["posting_url"] = url
    if posted:
        analysis["posted"] = posted

    day = out_dir()
    stem = f"{slug(company)}__{slug(role)}"
    path = day / f"{stem}__analysis.json"
    path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    say(f"\n  Level fit: {analysis.get('seniority_fit', '?')}")
    say(f"  Verdict:   {analysis.get('recommendation', '?')}")
    for g in analysis.get("gaps", []):
        if isinstance(g, dict):
            say(f"  [{g.get('severity', '?'):<11}] {g.get('requirement', '')}")
    if analysis.get("verdict"):
        say("\n" + textwrap.fill(analysis["verdict"], 76,
                                 initial_indent="  ", subsequent_indent="  "))
    say(f"\nWrote {path}")
    say("This is an opinion, not a gate. Nothing was drafted.")
    return _analysis_state(analysis), path, [], []


def _local_analysis(company, role, score, reason, desc_quality):
    """
    The analysis record, written from jev's verdict instead of a paid call.

    Same shape as the model's, because digest.py, ui.py and review_status()
    all read these files and should not care which route produced one.
    `decision_source` says which did, so a run can be audited later.

    The fields the hosted ANALYZE used to extract - duties, keywords,
    hard_requirements - are left empty on purpose. They existed to feed the
    drafting prompt a summary, and the drafting prompts now receive the
    posting itself, which is strictly better than a summary of it.
    """
    return {
        "company": company or "",
        "role": role or "",
        "seniority_fit": "unknown",
        "duties": [],
        "hard_requirements": [],
        "keywords": [],
        "strongest_matches": [],
        "gaps": [],
        "verdict": (f"Local scorer put this at {score}. {reason}"
                    if score is not None else
                    "Passed the local gates; no model was asked to judge it."),
        "recommendation": "apply",
        "jev_score": score,
        "jev_reason": reason,
        "desc_quality": desc_quality,
        "decision_source": "jev_local",
    }


def _save_description(uid, description, quality, say=print, url=None):
    """
    Persist a fetched description so the next run starts from it.

    Without this the fetch is thrown away when the process exits, and every
    retry pays for the same HTTP request. A paste-in run has no uid, so
    there is nothing to write and that is fine.
    """
    if not uid:
        return
    try:
        import store
        conn = store.connect()
        conn.execute(
            "UPDATE jobs SET description=?, desc_quality=? WHERE uid=?",
            (description, quality, uid))
        # Keep the employer's own link rather than the aggregator's tracking
        # hop. It is stable, it is what you want to open, and the next fetch
        # can skip the redirect entirely.
        if url:
            conn.execute("UPDATE jobs SET url=? WHERE uid=?", (url, uid))
        conn.commit()
        conn.close()
    except Exception as e:
        say(f"  ! could not store the fetched text: {type(e).__name__}: {e}")


def _voice_pass(draft, voice, cl, args, say=print):
    """
    Check the draft mechanically, and spend one call fixing what it finds.

    The 34 voice rules reach the model in the last third of a very long
    prompt, and almost all of them are prohibitions, which is the kind of
    instruction a model follows worst. Asking harder does not fix that.

    Checking does. Most of those rules are a regex - an em dash is a
    character, "I am excited to apply" is a string - so voicecheck.py finds
    the breaches perfectly and for free. This then hands the model the
    short, specific list, which is a job models are good at, instead of
    asking it to hold 34 prohibitions in mind while composing.

    One pass only. If a second attempt still breaks a rule the letter needs
    a person, not another call. Warnings are reported and never auto-fixed,
    because they are judgment calls.
    """
    if getattr(args, "no_voice_check", False):
        return draft
    try:
        import voicecheck
    except ImportError:
        return draft

    try:
        root = yaml.safe_load(open(getattr(args, "config", None)
                                   or "config.yaml", encoding="utf-8")) or {}
    except (FileNotFoundError, yaml.YAMLError):
        root = {}

    found, jev_ran = voicecheck.check_all(draft, root.get("jev"))
    errors = [f for f in found if f[0] == "ERROR"]
    warns = [f for f in found if f[0] == "WARN"]
    # A jev finding is a probability, not a fact, so only a confident one
    # earns a paid rewrite. The rest are reported for you to judge.
    thr = float((root.get("jev") or {}).get("voice_revise_at", 0.85))
    jev_strong = [f for f in warns if _jev_p(f[1]) is not None
                  and _jev_p(f[1]) >= thr]
    if jev_ran:
        say(f"  voice check: jev read {_para_count(draft)} paragraph(s)")
    elif not getattr(args, "quiet", False):
        say("  voice check: jev unavailable; mechanical rules only")

    if errors or jev_strong:
        say(f"  voice check: {len(errors)} rule breach(es)"
            + (f" and {len(jev_strong)} paragraph(s) jev flagged"
               if jev_strong else "") + "; revising...")
        problems = "\n".join(
            f"- {rid}: {fix}" + (f"\n  text: {ex}" if ex else "")
            for _, rid, fix, ex in errors + jev_strong)
        try:
            fixed = _strip_fences(ask(cl, REVISE.format(
                problems=problems, draft=draft, voice=voice), 2000, "letter"))
            after, _ = voicecheck.check_all(fixed, root.get("jev"))
            still = [f for f in after if f[0] == "ERROR"]
            still_jev = [f for f in after if f[0] == "WARN"
                         and _jev_p(f[1]) is not None and _jev_p(f[1]) >= thr]
            if len(still) + len(still_jev) < len(errors) + len(jev_strong):
                draft, errors, warns = fixed, still, [
                    f for f in after if f[0] == "WARN"]
                say(f"  voice check: {len(still) + len(still_jev) or 'no'} "
                    f"breach(es) left")
            else:
                say("  voice check: the revision did not improve it; "
                    "keeping the original")
        except Exception as e:
            say(f"  ! voice revision skipped: {type(e).__name__}: {e}")

    for sev, rid, fix, ex in errors:
        say(f"    [{sev}] {rid}: {fix}")
        if ex:
            say(f"           {ex}")
    for sev, rid, fix, ex in warns:
        say(f"    [{sev}] {rid}: {fix}")
        if ex:
            say(f"           {ex}")
    return draft


def _jev_p(rule_id):
    """The probability out of a jev finding's id, e.g. 'vague_claim (p2, 0.88)'."""
    m = re.search(r",\s*([01]?\.\d+)\)$", str(rule_id))
    return float(m.group(1)) if m else None


def _para_count(text):
    try:
        import voicecheck
        return len([p for p in re.split(r"\n\s*\n", voicecheck.body(text))
                    if len(" ".join(p.split())) >= 40])
    except Exception:
        return 0


def _candidate_name(facts_path="resume.yaml"):
    try:
        d = yaml.safe_load(open(facts_path, encoding="utf-8")) or {}
        return d.get("name") or "Candidate"
    except Exception:
        return "Candidate"


def _strip_fences(s):
    """Models sometimes wrap the document in ``` despite being told not to."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s


def cmd_screen(args):
    """
    Run the drafting gates and record the verdict. Draft nothing.

        python tailor.py screen --limit 200 --track career

    Everything here is free: the deterministic rules are string and date
    checks, and the pre-scan is the local jev model. No hosted call is made
    and no letter is written. What comes out is a verdict stored on each
    posting, so the ones worth writing to are a filter in the UI instead of
    something you rediscover by reading the whole list.

      recommended  passed every gate
      review       a gate was unsure, or the posting is too thin to judge
      no           a gate ruled it out, with the reason kept

    Re-running overwrites the previous verdict, so it is safe to run after
    changing a band or a threshold.
    """
    import store
    conn = store.connect()

    effective_min = args.min_score
    if effective_min is None:
        effective_min = _default_batch_min_score(args.config)
    rows = store.recent(conn, limit=100000, min_score=effective_min,
                        order="first_seen" if args.newest else "score",
                        track=None if args.track == "all" else args.track)

    todo = []
    for r in rows:
        if not (r.get("description") or "").strip():
            continue
        if not args.rescreen and r.get("draft_rec"):
            continue
        todo.append(r)
    todo = todo[:args.limit]
    if not todo:
        die(f"nothing to screen at score >= {effective_min}. "
            f"Pass --rescreen to redo postings already screened.")

    try:
        root = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    except (FileNotFoundError, yaml.YAMLError):
        root = {}
    facts, _voice = load_facts(args.facts)
    prof = None
    if getattr(args, "model", None):
        prof = llm._lookup(root, args.model, "jev")

    # Without jev this is close to useless. The deterministic rules catch
    # dates, civil-service wording and a thin posting; everything else -
    # a required licence, a senior title, eight years of experience - is
    # jev's judgment. Screening with it down would mark an RN post
    # "recommended", which is worse than not screening at all.
    if not getattr(args, "no_prescan", False):
        probe = None
        try:
            import jev as _probe_jev
            probe = _probe_jev.prescan(todo[0].get("description") or "",
                                       root.get("jev"), prof,
                                       track=todo[0].get("track") or "career")
        except ImportError:
            pass
        except Exception:
            pass
        if probe is None or probe[0] is None:
            die("jev is not answering, and without it screening would pass "
                "almost everything:\nrequired licences, senior titles and "
                "experience floors are its judgment, not\nthe free rules'. "
                "Start the local model and run this again, or pass\n"
                "--no-prescan to record the deterministic verdicts only.")

    print(f"Screening {len(todo)} posting(s) at score >= {effective_min}. "
          f"No drafting, no paid calls.\n")
    if getattr(args, "no_prescan", False):
        print("  (--no-prescan: deterministic rules only. A posting marked "
              "'recommended'\n  here has only cleared the date, wording and "
              "completeness checks.)\n")
    tally = {"recommended": 0, "review": 0, "no": 0}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []

    for r in todo:
        posting = r.get("description") or ""
        quality = effective_desc_quality(posting, r.get("desc_quality") or "?")
        rec, why = _screen_one(posting, facts, quality, r, root, prof, args)
        tally[rec] += 1
        results.append((r, rec, why))
        mark = {"recommended": "DRAFT", "review": "look", "no": "  no"}[rec]
        print(f"  {mark}  {(r.get('score') or 0):>3}  "
              f"{r['title'][:44]:<46}{r['company'][:20]:<22}{why[:40]}")

    if not args.dry_run:
        conn.executemany(
            "UPDATE jobs SET draft_rec=?, draft_reason=?, draft_screened_at=? "
            "WHERE uid=?",
            [(rec, why, now, r["uid"]) for r, rec, why in results])
        conn.commit()

    print(f"\n{tally['recommended']} to draft, {tally['review']} to look at, "
          f"{tally['no']} ruled out.")
    if args.dry_run:
        print("--dry-run: nothing written.")
    else:
        print("Filter the Postings list to \"Recommended to draft\" to work "
              "through them.")
    return 0


def _screen_one(posting, facts, quality, row, root, prof, args):
    """(verdict, reason) from the same gates the drafting path uses."""
    stop = deterministic_gate(posting, facts, quality,
                              posted=row.get("posted") or "")
    if stop[0] == "do_not_apply":
        return "no", stop[1]
    if stop[0]:
        return "review", stop[1]

    if not getattr(args, "no_prescan", False):
        try:
            import jev as _jev
            pscore, preview, preason = _jev.prescan(
                posting, root.get("jev"), prof,
                track=row.get("track") or "career")
        except ImportError:
            pscore = None
            preason = ""
            preview = False
        except Exception as e:
            pscore, preview = None, False
            preason = f"pre-scan skipped: {type(e).__name__}"
        if pscore is not None:
            floor = getattr(args, "prescan_min", 45)
            hard = floor - getattr(args, "prescan_margin", 15)
            if pscore < hard:
                return "no", f"local pre-scan {pscore}: {preason}"
            if pscore < floor:
                return "review", f"local pre-scan {pscore}, near the line"
            if preview:
                return "review", f"pre-scan flagged review: {preason}"
            return "recommended", f"passed, pre-scan {pscore}"
    return "recommended", "passed the deterministic gates"


def cmd_rewrite(args):
    """
    Rewrite a draft from the candidate's own notes.

    The voice check fixes rule breaches, which is a different thing from
    "this paragraph is boring" or "lead with the dedup story". That
    judgment is the person's, and until now the only way to act on it was
    to edit the file by hand or throw the draft away and pay for a fresh
    one that would not know what was wrong with the last.

    The previous version is kept beside the new one, because a rewrite
    that goes backwards should be recoverable.
    """
    path = Path(args.rewrite)
    if not path.exists():
        die(f"{path} not found")
    if path.suffix != ".md":
        die("give the .md draft, not the .docx. The .docx is rebuilt from it.")
    notes = (args.notes or "").strip()
    if not notes:
        die("say what you want changed, with --notes \"...\"")

    draft = path.read_text(encoding="utf-8")
    facts, voice = load_facts(args.facts)
    cl = client(args)

    print(f"Rewriting {path.name}...")
    out = _strip_fences(ask(cl, REWRITE.format(
        notes=notes, draft=draft, facts=facts, voice=voice), 2500, "letter"))
    if not out.strip():
        die("the model returned nothing; the draft is unchanged")

    if not path.name.endswith("__advice.md"):
        out = _voice_pass(out, voice, cl, args)

    backup = path.with_suffix(".md.prev")
    backup.write_text(draft, encoding="utf-8")
    path.write_text(out.strip() + "\n", encoding="utf-8")
    print(f"  wrote {path}")
    print(f"  previous version kept at {backup.name}")

    if not getattr(args, "no_docx", False):
        try:
            import docs
            made, pdf = docs.render(path, pdf=getattr(args, "pdf", False),
                                    facts_path=args.facts, quiet=True)
            print(f"  wrote {made}")
            if pdf:
                print(f"  wrote {pdf}")
        except ImportError:
            pass
        except Exception as e:
            print(f"  ! could not re-render: {type(e).__name__}: {e}")
    return 0


def cmd_batch(args):
    """
    Precision-oriented bulk tailoring.

    Discovery remains generous in score.py/digest.py, but batch tailoring only
    hands strong-scoring jobs to the tailor model by default. One viability call decides
    APPLY / MANUAL REVIEW / DO NOT APPLY. Only APPLY gets drafting calls.
    """
    import store
    conn = store.connect()

    effective_min = args.min_score
    if effective_min is None:
        effective_min = _default_batch_min_score(args.config)
        print(f"batch precision gate: score >= {effective_min} "
              f"(override with --min-score)")

    # Bridge postings are excluded by default. Drafting a 400-word letter
    # about econometrics for a night-shift warehouse job wastes a model call
    # and produces something nobody wants to read; those applications are a
    # web form. Pass --track all to override.
    rows = store.recent(conn, limit=10000, min_score=effective_min,
                        order="first_seen" if args.newest else "score",
                        track=None if args.track == "all" else args.track)

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
    print("Cost model: up to 1 viability call per full posting; "
          "only APPLY jobs get letter/resume calls. Incomplete postings and "
          "obvious eligibility failures can stop before any model call.")
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
    cl = client(args)
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
    ap.add_argument("--resume", action="store_true",
                    help="tailored resume document, no letter")
    ap.add_argument("--both", action="store_true",
                    help="letter AND tailored resume")
    ap.add_argument("--advice", action="store_true",
                    help="also write resume advice notes (not a document)")
    ap.add_argument("--pdf", action="store_true",
                    help="also write a PDF beside each .docx")
    ap.add_argument("--no-docx", action="store_true",
                    help="write Markdown only, skip the .docx")
    ap.add_argument("--no-fetch", action="store_true",
                    help="do not fetch the full text of a thin posting first")
    ap.add_argument("--rescreen", action="store_true",
                    help="with `screen`: redo postings already screened")
    ap.add_argument("--rewrite", metavar="PATH",
                    help="rewrite an existing draft. Give the .md file; "
                         "say what to change with --notes")
    ap.add_argument("--notes", metavar="TEXT",
                    help="with --rewrite: what you want changed")
    ap.add_argument("--no-voice-check", action="store_true",
                    help="skip the mechanical voice check and its one "
                         "revision call")
    ap.add_argument("--review", action="store_true",
                    help="ask a hosted model for a second opinion on this "
                         "posting and stop. Drafts nothing")
    ap.add_argument("--analyze", action="store_true",
                    help="also pay a hosted model for a viability analysis. "
                         "Off by default: jev makes that decision locally "
                         "and for free")
    ap.add_argument("--prescan-margin", type=int, default=15, metavar="N",
                    help="points below --prescan-min that still get manual "
                         "review rather than rejection (default 15)")
    ap.add_argument("--gate", choices=("both", "skip", "blocking", "off"),
                    default="both",
                    help="draft gate. both (default) = automatic materials ONLY "
                         "for APPLY; manual_review/do_not_apply stop after analysis. "
                         "skip = only do_not_apply stops; blocking = only literal "
                         "blocking gaps stop; off = never abort")
    ap.add_argument("--no-prescan", action="store_true",
                    help="skip the free local jev pre-gate and send every "
                         "posting straight to the paid viability call")
    ap.add_argument("--prescan-min", type=int, default=45,
                    help="local pre-scan score below which a posting is "
                         "rejected without a paid call (default 45)")
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
    ap.add_argument("--track", choices=("career", "bridge", "all"),
                    default="career",
                    help="which track to list or draft for. Defaults to "
                         "career: bridge postings are warehouse and clerical "
                         "work that does not get a tailored cover letter, and "
                         "a bridge score of 85 means something different from "
                         "a career score of 85")
    ap.add_argument("--min-score", type=int,
                    help="minimum cheap-scorer fit. For batch only, default is "
                         "tailoring.batch_min_score (or scoring.strong_at, usually 70)")
    ap.add_argument("--config", default="config.yaml",
                    help="config used to resolve the default batch score gate")
    ap.add_argument("--facts", default="resume.yaml")
    ap.add_argument("--model", metavar="PROFILE",
                    help="use this model profile (from models: in "
                         "config.yaml) for every step of this run")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="with `batch`: skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true",
                    help="with `batch`: list what would be processed, bill nothing")
    args = ap.parse_args()

    if args.rewrite:
        return cmd_rewrite(args)
    if args.job == "list":
        return cmd_list(args)
    if args.job == "screen":
        if "--limit" not in sys.argv:
            args.limit = 500      # free, so a generous default is fine
        return cmd_screen(args)
    if args.job == "batch":
        if "--limit" not in sys.argv:
            args.limit = 25          # a sane default spend; override with --limit
        return cmd_batch(args)

    # ---- get the posting text -------------------------------------------
    company = role = ""
    job = None
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
    cl = client(args)

    # ---- analyze + draft (shared with `batch`) ---------------------------
    try:
        state, analysis_path, written, stop = _run_one(
            posting, company, role, args, facts, voice, cl, job)
    except llm.LLMError as e:
        die(f"model call failed: {e}")
    _report_one(state, analysis_path, written, stop, args)


def _run_one(posting, company, role, args, facts, voice, cl, job):
    return process_job(
        posting, company, role, args, facts, voice, cl,
        url=(job or {}).get("url", "") if args.job else "",
        desc_quality=effective_desc_quality(
            (job or {}).get("description") or "",
            (job or {}).get("desc_quality") or "full") if args.job else "full",
        posted=((job or {}).get("posted") or "") if args.job else "")


def _report_one(state, analysis_path, written, stop, args):
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
