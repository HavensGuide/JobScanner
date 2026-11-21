#!/usr/bin/env python3
"""
score - replace the hard title blocklist with a cheap relevance score.

The old design gated on title substrings, which meant a posting either
survived intact or vanished without trace. This scores instead, so borderline
roles surface as "probably not, and here's why" rather than disappearing.

Three stages, in order of cost:

  1. hard_exclude()  free. Only things no score could rescue: licensed
                     professions, physical trades, seniority a new grad
                     cannot reach. Keep this list SHORT. Everything you put
                     here is invisible forever.
  2. score_jobs()    reads title AND description and returns 0-100 + a
                     one-line reason. Two methods: "rubric" (one generative
                     call per batch of ~20, prose rubric) and "jev" (band and
                     yes/no probabilities per posting). Either runs on any
                     model: see `models:` and `ai:` in config.yaml, and llm.py.
  3. thresholds      a DISPLAY decision, applied in digest.py. Scores are
                     stored, so you can retune without refetching anything.

If the model is unavailable the module falls back to a lexical score and marks
every job "unscored". The digest still sends. A cron job that silently mails
nothing is worse than one that mails noise.

CLI:
    python score.py --calibrate            # histogram of stored scores
    python score.py --backfill --limit 200 # score rows already in seen.db
    python score.py --test "Data Scientist, Developer Tools"

Requires: pip install requests pyyaml
"""

import argparse
import calendar
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import date, datetime, timezone

import yaml

import env
env.load()

DEFAULTS = {
    "enabled": True,
    "model": "claude-haiku-4-5-20251001",
    "batch_size": 20,
    "strong_at": 70,
    "maybe_at": 45,
    "max_per_run": 150,   # hard cost ceiling; excess is recorded unscored
    "desc_chars": 2200,   # focused posting text sent to the model
    # Bridge-track thresholds. Separate because a bridge score answers a
    # different question, so the same number does not mean the same thing.
    # main.py copies these in from config.yaml's bridge.scoring block.
    "bridge_strong_at": 70,
    "bridge_maybe_at": 45,
    # Scoring METHOD per track: "jev", "rubric" or "keyword". `method` sets
    # both; a track-specific key overrides it. Which MODEL each method runs on
    # is the `ai:` block in config.yaml, not this. "anthropic" is accepted as
    # the old name for "rubric", and `provider` as the old name for `method`.
    "method": "jev",
    "career_method": None,            # None -> inherit `method`
    "bridge_method": None,
}

METHODS = ("jev", "rubric", "keyword")
_METHOD_ALIASES = {"anthropic": "rubric", "hosted": "rubric",
                   "claude": "rubric", "local": "jev", "jevk5": "jev",
                   "none": "keyword", "off": "keyword"}


def normalize_method(v):
    v = (str(v or "")).strip().lower()
    v = _METHOD_ALIASES.get(v, v)
    if v and v not in METHODS:
        raise ValueError(f"unknown scoring method {v!r}; use one of "
                         f"{', '.join(METHODS)}")
    return v or None


def scorer_family(s):
    """Stored scorer -> method. Rows written before methods had names say
    'anthropic' (or NULL), which is the rubric."""
    return {"anthropic": "rubric", None: "rubric", "": "rubric"}.get(s, s)

TIERS = ("strong", "review", "maybe", "weak", "unscored")


# ---------------------------------------------------------------------------
# Stage 1: hard exclude
# ---------------------------------------------------------------------------
def hard_exclude(job, terms):
    """
    True if the job should never reach the scorer.

    Substring match on the lowercased title only. Deliberately not checking
    the description: "attorney" appears in plenty of policy analyst postings.
    """
    title = (job.get("title") or "").lower()
    if not title:
        return True
    return any(t in title for t in terms)


# ---------------------------------------------------------------------------
# Stage 1b: deterministic temporal eligibility
# ---------------------------------------------------------------------------
# A stale internship is not a weak fit; it is no longer an opportunity. Keep
# this deterministic and free so stale fixed-period roles never consume a
# model call. The parser is intentionally conservative: it only rejects dates
# that are explicit or occur in clear employment/program-period context.
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_RE = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
             r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
             r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")

_TIMEBOUND_RE = re.compile(
    r"\b(?:intern(?:ship)?s?|fellow(?:ship)?s?|cohorts?|apprentice(?:ship)?|"
    r"externship|residen(?:cy|t)|rotation(?:al)?|trainee(?:ship)?|"
    r"seasonal|temporary|fixed[- ]term|programs?|summer analyst|summer associate)\b",
    re.I,
)
_PERIOD_CONTEXT_RE = re.compile(
    r"\b(?:position|role|employment|job|assignment|term|duration|dates?|"
    r"programs?|intern(?:ship)?s?|fellow(?:ship)?s?|cohorts?|season|runs?|running|"
    r"work period|employment period|program period|start(?:ing)?|through)\b",
    re.I,
)
# A season word in prose is only an employment period when something right
# next to it says so. The old rule accepted any of "program", "class", "term",
# "starts", "begins" within 45 characters, which is a sentence's worth of the
# most common words in a job description — enough to read "The program
# launched in 2018. Summer programming includes workshops." as a stale summer
# internship and delete a permanent role. Two tight tests replace it:
#
#   ADJACENT  the season directly modifies a period noun, optionally through a
#             year: "summer internship", "Summer 2026 cohort", "fall analyst".
#   LEADIN    period-defining language runs straight into the season:
#             "start date: summer", "the program runs through summer".
#
# A season carrying an explicit year is also accepted when time-bound language
# sits nearby, which is what makes "interns for Summer 2025" resolvable.
_SEASON_ADJACENT_RE = re.compile(
    r"^(?:20\d{2}\s+)?(?:spring|summer|fall|autumn)\s+(?:of\s+)?(?:20\d{2}\s+)?"
    r"(?:intern(?:ship)?s?|fellow(?:ship)?s?|cohorts?|analysts?|associates?|"
    r"programs?|sessions?|semesters?|terms?|classes|class|starts?|start date|"
    r"hires?|hiring|recruit(?:ing|ment)?|externship|apprentice(?:ship)?|"
    r"residen(?:cy|ts?)|rotation(?:al)?|traineeships?|trainees?)\b",
    re.I,
)
_SEASON_LEADIN_RE = re.compile(
    r"\b(?:start(?:s|ing)?(?:[ \t]+date)?|begins?|beginning|runs?|running|"
    r"through|employment period|work period|program period|cohort|"
    r"intern(?:ship)?s?|fellow(?:ship)?s?)\b[^.!?]{0,20}$",
    re.I,
)
_OPEN_ENDED_RE = re.compile(
    r"\b(?:open[ -]year[ -]round|year[ -]round applications?|"
    r"applications? (?:are )?accepted year[ -]round|rolling applications?|"
    r"applications? (?:are )?accepted on a rolling basis|"
    r"multiple start dates?|start dates? (?:are )?(?:organized )?twice a year)\b",
    re.I,
)

_MONTH_RANGE_RE = re.compile(
    rf"\b(?P<m1>{_MONTH_RE})\s*"
    rf"(?P<d1>\d{{1,2}}(?:st|nd|rd|th)?)?\s*,?\s*"
    rf"(?P<y1>20\d{{2}})?\s*"
    rf"(?:-|–|—|to|through|thru)\s*"
    rf"(?P<m2>{_MONTH_RE})\s*"
    rf"(?P<d2>\d{{1,2}}(?:st|nd|rd|th)?)?\s*,?\s*"
    rf"(?P<y2>20\d{{2}})?\b",
    re.I,
)
_SEASON_RE = re.compile(
    r"\b(?:(?P<y1>20\d{2})\s+)?(?P<season>spring|summer|fall|autumn)"
    r"(?:\s+(?:of\s+)?(?P<y2>20\d{2}))?\b",
    re.I,
)

_DEADLINE_MARKER_RE = re.compile(
    r"\b(?:application\s+deadline|applications?\s+(?:close|closes|due)|"
    r"apply\s+by|deadline(?:[ \t]+date)?)\b\s*(?:is|on|:|-)?\s*",
    re.I,
)
_START_MARKER_RE = re.compile(
    r"\b(?:anticipated[ \t]+start[ \t]+date|expected[ \t]+start[ \t]+date|"
    r"start[ \t]+date|program[ \t]+(?:starts?|begins)|"
    r"internship[ \t]+(?:starts?|begins)|fellowship[ \t]+(?:starts?|begins)|"
    r"cohort[ \t]+(?:starts?|begins)|employment[ \t]+(?:starts?|begins)|"
    r"position[ \t]+(?:starts?|begins)|role[ \t]+(?:starts?|begins)|"
    r"job[ \t]+(?:starts?|begins)|term[ \t]+(?:starts?|begins))\b"
    r"[ \t]*(?:is|on|:|-)?[ \t]*",
    re.I,
)


def _month_num(raw):
    return _MONTHS.get((raw or "").lower().rstrip("."))


def _clean_day(raw, default=1):
    if not raw:
        return default
    m = re.match(r"\d{1,2}", str(raw))
    return int(m.group(0)) if m else default


def _posted_date(job):
    raw = str(job.get("posted") or "").strip()[:10]
    try:
        return date.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def _infer_year(job, today, end_month):
    """
    Infer the cycle year for a yearless period.

    Prefer the posting date because "Summer Internship" posted in Oct 2026
    normally means Summer 2027, while the same title posted in Jan 2026 means
    Summer 2026. Without a posting date, treat a period whose end month has not
    passed yet as this year's cycle; otherwise assume the next cycle rather
    than falsely deleting a likely upcoming posting.
    """
    ref = _posted_date(job) or today
    return ref.year if ref.month <= end_month else ref.year + 1


def _year_for_period(job, today, end_month, text, start, end, is_title=False):
    """Use a nearby recruiting-cycle year before falling back to inference."""
    lo, hi = max(0, start - 70), min(len(text), end + 70)
    nearby = []
    for m in re.finditer(r"20\d{2}", text[lo:hi]):
        pos = lo + m.start()
        dist = min(abs(pos - start), abs(pos - end))
        nearby.append((dist, int(m.group(0))))
    # A year printed near a season is only a recruiting-cycle year if it could
    # plausibly BE one. Job descriptions are full of founding dates, copyright
    # lines, and "since 2015" boilerplate; without this guard the nearest of
    # those dates the period and the posting is deleted as decades stale.
    ref = _posted_date(job) or today
    if nearby:
        nearby.sort()
        plausible = [y for _, y in nearby if y >= ref.year - 1]
        if plausible:
            return plausible[0]

    title_years = sorted({int(y) for y in re.findall(r"20\d{2}", str(job.get("title") or ""))})
    if is_title and len(title_years) == 1:
        return title_years[0]
    # If a title advertises multiple cycles (e.g. "2026/2027"), a yearless
    # period in the description is not grounds to delete the whole posting as
    # long as a future cycle is explicitly named. Prefer that future cycle.
    future = [y for y in title_years if y > today.year]
    if future:
        return min(future)
    return _infer_year(job, today, end_month)


def _safe_date(year, month, day=1, deadline=False):
    if deadline and day == 0:
        day = calendar.monthrange(year, month)[1]
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_date_phrase(fragment, job, today, deadline=False):
    """Parse a date immediately following a strong marker such as Start Date."""
    f = (fragment or "").lstrip()

    # ISO: 2026-09-15
    m = re.match(r"(?P<y>20\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b", f)
    if m:
        return _safe_date(int(m.group("y")), int(m.group("m")), int(m.group("d")))

    # US numeric: 9/15/2026 or 09-15-2026
    m = re.match(r"(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>20\d{2})\b", f)
    if m:
        return _safe_date(int(m.group("y")), int(m.group("m")), int(m.group("d")))

    # Month name with optional day/year: September 15, 2026 / September 2026.
    m = re.match(
        rf"(?P<m>{_MONTH_RE})\.?\s*"
        rf"(?P<d>\d{{1,2}}(?:st|nd|rd|th)?)?\s*,?\s*"
        rf"(?P<y>20\d{{2}})?\b",
        f, re.I,
    )
    if not m:
        return None
    month = _month_num(m.group("m"))
    if not month:
        return None
    year = int(m.group("y")) if m.group("y") else _infer_year(job, today, month)
    if m.group("d"):
        day = _clean_day(m.group("d"))
    elif deadline:
        day = calendar.monthrange(year, month)[1]
    else:
        day = 1
    return _safe_date(year, month, day)


def _period_context(text, start, end, title_timebound=False):
    if title_timebound:
        return True
    lo, hi = max(0, start - 100), min(len(text), end + 100)
    return bool(_PERIOD_CONTEXT_RE.search(text[lo:hi]))


def _temporal_scan(job, today=None):
    """
    Return (exclusion_reason, notes).

    `exclusion_reason` is non-empty only when the posting is definitely stale
    enough to drop before scoring. `notes` captures parsed future/current dates
    so the model receives temporal context as a redundant safety check.
    """
    today = today or date.today()
    title = str(job.get("title") or "")
    desc = str(job.get("description") or "")
    title_timebound = bool(_TIMEBOUND_RE.search(title))
    notes = []
    start_candidates = []

    # 1) Explicit application deadlines are universal hard gates.
    combined = f"{title}\n{desc}"
    open_ended = bool(_OPEN_ENDED_RE.search(combined))
    if open_ended:
        notes.append("posting explicitly allows rolling/year-round or multiple start dates")
    for m in _DEADLINE_MARKER_RE.finditer(combined):
        d = _parse_date_phrase(combined[m.end():m.end() + 60], job, today, deadline=True)
        if not d:
            continue
        notes.append(f"application deadline {d.isoformat()}")
        if d < today:
            return f"application deadline {d.isoformat()} has passed", notes

    # 2) Explicit start markers are strong enough without extra context.
    for m in _START_MARKER_RE.finditer(combined):
        d = _parse_date_phrase(combined[m.end():m.end() + 60], job, today)
        if not d:
            continue
        notes.append(f"stated start date {d.isoformat()}")
        start_candidates.append((d, f"stated start date {d.isoformat()} has passed"))

    # 3) Employment/program month ranges. Title ranges are high-signal. In the
    # description require nearby period language or a time-bound title.
    for text, is_title in ((title, True), (desc, False)):
        for m in _MONTH_RANGE_RE.finditer(text):
            if not is_title and not _period_context(text, m.start(), m.end(), title_timebound):
                continue
            m1, m2 = _month_num(m.group("m1")), _month_num(m.group("m2"))
            if not (m1 and m2):
                continue
            y1 = int(m.group("y1")) if m.group("y1") else None
            y2 = int(m.group("y2")) if m.group("y2") else None
            if y1:
                start_year = y1
            elif y2:
                start_year = y2 if m1 <= m2 else y2 - 1
            else:
                start_year = _year_for_period(
                    job, today, m2, text, m.start(), m.end(), is_title=is_title
                )
            start = _safe_date(start_year, m1, _clean_day(m.group("d1"), 1))
            if not start:
                continue
            label = re.sub(r"\s+", " ", m.group(0)).strip()
            notes.append(f"employment/program period '{label}' starts {start.isoformat()}")
            start_candidates.append((
                start, f"employment/program period starts {start.isoformat()} ({label})"
            ))

    # 4) Seasonal periods. Spring/Summer/Fall have conventional enough starts
    # for eligibility triage. Winter is deliberately omitted because the year
    # convention is ambiguous (Jan-Feb vs. Dec).
    seasons = {
        "spring": (3, 5),
        "summer": (6, 8),
        "fall": (9, 11),
        "autumn": (9, 11),
    }
    for text, is_title in ((title, True), (desc, False)):
        for m in _SEASON_RE.finditer(text):
            raw_year = m.group("y1") or m.group("y2")
            if not is_title:
                # "fall" is also an ordinary verb ("pay will fall within") and
                # "summer" turns up in event names, benefits copy, and company
                # history. Require the season to actually modify a period noun,
                # or to be introduced by period-defining language, before
                # treating a prose mention as the employment period.
                after = text[m.start():min(len(text), m.end() + 40)]
                before = text[max(0, m.start() - 60):m.start()]
                window = text[max(0, m.start() - 60):min(len(text), m.end() + 60)]
                if not (_SEASON_ADJACENT_RE.match(after)
                        or _SEASON_LEADIN_RE.search(before)
                        or (raw_year and _TIMEBOUND_RE.search(window))):
                    continue
            season = m.group("season").lower()
            start_month, end_month = seasons[season]
            year = (int(raw_year) if raw_year else _year_for_period(
                job, today, end_month, text, m.start(), m.end(), is_title=is_title
            ))
            start = date(year, start_month, 1)
            label = re.sub(r"\s+", " ", m.group(0)).strip()
            notes.append(f"{season} period '{label}' starts {start.isoformat()}")
            start_candidates.append((
                start, f"{season} period starts {start.isoformat()} ({label})"
            ))

    # A posting can advertise several cohorts. Keep it if ANY stated start is
    # today/future, or if it explicitly says applications are rolling/year-round.
    # Only reject when every parsed employment/program start is already past.
    if start_candidates and not open_ended:
        if any(d >= today for d, _ in start_candidates):
            return "", notes
        latest = max(start_candidates, key=lambda x: x[0])
        return latest[1], notes

    return "", notes


def temporal_exclusion_reason(job, today=None):
    """Human-readable reason for deterministic stale-job exclusion, or ''."""
    return _temporal_scan(job, today)[0]


def temporal_exclude(job, today=None):
    """True when a stated job/program start or application deadline is past."""
    return bool(temporal_exclusion_reason(job, today))


def temporal_summary(job, today=None):
    """Compact parsed temporal evidence for the model prompt."""
    reason, notes = _temporal_scan(job, today)
    if reason:
        return f"STALE: {reason}"
    return "; ".join(dict.fromkeys(notes)) if notes else "no explicit period parsed"


# ---------------------------------------------------------------------------
# Stage 2a: lexical fallback (no API needed)
# ---------------------------------------------------------------------------
def lexical_score(job, signals):
    """
    Crude keyword overlap, 0-100. Used only when the API is unavailable.
    Intentionally generous: when we cannot think properly, we show more, not
    less. Anything at or above the maybe threshold reaches your inbox.
    """
    text = f"{job.get('title', '')} {job.get('description', '')}".lower()
    title = (job.get("title") or "").lower()

    hits = sum(1 for s in signals if s in title) * 3
    hits += sum(1 for s in signals if s in text)
    return min(100, 40 + hits * 6)


# ---------------------------------------------------------------------------
# Stage 2b: the model
# ---------------------------------------------------------------------------
_TODAY = date.today().strftime("%B %d, %Y")

RUBRIC = """You are triaging job postings for one specific candidate. Score
each posting 0-100 on how good a fit it is FOR THIS CANDIDATE. You are not
judging whether the job is good, prestigious, or well paid. Only fit.

IMPORTANT: this candidate is broader than a quantitative analyst. Do not make
"analyticalness" a hidden requirement. A communications, policy, program,
research-coordination, client-support, data-operations, or stakeholder-facing
role can be a strong fit when its actual duties match the candidate record.
Score the overlap that exists; do not penalize a job merely for belonging to a
different function than economics/data analysis.

ANCHORS — use the full range. A run where everything lands 50-70 is useless.
  85-100  Squarely in scope. Candidate meets the stated requirements.
  70-84   Strong match. Most core duties are supported; gaps are minor/learnable.
  45-69   Plausible. Real gaps, OR not enough posting text to judge confidently.
  20-44   Stretch. Wrong level, or the central duties are only weakly adjacent.
   0-19   Wrong function/level, or a visible structural eligibility barrier.

WEIGHTING, in order:
  1. STRUCTURAL ELIGIBILITY + LEVEL. This candidate graduated in May 2026,
     which is in the PAST as of {today}. Treat them as a graduate holding the
     degree, not a student. A hard 5+ year floor, required graduate degree,
     required license, or people-management role can dominate the score.
  2. ACTUAL DUTIES. Compare the posting duty-by-duty against the entire factual
     record. Several direct duty matches can justify 70+ even when the title or
     function is not primarily analytical.
  3. TRANSFERABLE CAPABILITY CLUSTERS. Give full credit when the job draws on
     any substantial cluster documented in the record, including:
       - data/quant: Python, statistics, econometrics, Excel, data cleaning, QA,
         visualization, automation, large datasets;
       - policy/research: economics, public policy, labor policy, legal/regulatory
         research, literature review, source verification, policy writing;
       - communications/stakeholders: presentations, translating technical or
         policy material, earned-media/partnership work, liaison work, public
         speaking, cross-functional and external-stakeholder coordination;
       - program/project/operations: project delivery, evaluation design,
         documentation, records, process improvement, sensitive-data handling.
  4. DOMAIN + NAMED TOOLS. Domain familiarity helps but is not mandatory when
     duties transfer. Missing one named tool is normally a gap, not a verdict.

DO NOT down-score merely because:
  - the job is communications, policy, program, operations, client support,
    research coordination, or another non-quantitative function;
  - the title contains engineer/developer/scientist; read the actual duties;
  - one named tool is missing. SQL, R, Stata, Tableau, Databricks, Salesforce,
    etc. are gaps only to the degree the posting makes them central;
  - the candidate's closest evidence comes from an adjacent domain.

DO down-score hard for:
  - security clearance required and not realistically obtainable at entry level;
  - a professional license or graduate degree explicitly stated as required;
  - management of people as a core duty;
  - a hard experience minimum missed by years.

TEMPORAL ELIGIBILITY IS A HARD GATE. Today is {today}. If a posting states
an internship, fellowship, cohort, program, fixed-term assignment, or job
period whose START DATE has already passed, score it 0 even if the employer
left the posting online. If an explicit application deadline has passed, also
score it 0. A deterministic pre-filter should catch these first; this rule is
redundant protection. Do not confuse the posting publication date with the
employment/program start date.

Do NOT invent requirements from a title. For example, a title such as "Policy
Counsel" may ultimately require a JD/bar admission, but if an incomplete
posting does not actually state that, flag it for manual review rather than
assuming the requirement.

MISSING TOOLS. Distinguish a tool gap from a function gap. An early-career
role can still be a strong overall fit when most duties match and one required
tool is learnable. Conversely, if advanced use of that tool is the central daily
duty, keep the score below the strongest range and say why.

INCOMPLETE POSTINGS. Some entries are marked [EXCERPT ONLY] or [TITLE ONLY].
You are judging a fragment, not the posting. Absence of evidence in a truncated
excerpt is NOT evidence of a bad fit. If the visible title/text supports a
plausible or strong match but the excerpt does not contain enough duties, level,
or qualifications to make a confident decision, set "review": true. For such
entries:
  - do not score below 45 solely because information is missing;
  - score the evidence that IS visible, without inventing missing requirements;
  - begin the reason with "manual review:" and state what makes it promising
    and what must be checked in the full posting.
A visible disqualifier still scores low normally and should use review=false.

FULL POSTINGS can also use review=true when the text itself is genuinely
ambiguous or internally incomplete, but do not use manual review as a hedge for
every borderline score.

Return ONLY a JSON array, no fences, no preamble. One object per posting:
  {"i": <index>, "score": <0-100>, "review": <true|false>,
   "reason": "<max 20 words>"}

The reason must be concrete and specific to THIS posting. Name the actual
match or actual gap. Never say the candidate "lacks communications, labor,
policy, stakeholder, or project experience" if the candidate record contains
adjacent evidence in those areas. When you score something low, the reason is
the only thing the candidate will see, so make it the sentence that lets them
decide you were wrong."""


BRIDGE_RUBRIC = """You are triaging job postings for one specific person. These
postings are on the BRIDGE track: work that pays the bills while a separate
search for research, policy, and quantitative roles runs in parallel. The
candidate record is attached, but you are NOT scoring against it the way you
would for a career role.

THE QUESTION HERE IS DIFFERENT. Score 0-100 on two things only:

  1. CAN HE GET IT? Entry-level, hires off the street, no license or
     certification that takes months, no years of specific experience, no
     clearance, no CDL, no car assumed unless the location is transit-reachable.
  2. CAN HE STAND IT? The binding constraint is PUBLIC CONTACT, and it is not
     a yes/no question. What matters is how long one interaction lasts and
     whether he absorbs the other person's dissatisfaction. Sort every posting
     into one of three kinds:

       NONE. Warehouse, overnight stocking, records, lab support, mailroom,
       back-office processing. He may go a shift without speaking to anyone
       who does not work there. This is what he is looking for.

       TRANSACTIONAL. There is a queue and a task. The interaction is short,
       repeatable, and ends. A barista pulling shots, a mailroom window, a
       library circulation desk, a pharmacy fill counter. He has said this is
       acceptable. Do NOT score it as though it were the next kind.

       RELATIONAL. He owns someone's experience for an extended stretch and is
       the person they complain to. A waiter with a section, a retail floor
       associate on commission, a call-center queue, anything tipped, anything
       with a satisfaction score attached to his name. This is the actual no.

     The line runs between a barista and a waiter, not between indoors and
     outdoors. Read the duties and decide which kind the job really is; a
     title of "Cafe Associate" can be either.

A four-year degree and a research background are NEITHER a qualification NOR a
disqualification here. Never score a posting down as "overqualified" and never
score it up because it mentions data or analysis. Do not write reasons about
career fit, growth, or whether the job is beneath him. He knows.

ANCHORS — use the full range.
  85-100  No public contact, hires immediately, commutable, pay stated and
          liveable. Warehouse, overnight stocking, data entry, records, lab
          support, mailroom, back-office processing.
  70-84   Incidental contact only (a dock, a service window, covering a desk
          at lunch), or transactional counter work that is otherwise clean:
          steady hours, stated pay, no tips, no quota. A well-paid barista job
          at a stable employer belongs here, not below.
  45-69   Real friction: transactional contact plus something else bad, a
          physical demand worth naming, an erratic or on-call schedule,
          unstated pay, a short certification, or a location needing a car.
  20-44   Relational service work. He has not ruled this out absolutely, but
          he would be unhappy: tipped floor service, a section of tables, a
          commissioned sales floor, a named satisfaction metric.
   0-19   Cold sales, door-to-door, a call-center queue, or a hard gate (CDL,
          active clearance, licensure, 3+ years required).

READ THE POSTING, NOT THE TITLE. Titles lie in both directions on this track.
"Operations Associate" is frequently a call center. "Service Coordinator" may
be pure data entry. "Inventory Specialist" at a boutique is a sales floor job
with a stockroom attached. "Cafe Associate" is transactional at a kiosk and
relational at a sit-down place with tables. When the duties reveal a worse
kind of contact than the title implied, score it down and say where you saw
it. When they reveal a better kind, score it up the same way.

WHAT TO PUT IN THE REASON. The reason is the only thing he reads before
deciding, and for this track the useful facts are:
  - the pay, verbatim, if the posting states one ("$19.50/hr", "$44k-$52k");
  - the shift, if it is not standard business hours ("11p-7a", "weekends");
  - which of the three contact kinds it is, in plain words, and what in the
    posting told you ("counter queue, no tips" / "own section, tipped");
  - any hard barrier.
Lead with pay and shift when both are visible. Skip anything you had to guess.

PHYSICAL WORK IS FINE, say so plainly rather than treating it as a defect. A
50lb lifting requirement, standing for eight hours, or a cold warehouse is
information, not a penalty. Score it as a 45-69 only when the posting describes
something genuinely punishing, and name what it is.

TEMPORAL ELIGIBILITY IS STILL A HARD GATE. Today is {today}. A stated start
date or application deadline in the past scores 0. Seasonal work whose season
has passed scores 0.

INCOMPLETE POSTINGS marked [EXCERPT ONLY] or [TITLE ONLY]: do not invent duties
or pay. If the visible text suggests a workable fit but you cannot see the
public-contact level or the pay, set "review": true, do not score below 45 for
missing information alone, and begin the reason with "manual review:".

Return ONLY a JSON array, no fences, no preamble. One object per posting:
  {"i": <index>, "score": <0-100>, "review": <true|false>,
   "reason": "<max 20 words>"}"""


# ---------------------------------------------------------------------------
# Stage 1c: which rubric reads this posting
# ---------------------------------------------------------------------------
TRACKS = ("career", "bridge")


def professional_titles(cfg):
    """
    Titles never dropped by bridge.exclude, however they are worded.

    Without this, "Server Engineer" dies on the word "server" and "Customer
    Success Analyst" dies on "customer service". Lives in config because it
    is a job-search specification, not a parsing rule.
    """
    v = cfg.get("professional_titles")
    return tuple(t.lower() for t in v) if v else _PROFESSIONAL_TITLE


def bridge_never(cfg):
    """Title words that are never bridge work (bridge.never)."""
    return tuple(s.lower() for s in
                 ((cfg.get("bridge") or {}).get("never") or []))


def track_terms(cfg):
    """
    (career, bridge, exclude, tolerated) term lists from a loaded config.yaml.

    Shared by main.py and the score.py CLI so a title is classified the same
    way during a run and during a backfill. All lowercased here; callers must
    not lowercase again or re-derive their own lists.
    """
    career = [s.lower() for s in
              ((cfg.get("ats", {}).get("sweep", {}) or {}).get("title_must_match")
               or cfg.get("relevance_signals", []))]
    bridge_cfg = cfg.get("bridge") or {}
    if not bridge_cfg.get("enabled", False):
        return career, [], [], []
    return (career,
            [s.lower() for s in bridge_cfg.get("title_must_match", [])],
            [s.lower() for s in bridge_cfg.get("exclude", [])],
            [s.lower() for s in bridge_cfg.get("tolerated", [])])


# A title carrying one of these is a professional role whatever else it says,
# so it is never dropped by bridge.exclude. Without this, "Server Engineer"
# dies on "server" and "Customer Success Analyst" dies on "customer service".
# The scorer can read them and say no; a silent drop cannot be argued with.
# Default only. config.yaml's `professional_titles:` overrides it, and
# track_terms() reads that. Kept here so an older config still works.
_PROFESSIONAL_TITLE = (
    "engineer", "developer", "scientist", "analyst", "architect", "researcher",
    "economist", "statistician", "counsel", "strategist", "consultant",
    "fellow", "associate director", "program manager", "product manager",
)


def classify_track(job, career_terms=(), bridge_terms=(), bridge_exclude=(),
                   bridge_tolerated=(), professional=None, never=()):
    """
    "career", "bridge", or None to drop the posting entirely.

    Title-only and free, so it runs before anything is fetched or scored.

    Career wins ties: "Inventory Analyst" matches both lists and is still read
    as an analyst job, which keeps every title that worked before working the
    same way. A posting matching neither list also goes to career, because
    USAJOBS, NYC Open Data, and Adzuna have no title filter upstream and that
    is what they did before this track existed.

    `bridge_tolerated` beats `bridge_exclude`, and is also a bridge include
    term in its own right. That is what makes "Barista" reachable: the word
    sits inside the coffee-shop exclusions, and tolerating it has to override
    them without deleting them, or "Barista/Server" comes back too.

    None is the one silent drop in the pipeline, and it is deliberate: "Retail
    Stock Associate", "Call Center Data Entry Specialist". No description
    rescues a sales floor or a phone queue, so these never reach the model.
    main.py prints what it dropped.
    """
    title = (job.get("title") or "").lower()
    if not title:
        return "career"
    if any(t in title for t in career_terms):
        return "career"
    if any(t in title for t in bridge_tolerated):
        return "bridge"

    ruled_out = (any(t in title for t in bridge_exclude)
                 and not any(t in title
                             for t in (professional or _PROFESSIONAL_TITLE)))
    if any(t in title for t in bridge_terms):
        # A professional title that happens to contain a bridge word is a
        # career question: "Software Engineer, Backend, Accounts
        # Receivable" reached the bridge digest at 87 because "engineer"
        # only ever protected a title from bridge.exclude, it never moved
        # it off the track.
        if any(t in title for t in (professional or _PROFESSIONAL_TITLE)):
            return "career"
        if any(t in title for t in never):
            return None
        return None if ruled_out else "bridge"
    # Matches neither list. Before the bridge track this went straight to the
    # career rubric and scored ~10; now that the contact constraint is written
    # down, a bare "Cashier" or "Line Cook" from Adzuna is not worth a model
    # call either.
    return None if ruled_out else "career"


def _profile(facts_path):
    """Condense resume.yaml into the scoring context. Drops voice rules."""
    if not os.path.exists(facts_path):
        raise FileNotFoundError(f"{facts_path} not found")
    data = yaml.safe_load(open(facts_path, encoding="utf-8"))
    data.pop("voice", None)
    for key in ("phone", "email"):
        data.pop(key, None)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


QUALITY_TAG = {"excerpt": " [EXCERPT ONLY]", "none": " [TITLE ONLY]"}


def _focus_description(desc, limit):
    """
    Keep the parts of a long posting that actually determine fit.

    Taking the first N characters is badly biased toward employer boilerplate.
    Job descriptions commonly put responsibilities and qualifications after a
    long company introduction, which means the old 700-character slice often
    never reached the job itself. Prefer sections around common headings; fall
    back to a head/middle/tail sample when headings are absent.
    """
    d = (desc or "").strip()
    if not d or len(d) <= limit:
        return re.sub(r"\s+", " ", d).strip()

    low = d.lower().replace("’", "'")
    resp_markers = (
        "what you'll do", "what you will do", "responsibilities",
        "key responsibilities", "major duties", "the role", "your role",
        "what you'll be doing", "what you will be doing",
    )
    qual_markers = (
        "who you are", "qualifications", "requirements", "minimum qualifications",
        "preferred qualifications", "what we're looking for", "what we are looking for",
        "what you'll bring", "what you will bring", "about you",
    )

    def first_pos(markers):
        hits = [low.find(m) for m in markers if low.find(m) >= 0]
        return min(hits) if hits else None

    rpos, qpos = first_pos(resp_markers), first_pos(qual_markers)
    chunks = []

    # A little opening context, then spend most of the budget on the job.
    intro_n = min(300, max(120, limit // 8))
    chunks.append(d[:intro_n])

    remaining = limit - intro_n
    if rpos is not None or qpos is not None:
        if rpos is not None and qpos is not None and qpos > rpos:
            duty_n = int(remaining * 0.58)
            qual_n = remaining - duty_n
            chunks.append(d[rpos:rpos + duty_n])
            chunks.append(d[qpos:qpos + qual_n])
        elif rpos is not None:
            chunks.append(d[rpos:rpos + remaining])
        else:
            chunks.append(d[qpos:qpos + remaining])
    else:
        # No recognizable headings: sample across the document instead of
        # treating the first company-intro paragraph as the whole posting.
        head_n = int(remaining * 0.45)
        mid_n = int(remaining * 0.30)
        tail_n = remaining - head_n - mid_n
        chunks.append(d[intro_n:intro_n + head_n])
        mid_start = max(0, len(d) // 2 - mid_n // 2)
        chunks.append(d[mid_start:mid_start + mid_n])
        chunks.append(d[-tail_n:])

    focused = " ... ".join(c.strip() for c in chunks if c.strip())
    return re.sub(r"\s+", " ", focused).strip()[:limit]


def _batch_prompt(jobs, desc_chars):
    parts = []
    for i, j in enumerate(jobs):
        raw_desc = j.get("description") or ""
        desc = _focus_description(raw_desc, desc_chars)
        # Sources differ in how much text they hand over. Adzuna truncates its
        # descriptions, so the model must be told it is reading a fragment.
        tag = QUALITY_TAG.get(j.get("desc_quality") or _infer_quality(raw_desc), "")
        parts.append(
            f"[{i}] {j.get('title', '')}{tag}\n"
            f"    employer: {j.get('company', '')}\n"
            f"    location: {j.get('location', '')}\n"
            f"    posted:   {j.get('posted') or '(unknown)'}\n"
            f"    temporal: {temporal_summary(j)}\n"
            f"    posting:  {desc or '(no description available)'}"
        )
    return "Score these postings.\n\n" + "\n\n".join(parts)


def _infer_quality(desc):
    """
    Fallback for rows stored before desc_quality was persisted.

    Must stay in sync with sources._quality. The ellipsis test is the load
    bearing part: Adzuna excerpts run ~500 chars, so a length-only rule calls
    them "full", the [EXCERPT ONLY] tag never gets attached, and the rubric's
    floor for truncated postings never fires. That silently buries every
    Adzuna job at the bottom of the rankings.
    """
    d = (desc or "").strip()
    if len(d) < 40:
        return "none"
    if len(d) < 400 or d.endswith(("\u2026", "...")):
        return "excerpt"
    return "full"


def _parse(text, n):
    """Tolerant JSON array parse. Returns {index: (score, reason, review)}."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        rows = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", cleaned, re.S)
        if not m:
            return {}
        try:
            rows = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    out = {}
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        try:
            i = int(r.get("i"))
            s = max(0, min(100, int(r.get("score"))))
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            review = bool(r.get("review", False))
            out[i] = (s, str(r.get("reason", ""))[:200], review)
    return out


def thresholds(cfg, track="career"):
    """(strong_at, maybe_at) for a track. Bridge falls back to career."""
    if track == "bridge":
        return (cfg.get("bridge_strong_at", cfg["strong_at"]),
                cfg.get("bridge_maybe_at", cfg["maybe_at"]))
    return cfg["strong_at"], cfg["maybe_at"]


def tier_of(score, cfg, review=False, track="career"):
    if score is None:
        return "unscored"
    # Manual review is an attention state, not a low-fit state. It gets its
    # own digest section so promising incomplete postings never disappear in
    # the middle of ordinary maybes.
    if review:
        return "review"
    strong_at, maybe_at = thresholds(cfg, track)
    if score >= strong_at:
        return "strong"
    if score >= maybe_at:
        return "maybe"
    return "weak"


def _profile_for(cfg, method, track):
    """The llm profile a method uses on a track, or None for keyword/rule."""
    if method not in ("rubric", "jev"):
        return None
    import llm
    key = f"_{method}_profile_{track}"
    if key not in cfg:
        # --compare and tests can pin a profile; otherwise ask config.yaml.
        pinned = cfg.get(f"{method}_profile")
        cfg[key] = (llm.finalize(pinned, pinned.get("name", "pinned"))
                    if isinstance(pinned, dict) else
                    llm.resolve(cfg.get("_root") or {}, method, track))
    return cfg[key]


def score_jobs(jobs, cfg=None, facts_path="resume.yaml", signals=()):
    """
    Annotate each job in place with score / tier / reason / scored_at.
    Always returns the full list. Never drops anything.

    Each track has a METHOD (jev, rubric, keyword) and each method runs on
    whatever model config.yaml's `ai:` block assigns it. Failure falls
    through jev -> rubric -> keyword, so a dead local server or an empty
    credit balance costs accuracy for a run, never the run.
    """
    import llm
    cfg = {**DEFAULTS, **(cfg or {})}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def mark(j, sc, reason, tier=None, review=False, scorer="rubric",
             model=None):
        track = j.get("track") or "career"
        # If the model explicitly says "read this manually", never bury it in
        # weak just because uncertainty pulled the numeric score down.
        if review and sc is not None:
            sc = max(sc, thresholds(cfg, track)[1])
        j["score"] = sc
        j["reason"] = reason
        j["tier"] = tier or tier_of(sc, cfg, review=review, track=track)
        j["scored_at"] = now
        # Which method AND which model produced this number. A jev 72 and a
        # rubric 72 are different scales, and so are a Haiku 72 and a local
        # Qwen 72; --calibrate keeps them apart instead of pooling them.
        j["scorer"] = scorer
        j["scorer_model"] = model

    def keyword(js, why):
        for j in js:
            mark(j, lexical_score(j, signals), why, "unscored",
                 scorer="keyword")

    if not jobs:
        return jobs

    # Deterministic temporal gate. main.py already drops these before dedupe/
    # storage, but score_jobs() is also called directly by --test/backfill and
    # may be reused elsewhere. Never spend a model call on a known-stale role.
    eligible = []
    stale = 0
    for j in jobs:
        why = temporal_exclusion_reason(j)
        if why:
            stale += 1
            mark(j, 0, f"stale: {why}", "weak", scorer="rule")
        else:
            eligible.append(j)
    if stale:
        print(f"  - temporal gate: {stale} stale posting(s) skipped before scoring")
    if not eligible:
        return jobs

    if not cfg["enabled"]:
        print("  - scorer: scoring disabled; falling back to keyword scoring")
        keyword(eligible, "keyword fallback, not read by model")
        return jobs

    def track_of(j):
        return j.get("track") or "career"

    # Resolve every track's model up front so a typo in `ai:` is reported
    # once, not once per batch.
    methods = {t: method_for(cfg, t) for t in TRACKS}
    for t, m in list(methods.items()):
        try:
            _profile_for(cfg, m, t)
        except llm.LLMError as e:
            print(f"  ! scorer ({t}): {e}; using keyword scoring")
            methods[t] = "keyword"

    def costs_money(j):
        p = _profile_for(cfg, methods[track_of(j)], track_of(j))
        return p is not None and not llm.is_free(p)

    # max_per_run is a COST ceiling, so it rations only the postings that
    # cost something. A locally-scored posting is free and being cut by a
    # budget cap made no sense: on one run that silently discarded ~550
    # bridge postings the local model would have handled in four minutes.
    # "Free" is the model's `cost`, so a local model on the rubric is free
    # and a hosted model behind jev is not.
    free = [j for j in eligible if not costs_money(j)]
    paid = [j for j in eligible if costs_money(j)]

    # Career first among the paid ones, so the cap eats bridge postings
    # rather than research ones when both are billed. Stable sort: order
    # within a track is untouched.
    paid.sort(key=lambda j: track_of(j) == "bridge")

    over = paid[cfg["max_per_run"]:]
    jobs_to_score = paid[: cfg["max_per_run"]] + free
    # What is left of the budget if a free method fails and its postings
    # fall through to a paid rubric model.
    budget = max(0, cfg["max_per_run"] - len(paid))
    if over:
        n_bridge_cut = sum(1 for j in over if track_of(j) == "bridge")
        if n_bridge_cut < len(over):
            print(f"  ! cap reached with {len(over) - n_bridge_cut} career "
                  f"posting(s) still unscored; raise --max-score")
        print(f"  ! {len(over)} past the {cfg['max_per_run']} paid cap, "
              f"left unscored" + (f" ({len(free)} free, not capped)"
                                  if free else ""))
    for j in over:
        mark(j, None, "over per-run scoring cap", "unscored", scorer="none")

    # Keyword-method tracks need nothing else.
    kw = [j for j in jobs_to_score if methods[track_of(j)] == "keyword"]
    if kw:
        keyword(kw, "keyword score (scoring method: keyword)")

    # jev first. Whatever it cannot handle falls through to the rubric
    # unchanged, so a dead server or a missing package costs a warning
    # rather than a failed run.
    rubric_tracks = {t for t in TRACKS if methods[t] == "rubric"}
    for track in [t for t in TRACKS if methods[t] == "jev"]:
        of_track = [j for j in jobs_to_score
                    if track_of(j) == track and j.get("scored_at") != now]
        if not of_track:
            continue
        try:
            import jev
        except ImportError:
            print("  ! jev: jev.py not importable; using the rubric scorer")
            rubric_tracks.add(track)
            continue
        prof = _profile_for(cfg, "jev", track)
        n, err = jev.score_track(
            of_track, cfg.get("jev"),
            focus=lambda j: _focus_description(
                j.get("description") or "",
                int((cfg.get("jev") or {}).get("desc_chars")
                    or cfg["desc_chars"])),
            track=track, facts_path=facts_path, profile=prof)
        if err:
            print(f"  ! jev ({track}): {err}; falling back to the rubric "
                  f"scorer")
            rubric_tracks.add(track)
            for j in of_track:
                j.pop("_jev", None)
            continue
        for j in of_track:
            r = j.pop("_jev", None)
            if r:
                j["score_detail"] = r.get("detail") or ""
                mark(j, r["score"], r["reason"], review=r["review"],
                     scorer="jev", model=llm.label(prof))

    jobs_to_score = [j for j in jobs_to_score
                     if j.get("scored_at") != now
                     and track_of(j) in rubric_tracks]
    if not jobs_to_score:
        return jobs

    try:
        profile = _profile(facts_path)
    except FileNotFoundError as e:
        print(f"  ! scorer: {e}; falling back to keyword scoring")
        keyword(jobs_to_score, "keyword fallback, no resume.yaml")
        return jobs

    def system_for(track):
        # config.yaml's `rubrics:` block wins; the module constants are the
        # fallback.
        rub = cfg.get("rubrics") or {}
        rubric = (rub.get(track)
                  or (BRIDGE_RUBRIC if track == "bridge" else RUBRIC))
        return [
            {"text": rubric.replace("{today}", _TODAY)},
            {"text": f"CANDIDATE RECORD (complete — nothing outside this "
                     f"exists):\n<facts>\n{profile}\n</facts>",
             # Anthropic prompt caching. Only takes effect above the model's
             # minimum cacheable length; other providers ignore the flag.
             "cache": True},
        ]

    # Batch by track so a single call never mixes rubrics. A profile can set
    # its own batch_size / desc_chars: a local model with an 8k context
    # cannot take twenty full postings plus a 12k-character resume at once.
    batches = []
    for track in TRACKS:
        of_track = [j for j in jobs_to_score if track_of(j) == track]
        if not of_track:
            continue
        try:
            prof = _profile_for(cfg, "rubric", track)
        except llm.LLMError as e:
            print(f"  ! rubric ({track}): {e}; using keyword scoring")
            keyword(of_track, "keyword fallback; rubric model misconfigured")
            continue
        problem = llm.check(prof)
        if problem:
            print(f"  - rubric ({track}): {prof['name']}: {problem}; "
                  f"falling back to keyword scoring")
            keyword(of_track, f"keyword fallback; {problem}")
            continue
        ok, _ = llm.ensure_ready(prof, quiet=True)
        if not ok:
            print(f"  ! rubric ({track}): {llm.describe(prof)} is not "
                  f"answering; falling back to keyword scoring")
            keyword(of_track, "keyword fallback; model server unavailable")
            continue
        print(f"  - rubric ({track}): {llm.describe(prof)}")
        if not llm.is_free(prof):
            # Postings that were budgeted as free (a jev track that failed)
            # are not allowed to run up a bill past the cap.
            fell = [j for j in of_track if not costs_money(j)]
            if len(fell) > budget:
                cut = fell[budget:]
                ids = {id(j) for j in cut}
                of_track = [j for j in of_track if id(j) not in ids]
                for j in cut:
                    mark(j, None, "over per-run scoring cap", "unscored",
                         scorer="none")
                print(f"  ! {len(cut)} {track} posting(s) fell back to a "
                      f"paid model past the {cfg['max_per_run']} cap; left "
                      f"unscored")
            budget -= min(budget, len(fell))
            if not of_track:
                continue
        size = int(prof.get("batch_size") or cfg["batch_size"])
        for start in range(0, len(of_track), size):
            batches.append((track, prof, of_track[start:start + size]))
    if not batches:
        return jobs

    total = sum(len(b) for _, _, b in batches)
    n_bridge = sum(len(b) for t, _, b in batches if t == "bridge")
    if n_bridge:
        print(f"  - {total - n_bridge} career / {n_bridge} bridge")

    # Some failures are permanent: an exhausted credit balance, a revoked
    # key, a model name that does not exist, a local server that is gone.
    # Retrying those 24 times produces 24 identical error lines and delays
    # the keyword fallback for no benefit. Detect them once per model and
    # stop asking that model.
    scored_so_far = 0
    dead = {}
    for bnum, (track, prof, batch) in enumerate(batches, 1):
        if prof["name"] in dead:
            keyword(batch, "keyword estimate; scorer unavailable")
            scored_so_far += len(batch)
            continue
        desc_chars = int(prof.get("desc_chars") or cfg["desc_chars"])
        try:
            text = llm.complete(prof, system_for(track),
                                _batch_prompt(batch, desc_chars),
                                max_tokens=int(prof.get("max_tokens")
                                               or 200 * len(batch) + 200))
            scores = _parse(text, len(batch))
        except llm.LLMError as e:
            if e.permanent:
                dead[prof["name"]] = e
                remaining = sum(len(b) for _, p, b in batches[bnum - 1:]
                                if p["name"] == prof["name"])
                print(f"  ! scorer stopped at batch {bnum}: {e}")
                print(f"  ! this will not fix itself on a retry. The "
                      f"remaining {remaining} posting(s) on {prof['name']} "
                      f"get keyword estimates and are recorded unscored; "
                      f"re-run `python score.py --backfill` once it is "
                      f"resolved.")
            else:
                print(f"  ! scorer batch {bnum} ({track}) failed: {e}")
            scores = {}

        for i, j in enumerate(batch):
            if i in scores:
                sc, reason, review = scores[i]
                mark(j, sc, reason, review=review, scorer="rubric",
                     model=llm.label(prof))
            else:
                # Model skipped this one, or the whole batch failed. Do not
                # let a parse failure look like a low score.
                keyword([j], "scoring failed, keyword estimate")

        scored_so_far += len(batch)
        print(f"  scored {scored_so_far}/{total}")

    return jobs


# ---------------------------------------------------------------------------
# CLI: calibration and backfill
# ---------------------------------------------------------------------------
def _load_cfg(path="config.yaml"):
    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    return merge_scoring_cfg(cfg), cfg


def method_for(cfg, track):
    """Which method scores this track. Track key wins over the global one."""
    return (cfg.get(f"{track}_method") or cfg.get("method") or "rubric")


provider_for = method_for          # old name, still imported elsewhere


def merge_scoring_cfg(cfg):
    """
    scoring: block + bridge.scoring: thresholds, flattened into one dict.

    Kept here rather than in main.py so the CLI, the backfill, and a real run
    all resolve thresholds identically.
    """
    raw = dict(cfg.get("scoring") or {})
    scfg = {**DEFAULTS, **raw}
    scfg["method"] = normalize_method(raw.get("method") or raw.get("provider")
                                      or DEFAULTS["method"])
    for track in TRACKS:
        tsc = ((cfg.get(track) or {}).get("scoring") or {})
        m = tsc.get("method") or tsc.get("provider")
        if m:
            scfg[f"{track}_method"] = normalize_method(m)
        if track == "bridge":
            for key in ("strong_at", "maybe_at"):
                if key in tsc:
                    scfg[f"bridge_{key}"] = tsc[key]
    scfg["jev"] = cfg.get("jev") or {}
    scfg["rubrics"] = cfg.get("rubrics") or {}
    scfg["professional_titles"] = cfg.get("professional_titles") or []
    # The whole file, so llm.resolve can read `models:` and `ai:`.
    scfg["_root"] = cfg
    return scfg


def _side(spec, scfg, track):
    """
    "jev", "rubric", "jev:qwen", "rubric:haiku" -> (label, scfg for one run).
    The profile after the colon is a name under `models:` in config.yaml.
    """
    import llm
    method, _, name = spec.partition(":")
    method = normalize_method(method)
    if method not in ("jev", "rubric"):
        sys.exit(f"--compare side {spec!r}: use jev, rubric, jev:PROFILE "
                 f"or rubric:PROFILE")
    run = {**scfg, "method": method, f"{track}_method": method}
    if name:
        try:
            run[f"{method}_profile"] = llm._lookup(scfg["_root"], name, method)
        except llm.LLMError as e:
            sys.exit(str(e))
    prof = _profile_for(dict(run), method, track)
    return f"{method}:{prof['name']}", run


def cmd_compare(args):
    """
    Score the same stored postings two ways and diff them.

    Either side is a method on a model: `jev`, `rubric`, or either with a
    profile name, so this answers "can jev carry the track" and also "is the
    local model as good as the hosted one at the same rubric". Writes nothing.

    The left side defaults to the rubric scores ALREADY in seen.db: there are
    thousands, they cost nothing to read, and re-scoring them to learn the
    same thing is the expensive way. --live (or --left SPEC) scores fresh.

    What matters is agreement on the RANKING and on which side of strong_at
    a posting falls, not on the exact integers. Different scales were built
    to answer the same question, not to emit the same numbers.
    """
    import store
    conn = store.connect()
    track = args.track or "bridge"
    scfg, cfg = _load_cfg(args.config)
    left_spec = args.left or ("rubric" if args.live else None)
    right_spec = args.right or "jev"

    if left_spec:
        rows = list(conn.execute(
            "SELECT uid, title, company, location, posted, description, "
            "desc_quality, track, NULL, NULL FROM jobs "
            "WHERE track = ? AND description IS NOT NULL "
            "AND length(description) > 200 "
            "ORDER BY first_seen DESC LIMIT ?", (track, args.limit)))
    else:
        rows = list(conn.execute(
            "SELECT uid, title, company, location, posted, description, "
            "desc_quality, track, score, reason FROM jobs "
            "WHERE track = ? AND description IS NOT NULL "
            "AND length(description) > 200 "
            "AND score IS NOT NULL AND tier != 'unscored' "
            "AND (scorer IN ('rubric', 'anthropic') OR scorer IS NULL) "
            "ORDER BY first_seen DESC LIMIT ?", (track, args.limit)))
    if not rows:
        sys.exit(f"no {track} postings with both a description and a stored "
                 f"rubric score. Run `python main.py` first, or pass --live "
                 f"(or --left SPEC) to score a fresh batch.")

    cols = ("uid", "title", "company", "location", "posted", "description",
            "desc_quality", "track", "stored_score", "stored_reason")
    base = [dict(zip(cols, r)) for r in rows]

    if left_spec:
        lname, lrun = _side(left_spec, scfg, track)
        print(f"comparing {len(base)} {track} postings: {lname} (fresh) "
              f"vs {right_spec}\n")
        a = [dict(j) for j in base]
        score_jobs(a, lrun, args.facts)
    else:
        lname = "stored"
        print(f"comparing {len(base)} {track} postings against their stored "
              f"rubric scores (no calls on that side)\n")
        a = [dict(j, score=j["stored_score"], reason=j["stored_reason"],
                  scorer="rubric") for j in base]

    rname, rrun = _side(right_spec, scfg, track)
    b = [dict(j) for j in base]
    score_jobs(b, rrun, args.facts)

    # A side that fell back to keyword estimates has not scored anything.
    # Printing "+0" against another failed side looks like perfect agreement
    # and is the opposite: it is two failures matching. Refuse to draw it.
    def real(rows):
        n = sum(1 for j in rows if j.get("scorer") in ("rubric", "jev"))
        return n, len(rows)

    for name, rows in ((lname, a), (rname, b)):
        ok, total = real(rows)
        if ok < total / 2:
            reasons = Counter(j.get("scorer") for j in rows)
            sys.exit(
                f"\n{name} scored only {ok}/{total} postings for real "
                f"({', '.join(f'{k}={v}' for k, v in reasons.most_common())}).\n"
                f"The rest are keyword estimates, so a comparison would show "
                f"two failures agreeing with each other rather than two "
                f"scorers agreeing.\n"
                f"Fix: `python preflight.py --probe` shows which model is "
                f"unreachable and why.")

    strong_at, _ = thresholds(scfg, track)
    print(f"\n  left  = {lname}\n  right = {rname}")
    print(f"\n{'':>4} {'left':>6} {'right':>5} {'d':>5}  title")
    print("-" * 74)
    flips = 0
    pairs = []
    for x, y in zip(a, b):
        sx, sy = x.get("score"), y.get("score")
        if sx is None or sy is None:
            continue
        pairs.append((sx, sy))
        flip = (sx >= strong_at) != (sy >= strong_at)
        flips += flip
        print(f"{'!!' if flip else '':>4} {sx:>6} {sy:>5} {sy-sx:>+5}  "
              f"{(x.get('title') or '')[:48]}")
        if flip:
            print(f"{'':>18}left:  {(x.get('reason') or '')[:52]}")
            print(f"{'':>18}right: {(y.get('reason') or '')[:52]}")

    if len(pairs) < 2:
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    # Spearman on ranks: do they order the postings the same way? Exact
    # agreement on integers is not the goal and never was.
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(pairs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d2) / (n * (n * n - 1)) if n > 1 else 0.0

    print("-" * 74)
    print(f"  n {n}   mean left {statistics.mean(xs):.1f}   "
          f"mean right {statistics.mean(ys):.1f}   "
          f"mean shift {statistics.mean(ys)-statistics.mean(xs):+.1f}")
    print(f"  spread left {max(xs)-min(xs)}   right {max(ys)-min(ys)}")
    print(f"  rank correlation (Spearman) {rho:.2f}")
    print(f"  disagree on strong_at={strong_at}: {flips}/{n} "
          f"({flips/n:.0%}), marked !! above")
    print("\n  rho above ~0.8 with few flips means the right side can carry "
          "the track.")
    print("  Read the flipped rows: that is where the two scorers differ.")


def cmd_calibrate(args):
    import store
    conn = store.connect()
    rows = list(conn.execute(
        "SELECT title, company, score, tier, reason, track, scorer, "
        "scorer_model FROM jobs "
        "WHERE score IS NOT NULL ORDER BY score DESC"))
    if not rows:
        sys.exit("no scored jobs yet. Run `python score.py --backfill` first.")

    scfg, _ = _load_cfg(args.config)

    # Per track, always. A bridge 80 and a career 80 answer different
    # questions, so one pooled histogram would have you raising strong_at
    # because a distribution centre had a hiring day.
    present = [t for t in TRACKS
               if any((r[5] or "career") == t for r in rows)]
    if args.track:
        present = [args.track]

    for idx, track in enumerate(present):
        sub = [r for r in rows if (r[5] or "career") == track]
        # A jev 72 and a rubric 72 are different scales, and so are two
        # models on the same rubric. Pooling them is the same mistake as
        # pooling the two tracks, so say when a track is mixed.
        def who(r):
            fam = scorer_family(r[6])
            if fam in ("keyword", "rule", "none"):
                return None
            return f"{fam}:{r[7]}" if r[7] else f"{fam} (before model tags)"
        by_scorer = Counter(w for w in map(who, sub) if w)
        if len(by_scorer) > 1:
            print(f"\n  ! mixed scorers in this track: "
                  + ", ".join(f"{k}={v}" for k, v in by_scorer.most_common())
                  + "\n    the histogram below pools different scales")
        if not sub:
            print(f"\nno {track} postings scored yet.")
            continue
        strong_at, maybe_at = thresholds(scfg, track)

        if len(present) > 1 or args.track:
            print(f"\n{'=' * 58}\n{track.upper()}\n{'=' * 58}")
        elif idx:
            print()
        buckets = Counter((r[2] // 10) * 10 for r in sub)
        print(f"{len(sub)} scored postings\n")
        for lo in range(90, -1, -10):
            n = buckets.get(lo, 0)
            bar = "#" * min(50, n)
            print(f"  {lo:>3}-{lo + 9:<3} {n:>4}  {bar}")

        print(f"\nAt current thresholds "
              f"(strong>={strong_at}, maybe>={maybe_at}):")
        t = Counter(r[3] for r in sub)
        for name in TIERS:
            if t.get(name):
                print(f"  {name:<9} {t[name]:>4}")

        print("\nHighest-scoring:")
        for r in sub[:8]:
            print(f"  {r[2]:>3}  {r[0][:44]:<46} {(r[4] or '')[:44]}")
        print("\nJust below the maybe line — the ones a blocklist would have "
              "hidden:")
        near = [r for r in sub if maybe_at - 15 <= r[2] < maybe_at]
        for r in near[:8]:
            print(f"  {r[2]:>3}  {r[0][:44]:<46} {(r[4] or '')[:44]}")
        if not near:
            print("  (none)")


def cmd_backfill(args):
    import store
    scfg, cfg = _load_cfg(args.config)
    signals = [k.lower() for k in cfg.get("relevance_signals", [])]
    conn = store.connect()

    params = []
    if args.uid:
        # One posting, whatever its current state. Used by the UI's
        # "Rescore" button and handy after editing a single rubric line.
        where = "uid = ?"
        params.append(args.uid)
        note = f"scoring posting {args.uid}"
    elif args.rescore:
        # After editing a rubric or a band list, existing scores are stale.
        # This re-scores rows that already have a number, which the normal
        # backfill skips.
        #
        # It used to be WHERE 1=1, which overwrote every stored score
        # including the hosted ones. Those are the only free baseline for
        # --compare, and once a local score lands on top of them they are
        # gone. So: --scorer targets one provider, and a bare --rescore that
        # would clobber a provider you are no longer using stops and says so.
        if args.scorer:
            # A method (jev, rubric), a profile name, or name:model exactly
            # as stored. "rubric" also matches rows from before the rename.
            fam = normalize_method(args.scorer) if args.scorer.lower() in (
                *METHODS, *_METHOD_ALIASES) else None
            if fam == "rubric":
                where = ("score IS NOT NULL AND (scorer IN ('rubric', "
                         "'anthropic') OR scorer IS NULL)")
            elif fam:
                where = "score IS NOT NULL AND scorer = ?"
                params.append(fam)
            else:
                where = ("score IS NOT NULL AND (scorer_model = ? "
                         "OR scorer_model LIKE ?)")
                params += [args.scorer, f"{args.scorer}:%"]
            note = f"re-scoring stored postings last scored by {args.scorer}"
        else:
            in_use = {method_for(scfg, t) for t in TRACKS}
            at_risk = Counter(
                scorer_family(r[0]) for r in conn.execute(
                    "SELECT scorer FROM jobs WHERE score IS NOT NULL "
                    "AND tier != 'unscored'"))
            clobber = {k: v for k, v in at_risk.items()
                       if k not in in_use and k not in ("keyword", "none",
                                                        "rule")}
            if clobber and not args.yes:
                print("re-scoring everything would overwrite scores from a "
                      "method you are not currently using:")
                for k, v in sorted(clobber.items()):
                    print(f"  {v} row(s) scored by {k}")
                print("\nThose are your only free baseline for "
                      "`--compare`; once a new score lands on them it is "
                      "gone.\n"
                      "  python score.py --backfill --rescore --scorer jev"
                      "    # redo only jev scores\n"
                      "  python score.py --backfill --rescore --yes"
                      "              # overwrite everything anyway")
                return
            where = "score IS NOT NULL"
            note = "re-scoring ALL stored postings"
    else:
        # tier='unscored' means the model never actually read it — the row has
        # a keyword-fallback number, not a judgement. Those are backfillable.
        where = "score IS NULL OR tier = 'unscored' OR tier IS NULL"
        note = "scoring postings the model has not read"

    rows = list(conn.execute(
        f"SELECT uid, title, company, location, posted, description, "
        f"desc_quality, track FROM jobs "
        f"WHERE {where} ORDER BY first_seen DESC LIMIT ?",
        (*params, args.limit)))
    if not rows:
        print("nothing to do — every stored posting already has a real score.")
        print("Use --backfill --rescore to re-score them after a rubric edit.")
        return

    jobs = [dict(zip(("uid", "title", "company", "location", "posted",
                      "description", "desc_quality", "track"), r))
            for r in rows]

    # Rows stored before the bridge track existed have track = NULL. Classify
    # them now from the same config main.py uses, so a --rescore run reads
    # warehouse work with the warehouse rubric instead of the career one.
    (career_terms, bridge_terms, bridge_exclude,
     bridge_tolerated) = track_terms(cfg)
    filled = 0
    for j in jobs:
        if not j.get("track"):
            j["track"] = (classify_track(j, career_terms, bridge_terms,
                                         bridge_exclude, bridge_tolerated,
                                         professional_titles(cfg),
                                         bridge_never(cfg))
                          or "career")
            filled += 1
    if filled:
        print(f"  - classified {filled} row(s) with no stored track")
    print(f"{note}: {len(jobs)} rows")
    scfg = {**scfg, "max_per_run": max(scfg["max_per_run"], len(jobs))}
    score_jobs(jobs, scfg, args.facts, signals)
    store.update_scores(conn, jobs)

    got = sum(1 for j in jobs if j["tier"] != "unscored")
    print(f"\nUpdated {len(jobs)} rows ({got} scored by the model).")
    print("Next: python score.py --calibrate")


def cmd_test(args):
    scfg, cfg = _load_cfg(args.config)
    (career_terms, bridge_terms, bridge_exclude,
     bridge_tolerated) = track_terms(cfg)
    job = {"title": args.test, "company": "", "location": "",
           "posted": args.posted or "", "description": args.desc or ""}
    auto = classify_track(job, career_terms, bridge_terms, bridge_exclude,
                          bridge_tolerated, professional_titles(cfg),
                          bridge_never(cfg))
    if args.track:
        job["track"] = args.track
    elif auto is None:
        print(f"  {args.test!r} is dropped by bridge.exclude before scoring. "
              f"Forcing the career rubric; use --track bridge to override.")
        job["track"] = "career"
    else:
        job["track"] = auto
    print(f"  track: {job['track']}"
          + ("" if args.track else " (auto)"))
    score_jobs([job], scfg, args.facts,
               [k.lower() for k in cfg.get("relevance_signals", [])])
    print(f"\n  {job['score']}  [{job['tier']}]  {job['reason']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="with --backfill: re-score rows that already have a "
                         "score. Use after editing RUBRIC.")
    ap.add_argument("--test", help="score a single title, for prompt tuning")
    ap.add_argument("--desc", help="description to go with --test")
    ap.add_argument("--posted", help="YYYY-MM-DD posting date to go with --test")
    ap.add_argument("--uid",
                    help="with --backfill, score just this posting (full uid)")
    ap.add_argument("--scorer",
                    help="with --rescore, redo only rows last scored by this "
                         "method (jev, rubric, keyword) or model profile name")
    ap.add_argument("--yes", action="store_true",
                    help="with --rescore, overwrite scores from methods you "
                         "are no longer using")
    ap.add_argument("--live", action="store_true",
                    help="with --compare, same as --left rubric: a fresh "
                         "rubric pass instead of the scores already in seen.db")
    ap.add_argument("--left", metavar="SPEC",
                    help="with --compare: jev, rubric, jev:PROFILE or "
                         "rubric:PROFILE. Default: stored rubric scores")
    ap.add_argument("--right", metavar="SPEC", default=None,
                    help="with --compare: the side to test. Default: jev "
                         "on its configured model")
    ap.add_argument("--compare", action="store_true",
                    help="score stored postings two ways (see --left/--right) "
                         "and diff them. Writes nothing.")
    ap.add_argument("--track", choices=TRACKS,
                    help="force a rubric for --test, or pick the track for "
                         "--compare (default bridge)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--facts", default="resume.yaml")
    args = ap.parse_args()

    if args.compare:
        return cmd_compare(args)
    if args.calibrate:
        cmd_calibrate(args)
    elif args.backfill:
        cmd_backfill(args)
    elif args.test:
        cmd_test(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
