#!/usr/bin/env python3
"""
paths.py - sort career postings by YOUR rules instead of scoring them.

Two questions, both answered from what the posting itself says:

  FIELD   Is the title one of your career paths?          (config: paths.fields)
  LEVEL   Does the posting ask for more than you have?    (config: paths.level)

and three sections in the digest:

  Entry-level in your fields   in a path, and nothing stated is above you
  Stretch                      in a path, but it states something above you
                               (more years, a PhD, current enrolment...)
  Filtered out                 not one of your paths, or a rule says out;
                               one line each, with the reason

No model and no score. Every rule is a line in config.yaml under `paths:`,
and every posting carries the rule that placed it. When a posting states
nothing about experience it is entry-level, with "no minimum stated", not a
guess.

    python paths.py --test                 # rules vs your applications and seen.db
    python paths.py --test tracker.csv     # same, tracker from a CSV
    python paths.py --title "Data Analyst II" [--desc "..."]

Switch the digest to it with, in config.yaml:

    scoring:
      method: jev
      career:              # <- this block already exists as a comment
        scoring:
          method: filter

(or `career: {scoring: {method: filter}}` at the top level, which is what
score.merge_scoring_cfg reads).
"""
import copy
import re

DEFAULTS = {
    # Career paths, first match wins, so list the narrow ones first. A title
    # is in a path when it contains one of `titles` (whole words; a trailing
    # * is a prefix; "re:..." is a regular expression).
    "fields": [
        {"name": "ml_ai", "label": "ML & AI research",
         "titles": ["machine learning", "ml engineer", "ml researcher",
                    "research scientist", "research engineer",
                    "applied scientist", "ai research*", "researcher",
                    "member of technical staff", "nlp", "deep learning",
                    "ai engineer", "ai resident", "research resident",
                    "quantitative developer", "quantitative researcher"]},
        {"name": "econ_policy", "label": "Economics, policy & social research",
         "titles": ["economist", "economic*", "econometric*", "policy",
                    "research analyst", "research associate",
                    "research assistant", "research fellow",
                    "research coordinator", "program evaluation",
                    "evaluation", "statistician", "statistic*", "survey",
                    "social science", "quantitative research*", "research"]},
        {"name": "data", "label": "Data & business analysis",
         "titles": ["data analyst", "data scientist", "data science",
                    "data insights", "data specialist", "data associate",
                    "data consultant", "analytics", "business intelligence",
                    "bi analyst", "decision scien*", "quantitative",
                    "ai consultant", "analytics consultant",
                    "consulting analyst", "consulting associate", "analyst"]},
    ],

    # Never a career path, whatever else the title says. Your hard_exclude
    # list already removes most of these before anything is read.
    "not_titles": ["sales", "account executive", "recruiter", "sourcer",
                   "nurse", "teacher", "tutor", "driver", "product manager",
                   "technical program manager", "engineering manager",
                   "counsel", "attorney"],

    # Fields outside your record, by title word: filtered out.
    "other_fields": [
        "bioinformatic*", "biolog*", "biochem*", "chemist*", "chemical",
        "genomic*", "genetic*", "protein*", "molecular", "immunolog*",
        "neuroscien*", "pharmacolog*", "virolog*", "microbiolog*",
        "robot*", "hardware", "compiler", "firmware", "embedded", "physics",
        "physicist", "medical imaging", "histolog*", "wet lab", "wet-lab",
        "security", "network engineer", "devops", "site reliability"],

    # Lab science the title does not reveal ("Research Associate II"):
    # filtered out when the DESCRIPTION uses at least `other_fields_min` of
    # these. One alone can be context ("data from clinical trials").
    "other_fields_text": [
        "wet lab", "wet-lab", "cell culture", "assay*", "pcr", "pipett*",
        "in vivo", "in vitro", "flow cytometry", "protein expression",
        "mass spectrometry", "chromatography", "biosafety", "aseptic",
        "tissue", "antibod*", "reagent*", "bench work", "benchwork",
        "laboratory techniques", "sequencing librar*", "crispr"],
    "other_fields_min": 2,

    "level": {
        # Each rule sends a posting to "stretch" or "out" (or "ignore").
        "senior_title": "out",       # words in senior_words below
        "mid_title": "stretch",      # words in mid_words below
        "phd_required": "stretch",
        "masters_required": "stretch",
        "enrolment_required": "stretch",
        "clearance_required": "out",
        "language_required": "out",
        # A stated experience minimum, in years. YOUR numbers: at or above
        # stretch_years goes to Stretch, at or above out_years is filtered
        # out. Sentences that count internships or graduate research are
        # skipped, because those count your record.
        "stretch_years": 3,
        "out_years": 6,
    },
    "senior_words": [
        "lead", "manager", "managing", "supervisor", "supervisory", "head",
        "principal", "senior", "sr", "director", "chief", "architect",
        r"re:\bstaff\s+(?:[\w/&-]+\s+){0,3}?(?:engineer|scientist|"
        r"researcher|developer|designer|member)\b",
        r"re:\b(?:iii|iv)\b", r"re:\blevel\s*[3-9]\b"],
    # Not senior even though a word above appears.
    "not_senior_words": ["assistant manager", "associate manager",
                         "project manager", "program manager",
                         "product manager", "project lead", "lead generation",
                         "data lead*"],
    "mid_words": [r"re:\bii\b", r"re:\s2$", r"re:\b2\b(?=\s*[-–(,]|$)",
                  "1 or 2", "experienced", "level 2", "level ii",
                  "mid-level", "mid level"],
    "enrolment_words": ["intern", "interns", "internship", "internships",
                        "extern", "externship", "co-op", "coop",
                        "summer analyst", "summer associate",
                        "college aide", "graduate student", "student"],

    # Shown, or folded into Filtered out.
    "stretch": "show",
}

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def conf(root):
    """DEFAULTS merged with root['paths']. Dicts update, lists replace."""
    out = copy.deepcopy(DEFAULTS)
    for k, v in ((root or {}).get("paths") or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        elif v is not None:
            out[k] = v
    return out


def validate(root):
    problems = []
    p = conf(root)
    for k in ((root or {}).get("paths") or {}):
        if k not in DEFAULTS and k != "enabled":
            problems.append(f"paths.{k} is not a known key")
    for f in p["fields"]:
        if not f.get("titles"):
            problems.append(f"paths.fields: {f.get('name')!r} has no titles")
    for k, v in p["level"].items():
        if k.endswith("_years"):
            if v is not None and not isinstance(v, (int, float)):
                problems.append(f"paths.level.{k} must be a number or null")
        elif v not in ("stretch", "out", "ignore"):
            problems.append(f"paths.level.{k} must be stretch, out or ignore")
    for name in ("senior_words", "not_senior_words", "mid_words",
                 "enrolment_words", "other_fields", "not_titles",
                 "other_fields_text"):
        for w in p[name]:
            if str(w).startswith("re:"):
                try:
                    re.compile(w[3:])
                except re.error as e:
                    problems.append(f"paths.{name}: bad regex {w!r}: {e}")
    return problems


# --------------------------------------------------------------------------
# Word lists
# --------------------------------------------------------------------------
_CACHE = {}


def _rx(words):
    key = tuple(words or ())
    if key not in _CACHE:
        parts = []
        for w in key:
            w = str(w)
            if w.startswith("re:"):
                parts.append(f"(?:{w[3:]})")
            elif w.endswith("*"):
                parts.append(r"\b" + re.escape(w[:-1]) + r"\w*")
            else:
                parts.append(r"\b" + re.escape(w) + r"(?!\w)")
        _CACHE[key] = re.compile("|".join(parts) or r"(?!x)x", re.I)
    return _CACHE[key]


def hit(words, text):
    m = _rx(words).search(text or "")
    return " ".join(m.group(0).lower().split()) if m else None


def all_title_terms(root):
    """Every plain title term in your paths, for the fetch-time filters."""
    out = []
    for f in conf(root)["fields"]:
        for t in f.get("titles") or []:
            t = str(t)
            if not t.startswith("re:"):
                out.append(t.rstrip("*").lower())
    return out


# --------------------------------------------------------------------------
# Reading requirements out of the posting text
# --------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?])\s+|\n+|\s*[•·▪●]\s*|;\s+")
_SOFT = re.compile(r"\b(?:preferred|a plus|nice[- ]to[- ]have|bonus|desired|"
                   r"desirable|ideally|ideal candidate|helpful|advantageous|"
                   r"not required|or equivalent|equivalent experience)\b", re.I)
_ACADEMIC = re.compile(r"\b(?:intern(?:ship)?s?|academic|graduate "
                       r"(?:school|research|study|work)|coursework|research "
                       r"assistant\w*|fellowships?|including education)\b",
                       re.I)
_BOILER = re.compile(r"\b(?:we have|we've|has been|founded|since \d{4}|"
                     r"in business|(?:with|for) (?:over|more than|nearly) "
                     r"\d+ years)\b", re.I)
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_YEARS = re.compile(
    r"(?<![\d.$])\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(?:\(\d+\)\s*)?(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d{1,2}\s*)?\+?\s*"
    r"(?:years?|yrs?)\b", re.I)
_PHD = re.compile(r"\bph\.?\s?d\b\.?|\bdoctora(?:te|l)\b", re.I)
_MASTERS = re.compile(r"\bmaster'?s\b|\bm\.?s\.? (?:degree|in)\b|"
                      r"\bm\.?a\.? (?:degree|in)\b|\bmba\b", re.I)
_BACHELOR = re.compile(r"\b(?:bachelor'?s?|b\.?s\.?|b\.?a\.?|undergraduate)"
                       r"\b|\bbs/|\bba/", re.I)
_NEED = re.compile(r"\b(?:required|requires?|must|minimum|need(?:ed)?)\b",
                   re.I)
_ENROL_TEXT = re.compile(
    r"\b(?:currently enrolled|currently pursuing|must be enrolled|enrolled "
    r"(?:in|at) an? (?:accredited )?(?:college|university|degree|"
    r"bachelor|master|phd)|returning to (?:school|university)|rising "
    r"(?:junior|senior)s?|must be a (?:current )?student)\b", re.I)
_CLEAR = re.compile(r"\b(?:active|current)\s+(?:\w+\s+){0,3}clearance\b|"
                    r"\bts/sci\b|\btop secret\b", re.I)
_LANGS = ("Spanish|French|German|Italian|Portuguese|Mandarin|Cantonese|"
          "Chinese|Japanese|Korean|Arabic|Hindi|Urdu|Bengali|Punjabi|Russian|"
          "Ukrainian|Polish|Haitian Creole|Tagalog|Vietnamese|Hebrew|Farsi|"
          "Turkish|Greek")
_LANG = re.compile(rf"\b(?:fluen\w*|native|bilingual|proficien\w*)\b[^.]{{0,40}}"
                   rf"\b({_LANGS})\b|\b({_LANGS})\b[^.]{{0,20}}"
                   rf"\b(?:fluen\w*|required|proficien\w*)\b")


def sentences(text):
    return [s.strip() for s in _SENT.split(text or "") if s.strip()]


def years_required(text):
    """
    The stated years-of-experience minimum, or None. Within one sentence the
    smallest number wins ("5 years, or 3 with a master's"); across sentences
    the largest. Soft, academic-counting and company-history sentences are
    skipped.
    """
    floors = []
    for s in sentences(text):
        low = s.lower()
        if "experience" not in low and "work history" not in low:
            continue
        if _SOFT.search(s) or _ACADEMIC.search(s) or _BOILER.search(s):
            continue
        nums = []
        for m in _YEARS.finditer(s):
            raw = m.group(1).lower()
            n = _WORDNUM.get(raw) or (int(raw) if raw.isdigit() else 0)
            if 1 <= n <= 15:
                nums.append(n)
        if nums:
            floors.append(min(nums))
    return max(floors) if floors else None


def degree_required(text, rx):
    """A sentence naming this degree as needed, with no softer alternative."""
    for s in sentences(text):
        if (rx.search(s) and not _SOFT.search(s) and not _BACHELOR.search(s)
                and (_NEED.search(s) or len(s) < 90)):
            return True
    return False


def requirements(text):
    """{rule: evidence} for everything stated in the text."""
    t = text or ""
    out = {}
    y = years_required(t)
    if y:
        out["years"] = y
    if degree_required(t, _PHD):
        out["phd_required"] = "PhD required"
    elif degree_required(t, _MASTERS):
        out["masters_required"] = "master's required"
    if _ENROL_TEXT.search(t):
        out["enrolment_required"] = "current enrolment required"
    for s in sentences(t):
        if _CLEAR.search(s) and not re.search(r"\b(?:obtain|eligib|able to)\b",
                                              s, re.I):
            out["clearance_required"] = "active clearance required"
            break
    for s in sentences(t):
        m = _LANG.search(s)
        if m and not _SOFT.search(s):
            out["language_required"] = f"{m.group(1) or m.group(2)} required"
            break
    return out


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------
TIER = {"entry": "strong", "stretch": "maybe", "out": "weak"}


def field_of(title, p):
    if hit(p["not_titles"], title):
        return None, f"not a career path ({hit(p['not_titles'], title)})"
    other = hit(p["other_fields"], title)
    if other:
        return None, f"outside your fields ({other})"
    for f in p["fields"]:
        if hit(f.get("titles"), title):
            return f, None
    return None, "not in your fields"


def lab_science(desc, p):
    """The lab-science words a description uses, if there are enough."""
    found = {m.group(0).lower() for m in
             _rx(p["other_fields_text"]).finditer(desc or "")}
    return sorted(found) if len(found) >= int(p["other_fields_min"]) else None


def classify(job, p):
    """
    (section, path label or None, [reasons]). section: entry | stretch | out.
    """
    title = job.get("title") or ""
    path, why = field_of(title, p)
    if path is None:
        return "out", None, [why]
    lab = lab_science(job.get("description"), p)
    if lab:
        return "out", None, [f"lab science ({', '.join(lab[:3])})"]
    lv = p["level"]
    found = []                          # (action, reason)

    senior = (None if hit(p["not_senior_words"], title)
              else hit(p["senior_words"], title))
    if senior:
        found.append((lv["senior_title"], f"senior title ({senior})"))
    elif hit(p["mid_words"], title):
        found.append((lv["mid_title"], "mid-level title"))
    enrol = hit(p["enrolment_words"], title)
    if enrol:
        found.append((lv["enrolment_required"], f"for students ({enrol})"))

    req = requirements(job.get("description") or "")
    y = req.pop("years", None)
    if y is not None:
        if lv.get("out_years") is not None and y >= lv["out_years"]:
            found.append(("out", f"{y}+ years stated"))
        elif lv.get("stretch_years") is not None and y >= lv["stretch_years"]:
            found.append(("stretch", f"{y}+ years stated"))
    for rule, ev in req.items():
        if rule == "enrolment_required" and enrol:
            continue
        found.append((lv.get(rule, "stretch"), ev))

    found = [(a, r) for a, r in found if a != "ignore"]
    if any(a == "out" for a, _ in found):
        section = "out"
    elif found:
        section = "stretch" if p.get("stretch", "show") == "show" else "out"
    else:
        section = "entry"
    reasons = [r for _, r in found]
    if section == "entry":
        reasons = ["no minimum stated" if y is None else f"{y} year(s) stated"]
    return section, path["label"], reasons


def verdict(job, p):
    """(tier, reason) for score.py: strong=entry, maybe=stretch, weak=out."""
    section, label, reasons = classify(job, p)
    if section == "out":
        return "weak", "; ".join(reasons)
    head = label + (" · entry-level" if section == "entry" else " · stretch")
    return TIER[section], f"{head}: {'; '.join(reasons)}"


# --------------------------------------------------------------------------
# --test
# --------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ",
                                      (s or "").lower())).strip()


def run_test(args, root):
    import random
    import sqlite3
    from collections import Counter
    from datetime import date, timedelta
    p = conf(root)
    for prob in validate(root):
        print(f"! {prob}")

    # ---- your applications --------------------------------------------
    import fit
    apps = []
    try:
        jcfg = (root.get("jev") or {})
        apps, source = fit.load_tracker(args.test or None, jcfg)
    except Exception as e:
        print(f"(no tracker: {e})\n")
    conn = sqlite3.connect(args.db)
    have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    cols = ["uid", "title", "company", "location", "url", "description",
            "first_seen", "tier", "score", "track"]
    sel = ", ".join(c if c in have else f"NULL AS {c}" for c in cols)
    rows = [dict(zip(cols, r)) for r in conn.execute(
        f"SELECT {sel} FROM jobs WHERE (track = 'career' OR track IS NULL)")]
    if apps:
        found = fit.match_tracker(apps, rows)
        print(f"YOUR {len(apps)} APPLICATIONS "
              f"({len(found)} with stored text; the rest by title only)\n")
        counts = Counter()
        lost = []
        for n, a in enumerate(apps):
            job = found.get(n) or {"title": a["title"], "description": ""}
            sec, label, why = classify(job, p)
            counts[sec] += 1
            if sec != "entry":
                lost.append((sec, a, label, why, n in found))
        for sec in ("entry", "stretch", "out"):
            print(f"  {sec:<8} {counts[sec]:>3}")
        print("\n  Not entry-level (read these: 'out' is a job you applied to "
              "that you would stop seeing):")
        for sec, a, label, why, txt in sorted(lost, key=lambda x: x[0]):
            print(f"  {sec:<8} {a['title'][:44]:<46} {'; '.join(why)[:46]}"
                  + ("" if txt else "  [title only]"))
        print()

    # ---- stored postings ----------------------------------------------
    since = (date.today() - timedelta(days=args.days)).isoformat()
    recent = [r for r in rows if str(r.get("first_seen") or "") >= since
              and r.get("track") in ("career", None)]
    if not recent:
        print(f"no career postings first seen in the last {args.days} days")
        return
    runs = len({str(r["first_seen"])[:16] for r in recent})
    out = Counter()
    sec_of = {}
    for r in recent:
        sec, label, why = classify(r, p)
        sec_of[r["uid"]] = (sec, label, why)
        out[sec if sec != "out" else "out: " + re.sub(r"\s*\(.*", "",
                                                      why[0])] += 1
    n = len(recent)
    print(f"CAREER POSTINGS FIRST SEEN IN THE LAST {args.days} DAYS: {n} "
          f"over about {runs} runs\n")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>5}  {k:<36} about {v / max(runs, 1):.0f} per run")
    old_strong = [r for r in recent if r.get("tier") == "strong"]
    if old_strong:
        c = Counter(sec_of[r["uid"]][0] for r in old_strong)
        print(f"\n  Of the {len(old_strong)} the old scorer put in 'worth "
              f"applying to': " + ", ".join(f"{c[s]} {s}" for s in
                                             ("entry", "stretch", "out")))
    rnd = random.Random(0)
    for sec, k in (("entry", 15), ("stretch", 10)):
        pick = [r for r in recent if sec_of[r["uid"]][0] == sec]
        rnd.shuffle(pick)
        print(f"\n  A random {min(k, len(pick))} from {sec}:")
        for r in pick[:k]:
            _, label, why = sec_of[r["uid"]]
            print(f"    {r['title'][:46]:<48} {(label or '')[:18]:<19} "
                  f"{'; '.join(why)[:30]}")
    if any(not (r.get("description") or "").strip() for r in recent):
        k = sum(1 for r in recent if not (r.get("description") or "").strip())
        print(f"\n  {k} posting(s) have no description, so only their title "
              f"rules applied.")


def main():
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description="Career paths and level rules.")
    ap.add_argument("--test", nargs="?", const="", metavar="TRACKER",
                    help="check the rules against your applications and "
                         "seen.db (tracker defaults to jev.fit.tracker)")
    ap.add_argument("--title", help="classify one title")
    ap.add_argument("--desc", default="", help="description for --title")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--db", default="seen.db")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    root = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    if args.title:
        sec, label, why = classify({"title": args.title,
                                    "description": args.desc}, conf(root))
        return print(f"{sec}  {label or '-'}  {'; '.join(why)}")
    if args.test is not None:
        return run_test(args, root)
    ap.print_help()


if __name__ == "__main__":
    main()
