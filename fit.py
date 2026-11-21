#!/usr/bin/env python3
"""
fit.py - jev scoring where the model sees the candidate and the weights are
learned from the jobs you actually applied to.

How a career score is made
--------------------------
1. Your record (built from resume.yaml) goes above every posting.
2. The model answers a few questions about the posting RELATIVE to your
   record (`career_features`): how closely the duties match, whether it
   needs a graduate degree, current enrolment, other credentials, the
   experience it asks for, whether it is open now.
3. Those answers are combined with weights LEARNED from your applications
   (`python fit.py --learn tracker.csv`): a logistic regression of "did you
   apply" on each answer. The score is 100 x the fitted probability that a
   posting looks like one you applied to, with applied and not-applied
   weighted equally, so 50 means "as much like your applications as not".

Nobody picks the weights. An answer that does not predict what you apply
to ends up near zero; one the model answers pessimistically on every
posting has that bias absorbed by the intercept. Refit whenever the
tracker grows, or set jev.fit.tracker to your Google Sheet and it refits
by itself before a run when you have added applications. Until a fit exists the answers are multiplied together,
which is known to squash every score toward zero.

What it learns is what you APPLY to, not what you would be hired for: the
tracker records no outcomes yet. If you start recording interviews, those
rows can be weighted separately.

    python fit.py --candidate               # the text the model reads about you
    python fit.py --learn tracker.csv       # match, answer, fit, report
    python fit.py --learn                   # same, from jev.fit.tracker
    python fit.py --help-sheets             # connect your Google Sheet
    python fit.py --check                   # stored postings, every factor
    python fit.py --write-config            # refresh jev.fit in config.yaml

Every question, band and label is a default that `jev.fit:` in config.yaml
overrides. The fitted weights live in fit_weights.yaml, which --learn
writes: they are data computed from your applications, not settings.
"""
import copy
import hashlib
import json
import math
import os
import random
import re
from datetime import date, datetime, timedelta

DEFAULTS = {
    # ---- career questions --------------------------------------------------
    "career_match": {
        "instructions":
            "Compare the duties this job posting describes with the work in "
            "the candidate record above it. Judge the duties, not the job "
            "title or the employer. Ignore degree requirements, years of "
            "experience and dates entirely; those are asked separately.",
        "bands": [
            "The duties are a different occupation from anything in the "
            "candidate record.",
            "The duties are adjacent to the record: related subject matter, "
            "but not work the candidate has done.",
            "Some of the central duties are things the candidate has done; "
            "the rest would be new.",
            "Most of the central duties are things the candidate has done.",
            "The central duties are the work the candidate record "
            "describes.",
        ],
        "labels": ["different occupation", "adjacent work", "partial match",
                   "mostly matches", "direct match"],
    },
    # Single-clause, and about the posting text alone.
    "needs_graduate_degree": {
        "instructions":
            "Does this posting REQUIRE a master's degree, PhD or other "
            "graduate degree?",
        "true": "Yes. A graduate degree is stated as required.",
        "false": "No. A bachelor's degree is enough, or a graduate degree is "
                 "only preferred.",
    },
    "needs_enrolment": {
        "instructions":
            "Does this posting require the applicant to be CURRENTLY "
            "enrolled as a student?",
        "true": "Yes. It is for current students: an internship, co-op or "
                "programme that requires active enrolment.",
        "false": "No. Graduates can apply.",
    },
    # Relative to the record: the remaining credentials.
    "has_credentials": {
        "instructions":
            "Leaving degree level aside, does the candidate record show the "
            "field of study, licence, certification, security clearance or "
            "language this posting requires?",
        "true": "Yes. The record shows what the posting requires, or the "
                "posting requires none of these.",
        "false": "No. It requires a degree in a different field, or a "
                 "licence, certification, clearance or language the record "
                 "does not show.",
    },
    "has_experience": {
        "instructions":
            "Does the candidate record show the experience this posting "
            "REQUIRES, counting experience the way this posting says to "
            "count it?",
        "true": "Yes. The role is entry-level or states no minimum, or the "
                "amount and kind of experience it asks for is covered by "
                "the record, including assistantships and internships where "
                "the posting counts them.",
        "false": "No. It asks for more years than the record shows, a "
                 "senior, lead or people-management level, or hands-on "
                 "experience in a field or tool the record does not "
                 "contain.",
    },
    # Which questions feed the career score, in order. "open_now" is the
    # existing jev.temporal question. Remove a name to stop asking it; add
    # any key of this block that has instructions/true/false. Changing this
    # list invalidates fit_weights.yaml until you run --learn again.
    "career_features": ["career_match", "needs_graduate_degree",
                        "needs_enrolment", "has_credentials",
                        "has_experience", "open_now"],

    # Questions where YES is a barrier. Only used before a fit exists, to
    # multiply (1 - p) instead of p; a fit learns each sign by itself.
    "barrier_features": ["needs_graduate_degree", "needs_enrolment"],

    # ---- bridge (not yet learned: the tracker has no bridge jobs) ---------
    "can_get": {
        "instructions":
            "Could the candidate in the record above be hired for this job "
            "as the record stands?",
        "true": "Yes. The posting asks for nothing the record lacks: no "
                "licence, certification, clearance or language, and no "
                "years of experience in this specific line of work. A "
                "degree the job does not need is not a barrier.",
        "false": "No. The posting requires at least one thing the record "
                 "does not show: a licence, certification, clearance, "
                 "language, or prior experience in this specific line of "
                 "work.",
    },

    # ---- inputs -------------------------------------------------------------
    "candidate_chars": 2600,
    "requirement_chars": 600,
    "weights_file": "fit_weights.yaml",

    # ---- your tracker -------------------------------------------------------
    # A CSV path, or the Google Sheets address of your tracker. A sheet is
    # read with a Google service account (see fit.py --help-sheets); the key
    # file comes from service_account_file, else the
    # GOOGLE_APPLICATION_CREDENTIALS environment variable. Keep the key
    # OUTSIDE this folder so it can never be committed to GitHub.
    "tracker": "",
    "tracker_tab": "Job Search",      # used when the address has no gid=
    "service_account_file": "",
    # on_run: refit before scoring whenever the tracker has new rows or the
    # questions changed. manual: only `python fit.py --learn`.
    "refit": "on_run",

    # ---- learning -----------------------------------------------------------
    # Postings you did not apply to, per application, taken from seen.db
    # around the dates you applied. Postings the digest showed you come
    # first, because passing on a job you saw says more than missing one
    # you never saw.
    "negatives_per_positive": 4,
    # L2 strengths tried; the one with the best cross-validated AUC is used.
    "regularization": [0.1, 1.0, 10.0],

    # ---- display only: never changes a score --------------------------------
    "review_band": 0.15,
    "review_min_score": 45,
}

# Old default text, so --write-config can upgrade it when you never edited it.
_OLD_HAS_CREDENTIALS = (
    "Does the candidate record show every credential this posting "
    "REQUIRES: the degree level, the degree field, any licence, "
    "certification, security clearance or language, and current "
    "student enrolment if the role is for students?")


def conf(cfg):
    """DEFAULTS merged with cfg['fit']. Dicts update, the rest replaces."""
    out = copy.deepcopy(DEFAULTS)
    for k, v in ((cfg or {}).get("fit") or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        elif v is not None:
            out[k] = v
    return out


def _question(f, cfg, name):
    if name == "open_now":
        return (cfg or {}).get("temporal") or {}
    return f.get(name) or {}


def validate(cfg):
    """Complaints about jev.fit / jev.mode, for preflight."""
    problems = []
    mode = (cfg or {}).get("mode", "resume")
    if mode not in ("resume", "bands"):
        problems.append(f"jev.mode is {mode!r}; use resume or bands")
    for k in ((cfg or {}).get("fit") or {}):
        if k not in DEFAULTS and k not in ("has_credentials",):
            problems.append(f"jev.fit.{k} is not a known key")
    f = conf(cfg)
    m = f["career_match"]
    if len(m.get("bands") or []) < 2:
        problems.append("jev.fit.career_match.bands needs at least 2 levels")
    if len(m.get("labels") or []) != len(m.get("bands") or []):
        problems.append("jev.fit.career_match.labels must match bands 1:1")
    for name in f["career_features"]:
        if name == "career_match":
            continue
        q = _question(f, cfg, name)
        for key in ("instructions", "true", "false"):
            if not str(q.get(key) or "").strip():
                problems.append(f"jev.fit.career_features: {name!r} has no "
                                f"{key}")
    if "career_match" not in f["career_features"]:
        problems.append("jev.fit.career_features must include career_match")
    return problems


# --------------------------------------------------------------------------
# The candidate, as the model sees it
# --------------------------------------------------------------------------
def candidate(cfg=None, facts_path="resume.yaml"):
    """
    The candidate text placed above every posting. `jev.candidate` in
    config.yaml, if you have written one, is used as is; otherwise it is
    built from resume.yaml (facts only, no voice rules).
    """
    own = str((cfg or {}).get("candidate") or "").strip()
    if own:
        return own
    try:
        import yaml
        f = yaml.safe_load(open(facts_path, encoding="utf-8")) or {}
    except Exception:
        return ""
    if not isinstance(f, dict):
        return ""
    ed = f.get("education") or {}
    sk = f.get("skills") or {}
    jobs = f.get("experience") or []
    years = [int(y) for e in jobs
             for y in re.findall(r"(?:19|20)\d{2}", str(e.get("dates", "")))]
    roles = "; ".join(
        f"{e.get('role', '')}, {e.get('org', '')} ({e.get('dates', '')})"
        for e in jobs)
    span = (f" between {min(years)} and {max(years)}" if years else "")
    course = ", ".join(str(c) for c in ed.get("coursework") or [])
    projects = "; ".join(
        f"{p.get('name', '')} ({p.get('type', '')})"
        for p in f.get("projects") or [])
    parts = [
        "CANDIDATE RECORD (complete; nothing outside it exists).",
        f"Education: {ed.get('degree', '')}, {ed.get('school', '')}, "
        f"graduated {ed.get('graduated', '')}. Not currently enrolled as a "
        f"student." + (f" Coursework: {course}." if course else ""),
        "The record lists no graduate degree, professional licence, "
        "certification, security clearance, or language other than English.",
        f"Experience: {len(jobs)} positions{span}, all research "
        f"assistantships, internships, fellowships or temporary roles held "
        f"during or just after the bachelor's degree: {roles}.",
        "Technical skills: " + "; ".join(map(str, sk.get("technical") or []))
        + ".",
        "Research skills: " + "; ".join(map(str, sk.get("research") or []))
        + ".",
        f"Projects: {projects}." if projects else "",
    ]
    limit = int(conf(cfg)["candidate_chars"])
    return " ".join(p for p in parts if p).strip()[:limit]


# --------------------------------------------------------------------------
# The posting, as the model sees it
# --------------------------------------------------------------------------
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?\s*[kK]?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s*[•·▪●]\s*|;\s+")
_REQ = re.compile(
    r"\b(?:\d{1,2}\s*\+?\s*(?:-|–|to)?\s*\d{0,2}\s*\+?\s*(?:years?|yrs?)|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\(\d+\)\s+)?"
    r"years?|ph\.?\s?d|doctora|master'?s|bachelor'?s|degree|licen[cs]|"
    r"certif|clearance|fluen|bilingual|native speaker|enrolled|"
    r"currently pursuing|cdl|minimum qualifications?|required)\b", re.I)


def stated_salary(job):
    """The posting's own salary field as "$lo-$hi per year|hour", or None."""
    s = str((job or {}).get("salary") or "").strip()
    low = s.lower()
    if not s or "estimate" in low or re.search(r"daily|day\b|week", low):
        return None
    vals = []
    for m in _NUM.finditer(s):
        raw = m.group(0).strip()
        try:
            v = float(raw.rstrip("kK").replace(",", "").strip())
        except ValueError:
            continue
        vals.append(v * 1000 if raw[-1:] in "kK" else v)
    vals = [v for v in vals if v > 0]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    hourly = bool(re.search(r"hour|hr\b|/h\b", low)) or hi < 1000
    fmt = ((lambda v: f"${v:,.2f}".replace(".00", "")) if hourly
           else (lambda v: f"${v:,.0f}"))
    body = fmt(lo) if lo == hi else f"{fmt(lo)}-{fmt(hi)}"
    return body + (" per hour" if hourly else " per year")


def requirement_lines(desc, already="", limit=600):
    """Requirement sentences appended to the slice the model reads."""
    seen = (already or "").lower()
    out, n = [], 0
    for s in (x.strip() for x in _SENT_SPLIT.split(desc or "")):
        if not (20 <= len(s) <= 320) or not _REQ.search(s):
            continue
        if s[:60].lower() in seen:
            continue
        out.append(s)
        n += len(s) + 3
        if n >= limit:
            break
    return " | ".join(out)[:limit]


def state(job, body, cand, cfg=None):
    """Candidate first (a shared prefix the server can cache), then the job."""
    f = conf(cfg)
    pay = stated_salary(job)
    req = requirement_lines(job.get("description") or "", body,
                            int(f["requirement_chars"]))
    posting = (f"JOB POSTING: {job.get('title', '')} at "
               f"{job.get('company', '')}, {job.get('location', '')}."
               + (f" Pay: {pay}." if pay else "")
               + f" {body}"
               + (f" Stated requirements: {req}" if req else ""))
    return (f"{cand}\n\n{posting}" if cand else posting), pay


def focus(job, cfg):
    """The same slice of the posting the pipeline scores."""
    chars = int((cfg or {}).get("desc_chars") or 2000)
    desc = job.get("description") or ""
    try:
        import score
        return score._focus_description(desc, chars)
    except Exception:
        return re.sub(r"\s+", " ", desc).strip()[:chars]


# --------------------------------------------------------------------------
# Asking the questions
# --------------------------------------------------------------------------
def _noul(q, today=""):
    return {"type": "noul",
            "instructions": q["instructions"].replace("{today}", today),
            "criteria": {"true": q["true"], "false": q["false"]}}


def _bands(res, n_bands):
    """(expected value 0-1, index of the likeliest band)."""
    probs = {int(k): float(v) for k, v in
             (res.get("probabilities") or {}).items()}
    if not probs:
        return 0.0, 0
    top = max(1, n_bands - 1)
    ev = sum(k * p for k, p in probs.items()) / top
    return max(0.0, min(1.0, ev)), max(probs, key=probs.get)


def questions_key(cfg, cand):
    """Hash of every question and the candidate text. Weights fitted under
    one set of questions are meaningless under another."""
    f = conf(cfg)
    blob = json.dumps({n: (f["career_match"] if n == "career_match"
                           else _question(f, cfg, n))
                       for n in f["career_features"]}, sort_keys=True)
    return hashlib.sha1((blob + cand).encode("utf-8")).hexdigest()[:12]


def career_features(model, job, body, cfg, cand, today=None):
    """({feature: value 0-1}, band label, text sent, pay)."""
    f = conf(cfg)
    today = today or date.today().isoformat()
    text, pay = state(job, body, cand, cfg)
    vals, label = {}, ""
    for name in f["career_features"]:
        if name == "career_match":
            m = f["career_match"]
            ev, best = _bands(model.decide(
                text, {"type": "score", "instructions": m["instructions"],
                       "criteria": list(m["bands"])}), len(m["bands"]))
            vals[name] = ev
            label = m["labels"][min(best, len(m["labels"]) - 1)]
        else:
            q = _question(f, cfg, name)
            vals[name] = float(model.decide(text, _noul(q, today))
                               .get("noul", 0.0))
    return vals, label, text, pay


# --------------------------------------------------------------------------
# Combining the answers
# --------------------------------------------------------------------------
def _logit(p):
    p = min(0.99, max(0.01, float(p)))
    return math.log(p / (1 - p))


_WEIGHTS = {}


def load_weights(cfg, track="career"):
    """The fitted block for a track from fit_weights.yaml, or None."""
    path = conf(cfg)["weights_file"]
    if path not in _WEIGHTS:
        try:
            import yaml
            _WEIGHTS[path] = yaml.safe_load(open(path, encoding="utf-8")) or {}
        except Exception:
            _WEIGHTS[path] = {}
    return (_WEIGHTS[path] or {}).get(track)


def combine(vals, w):
    """100 x sigmoid(intercept + sum of weight x logit(answer))."""
    z = float(w["intercept"]) + sum(
        float(w["weights"].get(k, 0.0)) * _logit(v) for k, v in vals.items())
    return 100.0 / (1.0 + math.exp(-z)), z


_WARNED = set()


def score_one(model, job, body, cfg, track="career", cand="", today=None):
    """
    Score one posting. Returns the dict jev.score_track stores:
    score, review, reason, detail, explain.
    """
    f = conf(cfg)
    today = today or date.today().isoformat()
    if track != "career":
        return _score_bridge(model, job, body, cfg, cand, today)

    vals, label, _, pay = career_features(model, job, body, cfg, cand, today)
    w = load_weights(cfg, "career")
    key = questions_key(cfg, cand)
    if w and w.get("questions_key") != key:
        if "stale" not in _WARNED:
            print("  ! fit: the questions or your record changed since "
                  "fit_weights.yaml was fitted; using the unfitted product "
                  "until you run `python fit.py --learn`")
            _WARNED.add("stale")
        w = None

    if w:
        score, z = combine(vals, w)
        method = (f"fitted on {w.get('positives')} applications, "
                  f"cv AUC {w.get('cv_auc')}")
        parts = [(float(w["weights"].get(k, 0.0)) * _logit(v), k, v)
                 for k, v in vals.items()]
        working = (f"intercept {float(w['intercept']):+.2f} "
                   + " ".join(f"{c:+.2f} {k}={v:.2f}" for c, k, v
                              in sorted(parts, key=lambda t: -abs(t[0])))
                   + f" -> {score:.0f}")
    else:
        score = 100.0
        inv = set(f["barrier_features"])
        for k, v in vals.items():
            score *= (1.0 - v) if k in inv else v
        method = "unfitted product (run fit.py --learn)"
        working = " x ".join(f"{k} {v:.2f}" for k, v in vals.items()) + \
            f" = {score:.0f}"
    score = round(max(0.0, min(100.0, score)))

    band = float(f["review_band"])
    review = (score >= int(f["review_min_score"])
              and any(abs(v - 0.5) <= band for k, v in vals.items()
                      if k != "career_match"))
    short = {"needs_graduate_degree": "grad degree needed",
             "needs_enrolment": "enrolment needed",
             "has_credentials": "credentials", "has_experience": "experience",
             "open_now": "open now"}
    bits = [label] + [f"{short.get(k, k)} {v:.0%}" for k, v in vals.items()
                      if k != "career_match"]
    if pay:
        bits.append(pay)
    detail = f"{working}\n{method}" + (
        "\nFlagged for review: one answer is close to a coin flip."
        if review else "")
    explain = {"track": "career", "mode": "resume", "band": label,
               "final": score, "questions": {k: round(v, 3)
                                            for k, v in vals.items()},
               "adjustments": [], "review": bool(review), "pay": pay,
               "text": detail}
    return {"score": score, "review": bool(review),
            "reason": " · ".join(bits), "detail": detail, "explain": explain}


def _score_bridge(model, job, body, cfg, cand, today):
    """Bridge: contact x can-get x no-fee x open. Not learned yet."""
    f = conf(cfg)
    text, pay = state(job, body, cand, cfg)
    c = cfg["contact"]
    base, best = _bands(model.decide(
        text, {"type": "score", "instructions": c["instructions"],
               "criteria": list(c["bands"])}), len(c["bands"]))
    label = c["labels"][min(best, len(c["labels"]) - 1)]
    fee = float(model.decide(text, _noul(cfg["fee"])).get("noul", 0.0))
    factors = [
        ("can get it", float(model.decide(text, _noul(f["can_get"]))
                             .get("noul", 0.0))),
        ("asks for no money", 1.0 - fee),
        ("open now", float(model.decide(text, _noul(cfg["temporal"], today))
                           .get("noul", 1.0))),
    ]
    product = base
    for _, p in factors:
        product *= p
    score = round(max(0.0, min(100.0, product * 100.0)))
    review = (score >= int(f["review_min_score"])
              and any(abs(p - 0.5) <= float(f["review_band"])
                      for _, p in factors))
    import jev
    bits = [label, pay or jev.extract_pay(text) or "no pay stated"]
    shift = jev.extract_shift(body)
    if shift:
        bits.append(shift)
    bits += [f"{n} {p:.0%}" for n, p in factors]
    detail = " x ".join([f"contact {base:.2f} ({label})"]
                        + [f"{n} {p:.2f}" for n, p in factors]) + \
        f" = {score}"
    return {"score": score, "review": bool(review),
            "reason": " · ".join(bits), "detail": detail,
            "explain": {"track": "bridge", "mode": "resume", "band": label,
                        "final": score, "review": bool(review), "pay": pay,
                        "questions": {n: round(p, 3) for n, p in factors},
                        "adjustments": [], "text": detail}}


# --------------------------------------------------------------------------
# Logistic regression, plain Python (five or six features, a few hundred
# rows: no numpy needed)
# --------------------------------------------------------------------------
def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
            for i in range(n)]


def fit_logistic(X, y, lam=1.0, iters=50):
    """
    L2-regularised logistic regression by Newton's method, classes weighted
    equally. X rows are feature lists (no intercept column). Returns
    [intercept, w1, ...].
    """
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    sw = [len(y) / (2.0 * (n_pos if t else n_neg)) for t in y]
    d = len(X[0]) + 1
    beta = [0.0] * d
    for _ in range(iters):
        g = [0.0] * d
        H = [[0.0] * d for _ in range(d)]
        for xi, yi, si in zip(X, y, sw):
            row = [1.0] + list(xi)
            z = sum(b * x for b, x in zip(beta, row))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            for a in range(d):
                g[a] += si * (p - yi) * row[a]
                for c in range(d):
                    H[a][c] += si * p * (1 - p) * row[a] * row[c]
        for a in range(1, d):
            g[a] += lam * beta[a]
            H[a][a] += lam
        step = _solve(H, g)
        beta = [b - s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-6:
            break
    return beta


def auc(scores, y):
    """Probability a random application outranks a random non-application."""
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cross_val(X, y, lam, k=5, seed=0):
    """Out-of-fold predictions, stratified."""
    rnd = random.Random(seed)
    pos = [i for i, t in enumerate(y) if t]
    neg = [i for i, t in enumerate(y) if not t]
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    folds = [pos[i::k] + neg[i::k] for i in range(k)]
    pred = [0.0] * len(y)
    for f in folds:
        test = set(f)
        tr = [i for i in range(len(y)) if i not in test]
        beta = fit_logistic([X[i] for i in tr], [y[i] for i in tr], lam)
        for i in f:
            pred[i] = sum(b * x for b, x in zip(beta, [1.0] + X[i]))
    return pred


# --------------------------------------------------------------------------
# Matching your tracker to seen.db
# --------------------------------------------------------------------------
_SUFFIX = re.compile(r"\b(?:llc|inc|corp|corporation|co|ltd|lp|llp|the|"
                     r"company|technologies|technology|labs?|group)\b")


def _norm(s):
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", _SUFFIX.sub(" ", s)).strip()


def _ids(s):
    s = (s or "").lower()
    return set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
                          r"[0-9a-f]{12}", s)) | set(re.findall(r"\d{6,}", s))


def _company_match(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b or a.replace(" ", "") == b.replace(" ", ""):
        return True
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return True
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.5


def _parse_day(s):
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.split()[0][:10], fmt).date()
        except ValueError:
            continue
    return None


def parse_tracker(rows, where="tracker"):
    """[{title, company, url, applied, interview}] from the sheet's rows."""
    hdr = next((i for i, r in enumerate(rows) if "Position Title" in
                [str(c).strip() for c in r]), None)
    if hdr is None:
        raise ValueError(f"{where}: no 'Position Title' header row")
    H = [str(h).strip() for h in rows[hdr]]
    col = {h: H.index(h) for h in H}

    def get(r, name):
        i = col.get(name)
        return (str(r[i]) if i is not None and i < len(r) else "").strip()

    out = []
    for r in rows[hdr + 1:]:
        title = get(r, "Position Title")
        if not title or title.lower().startswith("e.g."):
            continue
        out.append({"title": " ".join(title.split()),
                    "company": " ".join(get(r, "Company").split()),
                    "url": get(r, "Link"),
                    "applied": _parse_day(get(r, "Date Applied")),
                    "interview": bool(get(r, "Interview Scheduled"))})
    return out


def read_tracker(path):
    import csv
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    return parse_tracker(rows, path)


SHEETS_HELP = """\
Reading your tracker from Google Sheets, privately

 1. pip install gspread
 2. https://console.cloud.google.com : create a project (any name), then
    APIs & Services > Library > enable "Google Sheets API".
 3. APIs & Services > Credentials > Create credentials > Service account.
    Any name; skip the optional steps. Open it > Keys > Add key > JSON.
    A .json file downloads.
 4. Move that file OUTSIDE the jobscan folder, for example
      C:\\Users\\Kai\\.jobscan\\google-service-account.json
    It is a password. Inside the folder it could be committed to GitHub.
 5. Open the .json, copy "client_email" (...@...iam.gserviceaccount.com).
    In your sheet: Share > paste that address > Viewer > untick Notify.
 6. In config.yaml under jev: fit:
      tracker: "https://docs.google.com/spreadsheets/d/<id>/edit?gid=<gid>"
      service_account_file: "C:/Users/Kai/.jobscan/google-service-account.json"
    or put GOOGLE_APPLICATION_CREDENTIALS=<that path> in .env instead.
 7. python fit.py --learn          (reads the sheet, fits, saves)
"""


def _sheet_rows(address, f):
    """All cell values of one tab, through a service account."""
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", address)
    if not m:
        raise ValueError(f"not a Google Sheets address: {address}")
    gid = re.search(r"[#&?]gid=(\d+)", address)
    try:
        import gspread
    except ImportError:
        raise RuntimeError("pip install gspread (see python fit.py "
                           "--help-sheets)")
    key = (f.get("service_account_file")
           or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "")
    if not key or not os.path.exists(key):
        raise RuntimeError("no service account key: set jev.fit."
                           "service_account_file or GOOGLE_APPLICATION_"
                           "CREDENTIALS (see python fit.py --help-sheets)")
    sh = gspread.service_account(filename=key).open_by_key(m.group(1))
    if gid:
        ws = next((w for w in sh.worksheets()
                   if str(w.id) == gid.group(1)), None)
        if ws is None:
            raise RuntimeError(f"the sheet has no tab with gid={gid.group(1)}")
    else:
        ws = sh.worksheet(f.get("tracker_tab") or "Job Search")
    return ws.get_all_values()


def load_tracker(source, cfg):
    """Applications from a CSV path or a Google Sheets address."""
    f = conf(cfg)
    source = source or f.get("tracker") or ""
    if not source:
        raise ValueError("no tracker: pass a CSV path, or set "
                         "jev.fit.tracker in config.yaml")
    if source.startswith("http"):
        return parse_tracker(_sheet_rows(source, f), "Google Sheet"), "sheet"
    return read_tracker(source), os.path.basename(source)


def tracker_hash(apps):
    blob = "\n".join(sorted(
        f"{a['title']}|{a['company']}|{a['url']}|{a['applied']}|"
        f"{a['interview']}" for a in apps))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def match_tracker(apps, jobs):
    """{app index: job} by URL id, else company + similar title."""
    import difflib
    by_id = {}
    for j in jobs:
        for i in _ids(j.get("url")) | _ids(j.get("uid")):
            by_id.setdefault(i, j)
    found = {}
    for n, a in enumerate(apps):
        hit = next((by_id[i] for i in _ids(a["url"]) if i in by_id), None)
        if hit is None:
            best, score = None, 0.0
            ta = _norm(a["title"])
            for j in jobs:
                if not _company_match(a["company"], j.get("company")):
                    continue
                r = difflib.SequenceMatcher(None, ta,
                                            _norm(j.get("title"))).ratio()
                if r > score:
                    best, score = j, r
            hit = best if score >= 0.72 else None
        if hit is not None:
            found[n] = hit
    return found


# --------------------------------------------------------------------------
# --learn
# --------------------------------------------------------------------------
def _load_model(root, cfg, track="career"):
    import sys
    import jev
    import llm
    profile = llm.resolve(root, "jev", track)
    print(f"model: {llm.describe(profile)}")
    if not jev.ensure_server(cfg, profile)[0]:
        sys.exit("jev model unavailable")
    return jev._model(cfg, profile)


def _db_jobs(db):
    import sqlite3
    conn = None
    if db is None:
        try:
            import store
            conn = store.connect()
        except Exception:
            conn = None
    conn = conn or sqlite3.connect(db or "seen.db")
    cols = ("uid", "title", "company", "location", "url", "description",
            "first_seen", "tier", "score", "salary", "status")
    have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    sel = ", ".join(c if c in have else f"NULL AS {c}" for c in cols)
    return [dict(zip(cols, r)) for r in conn.execute(
        f"SELECT {sel} FROM jobs WHERE (track = 'career' OR track IS NULL) "
        f"AND length(description) > 400")]


class FitError(Exception):
    pass


def fit_tracker(apps, cfg, cand, model_fn, db=None, say=print):
    """
    Match, ask, fit. Returns (weights block, report dict). Raises FitError
    when there is too little to learn from. `model_fn` is called only if
    some posting has no cached answers.
    """
    f = conf(cfg)
    jobs = _db_jobs(db)
    found = match_tracker(apps, jobs)
    pos = {}
    for n, j in found.items():
        pos.setdefault(j["uid"], j)
    for j in jobs:                       # marked applied in ui.py
        if (j.get("status") or "") == "applied":
            pos.setdefault(j["uid"], j)
    missing = [a for n, a in enumerate(apps) if n not in found]
    say(f"matched {len(found)} of {len(apps)} applications to a stored "
        f"posting ({len(pos)} positives in all)")
    if len(pos) < 15:
        raise FitError(f"only {len(pos)} applications match a stored "
                       f"posting; at least 15 are needed")

    days = [a["applied"] for a in apps if a["applied"]]
    lo = min(days) - timedelta(days=21) if days else date.min
    hi = max(days) + timedelta(days=3) if days else date.max
    pos_keys = {(_norm(j["company"]), _norm(j["title"])) for j in pos.values()}

    def eligible(j):
        d = _parse_day(str(j.get("first_seen") or "")[:10])
        return (j["uid"] not in pos and d is not None and lo <= d <= hi
                and (_norm(j["company"]), _norm(j["title"])) not in pos_keys)

    rnd = random.Random(0)
    pool = [j for j in jobs if eligible(j)]
    shown = [j for j in pool if j.get("tier") in ("strong", "review", "maybe")]
    rest = [j for j in pool if j.get("tier") not in ("strong", "review",
                                                     "maybe")]
    rnd.shuffle(shown)
    rnd.shuffle(rest)
    want = int(f["negatives_per_positive"]) * len(pos)
    neg = (shown + rest)[:want]
    say(f"{len(neg)} postings you did not apply to, from {lo} to {hi} "
        f"({min(len(shown), want)} of them shown in a digest)")

    key = questions_key(cfg, cand)
    cache_path = "fit_cache.json"
    try:
        cache = json.load(open(cache_path, encoding="utf-8"))
    except Exception:
        cache = {}
    if cache.get("key") != key:
        cache = {"key": key, "rows": {}}
    todo = [j for j in list(pos.values()) + neg
            if j["uid"] not in cache["rows"]]
    if todo:
        say(f"asking the model about {len(todo)} posting(s); about "
            f"{len(todo) * 3 // 60 + 1} minute(s)")
        model = model_fn()
        for i, j in enumerate(todo, 1):
            vals, _, _, _ = career_features(model, j, focus(j, cfg), cfg, cand)
            cache["rows"][j["uid"]] = vals
            if i % 25 == 0:
                say(f"  {i}/{len(todo)}")
                json.dump(cache, open(cache_path, "w", encoding="utf-8"))
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))

    names = list(f["career_features"])
    rows = [(cache["rows"][j["uid"]], 1, j) for j in pos.values()] + \
           [(cache["rows"][j["uid"]], 0, j) for j in neg]
    X = [[_logit(v[n]) for n in names] for v, _, _ in rows]
    y = [t for _, t, _ in rows]

    tries = []
    for lam in f["regularization"]:
        tries.append((float(lam), auc(cross_val(X, y, float(lam)), y)))
    lam, cv = max(tries, key=lambda t: t[1])
    beta = fit_logistic(X, y, lam)
    inv = set(f["barrier_features"])
    prod = [math.prod((1 - v[n]) if n in inv else v[n] for n in names)
            for v, _, _ in rows]
    stored = [(j.get("score") if j.get("score") is not None else -1)
              for _, _, j in rows]
    w = {"intercept": round(beta[0], 4),
         "weights": {n: round(b, 4) for n, b in zip(names, beta[1:])},
         "regularization": lam, "cv_auc": round(cv, 3),
         "product_auc": round(auc(prod, y), 3),
         "stored_score_auc": round(auc(stored, y), 3),
         "positives": len(pos), "negatives": len(neg),
         "fitted": date.today().isoformat(), "questions_key": key,
         "tracker_hash": tracker_hash(apps)}
    ranked = sorted(((sum(b * x for b, x in zip(beta, [1.0] + xi)), j, t)
                     for xi, t, (_, _, j) in zip(X, y, rows)),
                    key=lambda r: -r[0])
    return w, {"tries": tries, "missing": missing, "ranked": ranked,
               "names": names, "beta": beta}


def save_weights(cfg, w, source):
    """Write fit_weights.yaml, keeping the previous fit beside it."""
    import shutil
    import yaml
    path = conf(cfg)["weights_file"]
    if os.path.exists(path):
        shutil.copyfile(path, path.replace(".yaml", "") + ".prev.yaml")
    w = dict(w, tracker=source)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Written by fit.py from your applications. Refit rather "
                 "than editing.\n# The previous fit is kept as "
                 "fit_weights.prev.yaml.\n")
        yaml.safe_dump({"career": w}, fh, sort_keys=False)
    _WEIGHTS.clear()


def maybe_refit(cfg, cand, model, say=print):
    """
    Called by jev.score_track before a career run. Refits when the tracker
    has changed or the questions have, and never stops the run: any problem
    is printed and the existing weights (or the product) are used.
    """
    f = conf(cfg)
    if f.get("refit") != "on_run" or not f.get("tracker"):
        return
    try:
        apps, source = load_tracker(None, cfg)
    except Exception as e:
        say(f"  ! fit: could not read the tracker ({e}); keeping the "
            f"current weights")
        return
    old = load_weights(cfg) or {}
    if (old.get("tracker_hash") == tracker_hash(apps)
            and old.get("questions_key") == questions_key(cfg, cand)):
        return
    say(f"  - fit: tracker or questions changed; refitting from "
        f"{len(apps)} applications")
    try:
        w, _ = fit_tracker(apps, cfg, cand, lambda: model,
                           say=lambda m: say(f"    {m}"))
    except FitError as e:
        say(f"  ! fit: {e}; keeping the current weights")
        return
    save_weights(cfg, w, source)
    was = old.get("cv_auc")
    say(f"  - fit: refitted, AUC {w['cv_auc']}"
        + (f" (was {was})" if was is not None else "")
        + "; previous fit kept as fit_weights.prev.yaml")


def learn(args, root, cfg):
    import sys
    cand = candidate(cfg, args.facts)
    if not cand:
        sys.exit(f"{args.facts} unreadable; nothing to describe you with")
    try:
        apps, source = load_tracker(args.learn, cfg)
    except (ValueError, RuntimeError, OSError) as e:
        sys.exit(str(e))
    print(f"{len(apps)} applications in {source}")
    try:
        w, r = fit_tracker(apps, cfg, cand,
                           lambda: _load_model(root, cfg), db=args.db)
    except FitError as e:
        for a in r_missing(apps, args.db):
            print(f"  not in seen.db: {a['title'][:50]} - {a['company'][:28]}")
        sys.exit(f"\n{e}. Paste the missing postings in ui.py, or keep "
                 f"applying through jobscan's own digest.")
    for a in r["missing"]:
        print(f"  not in seen.db: {a['title'][:50]} - {a['company'][:28]}")
    print()
    for lam, a in r["tries"]:
        print(f"  regularization {lam:<5} cross-validated AUC {a:.3f}")
    print("\nHow well each ranks your applications above the rest "
          "(AUC: 0.5 is chance, 1.0 is perfect)")
    print(f"  fitted weights, cross-validated   {w['cv_auc']:.3f}")
    print(f"  answers multiplied (no fit)       {w['product_auc']:.3f}")
    print(f"  scores stored in seen.db          {w['stored_score_auc']:.3f}"
          f"   (old scorers, mixed)")
    print("\nWeights (each multiplies the log-odds of that answer):")
    print(f"  intercept {w['intercept']:+.2f}")
    for n, b in w["weights"].items():
        print(f"  {n:<24} {b:+.2f}")
    print("  positive = more of this answer looks more like your "
          "applications; near 0 = it does not predict them")
    print("\nThe postings it ranks highest that you did NOT apply to:")
    for z, j, t in [x for x in r["ranked"] if not x[2]][:8]:
        print(f"  {100 / (1 + math.exp(-z)):>3.0f}  {j['title'][:52]} "
              f"({(j['company'] or '')[:20]})")
    print("And the applications it ranks lowest:")
    for z, j, t in [x for x in r["ranked"] if x[2]][-5:]:
        print(f"  {100 / (1 + math.exp(-z)):>3.0f}  {j['title'][:52]} "
              f"({(j['company'] or '')[:20]})")
    if args.dry_run:
        print("\n--dry-run: fit_weights.yaml not written")
        return
    save_weights(cfg, w, source)
    print(f"\nwrote {conf(cfg)['weights_file']}. Rescore with:\n"
          f"  python score.py --backfill --rescore --scorer jev --limit 5000")


def r_missing(apps, db):
    try:
        return [a for n, a in enumerate(apps)
                if n not in match_tracker(apps, _db_jobs(db))]
    except Exception:
        return []


# --------------------------------------------------------------------------
# --write-config: refresh the jev.fit block in place
# --------------------------------------------------------------------------
def write_config(path):
    import shutil
    import yaml
    text = open(path, encoding="utf-8").read()
    root = yaml.safe_load(text) or {}
    user = ((root.get("jev") or {}).get("fit") or {})
    if (user.get("has_credentials") or {}).get("instructions") == \
            _OLD_HAS_CREDENTIALS:
        user.pop("has_credentials")
    merged = conf({"fit": user})
    block = "  fit:\n" + "".join(
        "    " + line + "\n" for line in yaml.safe_dump(
            merged, sort_keys=False, width=76,
            allow_unicode=True).splitlines())
    lines = text.splitlines(keepends=True)
    try:
        j0 = next(i for i, l in enumerate(lines) if l.rstrip() == "jev:")
        s = next(i for i in range(j0 + 1, len(lines))
                 if lines[i].rstrip() == "  fit:")
    except StopIteration:
        raise SystemExit(f"{path}: no jev.fit block. Run apply_fixes.py "
                         f"--write first.")
    e = s + 1
    while e < len(lines) and (not lines[e].strip()
                              or len(lines[e]) - len(lines[e].lstrip()) >= 4):
        e += 1
    new = "".join(lines[:s]) + block + "".join(lines[e:])
    yaml.safe_load(new)
    shutil.copyfile(path, path + ".bak-fit")
    open(path, "w", encoding="utf-8").write(new)
    print(f"updated jev.fit in {path} (previous copy: {path}.bak-fit)")


# --------------------------------------------------------------------------
# One employer, one title, many cities
# --------------------------------------------------------------------------
def collapse_duplicates(jobs):
    """(representatives, [(duplicate, representative), ...])."""
    first, reps, pairs = {}, [], []
    for j in jobs:
        co = _norm(j.get("company"))
        key = (j.get("track") or "career", co, _norm(j.get("title")))
        if co and key in first:
            pairs.append((j, first[key]))
        else:
            first[key] = j
            reps.append(j)
    return reps, pairs


def copy_to_duplicates(pairs):
    """Duplicates take the representative's score in the 'weak' tier."""
    count = {}
    for dup, rep in pairs:
        count[id(rep)] = (rep, count.get(id(rep), (rep, 0))[1] + 1)
        for k in ("score", "scored_at", "scorer", "scorer_model",
                  "score_detail"):
            if k in rep:
                dup[k] = rep[k]
        dup["tier"] = "weak"
        dup["reason"] = (f"duplicate of the {rep.get('location') or 'other'}"
                         f" posting")
    for rep, n in count.values():
        rep["reason"] = ((rep.get("reason") or "")
                         + f" · +{n} more location{'s' if n > 1 else ''}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
CONTROL = {
    "title": "Line Cook", "company": "Control Posting", "location": "",
    "description":
        "Prepare menu items to order on the grill and fry stations during "
        "dinner service. Keep the line stocked, follow food-safety "
        "procedures, and clean the station at close. Two years of "
        "restaurant kitchen experience and a food handler certificate are "
        "required.",
}


def check(args, root, cfg):
    import sqlite3
    import sys
    cand = candidate(cfg, args.facts)
    conn = sqlite3.connect(args.db)
    cols = "uid, title, company, location, description, score"
    if args.uid:
        rows = [r for u in args.uid for r in conn.execute(
            f"SELECT {cols} FROM jobs WHERE uid LIKE ?", (f"%{u}%",))]
    else:
        rows = list(conn.execute(
            f"SELECT {cols} FROM jobs WHERE track = ? AND score IS NOT NULL "
            f"AND length(description) > 400 ORDER BY first_seen DESC "
            f"LIMIT ?", (args.track, args.limit)))
    if not rows:
        sys.exit("no matching postings in seen.db")
    model = _load_model(root, cfg, args.track)
    print()
    for uid, title, company, loc, desc, stored in rows + [
            (None, CONTROL["title"], CONTROL["company"], "",
             CONTROL["description"], "ctl")]:
        job = {"title": title, "company": company, "location": loc,
               "description": desc}
        r = score_one(model, job, focus(job, cfg), cfg, args.track, cand)
        print(f"{r['score']:>4} (stored {stored})  {title[:60]}")
        print(f"      {r['reason'][:100]}")
        print(f"      {r['detail'].splitlines()[0][:100]}")
    print("\nThe last row is the control, a line-cook posting: it should "
          "score near zero.")


def main():
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description="Resume-aware jev scoring.")
    ap.add_argument("--candidate", action="store_true",
                    help="print the candidate text the model is shown")
    ap.add_argument("--learn", metavar="CSV_OR_SHEET", nargs="?", const="",
                    help="learn weights from your tracker: a CSV path, a "
                         "Google Sheets address, or nothing to use "
                         "jev.fit.tracker")
    ap.add_argument("--help-sheets", action="store_true",
                    help="how to connect a private Google Sheet")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --learn: report, but do not write weights")
    ap.add_argument("--check", action="store_true",
                    help="score stored postings and print the working")
    ap.add_argument("--write-config", action="store_true",
                    help="refresh the jev.fit block in config.yaml")
    ap.add_argument("--track", default="career", choices=("career", "bridge"))
    ap.add_argument("--uid", action="append", default=[])
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--db", default="seen.db")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--facts", default="resume.yaml")
    args = ap.parse_args()

    if args.help_sheets:
        return print(SHEETS_HELP)
    if args.write_config:
        return write_config(args.config)
    import jev
    root = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    cfg = jev.merged(root.get("jev"))
    for p in validate(cfg):
        print(f"! {p}")
    if args.learn is not None:
        return learn(args, root, cfg)
    if args.check:
        return check(args, root, cfg)
    cand = candidate(cfg, args.facts)
    print(cand or "(empty: resume.yaml not readable)")
    print(f"\n{len(cand)} characters")


if __name__ == "__main__":
    main()
