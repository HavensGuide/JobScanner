"""
Fetchers for each job source. Every fetcher returns a list of dicts with the
same shape, so main.py doesn't need to know where a job came from.

Slug-free by design. Greenhouse/Lever/Ashby have been removed: they required
maintaining a hand-curated list of company tokens, and a renamed or migrated
board 404s silently.

What that costs, and how this file compensates:

  - Adzuna returns EXCERPTS, not full descriptions. Its `description` field is
    truncated by design. The scorer therefore sees a fragment for these, and
    tailor.py cannot draft from them at all. Flagged per-job via desc_quality.
  - Nonprofits and think tanks (Mathematica, Urban, MDRC, Vera, RAND) were
    only reachable through the ATS layer. `adzuna.employers` partially
    recovers them with targeted phrase queries — see fetch_adzuna.
  - USAJOBS was the most underused source in the original. It now pages, uses
    occupational series codes instead of keyword guessing, filters to
    publicly-eligible announcements, and assembles a real description.

Canonical job dict:
    {
      "uid":      stable unique id, "<source>:<native_id>"
      "source":   "adzuna" | "usajobs" | "nyc"
      "title":    str
      "company":  str
      "location": str
      "url":      str
      "posted":   ISO date string or ""
      "salary":   str  (may be "")
      "description":  str (may be truncated)
      "desc_quality": "full" | "excerpt" | "none"
    }
"""

import html
import os
import re
import time

import requests

TIMEOUT = 30
UA = "jobscan/1.0 (personal job search script)"


def _get(url, **kwargs):
    """GET with retry on transient failures. Returns parsed JSON or None."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=TIMEOUT, **kwargs)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                print(f"  ! failed {url[:70]}: {type(e).__name__}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


# Descriptions are no longer sliced at 2k/4k — that cut off the duties and
# left only boilerplate. But seen.db is committed back to the repo by the
# GitHub Action, and an uncapped sweep writes a fresh multi-megabyte binary
# blob into git history twice a day. 40k characters is far above any real
# posting (the longest federal announcements run ~25k) while still bounding
# the file, and _quality() is computed on the text before this trims it.
MAX_DESC = 40000


def _clean(s):
    """
    Strip HTML tags and entities; collapse whitespace.

    Unescape BEFORE stripping tags. Some feeds entity-encode their markup
    (&lt;p&gt;), so stripping first would leave the literal tags behind. Two
    passes because double-encoding (&amp;nbsp;) shows up in the wild.
    """
    if not s:
        return ""
    s = html.unescape(html.unescape(str(s)))
    s = re.sub(r"<br\s*/?>|</p>|</li>|</div>|</h\d>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()[:MAX_DESC]


def _quality(desc):
    """
    How much the scorer can trust this text.

    Adzuna excerpts run a few hundred characters and usually end mid-sentence
    with an ellipsis. Treating those as full descriptions makes the model
    confidently score a fragment.
    """
    d = (desc or "").strip()
    if len(d) < 40:
        return "none"
    if len(d) < 400 or d.endswith(("\u2026", "...")):
        return "excerpt"
    return "full"


# ---------------------------------------------------------------------------
# Adzuna
# ---------------------------------------------------------------------------
def _adzuna_query(country, params_base, what, phrase=False):
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = dict(params_base)
    if phrase:
        # what_phrase matches the exact phrase, which is what you want when the
        # query IS an employer name. Plain `what` would match any of the words.
        params["what_phrase"] = what
    else:
        params["what"] = what
    data = _get(url, params=params, headers={"User-Agent": UA})
    if not data:
        return []

    out = []
    for j in data.get("results", []):
        salary = ""
        lo, hi = j.get("salary_min"), j.get("salary_max")
        if lo and hi:
            salary = f"${int(lo):,}-${int(hi):,}"
            # Adzuna models a salary when the posting doesn't state one.
            # Never present a predicted number as the employer's.
            if str(j.get("salary_is_predicted")) == "1":
                salary += " (Adzuna estimate, not stated)"
        desc = _clean(j.get("description"))
        out.append({
            "uid": f"adzuna:{j.get('id')}",
            "source": "adzuna",
            "title": (j.get("title") or "").strip(),
            "company": (j.get("company") or {}).get("display_name", ""),
            "location": (j.get("location") or {}).get("display_name", ""),
            "url": j.get("redirect_url", ""),
            "posted": (j.get("created") or "")[:10],
            "salary": salary,
            "description": desc[:2000],
            "desc_quality": _quality(desc),
        })
    return out


def fetch_adzuna(cfg):
    """
    Two query types:

      queries:   broad role searches ("research analyst")
      employers: exact-phrase employer searches ("Urban Institute")

    `employers` is the partial replacement for the ATS layer. It is not as
    good — Adzuna only has the employer if some board it aggregates carried
    the posting, and descriptions are still excerpts — but it needs no slug
    and it fails loudly instead of silently. A targeted employer query that
    returns nothing is reported, because that is real information: either the
    employer isn't hiring or Adzuna doesn't index them.

    Quota math: one call per query per run. 2 runs/day x 22 weekdays = 44
    calls per query per month against a ~1,000/month free tier. Roughly 20
    queries total is the safe ceiling.
    """
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        print("  - adzuna: no credentials, skipping")
        return []

    country = cfg.get("country", "us")
    base = {
        "app_id": app_id,
        "app_key": app_key,
        "where": cfg.get("location", ""),
        "max_days_old": cfg.get("max_days_old", 3),
        "results_per_page": cfg.get("results_per_page", 30),
        "content-type": "application/json",
    }

    out = []
    for q in cfg.get("queries", []):
        out += _adzuna_query(country, base, q)
        time.sleep(1)

    # Employer-targeted. Wider date window and no location filter: if a named
    # target employer is hiring, you want to know regardless.
    emp_base = dict(base)
    emp_base["max_days_old"] = cfg.get("employer_max_days_old", 14)
    emp_base["where"] = ""

    empty = []
    for name in cfg.get("employers", []):
        got = _adzuna_query(country, emp_base, name, phrase=True)
        if not got:
            empty.append(name)
        out += got
        time.sleep(1)

    if empty:
        print(f"  - adzuna: no current listings for {', '.join(empty)}")
    return out


# ---------------------------------------------------------------------------
# USAJOBS
# ---------------------------------------------------------------------------
# Occupational series are far more precise than keyword search: series 0110 is
# Economist, and filtering on it returns roughly what a person would call an
# economist job. Verify any code before adding it:
#
#   python sources.py --series economist
#
USAJOBS_SEARCH = "https://data.usajobs.gov/api/search"


def _usajobs_headers():
    email = os.environ.get("USAJOBS_EMAIL")
    key = os.environ.get("USAJOBS_API_KEY")
    if not (email and key):
        return None
    return {"Host": "data.usajobs.gov", "User-Agent": email.strip(),
            "Authorization-Key": key.strip()}


def _usajobs_collect(headers, params, max_pages):
    """Page through one query. ResultsPerPage caps at 500."""
    items, page = [], 1
    while page <= max_pages:
        p = dict(params)
        p["Page"] = page
        data = _get(USAJOBS_SEARCH, params=p, headers=headers)
        if not data:
            break
        result = data.get("SearchResult", {})
        batch = result.get("SearchResultItems", []) or []
        items += batch
        try:
            count = int(result.get("SearchResultCountAll") or 0)
        except (TypeError, ValueError):
            count = 0
        if len(batch) < int(params.get("ResultsPerPage", 50)) or len(items) >= count:
            break
        page += 1
        time.sleep(1)
    return items


def _usajobs_description(details):
    """
    Assemble a real description instead of just the summary.

    JobSummary alone is often two sentences of boilerplate. MajorDuties and
    QualificationSummary are where the actual requirements live, which is
    exactly what the scorer needs to judge level fit.
    """
    duties = details.get("MajorDuties")
    if isinstance(duties, list):
        duties = "\n".join(str(d) for d in duties)
    parts = [details.get("JobSummary"), duties,
             details.get("QualificationSummary")]
    return _clean("\n\n".join(str(p) for p in parts if p))


def _first(seq, key):
    try:
        return (seq or [{}])[0].get(key, "") or ""
    except (AttributeError, IndexError):
        return ""


def fetch_usajobs(cfg):
    headers = _usajobs_headers()
    if not headers:
        print("  - usajobs: no credentials, skipping")
        return []

    per_page = cfg.get("results_per_page", 100)
    max_pages = cfg.get("max_pages", 3)

    base = {"ResultsPerPage": per_page}
    if cfg.get("location"):
        base["LocationName"] = cfg["location"]
    # Most federal announcements are open only to current federal employees.
    # "All" restricts to announcements the public can actually apply to —
    # the single highest-value filter here for someone outside government.
    if cfg.get("public_only", True):
        base["WhoMayApply"] = "All"
    if cfg.get("min_salary"):
        base["RemunerationMinimumAmount"] = cfg["min_salary"]

    raw, seen = [], set()

    def collect(params):
        for item in _usajobs_collect(headers, params, max_pages):
            oid = item.get("MatchedObjectId")
            if oid and oid not in seen:
                seen.add(oid)
                raw.append(item)
        time.sleep(1)

    # Series codes first — precise, and they surface roles whose titles you
    # would never have thought to search for.
    for code in cfg.get("series", []):
        collect({**base, "JobCategoryCode": str(code)})

    # Keyword fallback for anything the series codes miss.
    for kw in cfg.get("keywords", []):
        collect({**base, "Keyword": kw})

    out = []
    for item in raw:
        d = item.get("MatchedObjectDescriptor", {}) or {}
        details = (d.get("UserArea") or {}).get("Details") or {}

        pay = (d.get("PositionRemuneration") or [{}])[0]
        salary = ""
        if pay.get("MinimumRange"):
            try:
                unit = (pay.get("RateIntervalCode") or "").lower()
                salary = (f"${float(pay['MinimumRange']):,.0f}-"
                          f"${float(pay.get('MaximumRange', 0)):,.0f} "
                          f"{unit}").strip()
            except (TypeError, ValueError):
                pass

        locs = d.get("PositionLocation") or []
        loc = ", ".join(dict.fromkeys(
            l.get("LocationName", "") for l in locs[:3] if l.get("LocationName")))

        desc = _usajobs_description(details)
        # Grade is the level signal the scorer cares most about — GS-07/09 is
        # entry, GS-13 is not.
        grade = " ".join(filter(None, [_first(d.get("JobGrade"), "Code"),
                                       _first(d.get("PositionSchedule"), "Name")]))
        out.append({
            "uid": f"usajobs:{item.get('MatchedObjectId')}",
            "source": "usajobs",
            "title": (d.get("PositionTitle") or "").strip(),
            "company": d.get("OrganizationName", ""),
            "location": loc,
            "url": d.get("PositionURI", ""),
            "posted": (d.get("PublicationStartDate") or "")[:10],
            "salary": salary,
            "description": (f"[{grade}] {desc}" if grade else desc),
            "desc_quality": _quality(desc),
        })
    return out


# ---------------------------------------------------------------------------
# NYC Open Data - Jobs NYC Postings (Socrata dataset kpav-sd4t)
# ---------------------------------------------------------------------------
def fetch_nyc(cfg):
    if not cfg.get("enabled", True):
        return []

    params = {
        "$limit": cfg.get("limit", 500),
        "$order": "posting_date DESC",
    }
    where = []
    agencies = cfg.get("agencies") or []
    if agencies:
        where.append("agency in(%s)" % ",".join(f"'{a}'" for a in agencies))
    # Skip internal-only postings, which you are not eligible for.
    if cfg.get("external_only", True):
        where.append("upper(posting_type) = 'EXTERNAL'")
    if where:
        params["$where"] = " AND ".join(where)

    headers = {"User-Agent": UA}
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token

    data = _get("https://data.cityofnewyork.us/resource/kpav-sd4t.json",
                params=params, headers=headers)
    if not data:
        return []

    out = []
    for j in data:
        jid = j.get("job_id", "")
        salary = ""
        if j.get("salary_range_from"):
            try:
                lo = float(j["salary_range_from"])
                hi = float(j.get("salary_range_to", 0))
                unit = (j.get("salary_frequency") or "").lower()
                salary = f"${lo:,.0f}-${hi:,.0f} {unit}".strip()
            except (TypeError, ValueError):
                pass

        desc = _clean("\n\n".join(filter(None, [
            j.get("job_description", ""),
            j.get("minimum_qual_requirements", ""),
            j.get("preferred_skills", ""),
        ])))
        out.append({
            "uid": f"nyc:{jid}",
            "source": "nyc",
            "title": (j.get("business_title")
                      or j.get("civil_service_title", "")).strip(),
            "company": j.get("agency", ""),
            "location": j.get("work_location", ""),
            "url": f"https://www.nyc.gov/jobs/{jid}",
            "posted": (j.get("posting_date") or "")[:10],
            "salary": salary,
            "description": desc,
            "desc_quality": _quality(desc),
        })
    return out


# ---------------------------------------------------------------------------
# Helper: look up USAJOBS occupational series codes by name.
#   python sources.py --series economist
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--series":
        term = sys.argv[2].lower()
        h = _usajobs_headers()
        data = _get("https://data.usajobs.gov/api/codelist/occupationalseries",
                    headers=h or {"User-Agent": UA})
        if not data:
            sys.exit("could not fetch the codelist")
        items = (data.get("CodeList") or [{}])[0].get("ValidValue", [])
        hits = [i for i in items
                if term in (i.get("Value", "") + i.get("Code", "")).lower()]
        if not hits:
            print(f"no occupational series matching '{term}'")
        for i in hits:
            print(f"  {i.get('Code', ''):<6} {i.get('Value', '')}")
    else:
        print("usage: python sources.py --series <search term>")
