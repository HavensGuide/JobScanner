"""
Company job boards, swept wholesale rather than curated by brand.

There are ~15,900 Greenhouse/Lever/Ashby boards in the public slug directory.
Hand-picking twenty employers throws away the rest, and brand matching is
unreliable anyway: a slug that returns HTTP 200 proves the board is live, not
that it belongs to the company you meant. So sweep everything, filter on role
and location, and ignore the brand.

Cost control, in order of importance:

  1. TITLE AND LOCATION FIRST, DESCRIPTION LATER. Greenhouse's list endpoint
     without `content=true` returns a few hundred bytes per job instead of
     several KB. Filter on that, then fetch full text only for survivors.
     Lever and Ashby include descriptions in the list response already, so
     they are single-pass.
  2. CONCURRENCY. 15,900 sequential requests takes over an hour; at 24
     workers it is about five minutes.
  3. PRIORITISATION. Most boards never produce a relevant title. After the
     first sweep, productive boards are checked every run and barren ones
     rotate, so a routine run touches a fraction of the corpus.

Workday stays curated: it needs POST, pages 20 at a time, and carries no
description in the list response, so a full sweep would cost thousands of
requests for a handful of hits.
"""

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from sources import _clean, _quality

TIMEOUT = 20
UA = {"User-Agent": "jobscan/1.0 (personal job search script)"}
CACHE = Path(".slug-cache")
STATS = Path(".board-stats.json")

DIRECTORY = {
    "greenhouse": "greenhouse_companies.json",
    "lever": "lever_companies.json",
    "ashby": "ashby_companies.json",
}


# ---------------------------------------------------------------------------
# Board productivity memory
# ---------------------------------------------------------------------------
def _load_stats():
    if STATS.exists():
        try:
            return json.loads(STATS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_stats(stats):
    try:
        STATS.write_text(json.dumps(stats), encoding="utf-8")
    except Exception as e:
        print(f"  ! could not write {STATS}: {e}")


def _pick_boards(platform, slugs, stats, cfg):
    """
    Which boards to check this run.

    First run: everything. After that, boards that have ever produced a
    relevant title are always checked; the rest rotate, so the whole corpus is
    revisited regularly without paying for it every run.
    """
    cold = [s for s in slugs if f"{platform}:{s}" not in stats]
    if cfg.get("full_sweep", False) or len(cold) > len(slugs) * 0.5:
        return list(slugs), "full"

    hot, cool = [], []
    for s in slugs:
        rec = stats.get(f"{platform}:{s}")
        if rec is None or rec.get("hits", 0) > 0:
            hot.append(s)
        else:
            cool.append(s)
    random.shuffle(cool)
    take = int(len(cool) * cfg.get("cold_share", 0.25))
    return hot + cool[:take], f"{len(hot)} productive + {take} rotating"


# ---------------------------------------------------------------------------
def _session():
    s = requests.Session()
    s.headers.update(UA)
    a = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    s.mount("https://", a)
    return s


def _list_greenhouse(sess, slug):
    r = sess.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
                 timeout=TIMEOUT)
    if r.status_code != 200:
        return []
    return [{
        "uid": f"greenhouse:{slug}:{j.get('id')}",
        "source": "greenhouse", "_slug": slug, "_id": j.get("id"),
        "title": (j.get("title") or "").strip(),
        "company": slug,
        "location": (j.get("location") or {}).get("name", ""),
        "url": j.get("absolute_url", ""),
        "posted": (j.get("updated_at") or "")[:10],
        "salary": "", "description": "", "desc_quality": "none",
    } for j in r.json().get("jobs", [])]


def _list_lever(sess, slug):
    r = sess.get(f"https://api.lever.co/v0/postings/{slug}?mode=json",
                 timeout=TIMEOUT)
    if r.status_code != 200:
        return []
    data = r.json()
    if not isinstance(data, list):
        return []
    out = []
    for j in data:
        cats = j.get("categories") or {}
        desc = _clean(j.get("descriptionPlain") or j.get("description"))
        posted = ""
        if j.get("createdAt"):
            try:
                posted = time.strftime("%Y-%m-%d",
                                       time.gmtime(j["createdAt"] / 1000))
            except (TypeError, ValueError, OSError):
                pass
        out.append({
            "uid": f"lever:{slug}:{j.get('id')}",
            "source": "lever", "_slug": slug, "_id": j.get("id"),
            "title": (j.get("text") or "").strip(),
            "company": slug,
            "location": cats.get("location", ""),
            "url": j.get("hostedUrl", ""),
            "posted": posted, "salary": "",
            "description": desc, "desc_quality": _quality(desc),
        })
    return out


def _list_ashby(sess, slug):
    r = sess.get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}",
                 params={"includeCompensation": "true"}, timeout=TIMEOUT)
    if r.status_code != 200:
        return []
    data = r.json()
    label = data.get("name") or slug
    out = []
    for j in data.get("jobs", []):
        desc = _clean(j.get("descriptionPlain") or j.get("descriptionHtml"))
        summary = (j.get("compensation") or {}).get("compensationTierSummary")
        out.append({
            "uid": f"ashby:{slug}:{j.get('id')}",
            "source": "ashby", "_slug": slug, "_id": j.get("id"),
            "title": (j.get("title") or "").strip(),
            "company": label,
            "location": j.get("location", ""),
            "url": j.get("jobUrl", ""),
            "posted": (j.get("publishedAt") or "")[:10],
            "salary": summary if isinstance(summary, str) else "",
            "description": desc, "desc_quality": _quality(desc),
        })
    return out


LISTERS = {"greenhouse": _list_greenhouse, "lever": _list_lever,
           "ashby": _list_ashby}


def _detail_greenhouse(sess, slug, job_id):
    """The list call omits content; fetch it only for jobs that survived."""
    r = sess.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs/{job_id}",
                 params={"questions": "false"}, timeout=TIMEOUT)
    if r.status_code != 200:
        return ""
    return _clean(r.json().get("content"))


# ---------------------------------------------------------------------------
def sweep(cfg, keep):
    """
    Sweep every board on the cheap platforms, keeping only jobs where
    keep(job) is True. `keep` sees title, location, and company but NOT the
    description — that is the point, since descriptions are what cost.
    """
    workers = cfg.get("workers", 24)
    stats = _load_stats()
    sess = _session()
    kept_all = []
    platforms = cfg.get("platforms") or list(DIRECTORY)

    for platform, fname in DIRECTORY.items():
        if platform not in platforms:
            continue
        path = CACHE / fname
        if not path.exists():
            print(f"  ! {platform}: no slug cache. Run: python slugs.py --refresh")
            continue
        slugs = json.loads(path.read_text(encoding="utf-8"))
        todo, why = _pick_boards(platform, slugs, stats, cfg)
        print(f"  {platform:<11} {len(todo):>5} of {len(slugs):>5} boards ({why})")

        lister = LISTERS[platform]
        raw, live = [], 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(lister, sess, s): s for s in todo}
            for i, f in enumerate(as_completed(futs), 1):
                slug = futs[f]
                try:
                    jobs = f.result()
                except Exception:
                    jobs = []
                if jobs:
                    live += 1
                hits = [j for j in jobs if keep(j)]
                raw += hits
                rec = stats.setdefault(f"{platform}:{slug}", {})
                rec["jobs"] = len(jobs)
                rec["hits"] = rec.get("hits", 0) + len(hits)
                rec["seen"] = int(time.time())
                if i % 2500 == 0:
                    print(f"    ...{i}/{len(todo)}, {len(raw)} matches so far")
        print(f"    {len(raw):>4} matches from {live} live boards "
              f"in {time.time() - t0:.0f}s")

        if platform == "greenhouse" and raw:
            cap = cfg.get("max_details", 400)
            targets = raw[:cap]
            if len(raw) > cap:
                print(f"    ! {len(raw)} matches over max_details={cap}; "
                      f"{len(raw) - cap} stored without descriptions")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_detail_greenhouse, sess, j["_slug"], j["_id"]): j
                        for j in targets}
                for f in as_completed(futs):
                    j = futs[f]
                    try:
                        d = f.result()
                    except Exception:
                        d = ""
                    j["description"] = d
                    j["desc_quality"] = _quality(d)
            print(f"    fetched {len(targets)} descriptions")

        kept_all += raw

    _save_stats(stats)
    for j in kept_all:
        j.pop("_slug", None)
        j.pop("_id", None)
    return kept_all


# ---------------------------------------------------------------------------
# Workday stays curated — POST, 20 per page, no description in the list.
# ---------------------------------------------------------------------------
def fetch_workday(specs, search_text="", max_pages=3, max_details=25):
    out = []
    sess = _session()
    for spec in specs or []:
        parts = str(spec).split("|")
        if len(parts) != 3:
            print(f"  ! workday: bad slug {spec!r}, want tenant|wdN|site")
            continue
        tenant, dc, site = parts
        base = (f"https://{tenant}.{dc}.myworkdayjobs.com"
                f"/wday/cxs/{tenant}/{site}/jobs")
        postings = []
        for page in range(max_pages):
            try:
                r = sess.post(base, timeout=TIMEOUT,
                              headers={"Content-Type": "application/json"},
                              json={"appliedFacets": {}, "limit": 20,
                                    "offset": page * 20,
                                    "searchText": search_text})
                if r.status_code != 200:
                    break
                batch = r.json().get("jobPostings", []) or []
            except Exception as e:
                print(f"  ! workday {tenant}: {type(e).__name__}: {e}")
                break
            postings += batch
            if len(batch) < 20:
                break
        for i, j in enumerate(postings):
            path = j.get("externalPath", "")
            desc = ""
            if i < max_details and path:
                try:
                    d = sess.get(f"https://{tenant}.{dc}.myworkdayjobs.com"
                                 f"/wday/cxs/{tenant}/{site}{path}",
                                 timeout=TIMEOUT)
                    if d.status_code == 200:
                        desc = _clean((d.json().get("jobPostingInfo") or {})
                                      .get("jobDescription"))
                except Exception:
                    pass
            out.append({
                "uid": f"workday:{tenant}:{path}",
                "source": "workday",
                "title": (j.get("title") or "").strip(),
                "company": tenant,
                "location": j.get("locationsText", ""),
                "url": f"https://{tenant}.{dc}.myworkdayjobs.com/{site}{path}",
                "posted": "", "salary": "",
                "description": desc, "desc_quality": _quality(desc),
            })
    return out


def fetch_ats(cfg, keep=None):
    """Entry point for main.py. `keep` filters on title/location during sweep."""
    cfg = cfg or {}
    out = []
    if (cfg.get("sweep") or {}).get("enabled", True) and keep is not None:
        out += sweep(cfg.get("sweep") or {}, keep)
    wd = cfg.get("workday")
    if wd:
        got = fetch_workday(wd, cfg.get("workday_search", ""),
                            cfg.get("workday_max_pages", 3),
                            cfg.get("workday_max_details", 25))
        print(f"  {'workday':<11} {len(got):>4} from {len(wd)} curated board(s)")
        out += got
    return out
