#!/usr/bin/env python3
"""
missed.py - why did jobscan never show you a job you applied to?

For every application in your tracker that is not in seen.db, this works
out where it was posted and walks it through the same stages main.py uses:

    platform     can jobscan read that site at all?
    board        is the employer's board in the slug directory, and how
                 often is it checked (every run, or 1 in 4 if it has never
                 produced a match)?
    live         is the posting still up? (it asks the board's public API)
    filters      hard_exclude, the title filter, the track, the date rule
                 and the location allowlist, each with the reason it fails

    python missed.py                     # tracker from jev.fit.tracker
    python missed.py tracker.csv
    python missed.py --pin               # also: check those employers' boards
                                         # every run from now on

Reads only public job-board APIs. Writes nothing unless you pass --pin.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

TIMEOUT = 20


def platform_of(url):
    """(platform, slug or None, job id or None, extra)."""
    u = (url or "").strip()
    pr = urlparse(u)
    host, path = pr.netloc.lower(), pr.path
    q = parse_qs(pr.query)
    m = re.match(r"^/([^/]+)/jobs/(\d+)", path)
    if "greenhouse.io" in host and m:
        return "greenhouse", m.group(1), m.group(2), ""
    if "gh_jid" in q:
        return "greenhouse", None, q["gh_jid"][0], host
    m = re.match(r"^/([^/]+)/([0-9a-f-]{36})", path)
    if host.endswith("lever.co") and m:
        return "lever", m.group(1), m.group(2), ""
    if "ashbyhq.com" in host and m:
        return "ashby", m.group(1), m.group(2), ""
    m = re.match(r"^([a-z0-9-]+)\.(wd\d+)\.myworkdayjobs\.com$", host)
    if m:
        parts = [x for x in path.split("/") if x]
        if parts and re.fullmatch(r"[a-z]{2}-[A-Z]{2}", parts[0]):
            parts = parts[1:]
        site = parts[0] if parts else ""
        return "workday", f"{m.group(1)}|{m.group(2)}|{site}", None, ""
    if re.search(r"(?:^|\.)(?:cityjobs\.nyc\.gov|jobs\.nyc\.gov|nyc\.gov)$",
                 host) or "nyc.gov" in host:
        return "nyc", None, None, host
    if "usajobs.gov" in host:
        return "usajobs", None, None, host
    return "other", None, None, host or "(no link)"


NO_READER = {
    "joinhandshake.com": "Handshake needs your login; there is no public API",
    "linkedin.com": "LinkedIn has no public jobs API",
    "indeed.com": "Indeed has no public jobs API",
    "icims.com": "iCIMS boards are not read by jobscan",
    "paylocity.com": "Paylocity boards are not read by jobscan",
    "acquiretm.com": "AcquireTM boards are not read by jobscan",
    "smartrecruiters.com": "SmartRecruiters is not read by jobscan",
    "jobvite.com": "Jobvite is not read by jobscan",
    "taleo.net": "Taleo is not read by jobscan",
    "successfactors": "SuccessFactors is not read by jobscan",
}


def no_reader(host):
    for k, v in NO_READER.items():
        if k in host:
            return v
    return f"{host} is the employer's own site; jobscan has no reader for it"


# --------------------------------------------------------------------------
def _get(sess, url):
    try:
        r = sess.get(url, timeout=TIMEOUT)
        return r.status_code, (r.json() if r.status_code == 200 else None)
    except Exception as e:
        return None, str(e)


def live_posting(sess, platform, slug, job_id):
    """(status, title, location). status: live | closed | unknown."""
    if platform == "greenhouse":
        code, d = _get(sess, f"https://boards-api.greenhouse.io/v1/boards/"
                             f"{slug}/jobs/{job_id}?questions=false")
        if code == 200:
            return "live", d.get("title", ""), (d.get("location") or {}).get(
                "name", "")
    elif platform == "lever":
        code, d = _get(sess, f"https://api.lever.co/v0/postings/{slug}/"
                             f"{job_id}")
        if code == 200:
            return "live", d.get("text", ""), (d.get("categories") or {}).get(
                "location", "")
    elif platform == "ashby":
        code, d = _get(sess, f"https://api.ashbyhq.com/posting-api/job-board/"
                             f"{slug}")
        if code == 200:
            j = next((j for j in d.get("jobs", []) if j.get("id") == job_id),
                     None)
            if j:
                return "live", j.get("title", ""), j.get("location", "")
            code = 404
    else:
        return "unknown", "", ""
    return ("closed" if code == 404 else "unknown"), "", ""


def find_greenhouse_slug(sess, slugs, company, job_id, host=""):
    """A custom-domain Greenhouse link carries the job id but not the board.
    Try boards whose name resembles the company or the site's own name, and
    keep the one that has this job."""
    labels = [x for x in (host or "").lower().split(".")
              if x not in ("www", "careers", "jobs", "com", "org", "io",
                           "net", "co")]
    wants = [re.sub(r"[^a-z0-9]", "", w) for w in
             [(company or "").lower()] + labels[:1]]
    wants = [w for w in wants if len(w) >= 3]
    cands = []
    for want in wants:
        for s in slugs:
            k = s.lower().replace("-", "")
            if (want[:6] in k or k in want) and s not in cands:
                cands.append(s)
    cands = cands[:15]
    if not cands:
        return None
    for s in cands:
        code, _ = _get(sess, f"https://boards-api.greenhouse.io/v1/boards/"
                             f"{s}/jobs/{job_id}?questions=false")
        if code == 200:
            return s
    return None


# --------------------------------------------------------------------------
def filters(job, cfg, track_hint=None):
    """[(stage, ok, note)] in main.py's order, for an ATS posting."""
    import main
    import score as scoring
    excludes = [t.lower() for t in cfg.get("hard_exclude", [])]
    career, bridge, excl, tol = scoring.track_terms(cfg)
    professional = scoring.professional_titles(cfg)
    never = (scoring.bridge_never(cfg) if hasattr(scoring, "bridge_never")
             else ())
    sweep = [s.lower() for s in
             (cfg.get("ats", {}).get("sweep", {}).get("title_must_match")
              or cfg.get("relevance_signals", []))]
    if (cfg.get("bridge") or {}).get("enabled"):
        sweep += bridge + tol
    if scoring.method_for(scoring.merge_scoring_cfg(cfg), "career") == \
            "filter":
        import paths
        sweep += paths.all_title_terms(cfg)
    title = (job.get("title") or "").lower()
    out = []
    hx = next((t for t in excludes if t in title), None)
    out.append(("hard_exclude", hx is None,
                f"title contains {hx!r}" if hx else ""))
    st = any(t in title for t in sweep)
    out.append(("title filter", st, "" if st else
                "no word from ats.sweep.title_must_match is in the title"))
    try:
        track = scoring.classify_track(job, career, bridge, excl, tol,
                                       professional, never)
    except TypeError:
        track = scoring.classify_track(job, career, bridge, excl, tol,
                                       professional)
    out.append(("track", track is not None,
                f"-> {track}" if track else
                "dropped as public-facing or bridge.never"))
    stale = scoring.temporal_exclusion_reason(
        {"title": job.get("title") or "", "posted": ""})
    out.append(("date rule", not stale, stale or ""))
    t = track or track_hint or "career"
    locs = [l.lower() for l in (cfg.get("bridge", {}).get(
        "location_allowlist", []) if t == "bridge" else
        cfg.get("location_allowlist", []))]
    block = [b.lower() for b in cfg.get("location_blocklist", [])]
    ok = main.location_ok(job, locs, block)
    out.append(("location", ok, "" if ok else
                f"{job.get('location')!r} is not on the {t} allowlist "
                f"or is blocklisted"))
    return out


def main_():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracker", nargs="?", default="")
    ap.add_argument("--pin", action="store_true",
                    help="check the boards of employers you applied to on "
                         "every run")
    ap.add_argument("--db", default="seen.db")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    import sqlite3
    import requests
    import ats
    import fit
    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    try:
        apps, source = fit.load_tracker(args.tracker or None,
                                        cfg.get("jev") or {})
    except Exception as e:
        sys.exit(str(e))

    conn = sqlite3.connect(args.db)
    rows = [dict(zip(("uid", "title", "company", "url", "track",
                      "description"), r)) for r in conn.execute(
        "SELECT uid, title, company, url, track, description FROM jobs")]
    found = fit.match_tracker(apps, rows)
    print(f"{len(apps)} applications in {source}; {len(found)} are in "
          f"{args.db}\n")
    for n, j in found.items():
        if j.get("track") == "bridge" or len((j.get("description") or "")
                                             .strip()) < 400:
            why = ("stored on the bridge track"
                   if j.get("track") == "bridge" else
                   "stored without a description")
            print(f"  in seen.db but {why}: {apps[n]['title'][:50]}")
    print()

    slugs = {}
    for p, fname in ats.DIRECTORY.items():
        f = ats.CACHE / fname
        try:
            slugs[p] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            slugs[p] = []
    stats = ats._load_stats()
    sess = ats._session()
    workday = [str(w) for w in (cfg.get("ats", {}).get("workday") or [])]
    cold_share = (cfg.get("ats", {}).get("sweep", {}) or {}).get(
        "cold_share", 0.25)
    nyc_cache = None
    verdicts = Counter()
    pins = []

    for n, a in enumerate(apps):
        if n in found:
            continue
        plat, slug, jid, extra = platform_of(a["url"])
        print(f"{a['title'][:56]}  ({a['company'][:30]})")

        if plat == "other":
            verdicts["no reader for that site"] += 1
            print(f"    platform  {no_reader(extra)}\n")
            continue
        if plat == "usajobs":
            verdicts["USAJOBS search missed it"] += 1
            print("    platform  USAJOBS: your search only covers series "
                  "0110 and 1560 and two keywords (usajobs: in config.yaml)\n")
            continue
        if plat == "workday":
            spec = slug
            have = any(w.split("|")[0] == spec.split("|")[0] for w in workday)
            if have:
                verdicts["Workday board read, posting not matched"] += 1
                print(f"    board     {spec} is in ats.workday; the search "
                      f"text {cfg.get('ats', {}).get('workday_search')!r} or "
                      f"the 3-page limit may have missed it\n")
            else:
                verdicts["Workday board not in your list"] += 1
                print(f"    board     not read. Add under ats.workday:\n"
                      f"                - \"{spec}\"\n")
            continue
        if plat == "nyc":
            if nyc_cache is None:
                import sources
                ncfg = dict(cfg.get("nyc_open_data") or {})
                lim = int(ncfg.get("limit", 500))
                try:
                    got = sources.fetch_nyc(ncfg) or []
                    big = sources.fetch_nyc(dict(ncfg, limit=max(5000,
                                                                 lim * 10)))
                except Exception as e:
                    got, big = [], []
                    print(f"    ! NYC Open Data: {e}")
                nyc_cache = (lim, got, big or [])
            lim, got, big = nyc_cache
            want = fit._norm(a["title"])
            in_big = any(fit._norm(j.get("title")) == want for j in big)
            in_got = any(fit._norm(j.get("title")) == want for j in got)
            capped = len(got) >= lim
            print(f"    source    NYC Open Data returned {len(got)} postings "
                  f"at limit {lim}" + (" (CAPPED)" if capped else "")
                  + f"; {len(big)} with a higher limit")
            if in_big and not in_got:
                verdicts["NYC Open Data limit too low"] += 1
                print(f"    verdict   found only above the limit. Raise "
                      f"nyc_open_data.limit to {max(2000, len(big) + 200)}\n")
            elif in_got:
                verdicts["NYC: fetched now"] += 1
                print("    verdict   fetched today; it was probably posted "
                      "and closed between runs, or the title changed\n")
            else:
                verdicts["NYC: not open now"] += 1
                print("    verdict   not in today's feed at any limit: "
                      "closed, or listed under another title\n")
            continue

        # Greenhouse / Lever / Ashby
        board_list = slugs.get(plat) or []
        if slug is None and plat == "greenhouse":
            slug = find_greenhouse_slug(sess, board_list, a["company"], jid,
                                        extra)
            if slug is None:
                verdicts["Greenhouse board not identified"] += 1
                print(f"    board     Greenhouse job {jid} on {extra}; no "
                      f"board in the directory matched {a['company']!r}. "
                      f"Find it with: python slugs.py --find "
                      f"\"{a['company']}\"\n")
                continue
        in_dir = slug in board_list
        rec = stats.get(f"{plat}:{slug}")
        when = (time.strftime("%Y-%m-%d", time.localtime(rec["seen"]))
                if rec and rec.get("seen") else "never")
        if not in_dir:
            freq = "never: not in the slug directory"
        elif rec is None or rec.get("hits", 0) > 0:
            freq = "every run"
        else:
            freq = f"about 1 run in {round(1 / max(cold_share, 0.01))}"
        print(f"    board     {plat}:{slug}  checked {freq}; last {when}")
        if in_dir or slug:
            pins.append((plat, slug, in_dir))
        status, title, loc = live_posting(sess, plat, slug, jid)
        job = {"title": title or a["title"], "location": loc,
               "company": a["company"]}
        print(f"    live      {status}" + (f": {title!r} in {loc!r}"
                                          if status == "live" else ""))
        fails = [(s, note) for s, ok, note in filters(job, cfg) if not ok]
        for s, note in fails:
            print(f"    FAILS     {s}: {note}")
        if fails:
            verdicts[f"dropped by {fails[0][0]}"] += 1
        elif not in_dir:
            verdicts["board not in directory"] += 1
        elif freq != "every run":
            verdicts["board checked 1 run in 4"] += 1
            print("    verdict   passes every filter; its board is only "
                  "checked on some runs, so the posting came and went")
        else:
            verdicts["passes now; missed for another reason"] += 1
            print("    verdict   passes every filter and the board is checked "
                  "every run: it may have been posted after your last run, "
                  "or matched under a different title")
        print()

    print("SUMMARY")
    for k, v in verdicts.most_common():
        print(f"  {v:>3}  {k}")

    if args.pin and pins:
        changed = 0
        for plat, slug, in_dir in pins:
            if not in_dir:
                f = ats.CACHE / ats.DIRECTORY[plat]
                data = slugs.get(plat)
                if isinstance(data, list):
                    data.append(slug)
                    f.write_text(json.dumps(data), encoding="utf-8")
                    print(f"  added {plat}:{slug} to the slug directory")
                else:
                    print(f"  ! could not add {slug}: unexpected format in {f}")
                    continue
            rec = stats.setdefault(f"{plat}:{slug}", {})
            if rec.get("hits", 0) < 1:
                rec["hits"] = 1
                changed += 1
        ats._save_stats(stats)
        print(f"\npinned {changed} board(s): checked every run from now on. "
              f"`python slugs.py --refresh` rebuilds the directory and drops "
              f"additions, so rerun --pin after a refresh.")
    elif pins:
        print(f"\n{len(pins)} of these are on boards jobscan can read. "
              f"`python missed.py --pin` checks those boards every run.")


if __name__ == "__main__":
    main_()
