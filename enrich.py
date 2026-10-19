#!/usr/bin/env python3
"""
enrich.py — fill in descriptions for postings that arrived without one.

Four rows in a recent digest read "no desc" and still scored 99, because the
scorer was judging a title. jev.py now refuses to score those at all, which
is correct but turns them into a pile of manual-review rows. Better to go and
get the text.

Where the gaps come from:

  workday   The list response carries no description and only the first
            `workday_max_details` (25) get a detail call.
  lever     Some boards publish an empty descriptionPlain.
  greenhouse  Anything past `max_details`, and the occasional board that
            returns content only on the detail endpoint.
  adzuna    Excerpts, which are short but not empty, so they are left alone.

Two routes, in order:

  1. THE PLATFORM API. greenhouse, lever and workday all have a single-posting
     endpoint, and ats.py already knows the shapes. Clean JSON beats scraping
     the rendered page every time.
  2. A PLAIN HTTP GET of the posting URL, stripped with sources._clean.

Then the screen. Fetching a job URL gets you a sign-in wall, a cookie banner,
a "this position is no longer available" notice or a board index about as
often as it gets the job, and all of those are long enough to look like a
description. JevK5 answers "is this the body of one job posting" in about
0.2s per page, and anything that fails is discarded rather than scored.

    python enrich.py                  # fill gaps in seen.db, then stop
    python enrich.py --limit 50 --dry-run
    python enrich.py --no-screen      # keep whatever comes back, unverified
"""
import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

import env
env.load()

from sources import _clean, _quality

TIMEOUT = 20
# A real browser string. An honest "jobscan/1.0" gets 403s or a bot
# interstitial from aggregators and most ATS platforms, which is not a
# crawl-policy question: these are pages a person is expected to open by
# clicking the link in the feed, one at a time, which is exactly what this
# does. The Accept headers matter too; some hosts serve a stub to a client
# that does not ask for HTML.
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Hosts that only ever redirect to the employer's real posting. Landing on
# one of these means the redirect did not complete, so whatever text is
# there is tracking furniture, not the job.
AGGREGATORS = ("adzuna.", "indeed.", "linkedin.com", "glassdoor.",
               "ziprecruiter.", "simplyhired.", "talent.com", "jooble.",
               "neuvoo.", "dice.com/jobs/redirect")

# Pages that are plainly not a posting, caught before spending a model call.
_JUNK = re.compile(
    r"(sign in to continue|enable javascript|access denied|403 forbidden"
    r"|404 not found|page not found|no longer accepting applications"
    r"|this position has been filled|cookie preferences)", re.I)


def _session():
    s = requests.Session()
    s.headers.update(UA)
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=16,
                                                      pool_maxsize=16))
    return s


# ---------------------------------------------------------------------------
# Route 1: the platform's own single-posting endpoint
# ---------------------------------------------------------------------------
def _from_api(sess, uid, url):
    """
    Description via the source's API, or "" if there is no API route.

    uid shapes, set in ats.py:
      greenhouse:<slug>:<id>
      lever:<slug>:<id>
      ashby:<slug>:<id>
      workday:<tenant>:<externalPath>
    """
    parts = (uid or "").split(":", 2)
    if len(parts) < 3:
        return ""
    source, slug, ident = parts

    try:
        if source == "greenhouse":
            r = sess.get(f"https://boards-api.greenhouse.io/v1/boards/"
                         f"{slug}/jobs/{ident}",
                         params={"questions": "false"}, timeout=TIMEOUT)
            if r.status_code == 200:
                return _clean(r.json().get("content"))

        elif source == "lever":
            r = sess.get(f"https://api.lever.co/v0/postings/{slug}/{ident}",
                         timeout=TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                body = j.get("descriptionPlain") or j.get("description") or ""
                # Lever splits the detail into lists the summary omits.
                for sec in (j.get("lists") or []):
                    body += "\n\n" + str(sec.get("text", "")) + "\n" + \
                            str(sec.get("content", ""))
                body += "\n\n" + str(j.get("additionalPlain")
                                     or j.get("additional") or "")
                return _clean(body)

        elif source == "workday":
            # uid drops the data-centre segment, but the stored URL has it:
            # https://<tenant>.<dc>.myworkdayjobs.com/<site><path>
            host = urlparse(url or "").netloc
            path = urlparse(url or "").path
            if host and path:
                bits = path.strip("/").split("/", 1)
                if len(bits) == 2:
                    site, rest = bits
                    r = sess.get(f"https://{host}/wday/cxs/{slug}/{site}/{rest}",
                                 timeout=TIMEOUT)
                    if r.status_code == 200:
                        info = r.json().get("jobPostingInfo") or {}
                        return _clean(info.get("jobDescription"))
    except Exception:
        return ""
    return ""


# ---------------------------------------------------------------------------
# Route 2: fetch the page and strip it
# ---------------------------------------------------------------------------
def resolve(sess, url):
    """
    (final_url, note) after following an aggregator's click-through.

    An Adzuna link is /land/ad/<id>, a tracking hop to the employer's own
    posting. The job text is never on that hop, so what matters is where it
    goes. Returned separately from the fetch because the destination is
    worth keeping: it is stable, it is re-fetchable, and it is the link you
    would actually want to open.
    """
    if not url or not url.startswith("http"):
        return url, "no link"
    try:
        r = sess.get(url, timeout=TIMEOUT, allow_redirects=True)
    except Exception as e:
        return url, f"{type(e).__name__}"
    final = str(r.url or url)
    if r.status_code != 200:
        return final, f"HTTP {r.status_code}"
    if any(a in urlparse(final).netloc.lower() or a in final.lower()
           for a in AGGREGATORS):
        return final, "stayed on the aggregator"
    return final, ""


def _from_page(sess, url):
    if not url or not url.startswith("http"):
        return ""
    try:
        r = sess.get(url, timeout=TIMEOUT, allow_redirects=True)
    except Exception:
        return ""
    if r.status_code != 200 or "html" not in \
            r.headers.get("Content-Type", "").lower():
        return ""
    body = r.text
    # Drop script and style wholesale first: _clean strips tags but would
    # otherwise leave minified JavaScript in the "description".
    body = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", body,
                  flags=re.I | re.S)
    # Prefer the main content region when the page marks one.
    m = re.search(r"<(main|article)[^>]*>(.*?)</\1>", body, re.I | re.S)
    if m:
        body = m.group(2)
    return _clean(body)


def fetch_one(sess, job):
    """
    (description, route, note) for one posting.

    An empty description means no luck, and `note` says why, because
    "could not fetch" with no reason is not something anyone can act on.

    Order matters. The platform API is tried on the stored URL first, since
    a greenhouse or lever link needs no redirect. Aggregator links go
    through resolve() to find the employer's real posting, and the API
    route is then retried against THAT, because the destination is often an
    ATS this already knows how to read properly.
    """
    uid, url = job.get("uid", ""), job.get("url", "")

    d = _from_api(sess, uid, url)
    if len(d.strip()) >= 200:
        return d, "api", ""

    final, note = resolve(sess, url)
    if final and final != url:
        job["resolved_url"] = final
        d = _from_api(sess, _uid_for(final), final)
        if len(d.strip()) >= 200:
            return d, "api", f"via {urlparse(final).netloc}"

    if note:
        return "", "none", note

    d = _from_page(sess, final or url)
    if len(d.strip()) >= 200:
        return d, "page", (f"via {urlparse(final).netloc}"
                           if final != url else "")
    return "", "none", ("page had no readable description"
                        if d else "nothing usable at that link")


def _uid_for(url):
    """
    A synthetic uid so _from_api can dispatch on a resolved URL.

    _from_api keys off the uid's source prefix, which the aggregator's uid
    does not carry. Reading the platform off the destination host lets a
    link that lands on greenhouse be read through greenhouse's own API
    rather than scraped.
    """
    host = urlparse(url or "").netloc.lower()
    path = urlparse(url or "").path.strip("/").split("/")
    if "greenhouse.io" in host and len(path) >= 2:
        # boards.greenhouse.io/<slug>/jobs/<id>
        return f"greenhouse:{path[0]}:{path[-1]}"
    if "lever.co" in host and len(path) >= 2:
        return f"lever:{path[0]}:{path[-1]}"
    if "myworkdayjobs.com" in host:
        return f"workday:{host.split('.')[0]}:"
    return ""


# ---------------------------------------------------------------------------
def needs_text(job, thin=False):
    """
    Is this posting worth fetching?

    Missing or unusably short always qualifies. With `thin`, an excerpt does
    too: Adzuna truncates its descriptions, which leaves a few hundred
    characters that clear a length test but are too incomplete to draft
    from, so tailor.py refuses them. Without this they fall between the two
    checks and nothing ever fixes them.
    """
    if not (job.get("url") or "").startswith("http"):
        return False
    d = (job.get("description") or "").strip()
    if len(d) < 200:
        return True
    return bool(thin and (job.get("desc_quality") == "excerpt"
                          or _quality(d) == "excerpt"))


def enrich(jobs, workers=8, screen=True, jev_cfg=None, quiet=False,
           profile=None, thin=False):
    """
    Fill in `description` and `desc_quality` in place. Returns a tally.

    Only touches postings whose description is missing or unusably short
    (and, with `thin`, excerpts), so it is safe to run over anything.
    """
    targets = [j for j in jobs if needs_text(j, thin)]
    tally = {"tried": len(targets), "api": 0, "page": 0, "none": 0,
             "rejected": 0, "filled": 0}
    if not targets:
        return tally

    sess = _session()
    got = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, sess, j): j for j in targets}
        for f in as_completed(futs):
            j = futs[f]
            try:
                desc, route, note = f.result()
            except Exception as e:
                desc, route, note = "", "none", type(e).__name__
            tally[route] = tally.get(route, 0) + 1
            if desc and not _JUNK.search(desc[:1200]):
                got.append((j, desc))
            elif desc:
                tally["rejected"] += 1
                note = note or "looked like boilerplate, not a posting"
            # Say why, per posting. A silent failure is one nobody can fix.
            if not desc and not quiet:
                print(f"    no text for {(j.get('title') or j.get('uid'))[:46]}"
                      f": {note or 'unknown'}")
                if j.get("resolved_url"):
                    print(f"      link resolved to {j['resolved_url'][:96]}")

    if not quiet:
        print(f"  - enrich: fetched {len(got)}/{len(targets)} in "
              f"{time.perf_counter() - t0:.0f}s "
              f"(api {tally['api']}, page {tally['page']}, "
              f"missing {tally['none']})")

    # The screen. A sign-in wall is longer than min_desc_chars and would be
    # scored as though it were the job.
    if screen and got:
        try:
            import jev
            verdicts = jev.screen_texts([d for _, d in got], jev_cfg,
                                        profile)
        except ImportError:
            verdicts = None
        if verdicts is None:
            if not quiet:
                print("  ! enrich: no model available to screen with; "
                      "keeping fetched text unscreened")
        else:
            kept = []
            for (j, d), (is_posting, p) in zip(got, verdicts):
                if is_posting:
                    kept.append((j, d))
                else:
                    tally["rejected"] += 1
                    if not quiet:
                        print(f"    not a posting (p={p:.2f}): "
                              f"{(j.get('title') or '')[:52]}")
            got = kept

    for j, d in got:
        # Replacing an excerpt is only an improvement if the new text is
        # actually fuller. A short landing page that passed the screen must
        # not overwrite a 400-character excerpt that at least came from the
        # employer's own feed.
        old = (j.get("description") or "").strip()
        if old and len(d.strip()) <= len(old) * 1.2:
            tally["rejected"] += 1
            if not quiet:
                print(f"    kept the existing text, the fetch was no fuller: "
                      f"{(j.get('title') or '')[:52]}")
            continue
        j["description"] = d
        j["desc_quality"] = _quality(d)
        tally["filled"] += 1
    if not quiet:
        print(f"  - enrich: filled {tally['filled']}, "
              f"rejected {tally['rejected']}")
    return tally


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--uid", metavar="UID",
                    help="enrich one stored posting, e.g. adzuna:5932893477")
    ap.add_argument("--excerpts", action="store_true",
                    help="also re-fetch truncated excerpts (Adzuna), which "
                         "tailor.py refuses to draft from")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-screen", action="store_true",
                    help="keep fetched text without asking jev whether it is "
                         "actually a posting")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report, write nothing")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", metavar="PROFILE",
                    help="model profile used to screen fetched pages "
                         "(default: ai.jev)")
    args = ap.parse_args()

    import yaml
    import store
    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}

    conn = store.connect()
    cols = ("uid", "title", "company", "location", "url", "posted",
            "description", "desc_quality", "track")
    select = f"SELECT {', '.join(cols)} FROM jobs WHERE "

    if args.uid:
        # One posting by id. No quality test: you asked for this one, so the
        # only question is whether there is a link to follow.
        rows = list(conn.execute(
            select + "uid = ? AND url LIKE 'http%'", (args.uid,)))
        if not rows:
            sys.exit(f"{args.uid}: not stored, or it has no http link.")
        what = f"posting {args.uid}"
    else:
        # An Adzuna excerpt runs a few hundred characters, so it clears a
        # length test while still being too thin to draft from: tailor.py
        # stops on it and nothing here used to pick it up. --excerpts closes
        # that gap. It stays opt-in because excerpts are numerous and each
        # one costs an HTTP request.
        where = "(description IS NULL OR length(trim(description)) < 200)"
        if args.excerpts:
            where = f"({where} OR desc_quality = 'excerpt')"
        rows = list(conn.execute(
            select + where + " AND url LIKE 'http%' "
            "ORDER BY first_seen DESC LIMIT ?", (args.limit,)))
        what = ("posting(s) without a usable description" if not args.excerpts
                else "posting(s) with a thin or missing description")

    jobs = [dict(zip(cols, r)) for r in rows]
    if not jobs:
        sys.exit("nothing to enrich: every stored posting already has text. "
                 "Adzuna excerpts are skipped unless you pass --excerpts.")

    print(f"{len(jobs)} {what}\n")
    prof = None
    if args.model:
        import llm
        try:
            prof = llm._lookup(cfg, args.model, "jev")
        except llm.LLMError as e:
            sys.exit(str(e))
    before = {j["uid"]: (j.get("description") or "") for j in jobs}
    enrich(jobs, workers=args.workers, screen=not args.no_screen,
           jev_cfg=cfg.get("jev"), profile=prof,
           thin=bool(args.excerpts or args.uid))

    # Write back only what actually changed, so a run that fetched nothing
    # cannot rewrite rows with the text they already had.
    filled = [j for j in jobs
              if len((j.get("description") or "").strip()) >= 200
              and (j.get("description") or "") != before.get(j["uid"])]
    if args.dry_run:
        print("\n--- dry run, not writing ---")
        for j in filled[:10]:
            print(f"  {len(j['description']):>6} chars  {j['title'][:56]}")
        return
    if filled:
        conn.executemany(
            "UPDATE jobs SET description=?, desc_quality=? WHERE uid=?",
            [(j["description"], j["desc_quality"], j["uid"]) for j in filled])
        # Replace an aggregator's tracking hop with the employer's own link.
        resolved = [(j["resolved_url"], j["uid"]) for j in filled
                    if j.get("resolved_url")]
        if resolved:
            conn.executemany("UPDATE jobs SET url=? WHERE uid=?", resolved)
            print(f"  - resolved {len(resolved)} link(s) to the employer's "
                  f"own posting")
        conn.commit()
        print(f"\nwrote {len(filled)} description(s).")
        print("Next: python score.py --backfill --rescore --scorer jev "
              "--limit 2000")


if __name__ == "__main__":
    main()
