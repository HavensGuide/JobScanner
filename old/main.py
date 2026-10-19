#!/usr/bin/env python3
"""
jobscan - pull job listings from free APIs, keep only what's new, email a digest.

    python main.py            # normal run
    python main.py --dry-run  # fetch + score, but don't record or email
    python main.py --no-score # skip the model, keyword fallback only (free)

Pipeline order matters:

    fetch -> hard_exclude -> location -> dedupe -> SCORE -> record ALL -> email

Scoring sits after dedupe so you never pay to score the same posting twice,
and every scored job is recorded even when it scores too low to email —
otherwise it looks new again in twelve hours and gets re-scored forever.

Sources are slug-free: Adzuna, USAJOBS, NYC Open Data. No per-company tokens
to maintain, nothing that 404s silently when an employer changes ATS.
"""

import sys
import yaml

import env
env.load()          # read .env before anything looks for a credential

import ats
import digest
import score as scoring
import sources
import store

# Sources whose postings are inherently in-region.
LOCAL_SOURCES = {"nyc"}


def location_ok(job, locations, blocked=(), explicit=False):
    """
    Location is still a hard gate. A model cannot fix a commute.

    `explicit` means the caller passed --location, i.e. named the places they
    want. That overrides both the config blocklist and the LOCAL_SOURCES
    bypass, because otherwise asking for "toronto" silently returns nothing
    (it is on the blocklist) while NYC results come back anyway (nyc is
    exempt). A request you typed should beat a default you did not.
    """
    if not explicit and job.get("source") in LOCAL_SOURCES:
        return True
    loc = (job.get("location") or "").lower()
    # Blocklist wins. "remote" as a bare token otherwise admits
    # "Remote - Lithuania" and "Global (Remote)".
    if not explicit and loc and any(b in loc for b in blocked):
        return False
    if not locations:
        return True
    # Keep jobs with a blank location rather than silently dropping them;
    # some ATS feeds just don't populate the field.
    return (not loc) or any(p in loc for p in locations)


def _arg(name, default=None):
    """Read --name value from argv."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    dry_run = "--dry-run" in sys.argv
    no_score = "--no-score" in sys.argv

    # --location "new york,newark"  narrow to these for one run
    # --source ashby,workday        only these sources
    # --max-score 400               raise the per-run scoring cap
    only_loc = [x.strip().lower() for x in (_arg("--location") or "").split(",") if x.strip()]
    only_src = [x.strip().lower() for x in (_arg("--source") or "").split(",") if x.strip()]
    cap_override = _arg("--max-score")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    excludes = [t.lower() for t in cfg.get("hard_exclude", [])]
    locations = only_loc or [l.lower() for l in cfg.get("location_allowlist", [])]
    if only_loc:
        print(f"(--location: restricted to {', '.join(only_loc)}; "
              f"config allowlist and blocklist ignored for this run)")
    # Locations that must never match, checked before the allowlist. Without
    # this, the bare token "remote" lets through "Remote - Lithuania" and
    # "Mexico City, Mexico (Remote)".
    loc_block = [b.lower() for b in cfg.get("location_blocklist", [])]
    signals = [s.lower() for s in cfg.get("relevance_signals", [])]
    scfg = {**scoring.DEFAULTS, **(cfg.get("scoring") or {})}
    if no_score:
        scfg["enabled"] = False
    if cap_override:
        scfg["max_per_run"] = int(cap_override)

    # The ATS sweep sees ~16,000 boards, so it needs a positive title filter,
    # not just the blocklist — otherwise it returns every job on the internet.
    # relevance_signals doubles as that filter here.
    sweep_terms = [s.lower() for s in
                   (cfg.get("ats", {}).get("sweep", {}).get("title_must_match")
                    or cfg.get("relevance_signals", []))]

    def ats_keep(job):
        title = (job.get("title") or "").lower()
        if not title:
            return False
        if scoring.hard_exclude(job, excludes):
            return False
        if sweep_terms and not any(t in title for t in sweep_terms):
            return False
        # Title-only temporal patterns ("Summer 2026 Internship") are worth
        # rejecting before an ATS description fetch. Deliberately title-only:
        # a full temporal scan costs ~0.6ms per KB of description, and Lever
        # and Ashby hand back descriptions for every job on every board, so
        # scanning them here would burn ~30s per sweep in the main thread on
        # jobs the title filter above has already rejected. Descriptions are
        # scanned once, later, on the few hundred survivors.
        if scoring.temporal_exclude({"title": job.get("title") or "",
                                     "posted": job.get("posted") or ""}):
            return False
        return location_ok(job, locations, loc_block, bool(only_loc))

    print("Fetching...")
    raw = []
    for label, fn, arg in [
        ("adzuna", sources.fetch_adzuna, cfg.get("adzuna", {})),
        ("usajobs", sources.fetch_usajobs, cfg.get("usajobs", {})),
        ("nyc", sources.fetch_nyc, cfg.get("nyc_open_data", {})),
        ("ats", ats.fetch_ats, cfg.get("ats", {})),
    ]:
        try:
            got = (fn(arg, ats_keep) if label == "ats" else fn(arg)) or []
        except Exception as e:
            print(f"  ! {label} blew up: {type(e).__name__}: {e}")
            got = []
        # With only three sources, a silent zero is a much bigger deal than it
        # was when a dead ATS slug just meant one missing company.
        flag = "  <-- returned nothing, check credentials" if not got else ""
        print(f"  {label:<12} {len(got):>4} raw{flag}")
        raw += got

    # --- free filters ------------------------------------------------------
    kept = [j for j in raw if not scoring.hard_exclude(j, excludes)]
    print(f"\n{len(raw)} fetched -> {len(kept)} past hard exclude "
          f"({len(raw) - len(kept)} dropped)")

    # A stated program/job start in the past or an expired application
    # deadline is a hard eligibility failure, not a low-fit score — but it is
    # decided by a regex reading prose, and a regex reading prose is wrong
    # sometimes. So stale postings are DEMOTED, not deleted: score_jobs marks
    # them weak for free without a model call, store.record keeps them so they
    # are never re-examined, and digest.py lists them as one-liners with the
    # parsed reason. That is the same contract every other low-scoring job
    # gets, and it is the only way a bad temporal read is ever visible.
    #
    # Title-only patterns ("Summer 2026 Internship") are still hard-dropped
    # during the ATS sweep, where precision is high and the alternative is
    # paying to fetch the description.
    stale = [(j, why) for j in kept
             if (why := scoring.temporal_exclusion_reason(j))]
    print(f"{len(kept) - len(stale)} temporally eligible "
          f"({len(stale)} stale/closed, kept as weak)")
    for j, why in stale[:5]:
        print(f"  stale: {j.get('title', '')[:52]} — {why}")
    if len(stale) > 5:
        print(f"  ...and {len(stale) - 5} more stale posting(s)")

    kept = [j for j in kept if location_ok(j, locations, loc_block, bool(only_loc))]
    print(f"{len(kept)} in target locations")

    if only_src:
        kept = [j for j in kept if j.get("source", "").lower() in only_src]
        print(f"{len(kept)} after --source {','.join(only_src)}")

    # Where the surviving jobs actually are — the fastest way to notice the
    # allowlist is admitting somewhere you don't want.
    from collections import Counter
    top = Counter((j.get("location") or "?").split(",")[0].strip()
                  for j in kept).most_common(8)
    if top:
        print("  by location: " + "  ".join(f"{k}={v}" for k, v in top))

    # --- dedupe BEFORE spending money -------------------------------------
    conn = store.connect()

    # Repair descriptions for still-open jobs stored by older versions. The
    # old 2k/4k caps could leave only company boilerplate and cut off the actual
    # duties. Dry runs remain read-only.
    if not dry_run:
        refreshed = store.refresh_descriptions(conn, kept)
        if refreshed:
            print(f"{refreshed} existing descriptions upgraded; "
                  f"run score.py --backfill --rescore to refresh their scores")

    fresh = store.filter_new(conn, kept)
    print(f"{len(fresh)} are new since last run")

    if not fresh:
        print("Nothing to send.")
        return

    # --- score -------------------------------------------------------------
    print(f"\nScoring {len(fresh)}...")
    scoring.score_jobs(fresh, scfg, cfg.get("facts_file", "resume.yaml"), signals)

    tiers = {}
    for j in fresh:
        tiers[j["tier"]] = tiers.get(j["tier"], 0) + 1
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(tiers.items())))

    capped = sum(1 for j in fresh if j.get("reason") == "over per-run scoring cap")
    if capped:
        print(f"\n  !! {capped} of {len(fresh)} were NEVER READ by the model "
              f"(max_per_run={scfg['max_per_run']}).")
        print(f"     The digest below is mostly unfiltered. Either raise the cap")
        print(f"     (--max-score {len(fresh)}) or tighten title_must_match.")

    # --- send --------------------------------------------------------------
    # Weak jobs are recorded but only listed as one-liners in the digest.
    mailable = [j for j in fresh if j["tier"] != "weak"]

    if dry_run:
        print("\n--- dry run, not recording ---")
        print(digest.build_text(fresh))
        return

    if mailable:
        digest.send(fresh)          # full list; digest handles the tiering
    else:
        print("  - email: nothing above the weak threshold, skipping send")

    store.append_to_sheet(fresh)
    store.record(conn, fresh)       # ALL of them, weak included
    print("Done.")


if __name__ == "__main__":
    main()
