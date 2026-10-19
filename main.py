#!/usr/bin/env python3
"""
jobscan - pull job listings from free APIs, keep only what's new, email a digest.

    python main.py                  # normal run, both tracks
    python main.py --track bridge   # rent work only (4 Adzuna calls, not 19)
    python main.py --track career   # research/policy/quant only
    python main.py --dry-run        # fetch + score, but don't record or email
    python main.py --no-score       # skip the model, keyword fallback (free)
    python main.py --no-email       # record everything, send no digest
    python main.py --no-enrich      # skip fetching missing descriptions

--track narrows a single run. The other track is not fetched where that is
possible and dropped before dedupe where it is not, so its postings are never
marked seen and the next full run still shows them.

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
    if not loc:
        return True
    # A leading "^" anchors the term to the start of the location string.
    # Needed because "new york" as a bare substring matches "Webster, New
    # York", six hours upstate, while "^new york" only matches a location
    # that opens with the city.
    return any(loc.startswith(p[1:]) if p.startswith("^") else p in loc
               for p in locations)


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
    # Record everything, send nothing. The scheduled GitHub Actions run uses
    # this: discovery has to stay on a schedule because postings expire, but
    # scoring happens locally on the GPU, so the cloud run has no scores worth
    # mailing. Without it the cron sends two all-unscored digests a day.
    no_email = "--no-email" in sys.argv
    # Enrichment makes one HTTP request per description-less posting.
    skip_enrich = "--no-enrich" in sys.argv

    # --location "new york,newark"  narrow to these for one run
    # --source ashby,workday        only these sources
    # --track bridge                only one track
    # --max-score 400               raise the per-run scoring cap
    only_loc = [x.strip().lower() for x in (_arg("--location") or "").split(",") if x.strip()]
    only_src = [x.strip().lower() for x in (_arg("--source") or "").split(",") if x.strip()]
    only_track = (_arg("--track") or "").strip().lower() or None
    cap_override = _arg("--max-score")

    if only_track and only_track not in scoring.TRACKS:
        sys.exit(f"--track must be one of {', '.join(scoring.TRACKS)}")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    excludes = [t.lower() for t in cfg.get("hard_exclude", [])]
    locations = only_loc or [l.lower() for l in cfg.get("location_allowlist", [])]

    # Bridge track: rent work, scored by a different rubric and filtered to a
    # commute rather than a list of metros. A warehouse job in Denver is not
    # an opportunity. --location overrides both allowlists, as before.
    bridge_cfg = cfg.get("bridge") or {}
    bridge_on = bool(bridge_cfg.get("enabled"))
    bridge_locations = only_loc or [l.lower() for l in
                                    bridge_cfg.get("location_allowlist", [])]
    (career_terms, bridge_terms, bridge_exclude,
     bridge_tolerated) = scoring.track_terms(cfg)
    professional = scoring.professional_titles(cfg)

    def track_of(job):
        """Classify, or None to drop. Career behaviour is unchanged."""
        if not bridge_on:
            return "career"
        return scoring.classify_track(job, career_terms, bridge_terms,
                                      bridge_exclude, bridge_tolerated,
                                      professional)

    def locations_for(track):
        return bridge_locations if track == "bridge" else locations

    if only_loc:
        print(f"(--location: restricted to {', '.join(only_loc)}; "
              f"config allowlist and blocklist ignored for this run)")
    if only_track:
        # The other track's postings are not fetched where that is possible,
        # and are dropped before dedupe where it is not. Dropping before
        # store.record is deliberate: they are never marked seen, so the next
        # full run still shows them. A narrowed run must not cost you jobs.
        print(f"(--track {only_track}: {only_track} postings only; the other "
              f"track is neither scored nor marked seen)")
    if only_track == "bridge" and not bridge_on:
        sys.exit("--track bridge, but bridge.enabled is false in config.yaml")
    # Locations that must never match, checked before the allowlist. Without
    # this, the bare token "remote" lets through "Remote - Lithuania" and
    # "Mexico City, Mexico (Remote)".
    loc_block = [b.lower() for b in cfg.get("location_blocklist", [])]
    signals = [s.lower() for s in cfg.get("relevance_signals", [])]
    scfg = scoring.merge_scoring_cfg(cfg)
    if no_score:
        scfg["enabled"] = False
    if cap_override:
        scfg["max_per_run"] = int(cap_override)

    # The ATS sweep sees ~16,000 boards, so it needs a positive title filter,
    # not just the blocklist — otherwise it returns every job on the internet.
    # relevance_signals doubles as that filter here.
    # Bridge titles join the sweep filter, otherwise the sweep never fetches a
    # warehouse posting in the first place. In practice Greenhouse/Lever/Ashby
    # are startup boards and contribute little here; Adzuna is the real pipe
    # for that track. Costs nothing to leave on.
    sweep_terms = [s.lower() for s in
                   (cfg.get("ats", {}).get("sweep", {}).get("title_must_match")
                    or cfg.get("relevance_signals", []))]
    if bridge_on:
        sweep_terms = sweep_terms + bridge_terms + bridge_tolerated
    if only_track == "bridge":
        sweep_terms = list(bridge_terms) + list(bridge_tolerated)
    elif only_track == "career":
        drop = set(bridge_terms) | set(bridge_tolerated)
        sweep_terms = [t for t in sweep_terms if t not in drop]

    def ats_keep(job):
        title = (job.get("title") or "").lower()
        if not title:
            return False
        if scoring.hard_exclude(job, excludes):
            return False
        if sweep_terms and not any(t in title for t in sweep_terms):
            return False
        track = track_of(job)
        if track is None:
            return False
        if only_track and track != only_track:
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
        return location_ok(job, locations_for(track), loc_block, bool(only_loc))

    print("Fetching...")
    raw = []
    for label, fn, arg in [
        ("adzuna", sources.fetch_adzuna, cfg.get("adzuna", {})),
        ("usajobs", sources.fetch_usajobs, cfg.get("usajobs", {})),
        ("nyc", sources.fetch_nyc, cfg.get("nyc_open_data", {})),
        ("ats", ats.fetch_ats, cfg.get("ats", {})),
    ]:
        # USAJOBS is searched by occupational series (Economist, Data Science)
        # and two analyst keywords. None of that produces bridge work, so a
        # bridge-only run skips the calls rather than filtering the results.
        if only_track == "bridge" and label == "usajobs":
            print(f"  {label:<12}    - skipped (--track bridge)")
            continue
        try:
            if label == "ats":
                got = fn(arg, ats_keep) or []
            elif label == "adzuna":
                # Skips the career queries and employer searches on a bridge
                # run, which is 15 of the 19 monthly-quota calls.
                got = fn(arg, only_track) or []
            else:
                got = fn(arg) or []
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

    # Track assignment. Title-only, so it is free and runs before anything is
    # scored. None means a bridge-shaped title carrying a ruled-out term (a
    # stockroom job at a clothing store); those are dropped, not demoted,
    # because no description rescues a retail floor.
    if bridge_on:
        tracked, ruled_out = [], []
        for j in kept:
            t = track_of(j)
            if t is None:
                ruled_out.append(j)
                continue
            j["track"] = t
            tracked.append(j)
        kept = tracked
        n_bridge = sum(1 for j in kept if j["track"] == "bridge")
        print(f"{len(kept)} tracked ({len(kept) - n_bridge} career, "
              f"{n_bridge} bridge; {len(ruled_out)} dropped as public-facing)")
        for j in ruled_out[:5]:
            print(f"  public-facing: {j.get('title', '')[:60]}")
        if only_track:
            before = len(kept)
            kept = [j for j in kept if j["track"] == only_track]
            print(f"  --track {only_track}: {before - len(kept)} other-track "
                  f"posting(s) set aside, not recorded")
    else:
        for j in kept:
            j["track"] = "career"

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

    # Location is gated per track. The career allowlist spans Chicago, Denver,
    # and Seattle because a research job is worth relocating for; the bridge
    # allowlist is the Morristown Line plus NYC plus remote, because a $19/hour
    # job you cannot reach without a car is not an opportunity.
    kept = [j for j in kept
            if location_ok(j, locations_for(j.get("track")), loc_block,
                           bool(only_loc))]
    n_bridge = sum(1 for j in kept if j.get("track") == "bridge")
    print(f"{len(kept)} in target locations "
          f"({len(kept) - n_bridge} career, {n_bridge} bridge)")

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

    # --- fill in missing descriptions --------------------------------------
    # Workday publishes no description past the first 25, some Lever boards
    # return an empty one, and jev refuses to score a posting it cannot read.
    # Rather than leave those as manual-review rows, go and get the text: the
    # platform's own API first, the rendered page second. Fetched text is
    # screened, because a sign-in wall is longer than min_desc_chars and would
    # otherwise be scored as if it were the job.
    #
    # NOT on a --no-score run. That is the scheduled cloud run, where there is
    # no local model to screen with, and storing unscreened text means a
    # sign-in wall gets scored later as though it were the job.
    gaps = [j for j in fresh
            if len((j.get("description") or "").strip()) < 200
            and (j.get("url") or "").startswith("http")]
    if gaps and no_score:
        print(f"{len(gaps)} posting(s) arrived without a description. "
              f"Run `python enrich.py` where the jev model is.")
    elif gaps and not skip_enrich:
        try:
            import enrich
            enrich.enrich(gaps, jev_cfg=cfg.get("jev"))
        except ImportError:
            print("  ! enrich.py not found; postings without a description "
                  "stay unscored")
        except Exception as e:
            print(f"  ! enrich failed: {type(e).__name__}: {e}")

    # --- score -------------------------------------------------------------
    print(f"\nScoring {len(fresh)}...")
    scoring.score_jobs(fresh, scfg, cfg.get("facts_file", "resume.yaml"), signals)

    for track in ("career", "bridge"):
        of_track = [j for j in fresh if (j.get("track") or "career") == track]
        if not of_track:
            continue
        tiers = {}
        for j in of_track:
            tiers[j["tier"]] = tiers.get(j["tier"], 0) + 1
        print(f"  {track:<7} " +
              "  ".join(f"{k}={v}" for k, v in sorted(tiers.items())))

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

    if no_email:
        print(f"  - email: skipped (--no-email); {len(fresh)} posting(s) "
              f"recorded. Score them with `python score.py --backfill`.")
    elif mailable:
        digest.send(fresh)          # full list; digest handles the tiering
    else:
        print("  - email: nothing above the weak threshold, skipping send")

    store.append_to_sheet(fresh)
    store.record(conn, fresh)       # ALL of them, weak included
    print("Done.")


if __name__ == "__main__":
    main()
