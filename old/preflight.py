#!/usr/bin/env python3
"""
preflight - verify everything is wired up before the first real run.

    python preflight.py

Checks config validity, every credential, and pokes each source with one small
request. Costs one Adzuna call and one USAJOBS call. Records nothing, sends
nothing, and never writes to seen.db.

Run this before `python main.py` the first time, and any time the digest
arrives emptier than expected.
"""

import os
import sys

import env
env.load()

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
problems = []


def report(status, label, detail=""):
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    if status == BAD:
        problems.append(label)


def check_imports():
    try:
        import requests, yaml  # noqa: F401
        report(OK, "requests + pyyaml installed")
    except ImportError as e:
        report(BAD, "dependencies", f"{e}. Run: pip install -r requirements.txt")
        return False
    try:
        import anthropic  # noqa: F401
        report(OK, "anthropic installed")
    except ImportError:
        report(WARN, "anthropic not installed",
               "scoring will fall back to keywords. pip install anthropic")
    return True


def check_config():
    import yaml
    if not os.path.exists("config.yaml"):
        report(BAD, "config.yaml", "not found. Are you in the jobscan folder?")
        return None
    try:
        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        report(BAD, "config.yaml", f"invalid YAML: {e}")
        return None

    stale = [k for k in ("title_keywords", "title_blocklist",
                         "title_prefix_blocklist", "greenhouse", "lever",
                         "ashby") if k in cfg]
    if stale:
        report(WARN, "config.yaml has obsolete keys",
               f"{', '.join(stale)} — no longer read, safe to delete")

    missing = [k for k in ("hard_exclude", "relevance_signals", "scoring",
                           "location_allowlist", "adzuna", "usajobs",
                           "nyc_open_data") if k not in cfg]
    if missing:
        report(BAD, "config.yaml missing sections", ", ".join(missing))
    else:
        report(OK, "config.yaml", f"{len(cfg.get('hard_exclude', []))} hard "
                                  f"excludes, "
                                  f"{len(cfg.get('location_allowlist', []))} "
                                  f"locations")

    s = cfg.get("scoring") or {}
    if s.get("maybe_at", 45) >= s.get("strong_at", 70):
        report(BAD, "scoring thresholds", "maybe_at must be below strong_at")

    bridge = cfg.get("bridge") or {}
    if bridge.get("enabled"):
        bs = bridge.get("scoring") or {}
        if bs.get("maybe_at", 45) >= bs.get("strong_at", 70):
            report(BAD, "bridge thresholds",
                   "bridge.scoring.maybe_at must be below strong_at")
        n_terms = len(bridge.get("title_must_match", []))
        n_excl = len(bridge.get("exclude", []))
        n_locs = len(bridge.get("location_allowlist", []))
        if not n_terms:
            report(BAD, "bridge track",
                   "enabled but title_must_match is empty; nothing will ever "
                   "be classified as bridge")
        elif not n_locs:
            report(WARN, "bridge track",
                   "no bridge.location_allowlist; falling back to the career "
                   "list, which spans the whole country")
        else:
            report(OK, "bridge track",
                   f"{n_terms} title terms, {n_excl} ruled out, "
                   f"{n_locs} commutable locations")
        # Overlap is a silent contradiction: a term in both lists produces a
        # title that can never be classified bridge. `tolerated` is exempt —
        # overriding an exclusion is the whole point of that list.
        both = (set(t.lower() for t in bridge.get("title_must_match", []))
                & set(t.lower() for t in bridge.get("exclude", [])))
        if both:
            report(BAD, "bridge track",
                   f"in both title_must_match and exclude: {', '.join(sorted(both))}")

        tol = [t.lower() for t in bridge.get("tolerated", [])]
        if tol:
            report(OK, "bridge tolerated",
                   f"{len(tol)} term(s) allowed past exclude: "
                   f"{', '.join(tol[:6])}")
    else:
        report(WARN, "bridge track", "disabled; career postings only")

    ad = cfg.get("adzuna", {})
    n_ad = (len(ad.get("queries", [])) + len(ad.get("employers", []))
            + len(ad.get("bridge_queries", [])))
    monthly = n_ad * 2 * 22
    status = OK if monthly < 900 else WARN
    report(status, "Adzuna quota",
           f"{n_ad} queries ({len(ad.get('bridge_queries', []))} bridge) "
           f"x 2 runs/day x 22 days = ~{monthly}/month of ~1000")
    return cfg


def check_facts(cfg):
    import yaml
    path = (cfg or {}).get("facts_file", "resume.yaml")
    if not os.path.exists(path):
        report(BAD, path, "not found — the scorer and tailor both need it")
        return
    data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    n_exp = len(data.get("experience", []))
    if not n_exp:
        report(BAD, path, "no experience entries; scores will be meaningless")
    else:
        report(OK, path, f"{n_exp} roles, {len(data.get('voice', []))} voice rules")


def check_env():
    groups = [
        ("Adzuna", ["ADZUNA_APP_ID", "ADZUNA_APP_KEY"], True),
        ("USAJOBS", ["USAJOBS_EMAIL", "USAJOBS_API_KEY"], True),
        ("Socrata (optional)", ["SOCRATA_APP_TOKEN"], False),
        ("Email", ["SMTP_USER", "SMTP_PASSWORD"], False),
        ("Scoring", ["ANTHROPIC_API_KEY"], False),
    ]
    for name, keys, required in groups:
        missing = [k for k in keys if not os.environ.get(k)]
        if not missing:
            report(OK, name, "credentials set")
        elif required:
            report(BAD, name, f"missing {', '.join(missing)}")
        else:
            report(WARN, name, f"missing {', '.join(missing)}")


def check_sources(cfg):
    import sources
    print()
    for label, fn, arg, key in [
        ("adzuna", sources.fetch_adzuna,
         {**cfg.get("adzuna", {}), "queries": ["analyst"], "employers": [],
          "results_per_page": 5}, "ADZUNA_APP_ID"),
        ("usajobs", sources.fetch_usajobs,
         {**cfg.get("usajobs", {}), "series": [], "keywords": ["analyst"],
          "results_per_page": 5, "max_pages": 1}, "USAJOBS_API_KEY"),
        ("nyc", sources.fetch_nyc,
         {**cfg.get("nyc_open_data", {}), "limit": 5}, None),
    ]:
        if key and not os.environ.get(key):
            report(WARN, f"{label} probe", "skipped, no credentials")
            continue
        try:
            got = fn(arg) or []
        except Exception as e:
            report(BAD, f"{label} probe", f"{type(e).__name__}: {e}")
            continue
        if not got:
            report(BAD, f"{label} probe", "returned 0 postings")
            continue
        q = {}
        for j in got:
            q[j.get("desc_quality", "?")] = q.get(j.get("desc_quality", "?"), 0) + 1
        report(OK, f"{label} probe",
               f"{len(got)} postings, descriptions: "
               + ", ".join(f"{v} {k}" for k, v in sorted(q.items())))
        print(f"         e.g. {got[0]['title'][:50]} — {got[0]['company'][:30]}")


def check_db():
    import store
    conn = store.connect()
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE score IS NOT NULL "
        "AND tier != 'unscored'").fetchone()[0]
    report(OK, "seen.db", f"{total} postings stored, {scored} model-scored")
    if total and not scored:
        print("         -> run: python score.py --backfill")


def main():
    print("jobscan preflight\n" + "-" * 60)
    if not check_imports():
        sys.exit(1)
    cfg = check_config()
    if cfg is None:
        sys.exit(1)
    check_facts(cfg)
    print()
    check_env()
    check_db()
    check_sources(cfg)

    print("-" * 60)
    if problems:
        print(f"{len(problems)} blocking problem(s): {', '.join(problems)}")
        sys.exit(1)
    print("All clear. Next: python main.py --no-score --dry-run")


if __name__ == "__main__":
    main()
