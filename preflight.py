#!/usr/bin/env python3
"""
preflight - verify everything is wired up before the first real run.

    python preflight.py
    python preflight.py --probe     # also make one tiny call to each model

Checks config validity, every credential, and pokes each source with one small
request. Costs one Adzuna call and one USAJOBS call. Records nothing, sends
nothing, and never writes to seen.db.

--probe sends one short request to every model the config uses (rubric, jev
per track, tailor), so a hosted model costs a fraction of a cent. For jev it
reports which readout actually worked: logprobs, or the verbalized fallback.

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

    # jev block, when either track is scored by jev. Which MODEL runs it is
    # checked in check_models().
    import score
    try:
        scfg = score.merge_scoring_cfg(cfg)
    except ValueError as e:
        report(BAD, "scoring.method", str(e))
        scfg = None
    methods = ({t: score.method_for(scfg, t) for t in score.TRACKS}
               if scfg else {})
    if methods:
        report(OK, "scoring methods",
               ", ".join(f"{k}={v}" for k, v in methods.items()))
    if "jev" in methods.values():
        try:
            import jev
            jcfg = jev.merged(cfg.get("jev"))
            import llm
            try:
                kinds = {llm.resolve(cfg, "jev", t)["kind"]
                         for t, m in methods.items() if m == "jev"}
            except llm.LLMError:
                kinds = {"jevk5"}       # reported in check_models
            problems = jev.validate(jcfg, None if kinds == {"jevk5"} else 16)
            if problems:
                for p in problems:
                    report(BAD, "jev config", p)
            else:
                n = len(jcfg["contact"]["bands"])
                nc = len(jcfg["career_fit"]["bands"])
                overrides = sorted(set(cfg.get("jev") or {})
                                   & {"contact", "gate", "fee", "thresholds"})
                report(OK, "jev config",
                       f"{n} contact bands, {nc} career bands, "
                       f"thresholds valid"
                       + (f"; overriding {', '.join(overrides)}" if overrides
                          else "; all judgment at built-in defaults"))
        except ImportError:
            report(BAD, "jev config",
                   "a track's scoring method is 'jev' but jev.py is missing")

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
    n_voice = len(data.get("voice", []))
    if not n_exp:
        report(BAD, path, "no experience entries; scores will be meaningless")
    else:
        report(OK, path, f"{n_exp} roles, {n_voice} voice rules")
    # load_facts() pops `voice` and joins it. With the block missing that
    # join returns "", every prompt gets an empty VOICE RULES section, and
    # nothing anywhere says so. The drafts just quietly come back generic.
    if not n_voice:
        report(BAD, path, "no `voice:` block — every draft would be written "
                          "with no voice rules at all, silently")


def _roles_in_use(cfg):
    """[(label, role, track)] for every model this config will call."""
    import score
    try:
        scfg = score.merge_scoring_cfg(cfg)
    except ValueError:
        return []
    out = []
    for t in score.TRACKS:
        m = score.method_for(scfg, t)
        if m == "jev":
            out.append((f"jev ({t})", "jev", t))
        if m in ("jev", "rubric"):
            # jev falls back to the rubric, so that model matters too.
            out.append((f"rubric ({t})" + ("" if m == "rubric"
                                           else ", fallback"), "rubric", t))
    for step in ("analyze", "letter", "resume"):
        out.append((f"tailor {step}", f"tailor_{step}", None))
    return out


def check_models(cfg, probe=False):
    """Resolve, sanity-check and optionally call every configured model."""
    import llm
    print()
    seen = {}
    for label, role, track in _roles_in_use(cfg):
        try:
            p = llm.resolve(cfg, role, track)
        except llm.LLMError as e:
            report(BAD, f"model: {label}", str(e))
            continue
        fallback = "fallback" in label
        problem = llm.check(p)
        if problem:
            # A missing fallback is a warning: the run still works.
            report(WARN if (fallback or role.startswith("tailor")) else BAD,
                   f"model: {label}", f"{llm.describe(p)}: {problem}")
            continue
        key = (p["name"], role == "jev")
        if key in seen:
            report(OK, f"model: {label}", f"{p['name']} (checked above)")
            continue
        seen[key] = p
        detail = llm.describe(p)
        srv = p.get("server") or {}
        for k in ("exe", "model_path"):
            if srv.get(k) and not os.path.exists(srv[k]):
                report(BAD, f"model: {label}", f"server {k} not found: {srv[k]}")
                break
        else:
            if llm._is_local(p) and llm._health_url(p):
                up = llm._alive(llm._health_url(p))
                if up:
                    detail += "; server answering"
                elif srv.get("exe") or srv.get("command"):
                    detail += "; not running, will be started on demand"
                else:
                    report(WARN if fallback else BAD, f"model: {label}",
                           f"{detail}; nothing answering at "
                           f"{llm._health_url(p)} and no server: block")
                    continue
            report(OK, f"model: {label}", detail)
        if probe:
            _probe(label, role, p)


def _probe(label, role, p):
    import time
    import llm
    t0 = time.perf_counter()
    try:
        if role == "jev":
            if not llm.ensure_ready(p, quiet=True)[0]:
                report(BAD, f"probe: {label}", "server did not start")
                return
            d = llm.decider(p)
            r = d.decide(
                "Warehouse associate, night shift. Pick and pack orders. No "
                "experience needed. $19/hr.",
                {"type": "noul", "instructions": "Does this job involve "
                 "selling to customers?",
                 "criteria": {"true": "Yes.", "false": "No."}})
            how = getattr(d, "readout", "jevk5")
            if how == "auto":
                how = "logprobs"     # auto only stays auto when they worked
            note = f"; logprobs unavailable: {d.switched}" if getattr(
                d, "switched", None) else ""
            report(OK, f"probe: {label}",
                   f"readout={how}, P(yes)={r['noul']:.2f} (expect low), "
                   f"{time.perf_counter() - t0:.1f}s{note}")
        else:
            text = llm.complete(p, "Reply with exactly the word: ready",
                                "Are you there?", 20)
            report(OK, f"probe: {label}",
                   f"replied {text.strip()[:30]!r} in "
                   f"{time.perf_counter() - t0:.1f}s")
    except Exception as e:
        report(BAD, f"probe: {label}", f"{type(e).__name__}: {e}")


def check_env():
    groups = [
        ("Adzuna", ["ADZUNA_APP_ID", "ADZUNA_APP_KEY"], True),
        ("USAJOBS", ["USAJOBS_EMAIL", "USAJOBS_API_KEY"], True),
        ("Socrata (optional)", ["SOCRATA_APP_TOKEN"], False),
        ("Email", ["SMTP_USER", "SMTP_PASSWORD"], False),
    ]
    # Model keys are checked per model in check_models(), against the
    # providers config.yaml actually uses.
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
    check_models(cfg, probe="--probe" in sys.argv)
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
