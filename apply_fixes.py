#!/usr/bin/env python3
"""
apply_fixes.py - patch jev.py, score.py, main.py and config.yaml in place.

    python apply_fixes.py            # check only: says what would change
    python apply_fixes.py --write    # apply; originals saved as *.bak
    python apply_fixes.py --restore  # put the *.bak originals back

Already applied an earlier version? Rerunning is safe; then run
`python fit.py --write-config` to bring the jev.fit block up to date.

Run it from the jobscan folder, with fit.py beside it. gates.py from the
earlier version is no longer used and can be deleted. Every edit is an
exact-text replacement that must match once. If ANY edit fails to match,
nothing is written, so a file that differs from the copy these edits were
written against is left alone rather than half-patched.
"""
import shutil
import sys

EDITS = {}

try:
    import yaml
    import fit
except ImportError as e:
    sys.exit(f"needs PyYAML and fit.py in this folder: {e}")

# The whole of fit.DEFAULTS, written into config.yaml so every question,
# band and label that runs is visible and editable there.
FIT_YAML = (
    "  # ------------------------------------------------------------------\n"
    "  # resume mode (fit.py). The model is shown your record above each\n"
    "  # posting and the score is the PRODUCT of its answers:\n"
    "  #   career  match x P(credentials) x P(experience) x P(open now)\n"
    "  #   bridge  contact x P(can get it) x P(no fee) x P(open now)\n"
    "  # No caps or penalties: `thresholds:` below is only used by\n"
    "  # mode: bands, the previous scorer. Edit the wording here; the\n"
    "  # band ORDER in career_match is lowest to highest.\n"
    "  #   python fit.py --candidate   the text the model reads about you\n"
    "  #   python fit.py --check       every factor, plus a control posting\n"
    "  # ------------------------------------------------------------------\n"
    "  mode: resume          # resume | bands\n"
    "  # Characters of each posting jev reads (was scoring.desc_chars, 700).\n"
    "  desc_chars: 2000\n"
    "  # Write your own summary here to replace the one generated from\n"
    "  # resume.yaml:\n"
    "  # candidate: >-\n"
    "  #   CANDIDATE RECORD. B.A. in Economics ...\n"
    "  fit:\n"
    + "".join("    " + line + "\n" for line in yaml.safe_dump(
        fit.DEFAULTS, sort_keys=False, default_flow_style=False,
        width=76, allow_unicode=True).splitlines()))


DONE_IF = {}


def edit(path, old, new, why, done_if=None):
    """`done_if`: text whose presence means this edit already happened in
    some version, so it is never applied twice (fit.py --write-config
    refreshes that block afterwards)."""
    EDITS.setdefault(path, []).append((old, new, why))
    if done_if:
        DONE_IF[(path, why)] = done_if


# ======================================================================
# jev.py: three hooks. The band scorer is left exactly as it was and
# still runs when jev.mode is "bands".
# ======================================================================
edit("jev.py",
     "import copy\nfrom datetime import date\nimport re\nimport time\n",
     "import copy\nfrom datetime import date\nimport re\nimport time\n\n"
     "import fit\n",
     "import fit")

edit("jev.py",
     "    t = cfg.get(\"thresholds\") or {}\n"
     "    if t.get(\"gate_soft\", 0)",
     "    problems += fit.validate(cfg)\n"
     "    t = cfg.get(\"thresholds\") or {}\n"
     "    if t.get(\"gate_soft\", 0)",
     "preflight validates jev.fit")

edit("jev.py",
     "    score_q, a_q, b_q, t_q = questions(cfg, track)\n",
     "    score_q, a_q, b_q, t_q = questions(cfg, track)\n"
     "    # resume mode: the candidate record goes above every posting and the\n"
     "    # score is the product of the model's answers (fit.py).\n"
     "    resume_mode = cfg.get(\"mode\", \"resume\") == \"resume\"\n"
     "    cand = fit.candidate(cfg, facts_path) if resume_mode else \"\"\n"
     "    if resume_mode and not cand:\n"
     "        print(f\"  ! jev: {facts_path} unreadable; using the band scorer\")\n"
     "        resume_mode = False\n",
     "build the candidate text once per run")

edit("jev.py",
     "    if resume_mode and not cand:\n"
     "        print(f\"  ! jev: {facts_path} unreadable; using the band scorer\")\n"
     "        resume_mode = False\n",
     "    if resume_mode and not cand:\n"
     "        print(f\"  ! jev: {facts_path} unreadable; using the band scorer\")\n"
     "        resume_mode = False\n"
     "    if resume_mode and track == \"career\":\n"
     "        # New applications in your tracker? Refit before scoring.\n"
     "        fit.maybe_refit(cfg, cand, model)\n",
     "refit from the tracker before a career run")

edit("jev.py",
     "        text = (f\"{j.get('title','')} at {j.get('company','')}, \"\n"
     "                f\"{j.get('location','')}. {body}\")\n"
     "        c = model.decide(text, score_q)\n",
     "        if resume_mode:\n"
     "            return fit.score_one(model, j, body, cfg, track, cand), False\n"
     "        text = (f\"{j.get('title','')} at {j.get('company','')}, \"\n"
     "                f\"{j.get('location','')}. {body}\")\n"
     "        c = model.decide(text, score_q)\n",
     "score_track uses fit.score_one in resume mode")

edit("jev.py",
     "    score_q, a_q, b_q, t_q = questions(cfg, track, today)\n"
     "    try:\n",
     "    if cfg.get(\"mode\", \"resume\") == \"resume\" and fit.candidate(cfg):\n"
     "        # Same scale as the digest, on the full posting.\n"
     "        try:\n"
     "            r = fit.score_one(model, {\"description\": text}, text, cfg,\n"
     "                              track, fit.candidate(cfg), today)\n"
     "        except Exception as e:\n"
     "            return None, False, f\"{type(e).__name__}: {e}\"\n"
     "        return r[\"score\"], r[\"review\"], r[\"reason\"]\n"
     "    score_q, a_q, b_q, t_q = questions(cfg, track, today)\n"
     "    try:\n",
     "prescan (tailor.py) uses the same scorer")

# ======================================================================
# score.py
# ======================================================================
edit("score.py",
     "def track_terms(cfg):\n",
     "def bridge_never(cfg):\n"
     "    \"\"\"Title words that are never bridge work (bridge.never).\"\"\"\n"
     "    return tuple(s.lower() for s in\n"
     "                 ((cfg.get(\"bridge\") or {}).get(\"never\") or []))\n"
     "\n\n"
     "def track_terms(cfg):\n",
     "bridge_never()")

edit("score.py",
     "                   bridge_tolerated=(), professional=None):\n",
     "                   bridge_tolerated=(), professional=None, never=()):\n",
     "classify_track takes never=")

edit("score.py",
     "    if any(t in title for t in bridge_terms):\n"
     "        return None if ruled_out else \"bridge\"\n",
     "    if any(t in title for t in bridge_terms):\n"
     "        # A professional title that happens to contain a bridge word is a\n"
     "        # career question: \"Software Engineer, Backend, Accounts\n"
     "        # Receivable\" reached the bridge digest at 87 because \"engineer\"\n"
     "        # only ever protected a title from bridge.exclude, it never moved\n"
     "        # it off the track.\n"
     "        if any(t in title for t in (professional or _PROFESSIONAL_TITLE)):\n"
     "            return \"career\"\n"
     "        if any(t in title for t in never):\n"
     "            return None\n"
     "        return None if ruled_out else \"bridge\"\n",
     "professional titles leave the bridge track; bridge.never drops")

edit("score.py",
     "                                         bridge_exclude, bridge_tolerated,\n"
     "                                         professional_titles(cfg))\n",
     "                                         bridge_exclude, bridge_tolerated,\n"
     "                                         professional_titles(cfg),\n"
     "                                         bridge_never(cfg))\n",
     "backfill passes never")

edit("score.py",
     "                          bridge_tolerated, professional_titles(cfg))\n",
     "                          bridge_tolerated, professional_titles(cfg),\n"
     "                          bridge_never(cfg))\n",
     "--test passes never")

edit("score.py",
     "            focus=lambda j: _focus_description(\n"
     "                j.get(\"description\") or \"\", cfg[\"desc_chars\"]),\n",
     "            focus=lambda j: _focus_description(\n"
     "                j.get(\"description\") or \"\",\n"
     "                int((cfg.get(\"jev\") or {}).get(\"desc_chars\")\n"
     "                    or cfg[\"desc_chars\"])),\n",
     "jev reads jev.desc_chars, not the rubric's 700")

# ======================================================================
# main.py
# ======================================================================
edit("main.py",
     "import sys\nimport yaml\n",
     "import re\nimport sys\nimport yaml\n",
     "import re")

edit("main.py",
     "import digest\nimport score as scoring\n",
     "import digest\nimport fit\nimport score as scoring\n",
     "import fit")

edit("main.py",
     "    professional = scoring.professional_titles(cfg)\n",
     "    professional = scoring.professional_titles(cfg)\n"
     "    never = scoring.bridge_never(cfg)\n",
     "read bridge.never")

edit("main.py",
     "                                      bridge_exclude, bridge_tolerated,\n"
     "                                      professional)\n",
     "                                      bridge_exclude, bridge_tolerated,\n"
     "                                      professional, never)\n",
     "main passes never")

edit("main.py",
     "def location_ok(job, locations, blocked=(), explicit=False):\n",
     "def _loc_hit(term, loc):\n"
     "    \"\"\"\n"
     "    One allowlist/blocklist term against a lowercased location.\n"
     "\n"
     "    \"re:...\" is a regular expression, for terms that are only safe with\n"
     "    word boundaries: \"uk\" as a bare substring matches Milwaukee. \"^...\"\n"
     "    anchors to the start. Anything else is a substring, as before.\n"
     "    \"\"\"\n"
     "    if term.startswith(\"re:\"):\n"
     "        return bool(re.search(term[3:], loc))\n"
     "    if term.startswith(\"^\"):\n"
     "        return loc.startswith(term[1:])\n"
     "    return term in loc\n"
     "\n\n"
     "def location_ok(job, locations, blocked=(), explicit=False):\n",
     "regex terms in location lists")

edit("main.py",
     "    if not explicit and loc and any(b in loc for b in blocked):\n",
     "    if not explicit and loc and any(_loc_hit(b, loc) for b in blocked):\n",
     "blocklist uses _loc_hit")

edit("main.py",
     "    return any(loc.startswith(p[1:]) if p.startswith(\"^\") else p in loc\n"
     "               for p in locations)\n",
     "    return any(_loc_hit(p, loc) for p in locations)\n",
     "allowlist uses _loc_hit")

edit("main.py",
     "          f\"({len(raw) - len(kept)} dropped)\")\n",
     "          f\"({len(raw) - len(kept)} dropped)\")\n"
     "\n"
     "    # Employers that are not employers: survey-panel and work-at-home\n"
     "    # lead farms that post one title to every city. The fee question\n"
     "    # cannot catch them because the posting never mentions the fee.\n"
     "    company_block = [c.lower() for c in cfg.get(\"company_blocklist\", [])]\n"
     "    if company_block:\n"
     "        before = len(kept)\n"
     "        kept = [j for j in kept\n"
     "                if not any(c in (j.get(\"company\") or \"\").lower()\n"
     "                           for c in company_block)]\n"
     "        if before - len(kept):\n"
     "            print(f\"{before - len(kept)} dropped by company_blocklist\")\n",
     "company blocklist")

edit("main.py",
     "    print(f\"\\nScoring {len(fresh)}...\")\n"
     "    scoring.score_jobs(fresh, scfg, cfg.get(\"facts_file\", \"resume.yaml\"), signals)\n",
     "    # One employer, one title, several cities: score it once.\n"
     "    reps, dup_pairs = fit.collapse_duplicates(fresh)\n"
     "    if dup_pairs:\n"
     "        print(f\"{len(dup_pairs)} same-employer same-title duplicate(s) \"\n"
     "              f\"collapsed\")\n"
     "    print(f\"\\nScoring {len(reps)}...\")\n"
     "    scoring.score_jobs(reps, scfg, cfg.get(\"facts_file\", \"resume.yaml\"), signals)\n"
     "    fit.copy_to_duplicates(dup_pairs)\n",
     "collapse multi-city duplicates before scoring")

edit("config.yaml",
     "  - \"crawling engineer\"\n  - biostatistician\n  - postdoctoral\n",
     "  - \"crawling engineer\"\n  - biostatistician\n  - postdoctoral\n"
     "  - sourcer            # \"Executive Talent Sourcer & Researcher\" is recruiting\n",
     "hard_exclude: sourcer")

edit("config.yaml",
     "    - \"field interviewer\"\n    - \"leasing consultant\"\n",
     "    - \"field interviewer\"\n    - \"leasing consultant\"\n"
     "\n"
     "  # Never bridge work, even though the title contains a bridge word. Only\n"
     "  # applied to titles that matched title_must_match, so it cannot remove\n"
     "  # anything from the career track. Titles with a professional word\n"
     "  # (engineer, consultant, counsel...) are not listed here: those are sent\n"
     "  # to the career scorer instead of being dropped.\n"
     "  never:\n"
     "    - manager\n"
     "    - supervisor\n"
     "    - \"account executive\"\n"
     "    - sales\n"
     "    - strategy\n"
     "    - strategic\n"
     "    - \"business development\"\n"
     "    - instructor\n"
     "    - \"information system\"      # ISSO matched \"security officer\"\n"
     "    - \"data warehouse\"\n",
     "bridge.never")

edit("config.yaml",
     "    - assembler\n    - packaging\n    - manufacturing\n",
     "    - assembler\n"
     "    # Bare \"packaging\" and \"manufacturing\" pulled in Packaging Engineer,\n"
     "    # Manufacturing Quality Engineer and Strategic Account Executive -\n"
     "    # Manufacturing from the startup boards.\n"
     "    - packager\n"
     "    - \"packaging associate\"\n"
     "    - \"packaging operator\"\n"
     "    - \"manufacturing associate\"\n"
     "    - \"manufacturing technician\"\n"
     "    - \"manufacturing operator\"\n",
     "narrow packaging/manufacturing")

edit("config.yaml",
     "    - \"data operations\"\n    - \"operations associate\"\n    - \"operations assistant\"\n",
     "    - \"data operations\"\n"
     "    # \"operations associate\" removed: on Greenhouse/Lever/Ashby it means\n"
     "    # Strategy & Ops, Product Ops, Legal Ops, People Ops at $100K+, which\n"
     "    # were about thirty of the rows in one digest. \"Fulfillment Operations\n"
     "    # Associate\" still matches on \"fulfillment\". Put the line back if you\n"
     "    # miss it.\n"
     "    # - \"operations associate\"\n"
     "    - \"operations assistant\"\n",
     "drop 'operations associate' from the bridge title filter")

edit("config.yaml",
     "  - \"global (remote)\"\n  - worldwide\n  - anywhere\n",
     "  - \"global (remote)\"\n  - worldwide\n  - anywhere\n"
     "  # \"re:\" entries are regular expressions. \"uk\" has to be one: as a bare\n"
     "  # substring it matches Milwaukee.\n"
     "  - \"re:\\\\buk\\\\b\"\n"
     "  - england\n"
     "  - scotland\n"
     "  - europe\n"
     "  - \"sri lanka\"\n"
     "  - bogot              # \"Bogotá, Bogotá D.C.\" passed on the allowlist's \"d.c.\"\n"
     "  - latam\n"
     "  - israel\n"
     "  - \"tel aviv\"\n"
     "  - nigeria\n"
     "  - kenya\n"
     "  - \"south africa\"\n"
     "  - egypt\n"
     "  - vietnam\n"
     "  - china\n"
     "  - \"hong kong\"\n"
     "  - taiwan\n"
     "  - \"new zealand\"\n"
     "  - italy\n"
     "  - sweden\n"
     "  - switzerland\n"
     "  - belgium\n"
     "  - denmark\n"
     "  - norway\n"
     "  - finland\n"
     "  - turkey\n"
     "  - dubai\n"
     "  - \"saudi\"\n"
     "  - indonesia\n"
     "  - malaysia\n"
     "  - thailand\n"
     "  - chile\n"
     "  - peru\n"
     "  - \"costa rica\"\n"
     "  - bangalore\n"
     "  - bengaluru\n"
     "\n"
     "# Employers dropped before scoring, matched as a substring of the company\n"
     "# name. For lead farms that post \"Remote Data Entry Clerk - Work at Home\"\n"
     "# and \"Research Panelist\" to every city: the fee question cannot see\n"
     "# them, because the posting never mentions the fee.\n"
     "company_blocklist:\n"
     "  - usasurveyjob\n",
     "location blocklist additions + company_blocklist")

edit("config.yaml",
     "  min_desc_chars: 120\n  profile_chars: 1200\n",
     "  min_desc_chars: 120\n  profile_chars: 1200\n" + FIT_YAML,
     "jev.mode, jev.desc_chars and the jev.fit questions",
     done_if="\n  fit:\n")


def main():
    if "--restore" in sys.argv:
        import os
        for path in EDITS:
            if os.path.exists(path + ".bak"):
                shutil.copyfile(path + ".bak", path)
                print(f"  restored {path} from {path}.bak")
        return
    write = "--write" in sys.argv
    plan, failed = {}, 0
    for path, edits in EDITS.items():
        try:
            raw = open(path, encoding="utf-8", newline="").read()
        except OSError as e:
            print(f"! {path}: {e}")
            failed += 1
            continue
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        for old, new, why in edits:
            n = text.count(old)
            # Checked first: most edits insert around their anchor, so the
            # anchor is still there afterwards and a second run would
            # otherwise apply the edit twice.
            marker = DONE_IF.get((path, why))
            if new in text or (marker and marker in text):
                print(f"  done {path}: {why} (already applied)")
            elif n == 1:
                text = text.replace(old, new)
                print(f"  ok   {path}: {why}")
            else:
                print(f"  FAIL {path}: {why} (anchor found {n} times)")
                failed += 1
        plan[path] = text.replace("\n", "\r\n") if crlf else text
    if failed:
        sys.exit(f"\n{failed} problem(s). Nothing was written.")
    if not write:
        print("\nAll edits match. Re-run with --write to apply.")
        return
    import os
    for path, text in plan.items():
        if not os.path.exists(path + ".bak"):   # never overwrite the original
            shutil.copyfile(path, path + ".bak")
        open(path, "w", encoding="utf-8", newline="").write(text)
    print("\nWritten. Originals saved as *.bak. Next: python preflight.py, "
          "then\n  python score.py --backfill --rescore --scorer jev")


if __name__ == "__main__":
    main()
