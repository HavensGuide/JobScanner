#!/usr/bin/env python3
"""
slugs - find and verify ATS board slugs for employers you actually want.

The public Greenhouse/Lever/Ashby/Workday job APIs are free and return full
descriptions, but none of them publishes a directory of company board tokens.
That gap is why the ATS sources were dropped. It is now filled by a
Common-Crawl-derived directory that is regenerated daily:

    https://github.com/Feashliaa/job-board-aggregator  (data/*_companies.json)

This tool downloads those lists, matches them against employer names you care
about, PROBES each candidate against the live API to throw out false matches,
and prints a block you paste into config.yaml.

    python slugs.py --refresh                     # download / update the lists
    python slugs.py --find "Urban Institute" "RAND Corporation"
    python slugs.py --find-file targets.txt       # one employer per line
    python slugs.py --verify greenhouse:urban     # probe one slug directly

Probing is the important half. Substring matching alone produces junk:
"mathematica" matches the slug "matic", "brookings" matches "kings". Only a
live 200 with a plausible company name proves a slug is real.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

CACHE = Path(".slug-cache")
BASE = ("https://raw.githubusercontent.com/Feashliaa/job-board-aggregator"
        "/main/data")
FILES = {
    "greenhouse": "greenhouse_companies.json",
    "lever": "lever_companies.json",
    "ashby": "ashby_companies.json",
    "workday": "workday_companies.json",
}
UA = {"User-Agent": "jobscan-slugfinder/1.0 (personal job search)"}
TIMEOUT = 20


# ---------------------------------------------------------------------------
def refresh():
    CACHE.mkdir(exist_ok=True)
    for ats, fname in FILES.items():
        url = f"{BASE}/{fname}"
        try:
            r = requests.get(url, timeout=60, headers=UA)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ! {ats}: {type(e).__name__}: {e}")
            continue
        (CACHE / fname).write_text(json.dumps(data), encoding="utf-8")
        print(f"  {ats:<11} {len(data):>6} slugs")
    print(f"\nCached in {CACHE}/")


def load():
    if not CACHE.exists():
        sys.exit("No cache. Run: python slugs.py --refresh")
    out = {}
    for ats, fname in FILES.items():
        p = CACHE / fname
        if p.exists():
            out[ats] = json.loads(p.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Words that carry no identifying signal — dropped before matching so
# "RAND Corporation" can match the slug "rand".
NOISE = {"inc", "llc", "corp", "corporation", "company", "co", "the", "group",
         "associates", "institute", "institution", "foundation", "center",
         "centre", "university", "college", "school", "of", "and", "for",
         "research", "international", "global", "partners", "consulting"}


def core(name):
    toks = [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]
    keep = [t for t in toks if t not in NOISE]
    return "".join(keep or toks)


def candidates(target, directories, max_per_ats=6):
    """Slugs that plausibly correspond to `target`, before verification."""
    n, c = norm(target), core(target)
    found = []
    for ats, slugs in directories.items():
        hits = []
        for raw in slugs:
            slug = raw.split("|")[0] if ats == "workday" else raw
            ns = norm(slug)
            if not ns:
                continue
            # exact on either the full name or its distinctive core
            if ns == n or ns == c:
                hits.append((0, raw))
            # containment, but only when the shorter side is long enough to
            # be meaningful. Below ~6 chars this generates pure noise.
            elif len(c) >= 6 and c in ns:
                hits.append((1, raw))
            elif len(ns) >= 6 and ns in c:
                hits.append((2, raw))
        hits.sort(key=lambda h: (h[0], len(h[1])))
        found += [(ats, raw) for _, raw in hits[:max_per_ats]]
    return found


# ---------------------------------------------------------------------------
def probe(ats, raw):
    """
    Hit the live board. Returns (ok, n_jobs, label) — label is whatever the
    API calls the company, which is how you spot a false positive.
    """
    try:
        if ats == "greenhouse":
            r = requests.get(
                f"https://boards-api.greenhouse.io/v1/boards/{raw}/jobs",
                timeout=TIMEOUT, headers=UA)
            if r.status_code != 200:
                return False, 0, ""
            d = r.json()
            return True, len(d.get("jobs", [])), raw

        if ats == "lever":
            r = requests.get(f"https://api.lever.co/v0/postings/{raw}?mode=json",
                             timeout=TIMEOUT, headers=UA)
            if r.status_code != 200:
                return False, 0, ""
            d = r.json()
            return isinstance(d, list), len(d) if isinstance(d, list) else 0, raw

        if ats == "ashby":
            r = requests.get(
                f"https://api.ashbyhq.com/posting-api/job-board/{raw}",
                timeout=TIMEOUT, headers=UA)
            if r.status_code != 200:
                return False, 0, ""
            d = r.json()
            return True, len(d.get("jobs", [])), d.get("name") or raw

        if ats == "workday":
            parts = raw.split("|")
            if len(parts) != 3:
                return False, 0, ""
            tenant, dc, site = parts
            url = (f"https://{tenant}.{dc}.myworkdayjobs.com"
                   f"/wday/cxs/{tenant}/{site}/jobs")
            r = requests.post(url, json={"appliedFacets": {}, "limit": 20,
                                         "offset": 0, "searchText": ""},
                              timeout=TIMEOUT,
                              headers={**UA, "Content-Type": "application/json"})
            if r.status_code != 200:
                return False, 0, ""
            d = r.json()
            return True, d.get("total", 0), f"{tenant} ({site})"
    except Exception:
        return False, 0, ""
    return False, 0, ""


# ---------------------------------------------------------------------------
ATS_ORDER = ("greenhouse", "lever", "ashby", "workday")


def write_config(confirmed, path, replace=False):
    """
    Splice verified slugs into the `ats:` block of config.yaml.

    Rewrites only that block and only its platform lists, so the surrounding
    comments survive. Default is merge, so hand-added slugs are never lost;
    --replace rewrites the lists to exactly what was just verified.
    """
    import yaml
    p = Path(path)
    if not p.exists():
        print(f"  ! {path} not found; printing instead")
        return False
    text = p.read_text(encoding="utf-8")

    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if l.rstrip() == "ats:"]
    if len(idx) != 1:
        print(f"  ! could not find a single top-level `ats:` block in {path}")
        return False
    start = idx[0]

    # the block runs until the next top-level key (a non-indented, non-comment
    # line that isn't blank)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l and not l[0].isspace() and not l.lstrip().startswith("#"):
            end = i
            break

    existing = yaml.safe_load("\n".join(lines[start:end])) or {}
    existing = existing.get("ats") or {}

    block = ["ats:"]
    for ats in ATS_ORDER:
        new = confirmed.get(ats) or []
        old = [] if replace else [v for v in (existing.get(ats) or [])]
        merged, seen = [], set()
        for slug, label in [(v, "") for v in old] + new:
            if slug in seen:
                continue
            seen.add(slug)
            merged.append((slug, label))
        if not merged:
            block.append(f"  {ats}: []")
            continue
        block.append(f"  {ats}:")
        for slug, label in merged:
            comment = f"    # {label}" if label and label != slug else ""
            block.append(f'    - "{slug}"{comment}')

    # keep any non-list settings (workday_search, caps, ...)
    for k, v in existing.items():
        if k not in ATS_ORDER:
            block.append(f"  {k}: {json.dumps(v) if not isinstance(v, str) else v}")

    p.write_text("\n".join(lines[:start] + block + lines[end:]) + "\n",
                 encoding="utf-8")
    total = sum(len(confirmed.get(a) or []) for a in ATS_ORDER)
    print(f"  wrote {total} verified board(s) into {path} "
          f"({'replaced' if replace else 'merged'})")
    return True


def cmd_find(targets, args):
    dirs = load()
    print(f"Loaded {sum(len(v) for v in dirs.values()):,} slugs across "
          f"{len(dirs)} platforms.\n")

    confirmed = {k: [] for k in FILES}
    for t in targets:
        cands = candidates(t, dirs, args.max_per_ats)
        if not cands:
            print(f"{t}\n    no candidates in the directory")
            continue
        print(f"{t}")
        any_ok = False
        for ats, raw in cands:
            if args.no_probe:
                print(f"    ?  {ats}:{raw}  (unverified)")
                confirmed[ats].append(raw)
                continue
            ok, n, label = probe(ats, raw)
            time.sleep(args.delay)
            if ok and n >= args.min_jobs:
                any_ok = True
                confirmed[ats].append((raw, t))
                print(f"    OK {ats}:{raw:<32} {n:>4} jobs   [{label}]")
            elif ok:
                print(f"    -  {ats}:{raw:<32} {n:>4} jobs   (below "
                      f"--min-jobs {args.min_jobs})")
            else:
                print(f"    x  {ats}:{raw:<32}       dead / not a board")
        if not any_ok and not args.no_probe:
            print("    -> nothing verified; this employer is probably on "
                  "Workday under a different tenant, or on iCIMS/Taleo")
        print()

    total = sum(len(v) for v in confirmed.values())
    if not total:
        print("Nothing verified.")
        return
    print("=" * 62)
    if args.write:
        write_config(confirmed, args.config, args.replace)
    else:
        print("Paste into config.yaml (or re-run with --write):\n")
        print("ats:")
        for ats in ATS_ORDER:
            vals = confirmed.get(ats) or []
            if not vals:
                continue
            print(f"  {ats}:")
            for slug, label in vals:
                print(f'    - "{slug}"    # {label}')
    print(f"\n{total} verified board(s). Each costs one request per run —")
    print("keep the list to employers you would genuinely apply to.")


def cmd_verify(spec, args):
    if ":" not in spec:
        sys.exit("format: platform:slug   e.g. greenhouse:urban")
    ats, raw = spec.split(":", 1)
    if ats not in FILES:
        sys.exit(f"unknown platform {ats!r}; one of {', '.join(FILES)}")
    ok, n, label = probe(ats, raw)
    print(f"{spec}: {'OK' if ok else 'DEAD'}  jobs={n}  label={label!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="download/update the slug directories")
    ap.add_argument("--find", nargs="+", metavar="EMPLOYER")
    ap.add_argument("--find-file", metavar="PATH",
                    help="file with one employer name per line")
    ap.add_argument("--verify", metavar="PLATFORM:SLUG")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip live verification (fast, but keeps false matches)")
    ap.add_argument("--min-jobs", type=int, default=1,
                    help="ignore boards with fewer than this many open jobs")
    ap.add_argument("--max-per-ats", type=int, default=6)
    ap.add_argument("--write", action="store_true",
                    help="merge verified slugs straight into config.yaml")
    ap.add_argument("--replace", action="store_true",
                    help="with --write: overwrite the lists instead of merging")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between probes; be polite")
    args = ap.parse_args()

    if args.refresh:
        return refresh()
    if args.verify:
        return cmd_verify(args.verify, args)

    targets = list(args.find or [])
    if args.find_file:
        targets += [l.strip() for l in open(args.find_file, encoding="utf-8")
                    if l.strip() and not l.startswith("#")]
    if not targets and Path("targets.txt").exists():
        targets = [l.strip() for l in open("targets.txt", encoding="utf-8")
                   if l.strip() and not l.startswith("#")]
        print(f"(no --find given; using targets.txt, {len(targets)} employers)\n")
    if not targets:
        ap.print_help()
        return
    cmd_find(targets, args)


if __name__ == "__main__":
    main()
