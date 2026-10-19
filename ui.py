#!/usr/bin/env python3
"""
ui.py - jobscan in the browser.

    python ui.py                 # serves http://127.0.0.1:8765 and opens it
    python ui.py --port 9000 --no-browser

Everything the command line does, through the same code: the page reads
seen.db, config.yaml and out/ directly, and runs main.py, tailor.py,
score.py, preflight.py, slugs.py and bench_jev.py as subprocesses with their
output streamed back. Nothing here re-implements a scoring or drafting rule.

Safety. It listens on 127.0.0.1 only. Requests that change anything must
carry an X-Jobscan header, which a web page on another site cannot send
without a CORS preflight this server never answers, and the Host header must
be this machine, which defeats DNS rebinding. Scripts and flags come from a
fixed list below; the browser cannot run an arbitrary command.

Config edits keep your comments. Structured changes (thresholds, which model
does what) rewrite only the line they touch, then re-parse the file and
refuse to save if the result is not exactly what was asked for. Every save
first copies the old file to backups/.

Standard library only, plus the PyYAML jobscan already uses.
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import env  # noqa: E402
env.load()
import llm  # noqa: E402
import score  # noqa: E402
import store  # noqa: E402

UI_FILE = ROOT / "ui" / "index.html"
OUT_DIR = ROOT / "out"
BACKUP_DIR = ROOT / "backups"
DOWNLOAD_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document",
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".json": "application/json",
    ".txt": "text/plain",
}
TIER_ORDER = ("strong", "review", "maybe", "weak", "unscored")
MY_STATUSES = ("new", "saved", "applied", "dismissed")


# ==========================================================================
# Config, read fresh on every request so edits elsewhere show up at once
# ==========================================================================
def config_path():
    return ROOT / "config.yaml"


def load_config():
    try:
        return yaml.safe_load(config_path().read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def facts_path(cfg=None):
    cfg = cfg if cfg is not None else load_config()
    return ROOT / (cfg.get("facts_file") or "resume.yaml")


def scoring_cfg(cfg=None):
    return score.merge_scoring_cfg(cfg if cfg is not None else load_config())


# ==========================================================================
# Comment-preserving YAML edits
# ==========================================================================
_PLAIN = re.compile(r"[A-Za-z0-9_.\-/:@+]+")
_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "~", ""}


def _fmt(v):
    """A Python value as an inline YAML value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    s = str(v)
    if _PLAIN.fullmatch(s) and s.lower() not in _RESERVED and not s[0].isdigit():
        return s
    return json.dumps(s)


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _is_content(line):
    t = line.strip()
    return bool(t) and not t.startswith("#")


def _block_end(lines, i):
    """Index after the last line belonging to the block that starts at i."""
    base = _indent(lines[i])
    j, last = i + 1, i
    while j < len(lines):
        if _is_content(lines[j]):
            if _indent(lines[j]) <= base:
                break
            last = j
        j += 1
    return last + 1


def _split_comment(rest):
    """' 70   # note' -> (' 70', '   # note'), leaving quoted '#' alone."""
    quote = None
    for k, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (k == 0 or rest[k - 1] in " \t"):
            m = re.search(r"\s*$", rest[:k])
            return rest[:m.start()], rest[m.start():]
    return rest, ""


_KEY = re.compile(r"^(\s*)(['\"]?)([^'\"#:]+?)\2\s*:(.*)$")


def yaml_set(text, path, value):
    """
    Set path (a list of keys) to value, touching as little text as possible.

    An existing scalar keeps its line, its position and its trailing comment.
    A key that held a nested block has that block replaced by an inline
    value. A missing key is added at the end of its parent block, creating
    the parents if they do not exist either. Raises ValueError if the result
    does not parse back to the requested value.
    """
    lines = text.split("\n")
    lo, hi, parent_indent = 0, len(lines), -1
    for depth, key in enumerate(path):
        last = depth == len(path) - 1
        child_indent, found = None, None
        for i in range(lo, hi):
            if not _is_content(lines[i]):
                continue
            ind = _indent(lines[i])
            if ind <= parent_indent:
                break
            if child_indent is None:
                child_indent = ind
            if ind != child_indent:
                continue
            m = _KEY.match(lines[i])
            if m and m.group(3).strip() == key:
                found = i
                break
        if child_indent is None:
            child_indent = parent_indent + 2 if parent_indent >= 0 else 0
        pad = " " * child_indent

        if found is None:
            # Insert after the last content line of the parent block.
            at = lo
            for i in range(lo, hi):
                if _is_content(lines[i]):
                    if _indent(lines[i]) <= parent_indent:
                        break
                    at = i + 1
            if parent_indent < 0:
                at = len(lines)
                while at > 0 and not lines[at - 1].strip():
                    at -= 1
            new = f"{pad}{key}: {_fmt(value)}" if last else f"{pad}{key}:"
            lines.insert(at, new)
            if last:
                break
            lo, hi, parent_indent = at + 1, at + 1, child_indent
            continue

        if last:
            m = _KEY.match(lines[found])
            _, comment = _split_comment(m.group(4))
            end = _block_end(lines, found)
            lines[found] = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(2)}: " \
                           f"{_fmt(value)}{comment}"
            # A nested block under this key is replaced by the inline value;
            # comment lines inside it go with it.
            del lines[found + 1:end]
            break
        m = _KEY.match(lines[found])
        if _split_comment(m.group(4))[0].strip():
            raise ValueError(f"{'.'.join(path[:depth + 1])} holds a value, "
                             f"not a section; cannot set {'.'.join(path)}")
        lo, hi, parent_indent = found + 1, _block_end(lines, found), child_indent

    out = "\n".join(lines)
    parsed = yaml.safe_load(out) or {}
    cur = parsed
    for key in path:
        cur = (cur or {}).get(key) if isinstance(cur, dict) else None
    if cur != value:
        raise ValueError(f"edit of {'.'.join(path)} did not round-trip "
                         f"(got {cur!r}); nothing was saved")
    return out


def backup(path):
    """Copy a file into backups/ before overwriting it. Keeps the last 30."""
    if not path.exists():
        return None
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = BACKUP_DIR / f"{path.name}.{stamp}"
    shutil.copy2(path, dest)
    olds = sorted(BACKUP_DIR.glob(f"{path.name}.*"))
    for old in olds[:-30]:
        old.unlink(missing_ok=True)
    return dest


def write_config_text(text):
    backup(config_path())
    config_path().write_text(text, encoding="utf-8")


def edit_config(path, value):
    text = config_path().read_text(encoding="utf-8")
    write_config_text(yaml_set(text, path, value))


def check_config_text(text):
    """(errors, warnings) for a candidate config.yaml. Errors block a save."""
    try:
        cfg = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        return [f"YAML: {e}"], []
    if not isinstance(cfg, dict):
        return ["the file must be a mapping of keys"], []
    errors, warnings = [], []
    try:
        scfg = score.merge_scoring_cfg(cfg)
    except ValueError as e:
        errors.append(str(e))
        scfg = None
    try:
        import jev
        problems = jev.validate(jev.merged(cfg.get("jev")))
        warnings += [f"jev: {p} (the run would fall back to the built-in "
                     f"questions)" for p in problems]
    except Exception as e:
        warnings.append(f"jev: {e}")
    for role in llm.ROLES:
        for track in (score.TRACKS if role in ("jev", "rubric") else (None,)):
            try:
                llm.resolve(cfg, role, track)
            except llm.LLMError as e:
                msg = f"{role}{'/' + track if track else ''}: {e}"
                (errors if role in ("jev", "rubric") else warnings).append(msg)
    if scfg:
        for t in score.TRACKS:
            s, m = score.thresholds(scfg, t)
            if not 0 <= m <= s <= 100:
                errors.append(f"{t}: maybe_at ({m}) must be at or below "
                              f"strong_at ({s}), both within 0-100")
    return errors, warnings


# ==========================================================================
# Database
# ==========================================================================
def db():
    conn = store.connect()
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


LIST_COLS = ("uid", "title", "company", "location", "salary", "url", "posted",
             "first_seen", "score", "tier", "reason", "source", "track",
             "scorer", "scorer_model", "status", "desc_quality",
             "bookmarked")


def display_tier(row, scfg):
    """
    Tier under the CURRENT thresholds. The stored tier was computed at
    scoring time; score.py treats thresholds as a display decision, so the UI
    re-applies them. Review and unscored are states, not thresholds.
    """
    t = row.get("tier")
    if t in ("review", "unscored") or row.get("score") is None:
        return t or "unscored"
    return score.tier_of(row["score"], scfg, track=row.get("track") or "career")


def _since_iso(days):
    return (datetime.now(timezone.utc) - timedelta(days=float(days))) \
        .isoformat(timespec="seconds")


def query_jobs(q):
    """Filtered, tiered, sorted rows plus counts for the filter panel."""
    scfg = scoring_cfg()
    where, args = [], []
    track = q.get("track")
    if track == "career":
        where.append("(track = 'career' OR track IS NULL)")
    elif track == "bridge":
        where.append("track = 'bridge'")
    if q.get("days"):
        where.append("first_seen >= ?")
        args.append(_since_iso(q["days"]))
    if q.get("run") == "last":
        where.append("first_seen = (SELECT MAX(first_seen) FROM jobs)")
    if q.get("source"):
        where.append("source = ?")
        args.append(q["source"])
    text = (q.get("q") or "").strip()
    if text:
        like = f"%{text}%"
        where.append("(title LIKE ? OR company LIKE ? OR location LIKE ? "
                     "OR reason LIKE ?" + (" OR description LIKE ?"
                                            if q.get("deep") else "") + ")")
        args += [like] * (5 if q.get("deep") else 4)
    if q.get("bookmarked"):
        where.append("bookmarked = 1")
    if q.get("scorer"):
        fam = q["scorer"]
        if fam == "rubric":
            where.append("(scorer IN ('rubric','anthropic') OR "
                         "(scorer IS NULL AND score IS NOT NULL))")
        else:
            where.append("scorer = ?")
            args.append(fam)
    sql = (f"SELECT {', '.join(LIST_COLS)} FROM jobs"
           + (" WHERE " + " AND ".join(where) if where else ""))
    conn = db()
    rows = [dict(zip(LIST_COLS, r)) for r in conn.execute(sql, args)]
    conn.close()

    for r in rows:
        r["tier_now"] = display_tier(r, scfg)
        r["status"] = r.get("status") or "new"
        r["track"] = r.get("track") or "career"
        r["bookmarked"] = bool(r.get("bookmarked"))
    tier_counts = {t: 0 for t in TIER_ORDER}
    status_counts = {s: 0 for s in MY_STATUSES}
    for r in rows:
        if q.get("status") in (None, "", r["status"]) or (
                q.get("status") == "open" and r["status"] in ("new", "saved")):
            tier_counts[r["tier_now"]] = tier_counts.get(r["tier_now"], 0) + 1
        if not q.get("tier") or r["tier_now"] == q["tier"]:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    st = q.get("status")
    if st == "open":
        rows = [r for r in rows if r["status"] in ("new", "saved")]
    elif st:
        rows = [r for r in rows if r["status"] == st]
    if q.get("tier"):
        rows = [r for r in rows if r["tier_now"] == q["tier"]]

    # Tailor state used to be attached to the page slice only, which is
    # cheaper but makes it impossible to filter on. It is a dict lookup
    # against an index already built once, so doing it for every row costs
    # little and lets "already looked at" become a filter rather than
    # something you have to notice yourself.
    idx = drafts_index()
    for r in rows:
        r["tailor"] = tailor_state(r, idx)
    seen_counts = {"review": 0, "skipped": 0, "drafted": 0}
    for r in rows:
        s = r["tailor"]["state"]
        if s in seen_counts:
            seen_counts[s] += 1
    if q.get("hide_reviewed"):
        rows = [r for r in rows if r["tailor"]["state"] != "review"]
    if q.get("hide_skipped"):
        rows = [r for r in rows if r["tailor"]["state"] != "skipped"]
    if q.get("hide_drafted"):
        rows = [r for r in rows if r["tailor"]["state"] != "drafted"]

    if q.get("sort") == "newest":
        rows.sort(key=lambda r: (r.get("first_seen") or "", r["score"] or -1),
                  reverse=True)
    else:
        rows.sort(key=lambda r: (r["score"] is not None, r["score"] or 0,
                                 r.get("first_seen") or ""), reverse=True)
    total = len(rows)
    off, lim = int(q.get("offset") or 0), int(q.get("limit") or 100)
    page = rows[off:off + lim]
    return {"total": total, "rows": page, "tiers": tier_counts,
            "statuses": status_counts, "seen": seen_counts}


def histogram(q):
    """Model scores 0-100 for the strip at the top of Postings."""
    scfg = scoring_cfg()
    track = q.get("track") or "career"
    where = ["score IS NOT NULL", "tier NOT IN ('unscored')",
             "(scorer IS NULL OR scorer IN ('rubric','anthropic','jev'))"]
    args = []
    where.append("(track = 'career' OR track IS NULL)" if track == "career"
                 else "track = 'bridge'")
    if q.get("days"):
        where.append("first_seen >= ?")
        args.append(_since_iso(q["days"]))
    conn = db()
    bins = [0] * 101
    review = 0
    for sc, tier in conn.execute(
            f"SELECT score, tier FROM jobs WHERE {' AND '.join(where)}", args):
        if tier == "review":
            review += 1
        bins[max(0, min(100, int(sc)))] += 1
    unscored = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE (tier = 'unscored' OR score IS NULL)"
        + (" AND (track = 'career' OR track IS NULL)" if track == "career"
           else " AND track = 'bridge'")).fetchone()[0]
    conn.close()
    strong, maybe = score.thresholds(scfg, track)
    return {"bins": bins, "strong_at": strong, "maybe_at": maybe,
            "review": review, "unscored": unscored}


def summary():
    cfg = load_config()
    scfg = scoring_cfg(cfg)
    conn = db()
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    last = conn.execute("SELECT MAX(first_seen) FROM jobs").fetchone()[0]
    last_n = conn.execute("SELECT COUNT(*) FROM jobs WHERE first_seen = ?",
                          (last,)).fetchone()[0] if last else 0
    conn.close()
    methods = {}
    for t in score.TRACKS:
        m = score.method_for(scfg, t)
        prof = None
        if m in ("jev", "rubric"):
            try:
                prof = llm.describe(llm.resolve(cfg, m, t))
            except llm.LLMError as e:
                prof = f"! {e}"
        methods[t] = {"method": m, "model": prof,
                      "strong_at": score.thresholds(scfg, t)[0],
                      "maybe_at": score.thresholds(scfg, t)[1]}
    return {"total": total, "last_run": last, "last_run_count": last_n,
            "methods": methods,
            "bridge_enabled": bool((cfg.get("bridge") or {}).get("enabled")),
            "root": str(ROOT)}


def get_job(uid):
    conn = db()
    cols = LIST_COLS + ("description",)
    row = conn.execute(f"SELECT {', '.join(cols)} FROM jobs WHERE uid = ?",
                       (uid,)).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(zip(cols, row))
    r["tier_now"] = display_tier(r, scoring_cfg())
    r["status"] = r.get("status") or "new"
    r["track"] = r.get("track") or "career"
    r["bookmarked"] = bool(r.get("bookmarked"))
    r["tailor"] = tailor_state(r, drafts_index())
    return r


def set_status(uid, status):
    if status not in MY_STATUSES:
        raise ValueError(f"status must be one of {', '.join(MY_STATUSES)}")
    conn = db()
    n = conn.execute("UPDATE jobs SET status = ? WHERE uid = ?",
                     (status, uid)).rowcount
    conn.commit()
    conn.close()
    if not n:
        raise KeyError(uid)


def set_bookmark(uid, on):
    """Flag or unflag one posting. Independent of status."""
    conn = db()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = conn.execute(
        "UPDATE jobs SET bookmarked = ?, bookmarked_at = ? WHERE uid = ?",
        (1 if on else 0, now if on else None, uid)).rowcount
    conn.commit()
    conn.close()
    if not n:
        raise KeyError(uid)
    return {"ok": True, "uid": uid, "bookmarked": bool(on)}


def set_description(uid, text):
    """
    Store a posting the user pasted in by hand.

    Some postings cannot be fetched: the employer's page needs JavaScript,
    or the host refuses anything that is not a person with a browser. The
    answer is not a one-off draft from a temporary file, which leaves the
    stored posting just as broken as it was. Writing it to seen.db fixes
    the row for good, so scoring, tailoring and every later run all see the
    real text.

    Marked `full` on purpose. A person who went and copied the posting has
    better evidence than any heuristic here, and leaving it `excerpt` would
    make the tailor refuse the text they just supplied.
    """
    text = (text or "").strip()
    if len(text) < 200:
        raise ValueError(f"that is only {len(text)} characters. Paste the "
                         f"whole posting: duties and requirements, not just "
                         f"the title.")
    conn = db()
    n = conn.execute(
        "UPDATE jobs SET description = ?, desc_quality = 'full' "
        "WHERE uid = ?", (text, uid)).rowcount
    conn.commit()
    conn.close()
    if not n:
        raise KeyError(uid)
    return {"ok": True, "chars": len(text), "desc_quality": "full"}


# ==========================================================================
# Drafts in out/, read the same way tailor.py reads them
# ==========================================================================
def _slug(s):
    import tailor
    return tailor.slug(s)


def drafts_index():
    """{stem: {"files": [...], "rec": ..., "newest": ...}} from out/."""
    idx = {}
    if not OUT_DIR.exists():
        return idx
    for f in OUT_DIR.rglob("*__*"):
        if f.suffix not in (".md", ".json", ".docx", ".pdf") or not f.is_file():
            continue
        stem, kind = f.stem.rsplit("__", 1)
        e = idx.setdefault(stem, {"files": [], "rec": None, "newest": 0})
        e["files"].append({"path": f.relative_to(ROOT).as_posix(),
                           "kind": kind, "day": f.parent.name,
                           "ext": f.suffix.lstrip("."),
                           "editable": f.suffix == ".md",
                           "mtime": f.stat().st_mtime})
        e["newest"] = max(e["newest"], f.stat().st_mtime)
        if kind == "analysis":
            try:
                a = json.loads(f.read_text(encoding="utf-8"))
                import tailor
                e["rec"] = tailor.normalize_recommendation(
                    a.get("recommendation")) or "manual_review"
                e["company"] = a.get("company") or ""
                e["role"] = a.get("role") or ""
                e["analysis"] = f.relative_to(ROOT).as_posix()
            except Exception:
                e["rec"] = e["rec"] or "manual_review"
    return idx


def tailor_state(row, idx):
    stem = f"{_slug(row.get('company') or '')}__{_slug(row.get('title') or '')}"
    e = idx.get(stem)
    if not e:
        return {"state": "new", "stem": stem, "files": []}
    kinds = {f["kind"] for f in e["files"]}
    if kinds & {"letter", "resume"}:
        state = "drafted"
    elif e["rec"] == "do_not_apply":
        state = "skipped"
    else:
        state = "review"
    return {"state": state, "stem": stem, "rec": e["rec"],
            "files": sorted(e["files"], key=lambda f: f["mtime"])}


def stem_to_uid():
    """
    {stem: uid} for every stored posting.

    A draft folder is named after the company and role, not the uid, so the
    drafts page had no way to act on the posting behind it. Rebuilding the
    same slug from the jobs table recovers the link, and it works for
    analyses written before anything recorded a uid.
    """
    conn = db()
    out = {}
    for uid, company, title in conn.execute(
            "SELECT uid, company, title FROM jobs"):
        out.setdefault(f"{_slug(company or '')}__{_slug(title or '')}", uid)
    conn.close()
    return out


def list_drafts():
    idx = drafts_index()
    uids = stem_to_uid()
    out = []
    for stem, e in idx.items():
        kinds = {f["kind"] for f in e["files"]}
        out.append({"stem": stem, "rec": e["rec"], "newest": e["newest"],
                    "uid": uids.get(stem),
                    "company": e.get("company") or "",
                    "role": e.get("role") or "",
                    "drafted": bool(kinds & {"letter", "resume"}),
                    "files": sorted(e["files"], key=lambda f: f["mtime"])})
    out.sort(key=lambda d: d["newest"], reverse=True)
    return out


def safe_out_path(rel):
    p = (ROOT / rel).resolve()
    if OUT_DIR.resolve() not in p.parents:
        raise PermissionError("only files under out/ can be read or written")
    return p


# ==========================================================================
# Models
# ==========================================================================
def models_view():
    cfg = load_config()
    scfg = scoring_cfg(cfg)
    profiles = []
    for name, raw in (cfg.get("models") or {}).items():
        try:
            p = llm.finalize(raw, name)
            profiles.append({"name": name, "describe": llm.describe(p),
                             "provider": p["provider"],
                             "model": p.get("model") or "",
                             "cost": p.get("cost"),
                             "where": p.get("base_url") or p.get("endpoint")
                             or "",
                             "problem": llm.check(p),
                             "generative": p["kind"] != "jevk5",
                             "yaml": yaml.safe_dump({name: raw},
                                                    sort_keys=False,
                                                    allow_unicode=True)})
        except llm.LLMError as e:
            profiles.append({"name": name, "problem": str(e),
                             "describe": name, "generative": False})
    ai = cfg.get("ai") or {}
    roles = []
    for role in llm.ROLES:
        for track in (score.TRACKS if role in ("jev", "rubric") else (None,)):
            try:
                p = llm.resolve(cfg, role, track)
                roles.append({"role": role, "track": track,
                              "profile": p["name"], "describe": llm.describe(p),
                              "problem": llm.check(p)})
            except llm.LLMError as e:
                roles.append({"role": role, "track": track, "profile": None,
                              "problem": str(e)})
    methods = {t: score.method_for(scfg, t) for t in score.TRACKS}
    return {"profiles": profiles, "roles": roles, "ai": ai,
            "methods": methods,
            "env_override": os.environ.get("JOBSCAN_AI") or ""}


def set_role(role, track, profile):
    if role not in llm.ROLES:
        raise ValueError(f"unknown role {role}")
    cfg = load_config()
    names = set((cfg.get("models") or {}))
    if profile not in names and profile != "default":
        raise ValueError(f"no profile named {profile!r} under models:")
    cur = (cfg.get("ai") or {}).get(role)
    if track:
        if isinstance(cur, dict) and "provider" not in cur:
            value = dict(cur)
        else:
            base = cur if isinstance(cur, str) else None
            value = {t: base for t in score.TRACKS if base}
        value[track] = profile
        if len(set(value.get(t) for t in score.TRACKS)) == 1 and all(
                t in value for t in score.TRACKS):
            value = profile          # both tracks agree: keep it simple
    else:
        value = profile
    edit_config(["ai", role], value)


def set_method(track, method):
    method = score.normalize_method(method)
    if track == "both":
        edit_config(["scoring", "method"], method)
        text = config_path().read_text(encoding="utf-8")
        cfg = yaml.safe_load(text) or {}
        # Per-track overrides would silently win over the global value.
        for t in score.TRACKS:
            tsc = ((cfg.get(t) or {}).get("scoring") or {})
            if "method" in tsc or "provider" in tsc:
                key = "method" if "method" in tsc else "provider"
                text = yaml_set(text, [t, "scoring", key], method)
        write_config_text(text)
    elif track in score.TRACKS:
        cfg = load_config()
        tsc = ((cfg.get(track) or {}).get("scoring") or {})
        key = "provider" if ("provider" in tsc and "method" not in tsc) \
            else "method"
        edit_config([track, "scoring", key], method)
    else:
        raise ValueError("track must be career, bridge or both")


def set_thresholds(track, strong, maybe):
    strong, maybe = int(strong), int(maybe)
    if not 0 <= maybe <= strong <= 100:
        raise ValueError("need 0 <= maybe <= strong <= 100")
    text = config_path().read_text(encoding="utf-8")
    base = ["scoring"] if track == "career" else ["bridge", "scoring"]
    text = yaml_set(text, base + ["strong_at"], strong)
    text = yaml_set(text, base + ["maybe_at"], maybe)
    write_config_text(text)


def env_status():
    cfg = load_config()
    wanted = {"ADZUNA_APP_ID": "Adzuna", "ADZUNA_APP_KEY": "Adzuna",
              "USAJOBS_EMAIL": "USAJOBS", "USAJOBS_API_KEY": "USAJOBS",
              "SOCRATA_APP_TOKEN": "NYC Open Data (optional)",
              "SMTP_USER": "Email digest", "SMTP_PASSWORD": "Email digest",
              "DIGEST_TO": "Email digest"}
    for name, raw in (cfg.get("models") or {}).items():
        try:
            p = llm.finalize(raw, name)
        except llm.LLMError:
            continue
        if p.get("api_key_env"):
            wanted.setdefault(p["api_key_env"], f"model {name}")
    return [{"key": k, "for": v, "set": bool(os.environ.get(k))}
            for k, v in wanted.items()]


# ==========================================================================
# Running the scripts
# ==========================================================================
RUNS = {}
RUNS_LOCK = threading.Lock()
SOURCES = ("adzuna", "usajobs", "nyc", "greenhouse", "lever", "ashby",
           "workday")


def _int(v, lo, hi, name):
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number")
    if not lo <= n <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return n


def _track(v, allow_all=False):
    ok = ("career", "bridge") + (("all",) if allow_all else ())
    if v not in ok:
        raise ValueError(f"track must be one of {', '.join(ok)}")
    return v


def _profile_name(v):
    if v not in (load_config().get("models") or {}):
        raise ValueError(f"no model profile named {v!r}")
    return v


def _spec(v):
    m, _, name = str(v).partition(":")
    if score.normalize_method(m) not in ("jev", "rubric"):
        raise ValueError("compare sides are jev, rubric, jev:PROFILE or "
                         "rubric:PROFILE")
    if name:
        _profile_name(name)
    return v


def build_command(kind, o):
    """(label, script, args) from a whitelisted kind and checked options."""
    o = o or {}
    a = []
    if kind == "scan":
        if o.get("dry_run"):
            a.append("--dry-run")
        if o.get("no_score"):
            a.append("--no-score")
        if o.get("track"):
            a += ["--track", _track(o["track"])]
        src = [s for s in (o.get("sources") or []) if s in SOURCES]
        if src and len(src) < len(SOURCES):
            a += ["--source", ",".join(src)]
        if o.get("location"):
            loc = str(o["location"])
            if not re.fullmatch(r"[\w ,.'\-]{1,200}", loc):
                raise ValueError("location may contain letters, spaces and "
                                 "commas only")
            a += ["--location", loc]
        if o.get("max_score"):
            a += ["--max-score", str(_int(o["max_score"], 1, 100000,
                                          "max score"))]
        label = "Scan" + (" (dry run)" if o.get("dry_run") else "")
        return label, "main.py", a
    if kind == "tailor":
        if o.get("uid"):
            a.append(str(o["uid"]))
            label = f"Tailor {o.get('title') or o['uid']}"
        elif o.get("posting"):
            text = str(o["posting"])
            if len(text.strip()) < 150:
                raise ValueError("paste the whole posting (150+ characters)")
            d = OUT_DIR / "_ui"
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"posting-{int(time.time())}.txt"
            f.write_text(text, encoding="utf-8")
            a += ["--file", str(f)]
            label = "Tailor pasted posting"
        else:
            raise ValueError("choose a posting or paste one")
        mode = o.get("mode") or "letter"
        if mode == "both":
            a.append("--both")
        elif mode == "resume":
            a.append("--resume")
        elif mode != "letter":
            raise ValueError("mode is letter, resume or both")
        if o.get("advice"):
            a.append("--advice")
        if o.get("pdf"):
            a.append("--pdf")
        if o.get("no_docx"):
            a.append("--no-docx")
        if o.get("force"):
            a.append("--force")
        if o.get("model"):
            a += ["--model", _profile_name(o["model"])]
        return label, "tailor.py", a
    if kind == "dedupe":
        # Duplicates already stored. New ones are prevented at fetch time by
        # the fingerprint; this is for rows written before that existed.
        a = ["--apply"] if o.get("apply") else []
        return ("Remove stored duplicates" if o.get("apply")
                else "Find stored duplicates", "store.py", a)
    if kind == "review":
        # A deliberate second opinion on one posting, on a model you pick.
        # Drafts nothing, so it cannot quietly become a billed letter.
        if not o.get("uid"):
            raise ValueError("choose a posting to review")
        a = [str(o["uid"]), "--review"]
        if o.get("model"):
            a += ["--model", _profile_name(o["model"])]
        return (f"Review {o.get('title') or o['uid']}", "tailor.py", a)
    if kind == "render":
        # Re-render a draft you edited in the browser. The .md stays the
        # source of truth; the .docx is rebuilt from it every time.
        rel = str(o.get("path") or "")
        if not rel:
            raise ValueError("no draft chosen")
        f = safe_out_path(rel)
        if f.suffix != ".md":
            raise ValueError("only a .md draft can be rendered")
        a = [str(f)]
        if o.get("pdf"):
            a.append("--pdf")
        return (f"Render {f.name}"
                + (" + PDF" if o.get("pdf") else ""), "docs.py", a)
    if kind == "batch":
        a = ["batch", "--yes"]
        if o.get("dry_run"):
            a.append("--dry-run")
        a += ["--limit", str(_int(o.get("limit", 25), 1, 500, "limit"))]
        if o.get("min_score") not in (None, ""):
            a += ["--min-score", str(_int(o["min_score"], 0, 100,
                                          "minimum score"))]
        a += ["--track", _track(o.get("track") or "career", allow_all=True)]
        if o.get("newest"):
            a.append("--newest")
        if o.get("model"):
            a += ["--model", _profile_name(o["model"])]
        return ("Batch tailor" + (" (dry run)" if o.get("dry_run") else ""),
                "tailor.py", a)
    if kind == "backfill":
        a = ["--backfill", "--limit",
             str(_int(o.get("limit", 200), 1, 100000, "limit"))]
        if o.get("uid"):
            a += ["--uid", str(o["uid"])]
        if o.get("rescore"):
            a.append("--rescore")
        if o.get("scorer"):
            if not re.fullmatch(r"[\w.:\-]{1,80}", str(o["scorer"])):
                raise ValueError("scorer is a method or profile name")
            a += ["--scorer", str(o["scorer"])]
        if o.get("yes"):
            a.append("--yes")
        label = ("Rescore " + (o.get("title") or o["uid"])) if o.get("uid") \
            else ("Rescore" if o.get("rescore") else "Backfill")
        return label, "score.py", a
    if kind == "compare":
        a = ["--compare", "--track", _track(o.get("track") or "bridge"),
             "--limit", str(_int(o.get("limit", 60), 2, 5000, "limit"))]
        if o.get("left"):
            a += ["--left", _spec(o["left"])]
        if o.get("right"):
            a += ["--right", _spec(o["right"])]
        return "Compare scorers", "score.py", a
    if kind == "test":
        title = str(o.get("title") or "").strip()
        if not title:
            raise ValueError("enter a title")
        a = ["--test", title]
        if o.get("desc"):
            a += ["--desc", str(o["desc"])]
        if o.get("track"):
            a += ["--track", _track(o["track"])]
        return f"Test: {title[:40]}", "score.py", a
    if kind == "calibrate":
        a = ["--calibrate"]
        if o.get("track"):
            a += ["--track", _track(o["track"])]
        return "Calibrate", "score.py", a
    if kind == "preflight":
        return ("Preflight" + (" with model probe" if o.get("probe") else ""),
                "preflight.py", ["--probe"] if o.get("probe") else [])
    if kind == "slugs":
        if o.get("find"):
            names = [n.strip() for n in str(o["find"]).split(",") if n.strip()]
            if not names or not all(re.fullmatch(r"[\w .&'\-]{1,80}", n)
                                    for n in names):
                raise ValueError("employer names, separated by commas")
            return "Find job boards", "slugs.py", ["--find", *names]
        return "Refresh job-board list", "slugs.py", ["--refresh"]
    if kind == "enrich":
        # Postings that arrived with a URL but no description. Fetches the
        # text and screens it with jev before storing, so a sign-in wall is
        # not scored as though it were the job.
        a = []
        if o.get("uid"):
            # One posting, by id. Used by "Fetch the full posting" on a job
            # whose stored text is only an excerpt.
            a += ["--uid", str(o["uid"])]
        else:
            a += ["--limit", str(_int(o.get("limit", 200), 1, 100000,
                                      "limit"))]
            if o.get("excerpts"):
                a.append("--excerpts")
        if o.get("dry_run"):
            a.append("--dry-run")
        if o.get("no_screen"):
            a.append("--no-screen")
        if o.get("model"):
            a += ["--model", _profile_name(o["model"])]
        label = ("Fetch the full posting" if o.get("uid") else
                 "Fetch thin and missing descriptions" if o.get("excerpts")
                 else "Fetch missing descriptions")
        return (label + (" (dry run)" if o.get("dry_run") else ""),
                "enrich.py", a)
    if kind == "bench":
        a = ["--n", str(_int(o.get("n", 12), 1, 500, "postings"))]
        if o.get("model"):
            a += ["--model", _profile_name(o["model"])]
        return "Benchmark jev", "bench_jev.py", a
    raise ValueError(f"unknown action {kind!r}")


class Run:
    def __init__(self, kind, label, script, args):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.label, self.script, self.args = kind, label, script, args
        self.lines = []
        self.started = time.time()
        self.ended = None
        self.code = None
        env_ = {**os.environ, "PYTHONUNBUFFERED": "1",
                # Windows pipes default to cp1252, which cannot encode the
                # dashes and arrows these scripts print.
                "PYTHONIOENCODING": "utf-8"}
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(
            [sys.executable, "-u", script, *args], cwd=ROOT, env=env_,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, **kw)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for raw in iter(self.proc.stdout.readline, b""):
            self.lines.append(raw.decode("utf-8", "replace").rstrip("\r\n"))
        self.code = self.proc.wait()
        self.ended = time.time()

    def stop(self):
        if self.code is None:
            self.proc.terminate()

    def view(self, since=0):
        return {"id": self.id, "kind": self.kind, "label": self.label,
                "command": " ".join(["python", self.script, *self.args]),
                "started": self.started, "ended": self.ended,
                "code": self.code, "running": self.code is None,
                "lines": self.lines[since:], "next": len(self.lines)}


ASSIST_SYSTEM = """You are the built-in assistant for jobscan, a personal \
job-search tool the user runs on their own machine.

You do two jobs:

1. ANSWER QUESTIONS about how the tool works, using the notes below. If the
   notes do not cover it, say so rather than guessing at a flag or a file that
   may not exist.
2. REVIEW A DRAFT when one is supplied. Be specific and concrete. Quote the
   line you mean and say what to put instead. A review that says "consider
   strengthening the opening" is useless.

When reviewing, hold the draft to the candidate's own voice rules, which are
in the fact base: no filler enthusiasm, no em dashes, no "I am excited to
apply", no manufactured X-not-Y contrasts, no claim that the facts do not
support. Flag any sentence that asserts experience with a tool or domain
absent from the fact base, because that is the one error that gets an
application thrown out.

Be brief. Lead with the single most important thing. Never invent a fact
about the candidate."""

ASSIST_NOTES = """HOW JOBSCAN WORKS

Two tracks, never mixed: `career` (research, policy, quantitative) and
`bridge` (rent work). Each is scored separately and shown separately.

Pipeline: fetch -> hard_exclude -> location filter -> dedupe -> score ->
record everything -> digest. Scoring runs after dedupe so nothing is scored
twice, and every scored posting is stored even when it scores too low to
email.

Scoring methods per track: `jev` (the local JevK5 classifier, free, generates
no tokens, returns calibrated probabilities), `rubric` (a generative model
reading a prose rubric), `keyword` (a free fallback estimate). Failure falls
through jev -> rubric -> keyword, so an outage costs accuracy, never the run.

Postings that arrive as a bare link with no text cannot be scored. `enrich.py`
fetches the body and screens it locally so a sign-in wall is not stored as a
description. In the UI that is "Fetch missing descriptions".

Tailoring writes to out/<date>/. The .md is the editable source; the .docx is
rebuilt from it and overwritten on every render, so edits belong in the .md.
`--resume` writes a tailored resume document, `--advice` writes notes about
what to change, `--both` writes letter and resume, `--pdf` adds a PDF.

Models live under `models:` in config.yaml and are assigned to roles under
`ai:`. Adding a profile there makes it appear in every dropdown."""


def assist(body):
    """
    Answer a question, optionally about a specific draft or posting.

    Runs on whichever profile `ai:` assigns to tailor_letter, because that is
    the model already chosen for prose. Costs whatever that model costs.
    """
    question = (body.get("question") or "").strip()
    if not question:
        raise ValueError("ask a question")

    cfg = load_config()
    try:
        name = body.get("model")
        p = (llm._lookup(cfg, name, "tailor_letter") if name
             else llm.resolve(cfg, "tailor_letter"))
    except llm.LLMError as e:
        raise ValueError(str(e))
    problem = llm.check(p)
    if problem:
        raise ValueError(f"{llm.describe(p)}: {problem}")
    ok, _ = llm.ensure_ready(p, quiet=True)
    if not ok:
        raise ValueError(f"{llm.describe(p)} is not answering. Start it, or "
                         f"pick another model.")

    parts = [ASSIST_NOTES]

    path = body.get("path")
    if path:
        f = safe_out_path(path)
        if f.exists() and f.suffix == ".md":
            parts.append(f"THE DRAFT UNDER REVIEW ({f.name}):\n"
                         f"<draft>\n{f.read_text(encoding='utf-8')}\n</draft>")

    if body.get("include_facts", True):
        fp = facts_path(cfg)
        if fp.exists():
            parts.append(f"CANDIDATE FACTS (nothing outside this exists):\n"
                         f"<facts>\n{fp.read_text(encoding='utf-8')}\n</facts>")

    uid = body.get("uid")
    if uid:
        job = get_job(uid)
        if job:
            parts.append(
                "THE POSTING:\n<posting>\n"
                f"{job.get('title','')} at {job.get('company','')}\n"
                f"{(job.get('description') or '')[:6000]}\n</posting>")

    parts.append(f"QUESTION:\n{question}")
    prompt = "\n\n".join(parts)
    answer = llm.complete(p, ASSIST_SYSTEM, prompt,
                          int(p.get("max_tokens") or 1500))
    return {"answer": (answer or "").strip(), "model": llm.describe(p)}


def start_run(kind, opts):
    label, script, args = build_command(kind, opts)
    r = Run(kind, label, script, args)
    with RUNS_LOCK:
        RUNS[r.id] = r
        for old in sorted(RUNS.values(), key=lambda x: x.started)[:-40]:
            if old.code is not None:
                RUNS.pop(old.id, None)
    return r


def digest_preview(q):
    import digest
    conn = db()
    if q.get("days"):
        where, args = "first_seen >= ?", [_since_iso(q["days"])]
    else:
        where, args = "first_seen = (SELECT MAX(first_seen) FROM jobs)", []
    cols = LIST_COLS + ("description",)
    rows = [dict(zip(cols, r)) for r in conn.execute(
        f"SELECT {', '.join(cols)} FROM jobs WHERE {where}", args)]
    conn.close()
    scfg = scoring_cfg()
    for r in rows:
        r["tier"] = display_tier(r, scfg)
        r["track"] = r.get("track") or "career"
    if not rows:
        return {"count": 0, "html": ""}
    return {"count": len(rows), "html": digest.build_html(rows)}


# ==========================================================================
# HTTP
# ==========================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "jobscan-ui"

    def log_message(self, fmt, *a):
        pass

    # ---- plumbing ------------------------------------------------------
    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in ("127.0.0.1", "localhost")

    def _send(self, code, body, ctype="application/json", extra=None):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        # charset belongs on text, not on a .docx: appending it to a binary
        # type makes some browsers try to display the bytes instead of
        # offering a download.
        if ctype.startswith("text/") or ctype in (
                "application/json", "application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _route(self, method):
        if not self._host_ok():
            return self._send(403, {"error": "this server only answers on "
                                             "127.0.0.1 / localhost"})
        if method != "GET" and self.headers.get("X-Jobscan") != "1":
            return self._send(403, {"error": "missing X-Jobscan header"})
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        parts = [p for p in url.path.split("/") if p]
        try:
            return self._dispatch(method, parts, q)
        except KeyError as e:
            return self._send(404, {"error": f"not found: {e}"})
        except PermissionError as e:
            return self._send(403, {"error": str(e)})
        except (ValueError, llm.LLMError) as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    # ---- routes --------------------------------------------------------
    def _dispatch(self, method, parts, q):
        if not parts or parts[0] in ("index.html",):
            if not UI_FILE.exists():
                return self._send(500, {"error": f"{UI_FILE} is missing"})
            return self._send(200, UI_FILE.read_bytes(), "text/html")
        if parts[0] != "api":
            raise KeyError(self.path)
        p = parts[1:]
        G, P = method == "GET", method in ("POST", "PUT")

        if G and p == ["summary"]:
            return self._send(200, summary())
        if G and p == ["jobs"]:
            return self._send(200, query_jobs(q))
        if G and p == ["histogram"]:
            return self._send(200, histogram(q))
        if p == ["job"] and G:
            job = get_job(q.get("uid") or "")
            if not job:
                raise KeyError(q.get("uid"))
            return self._send(200, job)
        if p == ["job", "status"] and P:
            b = self._json()
            set_status(b.get("uid") or "", b.get("status"))
            return self._send(200, {"ok": True})
        if p == ["job", "bookmark"] and P:
            b = self._json()
            return self._send(200, set_bookmark(b.get("uid") or "",
                                                bool(b.get("on"))))
        if p == ["job", "description"] and P:
            b = self._json()
            return self._send(200, set_description(b.get("uid") or "",
                                                   b.get("text")))
        if P and p == ["thresholds"]:
            b = self._json()
            set_thresholds(_track(b.get("track")), b.get("strong_at"),
                           b.get("maybe_at"))
            return self._send(200, {"ok": True})
        if G and p == ["drafts"]:
            return self._send(200, list_drafts())
        if p == ["file"]:
            f = safe_out_path(q.get("path") or "")
            if G:
                if not f.exists():
                    raise KeyError(q.get("path"))
                return self._send(200, {"path": q["path"],
                                        "text": f.read_text(encoding="utf-8")})
            if P:
                if f.suffix != ".md":
                    raise PermissionError("only .md drafts can be edited")
                f.write_text(self._json().get("text", ""), encoding="utf-8")
                return self._send(200, {"ok": True})
        if G and p == ["download"]:
            f = safe_out_path(q.get("path") or "")
            if not f.exists() or not f.is_file():
                raise KeyError(q.get("path"))
            ctype = DOWNLOAD_TYPES.get(f.suffix, "application/octet-stream")
            return self._send(200, f.read_bytes(), ctype,
                              extra={"Content-Disposition":
                                     f'attachment; filename="{f.name}"'})
        if p[:1] == ["config"] and len(p) == 2 and p[1] in ("config", "resume"):
            path = config_path() if p[1] == "config" else facts_path()
            if G:
                return self._send(200, {
                    "name": path.name,
                    "text": path.read_text(encoding="utf-8")
                    if path.exists() else ""})
            b = self._json()
            text = b.get("text", "")
            if p[1] == "config":
                errors, warnings = check_config_text(text)
            else:
                try:
                    data = yaml.safe_load(text)
                    errors = [] if isinstance(data, dict) else [
                        "resume.yaml must be a mapping of keys"]
                except yaml.YAMLError as e:
                    errors = [f"YAML: {e}"]
                warnings = []
            if b.get("check_only") or errors:
                return self._send(200 if not errors else 400,
                                  {"ok": not errors, "errors": errors,
                                   "warnings": warnings})
            saved = backup(path)
            path.write_text(text, encoding="utf-8")
            return self._send(200, {"ok": True, "warnings": warnings,
                                    "backup": saved.name if saved else None})
        if G and p == ["models"]:
            return self._send(200, models_view())
        if P and p == ["models", "role"]:
            b = self._json()
            set_role(b.get("role"), b.get("track"), b.get("profile"))
            return self._send(200, models_view())
        if P and p == ["models", "method"]:
            b = self._json()
            set_method(b.get("track"), b.get("method"))
            return self._send(200, models_view())
        if P and p == ["assist"]:
            return self._send(200, assist(self._json()))
        if G and p == ["env"]:
            return self._send(200, env_status())
        if G and p == ["digest"]:
            return self._send(200, digest_preview(q))
        if p == ["runs"]:
            if P:
                b = self._json()
                return self._send(200, start_run(b.get("kind"),
                                                 b.get("options")).view())
            with RUNS_LOCK:
                runs = sorted(RUNS.values(), key=lambda r: -r.started)
            return self._send(200, [dict(r.view(), lines=r.lines[-1:])
                                    for r in runs])
        if p[:1] == ["runs"] and len(p) >= 2:
            r = RUNS.get(p[1])
            if not r:
                raise KeyError(p[1])
            if P and p[2:] == ["stop"]:
                r.stop()
                return self._send(200, {"ok": True})
            return self._send(200, r.view(int(q.get("since") or 0)))
        raise KeyError("/".join(parts))


def main():
    ap = argparse.ArgumentParser(description="jobscan in the browser")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    store.connect().close()              # creates/migrates seen.db
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"jobscan UI on {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
