#!/usr/bin/env python3
"""
paths_ui.py - both tracks in the browser: career postings sorted by your
path rules, bridge postings with their scores.

    python paths_ui.py                 # http://127.0.0.1:8766, opens it
    python paths_ui.py --port 9001 --no-browser

Career: every career posting in seen.db is sorted LIVE with the rules in
config.yaml (`paths:`), so what you see is what the rules say now. Tabs:
Entry-level, Stretch, Filtered out; each posting shows the rule that put it
there.

Bridge: bridge postings are scored by the model, so this shows their stored
score, section and reason, in the bridge digest's sections. The rules panel
edits the `bridge:` block (title terms, exclusions, locations); its Preview
shows which stored bridge postings the new rules would drop or move to the
career track.

"Try a title" says which track a title goes to first, then where it lands.

The Rules panel edits the `paths:` block. Preview shows what would move
before anything is saved; Save backs up config.yaml first. "Apply to stored
postings" writes the sections into seen.db so ui.py and the digest agree;
it needs no model and takes seconds.

Same safety as ui.py: 127.0.0.1 only, the Host header must be this
machine, and every change needs an X-Jobscan header that another web page
cannot send. Standard library plus PyYAML.
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import webbrowser
from collections import Counter
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402

CONFIG = ROOT / "config.yaml"
BACKUPS = ROOT / "backups"
ON = "career:\n  scoring:\n    method: filter\n"
OFF = "# career:\n#   scoring:\n#     method: filter\n"


# --------------------------------------------------------------------------
# config.yaml
# --------------------------------------------------------------------------
def read_config():
    text = CONFIG.read_text(encoding="utf-8")
    return text, (yaml.safe_load(text) or {})


def backup():
    BACKUPS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = BACKUPS / f"config.yaml.{stamp}"
    shutil.copy2(CONFIG, dest)
    return dest


BLOCK = {"career": "paths", "bridge": "bridge"}


def paths_block(text, key="paths"):
    """(lines, start, end): line indexes of a top-level block."""
    lines = text.splitlines(keepends=True)
    s = next((i for i, l in enumerate(lines) if l.rstrip() == f"{key}:"),
             None)
    if s is None:
        return lines, None, None
    e = s + 1
    while e < len(lines):
        l = lines[e]
        if l.strip() and not l.startswith((" ", "\t")):
            break
        e += 1
    while e > s + 1 and not lines[e - 1].strip():
        e -= 1
    return lines, s, e


def current_rules_text(text, track="career"):
    key = BLOCK[track]
    lines, s, e = paths_block(text, key)
    if s is None:
        if key != "paths":
            raise ValueError(f"config.yaml has no `{key}:` block")
        return yaml.safe_dump({"paths": paths.DEFAULTS}, sort_keys=False,
                              width=76, allow_unicode=True)
    return "".join(lines[s:e])


def parse_rules(block, track="career"):
    """The root dict a rules text stands for, or ValueError."""
    key = BLOCK[track]
    try:
        d = yaml.safe_load(block) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"YAML: {e}")
    if not isinstance(d, dict) or set(d) != {key}:
        raise ValueError(f"the rules must be a single top-level `{key}:` "
                         f"block")
    if key == "paths":
        problems = paths.validate(d)
    else:
        b = d["bridge"] or {}
        problems = [f"bridge.{k} must be a list" for k in
                    ("tolerated", "exclude", "never", "title_must_match",
                     "location_allowlist") if k in b and
                    not isinstance(b[k], list)]
    if problems:
        raise ValueError("; ".join(problems))
    return d


def save_rules(block, track="career"):
    parse_rules(block, track)
    text, _ = read_config()
    lines, s, e = paths_block(text, BLOCK[track])
    block = block.rstrip("\n") + "\n"
    new = (text.rstrip("\n") + "\n\n" + block if s is None
           else "".join(lines[:s]) + block + "".join(lines[e:]))
    yaml.safe_load(new)                       # whole file still parses
    dest = backup()
    CONFIG.write_text(new, encoding="utf-8")
    return str(dest.name)


def filter_mode():
    try:
        import score
        _, cfg = read_config()
        return score.method_for(score.merge_scoring_cfg(cfg),
                                "career") == "filter"
    except Exception:
        return False


def set_filter_mode(on):
    text, _ = read_config()
    if on and OFF in text:
        new = text.replace(OFF, ON, 1)
    elif not on and ON in text:
        new = text.replace(ON, OFF, 1)
    elif on == filter_mode():
        return
    else:
        raise ValueError("could not find the career: scoring: method lines "
                         "to switch; edit config.yaml by hand")
    yaml.safe_load(new)
    backup()
    CONFIG.write_text(new, encoding="utf-8")


# --------------------------------------------------------------------------
# seen.db
# --------------------------------------------------------------------------
def db():
    try:
        import store
        conn = store.connect()
    except Exception:
        conn = sqlite3.connect(ROOT / "seen.db")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


COLS = ("uid", "title", "company", "location", "salary", "url", "posted",
        "first_seen", "source", "status", "bookmarked", "description",
        "tier", "scorer", "track", "score", "reason", "score_detail")


def career_rows(days):
    conn = db()
    have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    sel = ", ".join(c if c in have else f"NULL AS {c}" for c in COLS)
    where = "(track = 'career' OR track IS NULL)"
    args = []
    if days:
        since = (datetime.now(timezone.utc) - timedelta(days=float(days))
                 ).isoformat(timespec="seconds")
        where += " AND first_seen >= ?"
        args.append(since)
    rows = [dict(zip(COLS, r)) for r in conn.execute(
        f"SELECT {sel} FROM jobs WHERE {where}", args)]
    conn.close()
    return rows


def sort_rows(rows, root):
    p = paths.conf(root)
    out = []
    for r in rows:
        sec, label, why = paths.classify(r, p)
        out.append({k: r.get(k) for k in COLS if k != "description"}
                   | {"section": sec, "path": label, "reasons": why,
                      "has_text": bool((r.get("description") or "").strip())})
    order = {"entry": 0, "stretch": 1, "out": 2}
    out.sort(key=lambda r: str(r.get("posted") or r.get("first_seen") or ""),
             reverse=True)
    out.sort(key=lambda r: order[r["section"]])
    return out


def summarise(sorted_rows):
    c = Counter(r["section"] for r in sorted_rows)
    why = Counter(re.sub(r"\s*\(.*", "", r["reasons"][0])
                  for r in sorted_rows if r["section"] == "out"
                  and r["reasons"])
    return {"entry": c["entry"], "stretch": c["stretch"], "out": c["out"],
            "out_reasons": why.most_common()}


def bridge_rows(days):
    conn = db()
    have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    sel = ", ".join(c if c in have else f"NULL AS {c}" for c in COLS)
    where, args = "track = 'bridge'", []
    if days:
        since = (datetime.now(timezone.utc) - timedelta(days=float(days))
                 ).isoformat(timespec="seconds")
        where += " AND first_seen >= ?"
        args.append(since)
    rows = [dict(zip(COLS, r)) for r in conn.execute(
        f"SELECT {sel} FROM jobs WHERE {where}", args)]
    conn.close()
    return rows


def bridge_view(q):
    """Stored bridge postings in the bridge digest's sections, re-tiered at
    the CURRENT thresholds the way ui.py does."""
    import score
    _, root = read_config()
    scfg = score.merge_scoring_cfg(root)
    out = []
    for r in bridge_rows(q.get("days") or 30):
        t = r.get("tier")
        if not (t in ("review", "unscored") or r.get("score") is None):
            t = score.tier_of(r["score"], scfg, track="bridge")
        out.append({k: r.get(k) for k in COLS if k != "description"}
                   | {"section": t or "unscored",
                      "label": (r.get("reason") or "").split(" · ")[0],
                      "has_text": bool((r.get("description") or "")
                                       .strip())})
    out.sort(key=lambda r: -(r.get("score") or 0))
    c = Counter(r["section"] for r in out)
    labels = Counter(r["label"] for r in out if r["label"])
    strong, maybe = score.thresholds(scfg, "bridge")
    return {"rows": out, "summary": dict(c), "labels": labels.most_common(),
            "thresholds": {"strong_at": strong, "maybe_at": maybe},
            "method": score.method_for(scfg, "bridge")}


def jobs_view(q):
    _, root = read_config()
    n_bridge = len(bridge_rows(q.get("days") or 30))
    if q.get("track") == "bridge":
        return dict(bridge_view(q), track="bridge", n_career=len(
            career_rows(q.get("days") or 30)), n_bridge=n_bridge)
    rows = sort_rows(career_rows(q.get("days") or 30), root)
    return {"track": "career", "rows": rows, "summary": summarise(rows),
            "filter_mode": filter_mode(), "n_career": len(rows),
            "n_bridge": n_bridge}


def route_title(title, root):
    """'career', 'bridge' or None (dropped before scoring), with why."""
    import score
    career, bridge, excl, tol = score.track_terms(root)
    prof = score.professional_titles(root)
    never = (score.bridge_never(root) if hasattr(score, "bridge_never")
             else ())
    job = {"title": title}
    hx = next((t for t in (x.lower() for x in root.get("hard_exclude", []))
               if t in title.lower()), None)
    if hx:
        return None, f"dropped by hard_exclude ({hx!r})"
    try:
        t = score.classify_track(job, career, bridge, excl, tol, prof, never)
    except TypeError:
        t = score.classify_track(job, career, bridge, excl, tol, prof)
    if t is None:
        return None, "dropped before scoring (bridge.exclude or bridge.never)"
    low = title.lower()
    if t == "career":
        why = next((w for w in career if w in low), None)
        return t, (f"title has {why!r}" if why else
                   "no track word; unmatched titles default to career")
    why = next((w for w in list(tol) + list(bridge) if w in low), None)
    return t, f"title has {why!r}" if why else ""


def bridge_preview(block, days):
    """Which stored bridge postings the new bridge rules would drop, move
    to career, or put outside the commute. Postings the OLD rules dropped
    were never stored, so additions cannot be shown."""
    import main
    import score
    _, root = read_config()
    new = dict(root, bridge=parse_rules(block, "bridge")["bridge"])
    career, bridge, excl, tol = score.track_terms(new)
    prof = score.professional_titles(new)
    never = (score.bridge_never(new) if hasattr(score, "bridge_never")
             else ())
    locs = [l.lower() for l in (new.get("bridge") or {}).get(
        "location_allowlist", [])]
    block_l = [b.lower() for b in new.get("location_blocklist", [])]
    moved = []
    rows = bridge_rows(days)
    for r in rows:
        try:
            t = score.classify_track(r, career, bridge, excl, tol, prof, never)
        except TypeError:
            t = score.classify_track(r, career, bridge, excl, tol, prof)
        if t is None:
            moved.append({"title": r["title"], "was": "bridge",
                          "section": "dropped", "reasons": ["exclude/never"]})
        elif t == "career":
            moved.append({"title": r["title"], "was": "bridge",
                          "section": "career", "reasons": ["career word"]})
        elif not main.location_ok(r, locs, block_l):
            moved.append({"title": r["title"], "was": "bridge",
                          "section": "dropped",
                          "reasons": [f"location {r.get('location')!r}"]})
    c = Counter(m["section"] for m in moved)
    return {"bridge": True, "total": len(rows), "moved": moved[:300],
            "moved_count": len(moved), "dropped": c["dropped"],
            "to_career": c["career"]}


def preview(block, days, track="career"):
    if track == "bridge":
        return bridge_preview(block, days)
    _, root = read_config()
    new_root = dict(root, paths=parse_rules(block)["paths"])
    raw = career_rows(days)
    before = {r["uid"]: r for r in sort_rows(raw, root)}
    after = sort_rows(raw, new_root)
    moved = [dict(r, was=before[r["uid"]]["section"]) for r in after
             if before.get(r["uid"]) and
             before[r["uid"]]["section"] != r["section"]]
    return {"summary": summarise(after),
            "before": summarise(list(before.values())), "moved": moved[:300],
            "moved_count": len(moved)}


def apply_to_stored():
    """Write every career posting's section into seen.db (filter mode)."""
    if not filter_mode():
        raise ValueError("switch filter mode on first; otherwise the next "
                         "run would overwrite these with model scores")
    _, root = read_config()
    p = paths.conf(root)
    conn = db()
    have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    sel = ", ".join(c if c in have else f"NULL AS {c}" for c in COLS)
    rows = [dict(zip(COLS, r)) for r in conn.execute(
        f"SELECT {sel} FROM jobs WHERE (track = 'career' OR track IS NULL)")]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sets = ["tier = ?", "reason = ?", "score = NULL"]
    if "scorer" in have:
        sets.append("scorer = 'filter'")
    if "scorer_model" in have:
        sets.append("scorer_model = NULL")
    if "scored_at" in have:
        sets.append("scored_at = ?")
    n = 0
    for r in rows:
        tier, why = paths.verdict(r, p)
        args = [tier, why] + ([now] if "scored_at" in have else []) + [r["uid"]]
        n += conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE uid = ?",
                          args).rowcount
    conn.commit()
    conn.close()
    return n


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "jobscan-paths"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _route(self, method):
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        if host not in ("127.0.0.1", "localhost"):
            return self._send(403, {"error": "local only"})
        if method == "POST" and self.headers.get("X-Jobscan") != "1":
            return self._send(403, {"error": "missing X-Jobscan header"})
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if method == "GET" and url.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode("utf-8"),
                                  "text/html; charset=utf-8")
            if method == "GET" and url.path == "/api/jobs":
                return self._send(200, jobs_view(q))
            if method == "GET" and url.path == "/api/rules":
                text, _ = read_config()
                return self._send(200, {"text": current_rules_text(
                    text, q.get("track") or "career")})
            if method == "POST" and url.path == "/api/preview":
                b = self._body()
                return self._send(200, preview(b.get("text") or "",
                                               b.get("days") or 30,
                                               b.get("track") or "career"))
            if method == "POST" and url.path == "/api/rules":
                b = self._body()
                return self._send(200, {"backup": save_rules(
                    b.get("text") or "", b.get("track") or "career")})
            if method == "POST" and url.path == "/api/test":
                b = self._body()
                _, root = read_config()
                tr = b.get("track") or "career"
                if b.get("rules"):
                    key = BLOCK[tr]
                    root = dict(root, **{key: parse_rules(b["rules"], tr)[key]})
                title = b.get("title") or ""
                route, rwhy = route_title(title, root)
                if route != "career":
                    return self._send(200, {"route": route, "route_why": rwhy,
                                            "section": None, "path": None,
                                            "reasons": []})
                sec, label, why = paths.classify(
                    {"title": title, "description": b.get("desc") or ""},
                    paths.conf(root))
                return self._send(200, {"route": route, "route_why": rwhy,
                                        "section": sec, "path": label,
                                        "reasons": why})
            if method == "POST" and url.path == "/api/mode":
                set_filter_mode(bool(self._body().get("on")))
                return self._send(200, {"filter_mode": filter_mode()})
            if method == "POST" and url.path == "/api/apply":
                return self._send(200, {"updated": apply_to_stored()})
            return self._send(404, {"error": f"not found: {url.path}"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>jobscan · tracks</title>
<style>
:root{--bg:#fbfaf8;--card:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#e7e4de;
--entry:#1a7f37;--stretch:#9a6700;--out:#77766f;--review:#8250df;--accent:#2f4f9e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:14px 22px 0}
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:650;letter-spacing:-.01em}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}
.seg button{border:0;border-radius:0;padding:6px 14px;background:#fff;color:var(--mute)}
.seg button.on{background:var(--ink);color:#fff}.seg button b{font-weight:600;margin-left:5px;opacity:.8}
.mode{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);background:#fff;cursor:pointer}
.mode.on{background:#e8f4ec;border-color:#b5dcc1;color:var(--entry)}
.note{font-size:12px;color:var(--mute)}
.grow{flex:1}input,select,textarea,button{font:inherit}
input[type=search],select{padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:#fff}
button{padding:6px 12px;border-radius:7px;border:1px solid var(--line);background:#fff;cursor:pointer}
button.primary{background:var(--ink);color:#fff;border-color:var(--ink)}button:disabled{opacity:.5;cursor:default}
.tabs{display:flex;gap:4px;margin-top:12px;flex-wrap:wrap}
.tab{padding:8px 14px;border:0;border-bottom:2px solid transparent;background:none;border-radius:0;color:var(--mute)}
.tab.active{color:var(--ink);border-bottom-color:var(--ink);font-weight:600}.tab b{font-weight:600;margin-left:4px}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:22px;padding:18px 22px}
@media(max-width:1000px){main{grid-template-columns:1fr}}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 12px}
.chip{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);background:#fff;cursor:pointer}
.chip.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.job{background:var(--card);border:1px solid var(--line);border-left-width:3px;border-radius:9px;padding:11px 13px;margin-bottom:8px}
.job a{color:var(--ink);font-weight:600;text-decoration:none}.job a:hover{text-decoration:underline}
.meta{color:var(--mute);font-size:12.5px;margin-top:2px}.why{font-size:12.5px;margin-top:5px}.why .p{color:var(--accent);font-weight:550}
.c-entry,.c-strong{border-left-color:var(--entry)}.c-stretch,.c-maybe{border-left-color:var(--stretch)}
.c-review{border-left-color:var(--review)}.c-out,.c-weak,.c-unscored{border-left-color:#cfccc5}
.badge{display:inline-block;min-width:26px;text-align:center;color:#fff;font-size:11px;font-weight:600;border-radius:4px;padding:1px 5px;margin-right:7px;vertical-align:1px}
.tag{font-size:11px;color:var(--mute);border:1px solid var(--line);border-radius:4px;padding:0 5px;margin-left:6px}
details{margin-top:5px}summary{font-size:12px;color:var(--mute);cursor:pointer}
pre{font:11.5px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;margin:5px 0 0;color:#444}
aside{position:sticky;top:120px;align-self:start}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.panel h2{font-size:13px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute)}
textarea{width:100%;height:340px;border:1px solid var(--line);border-radius:7px;padding:9px;font:12px/1.45 ui-monospace,Consolas,monospace;background:#fcfcfb;resize:vertical}
.row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;align-items:center}
.msg{font-size:12.5px;margin-top:8px;white-space:pre-wrap}.err{color:#b42318}.ok{color:var(--entry)}
.moved{max-height:260px;overflow:auto;font-size:12.5px;margin-top:8px;border-top:1px solid var(--line);padding-top:6px}
.moved div{padding:2px 0}.arrow{color:var(--mute)}
.small{font-size:12px;color:var(--mute)}.empty{color:var(--mute);padding:30px 0;text-align:center}
#testres{font-size:13px;margin-top:6px}
</style></head><body>
<header><div class="top">
<h1>jobscan</h1>
<div class="seg"><button id="t-career" class="on">Career<b id="nc"></b></button><button id="t-bridge">Bridge<b id="nb"></b></button></div>
<button id="mode" class="mode" title="Whether runs and the digest sort career postings with these rules">filter mode: …</button>
<span id="bnote" class="note" hidden></span>
<div class="grow"></div>
<input id="q" type="search" placeholder="Search title, company, reason" size="28">
<select id="days"><option value="7">last 7 days</option><option value="30" selected>last 30 days</option>
<option value="90">last 90 days</option><option value="0">all time</option></select>
</div>
<div class="tabs" id="tabs"></div></header>
<main><section><div id="chips" class="chips"></div><div id="list"></div></section>
<aside>
<div class="panel"><h2>Try a title</h2>
<input id="ttitle" type="search" placeholder="e.g. Research Associate II" style="width:100%">
<textarea id="tdesc" style="height:70px;margin-top:6px" placeholder="optional: paste requirements text"></textarea>
<div id="testres"></div></div>
<div class="panel"><h2 id="rtitle">Rules</h2>
<div class="small" id="rhelp"></div>
<textarea id="rules" spellcheck="false"></textarea>
<div class="row"><button id="preview">Preview</button><button id="save" class="primary">Save</button><button id="reset">Reload</button></div>
<div id="rmsg" class="msg"></div><div id="moved" class="moved" hidden></div></div>
<div class="panel" id="apanel"><h2>Stored career postings</h2>
<div class="small">Write the sections into seen.db, so ui.py and the next digest show the same thing. No model; takes seconds. Needs filter mode on.</div>
<div class="row"><button id="apply">Apply to stored postings</button></div><div id="amsg" class="msg"></div></div>
</aside></main>
<script>
const $=s=>document.querySelector(s);
const H={"X-Jobscan":"1","Content-Type":"application/json"};
const TABS={career:[["entry","Entry-level","var(--entry)"],["stretch","Stretch","var(--stretch)"],["out","Filtered out","var(--out)"]],
 bridge:[["strong","Take it if you need it now","var(--entry)"],["review","Manual review","var(--review)"],["maybe","Workable with a catch","var(--stretch)"],["weak","Probably not","var(--out)"],["unscored","Not scored","var(--out)"]]};
const HELP={career:"Your <code>paths:</code> block: career fields and level rules. Preview shows what would move; nothing is saved until you press Save.",
 bridge:"Your <code>bridge:</code> block: which titles are bridge work, what is excluded, and the commute. Preview shows which stored bridge postings the new rules would drop or send to career. Scores come from the model and do not change here."};
let track="career",rows=[],tab="entry",chip=null,data=null;const ruleText={};
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:"POST",headers:H,body:JSON.stringify(body)});
 const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
function drawTabs(){const t=$("#tabs");t.innerHTML="";const sm=data?data.summary:{};
 for(const[k,l,c]of TABS[track]){const n=sm[k]||0;if(track==="bridge"&&k==="unscored"&&!n)continue;
  const b=document.createElement("button");b.className="tab"+(tab===k?" active":"");b.innerHTML=`${l}<b style="color:${c}">${n}</b>`;
  b.onclick=()=>{tab=k;chip=null;drawTabs();render()};t.append(b)}}
async function load(){$("#list").innerHTML='<div class="empty">Loading…</div>';
 data=await api(`/api/jobs?track=${track}&days=${$("#days").value}`);rows=data.rows;
 $("#nc").textContent=data.n_career;$("#nb").textContent=data.n_bridge;
 $("#mode").hidden=track!=="career";$("#bnote").hidden=track!=="bridge";$("#apanel").hidden=track!=="career";
 if(track==="career")setMode(data.filter_mode);
 else $("#bnote").textContent=`scored by ${data.method} · sections at ${data.thresholds.strong_at}+ and ${data.thresholds.maybe_at}+`;
 drawTabs();render()}
function setMode(on){const b=$("#mode");b.classList.toggle("on",on);b.textContent="filter mode: "+(on?"on":"off");b.dataset.on=on?1:0}
function chipsFor(){if(track==="career"&&tab==="out")return data.summary.out_reasons;
 if(track==="bridge"){const c={};for(const r of rows)if(r.section===tab&&r.label)c[r.label]=(c[r.label]||0)+1;return Object.entries(c).sort((a,b)=>b[1]-a[1])}return[]}
function keyOf(r){return track==="career"?(r.reasons[0]||"").replace(/\s*\(.*/,""):r.label}
function render(){const q=$("#q").value.trim().toLowerCase();const ch=$("#chips");ch.innerHTML="";
 for(const[k,n]of chipsFor()){const c=document.createElement("button");c.className="chip"+(chip===k?" on":"");c.textContent=`${k} · ${n}`;
  c.onclick=()=>{chip=chip===k?null:k;render()};ch.append(c)}
 let list=rows.filter(r=>r.section===tab);if(chip)list=list.filter(r=>keyOf(r)===chip);
 if(q)list=list.filter(r=>[r.title,r.company,r.location,r.reason,(r.reasons||[]).join(" "),r.path].join(" ").toLowerCase().includes(q));
 const col=(TABS[track].find(t=>t[0]===tab)||[,,"var(--out)"])[2];
 const one=r=>{const meta=esc([r.company,r.location,r.salary,r.posted||String(r.first_seen||"").slice(0,10)].filter(Boolean).join(" · "));
  const tags=(r.status&&r.status!=="new"?`<span class="tag">${esc(r.status)}</span>`:"")+(r.has_text?"":`<span class="tag" title="No description stored">title only</span>`);
  if(track==="career")return`<div class="job c-${r.section}"><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a>${tags}
   <div class="meta">${meta}</div><div class="why">${r.path?`<span class="p">${esc(r.path)}</span> · `:""}${esc((r.reasons||[]).join("; "))}</div></div>`;
  const badge=r.score!=null?`<span class="badge" style="background:${col}">${r.score}</span>`:"";
  return`<div class="job c-${r.section}">${badge}<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a>${tags}
   <div class="meta">${meta}</div><div class="why">${esc(r.reason||"")}</div>${r.score_detail?`<details><summary>how this was scored</summary><pre>${esc(r.score_detail)}</pre></details>`:""}</div>`};
 const shown=list.slice(0,400);
 $("#list").innerHTML=shown.length?shown.map(one).join("")+(list.length>400?`<div class="empty">…and ${list.length-400} more; search to narrow</div>`:""):'<div class="empty">Nothing here.</div>'}
function switchTrack(t){if(t===track)return;ruleText[track]=$("#rules").value;track=t;tab=TABS[t][0][0];chip=null;
 $("#t-career").classList.toggle("on",t==="career");$("#t-bridge").classList.toggle("on",t==="bridge");
 $("#rtitle").textContent=t==="career"?"Rules · career paths":"Rules · bridge track";$("#rhelp").innerHTML=HELP[t];
 $("#rmsg").textContent="";$("#moved").hidden=true;
 if(ruleText[t]!==undefined)$("#rules").value=ruleText[t];else loadRules();load();runTest()}
$("#t-career").onclick=()=>switchTrack("career");$("#t-bridge").onclick=()=>switchTrack("bridge");
$("#q").oninput=render;$("#days").onchange=load;
$("#mode").onclick=async()=>{try{const j=await api("/api/mode",{on:$("#mode").dataset.on!=="1"});setMode(j.filter_mode)}catch(e){box("#amsg",e.message,true)}};
function box(sel,t,bad){const m=$(sel);m.className="msg "+(bad?"err":"ok");m.textContent=t}
async function loadRules(){try{const j=await api("/api/rules?track="+track);$("#rules").value=j.text;delete ruleText[track]}catch(e){$("#rules").value="";box("#rmsg",e.message,true)}
 $("#moved").hidden=true}
$("#reset").onclick=()=>{loadRules();$("#rmsg").textContent=""};
$("#preview").onclick=async()=>{try{const j=await api("/api/preview",{text:$("#rules").value,days:+$("#days").value,track});
 if(j.bridge)box("#rmsg",`Of ${j.total} stored bridge postings: ${j.dropped} would be dropped, ${j.to_career} would go to the career track.\nPostings the current rules already drop were never stored, so new additions cannot be shown.`,false);
 else{const b=j.before,a=j.summary;box("#rmsg",`Entry ${b.entry} → ${a.entry}   Stretch ${b.stretch} → ${a.stretch}   Filtered out ${b.out} → ${a.out}\n${j.moved_count} posting(s) would move.`,false)}
 const m=$("#moved");m.hidden=!j.moved.length;m.innerHTML=j.moved.map(r=>`<div><span class="arrow">${esc(r.was)} → ${esc(r.section)}</span> ${esc(r.title)} <span class="small">${esc((r.reasons||[]).join("; "))}</span></div>`).join("")}
 catch(e){box("#rmsg",e.message,true)}};
$("#save").onclick=async()=>{try{const j=await api("/api/rules",{text:$("#rules").value,track});box("#rmsg","Saved. Previous config.yaml kept as backups/"+j.backup,false);$("#moved").hidden=true;delete ruleText[track];load()}
 catch(e){box("#rmsg",e.message,true)}};
let armed=false;$("#apply").onclick=async()=>{if(!armed){armed=true;$("#apply").textContent="Click again to confirm";setTimeout(()=>{armed=false;$("#apply").textContent="Apply to stored postings"},4000);return}
 armed=false;$("#apply").textContent="Apply to stored postings";$("#apply").disabled=true;
 try{const j=await api("/api/apply",{});box("#amsg",`Updated ${j.updated} stored career posting(s).`,false)}catch(e){box("#amsg",e.message,true)}$("#apply").disabled=false};
let tt;function runTest(){clearTimeout(tt);tt=setTimeout(async()=>{const t=$("#ttitle").value.trim();if(!t){$("#testres").innerHTML="";return}
 try{const j=await api("/api/test",{title:t,desc:$("#tdesc").value,rules:$("#rules").value,track});
  const route=j.route?`<div><b>${j.route==="career"?"Career":"Bridge"} track</b> <span class="small">· ${esc(j.route_why)}</span></div>`:`<div><b class="err">Never scored</b> <span class="small">· ${esc(j.route_why)}</span></div>`;
  let sec="";if(j.section){const c={entry:"var(--entry)",stretch:"var(--stretch)",out:"var(--out)"}[j.section];
   sec=`<div style="margin-top:3px"><b style="color:${c}">${j.section}</b>${j.path?" · "+esc(j.path):""}<div class="small">${esc(j.reasons.join("; "))}</div></div>`}
  else if(j.route==="bridge")sec=`<div class="small" style="margin-top:3px">Bridge postings are scored by the model on contact level, hiring bar and pay.</div>`;
  $("#testres").innerHTML=route+sec}catch(e){$("#testres").innerHTML=`<span class="err">${esc(e.message)}</span>`}},250)}
$("#ttitle").oninput=runTest;$("#tdesc").oninput=runTest;
$("#rhelp").innerHTML=HELP.career;$("#rtitle").textContent="Rules · career paths";
load().catch(e=>$("#list").innerHTML=`<div class="empty err">${esc(e.message)}</div>`);loadRules();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="career paths in the browser")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not CONFIG.exists():
        sys.exit(f"no {CONFIG}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"career paths on {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
