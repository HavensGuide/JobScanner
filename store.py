"""
Persistence: a SQLite file that remembers every job UID already seen, so the
digest only ever shows genuinely new postings.

The DB file is committed back to the repo by the GitHub Action. That's a slight
abuse of git, but for a single-user job search it means zero infrastructure.

Scoring note: every job that reaches the scorer is recorded, including the ones
that scored too low to email. If weak jobs were not recorded they would look
new again on the next run and get re-scored forever.
"""

import os
import re
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("JOBSCAN_DB", "seen.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    uid         TEXT PRIMARY KEY,
    source      TEXT,
    title       TEXT,
    company     TEXT,
    location    TEXT,
    url         TEXT,
    posted      TEXT,
    salary      TEXT,
    description TEXT,
    desc_quality TEXT,
    first_seen  TEXT,
    status      TEXT DEFAULT 'new',
    score       INTEGER,
    tier        TEXT,
    reason      TEXT,
    scored_at   TEXT,
    track       TEXT,
    scorer      TEXT,
    scorer_model TEXT
);
"""

# Run AFTER migrations — on an existing DB the score column does not exist yet
# when the CREATE TABLE above is a no-op, and indexing a missing column throws.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_first_seen ON jobs(first_seen);
CREATE INDEX IF NOT EXISTS idx_score ON jobs(score);
"""

# Columns added after the first release. Older seen.db files get them on open.
MIGRATIONS = {
    "description": "TEXT",
    "desc_quality": "TEXT",
    "score": "INTEGER",
    "tier": "TEXT",
    "reason": "TEXT",
    "scored_at": "TEXT",
    # "career" or "bridge" — which rubric read this posting. NULL on rows
    # stored before the bridge track existed; score.py --backfill fills those
    # in from config.yaml when it re-reads them.
    "track": "TEXT",
    # Which METHOD produced score/tier/reason: "rubric", "jev", "keyword",
    # "rule" (temporal gate) or "none" (over cap). "anthropic" on rows
    # written before the rubric could run on other models; it means rubric.
    # NULL on rows from before methods existed, also rubric or keyword.
    "scorer": "TEXT",
    # Which MODEL, as "profile:model" from llm.label(). NULL on older rows
    # and on keyword/rule/none rows, which used no model.
    "scorer_model": "TEXT",
    # How jev reached the score: the band distribution, the expected
    # value, every cap and penalty, and each question's probability. The
    # `reason` column says what the verdict was in one line; this says how
    # it got there, which is what you need when a score looks wrong.
    "score_detail": "TEXT",
    # The drafting gates' verdict, recorded without drafting anything:
    # "recommended", "review" or "no". Written by `tailor.py screen`, which
    # runs the free deterministic rules and the local jev pre-scan and
    # stops there. Separate from `status`, which is where YOU put a
    # posting, and from `tier`, which is how well it scored.
    "draft_rec": "TEXT",
    "draft_reason": "TEXT",
    "draft_screened_at": "TEXT",
    # A flag you set, kept apart from `status` on purpose. Status is where a
    # posting is in the workflow and its values are exclusive, so marking
    # something applied would otherwise lose the fact that you had flagged
    # it. A bookmark survives every status change.
    "bookmarked": "INTEGER DEFAULT 0",
    "bookmarked_at": "TEXT",
    # Content identity: employer + title + place, normalized. The uid is the
    # SOURCE's id for a posting, and Adzuna does not keep those stable - the
    # same job comes back under a new id after a re-crawl, so uid dedupe let
    # it through as new on every run. Backfilled on open for older rows.
    "fingerprint": "TEXT",
}


def connect(path=None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    for name, sqltype in MIGRATIONS.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sqltype}")
    conn.executescript(INDEXES)
    # Older rows predate the column. Fill them once, or every one of them
    # looks like a job never seen before the first time it is re-listed.
    if conn.execute("SELECT COUNT(*) FROM jobs WHERE fingerprint IS NULL"
                    ).fetchone()[0]:
        rows = conn.execute(
            "SELECT uid, company, title, location FROM jobs "
            "WHERE fingerprint IS NULL").fetchall()
        conn.executemany(
            "UPDATE jobs SET fingerprint = ? WHERE uid = ?",
            [(fingerprint({"company": c, "title": t, "location": l}), u)
             for u, c, t, l in rows])
    conn.executescript(
        "CREATE INDEX IF NOT EXISTS idx_fingerprint ON jobs(fingerprint);")
    conn.commit()
    return conn


# Punctuation, case and the decorations employers add to a title vary between
# re-listings of one job. Strip them so the key is the job, not the ad.
_WS = re.compile(r"[^a-z0-9]+")
# "new" is deliberately NOT in here. It is load-bearing in New York, New
# Jersey, New Relic and plenty of employer names, and stripping it would
# make "New York" and "York" the same place.
_NOISE = re.compile(
    r"\b(hybrid|remote|onsite|on site|full time|fulltime|part time|parttime|"
    r"contract|temporary|temp|permanent|perm|w2|c2c|urgent|hiring|"
    r"immediate|immediately|job|opening|opportunity|position|"
    r"careers?|office|based)\b")


def _norm(s):
    s = _WS.sub(" ", str(s or "").lower()).strip()
    s = _NOISE.sub(" ", s)
    return " ".join(s.split())


def fingerprint(job):
    """
    Content identity for one posting: employer, title, place.

    Location stays in the key on purpose. NORC advertises the same role in
    Chicago and in DC and those are two jobs you would choose between, not
    one job listed twice. What this collapses is the same role, at the same
    employer, in the same place, arriving twice under different source ids -
    which is what Adzuna does on a re-crawl, and what put six NORC postings
    in two consecutive digests.
    """
    return "|".join((_norm(job.get("company")), _norm(job.get("title")),
                     _norm(job.get("location"))))


FIELDS = ["uid", "title", "company", "location", "salary", "url", "posted",
          "description", "desc_quality", "first_seen", "score", "tier",
          "reason", "source", "track", "scorer"]


def recent(conn, limit=40, source=None, min_score=None, order="score",
           since=None, until=None, track=None):
    """
    Jobs for the tailor CLI to pick from.

    order="score" ranks the WHOLE table by fit and returns the best `limit`.
    Ordering by first_seen and then sorting the slice was the bug: it returned
    the 40 newest rows, which on a sweep run are arbitrary, and buried every
    strong match.

    `track` filters to "career" or "bridge". Career also matches rows stored
    before the column existed, which are all career postings by definition —
    the bridge track did not exist when they were written.

    `since`/`until` bound first_seen as a half-open interval [since, until).
    Pass UTC ISO strings in the same format record() writes, so the comparison
    is a plain lexicographic one and the idx_first_seen index still applies.
    Callers are responsible for turning a local calendar day into UTC bounds —
    the DB only ever stores UTC.
    """
    q = f"SELECT {', '.join(FIELDS)} FROM jobs"
    where, args = [], []
    if source:
        where.append("source = ?")
        args.append(source)
    if min_score is not None:
        where.append("score >= ?")
        args.append(min_score)
    if since:
        where.append("first_seen >= ?")
        args.append(since)
    if until:
        where.append("first_seen < ?")
        args.append(until)
    if track == "career":
        where.append("(track = ? OR track IS NULL)")
        args.append(track)
    elif track:
        where.append("track = ?")
        args.append(track)
    if where:
        q += " WHERE " + " AND ".join(where)
    if order == "score":
        # NULLs last: unscored rows should never outrank a real score.
        q += (" ORDER BY CASE WHEN score IS NULL THEN 1 ELSE 0 END, "
              "score DESC, first_seen DESC LIMIT ?")
    else:
        q += " ORDER BY first_seen DESC, rowid DESC LIMIT ?"
    args.append(limit)
    return [dict(zip(FIELDS, row)) for row in conn.execute(q, args)]


def day_counts(conn, to_local_day, limit=14):
    """
    {local day -> row count} for the most recent `limit` days that have rows.

    `to_local_day` converts a stored UTC ISO timestamp to a local date string;
    the caller owns that because the DB has no opinion about your timezone.
    Used to answer "you asked for a day with nothing in it, here are the days
    that do" instead of a bare empty list.
    """
    counts = {}
    for (ts,) in conn.execute(
            "SELECT first_seen FROM jobs WHERE first_seen IS NOT NULL"):
        d = to_local_day(ts)
        if d:
            counts[d] = counts.get(d, 0) + 1
    return dict(sorted(counts.items(), reverse=True)[:limit])


def get_job(conn, uid):
    """
    Resolve a job by full uid, by the part after the last colon, or by any
    unique suffix.

    The suffix case matters because the CLI shortens long ids for display —
    an Ashby uuid is 36 characters and a Workday path is longer still. If the
    id printed by `list` did not resolve here, it would be unusable.
    """
    sel = f"SELECT {', '.join(FIELDS)} FROM jobs WHERE "
    for clause, args in (("uid = ?", (uid,)),
                         ("uid LIKE ?", (f"%:{uid}",)),
                         ("uid LIKE ?", (f"%{uid}",))):
        rows = conn.execute(sel + clause + " LIMIT 2", args).fetchall()
        if len(rows) == 1:
            return dict(zip(FIELDS, rows[0]))
        if len(rows) > 1:
            print(f"  ! {uid!r} matches more than one job; "
                  f"use the full uid from `tailor.py list --uids`")
            return dict(zip(FIELDS, rows[0]))
    return None


def filter_new(conn, jobs, report=True):
    """
    Return only jobs that are genuinely new.

    Two tests, because the source's id is not trustworthy on its own.
    Adzuna reissues ids on a re-crawl, so the same NORC posting arrived as
    new in consecutive runs. The fingerprint catches that, and also catches
    one employer listing the same role twice in a single run.
    """
    if not jobs:
        return []
    known = {row[0] for row in conn.execute("SELECT uid FROM jobs")}
    known_fp = {row[0] for row in conn.execute(
        "SELECT fingerprint FROM jobs WHERE fingerprint IS NOT NULL")}
    seen_uid, seen_fp = set(), set()
    fresh, dup_uid, dup_fp = [], 0, 0
    for j in jobs:
        uid = j["uid"]
        fp = j.get("fingerprint") or fingerprint(j)
        j["fingerprint"] = fp
        if uid in known or uid in seen_uid:
            dup_uid += 1
            continue
        if fp and (fp in known_fp or fp in seen_fp):
            dup_fp += 1
            continue
        seen_uid.add(uid)
        seen_fp.add(fp)
        fresh.append(j)
    if report and dup_fp:
        print(f"  - dedupe: {dup_fp} posting(s) already seen under a "
              f"different id (same employer, title and place)")
    return fresh


def record(conn, jobs):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        """INSERT OR IGNORE INTO jobs
           (uid, source, title, company, location, url, posted, salary,
            description, desc_quality, first_seen, score, tier, reason,
            scored_at, track, scorer, scorer_model, fingerprint,
            score_detail)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(j["uid"], j["source"], j["title"], j["company"], j["location"],
          j["url"], j["posted"], j["salary"], j.get("description", ""),
          j.get("desc_quality"), now,
          j.get("score"), j.get("tier"), j.get("reason"), j.get("scored_at"),
          j.get("track") or "career", j.get("scorer"), j.get("scorer_model"),
          j.get("fingerprint") or fingerprint(j), j.get("score_detail"))
         for j in jobs],
    )
    conn.commit()


def update_scores(conn, jobs):
    """Write scores onto rows that already exist. Used by --backfill."""
    conn.executemany(
        "UPDATE jobs SET score=?, tier=?, reason=?, scored_at=?, track=?, "
        "scorer=?, scorer_model=?, score_detail=? WHERE uid=?",
        [(j.get("score"), j.get("tier"), j.get("reason"), j.get("scored_at"),
          j.get("track") or "career", j.get("scorer"), j.get("scorer_model"),
          j.get("score_detail"), j["uid"])
         for j in jobs],
    )
    conn.commit()


QUALITY_RANK = {"none": 0, "excerpt": 1, "full": 2}


def refresh_descriptions(conn, jobs):
    """
    Upgrade stored text for jobs we have already seen.

    Older versions deliberately capped several full-source descriptions at
    2,000-4,000 characters. That often preserved only employer boilerplate and
    cut off the actual duties/qualifications. A normal fetch still encounters
    those open postings before filter_new() drops them, so use that opportunity
    to repair the stored row.

    Returns the number of rows whose description/quality improved. This does
    not change the score automatically; after the first real run with the patch,
    run `python score.py --backfill --rescore --limit 200` (or a larger limit)
    to recalculate recent stored jobs against the repaired text.
    """
    if not jobs:
        return 0
    changed = 0
    for j in jobs:
        uid = j.get("uid")
        new = (j.get("description") or "").strip()
        if not uid or not new:
            continue
        row = conn.execute(
            "SELECT description, desc_quality FROM jobs WHERE uid = ?", (uid,)
        ).fetchone()
        if not row:
            continue
        old = (row[0] or "").strip()
        old_q = row[1] or "none"
        new_q = j.get("desc_quality") or "none"
        better = (len(new) > len(old) + 100 or
                  QUALITY_RANK.get(new_q, 0) > QUALITY_RANK.get(old_q, 0))
        if not better:
            continue
        conn.execute(
            "UPDATE jobs SET description=?, desc_quality=? WHERE uid=?",
            (new, new_q, uid),
        )
        changed += 1
    if changed:
        conn.commit()
    return changed


# ---------------------------------------------------------------------------
# Optional: mirror new rows into a Google Sheet.
# Requires: pip install gspread google-auth
# Set GOOGLE_SHEET_ID and GOOGLE_SERVICE_ACCOUNT_JSON (the full JSON blob).
# ---------------------------------------------------------------------------
def append_to_sheet(jobs):
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not (sheet_id and creds_json and jobs):
        return False

    try:
        import json
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("  - sheets: gspread not installed, skipping")
        return False

    try:
        creds = Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        ws = gspread.authorize(creds).open_by_key(sheet_id).sheet1

        if not ws.get_all_values():
            ws.append_row(["First seen", "Score", "Tier", "Reason", "Source",
                           "Title", "Company", "Location", "Salary", "Posted",
                           "URL", "Status", "Track", "Scorer"])

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Track goes on the END, not in the middle: an existing sheet already
        # has eleven columns of history and inserting would misalign all of it.
        rows = [[today, j.get("score", ""), j.get("tier", ""),
                 j.get("reason", ""), j["source"], j["title"], j["company"],
                 j["location"], j["salary"], j["posted"], j["url"], "new",
                 j.get("track") or "career", j.get("scorer") or ""]
                for j in jobs]
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        print(f"  ! sheets append failed: {type(e).__name__}: {e}")
        return False


# ==========================================================================
# Cleaning up duplicates that are already stored
# ==========================================================================
def duplicate_groups(conn):
    """
    [(fingerprint, [rows])] for every job stored more than once.

    Rows come back newest-id-last so the keeper is deterministic: the one
    with the most text, then the one seen first, because the earliest
    sighting is the honest first_seen date for a job.
    """
    groups = {}
    for row in conn.execute(
            "SELECT uid, title, company, location, first_seen, score, "
            "status, LENGTH(COALESCE(description,'')) AS n, fingerprint "
            "FROM jobs WHERE fingerprint IS NOT NULL AND fingerprint != '||' "
            "ORDER BY fingerprint, first_seen"):
        groups.setdefault(row[8], []).append({
            "uid": row[0], "title": row[1], "company": row[2],
            "location": row[3], "first_seen": row[4], "score": row[5],
            "status": row[6], "chars": row[7]})
    return [(fp, rows) for fp, rows in groups.items() if len(rows) > 1]


def dedupe_existing(conn, apply=False):
    """
    Collapse stored duplicates. Returns (groups, rows_removed).

    The keeper is the row with the most description text, breaking ties on
    the earliest first_seen. A row you have acted on - saved, applied,
    dismissed - always wins over an untouched one, because that status is
    the only thing here you cannot reconstruct.
    """
    rank = {"applied": 3, "saved": 2, "dismissed": 1, "new": 0}
    groups = duplicate_groups(conn)
    removed = 0
    for fp, rows in groups:
        rows = sorted(rows, key=lambda r: (
            -rank.get(r["status"] or "new", 0), -(r["chars"] or 0),
            r["first_seen"] or ""))
        for r in rows[1:]:
            removed += 1
            if apply:
                conn.execute("DELETE FROM jobs WHERE uid = ?", (r["uid"],))
    if apply:
        conn.commit()
    return groups, removed


if __name__ == "__main__":
    import sys
    args = set(sys.argv[1:])
    conn = connect()
    groups, removed = dedupe_existing(conn, apply="--apply" in args)
    if not groups:
        print("No duplicates stored.")
        raise SystemExit(0)
    for fp, rows in groups[:40]:
        print(f"\n{rows[0]['company']} — {rows[0]['title']}")
        for r in rows:
            print(f"   {r['uid']:<34} {r['first_seen'][:10]}  "
                  f"score {str(r['score']):>4}  {r['chars']:>6} chars  "
                  f"{r['status'] or 'new'}")
    extra = len(groups) - 40
    if extra > 0:
        print(f"\n...and {extra} more group(s)")
    print(f"\n{len(groups)} duplicated job(s), {removed} row(s) "
          f"{'removed' if '--apply' in args else 'would be removed'}")
    if "--apply" not in args:
        print("Re-run with --apply to remove them.")
