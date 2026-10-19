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
    scored_at   TEXT
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
}


def connect(path=None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    for name, sqltype in MIGRATIONS.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sqltype}")
    conn.executescript(INDEXES)
    conn.commit()
    return conn


FIELDS = ["uid", "title", "company", "location", "salary", "url", "posted",
          "description", "desc_quality", "first_seen", "score", "tier",
          "reason", "source"]


def recent(conn, limit=40, source=None, min_score=None, order="score",
           since=None, until=None):
    """
    Jobs for the tailor CLI to pick from.

    order="score" ranks the WHOLE table by fit and returns the best `limit`.
    Ordering by first_seen and then sorting the slice was the bug: it returned
    the 40 newest rows, which on a sweep run are arbitrary, and buried every
    strong match.

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


def filter_new(conn, jobs):
    """Return only jobs whose uid isn't already in the DB."""
    if not jobs:
        return []
    known = {row[0] for row in conn.execute("SELECT uid FROM jobs")}
    seen_this_run = set()
    fresh = []
    for j in jobs:
        uid = j["uid"]
        if uid in known or uid in seen_this_run:
            continue
        seen_this_run.add(uid)
        fresh.append(j)
    return fresh


def record(conn, jobs):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        """INSERT OR IGNORE INTO jobs
           (uid, source, title, company, location, url, posted, salary,
            description, desc_quality, first_seen, score, tier, reason,
            scored_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(j["uid"], j["source"], j["title"], j["company"], j["location"],
          j["url"], j["posted"], j["salary"], j.get("description", ""),
          j.get("desc_quality"), now,
          j.get("score"), j.get("tier"), j.get("reason"), j.get("scored_at"))
         for j in jobs],
    )
    conn.commit()


def update_scores(conn, jobs):
    """Write scores onto rows that already exist. Used by --backfill."""
    conn.executemany(
        "UPDATE jobs SET score=?, tier=?, reason=?, scored_at=? WHERE uid=?",
        [(j.get("score"), j.get("tier"), j.get("reason"), j.get("scored_at"),
          j["uid"]) for j in jobs],
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
                           "URL", "Status"])

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = [[today, j.get("score", ""), j.get("tier", ""),
                 j.get("reason", ""), j["source"], j["title"], j["company"],
                 j["location"], j["salary"], j["posted"], j["url"], "new"]
                for j in jobs]
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        print(f"  ! sheets append failed: {type(e).__name__}: {e}")
        return False
