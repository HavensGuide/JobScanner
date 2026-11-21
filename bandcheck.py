#!/usr/bin/env python3
"""
bandcheck.py - did changing a band change the right things?

    cp seen.db seen.before.db
    python score.py --backfill --rescore --scorer jev --yes --limit 2000
    python bandcheck.py --before seen.before.db --after seen.db --track career

score.py --compare is for two SCORERS: jev against the rubric, or one model
against another. It cannot answer this question, because both sides here are
jev and the only thing that differs is a band in config.yaml. Comparing jev
to jev scores the same postings twice with the same rules and reports perfect
agreement, which is true and useless.

So this compares two SNAPSHOTS of the same scorer instead: the database
before the rescore and the database after it.

What to look at, in order:

  TIER FLIPS. The only movement that changes what you see. A score going
  62 -> 68 changes nothing; a score crossing strong_at changes the digest.

  THE MOVERS. Read the titles, not the mean. A band change is correct when
  it moves the postings you meant it to move, and a mean shift of +3 can be
  one posting moving 60 points or sixty moving one.

  RANK AGREEMENT. Spearman between the two score sets. Near 1.0 means you
  nudged the scale; below about 0.9 means you changed what the scorer is
  measuring, which may be what you wanted but should not be a surprise.
"""

import argparse
import sqlite3
import sys


def load(path, track):
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        sys.exit(f"{path}: {e}")
    where = "score IS NOT NULL"
    args = []
    if track:
        where += " AND (track = ?" + (" OR track IS NULL)" if track == "career"
                                      else ")")
        args.append(track)
    try:
        rows = conn.execute(
            f"SELECT uid, title, company, score, tier, reason FROM jobs "
            f"WHERE {where}", args).fetchall()
    except sqlite3.Error as e:
        sys.exit(f"{path}: {e}")
    conn.close()
    return {r[0]: {"title": r[1], "company": r[2], "score": r[3],
                   "tier": r[4], "reason": r[5]} for r in rows}


def spearman(pairs):
    """Rank correlation, computed here to avoid a scipy dependency."""
    n = len(pairs)
    if n < 3:
        return None

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:                      # average ties, or they bias rho
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    a = ranks([p[0] for p in pairs])
    b = ranks([p[1] for p in pairs])
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else None


def main():
    ap = argparse.ArgumentParser(
        description="Compare two snapshots of the same scorer")
    ap.add_argument("--before", default="seen.before.db")
    ap.add_argument("--after", default="seen.db")
    ap.add_argument("--track", default="career",
                    help="career, bridge, or empty for both")
    ap.add_argument("--movers", type=int, default=12,
                    help="how many biggest movers to print per direction")
    args = ap.parse_args()

    before, after = load(args.before, args.track), load(args.after, args.track)
    shared = [u for u in before if u in after]
    if not shared:
        sys.exit("no postings scored in both snapshots. Did the rescore run, "
                 "and is --track right?")

    pairs, moved, flips = [], [], []
    for u in shared:
        b, a = before[u], after[u]
        pairs.append((b["score"], a["score"]))
        d = a["score"] - b["score"]
        if d:
            moved.append((d, b, a))
        if (b["tier"] or "") != (a["tier"] or ""):
            flips.append((b, a))

    n = len(shared)
    mean = sum(a - b for b, a in pairs) / n
    rho = spearman(pairs)

    print(f"{n} posting(s) scored in both snapshots"
          + (f", track {args.track}" if args.track else ""))
    print(f"  unchanged      {n - len(moved)}")
    print(f"  moved          {len(moved)}")
    print(f"  mean shift     {mean:+.1f}")
    print(f"  rank agreement {rho:.3f}" if rho is not None else
          "  rank agreement n/a")
    if rho is not None and rho < 0.9:
        print("  ! below 0.90: the scorer is measuring something different "
              "now, not just shifted. Make sure that is what you wanted.")

    print(f"\nTIER FLIPS ({len(flips)}) - the only changes that reach the "
          f"digest")
    if not flips:
        print("  none")
    for b, a in sorted(flips, key=lambda p: -(p[1]["score"] - p[0]["score"]))[:30]:
        print(f"  {b['tier'] or '?':>8} -> {a['tier'] or '?':<8} "
              f"{b['score']:>3} -> {a['score']:<3}  {b['title'][:46]}")
        print(f"           {(a['reason'] or '')[:86]}")

    # Split by direction rather than slicing a sorted list from both ends:
    # with only a few movers the same rows appear under both headings.
    rises = sorted((m for m in moved if m[0] > 0), key=lambda m: -m[0])
    falls = sorted((m for m in moved if m[0] < 0), key=lambda m: m[0])
    for label, group in (("ROSE", rises), ("FELL", falls)):
        if not group:
            continue
        print(f"\n{label} ({len(group)})")
        for d, b, a in group[:args.movers]:
            print(f"  {d:+4}  {b['score']:>3} -> {a['score']:<3}  "
                  f"{b['company'][:22]:<24} {b['title'][:40]}")
        if len(group) > args.movers:
            print(f"  ...and {len(group) - args.movers} more")

    print("\nRead the titles above, not the mean. The change is right if it "
          "moved\nthe postings you meant it to move.")


if __name__ == "__main__":
    main()
