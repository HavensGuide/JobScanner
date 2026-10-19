"""
Build and send the email digest, grouped by fit tier rather than by source.

The design goal: nothing vanishes silently. Strong matches lead, maybes get a
compact section, and weak ones still appear as a single line of titles with
their scores. If the scorer is wrong, you can see that it was wrong.
"""

import os
import html
import smtplib
from collections import defaultdict
from email.message import EmailMessage

SOURCE_LABEL = {
    "adzuna": "Adzuna",
    "usajobs": "USAJOBS",
    "nyc": "NYC gov",
}

TIER_HEADING = {
    "strong": ("Worth applying to", "#1a7f37"),
    "review": ("Manual review — promising / incomplete", "#8250df"),
    "maybe": ("Worth a look", "#9a6700"),
    "unscored": ("Not scored — review these yourself", "#0969da"),
    "weak": ("Probably not", "#666666"),
}

# Same tiers, different words. A bridge score answers "could you get it and
# stand it", so "worth applying to" would be the wrong promise on that side.
BRIDGE_TIER_HEADING = {
    "strong": ("Take it if you need it now", "#1a7f37"),
    "review": ("Manual review — pay or duties not visible", "#8250df"),
    "maybe": ("Workable with a catch", "#9a6700"),
    "unscored": ("Not scored — review these yourself", "#0969da"),
    "weak": ("Probably not", "#666666"),
}

ORDER = ["strong", "review", "maybe", "unscored", "weak"]

# Career first, always. Bridge postings are numerous and easy to qualify for,
# so without a hard split they would crowd out the jobs actually worth having.
TRACK_ORDER = ["career", "bridge"]
TRACK_HEADING = {
    "career": "Research, policy, and quantitative",
    "bridge": "Bridge work",
}


def headings_for(track):
    return BRIDGE_TIER_HEADING if track == "bridge" else TIER_HEADING


def split_tracks(jobs):
    """{track -> [jobs]}, preserving TRACK_ORDER and skipping empty tracks."""
    by_track = defaultdict(list)
    for j in jobs:
        by_track[j.get("track") or "career"].append(j)
    return [(t, by_track[t]) for t in TRACK_ORDER if by_track.get(t)]


def group(jobs):
    by_tier = defaultdict(list)
    for j in jobs:
        by_tier[j.get("tier") or "unscored"].append(j)
    for tier in by_tier:
        by_tier[tier].sort(key=lambda j: -(j.get("score") or 0))
    return by_tier


def counts(jobs, tier, track=None):
    return sum(1 for j in jobs
               if (j.get("tier") or "unscored") == tier
               and (track is None or (j.get("track") or "career") == track))


def _meta(j):
    src = SOURCE_LABEL.get(j.get("source"), j.get("source", ""))
    return " · ".join(filter(None, [j.get("company"), j.get("location"),
                                    j.get("salary"), src]))


# ---------------------------------------------------------------------------
def build_html(jobs):
    tracks = split_tracks(jobs)
    strong = counts(jobs, "strong", "career")
    review = counts(jobs, "review", "career")
    bridge_n = sum(1 for j in jobs if (j.get("track") or "career") == "bridge"
                   and (j.get("tier") or "") in ("strong", "review", "maybe"))

    p = ["<div style='font-family:-apple-system,Segoe UI,Helvetica,sans-serif;"
         "max-width:640px;color:#1a1a1a;line-height:1.5'>"]
    # On a --track bridge run there is no career news, and leading with
    # "0 worth applying to" reads like a bad day rather than a narrowed run.
    if not any((j.get("track") or "career") == "career" for j in jobs):
        head = f"{bridge_n} bridge"
    else:
        head = f"{strong} worth applying to · {review} manual review"
        if bridge_n:
            head += f" · {bridge_n} bridge"
    p.append(f"<h2 style='margin:0 0 2px'>{head}</h2>"
             f"<div style='color:#666;font-size:13px;margin-bottom:6px'>"
             f"{len(jobs)} new postings scored</div>")

    if counts(jobs, "unscored"):
        p.append("<div style='background:#fff8e1;border-left:3px solid #f0ad4e;"
                 "padding:8px 10px;font-size:13px;margin:12px 0'>"
                 "Some postings could not be scored by the model. They are "
                 "listed below unfiltered.</div>")

    show_track_headers = len(tracks) > 1
    for track, items_in_track in tracks:
        by_tier = group(items_in_track)
        headings = headings_for(track)

        if show_track_headers:
            p.append(f"<div style='margin:30px 0 0;padding-top:14px;"
                     f"border-top:2px solid #1a1a1a'>"
                     f"<span style='font-size:15px;font-weight:700'>"
                     f"{TRACK_HEADING[track]}</span>"
                     f"<span style='color:#888;font-size:13px'> · "
                     f"{len(items_in_track)}</span></div>")

        for tier in ORDER:
            items = by_tier.get(tier) or []
            if not items:
                continue
            label, color = headings[tier]

            p.append(f"<h3 style='margin:24px 0 8px;font-size:13px;"
                     f"text-transform:uppercase;letter-spacing:.06em;"
                     f"color:{color}'>{label} ({len(items)})</h3>")

            # Weak tier collapses to titles only — visible, but not competing
            # for attention with the things you should actually read.
            if tier == "weak":
                rows = "".join(
                    f"<div style='font-size:12px;color:#888;margin:0 0 3px'>"
                    f"<a href='{html.escape(j.get('url', ''))}' "
                    f"style='color:#888'>{html.escape(j.get('title', ''))}</a> "
                    f"<span style='color:#aaa'>· {j.get('score')} · "
                    f"{html.escape(j.get('reason') or '')}</span></div>"
                    for j in items)
                p.append(f"<div style='padding-left:10px'>{rows}</div>")
                continue

            for j in items:
                score = j.get("score")
                badge = ""
                if score is not None:
                    badge = (f"<span style='display:inline-block;min-width:26px;"
                             f"text-align:center;background:{color};color:#fff;"
                             f"font-size:11px;font-weight:600;border-radius:3px;"
                             f"padding:1px 5px;margin-right:6px'>{score}</span>")
                reason = html.escape(j.get("reason") or "")
                p.append(
                    f"<div style='margin:0 0 14px;padding-left:10px;"
                    f"border-left:2px solid #e0e0e0'>"
                    f"{badge}<a href='{html.escape(j.get('url', ''))}' "
                    f"style='font-weight:600;color:#1a1a1a;"
                    f"text-decoration:none'>"
                    f"{html.escape(j.get('title', ''))}</a><br>"
                    f"<span style='color:#666;font-size:13px'>"
                    f"{html.escape(_meta(j))}</span>"
                    + (f"<br><span style='color:#444;font-size:13px;"
                       f"font-style:italic'>{reason}</span>" if reason else "")
                    + "</div>")

    p.append("</div>")
    return "\n".join(p)


def build_text(jobs):
    tracks = split_tracks(jobs)
    bridge_n = sum(1 for j in jobs if (j.get("track") or "career") == "bridge"
                   and (j.get("tier") or "") in ("strong", "review", "maybe"))
    if jobs and not any((j.get("track") or "career") == "career" for j in jobs):
        head = f"{bridge_n} bridge"
    else:
        head = (f"{counts(jobs, 'strong', 'career')} worth applying to, "
                f"{counts(jobs, 'review', 'career')} manual review")
        if bridge_n:
            head += f", {bridge_n} bridge"
    lines = [f"{head} ({len(jobs)} new postings scored)", ""]

    show_track_headers = len(tracks) > 1
    for track, items_in_track in tracks:
        by_tier = group(items_in_track)
        headings = headings_for(track)
        if show_track_headers:
            lines += ["=" * 62,
                      f"{TRACK_HEADING[track].upper()}  ({len(items_in_track)})",
                      "=" * 62, ""]
        for tier in ORDER:
            items = by_tier.get(tier) or []
            if not items:
                continue
            lines += [headings[tier][0].upper() + f" ({len(items)})", ""]
            for j in items:
                sc = j.get("score")
                head_ = f"[{sc:>3}] " if sc is not None else "[ - ] "
                if tier == "weak":
                    lines.append(f"  {head_}{j.get('title', '')} — "
                                 f"{j.get('reason') or ''}")
                else:
                    lines += [f"  {head_}{j.get('title', '')}",
                              f"        {_meta(j)}",
                              f"        {j.get('reason') or ''}",
                              f"        {j.get('url', '')}", ""]
            lines.append("")
    return "\n".join(lines)


def send(jobs):
    """Send via SMTP. For Gmail use an App Password, not your login password."""
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("DIGEST_TO", user)

    if not (user and password):
        print("  - email: no SMTP credentials, printing instead\n")
        print(build_text(jobs))
        return False

    bridge_strong = counts(jobs, "strong", "bridge")
    bridge_maybe = (counts(jobs, "maybe", "bridge")
                    + counts(jobs, "review", "bridge"))
    if jobs and not any((j.get("track") or "career") == "career" for j in jobs):
        subject = (f"[jobscan] bridge: {bridge_strong} strong, "
                   f"{bridge_maybe} maybe")
    else:
        subject = (f"[jobscan] {counts(jobs, 'strong', 'career')} strong, "
                   f"{counts(jobs, 'review', 'career')} review, "
                   f"{counts(jobs, 'maybe', 'career')} maybe")
        if bridge_strong or bridge_maybe:
            subject += f" · bridge {bridge_strong}/{bridge_strong + bridge_maybe}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(build_text(jobs))
    msg.add_alternative(build_html(jobs), subtype="html")

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, password)
        s.send_message(msg)
    print(f"  - email: sent to {to_addr}")
    return True
