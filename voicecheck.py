#!/usr/bin/env python3
"""
voicecheck.py - hold a draft to the voice rules mechanically.

    python voicecheck.py out/2026-10-19/norc__data_scientist_i__letter.md
    python voicecheck.py --all out/2026-10-19
    python voicecheck.py --quiet draft.md      # exit 1 if anything is wrong

Why this exists. resume.yaml carries 34 voice rules, about 7,400 characters,
and they arrive in the last third of a 23,000-character prompt. Nearly all of
them are prohibitions, which is the hardest kind of instruction to follow:
the model has to produce the sentence and then not produce it. Naming a
banned phrase can even prime it.

Most of those rules, though, are not matters of judgment. "No em dashes" is
a character search. "Never say I am excited to apply" is a string. A regex
enforces those perfectly, for free, every time, which a model will not.

So the division of labour: the prompt still carries every rule, because the
model should aim at the right target; this catches the mechanical ones it
misses, and names the rule and the offending text so the fix is obvious.

What it cannot check is the part that actually matters: whether the letter
says something true and specific, and whether the claims trace to the fact
base. Read the letter.
"""

import argparse
import re
import sys
from pathlib import Path

# (id, severity, pattern, what to do instead). ERROR is a rule with no
# exceptions; WARN needs a human to decide.
RULES = [
    ("em-dash", "ERROR", r"—",
     "em dash: use a comma, colon, semicolon or a separate sentence"),
    ("nb-hyphen", "ERROR", r"‑",
     "Unicode non-breaking hyphen: use an ordinary hyphen"),
    ("excited", "ERROR",
     r"\b(?:i am|i'm)\s+(?:very\s+|really\s+|especially\s+)?"
     r"(?:excited|thrilled|delighted|eager|passionate)\b",
     "filler enthusiasm: open with a concrete fit, method or piece of work"),
    ("excited-noun", "ERROR",
     r"\b(?:excited|thrilled|delighted)\s+(?:to apply|about the opportunity|"
     r"by the opportunity)\b",
     "filler enthusiasm: say what you did instead"),
    ("contrastive", "WARN",
     r"\b(?:rather than|instead of|not (?:just|merely|simply) \w+ but)\b",
     "manufactured X-not-Y contrast: state the action directly"),
    # "X are not incidental to that problem, they are the problem" is the
    # same tic in a form the pattern above misses: the negation and the
    # correction are separate clauses.
    # "act on them, not just evaluate them" and "are prerequisites, not
    # refinements" are both rule 10, and both slipped past the patterns
    # above: the first has no "but", the second no repeated verb.
    ("contrastive-notjust", "WARN",
     r",\s*not\s+(?:just|merely|simply|only)\s+\w+",
     "X-not-Y: drop the 'not just ...' tail and state the thing"),
    ("contrastive-noun", "WARN",
     r"\b(?:is|are|was|were)\s+\w+(?:\s+\w+)?,\s*not\s+\w+(?:\s+\w+)?\s*[.;]",
     "X-not-Y: say what it is without naming what it is not"),
    ("contrastive-clause", "WARN",
     r"\b(?:is|are|was|were)\s+not\s+[^.;]{3,60}?[,;]\s*"
     r"(?:it|they|that|this|these|those)\s+(?:is|are|was|were)\b",
     "X-not-Y across two clauses: say what it is and stop"),
    # Rule 7 now states the positive target (stop at the outcome) and no
    # longer lists these, so the list has to live here or it is lost.
    ("grader-commentary", "WARN",
     r"\b(?:to ensure (?:rigou?r|accuracy|correctness|quality)|"
     r"to demonstrate (?:rigou?r|discipline|care|competence)|"
     r"so (?:the|that the) (?:result|output|finding|number)s? (?:was|were|is|"
     r"are) (?:trustworthy|reliable|defensible|sound)|"
     r"which (?:ensured|guaranteed) (?:rigou?r|accuracy|correctness))\b",
     "commentary for a grader: stop at what came out of it"),
    ("same-discipline", "ERROR",
     r"\b(?:the same (?:discipline|rigou?r|care|habit|instinct|question)|"
     r"that same (?:discipline|rigou?r|care)|"
     r"(?:depends|rests|turns) on the same)\b",
     "'the same discipline' tic: name the specific thing, or cut the bridge"),
    ("writes-at-posting", "ERROR",
     r"\b(?:as the (?:role|posting|job|description) (?:describes|states|"
     r"notes|says|mentions|outlines)|the role (?:involves|calls for|"
     r"requires) [^.]{0,60}(?:which|and) i\b|your (?:posting|listing|"
     r"job description) (?:describes|mentions|states))\b",
     "writing at the posting: say the thing, do not narrate the ad"),
    ("stock-transition", "ERROR",
     r"(?:that discipline carries|maps? (?:directly )?onto|the bridge from|"
     r"what i bring|the clearest starting point|that experience taught me|"
     r"this aligns directly|the core of what i have done|"
     r"i was drawn to|resonates with me|speaks to me)",
     "stock transition: use a plain one, or none"),
    ("flattery", "WARN",
     r"\b(?:your (?:impressive|outstanding|renowned|world[- ]class|leading|"
     r"prestigious|pioneering)|(?:impressive|outstanding|renowned|"
     r"world[- ]class|prestigious|pioneering) (?:work|research|mission|team|"
     r"reputation))\b",
     "flattery about the employer"),
    ("production-grade", "WARN", r"\bproduction[- ]grade\b",
     "only if the facts establish deployment, monitoring or ownership"),
    ("adjectives", "WARN",
     r"\b(?:rigorous|robust|sophisticated|cutting[- ]edge|state[- ]of[- ]the[- ]art|"
     r"deeply|highly motivated|uniquely positioned)\b",
     "manufactured sophistication: cut it unless it carries information"),
    ("category-label", "WARN",
     r"\b(?:longitudinal|cross[- ]sectional|stakeholder engagement|"
     r"research synthesis|methodological rigou?r)\b",
     "category label: prefer the human-scale description"),
    ("self-deprecation", "ERROR",
     r"\b(?:while i (?:lack|do not have|don't have)|although i (?:lack|am not)|"
     r"i may not (?:have|be)|i am not overstating|i will not represent)\b",
     "do not argue yourself out of the job"),
    ("hedge", "WARN",
     r"\b(?:i believe (?:that )?(?:my|i)|i feel (?:that )?(?:my|i)|"
     r"i am confident (?:that )?(?:my|i))\b",
     "hedge: state the fact"),
]

# Rule 15: an opening, at most three body paragraphs and preferably
# two, then a close, at roughly 300-400 words. The band is wider than
# the rule so the checker nags only on a real overrun.
WORD_MIN, WORD_MAX = 260, 430
# An opening, at most three body paragraphs, and a close. The rule
# prefers two body paragraphs, so four total is the happy case and
# five is the ceiling.
PARA_MIN, PARA_MAX = 3, 5

# Distinctive enough that using one twice reads as a tic.
DISTINCTIVE = [
    "walk-forward", "point-in-time", "changepoint", "fixed effects",
    "deduplicated", "reproducible", "calibrated", "inferential",
    "post-hoc", "effect size", "disclosure limitation", "state-space",
]


def body(text):
    """
    The letter without its furniture.

    The date, Re: line, salutation and sign-off are not prose anyone chose,
    and counting them distorts both the word count and the paragraph count.
    """
    lines, out, started = (text or "").splitlines(), [], False
    for ln in lines:
        s = ln.strip()
        low = s.lower()
        if not started:
            if low.startswith(("dear ", "to the ", "to whom")):
                started = True
            continue
        if low.startswith(("sincerely", "regards", "best regards",
                           "respectfully", "yours ")):
            break
        out.append(ln)
    return "\n".join(out).strip() if started else (text or "").strip()


def is_letter(text):
    """A letter has a salutation. A resume does not."""
    for ln in (text or "").splitlines():
        if ln.strip().lower().startswith(("dear ", "to the ", "to whom")):
            return True
    return False


def check(text, path="", kind=None):
    """
    [(severity, rule, detail, excerpt)] for one draft.

    `kind` is "letter" or "resume"; it is detected from the salutation when
    not given. It matters because the word count and paragraph count are
    letter rules. Run on a resume they fire every time - a resume is not
    350 words of prose in four paragraphs - and a checker that always
    complains is one you stop reading.
    """
    found, b = [], body(text)
    low = b.lower()
    kind = kind or ("letter" if is_letter(text) else "resume")

    for rid, sev, pat, fix in RULES:
        for m in re.finditer(pat, low, re.I):
            found.append((sev, rid, fix, _excerpt(b, m.start(), m.end())))

    paras = [p for p in re.split(r"\n\s*\n", b) if p.strip()]
    if kind == "letter":
        words = len(re.findall(r"\b[\w'-]+\b", b))
        if words and not (WORD_MIN <= words <= WORD_MAX):
            found.append(("WARN", "length",
                          f"{words} words; the rule is roughly 350-450", ""))
        if paras and not (PARA_MIN <= len(paras) <= PARA_MAX):
            found.append(("WARN", "paragraphs",
                          f"{len(paras)} paragraphs; the rule is an opening, "
                          f"at most three body paragraphs and a close", ""))

    # Hyphens and spaces are the same word here. "point in time" in one
    # paragraph and "point-in-time" in the next is the repetition the rule
    # is about, and matching the literal string missed it.
    flat = re.sub(r"[-‐‑\s]+", " ", low)
    for term in DISTINCTIVE:
        t = re.sub(r"[-\s]+", " ", term)
        n = len(re.findall(re.escape(t), flat))
        if n > 1:
            found.append(("WARN", "repetition",
                          f"{term!r} appears {n} times; once is the rule", ""))

    # Several bullets in a row is a resume habit, not a letter.
    if len([p for p in paras if re.match(r"^\s*[-*•]", p)]) >= 3:
        found.append(("WARN", "bullets",
                      "the letter is mostly bullets; it should be prose", ""))
    return found


def check_all(text, jev_cfg=None, profile=None, use_jev=True, kind=None):
    """
    Regex findings plus jev's, as one list.

    The split is by what each is good at. A regex settles a rule with no
    judgment in it, perfectly and for nothing. jev handles the rest: is
    this paragraph the resume in prose, does it argue the candidate out of
    the job. It runs locally, emits no tokens, and reports per paragraph,
    so a breach comes with a place and not just a verdict.

    jev findings are WARN, never ERROR. A regex match is a fact; a
    probability above a threshold is an opinion, and an opinion should not
    silently trigger a rewrite.

    Returns (findings, jev_ran). jev_ran is False when no local model
    answered, so a caller can say "not checked" rather than "clean".
    """
    found = check(text, kind=kind)
    if not use_jev:
        return found, False
    try:
        import jev
        hits = jev.voice_scan(text, jev_cfg, profile)
    except ImportError:
        return found, False
    except AttributeError:
        # An older jev.py with no voice_scan. Report the mechanical rules
        # rather than crashing, and let the caller say jev did not run.
        return found, False
    if hits is None:
        return found, False
    for rid, idx, p, excerpt in hits:
        fix = jev.VOICE_FIX.get(rid, "breaks a voice rule")
        found.append(("WARN", f"{rid} (p{idx + 1}, {p:.2f})", fix, excerpt))
    return found, True


def _excerpt(text, start, end, pad=34):
    a, b = max(0, start - pad), min(len(text), end + pad)
    return ("..." if a else "") + " ".join(text[a:b].split()) + \
           ("..." if b < len(text) else "")


def report(path, found, quiet=False):
    errs = [f for f in found if f[0] == "ERROR"]
    warns = [f for f in found if f[0] == "WARN"]
    if quiet and not found:
        return 0
    name = Path(path).name if path else "draft"
    if not found:
        print(f"  {name}: clean")
        return 0
    print(f"\n{name}: {len(errs)} error(s), {len(warns)} warning(s)")
    for sev, rid, fix, ex in errs + warns:
        print(f"  [{sev:<5}] {rid}: {fix}")
        if ex:
            print(f"           {ex}")
    return 1 if errs else 0


def main():
    ap = argparse.ArgumentParser(
        description="Check a draft against the voice rules")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all", metavar="DIR", help="check every draft in a folder")
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing when a draft is clean")
    ap.add_argument("--no-jev", action="store_true",
                    help="mechanical rules only; skip the local model")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    if not args.path and not args.all:
        ap.error("give a draft, or --all DIR")

    # This used to call check() and therefore never ran jev, so the command
    # line and the UI button both reported "clean" on letters that broke
    # exactly the rules jev exists to catch.
    jev_cfg = None
    if not args.no_jev:
        try:
            import yaml
            jev_cfg = (yaml.safe_load(open(args.config, encoding="utf-8"))
                       or {}).get("jev")
        except Exception:
            jev_cfg = None

    targets = (sorted(Path(args.all).glob("*.md")) if args.all
               else [Path(args.path)])
    targets = [t for t in targets if not t.stem.endswith("__advice")]
    if not targets:
        sys.exit("no drafts found")

    bad, jev_ran = 0, False
    for t in targets:
        try:
            found, ran = check_all(t.read_text(encoding="utf-8"), jev_cfg,
                                   use_jev=not args.no_jev)
            jev_ran = jev_ran or ran
            bad |= report(t, found, args.quiet)
        except OSError as e:
            print(f"  {t}: {e}")
    if not jev_ran and not args.no_jev:
        print("\nNote: jev did not answer, so only the mechanical rules ran. "
              "The judgment\nrules (opens with filler, resume in prose, "
              "praises the employer, vague claim)\nwere NOT checked. Start "
              "the local model and run this again.")
    if bad:
        print("\nErrors are rules with no exceptions. Warnings need your "
              "judgment.")
        print("What this cannot check: whether the letter says something "
              "true and specific.")
    sys.exit(bad)


if __name__ == "__main__":
    main()
