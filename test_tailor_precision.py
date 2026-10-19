#!/usr/bin/env python3
"""Offline regression tests for the precision-oriented tailor gate."""
import json
import tempfile
from pathlib import Path

import tailor

FACTS = """CANDIDATE STATUS AS OF August 21, 2026:\n  GRADUATE. Degree conferred May 2026.\neducation:\n  degree: B.A. in Economics\n"""


def expect_gate(text, expected, quality="full"):
    got, reason = tailor.deterministic_gate(text, FACTS, quality)
    assert got == expected, (text, expected, got, reason)


def main():
    expect_gate("Internship. Must be currently enrolled in a university.", "do_not_apply")
    got, _ = tailor.deterministic_gate("Policy Intern Summer 2025. June through August 2025.", FACTS, "full")
    assert got == "do_not_apply"
    expect_gate("Qualifications: Enrolled in a top-tier university with quantitative background.", "do_not_apply")
    expect_gate("Desired Qualifications: Bachelor's candidates graduating December 2026/Summer 2027.", "do_not_apply")
    expect_gate("Policy Counsel. JD required. Research and writing.", "do_not_apply")
    expect_gate("Attorney role. Must be admitted to the New York bar.", "do_not_apply")
    expect_gate("Analyst. SQL required. Build dashboards.", None)  # viability model decides
    expect_gate("Analyst. SQL preferred. Python and Excel.", None)
    expect_gate("Data analyst", "manual_review", quality="excerpt")
    assert tailor.effective_desc_quality("x" * 2000, "full") == "excerpt"
    assert tailor.effective_desc_quality("x" * 1999, "full") == "full"
    assert tailor.effective_desc_quality("x" * 40000, "full") == "excerpt"

    # The gate must split the title off the RAW posting. Flattening newlines
    # first made splitlines()[0] the whole document, and _temporal_scan skips
    # its season-context guard for titles — so every ordinary mention of a
    # season became an expired internship.
    multiline = (
        "Community Response Coordinator\n"
        "City of Newark\n"
        "Newark, NJ\n\n"
        "Lead outreach for public events such as DOT Summer Streets and "
        "community board meetings. Responsibilities include stakeholder "
        "coordination, drafting materials, and tracking metrics.\n"
        "Qualifications: Bachelor's degree, strong writing.\n")
    assert tailor.deterministic_gate(multiline, FACTS, "full",
                                     posted="2026-08-01")[0] is None
    founded = (
        "Research Analyst\nUrban Policy Group\nNew York, NY\n\n"
        "Founded in 2011, our organization runs a year-round research "
        "program. Responsibilities: data cleaning in Python, literature "
        "review, drafting briefs. Our summer speaker series brings in "
        "outside experts.\nQualifications: Bachelor's in economics.\n")
    assert tailor.deterministic_gate(founded, FACTS, "full",
                                     posted="2026-08-01")[0] is None

    # A genuinely expired posting must still stop before Sonnet.
    expired = ("Policy Fellowship\nSome Institute\nNew York, NY\n\n"
               "Application deadline: March 1, 2026. Program begins April 2026.\n")
    assert tailor.deterministic_gate(expired, FACTS, "full",
                                     posted="2026-01-05")[0] == "do_not_apply"

    assert tailor.normalize_recommendation("stretch") == "manual_review"
    assert tailor.normalize_recommendation("skip") == "do_not_apply"
    assert not tailor.gate_reasons({"recommendation": "apply", "gaps": []}, "both")
    assert tailor.gate_reasons({"recommendation": "manual_review", "gaps": []}, "both")
    assert tailor.gate_reasons({"recommendation": "do_not_apply", "gaps": []}, "both")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "config.yaml"
        p.write_text("scoring:\n  strong_at: 73\n", encoding="utf-8")
        assert tailor._default_batch_min_score(str(p)) == 73
        p.write_text("scoring:\n  strong_at: 73\ntailoring:\n  batch_min_score: 76\n", encoding="utf-8")
        assert tailor._default_batch_min_score(str(p)) == 76

    print("tailor precision tests passed")


if __name__ == "__main__":
    main()
