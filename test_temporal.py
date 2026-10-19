#!/usr/bin/env python3
"""Small deterministic regression test for stale-job temporal filtering."""
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score

TODAY = date(2026, 8, 21)


def job(title, desc="", posted="2026-01-15"):
    return {"title": title, "description": desc, "posted": posted}


def check(title, expected, desc="", posted="2026-01-15"):
    j = job(title, desc, posted)
    got = score.temporal_exclude(j, TODAY)
    assert got is expected, (title, expected, got, score.temporal_summary(j, TODAY))


# Required examples.
check("Quantitative Researcher Internship, May-August", True)
check("Summer 2026 Internship", True)
check("Internship, June 2026 - August 2026", True)
check("Fall 2026 Fellowship", False)
check("Internship, September-December 2026", False)
check("Spring 2027 Fellowship", False)
check("Summer 2027 Internship", False)
check("Internship, May-August 2027", False)

# Explicit start dates and application deadlines.
check("Research Internship", True,
      "Start Date: May 18, 2026. The internship runs through August.")
check("Research Internship", False,
      "Start Date: September 8, 2026. The internship runs through December.")
check("Policy Fellowship", True,
      "Application deadline: August 15, 2026. Program begins September 2026.")
check("Policy Fellowship", False,
      "Apply by September 15, 2026. Program begins October 2026.")

# The posted date disambiguates yearless recruiting cycles.
check("Summer Internship", False, posted="2026-10-15")  # Summer 2027
check("Summer Internship", True, posted="2026-02-15")   # Summer 2026

# Do not turn arbitrary historical month ranges into employment periods.
check("Data Analyst", False,
      "Sales were measured May-August 2025 and compared with the prior year.")

# Recruiting-cycle years can sit elsewhere in the title.
check("Summer Analyst 2027", False, posted="2026-07-24")
check("2027 Corporate Finance Summer Analyst", False, posted="2026-08-06")
check("Quantitative Developer Intern", True,
      "We are looking for quantitative developer interns for Summer 2025.",
      posted="2026-08-03")

# Multiple/rolling cohorts should not be deleted because one mentioned season
# has already begun.
check("Quant Research Intern", False,
      "This role is open year-round; we welcome summer internships, winter-break internships, or part-time roles.",
      posted="2026-06-15")
check("Associate Value Engineer - Orbit Program", False,
      "Start dates are organized twice a year. You may join a summer start date (June/July) or winter start date (January/February).",
      posted="2026-08-06")
check("Quant Developer / Quant Research Intern - 2026/2027", False,
      "12-week onsite internship — Fall 2026 (September-November) or Summer 2027 (June-August).",
      posted="2026-07-23")

# Ordinary uses of seasonal words are not employment periods.
check("Community Response Coordinator", False,
      "Lead outreach for public events such as DOT Summer Streets and community board meetings.",
      posted="2026-08-01")
check("Lead Data Scientist", False,
      "The successful candidate's starting pay will fall within the range below.",
      posted="2026-08-12")

# A year printed near a season is usually company history, not a cycle year.
# Without a plausibility guard the nearest match dates the period to 2014 and
# a permanent role is deleted as a decade-stale internship.
check("Research Analyst", False,
      "Founded in 2011, our organization runs a year-round research program. "
      "Our summer speaker series brings in outside experts.",
      posted="2026-08-01")
check("Data Analyst", False,
      "Copyright 2015. Reports are produced each fall for the program board.",
      posted="2026-08-01")

# "program", "class", "term", "starts", "begins" are among the most common
# words in any posting. Proximity to one of them is not evidence of a period.
check("Program Associate", False,
      "The program launched in 2018. Summer programming includes youth "
      "workshops that the associate helps coordinate.",
      posted="2026-08-01")
check("Communications Associate", False,
      "The team begins each cycle with a planning retreat. Our summer "
      "newsletter reaches 40,000 subscribers.",
      posted="2026-08-01")

# ...but a season that actually modifies a period noun still counts.
check("Research Intern", True,
      "This is our summer internship. Interns join in June.",
      posted="2026-08-01")
check("Policy Fellow", True,
      "The fellowship runs through summer, starting June.",
      posted="2026-07-01")

print("temporal filter tests passed")
