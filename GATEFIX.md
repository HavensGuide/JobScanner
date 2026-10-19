# Gate fix — two changes, config.yaml and tailor.py

Install on top of round3cd.

## 1. career_level was counting years, not reading the sentence

NORC asked for "4 years of relevant experience including graduate research,
internships, and applied professional experience." The old question asked
whether the role was open to someone with "under two years of full-time
professional experience", so the model saw a 4 and answered no — while the
same sentence said internships and research count, which is exactly the
experience you have.

The question now asks whether the posting's OWN definition of experience
admits research assistantships, internships and fellowships, whatever
number of years it names. Both bands stay single-clause on that one axis;
the lists inside them are alternatives, not conditions to satisfy together,
which is what broke the earlier conjunctive bands.

A genuinely senior posting still fails it: "8 years leading a team" and
"5 years professional experience required" with no mention of internships
both read false.

## 2. A borderline pre-scan was a hard rejection

The two errors are not worth the same. A false positive costs one cheap
viability call. A false negative costs a job you never hear about again,
and it is invisible, because the run stops before the analysis is written.

Only a clear reject is now a reject. Within 15 points of the floor you get
manual review instead, so the posting still reaches you:

    below 30   do_not_apply
    30 to 44   manual_review, "near the line"
    45 and up  proceeds to the paid call

Tunable with --prescan-margin (default 15). The NORC score of 30 now lands
in manual review rather than do_not_apply.

## Check it before trusting it

Changing a band changes every career score, so verify rather than assume:

    python score.py --compare --left jev --right jev --track career --limit 60

Watch for postings that move for the right reason. If a senior role starts
clearing the gate, the band went too far and I would rather tighten it than
leave it loose.

Then re-run the one that started this:

    python tailor.py adzuna:5932893477 --both
