# Re-review a posting with a model you pick

Four files: tailor.py, ui.py, ui/index.html, config.yaml. Includes the
jev-decides change and the career_level band fix, so install this instead
of those if you have not applied them yet.

## What it adds

On any posting, under the drafting buttons:

    Second opinion from [ model ]   [ Re-review this posting ]

jev still makes the routine decision, free, on every run. This is for when
you disagree with it: a posting it rejected that you think it misread, or
one it passed that looks too good. The chosen model reads the posting and
your fact base and argues the case.

It writes an opinion and drafts nothing, so it costs exactly one call. That
separation is deliberate: asking for an opinion must never quietly turn
into paying for a letter.

The dropdown lists your generative profiles only. jevk5 is excluded because
it returns calibrated probabilities, not an argument, and it has already
given its verdict by the time you are reading the posting. Adding an `opus`
profile under `models:` in config.yaml puts it in this dropdown like any
other.

## On the command line

    python tailor.py adzuna:5932893477 --review
    python tailor.py adzuna:5932893477 --review --model sonnet

Gates are skipped: you have already decided you want this one looked at.
The result is written to the usual analysis file, marked
`"decision_source": "model_review"` with a `reviewed_at` timestamp, so a
model's opinion is never mistaken later for jev's decision.

## Tested

One paid call, nothing drafted, state and source recorded correctly. The
command builder rejects a missing uid and refuses a model name that is not
a defined profile, so nothing from the browser reaches the command line
unchecked. Driven in a browser: the button and dropdown render, the
dropdown lists haiku and sonnet plus the configured default, no page
errors.
