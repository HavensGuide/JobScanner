# Flagging what you have already looked at

Five files. Carries bookmarks, dedupe, re-review, jev-decides and the
career_level band, so this is the one to install if you are behind.

## Postings

A past tailor run has three outcomes, and only one of them was visible.
"Drafted" showed a pill; analysed-and-stopped and analysed-and-rejected
looked identical to a posting nobody had touched. Each now carries its
own:

    Drafted        has a letter or a resume
    Reviewed       analysed, stopped before drafting
    Not worth it   analysed as not worth applying to

Under the filter bar there is now an "Already looked at" row with three
checkboxes: Hide drafted, Hide reviewed, Hide not worth it. Each shows its
count, so hiding is never the same as forgetting.

This needed a change in how rows are built. Tailor state used to be
attached only to the page slice, after pagination, which is cheaper but
makes it impossible to filter on. It is a dictionary lookup against an
index already built once per request, so it now runs for every row.

## Drafts

The same three checkboxes, counted the same way, above the list.

Re-review is on the Analysis tab: a model picker and "Run it again". It
runs the same review as the postings page, writes the analysis back, and
drafts nothing.

That needed a link that did not exist. A draft folder is named after the
company and role, never the uid, so the drafts page had no way to act on
the posting behind it. `stem_to_uid()` rebuilds the same slug from the
jobs table to recover it, which works for analyses written long before
anything thought to record a uid. An analysis with no matching posting
says so plainly and hides the control rather than offering a button that
cannot work.

Each analysis also now says who decided it: "Decided by jev, locally" or
"by a model you asked", with the time.

## Tested

Seeded a reviewed posting and a rejected one. Counts came back
{review: 1, skipped: 1, drafted: 0}; hiding either dropped the list from 3
to 2, hiding both to 1. In a browser both pills render, every checkbox
filters, the drafts list filters and renumbers, the model picker fills,
and the re-review control appears only on drafts whose posting still
exists. No page errors.

One thing the drafts list now does that it did not: if the filters hide
whatever was selected, the selection moves to the first visible row
instead of leaving the detail pane describing something no longer listed.
