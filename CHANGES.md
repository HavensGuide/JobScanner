# Merge: forward-port onto the llm.py / UI branch

The `llm.py` + `ui.py` refactor from the other chat branched before the last
two rounds of work here, so installing it silently reverted three features.
This package puts them back on top of that branch and fixes two bugs that
shipped with it.

Nothing from `llm.py` or the UI was undone. Every AI call still goes through
`llm.py` profiles.

## Restored

**1. `enrich.py` (new file) — fetch descriptions that the feed left out.**
A posting that arrives as a bare URL with no body can't be scored; the old
behavior was to drop it. `enrich.py` fetches it two ways: the platform's own
single-posting API first (greenhouse, lever, workday), then a plain GET with
`<script>`/`<style>` stripped and `<main>`/`<article>` preferred. What comes
back is run through a junk regex and then a JevK5 screen, so a sign-in wall or
an expired-posting notice is rejected instead of being stored as a description.

Routed through `llm.py`: `--model` resolves a profile name via `llm._lookup`.

**2. Local pre-scan in `tailor.py`.**
Sits between the deterministic gate and the paid `ANALYZE` call. A posting
that scores under the floor is marked `do_not_apply` locally and never reaches
the paid model. New flags: `--no-prescan`, `--prescan-min` (default 45).

**3. `--no-email` / `--no-enrich` in `main.py`.**
`--no-email` records postings and skips the digest, so the scheduled cloud run
can discover without mailing. `main.py` also enriches gap postings after
`filter_new`, unless `--no-score` (the cloud run has no model to screen with,
so it prints a note telling you to run `enrich.py` locally instead).

Supporting changes in `jev.py`: a new `posting_text` question, added to
`validate()`, plus `screen_texts()` and `prescan()` — both routed through
`llm.py`.

**4. A "Fetch missing descriptions" control on the Scoring page.**
`ui.py` already accepted an `enrich` action but nothing in `ui/index.html`
could reach it. Added under Backfill, with limit, model, dry run, and skip
the screen. Verified end to end: the button posts, the run starts, and the
label comes back correct.

## Fixed

**`config.yaml` — truncated rubrics.** Both prose rubrics had lost their
leading character and began `"ou are triaging job postings..."`. The model was
being handed a malformed first sentence on every single call.

**`store.py` — Sheets column mismatch.** The mirror wrote 14 headers and 13
values: a `Scorer` header with nothing under it, which shifted every column
after it by one. Now 14/14.

**`jobscan.yml` — discovery only.** Runs `main.py --no-score --no-email`. The
env block carries only the feed keys and the Sheets credentials. No model keys
and no SMTP credentials, deliberately: if something in the cloud run tries to
score or mail, it fails loudly instead of quietly spending money or sending a
half-scored digest.

## Known gaps

- `store.py` `FIELDS` does not include `scorer_model`. Left alone in case
  that's intentional in the UI branch.
- `enrich.py` has never run against a live job URL. The sandbox can't reach
  those hosts, so it was tested against a stub. Run `python enrich.py --limit 5
  --dry-run` first.
- `jev.prescan` has never run against the real model, only the stub.

## Install

Copy these over your current tree:

    jev.py  enrich.py  tailor.py  main.py  config.yaml  store.py  jobscan.yml  ui.py

Then:

    python enrich.py --limit 5 --dry-run     # check the fetcher on real URLs
    python tailor.py <id>                    # watch for the "pre-scan (local, free)" line


---

# Round 2: documents, editing, and an assistant

## New: `docs.py`

Renders a draft into a document you can send. Two layouts, chosen from the
filename: `__letter.md` becomes a block letter, anything else becomes the
one-page technical resume. Both are modelled on the documents you already
send (the OpenAI-format resume and the Technical Documentation Writer
letter), so the output should look familiar rather than like a template.

Contact details come from `resume.yaml`, never from the prompt, so a phone
number cannot drift between documents or be invented by a model.

    python docs.py out/2026-10-18/acme__data-analyst__letter.md
    python docs.py --all out/2026-10-18 --pdf

**The .md stays the source.** The .docx is a build artifact and is
overwritten on every render. Edit the Markdown, re-render, and the document
follows. Editing the .docx directly works until the next render, then the
change is gone. The UI says this next to the editor.

PDF is optional and degrades rather than failing. LibreOffice is tried first
because it renders the .docx itself and therefore agrees with it exactly;
reportlab is the fallback and re-lays the text, which is close but not
identical. With neither installed the .docx is still written and the PDF is
reported as skipped.

## Resume advice became resume tailoring

`--resume` used to return notes about what you should change. It now writes
the finished tailored resume: roles selected and ordered for the posting,
bullets cut to the ones that speak to its duties, tagline and summary
rewritten for the role, skills grouped so the asked-for ones come first.

The constraint that makes this safe is that it only ever reorders, selects
and rephrases what is already in `resume.yaml`. The prompt forbids adding a
tool the facts do not contain however central the posting makes it, forbids
upgrading what a bullet describes, and requires numbers be copied exactly.

The old advice output is still there under `--advice`, as
`__advice.md`. Advice is notes to yourself, so it is never rendered to a
document.

    python tailor.py 787428 --resume          # tailored resume
    python tailor.py 787428 --both --pdf      # letter + resume, with PDFs
    python tailor.py 787428 --advice          # the old notes
    python tailor.py 787428 --no-docx         # Markdown only

## UI

**Tailor my resume** sits beside Draft a letter on every posting, with a
"PDF too" checkbox.

**Both documents are editable.** The Letter and Resume tabs open the
Markdown in an editor with Save, Rebuild .docx, and download buttons for the
.docx and the PDF. Rebuild saves first, so what is on screen is what gets
rendered rather than whatever was last written to disk.

**Downloads** are served with the right MIME type and a
`Content-Disposition` filename, through the same `out/`-only path guard as
the editor, so a traversal attempt is refused.

**An assistant**, from "Ask for help" in the sidebar or "Ask for a review"
on any draft. Opened from a draft it gets the draft, your fact base and your
voice rules, and is told to quote the line it means and give the
replacement. It flags any sentence claiming experience the facts do not
support, which is the error that gets an application binned. Opened from the
sidebar it answers questions about the tool from a written summary of how it
works, and says so when the answer is not in that summary. It runs on
whichever profile `ai:` assigns to `tailor_letter`, so it costs what that
model costs.

## Dependency

`python-docx` is now required and is in requirements.txt. `reportlab` is
listed but commented out; install it only if you want PDFs without
LibreOffice.

## Tested

The renderer was run on both layouts and the PDFs were rasterised and
checked by eye against your originals. The UI was driven in a real browser:
no page errors, tabs render, downloads serve a valid `Microsoft Word 2007+`
file, Rebuild produces both .docx and .pdf, and the assistant opens with the
draft in context. Path traversal and rendering a non-.md are both refused.
A full `process_job` run with a stubbed model produced letter, resume and
advice with the documents built and advice correctly left as notes.

Not tested: a real model writing a tailored resume, since that needs your
API key or JevK5. Check the first one closely against resume.yaml before
you send it.
