#!/usr/bin/env python3
"""
docs.py - turn a tailored draft into a document you can actually send.

    python docs.py out/2026-10-18/acme__data-analyst__letter.md
    python docs.py out/2026-10-18/acme__data-analyst__resume.md --pdf
    python docs.py --all out/2026-10-18 --pdf

tailor.py writes Markdown because Markdown is easy to diff, edit and re-read.
Nobody accepts a .md as a job application. This renders the same text into a
.docx laid out like the documents Kai already sends, and optionally a PDF.

WHY MARKDOWN STAYS THE SOURCE. The .md is the editable copy: change it in the
UI or any editor and re-render. The .docx is a build artifact and is
overwritten on every render, so never hand-edit the .docx and expect it to
survive. That one rule is what keeps the UI's editor and this renderer from
fighting over the same file.

TWO LAYOUTS, picked from the filename:

  __letter.md   block letter. Name and contact header, date, Re: line,
                salutation, body paragraphs, sign-off.
  __resume.md   one-page technical resume. Name, tagline, contact rule,
                then SUMMARY / CORE SKILLS / EXPERIENCE / PROJECTS /
                EDUCATION with tight spacing and hanging bullets.

Both read contact details from resume.yaml, so a phone number is never
duplicated into a prompt and can never drift between documents.

PDF. Optional, because it needs something that can lay out a .docx.
LibreOffice is tried first since it renders the .docx we just built and the
two therefore agree. reportlab is the fallback and re-renders from the
Markdown, which is close but not identical. If neither is present the .docx
is still written and the PDF is reported as skipped, never as an error.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# Matches the existing documents: a dark near-black for text, a rule under
# the contact line, and a muted grey for the tagline.
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x55, 0x55)
RULE = "999999"
BODY_FONT = "Calibri"
HEAD_FONT = "Calibri"


# ==========================================================================
# Small docx helpers
# ==========================================================================
def _style(doc):
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(10)
    st.font.color.rgb = INK
    pf = st.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    return doc


def _margins(doc, top=0.5, bottom=0.5, left=0.7, right=0.7):
    for s in doc.sections:
        s.top_margin = Inches(top)
        s.bottom_margin = Inches(bottom)
        s.left_margin = Inches(left)
        s.right_margin = Inches(right)


def _p(doc, text="", size=10, bold=False, italic=False, color=None,
       before=0, after=0, align=None, caps=False, spacing=None):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if align is not None:
        par.alignment = align
    if text:
        _runs(par, text, size, bold, italic, color, caps, spacing)
    return par


def _runs(par, text, size=10, bold=False, italic=False, color=None,
          caps=False, spacing=None):
    """Add text to a paragraph, honouring **bold** and *italic* inline."""
    for chunk, b, i in _inline(text):
        r = par.add_run(chunk.upper() if caps else chunk)
        r.font.name = BODY_FONT
        r.font.size = Pt(size)
        r.bold = bold or b
        r.italic = italic or i
        r.font.color.rgb = color or INK
        if spacing:
            _char_spacing(r, spacing)
    return par


_INLINE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*(?!\*).+?(?<!\*)\*(?!\*))")


def _inline(text):
    """[(text, bold, italic)] from a string with ** and * markers."""
    out = []
    for part in _INLINE.split(text or ""):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((part[2:-2], True, False))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append((part[1:-1], False, True))
        else:
            out.append((part, False, False))
    return out or [(text or "", False, False)]


def _char_spacing(run, twips):
    """Letter-spacing, used for the federal-style section headings."""
    rPr = run._element.get_or_add_rPr()
    el = OxmlElement("w:spacing")
    el.set(qn("w:val"), str(int(twips)))
    rPr.append(el)


def _rule(par, color=RULE, size=6):
    """A horizontal rule under a paragraph."""
    pPr = par._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "2")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)
    pPr.append(borders)


def _bullet(doc, text, size=10, after=2):
    """
    A hanging-indent bullet.

    Word's built-in List Bullet style drags in list numbering that behaves
    differently depending on the template, and a resume only needs one level.
    A literal bullet with a hanging indent is predictable everywhere.
    """
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.left_indent = Inches(0.16)
    pf.first_line_indent = Inches(-0.16)
    pf.space_before = Pt(0)
    pf.space_after = Pt(after)
    r = par.add_run("\u2022  ")
    r.font.name = BODY_FONT
    r.font.size = Pt(size)
    r.font.color.rgb = INK
    _runs(par, text, size)
    return par


def _section(doc, title, federal=False):
    """A section heading. Federal style letter-spaces it, as in the original."""
    par = _p(doc, before=9, after=3)
    _runs(par, title, size=9.5, bold=True, caps=True,
          spacing=60 if federal else 20)
    _rule(par)
    return par


# ==========================================================================
# Contact details
# ==========================================================================
def contact(facts_path="resume.yaml"):
    """Name, location, phone, email, linkedin from resume.yaml."""
    d = {}
    if yaml:
        try:
            d = yaml.safe_load(open(facts_path, encoding="utf-8")) or {}
        except Exception:
            d = {}
    li = (d.get("linkedin") or "").replace("https://", "").replace("www.", "")
    return {
        "name": d.get("name") or "",
        "location": d.get("location") or "",
        "phone": d.get("phone") or "",
        "email": d.get("email") or "",
        "linkedin": li,
    }


def _contact_line(c, parts=("location", "phone", "email", "linkedin")):
    return "  |  ".join(c[k] for k in parts if c.get(k))


# ==========================================================================
# Markdown parsing
# ==========================================================================
def parse(md):
    """
    Markdown -> [(kind, text)] where kind is h1/h2/h3/bullet/text/blank.

    Deliberately small. The drafts are headings, bullets and paragraphs; a
    full Markdown parser would add a dependency to gain nothing.
    """
    out = []
    for raw in (md or "").splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s:
            out.append(("blank", ""))
        elif s.startswith("### "):
            out.append(("h3", s[4:].strip()))
        elif s.startswith("## "):
            out.append(("h2", s[3:].strip()))
        elif s.startswith("# "):
            out.append(("h1", s[2:].strip()))
        elif re.match(r"^[-*\u2022]\s+", s):
            out.append(("bullet", re.sub(r"^[-*\u2022]\s+", "", s)))
        elif re.match(r"^\d+[.)]\s+", s):
            out.append(("bullet", re.sub(r"^\d+[.)]\s+", "", s)))
        elif set(s) <= {"-", "=", "_"} and len(s) >= 3:
            out.append(("blank", ""))
        else:
            out.append(("text", s))
    return out


# ==========================================================================
# Cover letter
# ==========================================================================
def build_letter(md, out_path, c=None, re_line=None):
    """
    Block letter matching the existing one: header, date, Re:, salutation,
    body, sign-off.

    The draft usually carries its own date, Re: line and "Dear ..." because
    the model was asked for a complete letter. Those are detected and
    rendered as letter furniture instead of as body text, so a draft that
    includes them and one that does not both come out identical.
    """
    c = c or contact()
    doc = _style(Document())
    _margins(doc, top=0.7, bottom=0.7, left=0.9, right=0.9)

    head = _p(doc, after=1)
    _runs(head, c["name"], size=15, bold=True)
    sub = _p(doc, after=6)
    _runs(sub, _contact_line(c), size=9, color=MUTED)
    _rule(sub)

    blocks = parse(md)
    body, date_txt, subject, salutation, signoff = [], None, None, None, []
    seen_salutation = False

    for kind, text in blocks:
        if kind == "blank":
            if body and body[-1] != ("blank", ""):
                body.append(("blank", ""))
            continue
        low = text.lower()
        if kind in ("h1", "h2", "h3"):
            # A draft that titles itself "Cover Letter" or repeats the name
            # should not print that as a heading in a block letter.
            if "cover letter" in low or text.strip() == c["name"]:
                continue
            if low.startswith("re:") or low.startswith("subject:"):
                subject = text
                continue
            body.append(("text", text))
            continue
        if not seen_salutation:
            if date_txt is None and _is_date(text):
                date_txt = text
                continue
            if low.startswith(("re:", "subject:")):
                subject = text
                continue
            if low.startswith(("dear ", "to the ", "to whom")):
                salutation = text
                seen_salutation = True
                continue
            # Contact lines the model restated above the salutation.
            if _is_contact_echo(text, c):
                continue
        if low.startswith(("sincerely", "regards", "best regards",
                           "respectfully", "yours ")):
            signoff.append(text)
            continue
        if signoff:
            signoff.append(text)
            continue
        body.append((kind, text))

    _p(doc, date_txt or _today_long(), size=10, before=8, after=8)
    if subject or re_line:
        s = subject or f"Re: {re_line}"
        if not s.lower().startswith(("re:", "subject:")):
            s = f"Re: {s}"
        _p(doc, s, size=10, bold=True, after=8)
    _p(doc, salutation or "Dear Hiring Team,", size=10, after=8)

    for kind, text in body:
        if kind == "blank":
            continue
        if kind == "bullet":
            _bullet(doc, text, after=4)
        else:
            _p(doc, text, size=10, after=8, align=WD_ALIGN_PARAGRAPH.LEFT)

    _p(doc, (signoff[0] if signoff else "Sincerely,"), size=10,
       before=4, after=20)
    _p(doc, (signoff[1] if len(signoff) > 1 else c["name"]), size=10)

    out_path = Path(out_path)
    doc.save(str(out_path))
    return out_path


_DATE = re.compile(
    r"^(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}$|^\d{4}-\d{2}-\d{2}$", re.I)


def _is_date(s):
    return bool(_DATE.match((s or "").strip()))


def _is_contact_echo(text, c):
    t = (text or "").lower()
    for k in ("phone", "email", "location"):
        v = (c.get(k) or "").lower()
        if v and v in t:
            return True
    return False


def _today_long():
    from datetime import date
    return date.today().strftime("%B %-d, %Y") if sys.platform != "win32" \
        else date.today().strftime("%B %d, %Y").replace(" 0", " ")


# ==========================================================================
# Resume
# ==========================================================================
def build_resume(md, out_path, c=None, federal=False):
    """
    One-page technical resume, or the longer federal layout.

    Structure comes from the Markdown: `## ` starts a section, `### ` is a
    role or project heading, and a line immediately under a role heading that
    looks like dates is set as the dated sub-line rather than a paragraph.
    """
    c = c or contact()
    doc = _style(Document())
    _margins(doc, top=0.45, bottom=0.45,
             left=0.6 if not federal else 0.7,
             right=0.6 if not federal else 0.7)

    blocks = parse(md)

    # A leading h1 is the name; the line after it, if short and not a
    # section, is the tagline. Both are consumed so they are not repeated.
    idx = 0
    name = c["name"]
    tagline = None
    while idx < len(blocks) and blocks[idx][0] == "blank":
        idx += 1
    if idx < len(blocks) and blocks[idx][0] == "h1":
        name = blocks[idx][1] or name
        idx += 1
        j = idx
        while j < len(blocks) and blocks[j][0] == "blank":
            j += 1
        if (j < len(blocks) and blocks[j][0] == "text"
                and len(blocks[j][1]) < 110
                and not _is_contact_echo(blocks[j][1], c)):
            tagline = blocks[j][1]
            idx = j + 1

    head = _p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    _runs(head, name, size=17, bold=True, spacing=24)
    if tagline:
        t = _p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
        _runs(t, tagline.upper(), size=8.5, color=MUTED, spacing=30)
    cl = _p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    _runs(cl, _contact_line(c, ("location", "phone", "email", "linkedin")),
          size=9, color=MUTED)
    _rule(cl)

    prev = None
    for k in range(idx, len(blocks)):
        kind, text = blocks[k]
        if kind == "blank":
            prev = kind
            continue
        if kind in ("h1", "h2"):
            _section(doc, text, federal=federal)
        elif kind == "h3":
            par = _p(doc, before=5, after=0)
            _runs(par, text, size=10.5, bold=True)
        elif kind == "bullet":
            _bullet(doc, text, after=2)
        else:
            # A dates / meta line directly under a role heading.
            if prev == "h3" and _is_meta(text):
                par = _p(doc, after=2)
                _runs(par, text, size=9, italic=True, color=MUTED)
            else:
                _p(doc, text, size=10, after=3)
        prev = kind

    out_path = Path(out_path)
    doc.save(str(out_path))
    return out_path


_META = re.compile(
    r"(\d{4})|present|current|hours per week|supervisor|\bgpa\b|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.I)


def _is_meta(s):
    s = (s or "").strip()
    return len(s) < 170 and bool(_META.search(s))


# ==========================================================================
# PDF
# ==========================================================================
def to_pdf(docx_path, pdf_path=None, quiet=False):
    """
    .docx -> .pdf. Returns the path, or None if nothing can render one.

    LibreOffice first: it renders the .docx itself, so the PDF and the Word
    file agree. reportlab is the fallback and lays the text out again, which
    is readable but will not match line for line.
    """
    docx_path = Path(docx_path)
    pdf_path = Path(pdf_path) if pdf_path else docx_path.with_suffix(".pdf")
    say = (lambda *a: None) if quiet else print

    soffice = (shutil.which("soffice") or shutil.which("libreoffice")
               or _soffice_windows())
    if soffice:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                subprocess.run(
                    [soffice, "--headless", "--norestore", "--convert-to",
                     "pdf", "--outdir", tmp, str(docx_path)],
                    check=True, timeout=180,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                made = Path(tmp) / (docx_path.stem + ".pdf")
                if made.exists():
                    shutil.move(str(made), str(pdf_path))
                    return pdf_path
            except (subprocess.SubprocessError, OSError) as e:
                say(f"  ! libreoffice could not render a PDF: "
                    f"{type(e).__name__}: {e}")

    made = _pdf_reportlab(docx_path, pdf_path)
    if made:
        return made
    say("  ! no PDF: install LibreOffice, or `pip install reportlab`. "
        "The .docx was still written.")
    return None


def _soffice_windows():
    for p in (r"C:\Program Files\LibreOffice\program\soffice.exe",
              r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"):
        if Path(p).exists():
            return p
    return None


def _pdf_reportlab(docx_path, pdf_path):
    """Re-lay the .docx text with reportlab. Approximate, but readable."""
    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer)
    except ImportError:
        return None
    try:
        src = Document(str(docx_path))
    except Exception:
        return None

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5,
                          leading=12.5, alignment=TA_LEFT, textColor="#1a1a1a")
    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    flow = []
    for par in src.paragraphs:
        text = "".join(r.text for r in par.runs).strip()
        if not text:
            flow.append(Spacer(1, 5))
            continue
        bold = any(r.bold for r in par.runs if r.text.strip())
        size = max((r.font.size.pt for r in par.runs
                    if r.font.size is not None), default=9.5)
        st = ParagraphStyle(
            "s", parent=body, fontSize=size,
            leading=size * 1.32,
            fontName="Helvetica-Bold" if bold else "Helvetica")
        flow.append(Paragraph(_xml_escape(text), st))
    try:
        doc.build(flow)
    except Exception:
        return None
    return pdf_path


def _xml_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ==========================================================================
# Entry point used by tailor.py and ui.py
# ==========================================================================
def render(md_path, pdf=False, facts_path="resume.yaml", quiet=False,
           re_line=None, federal=False):
    """
    One .md -> (.docx, .pdf or None).

    The layout is chosen from the filename suffix that tailor.py writes:
    __letter.md gets the block letter, anything else the resume.
    """
    md_path = Path(md_path)
    md = md_path.read_text(encoding="utf-8")
    c = contact(facts_path)
    docx_path = md_path.with_suffix(".docx")

    if md_path.stem.endswith("__letter") or md_path.stem.endswith("letter"):
        build_letter(md, docx_path, c, re_line=re_line)
    else:
        build_resume(md, docx_path, c, federal=federal)

    made_pdf = to_pdf(docx_path, quiet=quiet) if pdf else None
    return docx_path, made_pdf


def render_all(folder, pdf=False, facts_path="resume.yaml", quiet=False):
    """Render every draft in a folder. Advice files are skipped."""
    out = []
    for f in sorted(Path(folder).glob("*.md")):
        if f.stem.endswith("__advice"):
            continue
        out.append(render(f, pdf=pdf, facts_path=facts_path, quiet=quiet))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Render a tailored draft as .docx, and optionally .pdf")
    ap.add_argument("path", nargs="?", help="a draft .md file")
    ap.add_argument("--all", metavar="DIR",
                    help="render every draft in this folder")
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    ap.add_argument("--facts", default="resume.yaml")
    ap.add_argument("--federal", action="store_true",
                    help="federal resume layout (longer, letter-spaced heads)")
    args = ap.parse_args()

    if not args.path and not args.all:
        ap.error("give a draft .md, or --all DIR")

    targets = []
    if args.all:
        targets = render_all(args.all, pdf=args.pdf, facts_path=args.facts)
    else:
        targets = [render(args.path, pdf=args.pdf, facts_path=args.facts,
                          federal=args.federal)]

    for docx_path, pdf_path in targets:
        print(f"  wrote {docx_path}")
        if pdf_path:
            print(f"  wrote {pdf_path}")


if __name__ == "__main__":
    main()
