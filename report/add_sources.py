"""Add a source attribution line beneath every figure caption.

Supervisor's note: "Insert the sources of your diagrams and drawings."

Every figure in this report is the author's own: the conceptual diagrams in
Chapters 2 and 3 were drawn by the author, and the three figures in Chapter 4
are output produced by the author's own pipeline from the experimental runs.
The Chapter 4 wording says so explicitly, since "own work" for a generated
plot means something slightly different from "own work" for a drawing.

Writes back to the same file. If Word has it open the file is locked, so a
timestamped copy is written alongside instead of failing.
"""
import copy
import os
import re
import time

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "DE_DNN_IDS.docx")

FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)

CONCEPTUAL = "Source: Author's own work"
GENERATED = ("Source: Author's own work, generated from the experimental "
             "results of this study")

doc = Document(SRC)
body = doc.element.body
cap_re = re.compile(r"^Figure\s+\d+\.\s*\d*\s*\d*\s*:", re.I)


def is_p(el):
    return el.tag == qn("w:p")


def make_source_line(model, text):
    """Build the attribution paragraph, cloning the caption's paragraph
    properties so indentation and alignment match whatever the caption uses."""
    p = copy.deepcopy(model._p)
    # strip every run, keep the pPr
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    par = Paragraph(p, model._parent)
    par.alignment = model.alignment or WD_ALIGN_PARAGRAPH.CENTER
    pf = par.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(10)
    pf.keep_with_next = False
    r = par.add_run(text)
    r.italic = True
    r.bold = False
    r.font.name = FONT
    r.font.size = Pt(10)
    r.font.color.rgb = BLACK
    rpr = r._r.get_or_add_rPr()
    rf = OxmlElement("w:rFonts")
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    rpr.insert(0, rf)
    return p


chapter = None
added, skipped = 0, 0
for el in list(body.iterchildren()):
    if not is_p(el):
        continue
    par = Paragraph(el, doc)
    if par.style.name.lower().startswith(("toc", "table of")):
        continue
    t = par.text.strip()
    m = re.match(r"CHAPTER (ONE|TWO|THREE|FOUR|FIVE)$", t)
    if m:
        chapter = m.group(1)
    if not cap_re.match(t):
        continue

    # Don't double-add if the script is re-run.
    nxt = el.getnext()
    if nxt is not None and is_p(nxt) and \
            Paragraph(nxt, doc).text.strip().startswith("Source:"):
        skipped += 1
        continue

    text = GENERATED if chapter == "FOUR" else CONCEPTUAL
    # Keep the caption glued to its attribution line.
    par.paragraph_format.keep_with_next = True
    el.addnext(make_source_line(par, text))
    added += 1
    print("  %-58s -> %s" % (t[:58], text[:34] + "..."))

target = SRC
try:
    doc.save(target)
except PermissionError:
    target = SRC.replace(".docx", time.strftime("_%H%M%S.docx"))
    doc.save(target)
    print("\n!! %s is open in Word; wrote %s instead."
          % (os.path.basename(SRC), os.path.basename(target)))

print("\nadded %d source line(s), skipped %d already present" % (added, skipped))
print("saved:", target)
