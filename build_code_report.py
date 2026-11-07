"""Knits the executed notebook (code, comments, outputs, charts) into a Word document."""
import base64, io, json, re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

NB = "energy_demand_forecasting.ipynb"
OUT = "ANA500_MicroProject4_Code_Report.docx"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(0.8)
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)


def shade(paragraph, hex_fill):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    pPr.append(shd)


def add_inline(paragraph, text):
    """Adds text with **bold** and `code` spans as separate runs."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        elif part.startswith("`"):
            r = paragraph.add_run(part[1:-1]); r.font.name = "Courier New"
        else:
            paragraph.add_run(part.replace("\\$", "$"))


def monospace_block(text, fill, size, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    shade(p, fill)
    lines = text.rstrip("\n").split("\n")
    for i, line in enumerate(lines):
        r = p.add_run(line)
        r.font.name = "Courier New"; r.font.size = Pt(size)
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
        if color:
            r.font.color.rgb = color
        if i < len(lines) - 1:
            r.add_break()


def add_md_table(rows):
    """Renders markdown table rows as a Word table with a bold header row; the --- separator row is skipped."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Light Grid Accent 1"
    for i, row in enumerate(cells):
        for j, val in enumerate(row):
            p = t.cell(i, j).paragraphs[0]
            add_inline(p, val)
            for r in p.runs:
                r.font.size = Pt(9.5)
                if i == 0:
                    r.bold = True
    doc.add_paragraph()


nb = json.load(open(NB))
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        # Walk the cell in order: markdown tables ("| a | b |" rows) become Word tables, the rest becomes text
        blocks = []
        for line in src.split("\n"):
            is_table = line.strip().startswith("|")
            if blocks and blocks[-1][0] == is_table:
                blocks[-1][1].append(line)
            else:
                blocks.append((is_table, [line]))
        for is_table, block in blocks:
            if is_table:
                add_md_table(block)
                continue
            # Join soft-wrapped lines into one paragraph; headings, list items and "**Label:**" lines start new ones
            lines = []
            for line in block:
                starts_block = re.match(r"^(#+\s|\s*[*-]\s|\d+\.\s|\*\*)", line)
                if lines and line.strip() and lines[-1].strip() and not starts_block and not lines[-1].startswith("#"):
                    lines[-1] += " " + line.strip()
                else:
                    lines.append(line)
            for line in lines:
                if not line.strip():
                    continue
                m = re.match(r"^(#+)\s+(.*)", line)
                if m:
                    doc.add_heading(m.group(2), level=min(len(m.group(1)) - 1, 3) if len(m.group(1)) > 1 else 0)
                elif re.match(r"^\s*[*-]\s+", line):
                    add_inline(doc.add_paragraph(style="List Bullet"), re.sub(r"^\s*[*-]\s+", "", line))
                elif re.match(r"^\d+\.\s+", line):
                    add_inline(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\.\s+", "", line))
                else:
                    add_inline(doc.add_paragraph(), line)
        continue

    monospace_block(src, "F2F4F5", 8.5)
    for out in cell.get("outputs", []):
        if out["output_type"] == "stream":
            monospace_block(ANSI.sub("", "".join(out["text"])), "FFFFFF", 8, RGBColor(0x33, 0x33, 0x33))
        elif "data" in out:
            if "image/png" in out["data"]:
                doc.add_picture(io.BytesIO(base64.b64decode(out["data"]["image/png"])), width=Inches(6.4))
            elif "text/plain" in out["data"]:
                monospace_block(ANSI.sub("", "".join(out["data"]["text/plain"])), "FFFFFF", 8, RGBColor(0x33, 0x33, 0x33))
        elif out["output_type"] == "error":
            raise SystemExit("Notebook contains an error output; re-run it first.")

doc.save(OUT)
print("Saved", OUT)
