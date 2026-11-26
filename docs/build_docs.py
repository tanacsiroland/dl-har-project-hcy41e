"""Build docs/dokumentacio.docx from docs/dokumentacio.md with pandoc.

Run from the project root:  python docs/build_docs.py
Needs the packages in docs/requirements.txt (pypandoc_binary ships its own pandoc).
"""

import subprocess
import tempfile
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph

DOCS_DIR = Path(__file__).resolve().parent
SOURCE = DOCS_DIR / "dokumentacio.md"
TARGET = DOCS_DIR / "dokumentacio.docx"
FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)
LATER_TBLPR_TAGS = {
    qn(f"w:{name}")
    for name in ("shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription")
}


def set_style(doc, name, size, bold=False, italic=False, before=0, after=6, align=None):
    style = doc.styles[name]
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = BLACK
    # East Asian and complex-script font slots would otherwise keep the theme font.
    rfonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(attr), None)  # a theme font would override FONT
    fmt = style.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    if align is not None:
        fmt.alignment = align


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    run.append(text)
    field.append(run)
    paragraph._p.append(field)


def make_reference_docx(path: Path) -> None:
    """Start from pandoc's default reference document and set fonts, sizes and margins."""
    default = subprocess.run(
        [pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"],
        check=True,
        capture_output=True,
    ).stdout
    path.write_bytes(default)
    doc = Document(str(path))

    set_style(doc, "Normal", 12)
    for name in ("Body Text", "First Paragraph", "Compact"):
        set_style(doc, name, 12, after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        doc.styles[name].paragraph_format.line_spacing = 1.15
    set_style(doc, "Title", 20, bold=True, before=120, after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_style(doc, "Subtitle", 14, after=36, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_style(doc, "Author", 12, after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_style(doc, "Heading 1", 14, bold=True, before=18, after=6)
    set_style(doc, "Heading 2", 12, bold=True, before=12, after=4)
    set_style(doc, "Image Caption", 11, italic=True, after=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_style(doc, "Table Caption", 11, italic=True, before=6, after=4)
    table_text = doc.styles.add_style("Table Text", WD_STYLE_TYPE.PARAGRAPH)
    table_text.base_style = doc.styles["Compact"]
    set_style(doc, "Table Text", 10, after=0, align=WD_ALIGN_PARAGRAPH.LEFT)
    table_text.paragraph_format.line_spacing = 1.0

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(2.5))
    section.header_distance, section.footer_distance, section.gutter = Cm(1.25), Cm(1.25), Cm(0)
    section.different_first_page_header_footer = True  # no page number on the title page
    footer = section.footer
    footer.is_linked_to_previous = False
    add_page_number(footer.paragraphs[0])
    doc.save(str(path))


def format_tables(doc) -> None:
    """Thin horizontal lines in every table and some space before the next paragraph."""
    for table in doc.tables:
        following = table._tbl.getnext()
        if following is not None and following.tag == qn("w:p"):
            Paragraph(following, table._parent).paragraph_format.space_before = Pt(8)
        last_row = len(table.rows) - 1
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.style = doc.styles["Table Text"]
                    # Keep short tables on one page.
                    paragraph.paragraph_format.keep_with_next = row_index < last_row
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "insideH"):
            element = OxmlElement(f"w:{edge}")
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "4")
            element.set(qn("w:color"), "808080")
            borders.append(element)
        tbl_pr = table._tbl.tblPr
        # Schema order: tblBorders must come before these elements.
        later = [child for child in tbl_pr if child.tag in LATER_TBLPR_TAGS]
        if later:
            later[0].addprevious(borders)
        else:
            tbl_pr.append(borders)


def format_references(doc) -> None:
    """Hanging indent for the IEEE-style "[n] ..." entries after the last heading."""
    in_references = False
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            in_references = paragraph.text.endswith("Irodalomjegyzék")
        elif in_references and paragraph.text.startswith("["):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.left_indent = Cm(0.9)
            paragraph.paragraph_format.first_line_indent = Cm(-0.9)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "reference.docx"
        make_reference_docx(reference)
        pypandoc.convert_file(
            str(SOURCE),
            "docx",
            outputfile=str(TARGET),
            extra_args=[f"--reference-doc={reference}", f"--resource-path={DOCS_DIR}"],
        )
    doc = Document(str(TARGET))
    format_tables(doc)
    format_references(doc)
    doc.save(str(TARGET))
    print(f"Saved {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
