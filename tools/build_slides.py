"""Build docs/prezentacio.pptx: four plain slides that summarize the project.

Run from the project root:  python tools/build_slides.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

TARGET = Path(__file__).resolve().parent.parent / "docs" / "prezentacio.pptx"
FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(0x59, 0x59, 0x59)
TABLE_GRID_STYLE = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
LEFT = Inches(0.7)
WIDTH = Inches(11.93)

TITLE = (
    "Emberi tevékenységek felismerése\n"
    "okostelefon-szenzorjelekből\n"
    "kis méretű 1D konvolúciós hálóval"
)
AUTHOR = (
    "Tanácsi Roland (HCY41E)\n"
    "Mély gépi tanulás (NSXMG1HMLF), Óbudai Egyetem\n"
    "2026. november 26."
)

SLIDES = [
    {
        "title": "Feladat és módszer",
        "bullets": [
            "UCI HAR: 30 személy, 6 tevékenység, 9 csatorna × 128 minta",
            "Felosztás személyek szerint: 16 tanító, 5 validációs, 9 teszt",
            "Klasszikus modellek: 72 statisztika, logisztikus regresszió, véletlen erdő",
            "1D-CNN: 3 konvolúciós blokk, 74 582 paraméter, keresztvalidációval hangolva",
        ],
    },
    {
        "title": "Eredmények a teszthalmazon (9 személy)",
        "table": [
            ["Modell", "Makro-F1", "Pontosság", "Hibás ablakok"],
            ["Logisztikus regresszió", "0,902", "90,4%", "284"],
            ["Véletlen erdő", "0,843", "84,6%", "455"],
            ["Alap 1D-CNN", "0,926", "92,5%", "221"],
            ["Hangolt 1D-CNN", "0,928", "92,7%", "215"],
        ],
        "caption": "A hangolt háló mindkét klasszikus modellnél szignifikánsan jobb "
        "(McNemar-próba, p < 0,001).\nMozgásnál jobb (F1: 0,96–1,00), ülésnél és állásnál "
        "gyengébb: a 215 hibából 174 az ülés és az állás felcserélése.",
    },
    {
        "title": "Összegzés",
        "bullets": [
            "H1 nagyrészt teljesül: a háló 0,928, a klasszikus modellek 0,902 és 0,843",
            "H2 teljesül: gravitáció nélkül a fekvés F1 értéke 1,00-ról 0,87-re esik",
            "Korlát: csak 30 személy, a telefon mindig a derékon",
            "Következő lépés: forgatásos adatbővítés, tanítás mind a 21 személyen",
        ],
    },
]


def add_text(slide, text, top, height, size, bold=False, color=BLACK):
    box = slide.shapes.add_textbox(LEFT, top, WIDTH, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        run = paragraph.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return frame


def add_bullets(slide, bullets):
    frame = add_text(slide, "\n".join("•  " + b for b in bullets), Inches(1.6), Inches(5.0), 24)
    for paragraph in frame.paragraphs:
        paragraph.space_after = Pt(14)


def add_table(slide, rows):
    shape = slide.shapes.add_table(
        len(rows), len(rows[0]), LEFT, Inches(1.6), Inches(10.0), Inches(0.5) * len(rows)
    )
    shape._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId"
    ).text = TABLE_GRID_STYLE
    table = shape.table
    for col, width in enumerate((3.8, 1.9, 1.9, 2.4)):
        table.columns[col].width = Inches(width)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            paragraph = table.cell(r, c).text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = paragraph.add_run()
            run.text = value
            run.font.name = FONT
            run.font.size = Pt(20)
            run.font.bold = r == 0 or row[0] == "Hangolt 1D-CNN"
            run.font.color.rgb = BLACK


def add_slide_number(slide, number):
    box = slide.shapes.add_textbox(Inches(12.13), Inches(6.95), Inches(0.9), Inches(0.4))
    paragraph = box.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.RIGHT
    run = paragraph.add_run()
    run.text = str(number)
    run.font.name = FONT
    run.font.size = Pt(12)
    run.font.color.rgb = GRAY


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    title_slide = prs.slides.add_slide(blank)
    add_text(title_slide, TITLE, Inches(1.9), Inches(2.0), 32, bold=True)
    add_text(title_slide, AUTHOR, Inches(4.4), Inches(1.5), 20, color=GRAY)

    for number, spec in enumerate(SLIDES, start=2):
        slide = prs.slides.add_slide(blank)
        add_text(slide, spec["title"], Inches(0.5), Inches(0.9), 30, bold=True)
        if "bullets" in spec:
            add_bullets(slide, spec["bullets"])
        if "table" in spec:
            add_table(slide, spec["table"])
            add_text(slide, spec["caption"], Inches(4.6), Inches(1.6), 20)
        add_slide_number(slide, number)

    prs.save(str(TARGET))
    print(f"Saved {TARGET} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
