"""Build docs/prezentacio.pptx: clean 8-slide Hungarian project summary.

Run from the project root:  python tools/build_slides.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "tools" / "report_source" / "figures"
TARGET = ROOT / "docs" / "prezentacio.pptx"
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
    "2026. december 5."
)


def add_text(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    size: int,
    bold: bool = False,
    color: RGBColor = BLACK,
):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
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


def add_bullets(
    slide,
    bullets: list[str],
    left: float = 0.7,
    top: float = 1.5,
    width: float = 11.93,
    height: float = 5.0,
    size: int = 22,
    space_after: int = 12,
) -> None:
    frame = add_text(
        slide,
        "\n".join("•  " + b for b in bullets),
        left=left,
        top=top,
        width=width,
        height=height,
        size=size,
    )
    for paragraph in frame.paragraphs:
        paragraph.space_after = Pt(space_after)


def add_table(slide, rows: list[list[str]], top: float = 1.55) -> None:
    shape = slide.shapes.add_table(
        len(rows), len(rows[0]), LEFT, Inches(top), Inches(11.6), Inches(0.5) * len(rows)
    )
    shape._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId"
    ).text = TABLE_GRID_STYLE
    table = shape.table
    for col, width in enumerate((3.6, 1.8, 1.8, 2.1, 2.3)):
        table.columns[col].width = Inches(width)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            paragraph = table.cell(r, c).text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = paragraph.add_run()
            run.text = value
            run.font.name = FONT
            run.font.size = Pt(19)
            run.font.bold = r == 0 or row[0] == "Hangolt 1D-CNN"
            run.font.color.rgb = BLACK


def add_slide_number(slide, number: int) -> None:
    box = slide.shapes.add_textbox(Inches(12.13), Inches(6.95), Inches(0.9), Inches(0.4))
    paragraph = box.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.RIGHT
    run = paragraph.add_run()
    run.text = str(number)
    run.font.name = FONT
    run.font.size = Pt(12)
    run.font.color.rgb = GRAY


def new_content_slide(prs: Presentation, blank, title: str, number: int):
    slide = prs.slides.add_slide(blank)
    add_text(slide, title, left=0.7, top=0.45, width=11.93, height=0.8, size=28, bold=True)
    add_slide_number(slide, number)
    return slide


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Title slide
    s1 = prs.slides.add_slide(blank)
    add_text(s1, TITLE, left=0.7, top=1.9, width=11.93, height=2.0, size=32, bold=True)
    add_text(s1, AUTHOR, left=0.7, top=4.4, width=11.93, height=1.5, size=20, color=GRAY)

    # 2. Task, dataset, split, and hypotheses
    s2 = new_content_slide(prs, blank, "Feladat, adatkészlet és hipotézisek", 2)
    add_bullets(
        s2,
        [
            "UCI HAR adatkészlet: 30 személy, 6 tevékenység, 9 csatorna × 128 minta (50 Hz; 2,56 s)",
            "Személyek szerinti felosztás: 16 tanító (5 563), 5 validációs (1 789), 9 teszt (2 947 ablak)",
            "H1: a nyers jelekre tanított 1D-CNN jobb makro-F1-et ér el, főleg mozgásnál",
            "H2: a teljes gyorsulás (gravitáció) elhagyása rontja a nyugalmi testhelyzetek felismerését",
        ],
        space_after=16,
    )

    # 3. Compared models and training
    s3 = new_content_slide(prs, blank, "Összehasonlított modellek és tanítás", 3)
    add_bullets(
        s3,
        [
            "Klasszikus modellek: ablakonként 72 statisztika (logisztikus regresszió, véletlen erdő)",
            "Alap 1D-CNN: 3 konvolúciós blokk, globális átlagolás az idő mentén (38 054 paraméter)",
            "Hangolt 1D-CNN: 48–96–160 szűrő, 9 hosszú első szűrő (74 582 paraméter, 0,3 MB)",
            "Tanítás: standardizálás csak a tanítóadaton, AdamW, koszinusz ütemezés, korai leállítás",
        ],
        space_after=16,
    )

    # 4. Tuning and 5-fold CV
    s4 = new_content_slide(prs, blank, "Hiperparaméter-hangolás és 5-szörös keresztvalidáció", 4)
    add_bullets(
        s4,
        [
            "8 beállítás futott a validációs halmazon, 4 beállítás az 5-szörös keresztvalidációban",
            "Győztes: Hangolt 1D-CNN (0,956 ± 0,033 keresztvalidációs, 0,973 validációs makro-F1)",
            "3 véletlenmag átlaga: 0,956 ± 0,012 — a kis validációs halmaz túl kedvező képet adhat",
        ],
        left=0.7,
        top=1.6,
        width=4.65,
        height=4.9,
        size=20,
        space_after=14,
    )
    s4.shapes.add_picture(
        str(FIG_DIR / "model_comparison.png"),
        Inches(5.55),
        Inches(1.50),
        width=Inches(7.1),
    )

    # 5. Channel ablation (H2)
    s5 = new_content_slide(prs, blank, "Elhagyásos vizsgálat (H2 igazolása)", 5)
    add_bullets(
        s5,
        [
            "Mind a 9 csatorna: 0,956 ± 0,033 makro-F1 (Fekvés F1: 1,000)",
            "Gravitáció nélkül (6 csatorna): a makro-F1 0,925-re, a fekvés F1 0,866-ra esik",
            "Teljes gyorsulás + szögsebesség (6 csatorna): 0,949 ± 0,040 — a háló kinyeri a mozgást",
        ],
        left=0.7,
        top=1.35,
        width=11.93,
        height=1.75,
        size=19,
        space_after=8,
    )
    s5.shapes.add_picture(
        str(FIG_DIR / "channel_ablation.png"),
        Inches(2.90),
        Inches(3.25),
        width=Inches(7.5),
    )

    # 6. Locked test set results
    s6 = new_content_slide(prs, blank, "Végső eredmények a teszthalmazon (9 személy, 2 947 ablak)", 6)
    add_table(
        s6,
        [
            ["Modell", "Makro-F1", "Pontosság", "Hibás ablakok", "Személyenkénti átlag"],
            ["Logisztikus regresszió", "0,902", "90,4%", "284", "0,896 ± 0,067"],
            ["Véletlen erdő", "0,843", "84,6%", "455", "0,823 ± 0,106"],
            ["Alap 1D-CNN", "0,926", "92,5%", "221", "0,914 ± 0,090"],
            ["Hangolt 1D-CNN", "0,928", "92,7%", "215", "0,916 ± 0,093"],
        ],
        top=1.55,
    )
    add_bullets(
        s6,
        [
            "A hangolt háló mindkét klasszikus modellnél szignifikánsan jobb (McNemar-próba, p < 0,001)",
            "A két 1D-CNN közötti különbség nem szignifikáns (p = 0,49; 6 ablak eltérés)",
            "Futási idő (1 ablak): 0,30 ms GPU-n (RTX 5090), 0,39 ms asztali CPU-n",
        ],
        left=0.7,
        top=4.35,
        width=11.93,
        height=2.3,
        size=19,
        space_after=8,
    )

    # 7. Error analysis and per-subject generalization
    s7 = new_content_slide(prs, blank, "Hibaelemzés és személyenkénti általánosítás", 7)
    s7.shapes.add_picture(
        str(FIG_DIR / "test_confusion_matrices.png"),
        Inches(0.65),
        Inches(1.35),
        height=Inches(5.35),
    )
    add_bullets(
        s7,
        [
            "Mozgások: F1 0,96–1,00 (Séta 496/496, Lépcsőn le 420/420); Fekvés 0,996",
            "A 215 hibából 174 (80,9%) az ülés és állás felcserélése (logisztikus regresszió itt jobb)",
            "9 személyből 7-en a háló vezet; a 2. és 9. személyen a logisztikus regresszió erősebb",
        ],
        left=6.25,
        top=1.35,
        width=6.4,
        height=2.3,
        size=17,
        space_after=8,
    )
    s7.shapes.add_picture(
        str(FIG_DIR / "test_per_subject_f1.png"),
        Inches(6.25),
        Inches(3.75),
        width=Inches(6.2),
    )

    # 8. Summary, limitations, next steps
    s8 = new_content_slide(prs, blank, "Összegzés, korlátok és továbbfejlesztés", 8)
    add_bullets(
        s8,
        [
            "H1 nagyrészt teljesül: 0,928 vs 0,902 és 0,843 (előny a mozgásoknál, ülés/állás gyengébb)",
            "H2 teljesül: gravitáció nélkül a fekvés F1 értéke 1,000-ról 0,866-ra esik",
            "Korlátok: 30 személy, derékon rögzített telefon, 16 tanítószemély, telefonos mérés nincs",
            "Következő lépés: forgatásos adatbővítés, gyorsulás-statisztikák, tanítás 21 személyen",
        ],
        space_after=16,
    )

    prs.save(str(TARGET))
    print(f"Saved {TARGET} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
