"""Доп. задание 3: экспорт вопросов в PDF (шрифт DejaVu — кириллица и казахские буквы)."""
import os

from fpdf import FPDF

from .generator import LEVELS

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")


def _pdf():
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    return pdf


def _line(pdf, text, size=10, bold=False, indent=0):
    pdf.set_font("DejaVu", "B" if bold else "", size)
    pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - indent, size * 0.5 + 1, str(text),
                   new_x="LMARGIN", new_y="NEXT")


def questions_pdf(questions, title="Вопросы", with_answers=True) -> bytes:
    pdf = _pdf()
    _line(pdf, title, 16, True)
    pdf.ln(3)
    by_doc = {}
    for q in questions:
        by_doc.setdefault((q.get("doc_name", ""), q["topic_title"]), []).append(q)
    num = 0
    for (doc, topic), qs in by_doc.items():
        pdf.ln(2)
        _line(pdf, f"{doc} — {topic}", 12, True)
        for q in sorted(qs, key=lambda x: (x["level"], x["id"])):
            num += 1
            kind = "тест" if q["qtype"] == "mcq" else "открытый"
            pdf.ln(1.5)
            _line(pdf, f"{num}. [Уровень {q['level']} — {LEVELS[q['level']][0]}, {kind}] {q['question']}", 10, True)
            if q["qtype"] == "mcq" and q.get("options"):
                for i, o in enumerate(q["options"]):
                    _line(pdf, f"{'ABCD'[i]}) {o}", 10, indent=5)
            if with_answers:
                ans = q["answer"]
                if q["qtype"] == "mcq" and q.get("correct_index") is not None:
                    ans = f"{'ABCD'[q['correct_index']]}) {ans}"
                _line(pdf, f"Ответ: {ans}", 9, indent=5)
                if q.get("explanation"):
                    _line(pdf, f"Объяснение: {q['explanation']}", 9, indent=5)
                if q.get("source_page"):
                    _line(pdf, f"Источник: стр./раздел {q['source_page']}", 8, indent=5)
    return bytes(pdf.output())
