"""Генерация вопросов 5 уровней сложности (тестовые и открытые) по фрагментам текста."""
import random

from .llm import chat_json

LEVELS = {
    1: ("Очень лёгкий", "Простое определение или факт, прямо названный в тексте. Ответ — слово или одно предложение."),
    2: ("Лёгкий", "Проверка понимания: объяснить своими словами, сравнить два понятия, привести пример из текста."),
    3: ("Средний", "Применение знаний: небольшая практическая задача или ситуация, где нужно применить правило/метод из текста "
                   "(для программирования — написать короткий код, для точных наук — посчитать)."),
    4: ("Сложный", "Анализ: найти ошибку и объяснить причину, разобрать ситуацию, определить последствия, сравнить подходы."),
    5: ("Очень сложный", "Нестандартная задача: объединяет несколько понятий темы, требует обоснования выбора решения "
                         "и многошагового рассуждения."),
}

LANGS = {"Русский": "Russian", "Қазақша": "Kazakh", "English": "English"}
QTYPES = {"mcq": "Тестовые (4 варианта)", "open": "Открытые"}


def _build_prompt(topic, level, qtype, n, context, avoid, language):
    lname, ldesc = LEVELS[level]
    if qtype == "mcq":
        fmt = (
            'Multiple-choice questions. For each: "question", "answer" (the single correct option), '
            '"distractors" (exactly 3 plausible but clearly wrong options, similar length/style to the correct one; '
            'never "all of the above"/"none of the above"), "explanation" (1–2 sentences why the answer is correct), '
            '"source_page" (integer taken from the [стр. N] labels).'
        )
    else:
        fmt = (
            'Open questions. For each: "question", "answer" (a complete model answer or solution), '
            '"distractors": [] (empty list), "explanation" (1–2 sentences about the approach), '
            '"source_page" (integer taken from the [стр. N] labels).'
        )
    avoid_txt = "\n".join(f"- {a}" for a in avoid[-30:]) or "(none)"
    system = (
        "You are an expert teacher who writes exam questions strictly from a given study material. "
        "Use ONLY facts contained in the provided fragments; never add outside knowledge. "
        "Return ONLY a JSON object."
    )
    user = f"""Topic: {topic['title']}
Number of questions: {n}
Difficulty level: {level} of 5 — "{lname}": {ldesc}
Question type: {fmt}
Write everything in {language}. Keep code, formulas and terms from the material unchanged.

Rules:
- Every question must be answerable from the fragments below and must belong to the topic "{topic['title']}".
- The difficulty must clearly match level {level}; do not make all questions look alike.
- Do not repeat or paraphrase these already existing questions:
{avoid_txt}

JSON format: {{"questions": [{{"question": "...", "answer": "...", "distractors": [], "explanation": "...", "source_page": 1}}]}}

MATERIAL FRAGMENTS:
{context}"""
    return system, user


def generate(*, topic, level, qtype, n, chunks, avoid=None, language="Russian", api_key=None, model=None):
    context = "\n\n".join(f"[стр. {c['page']}]\n{c['text']}" for c in chunks)
    valid_pages = {c["page"] for c in chunks}
    system, user = _build_prompt(topic, level, qtype, n, context, avoid or [], language)
    data = chat_json(system, user, api_key=api_key, model=model, temperature=0.6)

    out = []
    for item in data.get("questions", []):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        try:
            page = int(item.get("source_page"))
        except (TypeError, ValueError):
            page = None
        if page not in valid_pages:
            page = None
        rec = {
            "level": level,
            "qtype": qtype,
            "question": q,
            "answer": a,
            "explanation": str(item.get("explanation", "")).strip(),
            "source_page": page,
            "options": None,
            "correct_index": None,
        }
        if qtype == "mcq":
            seen, dis = {a.lower()}, []
            for d in item.get("distractors", []) or []:
                d = str(d).strip()
                if d and d.lower() not in seen:
                    seen.add(d.lower())
                    dis.append(d)
            if len(dis) < 3:
                continue  # некорректный тест — пропускаем
            options = [a] + dis[:3]
            random.shuffle(options)  # правильный ответ не всегда на одной позиции
            rec["options"] = options
            rec["correct_index"] = options.index(a)
        out.append(rec)
        if len(out) >= n:
            break
    return out
