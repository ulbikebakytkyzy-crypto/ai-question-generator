"""Базовая проверка качества: дубликаты + проверка вторым запросом к AI."""
import json
import re
from difflib import SequenceMatcher

from .generator import LEVELS
from .llm import chat_json


def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def is_duplicate(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    return jaccard >= 0.7 or SequenceMatcher(None, na, nb).ratio() >= 0.85


def dedupe(new_questions, existing_questions):
    """Убирает вопросы, похожие на уже существующие или друг на друга."""
    kept, dropped = [], []
    pool = [q["question"] for q in existing_questions]
    for q in new_questions:
        if any(is_duplicate(q["question"], p) for p in pool):
            dropped.append(q)
        else:
            kept.append(q)
            pool.append(q["question"])
    return kept, dropped


def ai_check(questions, topic, chunks, language="Russian", api_key=None, model=None):
    """Дополнительный запрос к AI. Дописывает quality_status / quality_comment каждому вопросу."""
    if not questions:
        return questions
    context = "\n\n".join(f"[стр. {c['page']}]\n{c['text']}" for c in chunks)
    items = []
    for i, q in enumerate(questions):
        item = {
            "id": i,
            "level": q["level"],
            "level_meaning": LEVELS[q["level"]][1],
            "question": q["question"],
            "correct_answer": q["answer"],
        }
        if q["qtype"] == "mcq":
            item["options"] = q["options"]
        items.append(item)

    system = "You are a strict reviewer of exam questions. Return ONLY a JSON object."
    user = f"""Topic: {topic['title']}
Check every question against the MATERIAL. For each question decide:
- "on_topic": the question is about the topic;
- "grounded_and_correct": the correct answer is supported by the material and is truly correct;
- "level_ok": the difficulty matches the declared level (level_meaning);
- "single_correct": (tests only) exactly one option is correct, otherwise true;
- "comment": one short sentence in {language} describing the problem, or "" if everything is fine.

JSON format: {{"results": [{{"id": 0, "on_topic": true, "grounded_and_correct": true, "level_ok": true, "single_correct": true, "comment": ""}}]}}

QUESTIONS:
{json.dumps(items, ensure_ascii=False)}

MATERIAL:
{context}"""
    try:
        data = chat_json(system, user, api_key=api_key, model=model, temperature=0)
    except Exception as e:  # noqa: BLE001
        for q in questions:
            q["quality_status"] = "unchecked"
            q["quality_comment"] = f"AI-проверка не выполнена: {e}"
        return questions

    by_id = {r.get("id"): r for r in data.get("results", []) if isinstance(r, dict)}
    for i, q in enumerate(questions):
        r = by_id.get(i)
        if not r:
            q["quality_status"], q["quality_comment"] = "unchecked", "Нет ответа проверки"
            continue
        flags = [r.get(k, True) for k in ("on_topic", "grounded_and_correct", "level_ok", "single_correct")]
        q["quality_status"] = "ok" if all(flags) else "warning"
        q["quality_comment"] = str(r.get("comment", "")).strip()
    return questions
