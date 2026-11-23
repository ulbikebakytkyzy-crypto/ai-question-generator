"""Доп. задание 2: проверка открытых ответов пользователя через AI."""
import json

from .llm import chat_json


def grade_open_answers(items, language="Russian", api_key=None, model=None):
    """items: список dict(question, reference, user_answer).
    Возвращает список dict(score 0..2, verdict, feedback) в том же порядке."""
    if not items:
        return []
    payload = [{"id": i, "question": it["question"], "reference_answer": it["reference"],
                "student_answer": it["user_answer"]} for i, it in enumerate(items)]
    system = "You are a fair but strict teacher grading short student answers. Return ONLY a JSON object."
    user = f"""Compare each student answer with the reference answer.
Score: 2 = correct and complete, 1 = partially correct (key idea present but incomplete or with a minor mistake),
0 = wrong, empty or off-topic. Accept answers in other words or another language if the meaning matches.
"feedback": 1–2 short sentences in {language}: what is right and what is missing.

JSON format: {{"results": [{{"id": 0, "score": 2, "feedback": "..."}}]}}

ANSWERS:
{json.dumps(payload, ensure_ascii=False)}"""
    data = chat_json(system, user, api_key=api_key, model=model, temperature=0)
    by_id = {r.get("id"): r for r in data.get("results", []) if isinstance(r, dict)}
    out = []
    for i, it in enumerate(items):
        r = by_id.get(i, {})
        try:
            score = max(0, min(2, int(r.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        if not it["user_answer"].strip():
            score = 0
        out.append({"score": score, "feedback": str(r.get("feedback", "")).strip() or "—"})
    return out
