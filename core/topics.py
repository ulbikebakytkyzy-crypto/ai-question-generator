"""Определение основных тем документа с помощью AI."""
from .llm import chat_json

MAX_SAMPLE_CHARS = 24000


def _sample(chunks):
    total = sum(len(c["text"]) for c in chunks)
    if total <= MAX_SAMPLE_CHARS:
        picked = chunks
    else:  # равномерно по всему документу
        step = total / MAX_SAMPLE_CHARS
        picked, acc, nxt = [], 0, 0.0
        for c in chunks:
            if acc >= nxt:
                picked.append(c)
                nxt += step * len(c["text"])
            acc += len(c["text"])
    text, size = [], 0
    for c in picked:
        piece = f"[стр. {c['page']}] {c['text']}"
        if size + len(piece) > MAX_SAMPLE_CHARS:
            break
        text.append(piece)
        size += len(piece)
    return "\n\n".join(text)


def detect_topics(chunks, language="Russian", api_key=None, model=None):
    system = (
        "You are an experienced teacher who analyses textbooks. "
        "Return ONLY a JSON object."
    )
    user = f"""Below are excerpts from a study material. Identify its 4–10 main topics
(chapters / key subjects) in the order they appear.

For each topic return:
- "title": short topic name, written in {language}
- "description": one sentence, in {language}
- "keywords": 5–10 important terms EXACTLY as they are written in the material
  (in the material's own language) — they are used to search the text.

JSON format: {{"topics": [{{"title": "...", "description": "...", "keywords": ["..."]}}]}}

MATERIAL:
{_sample(chunks)}"""
    data = chat_json(system, user, api_key=api_key, model=model, temperature=0.2)
    out = []
    for t in data.get("topics", []):
        title = str(t.get("title", "")).strip()
        if not title:
            continue
        kws = [str(k).strip() for k in t.get("keywords", []) if str(k).strip()]
        out.append(
            {
                "title": title,
                "description": str(t.get("description", "")).strip(),
                "keywords": kws,
            }
        )
    return out
