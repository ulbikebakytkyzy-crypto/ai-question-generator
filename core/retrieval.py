"""Простой поиск релевантных фрагментов (TF-IDF без внешних библиотек)."""
import math
import random
import re
from collections import Counter


def tokens(text: str):
    # обрезка до 5 символов — грубая «стемминг»-замена для русского/казахского/английского
    return [w[:5] for w in re.findall(r"\w{3,}", text.lower())]


def select_chunks(chunks, query_terms, max_chars=9000, randomize=True, seed=None):
    """Выбирает фрагменты, наиболее связанные с темой, в порядке следования в документе."""
    if not chunks:
        return []
    q = set(tokens(" ".join(query_terms)))
    docs = [Counter(tokens(c["text"])) for c in chunks]
    n = len(chunks)
    df = Counter()
    for d in docs:
        df.update(d.keys())

    scored = []
    for i, d in enumerate(docs):
        s = sum((1 + math.log(d[t])) * math.log(1 + n / df[t]) for t in q if t in d)
        scored.append((s, i))
    scored.sort(reverse=True)
    ranked = [i for s, i in scored if s > 0]

    if not ranked:  # ничего не нашли — берём равномерно по документу
        step = max(1, n // 8)
        ranked = list(range(0, n, step))

    pool, total = [], 0
    for i in ranked:
        pool.append(i)
        total += len(chunks[i]["text"])
        if total >= max_chars * 2:
            break

    order = pool
    if randomize and len(pool) > 3:
        rng = random.Random(seed)
        rest = pool[3:]
        rng.shuffle(rest)
        order = pool[:3] + rest  # три лучших всегда, остальное — для разнообразия

    picked, total = [], 0
    for i in order:
        ln = len(chunks[i]["text"])
        if picked and total + ln > max_chars:
            continue
        picked.append(i)
        total += ln
    return [chunks[i] for i in sorted(picked)]
