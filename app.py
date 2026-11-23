"""AI-генератор вопросов по учебным материалам (Streamlit)."""
import json
import os
import random

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core import db, extract, retrieval, validator  # noqa: E402
from core import topics as topics_mod  # noqa: E402
from core.generator import LANGS, LEVELS, QTYPES, generate  # noqa: E402
from core.llm import DEFAULT_MODEL, LLMError  # noqa: E402

st.set_page_config(page_title="AI-генератор вопросов", page_icon="📚", layout="wide")
db.init()

STATUS_ICON = {"ok": "✅", "warning": "⚠️", "unchecked": "⏺"}
VERDICTS = ["", "ок", "ошибка в ответе", "не по теме", "не тот уровень", "повтор", "другое"]

# ---------------------------------------------------------------- боковая панель
st.sidebar.title("📚 AI-генератор вопросов")
api_key = st.sidebar.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
model = st.sidebar.text_input("Модель", value=DEFAULT_MODEL)
lang_label = st.sidebar.selectbox("Язык вопросов", list(LANGS.keys()))
language = LANGS[lang_label]

docs = db.list_documents()
doc = None
if docs:
    ids = [d["id"] for d in docs]
    cur = st.session_state.get("doc_id")
    sel = st.sidebar.selectbox(
        "Документ", ids, index=ids.index(cur) if cur in ids else 0,
        format_func=lambda i: next(d["name"] for d in docs if d["id"] == i),
    )
    st.session_state["doc_id"] = sel
    doc = next(d for d in docs if d["id"] == sel)
    st.sidebar.caption(f"Загружен: {doc['uploaded_at']} · страниц/разделов: {doc['n_pages']}")
    if st.sidebar.button("🗑 Удалить документ и его вопросы"):
        db.delete_document(doc["id"])
        st.session_state.pop("doc_id", None)
        st.rerun()

tab_doc, tab_gen, tab_q, tab_quiz, tab_export = st.tabs(
    ["1. Документ и темы", "2. Генерация", "3. Вопросы", "4. Мини-тест", "5. Экспорт и проверка"]
)

# ---------------------------------------------------------------- 1. документ и темы
with tab_doc:
    up = st.file_uploader("Загрузите учебный материал (PDF с текстом или TXT)", type=["pdf", "txt"])
    if up and st.button("Обработать документ", type="primary"):
        try:
            with st.spinner("Извлекаю текст..."):
                pages = extract.read_document(up.name, up.getvalue())
                chunks = extract.make_chunks(pages)
                new_id = db.add_document(up.name, len(pages), chunks)
            with st.spinner("AI определяет темы..."):
                db.save_topics(new_id, topics_mod.detect_topics(chunks, language, api_key, model))
            st.session_state["doc_id"] = new_id
            st.rerun()
        except (ValueError, RuntimeError, LLMError) as e:
            st.error(str(e))

    if doc:
        st.subheader(f"📄 {doc['name']}")
        chunks = db.get_chunks(doc["id"])
        st.caption(f"Фрагментов текста: {len(chunks)} · символов: {sum(len(c['text']) for c in chunks):,}")
        with st.expander("Превью извлечённого текста"):
            st.text("\n\n".join(c["text"] for c in chunks[:3])[:3000])
        topics = db.get_topics(doc["id"])
        c1, c2 = st.columns([4, 1])
        c1.subheader("Найденные темы")
        if c2.button("🔄 Определить заново"):
            try:
                with st.spinner("AI определяет темы..."):
                    db.save_topics(doc["id"], topics_mod.detect_topics(chunks, language, api_key, model))
                st.rerun()
            except LLMError as e:
                st.error(str(e))
        if not topics:
            st.info("Темы не найдены — нажмите «Определить заново».")
        for i, t in enumerate(topics, 1):
            st.markdown(f"**{i}. {t['title']}** — {t['description']}")
            st.caption("Ключевые слова: " + ", ".join(t["keywords"]))
    else:
        st.info("Загрузите документ, чтобы начать.")

# ---------------------------------------------------------------- 2. генерация
with tab_gen:
    if not doc:
        st.info("Сначала загрузите документ.")
    else:
        topics = db.get_topics(doc["id"])
        if not topics:
            st.warning("У документа нет тем.")
        else:
            topic = st.selectbox("Тема", topics, format_func=lambda t: t["title"])
            qtypes = st.multiselect("Тип вопросов", list(QTYPES), default=list(QTYPES), format_func=QTYPES.get)
            levels = st.multiselect(
                "Уровни сложности", list(LEVELS), default=list(LEVELS),
                format_func=lambda l: f"{l} — {LEVELS[l][0]}",
            )
            n = st.slider("Количество вопросов (на каждый уровень и каждый тип)", 1, 10, 3)
            check = st.checkbox("Проверять качество дополнительным запросом к AI", value=True)

            if st.button("✨ Сгенерировать", type="primary", disabled=not (qtypes and levels)):
                chunks = db.get_chunks(doc["id"])
                terms = [topic["title"]] + topic["keywords"]
                existing = db.list_questions(doc_id=doc["id"], topic_id=topic["id"])
                combos = [(l, t) for l in sorted(levels) for t in qtypes]
                bar = st.progress(0.0, text="Начинаю...")
                added = dups_total = 0
                for i, (lvl, qt) in enumerate(combos):
                    bar.progress(i / len(combos), text=f"Уровень {lvl} · {QTYPES[qt]}")
                    ctx = retrieval.select_chunks(chunks, terms, seed=random.random())
                    try:
                        qs = generate(
                            topic=topic, level=lvl, qtype=qt, n=n, chunks=ctx,
                            avoid=[q["question"] for q in existing], language=language,
                            api_key=api_key, model=model,
                        )
                    except LLMError as e:
                        st.error(str(e))
                        break
                    qs, dropped = validator.dedupe(qs, existing)
                    dups_total += len(dropped)
                    if check:
                        validator.ai_check(qs, topic, ctx, language, api_key, model)
                    db.add_questions(doc["id"], topic, qs)
                    existing += qs
                    added += len(qs)
                bar.progress(1.0, text="Готово")
                st.success(f"Добавлено вопросов: {added}. Отброшено повторов: {dups_total}. "
                           "Смотрите вкладку «3. Вопросы».")

# ---------------------------------------------------------------- 3. вопросы
with tab_q:
    if not doc:
        st.info("Сначала загрузите документ.")
    else:
        topics = db.get_topics(doc["id"])
        f1, f2, f3, f4 = st.columns(4)
        ft = f1.selectbox("Тема", [None] + topics, format_func=lambda t: "Все" if t is None else t["title"])
        fl = f2.selectbox("Уровень", [None, *LEVELS], format_func=lambda l: "Все" if l is None else f"{l} — {LEVELS[l][0]}")
        fq = f3.selectbox("Тип", [None, *QTYPES], format_func=lambda k: "Все" if k is None else QTYPES[k])
        fs = f4.selectbox("Статус AI", [None, "ok", "warning", "unchecked"],
                          format_func=lambda s: "Все" if s is None else f"{STATUS_ICON[s]} {s}")
        questions = db.list_questions(doc_id=doc["id"], topic_id=ft["id"] if ft else None,
                                      level=fl, qtype=fq, status=fs)
        st.caption(f"Показано вопросов: {len(questions)}")

        for q in questions:
            mark = "🧑‍🏫" if q["manual_reviewed"] else ""
            title = (f"{STATUS_ICON.get(q['quality_status'], '')} {mark} [Ур. {q['level']}] "
                     f"{'Тест' if q['qtype'] == 'mcq' else 'Открытый'} · {q['question'][:110]}")
            with st.expander(title):
                st.caption(f"Тема: {q['topic_title']} · источник: стр./раздел {q['source_page'] or '—'}")
                if q["quality_comment"]:
                    st.warning(f"AI-проверка: {q['quality_comment']}")
                with st.form(f"form_{q['id']}"):
                    txt = st.text_area("Вопрос", q["question"])
                    opts, ci = None, None
                    if q["qtype"] == "mcq":
                        opts = [st.text_input(f"Вариант {'ABCD'[i]}", q["options"][i]) for i in range(4)]
                        ci = st.radio("Правильный вариант", [0, 1, 2, 3], index=q["correct_index"],
                                      format_func=lambda i: "ABCD"[i], horizontal=True)
                        ans = opts[ci]
                    else:
                        ans = st.text_area("Правильный ответ", q["answer"])
                    expl = st.text_area("Объяснение", q["explanation"])
                    lvl = st.selectbox("Уровень", list(LEVELS), index=q["level"] - 1,
                                       format_func=lambda l: f"{l} — {LEVELS[l][0]}")
                    st.markdown("**Ручная проверка**")
                    r1, r2, r3 = st.columns([1, 2, 3])
                    rev = r1.checkbox("Проверено", bool(q["manual_reviewed"]))
                    verdict = r2.selectbox("Вердикт", VERDICTS,
                                           index=VERDICTS.index(q["manual_verdict"]) if q["manual_verdict"] in VERDICTS else 0)
                    cmt = r3.text_input("Комментарий / найденная ошибка", q["manual_comment"])
                    b1, b2, _ = st.columns([1, 1, 4])
                    save = b1.form_submit_button("💾 Сохранить")
                    delete = b2.form_submit_button("🗑 Удалить")
                if save:
                    db.update_question(q["id"], question=txt, answer=ans, explanation=expl, level=lvl,
                                       options=opts, correct_index=ci, manual_reviewed=int(rev),
                                       manual_verdict=verdict, manual_comment=cmt)
                    st.rerun()
                if delete:
                    db.delete_question(q["id"])
                    st.rerun()

# ---------------------------------------------------------------- 4. мини-тест
with tab_quiz:
    if not doc:
        st.info("Сначала загрузите документ.")
    else:
        pool = db.list_questions(doc_id=doc["id"], qtype="mcq")
        if len(pool) < 1:
            st.info("Нет тестовых вопросов. Сгенерируйте их на вкладке «Генерация».")
        else:
            k = st.slider("Количество вопросов в тесте", 1, min(20, len(pool)), min(5, len(pool)))
            if st.button("▶ Начать тест"):
                st.session_state["quiz"] = [q["id"] for q in random.sample(pool, k)]
                st.session_state["quiz_nonce"] = random.random()
            ids = st.session_state.get("quiz")
            if ids:
                by_id = {q["id"]: q for q in pool}
                quiz = [by_id[i] for i in ids if i in by_id]
                with st.form("quiz_form"):
                    answers = {}
                    for n_, q in enumerate(quiz, 1):
                        answers[q["id"]] = st.radio(
                            f"{n_}. {q['question']}", [0, 1, 2, 3], index=None,
                            format_func=lambda i, q=q: q["options"][i],
                            key=f"quiz_{st.session_state['quiz_nonce']}_{q['id']}",
                        )
                    done = st.form_submit_button("Завершить и проверить")
                if done:
                    score = sum(1 for q in quiz if answers[q["id"]] == q["correct_index"])
                    st.success(f"Результат: {score} из {len(quiz)}")
                    for q in quiz:
                        ok = answers[q["id"]] == q["correct_index"]
                        st.markdown(f"{'✅' if ok else '❌'} **{q['question']}**  \n"
                                    f"Правильный ответ: {q['answer']}  \n_{q['explanation']}_")

# ---------------------------------------------------------------- 5. экспорт и проверка
with tab_export:
    allq = db.list_questions()
    df = pd.DataFrame(allq)
    st.subheader("Требования к демонстрации")
    n_docs = len(db.list_documents())
    reviewed = [q for q in allq if q["manual_reviewed"]]
    per_level = {l: sum(1 for q in allq if q["level"] == l) for l in LEVELS}
    checks = [
        (f"Документов: {n_docs} (нужно ≥ 2)", n_docs >= 2),
        (f"Всего вопросов: {len(allq)} (нужно ≥ 25)", len(allq) >= 25),
        (f"Минимум 5 вопросов на каждом уровне: {per_level}", all(v >= 5 for v in per_level.values())),
        (f"Проверено вручную: {len(reviewed)} (нужно ≥ 15)", len(reviewed) >= 15),
    ]
    for label, ok in checks:
        st.write(("✅ " if ok else "⬜ ") + label)

    if df.empty:
        st.info("Вопросов пока нет.")
    else:
        flat = df.copy()
        flat["options"] = flat["options"].apply(
            lambda o: " | ".join(f"{'ABCD'[i]}) {t}" for i, t in enumerate(o)) if isinstance(o, list) else "")
        cols = ["doc_name", "topic_title", "level", "qtype", "question", "options", "answer", "explanation",
                "source_page", "quality_status", "quality_comment", "manual_reviewed", "manual_verdict", "manual_comment"]
        st.subheader("Статистика")
        st.dataframe(pd.crosstab(df["level"], df["qtype"], margins=True), use_container_width=True)
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇ Все вопросы (CSV)", flat[cols].to_csv(index=False).encode("utf-8-sig"),
                           "questions.csv", "text/csv")
        e2.download_button("⬇ Все вопросы (JSON)", json.dumps(allq, ensure_ascii=False, indent=2),
                           "questions.json", "application/json")
        if reviewed:
            rv = flat[flat["manual_reviewed"] == 1][cols]
            e3.download_button("⬇ Отчёт ручной проверки (CSV)", rv.to_csv(index=False).encode("utf-8-sig"),
                               "manual_review.csv", "text/csv")
            st.subheader("Итоги ручной проверки")
            st.dataframe(rv["manual_verdict"].replace("", "—").value_counts().rename("кол-во"))
