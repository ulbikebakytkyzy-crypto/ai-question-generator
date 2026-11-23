"""AI-генератор вопросов по учебным материалам (Streamlit)."""
import json
import os
import random
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI-генератор вопросов", page_icon="📚", layout="wide")

# ключ из Streamlit Secrets (на сервере) → переменная окружения; в браузер он не передаётся
try:
    for _k in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # noqa: BLE001 — secrets.toml может отсутствовать
    pass

from core import db, extract, grader, retrieval, validator  # noqa: E402
from core import topics as topics_mod  # noqa: E402
from core.export_pdf import questions_pdf  # noqa: E402
from core.generator import LANGS, LEVELS, QTYPES, generate  # noqa: E402
from core.llm import DEFAULT_MODEL, MODELS, LLMError  # noqa: E402

db.init()

STATUS_ICON = {"ok": "✅", "warning": "⚠️", "unchecked": "⏺"}
STATUS_NAME = {"ok": "без замечаний", "warning": "есть замечания", "unchecked": "не проверен"}
VERDICTS = ["", "ок", "ошибка в ответе", "не по теме", "не тот уровень", "повтор", "другое"]
QT_SHORT = {"mcq": "Тест", "open": "Открытый"}


def level_label(l):
    return f"{l} — {LEVELS[l][0]}"


def model_picker(label, key, default=DEFAULT_MODEL):
    opts = list(dict.fromkeys([default, *MODELS])) + ["другая…"]
    choice = st.selectbox(label, opts, key=key)
    if choice == "другая…":
        return st.text_input("Название модели", key=f"{key}_custom").strip() or default
    return choice


# ---------------------------------------------------------------- боковая панель
st.sidebar.title("📚 AI-генератор вопросов")
server_key = bool(os.getenv("OPENAI_API_KEY"))
user_key = st.sidebar.text_input(
    "OpenAI API key", type="password",
    placeholder="ключ уже подключён на сервере" if server_key else "sk-...",
    help="Ключ используется только в этой сессии и нигде не сохраняется.",
)
api_key = user_key.strip() or None
if not (api_key or server_key):
    st.sidebar.warning("Введите OpenAI API key, чтобы работали темы и генерация.")
with st.sidebar:
    model = model_picker("Модель", "model_main")
lang_label = st.sidebar.selectbox("Язык вопросов", list(LANGS.keys()),
                                  help="Материал может быть на любом языке — вопросы будут на выбранном.")
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

with st.sidebar.expander("💾 Резервная копия базы"):
    st.caption("На бесплатном хостинге база может обнулиться при перезапуске. "
               "Скачивайте копию после работы и восстанавливайте её при необходимости.")
    st.download_button("⬇ Скачать копию (JSON)", json.dumps(db.export_all(), ensure_ascii=False),
                       "backup.json", "application/json", width="stretch")
    bk = st.file_uploader("Восстановить из копии", type=["json"], key="backup_up")
    if bk and st.button("Восстановить (заменит текущие данные)", width="stretch"):
        try:
            n = db.import_all(json.loads(bk.getvalue().decode("utf-8")))
            st.session_state.pop("doc_id", None)
            st.success(f"Восстановлено вопросов: {n}")
            st.rerun()
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            st.error(f"Не удалось восстановить: {e}")

st.markdown("""
<style>
[data-testid="stSidebar"] {background: linear-gradient(180deg, #ede4ff 0%, #ffe6f2 100%);}
.hero {background: linear-gradient(120deg, #7c3aed 0%, #db2777 55%, #f59e0b 100%); color: white;
       padding: 22px 28px; border-radius: 18px; margin-bottom: 14px; box-shadow: 0 8px 24px rgba(124,58,237,.25);}
.hero h1 {color: white !important; margin: 0; font-size: 2.1rem;}
.hero p {margin: 6px 0 0; opacity: .95;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #efe9ff; border-radius: 12px 12px 0 0; padding: 8px 14px;}
.stTabs [aria-selected="true"] {background: linear-gradient(120deg, #7c3aed, #db2777); color: white !important;}
.stTabs [aria-selected="true"] p {color: white !important;}
[data-testid="stMetric"] {background: white; border-radius: 14px; padding: 12px 16px;
                          border-left: 6px solid #db2777; box-shadow: 0 2px 10px rgba(0,0,0,.05);}
[data-testid="stExpander"] details {background: white; border-radius: 12px;}
.stButton button[kind="primary"] {background: linear-gradient(120deg, #7c3aed, #db2777); border: 0;}
.stDownloadButton button {border: 2px solid #7c3aed;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📚 AI-генератор вопросов по учебным материалам</h1>'
            '<p>Загрузка документа → анализ текста → темы → вопросы 5 уровней → правильные ответы → проверка</p></div>',
            unsafe_allow_html=True)

tab_doc, tab_gen, tab_q, tab_quiz, tab_cmp, tab_export = st.tabs(
    ["1. Документ и темы", "2. Генерация", "3. Вопросы", "4. Мини-тест", "5. Сравнение моделей",
     "6. Экспорт и проверка"]
)

# ---------------------------------------------------------------- 1. документ и темы
with tab_doc:
    ups = st.file_uploader("Загрузите учебный материал (PDF с текстом или TXT). Можно несколько файлов.",
                           type=["pdf", "txt"], accept_multiple_files=True)
    if ups and st.button("Обработать документ" + ("ы" if len(ups) > 1 else ""), type="primary"):
        for up in ups:
            try:
                with st.spinner(f"{up.name}: извлекаю текст..."):
                    pages = extract.read_document(up.name, up.getvalue())
                    chunks = extract.make_chunks(pages)
                    new_id = db.add_document(up.name, len(pages), chunks)
                st.session_state["doc_id"] = new_id
                with st.spinner(f"{up.name}: AI определяет темы..."):
                    db.save_topics(new_id, topics_mod.detect_topics(chunks, language, api_key, model))
                st.success(f"{up.name}: готово")
            except (ValueError, RuntimeError) as e:
                st.error(f"{up.name}: {e}")
            except LLMError as e:
                st.error(f"{up.name}: текст сохранён, но темы не определены — {e}")
        st.rerun()

    if doc:
        st.subheader(f"📄 {doc['name']}")
        chunks = db.get_chunks(doc["id"])
        n_q = len(db.list_questions(doc_id=doc["id"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Страниц / разделов", doc["n_pages"])
        m2.metric("Фрагментов текста", len(chunks))
        m3.metric("Вопросов по документу", n_q)
        with st.expander("Превью извлечённого текста"):
            st.text("\n\n".join(f"[стр. {c['page']}] {c['text']}" for c in chunks[:3])[:3000])
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
            st.info("Темы не найдены — нажмите «Определить заново» или добавьте тему вручную.")
        for i, t in enumerate(topics, 1):
            st.markdown(f"**{i}. {t['title']}** — {t['description']}")
            st.caption("Ключевые слова: " + ", ".join(t["keywords"]))
        with st.expander("➕ Добавить тему вручную"):
            with st.form("add_topic", clear_on_submit=True):
                tt = st.text_input("Название темы")
                td = st.text_input("Краткое описание")
                tk = st.text_input("Ключевые слова через запятую (как в тексте — по ним ищутся фрагменты)")
                if st.form_submit_button("Добавить") and tt.strip():
                    db.add_topic(doc["id"], tt.strip(), td.strip(), [k.strip() for k in tk.split(",") if k.strip()])
                    st.rerun()
    else:
        st.info("Загрузите документ, чтобы начать. Для проверки можно взять файлы из папки `samples/` репозитория.")

    with st.expander("ℹ️ Уровни сложности"):
        st.table(pd.DataFrame([{"Уровень": l, "Название": n, "Что проверяет": d} for l, (n, d) in LEVELS.items()])
                 .set_index("Уровень"))

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
            g1, g2 = st.columns(2)
            qtypes = g1.multiselect("Тип вопросов", list(QTYPES), default=list(QTYPES), format_func=QTYPES.get)
            levels = g2.multiselect("Уровни сложности", list(LEVELS), default=list(LEVELS), format_func=level_label)
            n = st.slider("Количество вопросов (на каждый уровень и каждый тип)", 1, 10, 2)
            check = st.checkbox("Проверять качество дополнительным запросом к AI", value=True)
            st.caption(f"Будет создано до **{n * len(qtypes) * len(levels)}** вопросов "
                       f"({len(levels)} ур. × {len(qtypes)} тип × {n}). Модель: `{model}`.")

            if st.button("✨ Сгенерировать", type="primary", disabled=not (qtypes and levels)):
                chunks = db.get_chunks(doc["id"])
                terms = [topic["title"]] + topic["keywords"]
                existing = db.list_questions(doc_id=doc["id"], topic_id=topic["id"])
                combos = [(l, t) for l in sorted(levels) for t in qtypes]
                bar = st.progress(0.0, text="Начинаю...")
                added = dups_total = warn = 0
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
                    for q in qs:
                        q["model"] = model
                    warn += sum(1 for q in qs if q.get("quality_status") == "warning")
                    db.add_questions(doc["id"], topic, qs)
                    existing += qs
                    added += len(qs)
                bar.progress(1.0, text="Готово")
                st.success(f"Добавлено вопросов: {added} (из них с замечаниями AI-проверки: {warn}). "
                           f"Отброшено повторов: {dups_total}. Смотрите вкладку «3. Вопросы».")

            st.divider()
            st.markdown("**Сколько вопросов уже есть по этому документу**")
            dq = db.list_questions(doc_id=doc["id"])
            if dq:
                ddf = pd.DataFrame(dq)
                tab = pd.crosstab(ddf["topic_title"], ddf["level"]).reindex(columns=list(LEVELS), fill_value=0)
                tab.columns = [f"Ур. {c}" for c in tab.columns]
                tab.index.name = "Тема"
                st.dataframe(tab, width="stretch")
            else:
                st.caption("Пока нет.")

# ---------------------------------------------------------------- 3. вопросы
with tab_q:
    if not doc:
        st.info("Сначала загрузите документ.")
    else:
        topics = db.get_topics(doc["id"])
        mode = st.radio("Режим", ["👀 Просмотр", "✏️ Редактирование и проверка"], horizontal=True)
        f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 3])
        ft = f1.selectbox("Тема", [None] + topics, format_func=lambda t: "Все" if t is None else t["title"])
        fl = f2.selectbox("Уровень", [None, *LEVELS], format_func=lambda l: "Все" if l is None else level_label(l))
        fq = f3.selectbox("Тип", [None, *QTYPES], format_func=lambda k: "Все" if k is None else QTYPES[k])
        fs = f4.selectbox("Статус AI", [None, "ok", "warning", "unchecked"],
                          format_func=lambda s: "Все" if s is None else f"{STATUS_ICON[s]} {STATUS_NAME[s]}")
        search = f5.text_input("Поиск по тексту").strip().lower()
        questions = db.list_questions(doc_id=doc["id"], topic_id=ft["id"] if ft else None,
                                      level=fl, qtype=fq, status=fs)
        if search:
            questions = [q for q in questions if search in (q["question"] + " " + q["answer"]).lower()]
        st.caption(f"Показано вопросов: {len(questions)} · ✅ без замечаний · ⚠️ есть замечания AI-проверки · "
                   "⏺ не проверен · 🧑‍🏫 проверен вручную")

        for q in questions:
            mark = "🧑‍🏫" if q["manual_reviewed"] else ""
            head = (f"{STATUS_ICON.get(q['quality_status'], '')} {mark} [Ур. {q['level']}] "
                    f"{QT_SHORT[q['qtype']]} · {q['question'][:110]}")
            meta = (f"Тема: {q['topic_title']} · уровень {level_label(q['level'])} · "
                    f"источник: стр./раздел {q['source_page'] or '—'}" + (f" · модель: {q['model']}" if q.get("model") else ""))

            if mode.startswith("👀"):
                with st.container(border=True):
                    st.markdown(f"**{head.split(' · ', 1)[0]}** {q['question']}")
                    st.caption(meta)
                    if q["qtype"] == "mcq":
                        st.markdown("\n".join(f"- {'ABCD'[i]}) {o}" for i, o in enumerate(q["options"])))
                    with st.expander("Показать ответ"):
                        if q["qtype"] == "mcq":
                            st.markdown(f"**Правильный вариант: {'ABCD'[q['correct_index']]}) {q['answer']}**")
                        else:
                            st.markdown(f"**Ответ:** {q['answer']}")
                        if q["explanation"]:
                            st.markdown(f"_Объяснение:_ {q['explanation']}")
                        if q["quality_comment"]:
                            st.warning(f"AI-проверка: {q['quality_comment']}")
                continue

            with st.expander(head):
                st.caption(meta)
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
                    lvl = st.selectbox("Уровень", list(LEVELS), index=q["level"] - 1, format_func=level_label)
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
                    if q["qtype"] == "mcq" and len({o.strip().lower() for o in opts}) < 4:
                        st.error("Варианты ответа не должны повторяться.")
                    else:
                        db.update_question(q["id"], question=txt, answer=ans, explanation=expl, level=lvl,
                                           options=opts, correct_index=ci, manual_reviewed=int(rev or bool(verdict)),
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
        st.caption("Тестовые вопросы проверяются автоматически, открытые — оценивает AI (0–2 балла) "
                   "по сравнению с эталонным ответом.")
        z1, z2, z3 = st.columns(3)
        quiz_types = z1.multiselect("Типы вопросов", list(QTYPES), default=["mcq"], format_func=QTYPES.get, key="qz_t")
        quiz_levels = z2.multiselect("Уровни", list(LEVELS), default=list(LEVELS), format_func=level_label, key="qz_l")
        pool = [q for q in db.list_questions(doc_id=doc["id"])
                if q["qtype"] in quiz_types and q["level"] in quiz_levels]
        if not pool:
            st.info("Нет подходящих вопросов. Сгенерируйте их на вкладке «Генерация».")
        else:
            k = z3.number_input("Количество вопросов", 1, min(20, len(pool)), min(5, len(pool)))
            if st.button("▶ Начать тест"):
                st.session_state["quiz"] = [q["id"] for q in random.sample(pool, int(k))]
                st.session_state["quiz_nonce"] = random.random()
                st.session_state.pop("quiz_result", None)
            ids = st.session_state.get("quiz")
            if ids:
                by_id = {q["id"]: q for q in db.list_questions(doc_id=doc["id"])}
                quiz = [by_id[i] for i in ids if i in by_id]
                nonce = st.session_state["quiz_nonce"]
                with st.form("quiz_form"):
                    answers = {}
                    for n_, q in enumerate(quiz, 1):
                        label = f"{n_}. [Ур. {q['level']}] {q['question']}"
                        if q["qtype"] == "mcq":
                            answers[q["id"]] = st.radio(label, [0, 1, 2, 3], index=None,
                                                        format_func=lambda i, q=q: q["options"][i],
                                                        key=f"quiz_{nonce}_{q['id']}")
                        else:
                            answers[q["id"]] = st.text_area(label, key=f"quiz_{nonce}_{q['id']}")
                    done = st.form_submit_button("Завершить и проверить")
                if done:
                    open_q = [q for q in quiz if q["qtype"] == "open"]
                    grades = {}
                    if open_q:
                        try:
                            with st.spinner("AI проверяет открытые ответы..."):
                                res = grader.grade_open_answers(
                                    [{"question": q["question"], "reference": q["answer"],
                                      "user_answer": answers[q["id"]] or ""} for q in open_q],
                                    language, api_key, model)
                            grades = {q["id"]: r for q, r in zip(open_q, res)}
                        except LLMError as e:
                            st.error(f"Открытые ответы не проверены: {e}")
                    st.session_state["quiz_result"] = {"answers": answers, "grades": grades}
                result = st.session_state.get("quiz_result")
                if result:
                    answers, grades = result["answers"], result["grades"]
                    total = 0.0
                    lines = []
                    for q in quiz:
                        if q["qtype"] == "mcq":
                            pts = 1.0 if answers.get(q["id"]) == q["correct_index"] else 0.0
                            icon = "✅" if pts else "❌"
                            extra = ""
                        else:
                            g = grades.get(q["id"])
                            pts = g["score"] / 2 if g else 0.0
                            icon = {1.0: "✅", 0.5: "🟡"}.get(pts, "❌") if g else "⏺"
                            extra = (f"  \nВаш ответ: {answers.get(q['id']) or '—'}  \n"
                                     f"Оценка AI: {g['score']}/2 — {g['feedback']}" if g else "")
                        total += pts
                        lines.append(f"{icon} **{q['question']}**  \nПравильный ответ: {q['answer']}{extra}  \n"
                                     f"_{q['explanation']}_")
                    st.success(f"Результат: {total:g} из {len(quiz)} ({100 * total / len(quiz):.0f}%)")
                    for ln in lines:
                        st.markdown(ln)

# ---------------------------------------------------------------- 5. сравнение моделей
with tab_cmp:
    st.markdown("Одна и та же тема и **одни и те же фрагменты текста** отправляются в две модели. "
                "Затем одна модель-судья проверяет все вопросы по одинаковым критериям.")
    if not doc:
        st.info("Сначала загрузите документ.")
    elif not db.get_topics(doc["id"]):
        st.warning("У документа нет тем.")
    else:
        topics = db.get_topics(doc["id"])
        c_topic = st.selectbox("Тема", topics, format_func=lambda t: t["title"], key="cmp_topic")
        a1, a2, a3 = st.columns(3)
        with a1:
            model_a = model_picker("Модель A", "cmp_a", MODELS[0])
        with a2:
            model_b = model_picker("Модель B", "cmp_b", MODELS[1])
        with a3:
            judge = model_picker("Модель-судья", "cmp_j", MODELS[2])
        b1, b2, b3 = st.columns(3)
        c_levels = b1.multiselect("Уровни", list(LEVELS), default=[1, 3, 5], format_func=level_label, key="cmp_l")
        c_type = b2.selectbox("Тип", list(QTYPES), format_func=QTYPES.get, key="cmp_qt")
        c_n = b3.slider("Вопросов на уровень", 1, 5, 2, key="cmp_n")
        c_save = st.checkbox("Сохранить созданные вопросы в базу (с пометкой модели)", value=False)
        if st.button("⚖️ Сравнить", type="primary", disabled=not c_levels or model_a == model_b):
            chunks = db.get_chunks(doc["id"])
            terms = [c_topic["title"]] + c_topic["keywords"]
            stats = {m: {"Вопросов": 0, "Без замечаний": 0, "С замечаниями": 0, "Повторов": 0,
                         "Время, с": 0.0, "Ошибок API": 0, "_len": 0} for m in (model_a, model_b)}
            samples = []
            bar = st.progress(0.0)
            steps = len(c_levels) * 2
            step = 0
            for lvl in sorted(c_levels):
                ctx = retrieval.select_chunks(chunks, terms, seed=lvl)  # одинаковый контекст для обеих моделей
                for m in (model_a, model_b):
                    step += 1
                    bar.progress(step / steps, text=f"Уровень {lvl} · {m}")
                    t0 = time.time()
                    try:
                        qs = generate(topic=c_topic, level=lvl, qtype=c_type, n=c_n, chunks=ctx,
                                      language=language, api_key=api_key, model=m)
                    except LLMError as e:
                        stats[m]["Ошибок API"] += 1
                        st.error(f"{m}: {e}")
                        continue
                    stats[m]["Время, с"] += time.time() - t0
                    qs, dropped = validator.dedupe(qs, [])
                    validator.ai_check(qs, c_topic, ctx, language, api_key, judge)
                    s = stats[m]
                    s["Вопросов"] += len(qs)
                    s["Повторов"] += len(dropped)
                    s["Без замечаний"] += sum(q["quality_status"] == "ok" for q in qs)
                    s["С замечаниями"] += sum(q["quality_status"] == "warning" for q in qs)
                    s["_len"] += sum(len(q["question"]) for q in qs)
                    for q in qs:
                        q["model"] = m
                        samples.append({"Модель": m, "Уровень": lvl, "Вопрос": q["question"], "Ответ": q["answer"],
                                        "AI-проверка": STATUS_ICON[q["quality_status"]] + " " + q["quality_comment"]})
                    if c_save:
                        db.add_questions(doc["id"], c_topic, qs)
            bar.empty()
            rows = []
            for m, s in stats.items():
                rows.append({"Модель": m, "Вопросов": s["Вопросов"], "Без замечаний": s["Без замечаний"],
                             "С замечаниями": s["С замечаниями"],
                             "Доля без замечаний": f"{100 * s['Без замечаний'] / s['Вопросов']:.0f}%" if s["Вопросов"] else "—",
                             "Повторов": s["Повторов"], "Ср. длина вопроса": round(s["_len"] / s["Вопросов"]) if s["Вопросов"] else 0,
                             "Время, с": round(s["Время, с"], 1), "Ошибок API": s["Ошибок API"]})
            st.session_state["cmp_result"] = {"summary": rows, "samples": samples, "judge": judge}
        res = st.session_state.get("cmp_result")
        if res:
            st.subheader("Итоги сравнения")
            st.dataframe(pd.DataFrame(res["summary"]).set_index("Модель"), width="stretch")
            st.caption(f"Судья: `{res['judge']}`. «Без замечаний» — вопрос по теме, ответ верен и подтверждается "
                       "текстом, уровень соответствует, у теста один правильный вариант.")
            sdf = pd.DataFrame(res["samples"])
            if not sdf.empty:
                st.dataframe(sdf, width="stretch", hide_index=True)
                st.download_button("⬇ Результаты сравнения (CSV)", sdf.to_csv(index=False).encode("utf-8-sig"),
                                   "model_comparison.csv", "text/csv")

# ---------------------------------------------------------------- 6. экспорт и проверка
with tab_export:
    allq = db.list_questions()
    df = pd.DataFrame(allq)
    st.subheader("Минимальные требования к демонстрации")
    n_docs = len(db.list_documents())
    reviewed = [q for q in allq if q["manual_reviewed"]]
    per_level = {l: sum(1 for q in allq if q["level"] == l) for l in LEVELS}
    per_type = {k: sum(1 for q in allq if q["qtype"] == k) for k in QTYPES}
    checks = [
        (f"Учебных материалов: {n_docs} (нужно ≥ 2)", n_docs >= 2),
        (f"Всего вопросов: {len(allq)} (нужно ≥ 25)", len(allq) >= 25),
        ("Минимум 5 вопросов на каждом уровне: " + ", ".join(f"ур. {l} — {v}" for l, v in per_level.items()),
         all(v >= 5 for v in per_level.values())),
        (f"Есть оба типа вопросов: тестовых {per_type['mcq']}, открытых {per_type['open']}",
         all(per_type.values())),
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
                "source_page", "model", "quality_status", "quality_comment", "manual_reviewed", "manual_verdict",
                "manual_comment"]
        st.subheader("Статистика")
        s1, s2 = st.columns(2)
        ct = pd.crosstab(df["level"], df["qtype"].map(QT_SHORT), margins=True, margins_name="Всего")
        ct.index = ct.index.map(str)
        ct.index.name = "Уровень"
        s1.dataframe(ct, width="stretch")
        qs_ = df["quality_status"].map(lambda s: f"{STATUS_ICON.get(s, '')} {STATUS_NAME.get(s, s)}").value_counts()
        s2.dataframe(qs_.rename("кол-во"), width="stretch")

        st.subheader("Скачать")
        e1, e2, e3, e4 = st.columns(4)
        e1.download_button("⬇ Все вопросы (CSV)", flat[cols].to_csv(index=False).encode("utf-8-sig"),
                           "questions.csv", "text/csv", width="stretch")
        e2.download_button("⬇ Все вопросы (JSON)", json.dumps(allq, ensure_ascii=False, indent=2),
                           "questions.json", "application/json", width="stretch")
        try:
            e3.download_button("⬇ PDF с ответами", questions_pdf(allq, "Вопросы с ответами"),
                               "questions_with_answers.pdf", "application/pdf", width="stretch")
            e4.download_button("⬇ PDF без ответов (бланк)", questions_pdf(allq, "Вопросы", with_answers=False),
                               "questions.pdf", "application/pdf", width="stretch")
        except Exception as e:  # noqa: BLE001
            e3.error(f"PDF не создан: {e}")

        st.subheader("Ручная проверка")
        if reviewed:
            rv = flat[flat["manual_reviewed"] == 1][cols]
            st.download_button("⬇ Отчёт ручной проверки (CSV)", rv.to_csv(index=False).encode("utf-8-sig"),
                               "manual_review.csv", "text/csv")
            v1, v2 = st.columns([1, 2])
            v1.dataframe(rv["manual_verdict"].replace("", "—").value_counts().rename("кол-во"))
            errs = rv[rv["manual_verdict"].isin(VERDICTS[2:])][["level", "question", "manual_verdict", "manual_comment"]]
            if not errs.empty:
                v2.markdown("**Найденные ошибки** (для презентации)")
                v2.dataframe(errs, hide_index=True, width="stretch")
            agree = rv[rv["quality_status"].isin(["ok", "warning"]) & (rv["manual_verdict"] != "")]
            if not agree.empty:
                same = ((agree["quality_status"] == "ok") == (agree["manual_verdict"] == "ок")).mean()
                st.caption(f"Совпадение AI-проверки с ручной: {100 * same:.0f}% (на {len(agree)} вопросах).")
        else:
            st.info("Отметьте вопросы на вкладке «3. Вопросы» → режим «Редактирование и проверка».")
