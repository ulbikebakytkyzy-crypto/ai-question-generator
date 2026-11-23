"""Хранение данных в SQLite."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "data/app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, n_pages INTEGER, uploaded_at TEXT);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER, idx INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS topics(
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  title TEXT, description TEXT, keywords TEXT);
CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  topic_id INTEGER, topic_title TEXT,
  level INTEGER, qtype TEXT, question TEXT, options TEXT, correct_index INTEGER,
  answer TEXT, explanation TEXT, source_page INTEGER,
  quality_status TEXT DEFAULT 'unchecked', quality_comment TEXT DEFAULT '',
  manual_reviewed INTEGER DEFAULT 0, manual_verdict TEXT DEFAULT '', manual_comment TEXT DEFAULT '',
  created_at TEXT, model TEXT DEFAULT '');
"""

# колонки, добавленные позже (миграция старых баз)
MIGRATIONS = [("questions", "model", "TEXT DEFAULT ''")]

EDITABLE = {
    "level", "question", "options", "correct_index", "answer", "explanation",
    "manual_reviewed", "manual_verdict", "manual_comment", "quality_status", "quality_comment",
}


@contextmanager
def conn():
    d = os.path.dirname(DB_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)
        for table, col, decl in MIGRATIONS:
            cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ---------- документы ----------
def add_document(name, n_pages, chunks):
    with conn() as c:
        cur = c.execute("INSERT INTO documents(name, n_pages, uploaded_at) VALUES(?,?,?)", (name, n_pages, _now()))
        doc_id = cur.lastrowid
        c.executemany(
            "INSERT INTO chunks(doc_id, page, idx, text) VALUES(?,?,?,?)",
            [(doc_id, ch["page"], ch["idx"], ch["text"]) for ch in chunks],
        )
    return doc_id


def list_documents():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM documents ORDER BY id DESC")]


def delete_document(doc_id):
    with conn() as c:
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def get_chunks(doc_id):
    with conn() as c:
        rows = c.execute("SELECT page, idx, text FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))
        return [dict(r) for r in rows]


# ---------- темы ----------
def save_topics(doc_id, topics):
    with conn() as c:
        c.execute("DELETE FROM topics WHERE doc_id=?", (doc_id,))
        c.executemany(
            "INSERT INTO topics(doc_id, title, description, keywords) VALUES(?,?,?,?)",
            [(doc_id, t["title"], t.get("description", ""), json.dumps(t.get("keywords", []), ensure_ascii=False))
             for t in topics],
        )


def get_topics(doc_id):
    with conn() as c:
        out = []
        for r in c.execute("SELECT * FROM topics WHERE doc_id=? ORDER BY id", (doc_id,)):
            d = dict(r)
            d["keywords"] = json.loads(d["keywords"] or "[]")
            out.append(d)
        return out


# ---------- вопросы ----------
def add_questions(doc_id, topic, questions):
    with conn() as c:
        for q in questions:
            c.execute(
                """INSERT INTO questions(doc_id, topic_id, topic_title, level, qtype, question, options,
                   correct_index, answer, explanation, source_page, quality_status, quality_comment, created_at, model)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc_id, topic["id"], topic["title"], q["level"], q["qtype"], q["question"],
                    json.dumps(q["options"], ensure_ascii=False) if q.get("options") else None,
                    q.get("correct_index"), q["answer"], q.get("explanation", ""), q.get("source_page"),
                    q.get("quality_status", "unchecked"), q.get("quality_comment", ""), _now(),
                    q.get("model", ""),
                ),
            )


def _row_to_q(r):
    d = dict(r)
    d["options"] = json.loads(d["options"]) if d["options"] else None
    return d


def list_questions(doc_id=None, topic_id=None, level=None, qtype=None, status=None, reviewed=None):
    sql, args = "SELECT q.*, d.name AS doc_name FROM questions q JOIN documents d ON d.id=q.doc_id WHERE 1=1", []
    for col, val in (("q.doc_id", doc_id), ("q.topic_id", topic_id), ("q.level", level),
                     ("q.qtype", qtype), ("q.quality_status", status), ("q.manual_reviewed", reviewed)):
        if val is not None:
            sql += f" AND {col}=?"
            args.append(val)
    sql += " ORDER BY q.level, q.id"
    with conn() as c:
        return [_row_to_q(r) for r in c.execute(sql, args)]


def update_question(qid, **fields):
    fields = {k: v for k, v in fields.items() if k in EDITABLE}
    if "options" in fields and fields["options"] is not None:
        fields["options"] = json.dumps(fields["options"], ensure_ascii=False)
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE questions SET {sets} WHERE id=?", [*fields.values(), qid])


def delete_question(qid):
    with conn() as c:
        c.execute("DELETE FROM questions WHERE id=?", (qid,))


# ---------- резервная копия (на Streamlit Cloud база стирается при перезапуске) ----------
TABLES = ("documents", "chunks", "topics", "questions")


def export_all():
    """Вся база в виде dict — для скачивания JSON-бэкапа."""
    with conn() as c:
        return {t: [dict(r) for r in c.execute(f"SELECT * FROM {t} ORDER BY id")] for t in TABLES}


def import_all(data, replace=True):
    """Восстанавливает базу из JSON-бэкапа. Возвращает число восстановленных вопросов."""
    if not isinstance(data, dict) or "documents" not in data:
        raise ValueError("Это не файл резервной копии приложения.")
    with conn() as c:
        if replace:
            for t in reversed(TABLES):
                c.execute(f"DELETE FROM {t}")
        id_map = {}
        for d in data.get("documents", []):
            cur = c.execute("INSERT INTO documents(name, n_pages, uploaded_at) VALUES(?,?,?)",
                            (d["name"], d.get("n_pages"), d.get("uploaded_at")))
            id_map[d["id"]] = cur.lastrowid
        for ch in data.get("chunks", []):
            if ch["doc_id"] in id_map:
                c.execute("INSERT INTO chunks(doc_id, page, idx, text) VALUES(?,?,?,?)",
                          (id_map[ch["doc_id"]], ch.get("page"), ch.get("idx"), ch.get("text")))
        topic_map = {}
        for t in data.get("topics", []):
            if t["doc_id"] in id_map:
                cur = c.execute("INSERT INTO topics(doc_id, title, description, keywords) VALUES(?,?,?,?)",
                                (id_map[t["doc_id"]], t.get("title"), t.get("description"), t.get("keywords")))
                topic_map[t["id"]] = cur.lastrowid
        cols = ["topic_title", "level", "qtype", "question", "options", "correct_index", "answer", "explanation",
                "source_page", "quality_status", "quality_comment", "manual_reviewed", "manual_verdict",
                "manual_comment", "created_at", "model"]
        n = 0
        for q in data.get("questions", []):
            if q["doc_id"] not in id_map:
                continue
            opts = q.get("options")
            if isinstance(opts, list):
                q = {**q, "options": json.dumps(opts, ensure_ascii=False)}
            vals = [q.get(k) for k in cols]
            c.execute(f"INSERT INTO questions(doc_id, topic_id, {', '.join(cols)}) "
                      f"VALUES(?,?,{','.join('?' * len(cols))})",
                      [id_map[q["doc_id"]], topic_map.get(q.get("topic_id")), *vals])
            n += 1
    return n


def add_topic(doc_id, title, description="", keywords=None):
    with conn() as c:
        c.execute("INSERT INTO topics(doc_id, title, description, keywords) VALUES(?,?,?,?)",
                  (doc_id, title, description, json.dumps(keywords or [], ensure_ascii=False)))
