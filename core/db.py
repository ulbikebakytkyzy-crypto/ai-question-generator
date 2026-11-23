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
  created_at TEXT);
"""

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
                   correct_index, answer, explanation, source_page, quality_status, quality_comment, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc_id, topic["id"], topic["title"], q["level"], q["qtype"], q["question"],
                    json.dumps(q["options"], ensure_ascii=False) if q.get("options") else None,
                    q.get("correct_index"), q["answer"], q.get("explanation", ""), q.get("source_page"),
                    q.get("quality_status", "unchecked"), q.get("quality_comment", ""), _now(),
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
