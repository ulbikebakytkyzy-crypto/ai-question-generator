"""Чтение PDF/TXT и разбиение текста на фрагменты (chunks)."""
import re

MAX_CHARS_PER_PSEUDO_PAGE = 3000


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp866"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _split_text_to_sections(text: str):
    """TXT не имеет страниц — режем на «разделы» по ~3000 символов."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > MAX_CHARS_PER_PSEUDO_PAGE:
            sections.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        sections.append(buf)
    return [(i + 1, s) for i, s in enumerate(sections)]


def read_document(name: str, data: bytes):
    """Возвращает список (номер_страницы, текст)."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        try:
            import pymupdf as fitz  # PyMuPDF
        except ImportError:
            try:
                import fitz  # старые версии PyMuPDF
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("Не установлен PyMuPDF: pip install pymupdf") from e
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            t = page.get_text().strip()
            if t:
                pages.append((i + 1, t))
        doc.close()
        if sum(len(t) for _, t in pages) < 200:
            raise ValueError(
                "В PDF почти нет текста — вероятно, это скан. Сканированные документы не поддерживаются."
            )
        return pages
    if lower.endswith(".txt"):
        text = _decode(data).strip()
        if len(text) < 200:
            raise ValueError("Текст слишком короткий (меньше 200 символов).")
        return _split_text_to_sections(text)
    raise ValueError("Поддерживаются только PDF и TXT.")


def _hard_split(p: str, limit: int):
    parts = []
    while len(p) > limit:
        cut = p.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(p[:cut].strip())
        p = p[cut:].strip()
    if p:
        parts.append(p)
    return parts


def make_chunks(pages, max_chars: int = 1800):
    """Список dict(page, idx, text) — фрагменты не длиннее max_chars."""
    chunks, idx = [], 0
    for page, text in pages:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        parts = []
        for p in paras:
            parts.extend(_hard_split(p, max_chars) if len(p) > max_chars else [p])
        buf = ""
        for p in parts:
            if buf and len(buf) + len(p) + 2 > max_chars:
                chunks.append({"page": page, "idx": idx, "text": buf})
                idx += 1
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            chunks.append({"page": page, "idx": idx, "text": buf})
            idx += 1
    return chunks
