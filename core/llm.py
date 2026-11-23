"""Тонкая обёртка над OpenAI API: возвращает JSON-ответ модели."""
import json
import os
import time

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class LLMError(Exception):
    pass


def _client(api_key=None):
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("Не указан OpenAI API key (поле в боковой панели или файл .env).")
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise LLMError("Не установлен пакет openai: pip install -r requirements.txt") from e
    return OpenAI(api_key=key)


def chat_json(system, user, *, api_key=None, model=None, temperature=0.4, retries=3):
    """Отправляет запрос и гарантированно возвращает dict (или бросает LLMError)."""
    client = _client(api_key)
    kwargs = dict(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    last = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            data = json.loads(resp.choices[0].message.content)
            if not isinstance(data, dict):
                raise ValueError("ответ не является JSON-объектом")
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            # некоторые модели не принимают нестандартную temperature
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            if "401" in msg or "invalid_api_key" in msg or "incorrect api key" in msg:
                raise LLMError("Неверный OpenAI API key.") from e
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"Ошибка обращения к модели: {last}")
