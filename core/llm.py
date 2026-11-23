"""Обёртка над LLM API (OpenAI-совместимый протокол): OpenAI, Google Gemini, Groq.
Gemini и Groq имеют бесплатные тарифы — ключ без банковской карты."""
import json
import os
import re
import time

PROVIDERS = {
    "gemini": {
        "label": "Google Gemini (бесплатно)",
        "env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
        "key_url": "https://aistudio.google.com/apikey",
    },
    "groq": {
        "label": "Groq (бесплатно)",
        "env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"],
        "key_url": "https://console.groq.com/keys",
    },
    "openai": {
        "label": "OpenAI (платно)",
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        "key_url": "https://platform.openai.com/api-keys",
    },
}


def provider_for(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("gemini") or m.startswith("gemma"):
        return "gemini"
    if m.startswith(("gpt-4", "gpt-3", "gpt-5", "o1", "o3", "o4")):
        return "openai"
    return "groq"


def available_providers():
    """Провайдеры, для которых на сервере есть ключ (в порядке приоритета)."""
    return [p for p, cfg in PROVIDERS.items() if os.getenv(cfg["env"])]


def default_provider():
    av = available_providers()
    return av[0] if av else "gemini"


DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or PROVIDERS[default_provider()]["models"][0]
MODELS = [m for p in PROVIDERS.values() for m in p["models"]]


class LLMError(Exception):
    pass


def _client(api_key=None, model=None):
    prov = provider_for(model)
    cfg = PROVIDERS[prov]
    if isinstance(api_key, dict):
        api_key = api_key.get(prov)
    key = api_key or os.getenv(cfg["env"])
    if not key:
        raise LLMError(f"Нет ключа для {cfg['label']} (модель {model}). Введите ключ в боковой панели "
                       f"или добавьте {cfg['env']} в Secrets. Получить ключ: {cfg['key_url']}")
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise LLMError("Не установлен пакет openai: pip install -r requirements.txt") from e
    return OpenAI(api_key=key, base_url=cfg["base_url"]) if cfg["base_url"] else OpenAI(api_key=key)


def _parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _wait_seconds(msg: str) -> float:
    m = re.search(r"(?:try again in|retry in|retry after|retrydelay\"?:?\s*\"?)\s*([\d.]+)\s*(ms|s)?", msg)
    if m:
        v = float(m.group(1))
        return min(60.0, v / 1000 if m.group(2) == "ms" else v) + 1
    return 20.0


def chat_json(system, user, *, api_key=None, model=None, temperature=0.4, retries=6):
    """Отправляет запрос и гарантированно возвращает dict (или бросает LLMError)."""
    model = model or DEFAULT_MODEL
    client = _client(api_key, model)
    kwargs = dict(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system + " Respond with valid JSON only."},
            {"role": "user", "content": user},
        ],
    )
    last = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            data = _parse_json(resp.choices[0].message.content)
            if not isinstance(data, dict):
                raise ValueError("ответ не является JSON-объектом")
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            if "response_format" in msg and "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            if "401" in msg or "invalid_api_key" in msg or "incorrect api key" in msg or "api key not valid" in msg:
                raise LLMError(f"Неверный API-ключ для модели {model}.") from e
            if "no credits" in msg or "insufficient_quota" in msg:
                raise LLMError(f"На аккаунте закончились деньги/кредиты ({model}). "
                               "Выберите бесплатную модель Gemini или Groq.") from e
            if "model" in msg and ("not found" in msg or "does not exist" in msg or "decommissioned" in msg):
                raise LLMError(f"Модель {model} недоступна. Выберите другую в боковой панели.") from e
            if "429" in msg or "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg:
                time.sleep(_wait_seconds(msg))  # бесплатный тариф: ждём и повторяем
                continue
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"Ошибка обращения к модели {model}: {last}")
