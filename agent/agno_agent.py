# agent/agno_agent.py
"""LLM coding agent wrapper (Agno + OpenRouter)"""

import os
import time
from typing import Iterable

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools

SYSTEM_PROMPT = """Ты - coding agent, помощник программиста.

Твои возможности:
- Работа с файлами: читать, писать, список файлов
- Выполнение shell команд

Правила:
1. Сначала изучи структуру проекта
2. Читай файлы перед изменением
3. Объясняй что делаешь
4. При изменении файла пиши ПОЛНОЕ содержимое файла
5. После правок запусти команды проверки (lint/test/build), если они есть
6. Не трогай node_modules, dist, build, .git
7. Меняй минимально необходимое количество файлов

Отвечай на русском языке.
"""


ALLOWLIST_DEFAULT = [
    "tngtech/deepseek-r1t2-chimera:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
]


def _make_agent(model_id: str, api_key: str) -> Agent:
    return Agent(
        model=OpenRouter(id=model_id, api_key=api_key),
        tools=[FileTools(), ShellTools()],
        instructions=SYSTEM_PROMPT,
        markdown=True,
    )


def _looks_like_transient_error(msg: str) -> bool:
    """Ошибки, при которых имеет смысл попробовать другую модель/повторить."""
    s = msg.lower()
    markers = [
        "rate limit",
        "rate-limited",
        "temporarily rate-limited",
        "provider returned error",
        "insufficient credits",
        "no models provided",
        "error code: 429",
        "error code: 402",
        "code': 429",
        "code': 402",
    ]
    return any(m in s for m in markers)


def _iter_models() -> Iterable[str]:
    # Если MODEL задан — пробуем его первым (но только если не пустой)
    env_model = (os.getenv("MODEL") or "").strip()
    if env_model:
        yield env_model

    seen = {env_model} if env_model else set()
    for m in ALLOWLIST_DEFAULT:
        if m not in seen:
            seen.add(m)
            yield m


def run_coding_agent(task: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    last_err: Exception | None = None
    last_msg: str | None = None

    for model_id in _iter_models():
        try:
            print(f"[LLM] Trying model: {model_id}")
            agent = _make_agent(model_id=model_id, api_key=api_key)
            resp = agent.run(task)
            return resp.content
        except Exception as e:
            last_err = e
            last_msg = str(e)
            print(f"[LLM] Model failed: {model_id} | error: {last_msg}")

            if _looks_like_transient_error(last_msg):
                time.sleep(1.0) 
                continue

            raise

    raise RuntimeError(f"All models failed. Last error: {last_msg or last_err}")
