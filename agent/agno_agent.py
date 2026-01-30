# agent/agno_agent.py
"""LLM coding agent wrapper (Agno)"""

import os
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

Дополнительные ограничения:
- Меняй только минимально необходимые файлы
- Не форматируй код целиком без необходимости
- Если не уверен — сначала прочитай файл
- Если задача неясна — выбери самый простой вариант

Отвечай на русском языке.
"""

def _make_agent(model_id: str, api_key: str) -> Agent:
    return Agent(
        model=OpenRouter(id=model_id, api_key=api_key),
        tools=[FileTools(), ShellTools()],
        instructions=SYSTEM_PROMPT,
        markdown=True,
    )

def run_coding_agent(task: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    primary = (os.getenv("MODEL") or "").strip() or "openai/gpt-4o-mini"
    fallbacks = [
        primary,
        "openai/gpt-4o-mini",
        "openai/gpt-4.1-mini",   # если доступно в OpenRouter
    ]

    last_err = None
    for m in fallbacks:
        try:
            agent = _make_agent(model_id=m, api_key=api_key)
            resp = agent.run(task)
            return resp.content
        except Exception as e:
            last_err = e
            # если это не 429, можно сразу пробросить; но для простоты пробуем дальше
            continue

    raise RuntimeError(f"All models failed, last error: {last_err}")
