FROM python:3.11-slim

# System deps (git нужен для работы агента)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . /app

# Python deps
RUN pip install --upgrade pip \
    && pip install PyGithub openai

# Default command: show help
CMD ["python", "-m", "agent.code_agent"]
