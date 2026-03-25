FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Включаем компиляцию байткода для uv (немного ускоряет старт бота)
ENV UV_COMPILE_BYTECODE=1

# Добавляем инструменты для проверки сети
RUN apt-get update && apt-get install -y curl iputils-ping ffmpeg libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*

# Копируем бинарники uv из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

COPY . /app/

CMD ["uv", "run", "--no-sync", "python", "main.py"]
