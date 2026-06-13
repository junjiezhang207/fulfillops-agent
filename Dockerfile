# syntax=docker/dockerfile:1

FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_BASE=/api/v1
RUN npm run build


FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS app

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY app/ ./app/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY knowledge_base/ ./knowledge_base/
COPY main.py ./
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN uv sync --frozen

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
