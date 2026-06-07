FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt && apt-get purge -y build-essential && apt-get autoremove -y

COPY app_web.py cache.py courtlistener.py ./
COPY full_ifp.json full_counsel.json full_extension.json ./
COPY A3_Lopez_Perez.md ./

RUN useradd -m -u 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/healthz || exit 1

CMD ["sh", "-c", "gunicorn -w 2 -t 120 -b 0.0.0.0:${PORT} app_web:app"]
