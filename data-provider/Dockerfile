FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache datasets requests

COPY init_script.sh .
COPY init_data.py .

RUN chmod +x init_script.sh

CMD ["./init_script.sh"]
