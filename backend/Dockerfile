FROM continuumio/miniconda3:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install faiss-cpu via conda (and force python 3.11 to avoid miniconda's default python 3.13 conflict)
RUN conda install -y -c pytorch python=3.11 faiss-cpu=1.8.0

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
