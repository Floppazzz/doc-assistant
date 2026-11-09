FROM python:3.11-slim

# Устанавливаем системные зависимости для компиляции (нужны для faiss / numpy)

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app 

# Перенаправляем кэш Huggin Face в отдельную папку, чтобы сохранить модели на хосте

ENV HF_HOME=/app/cache/hugginface \
    PYTHONPATH=/app\
    PYTHONUNBUFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.app:app"]"]