FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY frontend/ /app/frontend/
RUN mkdir -p /app/chroma_db /app/vectorizer_cache

EXPOSE 8000

CMD ["uvicorn", "esco_mapper.backend:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/src"]
