FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirementsFront.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /src/

EXPOSE 8001

ENV BACKEND_URL=http://backend:8000

CMD ["uvicorn", "esco_frontend.app:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "/app/src"]
