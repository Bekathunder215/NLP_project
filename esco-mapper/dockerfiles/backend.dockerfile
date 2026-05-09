FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirementsBase.txt .
COPY requirementsML.txt .
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_CUDA_ARCH_LIST="cpu"
ENV FORCE_CUDA="0"

RUN pip install --no-cache-dir -r requirementsBase.txt
RUN pip install --no-cache-dir -r requirementsML.txt

COPY src/ ./src/

# Optional env vars
ENV ESCO_LAZY_LOAD=false
ENV QLEVER_URL=http://qlever:7654

EXPOSE 8000

CMD ["uvicorn", "esco_mapper.backend:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/src"]
