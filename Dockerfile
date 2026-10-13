FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# One worker process holds one copy of the model in memory; ONNX Runtime
# releases the GIL during inference, so threads handle concurrent requests.
CMD ["sh", "-c", "exec gunicorn run:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 60 --access-logfile -"]
