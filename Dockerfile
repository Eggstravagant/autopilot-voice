# Autopilot's voice (Kokoro-82M) for Google Cloud Run. deploy.sh builds and deploys it; see README.md.
FROM python:3.12-slim-bookworm

# fp16: the fastest on CPU and the best sound (peaks ~450 MB). int8: half the memory and ~10% slower,
# for hosts with 512 MB (docker build --build-arg MODEL=int8).
ARG MODEL=fp16
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 KOKORO_MODEL=kokoro-v1.0.${MODEL}.onnx

RUN useradd --create-home --uid 1000 voice
WORKDIR /app

# Dependencies first: this layer only rebuilds when requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# The model (~180 MB) gets its own layer so code changes don't download it again
COPY fetch_model.py .
RUN python fetch_model.py "$MODEL"

COPY app.py .
USER voice
EXPOSE 8080
# One worker: the model is loaded once and uses every CPU. No access log: Cloud Run logs each request.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port \"$PORT\" --workers 1 --no-access-log --timeout-keep-alive 30 --timeout-graceful-shutdown 8"]
