# Two stages so the 30 MB corpus and pip's build cache stay out of the final image;
# only the trained artifact crosses over.
FROM python:3.12-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fnd/ fnd/
RUN python -m fnd.cli download && python -m fnd.cli train


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY fnd/ fnd/
COPY tests/ tests/
COPY --from=builder /app/artifacts/baseline.joblib artifacts/baseline.joblib

# ponytail: no healthcheck — this is a batch CLI, not a long-running service.
RUN useradd --create-home --uid 10001 fnd && chown -R fnd:fnd /app
USER fnd

ENTRYPOINT ["python", "-m", "fnd.cli"]
CMD ["predict", "--text", "Paste an article here to classify it."]


# HTTP service variant: docker build --target serve -t fnd-api .
FROM python:3.12-slim AS serve

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-api.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY fnd/ fnd/
COPY --from=builder /app/artifacts/baseline.joblib artifacts/baseline.joblib

RUN useradd --create-home --uid 10001 fnd && chown -R fnd:fnd /app
USER fnd

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
ENTRYPOINT ["uvicorn", "fnd.api:app", "--host", "0.0.0.0", "--port", "8000"]
