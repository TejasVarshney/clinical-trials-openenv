FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

WORKDIR /app

COPY clinical_trial_env ./clinical_trial_env
COPY inference.py ./inference.py
COPY README.md ./README.md
COPY openenv.yaml ./openenv.yaml
COPY pyproject.toml ./pyproject.toml

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "clinical_trial_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
