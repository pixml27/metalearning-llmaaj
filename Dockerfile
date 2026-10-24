FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LITELLM_MODE=PRODUCTION \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    PYDANTIC_AI_NO_BANNER=1

WORKDIR /app
COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements.lock.txt
COPY solution ./solution
COPY src/common.py ./src/common.py

ENTRYPOINT ["python", "-m", "solution"]
CMD ["--help"]
