FROM python:3.12 AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


WORKDIR /app

COPY . /
RUN uv pip install --system --no-cache-dir -r orchestrator/requirements.txt


CMD [ "fastapi", "run", "orchestrator/main.py"  , "--port", "80"]
