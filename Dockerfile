FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/cyber-agent

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir . && \
    python -m pip install --no-cache-dir "pytest>=8.0,<9"

RUN groupadd --gid 10001 agent && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin agent

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["python", "-m", "cyber_agent.sandbox_worker"]

