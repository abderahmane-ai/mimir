# The CUDA runtime image: the package and ONNX Runtime GPU, never the model. onnxruntime-gpu
# 1.30.0 is built against CUDA 13.0 and cuDNN 9, which this base image provides.
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04 AS base
RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3.12 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/bin/python3.12
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --extra local-gpu --extra server --extra mcp

FROM base
LABEL io.modelcontextprotocol.server.name="io.github.abderahmane-ai/mimir" \
      org.opencontainers.image.title="mimir" \
      org.opencontainers.image.description="MIMIR decision server, CUDA" \
      org.opencontainers.image.source="https://github.com/abderahmane-ai/mimir" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN useradd --create-home --uid 10001 mimir && install -d -o mimir /models
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1
USER mimir
VOLUME /models
EXPOSE 8000
ENTRYPOINT ["mimir"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--model-cache", "/models"]
