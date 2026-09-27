# The CPU runtime image: the package and ONNX Runtime, never the model. The model is downloaded
# at start into /models from the revision this package version pins.
FROM python:3.12.14-slim-trixie AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --extra local --extra server --extra mcp

FROM python:3.12.14-slim-trixie
LABEL io.modelcontextprotocol.server.name="io.github.Mythologic/mimir" \
      org.opencontainers.image.title="mimir" \
      org.opencontainers.image.description="MIMIR decision server, CPU" \
      org.opencontainers.image.source="https://github.com/Mythologic/mimir" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN useradd --create-home --uid 10001 mimir && install -d -o mimir /models
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1
USER mimir
VOLUME /models
EXPOSE 8000
ENTRYPOINT ["mimir"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--model-cache", "/models"]
