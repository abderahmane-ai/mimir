# Containers

```bash
docker run -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/vathosai/mimir:1.0.0-cpu
docker run --gpus all -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/vathosai/mimir:1.0.0-cuda
```

Images carry the runtime, never the model. On first start the model is downloaded from the
revision the image's package version pins, verified, and kept in `/models`. To run from that
cache with no network access, end the command with
`serve --host 0.0.0.0 --model-cache /models --offline`.

- The entry point is `mimir` and the default command is
  `serve --host 0.0.0.0 --port 8000 --model-cache /models`; any other command, such as `mcp`,
  replaces it.
- Both run as a non-root user and need keys in `MIMIR_API_KEYS`, since they listen on every
  address.
- The CUDA image needs a driver for CUDA 13.0.
- Images are signed with Sigstore by the release workflow:

```bash
cosign verify ghcr.io/vathosai/mimir:1.0.0-cpu \
  --certificate-identity https://github.com/vathosai/mimir/.github/workflows/release.yml@refs/heads/main \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```
