# Containers

```bash
docker run -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/abderahmane-ai/mimir:1.1.0-cpu
docker run --gpus all -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models ghcr.io/abderahmane-ai/mimir:1.1.0-cuda
```

Images carry the runtime, never the model weights. On first start, the model is downloaded from the Hugging Face Hub at the revision the image's package version pins, verified, and cached in `/models`. To run entirely offline after that first fetch:

```bash
docker run -p 8000:8000 -e MIMIR_API_KEYS=... -v mimir-models:/models \
  ghcr.io/abderahmane-ai/mimir:1.1.0-cpu \
  serve --host 0.0.0.0 --model-cache /models --offline
```

## Details

- **Entry point.** The entry point is `mimir` and the default command is `serve --host 0.0.0.0 --port 8000 --model-cache /models`. Pass any other `mimir` subcommand (e.g. `mcp`) to override it.
- **Security.** Both images run as a non-root user. They bind on every interface by default and therefore require keys in `MIMIR_API_KEYS`.
- **CUDA image.** The CUDA image requires a host driver compatible with CUDA 13.0. It serves the same weights and policy as the CPU image; on hardware the certificate does not list, the first start runs the release's equivalence set once.
- **Tags.** Tags follow `{version}-{variant}`, e.g. `1.1.0-cpu` and `1.1.0-cuda`. There is no `latest` tag; pin the version.

## Verifying images

Images are signed with Sigstore by the release workflow. Verify before running in production:

```bash
cosign verify ghcr.io/abderahmane-ai/mimir:1.1.0-cpu \
  --certificate-identity https://github.com/abderahmane-ai/mimir/.github/workflows/release.yml@refs/heads/main \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```
