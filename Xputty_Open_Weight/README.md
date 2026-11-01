# Xputty Open Weight

**Your material → your model.** A local-first browser canvas for collecting material, reviewing it, training a small language model, auditioning it and exporting portable weights.

Version **0.3.0**. Windows CPU acceptance is tested locally; GitHub Actions runs the same suite on Windows, Linux and macOS and builds per-OS packages. Successful branch builds attach packages to the namespaced `xow-v0.3.0` GitHub release. An OS package includes source, launchers, the English Vosk model and Apache Tika. Python packages are installed on first launch: these are not standalone native executables or fully offline installers.

## Open the canvas

Install **64-bit Python 3.11** and extract a distribution ZIP.

| System | Start |
|---|---|
| Windows x64 | Double-click `START-WINDOWS.cmd` |
| macOS Intel / Apple Silicon | Open `START-MAC.command` (or `sh start.sh` in Terminal) |
| Linux x64 | Run `sh start.sh` |

First start creates an isolated `.venv` and installs requirements. Chrome opens at **http://127.0.0.1:8788** with the worker access key. Another installed browser is used if Chrome is absent. Keep the worker running while building; closing a browser tab does not stop a build. `python launch.py --no-browser` starts without opening a window. The old VOX studio on port 8787 is unaffected.

1. **Add material.** Drop files or paste a webpage URL and press Capture link. At least three separate episodes with usable text are needed to preserve whole-episode holdouts.
2. **Review.** Edit text, replace binary files, remove unwanted material or undo removal. Original bytes remain intact; training uses the working copies.
3. **Build.** Preparation, training, evaluation and export run as a background job. Activity shows errors and the most recent checkpoint. Stop/resume preserves completed saves.
4. **Try & export.** Enter a dialogue fragment. Download GGUF, Safetensors, tokenizer, Modelfile and evaluation reports. The model ZIP excludes source transcripts.

The default is a **1.6M-parameter research model trained from scratch**, with 512-token training windows and a train-only 4,096-token vocabulary. It is not a pretrained assistant, speech model or production podcast host. Source timestamps are not measured speech pauses. Local runtime checks are separate from conversational quality acceptance.

## Input and compute

Text, structured text, subtitles, Word/documents, PDF, audio, video, images and UPOD use modality-specific adapters. Audio uses local Vosk. Images/scanned PDFs use local OCR; image/video descriptions use the configured local Ollama vision model and remain distinct from dialogue. Binary DOC requires Java on PATH and the bundled Tika JAR. Ordinary text ingestion and model training require neither Java nor Ollama.

Manual URL capture processes **one public HTML page per click**, preserving original HTML, extracted text, URL and checksums. It does not execute JavaScript, access private network services or follow links recursively. The source ZIP includes both the original extraction and current edited text.

Uploads: 256 MiB/file, 8 GiB of active material/project. Edits: 16 MiB. A source set locks when training starts to keep resumed weights tied to the same corpus. Projects live under `storage/` or `XOW_DATA`. Access keys are generated locally and are never shipped in packages. This is a **single-owner worker**, not multi-tenant hosting: everyone with its key can access its projects.

`XOW_DEVICE=auto` uses CUDA when available; `cpu` or `cuda` selects it explicitly. MPS/Apple GPU training is not implemented: macOS uses CPU. CUDA requires a compatible PyTorch installation and driver. The tiny model may not benefit from a large GPU.

## Deployment packages

| Target | Deployment |
|---|---|
| Local computer | OS launcher above |
| Shared/static hosting, including GoDaddy | Upload the **shared-hosting ZIP contents** as a static site; connect it to an HTTPS compute worker |
| OVHcloud / other VPS | `docker compose up -d --build`; put HTTPS in front of local port 8788 |
| NVIDIA Docker worker | `docker compose -f compose.yaml -f compose.gpu.yaml up -d --build` |
| Runpod Pod | Use the GPU Dockerfile or a CUDA PyTorch Pod; run `python server.py --host 0.0.0.0 --port 8787`, expose HTTP port 8787 and persist `XOW_DATA` |

Shared hosting serves the canvas; the worker performs training. Set `XOW_ORIGINS=https://your-canvas.example` for a separately hosted UI. Configure proxy request limits for your uploads. Runpod provides HTTPS proxy URLs; see its [port configuration](https://docs.runpod.io/pods/configuration/expose-ports). CPU/GPU Docker templates are supplied; a CUDA execution claim requires actual GPU acceptance.

Optional `deploy/xputty-open-weight.service` is a systemd template for an administrator-managed Linux install. Create the `xputty` user, application directory and writable data directory before enabling it. `Caddyfile` is an HTTPS proxy template; set your domain and point it at the selected worker port.

## Source installation and builds

From a source checkout, install requirements and run `python tools/fetch_assets.py` to download checksum-pinned media assets. They are prebundled in full distribution ZIPs, not committed as binary Git history. Review `assets-manifest.json` and `THIRD-PARTY-NOTICES.txt`.

```
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python tools/fetch_assets.py
python tools/build_packages.py --assets .
```

The builder uses a strict source allowlist. It excludes projects, credentials, `.venv`, logs, downloaded podcast captures, research weights and other archives. It sets executable ZIP permissions for shell launchers and emits SHA256 manifests. Runtime dependencies remain installed separately.

The GitHub repository contains the application under `Xputty_Open_Weight/`; its root workflow is copied from `ci/github-actions.yml`. No automatic deployment to a paid provider or public weight publication occurs.

## Verification

See `VERIFICATION.json` for actual tested scope, not inferred platform support. Tests cover authenticated ingestion, URL extraction, private-target rejection, editing/replacement/removal, immutable originals, source-set locking, training, evaluation, export, audition and checkpoint resume. HF tokenization/logit parity is required for completion. llama.cpp verification runs if installed (or `XOW_NATIVE_PYTHON` points to its Python). Ollama registration/generation runs when the loopback service is available; `XOW_OLLAMA_VERIFY=0` disables it.

The phase journal is SHA-linked and timestamped. It is not represented as a completed binary phase ledger or PyQoS implementation. The release does not embed Shadow Core weights or execute its router. User-provided research concepts remain separate from demonstrated capabilities.

Code follows the repository’s existing AGPL-3.0 license; see LICENSE. Third-party assets retain their respective licenses.
