---
title: MindCare
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Empathetic chat demo (not medical advice); text and audio.
---

# MindCare (Hugging Face Space)

Multimodal demo: Whisper, optional Wav2Vec2, and a fine-tuned LLM. **Not medical or psychological counselling**; seek real-world help in a crisis.

## Deploy on Hugging Face

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space), **SDK: Gradio**, then push this repo or sync from GitHub.
2. **Hardware**: free CPU is unlikely to run this stack well. In **Settings → Hardware**, choose a **GPU** (e.g. T4 / A10G) according to quota and budget.
3. **Secrets** (if needed): for private or gated models, add `HF_TOKEN` under **Settings → Repository secrets** (same token as `huggingface-cli login`).
4. **First load** downloads Whisper, optional Wav2Vec2, and the LLM; the queue helps with long runs. **Text-only** chats load the LLM on demand without pre-loading speech models.

## Dependencies and local dev

- Spaces use root `requirements.txt` (**CUDA 12.1** PyTorch wheels for GPU Spaces). On a machine **without NVIDIA GPU**, install CPU `torch` / `torchaudio` yourself, or temporarily remove the `--extra-index-url` lines and the `+cu121` torch pins.
- System packages: see `apt.txt` (includes `ffmpeg` for audio).

## Environment variables (optional)

| Variable | Meaning |
|----------|---------|
| `HF_TOKEN` | In Space Secrets, for private/gated models |
| `MINDCARE_LOG_INTERACTIONS=1` | Enable SQLite logging (ephemeral disk on Spaces) |
| `MINDCARE_EAGER_LOAD=1` | Preload all models on **non-Space** hosts |
| `GRADIO_AUTH_USERNAME` / `GRADIO_AUTH_PASSWORD` | HTTP basic auth for Gradio if both set |
| `MINDCARE_WHISPER_SIZE` | `tiny` / `base` / `small` (default) / `medium` / `large` — smaller saves VRAM, slightly worse ASR |
| `MINDCARE_EMOTION_MODE` | `model` (default, Wav2Vec2) / `text` (keyword heuristic, no emotion model) / `neutral` |
| `MINDCARE_LLM_LOAD_IN_4BIT=1` | Try 4-bit LLM on **CUDA** (`bitsandbytes`); falls back on failure |
| `MINDCARE_LLM_REPO` | Override default Hugging Face model id |
| `MINDCARE_PARALLEL_AUDIO=1` | Parallel STT + acoustic emotion (**do not** use with `MINDCARE_EMOTION_MODE=text`) |
| `MINDCARE_MAX_NEW_TOKENS` | Max new LLM tokens per reply (default `200`, clamped 32–512) |

## Google Colab

Use a **GPU** runtime (free tier has limits). **Do not** install the pinned CUDA `requirements.txt` on top of Colab’s PyTorch; install the other packages on Colab’s existing `torch`. See [`notebooks/mindcare_colab.ipynb`](notebooks/mindcare_colab.ipynb).

## Repo layout

- App entry: `app.py`
- Pipeline: `modules/pipelines/p.py`
- Crisis / violence keywords and canned replies: `modules/safety.py`
- Colab walkthrough: `notebooks/mindcare_colab.ipynb`

`modules/safety.py` still contains **Chinese substrings inside keyword lists** (`CRISIS_KEYWORDS_ZH`, `VIOLENCE_TO_OTHERS_KEYWORDS_ZH`) so messages typed in Chinese still trigger safety routing. All **user-facing canned replies** are English.

Legacy scripts under `test/` are stubbed; the supported entry point is `app.py`.
