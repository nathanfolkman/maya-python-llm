# maya-python-llm

A fine-tuned LLM optimised for writing Maya Python scripts — covering both `maya.cmds` and the Maya API 2.0 (`maya.api.OpenMaya`).

The pipeline scrapes the Maya API 2.0 docs, generates synthetic QA pairs using a teacher model, then fine-tunes a small local model (Qwen2.5-Coder-3B) that can run anywhere including inside Maya itself.

---

## Pipeline overview

```
scrape.py          →  maya_api_2_raw.json
build_dataset.py   →  api_pairs.jsonl          (ground-truth doc pairs)
generate_synthetic →  synthetic_pairs.jsonl    (teacher-generated QA pairs)
                   →  train.jsonl              (merged final dataset)
finetune.py        →  maya-coder-3b/*.gguf     (fine-tuned model)
```

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — `pip install uv`
- NVIDIA GPU with CUDA 12.x (for fine-tuning)
- [Ollama](https://ollama.com) (for local generation and inference)

---

## Step 1 — Scrape the Maya API docs

```bash
uv run python scrape.py
```

Output: `maya_api_2_raw.json` (~310 classes with methods, parameters, descriptions)

---

## Step 2 — Build the ground-truth dataset

```bash
uv run python build_dataset.py
```

Output: `api_pairs.jsonl` (~9,000 doc-derived QA pairs)

---

## Step 3 — Generate synthetic training data

Choose **one** approach (or run both in parallel to compare quality):

### Option A — Gemini free tier (cloud)

Requires a free [Google AI Studio](https://aistudio.google.com) API key from a project **without billing enabled**.

Free tier limits: 15 RPM / 500 RPD / 215K TPM — full run takes ~2 days.

```bash
# Add your key to .env
echo "GOOGLE_AI_STUDIO_KEY=your-key-here" > .env

# Run (resumes automatically from cache if interrupted)
GOOGLE_AI_STUDIO_KEY=$(grep GOOGLE_AI_STUDIO_KEY .env | cut -d= -f2) uv run python -u generate_synthetic.py
```

Output: `synthetic_pairs.jsonl`, `train.jsonl`

### Option B — Local model via Ollama (no rate limits)

Requires [Ollama](https://ollama.com) running locally.

```bash
# Pull the model
ollama pull qwen2.5-coder:7b

# Run (909 classes × 3 variants × 3 calls = 2,727 total calls)
uv run python -u generate_synthetic_local.py
```

Output: `synthetic_pairs_local.jsonl`, `train_local.jsonl`

> **Tip:** To improve Ollama throughput, enable parallel inference:
> ```bash
> sudo bash restart_ollama.sh   # sets OLLAMA_NUM_PARALLEL=2
> ```
> Then set `WORKERS = 6` in `generate_synthetic_local.py`.

---

## Step 4 — Fine-tune

Install training dependencies (once):

```bash
bash install_training.sh
```

Smoke test (100 steps):

```bash
uv run python finetune.py --max-steps 100
```

Full training (~1-2 hrs on RTX 4090):

```bash
uv run python finetune.py
```

By default this trains on both `train.jsonl` and `train_local.jsonl` merged together. Training config: Qwen2.5-Coder-3B-Instruct + QLoRA (r=16) + Unsloth, exported to Q4_K_M GGUF.

---

## Step 5 — Register with Ollama and run

```bash
ollama create maya-coder -f maya-coder-3b/Modelfile
ollama run maya-coder "How do I get the world space position of a joint using API 2.0?"
```

---

## Project structure

```
scrape.py                  Maya API 2.0 doc scraper
build_dataset.py           Ground-truth QA pair builder
generate_synthetic.py      Gemini-based synthetic data generator
generate_synthetic_local.py  Ollama-based synthetic data generator
finetune.py                Unsloth QLoRA fine-tuning + GGUF export
install_training.sh        One-time training dependency installer
restart_ollama.sh          Helper to set OLLAMA_NUM_PARALLEL via systemd
maya-coder-3b/Modelfile    Ollama model definition with system prompt
.env                       API keys (not committed)
```
