"""
finetune.py

Fine-tunes Qwen2.5-Coder-3B-Instruct on the merged Maya Python dataset using
Unsloth + QLoRA, then exports to GGUF (Q4_K_M) for use with Ollama.

Training data: train.jsonl + train_local.jsonl (merged, deduplicated)
Output:        maya-coder-3b/  (GGUF + tokenizer)

Run: uv run python finetune.py
     uv run python finetune.py --max-steps 100   # quick smoke test
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ── Config ──────────────────────────────────────────────────────────────────

BASE_MODEL   = "unsloth/Qwen2.5-Coder-3B-Instruct"
OUTPUT_DIR   = Path("maya-coder-3b")
DATA_SOURCES = [Path("train.jsonl"), Path("train_local.jsonl")]

MAX_SEQ_LENGTH = 2048
LORA_RANK      = 16
LORA_ALPHA     = 16
BATCH_SIZE     = 2
GRAD_ACCUM     = 4       # effective batch = 8
LR             = 2e-4
EPOCHS         = 3
WARMUP_RATIO   = 0.05
GGUF_QUANT     = "q4_k_m"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset() -> Dataset:
    rows = []
    seen = set()

    for src in DATA_SOURCES:
        if not src.exists():
            print(f"  Warning: {src} not found, skipping")
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Deduplicate on the user message content
                key = obj["messages"][1]["content"]
                if key in seen:
                    continue
                seen.add(key)
                rows.append(obj)
        print(f"  Loaded {src}: {len(rows)} total after dedup")

    print(f"  Final dataset: {len(rows)} pairs")
    return Dataset.from_list(rows)


# ── Formatting ───────────────────────────────────────────────────────────────

def make_formatter(tokenizer):
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    return format_example


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap training steps (use 100 for a smoke test)")
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,           # auto-detect bf16/fp16
        load_in_4bit=True,
    )

    print("Applying QLoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("Loading dataset...")
    dataset = load_dataset()
    dataset = dataset.map(make_formatter(tokenizer))

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS if args.max_steps == -1 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,           # pack short examples together for efficiency
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print(f"Training on {len(dataset)} examples...")
    trainer.train()

    print(f"\nExporting to GGUF ({GGUF_QUANT})...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.save_pretrained_gguf(
        str(OUTPUT_DIR),
        tokenizer,
        quantization_method=GGUF_QUANT,
    )
    gguf_files = list(OUTPUT_DIR.glob("*.gguf"))
    gguf_name = gguf_files[0].name if gguf_files else f"maya-coder-3b-unsloth-{GGUF_QUANT.upper()}.gguf"
    print(f"Done! GGUF saved to {OUTPUT_DIR}/{gguf_name}")

    # Update Modelfile FROM line to match actual filename
    modelfile = OUTPUT_DIR / "Modelfile"
    if modelfile.exists():
        text = modelfile.read_text()
        text = text.splitlines()
        text[0] = f"FROM ./{gguf_name}"
        modelfile.write_text("\n".join(text) + "\n")

    print(f"\nTo use with Ollama:")
    print(f"  ollama create maya-coder -f {OUTPUT_DIR}/Modelfile")
    print(f"  ollama run maya-coder")


if __name__ == "__main__":
    main()
