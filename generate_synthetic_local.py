"""
generate_synthetic_local.py

Uses a local Ollama model (qwen2.5-coder:7b) as a teacher model to generate
synthetic Maya Python QA pairs in three variants per class:

  api2        — OpenMaya API 2.0 code examples (the "professional SDK" way)
  cmds        — maya.cmds code examples (the "quick & dirty" way)
  translation — cmds ↔ OpenMaya API 2.0 conversion pairs

No rate limits or API keys — runs fully local via Ollama's OpenAI-compatible API.

Run: uv run python generate_synthetic_local.py
Output: synthetic_pairs_local.jsonl  (merged with api_pairs.jsonl → train_local.jsonl)
"""

import json
import os
import sys
import time
import ast
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────

INPUT_FILE = Path("maya_api_2_raw.json")
CACHE_DIR = Path("cache/synthetic_local")
OUTPUT_FILE = Path("synthetic_pairs_local.jsonl")
TRAIN_FILE = Path("train_local.jsonl")

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5-coder:7b"
WORKERS = 3  # concurrent requests; keeps Ollama queue full with no idle gaps

MAX_RETRIES = 3
RETRY_BASE_WAIT = 5

# Generation settings
QA_PAIRS_PER_CLASS = 5        # per call; 3 calls per variant → 15 total per variant per class
QA_CALLS_PER_VARIANT = 3      # repeat each variant 3x to reach 15 pairs total
PROMPT_VARIANTS = ["api2", "cmds", "translation"]
MAX_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are an expert Maya Python developer with deep knowledge of both "
    "maya.cmds and the Maya API 2.0 (maya.api.OpenMaya). "
    "Generate realistic developer questions and correct, runnable Python code answers. "
    "Always include necessary imports. Write clean, PEP-8 compliant code with brief comments."
)


# ── Ollama client ────────────────────────────────────────────────────────────

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _client


# ── Cache ────────────────────────────────────────────────────────────────────

def cache_path(class_name: str, variant: str, call_idx: int) -> Path:
    safe = re.sub(r"[^\w]", "_", class_name)
    return CACHE_DIR / variant / f"{safe}_{call_idx}.json"


def load_cache(class_name: str, variant: str, call_idx: int) -> list[dict] | None:
    p = cache_path(class_name, variant, call_idx)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(class_name: str, variant: str, call_idx: int, pairs: list[dict]):
    p = cache_path(class_name, variant, call_idx)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)


# ── Local model call ──────────────────────────────────────────────────────────

def call_local(prompt: str) -> str | None:
    wait = RETRY_BASE_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.8,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        except Exception as e:
            tqdm.write(f"    [error attempt {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
                wait *= 2

    tqdm.write(f"    [failed after {MAX_RETRIES} attempts]")
    return None


# ── Prompt builders ──────────────────────────────────────────────────────────

def _method_summary(cls: dict, max_methods: int = 10) -> str:
    methods = [
        m for m in cls.get("methods", [])
        if m.get("type") == "method"
        and not m["name"].startswith("__")
        and (m.get("description") or m.get("signature"))
    ]
    methods.sort(
        key=lambda m: (bool(m.get("parameters")), bool(m.get("description"))),
        reverse=True,
    )
    lines = []
    for m in methods[:max_methods]:
        sig = m.get("signature") or m["name"] + "()"
        ret = m.get("return_type") or ""
        desc = (m.get("description") or "")[:100]
        params = m.get("parameters") or []
        param_str = ", ".join(
            f'{p["name"]}:{p.get("type","?")}' for p in params[:4]
        )
        line = f"- {sig}"
        if ret:
            line += f" → {ret}"
        if param_str:
            line += f"  [{param_str}]"
        if desc:
            line += f"  # {desc}"
        lines.append(line)
    return "\n".join(lines)


def build_api2_prompt(cls: dict) -> str:
    class_name = cls["class"]
    module = cls.get("module", "OpenMaya")
    description = cls.get("description", "")
    methods_text = _method_summary(cls)

    return f"""I'm building training data for a Maya Python API 2.0 code generation assistant.

Class: {module}.{class_name}
Description: {description}

Key methods:
{methods_text}

Generate exactly {QA_PAIRS_PER_CLASS} diverse question-answer training pairs using Maya API 2.0.
Each pair must:
1. Ask a realistic developer question about a specific task (not just "what does X do")
2. Answer with complete, runnable Python code using `import maya.api.{module} as om`
3. Cover a range of tasks: basic instantiation, querying data, modifying data, iteration, error handling, combining with other classes

Format as a JSON object with a single "pairs" key containing the array (no markdown fences, no extra text):
{{"pairs": [{{"question": "...", "answer": "```python\\n...\\n```"}}, ...]}}"""


def build_cmds_prompt(cls: dict) -> str:
    class_name = cls["class"]
    module = cls.get("module", "OpenMaya")
    description = cls.get("description", "")
    methods_text = _method_summary(cls, max_methods=6)

    return f"""I'm building training data for a Maya Python scripting assistant.

The Maya API 2.0 class {module}.{class_name} ({description}) is the low-level SDK equivalent of common maya.cmds operations.

Key API 2.0 methods for context:
{methods_text}

Generate exactly {QA_PAIRS_PER_CLASS} diverse question-answer training pairs using maya.cmds (NOT the API).
Each pair must:
1. Ask a realistic rigging/modeling/animation/scene task question
2. Answer with complete, runnable Python code using `import maya.cmds as cmds`
3. Cover a range of tasks related to what this class handles conceptually
4. Include practical patterns: querying attributes, creating/modifying nodes, working with selections, loops over scene objects

Format as a JSON object with a single "pairs" key containing the array (no markdown fences, no extra text):
{{"pairs": [{{"question": "...", "answer": "```python\\n...\\n```"}}, ...]}}"""


def build_translation_prompt(cls: dict) -> str:
    class_name = cls["class"]
    module = cls.get("module", "OpenMaya")
    description = cls.get("description", "")
    methods_text = _method_summary(cls, max_methods=6)

    return f"""I'm building training data to help developers migrate from maya.cmds to Maya API 2.0.

The class {module}.{class_name} ({description}) provides API 2.0 equivalents for common cmds operations.

Key API 2.0 methods:
{methods_text}

Generate exactly {QA_PAIRS_PER_CLASS} question-answer pairs that show BOTH approaches side by side.
Mix of directions:
- "Convert this maya.cmds code to API 2.0: [snippet]" → API 2.0 version
- "What is the maya.cmds equivalent of this API 2.0 code: [snippet]" → cmds version
- "Rewrite this script using API 2.0 for better performance: [cmds snippet]" → API 2.0 version

Each answer must show BOTH the original and the converted code with a brief explanation of the key differences.

Format as a JSON object with a single "pairs" key containing the array (no markdown fences, no extra text):
{{"pairs": [{{"question": "...", "answer": "```python\\n...\\n```"}}, ...]}}"""


PROMPT_BUILDERS = {
    "api2": build_api2_prompt,
    "cmds": build_cmds_prompt,
    "translation": build_translation_prompt,
}


# ── Response parsing ──────────────────────────────────────────────────────────

def _sanitize_json(text: str) -> str:
    """Fix common LLM JSON mistakes: bad escapes inside string values."""
    # Replace invalid backslash escapes inside JSON strings.
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    # Everything else (e.g. \p \e \( ) must be double-escaped.
    def fix_escapes(m: re.Match) -> str:
        s = m.group(0)
        # Re-escape any backslash not followed by a valid JSON escape char
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        return s
    # Match JSON string literals (handles escaped quotes inside)
    return re.sub(r'"(?:[^"\\]|\\.)*"', fix_escapes, text, flags=re.DOTALL)


def parse_response(text: str, class_name: str, module: str, variant: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []

    # Strip outer markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try parsing as-is, then with escape sanitization
    parsed = None
    for candidate in [text, _sanitize_json(text)]:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if parsed is None:
        tqdm.write(f"    [parse fail] {class_name}:{variant} — could not decode JSON")
        return []

    # Unwrap {"pairs": [...]} envelope or use bare list
    if isinstance(parsed, dict):
        items = parsed.get("pairs") or next(
            (v for v in parsed.values() if isinstance(v, list)), None
        )
    else:
        items = parsed

    if not isinstance(items, list):
        return []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if not question or not answer:
            continue

        # Validate any Python code blocks
        for code_match in re.finditer(r"```python\n(.*?)```", answer, re.DOTALL):
            code = code_match.group(1).strip()
            try:
                ast.parse(code)
            except SyntaxError as e:
                tqdm.write(f"    [syntax err] {e}")
                answer = None
                break

        if answer is None:
            continue

        results.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "_source": f"synthetic_local:{variant}:{module}.{class_name}",
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run scrape.py first.", file=sys.stderr)
        sys.exit(1)

    # Verify Ollama is reachable
    try:
        models = get_client().models.list()
        available = [m.id for m in models.data]
        if MODEL not in available:
            print(f"Error: model '{MODEL}' not found in Ollama. Available: {available}", file=sys.stderr)
            print(f"Run: ollama pull {MODEL}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error: cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}", file=sys.stderr)
        print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        api_data: list[dict] = json.load(f)

    classes = [
        c for c in api_data
        if c["class"] != c.get("module")
        and any(m.get("type") == "method" for m in c.get("methods", []))
    ]

    total_calls = len(classes) * len(PROMPT_VARIANTS) * QA_CALLS_PER_VARIANT
    print(f"Loaded {len(api_data)} classes | Model: {MODEL}")
    print(f"Classes: {len(classes)} × {len(PROMPT_VARIANTS)} variants × {QA_CALLS_PER_VARIANT} calls = {total_calls} total calls\n")

    all_pairs: list[dict] = []
    file_lock = Lock()
    stats = {"cached": 0, "new": 0}

    # Build full work list, skipping already-cached calls
    work_items = []
    pre_cached_pairs = []
    for cls in classes:
        class_name = cls["class"]
        for variant in PROMPT_VARIANTS:
            for call_idx in range(QA_CALLS_PER_VARIANT):
                cached = load_cache(class_name, variant, call_idx)
                if cached is not None:
                    stats["cached"] += 1
                    pre_cached_pairs.extend(cached)
                else:
                    work_items.append((cls, variant, call_idx))

    def process(item: tuple) -> tuple[str, str, str, int, list[dict]]:
        cls, variant, call_idx = item
        class_name = cls["class"]
        module = cls.get("module", "OpenMaya")
        prompt = PROMPT_BUILDERS[variant](cls)
        raw = call_local(prompt)
        if raw is None:
            save_cache(class_name, variant, call_idx, [])
            return class_name, module, variant, call_idx, []
        pairs = parse_response(raw, class_name, module, variant)
        save_cache(class_name, variant, call_idx, pairs)
        return class_name, module, variant, call_idx, pairs

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for pair in pre_cached_pairs:
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        all_pairs.extend(pre_cached_pairs)

        bar = tqdm(
            total=total_calls,
            initial=stats["cached"],
            desc="Generating",
            unit="call",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(process, item): item for item in work_items}
            for future in as_completed(futures):
                class_name, module, variant, call_idx, pairs = future.result()
                count = len(pairs)
                status = f"✓{count}" if count else "FAIL"
                tqdm.write(f"  {class_name}:{variant}[{call_idx}] {status}")
                with file_lock:
                    for pair in pairs:
                        out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    all_pairs.extend(pairs)
                stats["new"] += 1
                bar.update(1)
                bar.set_postfix_str(f"{class_name}:{variant}", refresh=True)

        bar.close()

    print(f"\nDone!")
    print(f"  Cached variant calls   : {stats['cached']}")
    print(f"  New model calls        : {stats['new']}")
    print(f"  Total synthetic pairs  : {len(all_pairs)}")
    print(f"  Saved to {OUTPUT_FILE}")

    # Merge api_pairs.jsonl + synthetic_pairs_local.jsonl → train_local.jsonl
    print(f"\nMerging into {TRAIN_FILE}...")
    api_pairs_file = Path("api_pairs.jsonl")
    total_train = 0
    with open(TRAIN_FILE, "w", encoding="utf-8") as train_f:
        for src in [api_pairs_file, OUTPUT_FILE]:
            if src.exists():
                with open(src, encoding="utf-8") as src_f:
                    for line in src_f:
                        line = line.strip()
                        if line:
                            train_f.write(line + "\n")
                            total_train += 1
            else:
                print(f"  Warning: {src} not found")

    print(f"  Total training pairs in {TRAIN_FILE}: {total_train}")


if __name__ == "__main__":
    main()
