"""
generate_synthetic.py

Uses Google AI Studio (gemini-3.1-flash-lite-preview) as a teacher model to generate
synthetic Maya Python QA pairs in three variants per class:

  api2        — OpenMaya API 2.0 code examples (the "professional SDK" way)
  cmds        — maya.cmds code examples (the "quick & dirty" way)
  translation — cmds ↔ OpenMaya API 2.0 conversion pairs

Rate limits (free tier): 15 RPM / 215K TPM / 500 RPD
SLEEP_BETWEEN_CALLS = 6.0s → 10 RPM, safely under 15 RPM free tier limit.

Run: GOOGLE_AI_STUDIO_KEY=<key> uv run python generate_synthetic.py
Output: synthetic_pairs.jsonl  (merged with api_pairs.jsonl → train.jsonl)
"""

import json
import os
import sys
import time
import ast
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────

INPUT_FILE = Path("maya_api_2_raw.json")
CACHE_DIR = Path("cache/synthetic")
REQUEST_LOG = Path("cache/request_log.json")
OUTPUT_FILE = Path("synthetic_pairs.jsonl")
TRAIN_FILE = Path("train.jsonl")

API_KEY = os.environ.get("GOOGLE_AI_STUDIO_KEY", "")
MODEL = "gemini-3.1-flash-lite-preview"

# Rate limiting (actual limits: 4,000 RPM / 4M TPM)
SLEEP_BETWEEN_CALLS = 6.0  # 6s → 10 RPM, safely under 15 RPM free tier limit
DAILY_REQUEST_HARD_CAP = 500
MAX_RETRIES = 3
RETRY_BASE_WAIT = 10

# Generation settings
QA_PAIRS_PER_CLASS = 15       # per variant per class
PROMPT_VARIANTS = ["api2", "cmds", "translation"]
MAX_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are an expert Maya Python developer with deep knowledge of both "
    "maya.cmds and the Maya API 2.0 (maya.api.OpenMaya). "
    "Generate realistic developer questions and correct, runnable Python code answers. "
    "Always include necessary imports. Write clean, PEP-8 compliant code with brief comments."
)


# ── Rate limit tracking ──────────────────────────────────────────────────────

def load_request_log() -> dict:
    if REQUEST_LOG.exists():
        with open(REQUEST_LOG) as f:
            return json.load(f)
    return {}


def save_request_log(log: dict):
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(REQUEST_LOG, "w") as f:
        json.dump(log, f, indent=2)


def get_today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_daily_count(log: dict) -> int:
    return log.get(get_today_key(), 0)


def increment_daily_count(log: dict) -> dict:
    key = get_today_key()
    log[key] = log.get(key, 0) + 1
    return log


def check_daily_cap(log: dict):
    count = get_daily_count(log)
    if count >= DAILY_REQUEST_HARD_CAP:
        print(
            f"\n[STOP] Daily request cap reached ({count}/{DAILY_REQUEST_HARD_CAP}).",
            file=sys.stderr,
        )
        sys.exit(0)


# ── Cache ────────────────────────────────────────────────────────────────────

def cache_path(class_name: str, variant: str) -> Path:
    safe = re.sub(r"[^\w]", "_", class_name)
    return CACHE_DIR / variant / f"{safe}.json"


def load_cache(class_name: str, variant: str) -> list[dict] | None:
    p = cache_path(class_name, variant)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(class_name: str, variant: str, pairs: list[dict]):
    p = cache_path(class_name, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)


# ── Gemini API call ──────────────────────────────────────────────────────────

_client: genai.Client | None = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client


def call_gemini(prompt: str, log: dict) -> str | None:
    check_daily_cap(log)

    wait = RETRY_BASE_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = get_client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_TOKENS,
                    temperature=0.8,
                ),
            )
            increment_daily_count(log)
            save_request_log(log)
            return response.text

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                tqdm.write(f"    [429] waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                wait *= 2
            else:
                tqdm.write(f"    [error] {e}")
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

Format as a JSON array only (no markdown fences, no extra text):
[
  {{"question": "...", "answer": "```python\\n...\\n```"}},
  ...
]"""


def build_cmds_prompt(cls: dict) -> str:
    class_name = cls["class"]
    module = cls.get("module", "OpenMaya")
    description = cls.get("description", "")
    methods_text = _method_summary(cls, max_methods=6)

    # Map API 2.0 class to likely maya.cmds equivalents
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

Format as a JSON array only (no markdown fences, no extra text):
[
  {{"question": "...", "answer": "```python\\n...\\n```"}},
  ...
]"""


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

Format as a JSON array only (no markdown fences, no extra text):
[
  {{"question": "...", "answer": "```python\\n...\\n```"}},
  ...
]"""


PROMPT_BUILDERS = {
    "api2": build_api2_prompt,
    "cmds": build_cmds_prompt,
    "translation": build_translation_prompt,
}


# ── Response parsing ──────────────────────────────────────────────────────────

def _sanitize_json(text: str) -> str:
    """Fix common LLM JSON mistakes: bad escapes inside string values."""
    def fix_escapes(m: re.Match) -> str:
        s = m.group(0)
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        return s
    return re.sub(r'"(?:[^"\\]|\\.)*"', fix_escapes, text, flags=re.DOTALL)


def parse_response(text: str, class_name: str, module: str, variant: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []

    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group()

    items = None
    for candidate in [text, _sanitize_json(text)]:
        try:
            items = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if items is None:
        tqdm.write(f"    [parse fail] {class_name}:{variant} — could not decode JSON")
        return []

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
                print(f"    [syntax err] {e}")
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
            "_source": f"synthetic:{variant}:{module}.{class_name}",
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("Error: GOOGLE_AI_STUDIO_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run scrape.py first.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        api_data: list[dict] = json.load(f)

    log = load_request_log()
    today_count = get_daily_count(log)
    print(f"Loaded {len(api_data)} classes | Today's calls: {today_count}/{DAILY_REQUEST_HARD_CAP}")
    print(f"Variants: {PROMPT_VARIANTS} | {QA_PAIRS_PER_CLASS} pairs each")

    classes = [
        c for c in api_data
        if c["class"] != c.get("module")
        and any(m.get("type") == "method" for m in c.get("methods", []))
    ]
    print(f"Classes to process: {len(classes)} × {len(PROMPT_VARIANTS)} variants = "
          f"{len(classes) * len(PROMPT_VARIANTS)} total API calls\n")

    all_pairs: list[dict] = []
    cached_calls = 0
    new_calls = 0

    total_variants = len(classes) * len(PROMPT_VARIANTS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        bar = tqdm(
            total=total_variants,
            desc="Generating",
            unit="variant",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for cls in classes:
            class_name = cls["class"]
            module = cls.get("module", "OpenMaya")
            class_pairs: list[dict] = []
            variant_results: list[str] = []

            for variant in PROMPT_VARIANTS:
                cached = load_cache(class_name, variant)
                if cached is not None:
                    cached_calls += 1
                    class_pairs.extend(cached)
                    variant_results.append(f"{variant}:✓{len(cached)}")
                    bar.update(1)
                    bar.set_postfix_str(
                        f"{class_name} | today: {get_daily_count(log)}/{DAILY_REQUEST_HARD_CAP}",
                        refresh=True,
                    )
                    continue

                check_daily_cap(log)
                prompt = PROMPT_BUILDERS[variant](cls)
                bar.set_postfix_str(
                    f"{class_name}:{variant} | today: {get_daily_count(log)}/{DAILY_REQUEST_HARD_CAP}",
                    refresh=True,
                )
                raw = call_gemini(prompt, log)
                time.sleep(SLEEP_BETWEEN_CALLS)

                if raw is None:
                    save_cache(class_name, variant, [])
                    variant_results.append(f"{variant}:FAIL")
                    bar.update(1)
                    continue

                pairs = parse_response(raw, class_name, module, variant)
                save_cache(class_name, variant, pairs)
                class_pairs.extend(pairs)
                new_calls += 1
                variant_results.append(f"{variant}:{len(pairs)}")
                bar.update(1)
                bar.set_postfix_str(
                    f"{class_name} | today: {get_daily_count(log)}/{DAILY_REQUEST_HARD_CAP}",
                    refresh=True,
                )

            for pair in class_pairs:
                out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            all_pairs.extend(class_pairs)

            tqdm.write(f"  {class_name}: {' '.join(variant_results)}")

        bar.close()

    print(f"\nDone!")
    print(f"  Cached variant calls   : {cached_calls}")
    print(f"  New API calls          : {new_calls}")
    print(f"  Total synthetic pairs  : {len(all_pairs)}")
    print(f"  Saved to {OUTPUT_FILE}")

    # Merge api_pairs.jsonl + synthetic_pairs.jsonl → train.jsonl
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
    print(f"\nEstimated breakdown:")
    print(f"  API reference pairs  : ~9,023")
    print(f"  Synthetic (api2)     : ~{len(classes) * QA_PAIRS_PER_CLASS:,}")
    print(f"  Synthetic (cmds)     : ~{len(classes) * QA_PAIRS_PER_CLASS:,}")
    print(f"  Synthetic (transl.)  : ~{len(classes) * QA_PAIRS_PER_CLASS:,}")


if __name__ == "__main__":
    main()
