"""
build_dataset.py

Converts maya_api_2_raw.json into a chat-format JSONL file (api_pairs.jsonl).
Generates API reference question-answer pairs from the structured scrape data.
This is the "ground truth" dataset from the docs — no LLM calls needed.

Run: uv run python build_dataset.py
Output: api_pairs.jsonl
"""

import json
import sys
from pathlib import Path

INPUT_FILE = Path("maya_api_2_raw.json")
OUTPUT_FILE = Path("api_pairs.jsonl")

SYSTEM_PROMPT = (
    "You are an expert Maya Python developer with deep knowledge of both "
    "maya.cmds and the Maya API 2.0 (maya.api.OpenMaya). "
    "Provide accurate, concise answers about Maya Python API methods, classes, "
    "and best practices."
)


def fmt_params(parameters: list[dict]) -> str:
    """Format a parameter list as a readable string."""
    if not parameters:
        return "None"
    lines = []
    for p in parameters:
        name = p.get("name", "?")
        typ = p.get("type") or ""
        desc = p.get("description") or ""
        if typ and desc:
            lines.append(f"  - {name} ({typ}): {desc}")
        elif typ:
            lines.append(f"  - {name} ({typ})")
        elif desc:
            lines.append(f"  - {name}: {desc}")
        else:
            lines.append(f"  - {name}")
    return "\n".join(lines)


def make_message(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def pairs_for_method(class_name: str, module: str, method: dict) -> list[dict]:
    """Generate one or more training pairs for a single method entry."""
    pairs: list[dict] = []
    name = method.get("name", "")
    item_type = method.get("type", "method")
    description = (method.get("description") or "").strip()
    signature = (method.get("signature") or "").strip()
    return_type = (method.get("return_type") or "").strip()
    parameters = method.get("parameters") or []
    is_static = method.get("is_static", False)

    # ── Constants: simple value lookup ──
    if item_type == "constant":
        if not description:
            return []
        user = f"What is {class_name}.{name} in Maya API 2.0?"
        assistant = (
            f"`{class_name}.{name}` is a constant in `{module}.{class_name}`.\n"
            f"Value type: {method.get('value_type') or 'Int'}\n"
            f"Description: {description}"
        )
        pairs.append(make_message(user, assistant))
        return pairs

    # ── Properties ──
    if item_type == "property":
        if not description:
            return []
        access = method.get("access") or "RW"
        value_type = method.get("value_type") or ""
        user = f"What is the `{name}` property on `{class_name}` in Maya API 2.0?"
        assistant = (
            f"`{module}.{class_name}.{name}` is a {access} property.\n"
            f"Type: {value_type}\n"
            f"Description: {description}"
        )
        pairs.append(make_message(user, assistant))
        return pairs

    # ── Methods ──
    if not description and not signature:
        return []

    # Pair 1: "What does X do?" — doc lookup
    if description:
        user = f"What does `{class_name}.{name}()` do in Maya API 2.0?"
        lines = [f"`{module}.{class_name}.{name}` — {description}"]
        if signature:
            lines.append(f"\nSignature: `{signature}`")
        if return_type:
            lines.append(f"Returns: {return_type}")
        if parameters:
            lines.append(f"Parameters:\n{fmt_params(parameters)}")
        if is_static:
            lines.append("\nNote: This is a static method.")
        pairs.append(make_message(user, "\n".join(lines)))

    # Pair 2: "What arguments does X take?" — parameter focus
    if parameters:
        user = f"What arguments does `{class_name}.{name}()` accept in Maya API 2.0?"
        if signature:
            answer = f"Signature: `{signature}`\n\nParameters:\n{fmt_params(parameters)}"
        else:
            answer = f"`{class_name}.{name}()` parameters:\n{fmt_params(parameters)}"
        if return_type:
            answer += f"\n\nReturns: {return_type}"
        pairs.append(make_message(user, answer))

    # Pair 3: "What does X return?" — return type focus
    if return_type and return_type.lower() not in ("none", "none.", ""):
        user = f"What does `{class_name}.{name}()` return in Maya API 2.0?"
        answer = f"`{module}.{class_name}.{name}()` returns: **{return_type}**"
        if description:
            answer += f"\n\n{description}"
        pairs.append(make_message(user, answer))

    return pairs


def pairs_for_class(cls: dict) -> list[dict]:
    """Generate class-level and method-level pairs for one class entry."""
    pairs: list[dict] = []
    class_name = cls.get("class", "")
    module = cls.get("module", "OpenMaya")
    description = (cls.get("description") or "").strip()

    # Skip module-level index pages (no real methods)
    if class_name == module:
        return []

    # Class overview pair
    if description and description != "No description available.":
        user = f"What is `{class_name}` in Maya API 2.0?"
        methods_list = [
            m["name"]
            for m in cls.get("methods", [])
            if m.get("type") == "method" and not m["name"].startswith("__")
        ]
        answer = f"`{module}.{class_name}` — {description}"
        if methods_list:
            shown = methods_list[:15]
            answer += f"\n\nKey methods: {', '.join(shown)}"
            if len(methods_list) > 15:
                answer += f" ... ({len(methods_list) - 15} more)"
        pairs.append(make_message(f"Tell me about `{class_name}` in Maya Python API 2.0.", answer))

    # Per-method pairs
    for method in cls.get("methods", []):
        pairs.extend(pairs_for_method(class_name, module, method))

    return pairs


def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run scrape.py first.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        api_data: list[dict] = json.load(f)

    print(f"Loaded {len(api_data)} classes from {INPUT_FILE}")

    total_pairs = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for cls in api_data:
            pairs = pairs_for_class(cls)
            for pair in pairs:
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)

    print(f"Written {total_pairs} training pairs to {OUTPUT_FILE}")
    print(f"Average: {total_pairs / max(len(api_data), 1):.1f} pairs per class")


if __name__ == "__main__":
    main()
