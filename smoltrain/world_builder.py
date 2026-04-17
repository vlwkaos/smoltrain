"""Build world.json from taxonomy.yaml using Claude API.

Usage:
    python -m smoltrain.world_builder
    python -m smoltrain.world_builder --taxonomy taxonomy.yaml --output data/world.json
"""
import argparse
import json
from pathlib import Path

import os

import yaml

# Auth: try ANTHROPIC_API_KEY first, then OpenRouter (OPENROUTER_API_KEY).
# OpenRouter exposes an OpenAI-compatible endpoint with model="anthropic/claude-sonnet-4-6".
def _make_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=api_key)
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if or_key:
        from openai import OpenAI
        client = OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1")
        return "openrouter", client
    raise RuntimeError(
        "No API key found. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY."
    )

# smoltrain/world_builder.py — generated world schema: data/world.json

WORLD_SCHEMA = """\
{
  "classes": {
    "<class_name>": {
      "templates": {
        "en": {
          "imperative": [...],
          "casual": [...],
          "frustrated": [...],
          "uncertain": [...],
          "indirect": [...],
          "implicit": [...],
          "fragment": [...]
        },
        "ko": { /* same 7 registers */ },
        "mixed": { /* 5+ code-switching examples per register, en+ko */ }
      },
      "slot_lists": {
        "<slot_name>": ["...", ...]
      },
      "situation_corpus": {
        "en": [...],
        "ko": [...]
      },
      "conversational_wrappers": {
        "en": [...],
        "ko": [...]
      },
      "cross_class_discriminators": {
        "vs_<other_class>": {
          "key_signals": [...],
          "confusion_patterns": [...]
        }
      },
      "length_profile": {
        "short_pct": 0.3,
        "medium_pct": 0.5,
        "long_pct": 0.2
      }
    }
  },
  "compound_bridges": {
    "en": [...],
    "ko": [...]
  },
  "noise_patterns": {
    "typo_rules": [...],
    "abbreviations": {...},
    "filler_words": {
      "en": [...],
      "ko": [...]
    }
  }
}"""

PROMPT_TEMPLATE = """\
You are generating a structured world object for a multilingual intent classification training system.

## Classes

{class_definitions}

## Languages

{languages}

## Requirements

Generate a complete world object in JSON following this schema exactly:

{schema}

### Quantity requirements per class:
- templates.en / templates.ko: each register needs at minimum:
  - imperative: 8+ examples
  - casual: 8+ examples
  - frustrated: 5+ examples
  - uncertain: 5+ examples
  - indirect: 5+ examples
  - implicit: 5+ examples
  - fragment: 5+ examples
- templates.mixed: 5+ code-switching examples per register
- slot_lists: each slot needs 30+ entries
- situation_corpus.en / .ko: 15+ entries each
- conversational_wrappers.en / .ko: 15+ entries each
- cross_class_discriminators: one entry per other class pair
- compound_bridges.en / .ko: 10+ entries each
- noise_patterns.typo_rules: 5+ entries
- noise_patterns.filler_words.en / .ko: 10+ entries each

### Critical quality requirements:

KOREAN TEMPLATES:
- Must be natively written Korean developer speech, NOT translated from English
- Reflect actual Korean developer communication style: mix of formal/informal, 해요체, 해라체, 반말 depending on context
- Include Konglish (English tech terms naturally embedded in Korean sentences)
- Examples should feel like real Slack messages or chat logs from a Korean dev team

CODE-SWITCHING (mixed):
- Reflect how bilingual developers actually write: natural, not forced
- English technical terms mid-Korean sentence is common and expected
- Korean particles/grammar around English words (e.g., "PR을 merge해줘", "bug fix하고 싶은데")
- Not 50/50 alternation — organic mixing based on what feels natural per context

TEMPLATES diversity:
- Cover the FULL range of how real users phrase these requests
- Different vocabulary, sentence structures, tones across examples within same register
- No two examples should be near-paraphrases of each other

DISCRIMINATORS must be SPECIFIC and PRECISE:
- BAD: "agentic requests are longer"
- GOOD: specific trigger words, verb patterns, presence of "then", "also", "and after that", multi-step conjunctions
- Include actual words/phrases that are diagnostic signals
- confusion_patterns: describe EXACTLY how this class is misclassified as the other (e.g., "short imperative commands that happen to involve file editing look agentic but are simple if they're one-shot")

SLOT LISTS:
- Generate realistic, diverse developer domain values
- Vary across different tech stacks, project types, language paradigms
- Include edge cases and unusual but valid entries

OUTPUT:
- Output ONLY valid JSON. No markdown fences, no explanation, no comments inside the JSON.
- The JSON must be parseable by Python's json.loads() directly.
- Replace all /* comment */ markers with actual content.
- Replace all [...] and {{...}} with actual arrays/objects.
"""


def build_prompt(taxonomy: dict) -> str:
    class_defs = []
    for name, cls in taxonomy["classes"].items():
        class_defs.append(f"- **{name}**: {cls['description']}")

    languages_str = ", ".join(taxonomy["languages"])

    return PROMPT_TEMPLATE.format(
        class_definitions="\n".join(class_defs),
        languages=languages_str,
        schema=WORLD_SCHEMA,
    )


REQUIRED_KEYS = {
    "classes",
    "compound_bridges",
    "noise_patterns",
}

REQUIRED_CLASS_KEYS = {
    "templates",
    "slot_lists",
    "situation_corpus",
    "conversational_wrappers",
    "cross_class_discriminators",
    "length_profile",
}

REQUIRED_REGISTERS = {
    "imperative", "casual", "frustrated", "uncertain", "indirect", "implicit", "fragment"
}


def validate_world(world: dict, taxonomy: dict) -> list[str]:
    errors = []

    for key in REQUIRED_KEYS:
        if key not in world:
            errors.append(f"missing top-level key: {key}")

    classes = world.get("classes", {})
    for class_name in taxonomy["classes"]:
        if class_name not in classes:
            errors.append(f"missing class: {class_name}")
            continue
        cls = classes[class_name]
        for key in REQUIRED_CLASS_KEYS:
            if key not in cls:
                errors.append(f"{class_name}: missing key '{key}'")

        templates = cls.get("templates", {})
        for lang in ["en", "ko", "mixed"]:
            if lang not in templates:
                errors.append(f"{class_name}.templates: missing language '{lang}'")
            else:
                for reg in REQUIRED_REGISTERS:
                    if reg not in templates[lang]:
                        errors.append(f"{class_name}.templates.{lang}: missing register '{reg}'")

    return errors


def print_summary(world: dict) -> None:
    classes = world.get("classes", {})
    print(f"\nWorld summary: {len(classes)} classes")
    for class_name, cls in classes.items():
        templates = cls.get("templates", {})
        counts = {}
        for lang in ["en", "ko", "mixed"]:
            lang_templates = templates.get(lang, {})
            total = sum(len(v) for v in lang_templates.values() if isinstance(v, list))
            counts[lang] = total
        slots = cls.get("slot_lists", {})
        slot_count = sum(len(v) for v in slots.values() if isinstance(v, list))
        print(
            f"  {class_name}: en={counts.get('en', 0)} ko={counts.get('ko', 0)} "
            f"mixed={counts.get('mixed', 0)} slot_entries={slot_count}"
        )

    bridges_en = len(world.get("compound_bridges", {}).get("en", []))
    bridges_ko = len(world.get("compound_bridges", {}).get("ko", []))
    print(f"  compound_bridges: en={bridges_en} ko={bridges_ko}")


def build_world(taxonomy_path: str = "taxonomy.yaml", output_path: str = "data/world.json") -> None:
    taxonomy_file = Path(taxonomy_path)
    if not taxonomy_file.exists():
        raise FileNotFoundError(f"taxonomy file not found: {taxonomy_path}")

    with taxonomy_file.open() as f:
        taxonomy = yaml.safe_load(f)

    print(f"Loaded taxonomy: {list(taxonomy['classes'].keys())}")

    prompt = build_prompt(taxonomy)
    provider, client = _make_client()
    print(f"Using provider: {provider}")

    print("Calling Claude API (this may take a moment)...")
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # ^ strip any accidental markdown fences from the model
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("```")) + 1
        end = next(
            (i for i, l in enumerate(lines[start:], start) if l.strip() == "```"),
            len(lines),
        )
        raw = "\n".join(lines[start:end])

    try:
        world = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"API response is not valid JSON: {e}\n\nRaw (first 500 chars):\n{raw[:500]}")

    errors = validate_world(world, taxonomy)
    if errors:
        print(f"Validation warnings ({len(errors)}):")
        for err in errors:
            print(f"  ! {err}")
    else:
        print("Validation passed.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(world, f, ensure_ascii=False, indent=2)

    print(f"Written: {out}")
    print_summary(world)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build world.json from taxonomy.yaml")
    parser.add_argument("--taxonomy", default="taxonomy.yaml")
    parser.add_argument("--output", default="data/world.json")
    args = parser.parse_args()
    build_world(args.taxonomy, args.output)


if __name__ == "__main__":
    main()
