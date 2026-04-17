"""Procedural expansion engine: world.json + taxonomy.yaml -> dataset JSONL.

Usage:
    python -m smoltrain.expander
    python -m smoltrain.expander --world data/world.json --taxonomy taxonomy.yaml --output data/dataset.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONV_TYPE_TARGETS = [
    ("A", 0.25),
    ("B", 0.20),
    ("C", 0.20),
    ("D", 0.15),
    ("E", 0.10),
    ("F", 0.10),
]
TRANSFORM_TYPES = {"A", "F"}
TRANSFORM_PROB = 0.30
NOISE_PROB = 0.20
BASE_LANG_WEIGHTS = {"en": 0.40, "ko": 0.40, "mixed": 0.20}

# ---------------------------------------------------------------------------
# Slot filling
# ---------------------------------------------------------------------------

def fill_slots(text: str, slot_lists: dict) -> str:
    def replace(m):
        slot = m.group(1)
        choices = slot_lists.get(slot)
        if choices:
            return random.choice(choices)
        return m.group(0)
    return re.sub(r"\{(\w+)\}", replace, text)

# ---------------------------------------------------------------------------
# Language selection
# ---------------------------------------------------------------------------

def pick_lang(cls_data: dict, base_languages: list) -> str:
    """Pick language weighted by availability. Adds 'mixed' if templates exist."""
    cls_templates = cls_data.get("templates", {})
    candidates = list(base_languages)
    if "mixed" not in candidates and "mixed" in cls_templates:
        candidates.append("mixed")

    available = []
    weights = []
    for lang in candidates:
        lang_t = cls_templates.get(lang, {})
        if any(isinstance(v, list) and len(v) > 0 for v in lang_t.values()):
            available.append(lang)
            weights.append(BASE_LANG_WEIGHTS.get(lang, 0.10))

    if not available:
        return base_languages[0] if base_languages else "en"

    total = sum(weights)
    normalized = [w / total for w in weights]
    return random.choices(available, weights=normalized, k=1)[0]

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def _flat_templates(cls_data: dict, lang: str) -> list:
    templates = cls_data.get("templates", {}).get(lang, {})
    return [t for pool in templates.values() if isinstance(pool, list) for t in pool]


def _register_templates(cls_data: dict, lang: str) -> list:
    """Return list of (register, template) pairs for all non-empty registers."""
    templates = cls_data.get("templates", {}).get(lang, {})
    return [
        (reg, t)
        for reg, pool in templates.items()
        if isinstance(pool, list)
        for t in pool
    ]


def _pick_any(cls_data: dict, lang: str):
    """Return (text, register) from any register."""
    pairs = _register_templates(cls_data, lang)
    if not pairs:
        return "", "unknown"
    reg, text = random.choice(pairs)
    return fill_slots(text, cls_data.get("slot_lists", {})), reg


def _pick_register(cls_data: dict, lang: str, register: str):
    """Return text from a specific register, fallback to any."""
    templates = cls_data.get("templates", {}).get(lang, {})
    pool = templates.get(register, [])
    if pool:
        return fill_slots(random.choice(pool), cls_data.get("slot_lists", {}))
    # fallback
    flat = _flat_templates(cls_data, lang)
    if flat:
        return fill_slots(random.choice(flat), cls_data.get("slot_lists", {}))
    return ""

# ---------------------------------------------------------------------------
# Conversation type generators
# ---------------------------------------------------------------------------

_CONNECTORS = ["{s}. {t}", "{s}, so {t}", "{s} — {t}"]


def gen_type_a(cls_name: str, cls_data: dict, lang: str):
    text, reg = _pick_any(cls_data, lang)
    return text, reg


def gen_type_b(cls_name: str, cls_data: dict, lang: str):
    text, reg = _pick_any(cls_data, lang)
    situations = cls_data.get("situation_corpus", {}).get(lang, [])
    if situations:
        sit = random.choice(situations)
        connector = random.choice(_CONNECTORS)
        text = connector.format(s=sit, t=text)
    return text, reg


def gen_type_c(cls_name: str, cls_data: dict, lang: str):
    text, reg = _pick_any(cls_data, lang)
    wrappers = cls_data.get("conversational_wrappers", {}).get(lang, [])
    if wrappers:
        wrapper = random.choice(wrappers)
        if "[INTENT]" in wrapper:
            text = wrapper.replace("[INTENT]", text)
        else:
            text = f"{wrapper}, {text}"
    return text, reg


def gen_type_d(cls_name: str, cls_data: dict, lang: str):
    text = _pick_register(cls_data, lang, "fragment")
    reg = "fragment" if text else "unknown"
    if not text:
        text, reg = _pick_any(cls_data, lang)

    situations = cls_data.get("situation_corpus", {}).get(lang, [])
    if situations:
        sit = random.choice(situations)
        prefix = " ".join(sit.split()[:15])
        text = f"{prefix} — {text}"
    return text, reg


def gen_type_e(cls_name: str, cls_data: dict, lang: str, all_classes: dict, compound_bridges: dict):
    bridges = compound_bridges.get(lang, [])
    if not bridges:
        return gen_type_a(cls_name, cls_data, lang)

    bridge = random.choice(bridges)

    flat_a = _flat_templates(cls_data, lang)
    intent_a = fill_slots(
        random.choice(flat_a) if flat_a else cls_name,
        cls_data.get("slot_lists", {})
    )

    other_names = [c for c in all_classes if c != cls_name]
    if other_names:
        sec_name = random.choice(other_names)
        sec_data = all_classes[sec_name]
        flat_b = _flat_templates(sec_data, lang)
        intent_b = fill_slots(
            random.choice(flat_b) if flat_b else sec_name,
            sec_data.get("slot_lists", {})
        )
    else:
        intent_b = cls_name

    if "[INTENT_A]" in bridge and "[INTENT_B]" in bridge:
        text = bridge.replace("[INTENT_A]", intent_a).replace("[INTENT_B]", intent_b)
    else:
        text = f"{intent_a} {bridge} {intent_b}"

    return text, "compound"


def gen_type_f(cls_name: str, cls_data: dict, lang: str):
    text = _pick_register(cls_data, lang, "implicit")
    if not text:
        text, _ = _pick_any(cls_data, lang)

    situations = cls_data.get("situation_corpus", {}).get(lang, [])
    if situations and random.random() < 0.5:
        sit = random.choice(situations)
        prefix = " ".join(sit.split()[:8])
        text = f"{prefix} — {text}"
    return text, "implicit"

# ---------------------------------------------------------------------------
# Syntactic transforms
# ---------------------------------------------------------------------------

def _to_question(text: str) -> str:
    text = text.rstrip("?.,! ")
    prefix = random.choice(["can you ", "could you ", ""])
    return f"{prefix}{text}?"


def _to_passive(text: str) -> str:
    if random.random() < 0.5:
        return f"this needs to be {text}"
    return f"{text} — can someone handle this"


def _add_prefix(text: str, fillers: list) -> str:
    if not fillers:
        return text
    return f"{random.choice(fillers)} {text}"


def _add_suffix(text: str) -> str:
    return f"{text} {random.choice(['for me', 'if possible', 'when you get a chance', 'asap'])}"


def _to_formal(text: str) -> str:
    return f"{random.choice(['please ', 'would you mind '])}{text}"


def apply_transform(text: str, fillers: list) -> str:
    fns = [_to_question, _to_passive, _add_suffix, _to_formal]
    if fillers:
        fns.append(lambda t: _add_prefix(t, fillers))
    return random.choice(fns)(text)

# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def _abbrev_sub(text: str, abbreviations: dict) -> str:
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in abbreviations and random.random() < 0.05:
            result.append(random.choice(abbreviations[lower]))
        else:
            result.append(word)
    return " ".join(result)


def _typo_inject(text: str) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 4]
    if not candidates:
        return text
    idx = random.choice(candidates)
    chars = list(words[idx])
    pos = random.randint(0, len(chars) - 2)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    words[idx] = "".join(chars)
    return " ".join(words)


def _filler_inject(text: str, fillers: list) -> str:
    if not fillers:
        return text
    words = text.split()
    pos = random.randint(0, len(words))
    words.insert(pos, random.choice(fillers))
    return " ".join(words)


def apply_noise(text: str, lang: str, noise_patterns: dict) -> str:
    abbreviations = noise_patterns.get("abbreviations", {})
    fillers = noise_patterns.get("filler_words", {}).get(lang, [])

    fns = [_typo_inject]
    if abbreviations:
        fns.append(lambda t: _abbrev_sub(t, abbreviations))
    if fillers:
        fns.append(lambda t: _filler_inject(t, fillers))

    return random.choice(fns)(text)

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

_CONV_TYPES = [t for t, _ in CONV_TYPE_TARGETS]
_CONV_WEIGHTS = [w for _, w in CONV_TYPE_TARGETS]


def generate_dataset(world: dict, config: dict) -> list[dict]:
    samples_per_class = config.get("samples_per_class", 500)
    languages = config.get("languages", ["en"])

    all_classes = world["classes"]
    compound_bridges = world.get("compound_bridges", {})
    noise_patterns = world.get("noise_patterns", {})

    records = []

    for cls_name, cls_data in all_classes.items():
        for _ in range(samples_per_class):
            lang = pick_lang(cls_data, languages)
            conv_type = random.choices(_CONV_TYPES, weights=_CONV_WEIGHTS, k=1)[0]

            if conv_type == "A":
                text, register = gen_type_a(cls_name, cls_data, lang)
            elif conv_type == "B":
                text, register = gen_type_b(cls_name, cls_data, lang)
            elif conv_type == "C":
                text, register = gen_type_c(cls_name, cls_data, lang)
            elif conv_type == "D":
                text, register = gen_type_d(cls_name, cls_data, lang)
            elif conv_type == "E":
                text, register = gen_type_e(cls_name, cls_data, lang, all_classes, compound_bridges)
            else:  # F
                text, register = gen_type_f(cls_name, cls_data, lang)

            if conv_type in TRANSFORM_TYPES and random.random() < TRANSFORM_PROB:
                fillers = noise_patterns.get("filler_words", {}).get(lang, [])
                text = apply_transform(text, fillers)

            noised = False
            if random.random() < NOISE_PROB:
                text = apply_noise(text, lang, noise_patterns)
                noised = True

            records.append({
                "text": text,
                "label": cls_name,
                "lang": lang,
                "conv_type": conv_type,
                "register": register,
                "noised": noised,
            })

    random.shuffle(records)
    return records

# ---------------------------------------------------------------------------
# Stats / summary
# ---------------------------------------------------------------------------

def print_summary(records: list[dict]) -> None:
    from collections import Counter
    total = len(records)
    by_label = Counter(r["label"] for r in records)
    by_lang = Counter(r["lang"] for r in records)
    by_type = Counter(r["conv_type"] for r in records)
    noised = sum(1 for r in records if r["noised"])

    print(f"\nDataset: {total} samples")
    print("  by label:    " + "  ".join(f"{k}={v}" for k, v in sorted(by_label.items())))
    print("  by lang:     " + "  ".join(f"{k}={v}" for k, v in sorted(by_lang.items())))
    print("  by conv type:" + "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"  noised: {noised} ({100*noised/total:.1f}%)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def expand(world_path: str, taxonomy_path: str, output_path: str) -> None:
    world_file = Path(world_path)
    taxonomy_file = Path(taxonomy_path)

    if not world_file.exists():
        raise FileNotFoundError(f"world file not found: {world_path}")
    if not taxonomy_file.exists():
        raise FileNotFoundError(f"taxonomy file not found: {taxonomy_path}")

    with world_file.open(encoding="utf-8") as f:
        world = json.load(f)

    with taxonomy_file.open() as f:
        taxonomy = yaml.safe_load(f)

    config = {
        "samples_per_class": taxonomy.get("config", {}).get("samples_per_class", 500),
        "languages": taxonomy.get("languages", ["en"]),
    }

    print(f"Classes: {list(world['classes'].keys())}")
    print(f"Languages: {config['languages']}")
    print(f"Samples per class: {config['samples_per_class']}")

    records = generate_dataset(world, config)
    print_summary(records)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWritten: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand world.json into dataset JSONL")
    parser.add_argument("--world", default="data/world.json")
    parser.add_argument("--taxonomy", default="taxonomy.yaml")
    parser.add_argument("--output", default="data/dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    expand(args.world, args.taxonomy, args.output)


if __name__ == "__main__":
    main()
