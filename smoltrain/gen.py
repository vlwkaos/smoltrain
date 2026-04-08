"""Generate training data using a supervisor LLM."""
import json
import os
import subprocess
from pathlib import Path
from collections import Counter

import requests

from . import config as cfg_mod


# ---------------------------------------------------------------------------
# Auth / endpoint helpers
# ---------------------------------------------------------------------------

def _get_claude_oauth_token():
    """Try macOS keychain, then ANTHROPIC_TOKEN env var."""
    # 1. macOS keychain
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            raw = result.stdout.strip()
            data = json.loads(raw)
            token = data.get("claudeAiOauth", {}).get("accessToken")
            if token:
                return token
    except Exception:
        pass

    # 2. Env var
    return os.environ.get("ANTHROPIC_TOKEN")


def _build_client(supervisor: str, supervisor_model: str):
    """Return (call_fn) where call_fn(prompt) -> str."""

    if supervisor == "claude-oauth":
        token = _get_claude_oauth_token()
        if token:
            return _make_anthropic_bearer_client(token, supervisor_model)
        # Fallback: ANTHROPIC_API_KEY
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            return _make_anthropic_key_client(api_key, supervisor_model)
        raise RuntimeError(
            "claude-oauth: no token found in keychain, ANTHROPIC_TOKEN, or ANTHROPIC_API_KEY"
        )

    if supervisor == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return _make_anthropic_key_client(api_key, supervisor_model)

    if supervisor == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        return _make_openai_compat_client(
            "https://api.groq.com/openai/v1/chat/completions",
            api_key, supervisor_model
        )

    if supervisor == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return _make_openai_compat_client(
            "https://api.openai.com/v1/chat/completions",
            api_key, supervisor_model
        )

    if supervisor.startswith("local:"):
        url = supervisor[len("local:"):]
        return _make_openai_compat_client(url, None, supervisor_model)

    raise ValueError(f"Unknown supervisor: {supervisor!r}")


def _make_anthropic_bearer_client(token: str, model: str):
    def call(prompt: str) -> str:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    return call


def _make_anthropic_key_client(api_key: str, model: str):
    def call(prompt: str) -> str:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    return call


def _make_openai_compat_client(url: str, api_key, model: str):
    def call(prompt: str) -> str:
        headers = {"content-type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return call


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(cls: str, n: int, cfg) -> str:
    all_classes = cfg.classes
    descs = cfg.class_descriptions

    if descs:
        all_with_desc = ", ".join(
            f'"{c}" ({descs[c]})' if c in descs else f'"{c}"'
            for c in all_classes
        )
        other_with_desc = ", ".join(
            f'"{c}" ({descs[c]})' if c in descs else f'"{c}"'
            for c in all_classes if c != cls
        )
        desc_line = descs.get(cls, "")
        return (
            f'Generate {n} diverse, realistic examples for label "{cls}".\n'
            f"Task: {cfg.goal}\n"
            f"Class definition: {desc_line}\n"
            f"All labels: {all_with_desc}\n"
            f"Rules: one per line, plain text, no numbering, varied length.\n"
            f'Focus on boundary cases — clearly "{cls}", NOT: {other_with_desc}\n'
            f"Output exactly {n} lines:"
        )
    else:
        all_labels = ", ".join(f'"{c}"' for c in all_classes)
        return (
            f'Generate {n} diverse, realistic examples for label "{cls}".\n'
            f"Task: {cfg.goal}\n"
            f"All labels: {all_labels}\n"
            f'Rules: one per line, plain text, no numbering, varied length, clearly belongs to "{cls}".\n'
            f"Output exactly {n} lines:"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(cfg):
    ds_path = cfg_mod.dataset_path(cfg.name)
    ds_path.parent.mkdir(parents=True, exist_ok=True)

    # Count existing examples per class
    existing: Counter = Counter()
    if ds_path.exists():
        with open(ds_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        existing[obj["label"]] += 1
                    except Exception:
                        pass

    call = _build_client(cfg.supervisor, cfg.supervisor_model)

    with open(ds_path, "a") as out:
        for cls in cfg.classes:
            have = existing.get(cls, 0)
            need = cfg.n_per_class - have
            if need <= 0:
                print(f"  {cls}: already have {have} examples, skipping")
                continue

            print(f"  {cls}: have {have}, generating {need} more...")
            prompt = _build_prompt(cls, need, cfg)
            try:
                text = call(prompt)
            except Exception as e:
                print(f"  ERROR generating for {cls}: {e}")
                continue

            lines = [l.strip() for l in text.splitlines() if l.strip()]
            # Remove leading numbering/bullets if model added them
            cleaned = []
            for l in lines:
                # strip "1. " or "- " prefixes
                import re
                l = re.sub(r"^\d+\.\s+", "", l)
                l = re.sub(r"^[-*]\s+", "", l)
                if l:
                    cleaned.append(l)

            written = 0
            for example in cleaned[:need]:
                obj = {"text": example, "label": cls}
                out.write(json.dumps(obj) + "\n")
                written += 1

            print(f"  {cls}: wrote {written} examples")

    print(f"Dataset saved to {ds_path}")
