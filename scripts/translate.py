"""
LLM Translation of Latin Inscriptions
======================================
Translates a Latin inscription into a target language using one of twelve LLM
models accessed through three commercial APIs (Google, Anthropic, OpenAI) and
one open-source hosting provider (DeepInfra via its OpenAI-compatible endpoint).

The ``MODELS`` list and ``translate_single_lang`` function are the shared
interface imported by ``evaluate_testset.py`` and ``compute_extra_metrics.py``.
"""

import os
import time

from google import genai
from google.genai import types
import anthropic
import openai

# ── All twelve models used in the experiment ──────────────────────────
# Commercial (direct API)
COMMERCIAL_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "gpt-5",
    "gpt-5-nano-2025-08-07",
]

# Open-source (hosted on DeepInfra, accessed via OpenAI-compatible API)
DEEPINFRA_MODELS = [
    "deepseek-v3",
    "llama-3.1-8b",
    "llama-3.1-70b",
    "qwen-2.5-72b",
    "apertus-8b",
    "apertus-70b",
]

MODELS = COMMERCIAL_MODELS + DEEPINFRA_MODELS

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


def _route(model):
    """Return (provider, api_key_env_var) for a model name."""
    if "gemini" in model:
        return "gemini", "GEMINI_API_KEY"
    if "claude" in model or "sonnet" in model:
        return "anthropic", "ANTHROPIC_API_KEY"
    if "gpt" in model:
        return "openai", "OPENAI_API_KEY"
    # Everything else goes through DeepInfra's OpenAI-compatible endpoint
    return "deepinfra", "DEEPINFRA_API_KEY"


def translate_single_lang(text, target_lang, model):
    """Translate ``text`` into ``target_lang`` using ``model``.

    Returns the translated string, or ``None`` on failure.
    """
    system_prompt = (
        f"You are an expert translator of Latin epigraphy. Translate the "
        f"following Latin inscription into strictly {target_lang}. "
        "Maintain the historical accuracy. Preserve the meaning of brackets "
        "[] () where they indicate missing or supplied letters.\n"
        "You MUST respond with ONLY the translation as plain text. "
        "Do not add conversational text or wrap in quotes. "
        "Make sure to ONLY translate."
    )

    provider, key_env = _route(model)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                api_key = os.environ.get(key_env)
                if not api_key:
                    return "ERROR: NO KEY"
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                    ),
                )
                return response.text.strip()

            elif provider == "anthropic":
                api_key = os.environ.get(key_env)
                if not api_key:
                    return "ERROR: NO KEY"
                client = anthropic.Anthropic(api_key=api_key, max_retries=0)
                response = client.messages.create(
                    model=model,
                    max_tokens=10000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}],
                    temperature=0.1,
                )
                return response.content[0].text.strip()

            elif provider in ("openai", "deepinfra"):
                api_key = os.environ.get(key_env)
                if not api_key:
                    return "ERROR: NO KEY"
                kwargs = {"api_key": api_key, "max_retries": 0}
                if provider == "deepinfra":
                    kwargs["base_url"] = DEEPINFRA_BASE_URL
                client = openai.OpenAI(**kwargs)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            err_msg = str(e).lower()
            if any(w in err_msg for w in ("rate limit", "429", "quota", "overloaded")):
                time.sleep(1 + (2 ** attempt))
            else:
                return None
    return None
