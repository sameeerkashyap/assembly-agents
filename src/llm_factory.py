"""
LLM factory — returns a LangChain chat model for use across all agents.

Priority order (first available wins):
  1. ANTHROPIC_API_KEY env var  → ChatAnthropic (claude-haiku-4-5 for debate, sonnet for votes)
  2. HF_MODEL env var           → local HuggingFace model on MPS (Apple Silicon) or CPU
  3. config/simulation.yaml     → local HuggingFace model from config

Apple Silicon (MPS) is used automatically for local models — no manual setup needed.
The underlying HF pipeline is a singleton so the model loads once per process.

Usage:
    from llm_factory import get_llm, get_llm_for_votes

    debate_llm = get_llm()                   # fast/cheap — haiku or local 3B
    vote_llm   = get_llm_for_votes()         # higher quality — sonnet or local 3B at low temp
"""

from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "simulation.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _use_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _get_anthropic_llm(model: str, max_tokens: int, temperature: float):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model_name=model, max_tokens_to_sample=max_tokens, temperature=temperature, timeout=None, stop=None)


# ── Local HuggingFace on MPS ──────────────────────────────────────────────────

def _load_model_name() -> str:
    env = os.environ.get("HF_MODEL")
    if env:
        return env
    return _load_config()["llm"]["model"]


@lru_cache(maxsize=1)
def _get_pipeline(model_name: str):
    """Load model once, use MPS on Apple Silicon, fall back to CPU."""
    from transformers import pipeline
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[llm_factory] Loading {model_name} on {device}")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=device,
        return_full_text=False,
        truncation=True,
        torch_dtype=torch.float16,  # halves VRAM/RAM usage vs float32
    )
    print("[llm_factory] Model ready")
    return pipe


def _get_hf_llm(max_new_tokens: int, temperature: float):
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    raw_pipe = _get_pipeline(_load_model_name())
    hf_pipeline = HuggingFacePipeline(
        pipeline=raw_pipe,
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        },
    )
    return ChatHuggingFace(llm=hf_pipeline)


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm(max_new_tokens: int = 256, temperature: float = 0.8):
    """
    Debate LLM — fast and cheap.
      Anthropic: claude-haiku-4-5
      Local:     configured model on MPS, float16, 256 tokens
    """
    if _use_anthropic():
        return _get_anthropic_llm("claude-haiku-4-5-20251001", max_tokens=max_new_tokens, temperature=temperature)
    return _get_hf_llm(max_new_tokens=max_new_tokens, temperature=temperature)


def get_llm_for_votes(max_new_tokens: int = 512, temperature: float = 0.4):
    """
    Vote reasoning LLM — higher quality, used for official vote records.
      Anthropic: claude-sonnet-4-6
      Local:     same model as debate but lower temperature, more tokens
    """
    if _use_anthropic():
        return _get_anthropic_llm("claude-sonnet-4-6", max_tokens=max_new_tokens, temperature=temperature)
    return _get_hf_llm(max_new_tokens=max_new_tokens, temperature=temperature)


def get_raw_pipeline(max_new_tokens: int = 4096, temperature: float = 0.3):
    """
    Raw transformers pipeline for generate_profiles.py.
    Returns (pipeline, max_new_tokens, temperature).
    """
    return _get_pipeline(_load_model_name()), max_new_tokens, temperature
