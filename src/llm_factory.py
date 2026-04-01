"""
LLM factory — builds a ChatHuggingFace instance from a local transformers pipeline.

Usage:
    from llm_factory import get_llm

    debate_llm = get_llm(max_new_tokens=256, temperature=0.8)
    vote_llm   = get_llm(max_new_tokens=512, temperature=0.4)

The underlying pipeline is a module-level singleton so the model is loaded
once per process regardless of how many agents are created.

Config is read from config/simulation.yaml (llm.model).
Override the model at runtime by setting the HF_MODEL env var:
    HF_MODEL=unsloth/Qwen3.5-0.8B-GGUF python src/main.py ...
"""

from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache

import yaml

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "simulation.yaml"

def _load_model_name() -> str:
    env = os.environ.get("HF_MODEL")
    if env:
        return env
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["llm"]["model"]


# ── Pipeline singleton ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_pipeline(model_name: str):
    """
    Load the model once and cache it for the lifetime of the process.
    Transformers >=4.45 loads GGUF repos directly — no manual model/tokenizer setup needed.
    """
    from transformers import pipeline

    print(f"[llm_factory] Loading model: {model_name}")
    pipe = pipeline("text-generation", model=model_name, return_full_text=False)
    print(f"[llm_factory] Model ready")
    return pipe


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm(max_new_tokens: int = 512, temperature: float = 0.7):
    """
    Return a LangChain ChatHuggingFace instance backed by the local pipeline.

    max_new_tokens: cap generation length (use 256 for debate, 512 for vote)
    temperature:    sampling temperature (0.4 for voting, 0.8 for debate)

    The returned object is a drop-in replacement for ChatAnthropic:
    it accepts [SystemMessage, HumanMessage] and returns an AIMessage.
    """
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    model_name = _load_model_name()
    raw_pipe = _get_pipeline(model_name)

    # Wrap pipeline with per-call generation kwargs
    hf_pipeline = HuggingFacePipeline(
        pipeline=raw_pipe,
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        },
    )
    return ChatHuggingFace(llm=hf_pipeline)


def get_raw_pipeline(max_new_tokens: int = 4096, temperature: float = 0.3):
    """
    Return the raw transformers pipeline for use in generate_profiles.py,
    which calls the model directly rather than through LangChain.
    """
    model_name = _load_model_name()
    raw_pipe = _get_pipeline(model_name)
    return raw_pipe, max_new_tokens, temperature
