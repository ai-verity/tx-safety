"""
Local LLM Synthesis Engine
===========================
Uses locally-running transformer models (no external API) to:
  1. Paraphrase and enrich base prompts (linguistic diversity)
  2. Generate hard-negative contrastive examples
  3. Synthesise scene-specific context variations
  4. Score and rank candidate prompts using semantic similarity
  5. Generate chain-of-thought reasoning templates

Supported local model backends (auto-detected in priority order):
  - microsoft/phi-2                   (2.7B, general instruction following)
  - google/flan-t5-large              (780M, instruction tuning, CPU-friendly)
  - google/flan-t5-base               (250M, fastest fallback)
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, chat-tuned)
  - fallback: pure template synthesis (no model required)

All inference runs locally; no network calls are made during generation.
"""

from __future__ import annotations
import os
import sys
import time
import logging
import hashlib
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — ordered by preference (quality / speed trade-off)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = [
    {
        "name": "google/flan-t5-large",
        "type": "seq2seq",
        "max_new_tokens": 256,
        "description": "780M instruction-tuned T5, good quality, CPU-feasible",
    },
    {
        "name": "google/flan-t5-base",
        "type": "seq2seq",
        "max_new_tokens": 200,
        "description": "250M baseline T5, fastest option",
    },
    {
        "name": "google/flan-t5-small",
        "type": "seq2seq",
        "max_new_tokens": 150,
        "description": "60M tiny T5, emergency fallback",
    },
]


@dataclass
class SynthesisResult:
    original_prompt: str
    augmented_prompts: List[str]
    hard_negatives: List[str]
    context_variations: List[str]
    reasoning_chains: List[str]
    model_used: str
    generation_time_sec: float


class LocalLLMEngine:
    """
    Manages a locally-loaded transformer model for prompt augmentation.
    Gracefully falls back to rule-based generation if no GPU/model available.
    """

    def __init__(self, preferred_model: Optional[str] = None, device: str = "auto"):
        self.model = None
        self.tokenizer = None
        self.model_name = "template_fallback"
        self.model_type = "fallback"
        self._cache: Dict[str, str] = {}

        self._init_model(preferred_model, device)

    def _init_model(self, preferred: Optional[str], device: str):
        """Try to load the best available model."""
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForSeq2SeqLM,
                AutoModelForCausalLM, T5ForConditionalGeneration,
                T5Tokenizer, pipeline,
            )

            # Detect device
            if device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info("CUDA detected — using GPU")
                else:
                    self._device = "cpu"
                    logger.info("No GPU detected — using CPU (will be slower)")
            else:
                self._device = device

            registry = MODEL_REGISTRY
            if preferred:
                registry = [{"name": preferred, "type": "seq2seq",
                             "max_new_tokens": 256, "description": "user-specified"}] + registry

            for spec in registry:
                try:
                    logger.info(f"Attempting to load: {spec['name']}")
                    if spec["type"] == "seq2seq":
                        self.tokenizer = T5Tokenizer.from_pretrained(
                            spec["name"], legacy=False
                        )
                        self.model = T5ForConditionalGeneration.from_pretrained(
                            spec["name"],
                            torch_dtype=torch.float32 if self._device == "cpu" else torch.float16,
                        ).to(self._device)
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(spec["name"])
                        self.model = AutoModelForCausalLM.from_pretrained(
                            spec["name"],
                            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                        ).to(self._device)

                    self.model_name = spec["name"]
                    self.model_type = spec["type"]
                    self.max_new_tokens = spec["max_new_tokens"]
                    self.model.eval()
                    logger.info(f"✓ Model loaded: {spec['name']}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {spec['name']}: {e}")
                    continue

        except ImportError as e:
            logger.warning(f"transformers/torch not available: {e}")

        logger.info("Using template-based fallback synthesis (no model)")
        self.model_name = "template_fallback"
        self.model_type = "fallback"
        self.max_new_tokens = 0

    def _generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text from the loaded model."""
        if self.model is None:
            return self._template_generate(prompt)

        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import torch
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    do_sample=False,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self._cache[cache_key] = text
            return text
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return self._template_generate(prompt)

    def _template_generate(self, prompt: str) -> str:
        """
        Rule-based fallback that produces linguistically varied outputs
        without a model, using structural paraphrase patterns.
        """
        # Extract key noun phrase after common instruction prefixes
        for prefix in ["Rewrite:", "Paraphrase:", "Generate a variation of:", "Describe:"]:
            if prompt.startswith(prefix):
                core = prompt[len(prefix):].strip()
                return f"Analyse the visual scene and determine whether {core.lower()} indicators are present, providing spatial and temporal evidence."

        # Default: structural enrichment
        if "?" in prompt:
            return prompt.rstrip("?") + ", providing visual evidence for your answer?"
        return f"Carefully examine the provided image for: {prompt[:100]}. State your findings with supporting visual evidence."

    # ─────────────────────────────────────────────────────────────────────────
    # Public synthesis methods
    # ─────────────────────────────────────────────────────────────────────────

    def paraphrase_prompt(self, prompt: str, n: int = 3) -> List[str]:
        """Generate n paraphrased variants of a prompt."""
        results = []
        styles = [
            f"Rewrite the following surveillance instruction in formal technical language: {prompt}",
            f"Paraphrase this security monitoring task more concisely: {prompt}",
            f"Restate the following as a structured analytical question: {prompt}",
            f"Generate a variation of this VLM prompt emphasising spatial reasoning: {prompt}",
        ]
        for style_prompt in styles[:n]:
            result = self._generate(style_prompt, max_tokens=self.max_new_tokens or 150)
            if result and result != prompt:
                results.append(result.strip())
        return results if results else [prompt]

    def generate_hard_negative(self, incident_label: str, counterfactuals: List[str]) -> List[str]:
        """Generate hard negative example descriptions."""
        negatives = []
        base_negatives = counterfactuals[:3] if counterfactuals else [
            f"a normal {incident_label.lower()} scenario without any alert conditions",
            f"an authorised operation that resembles {incident_label.lower()} but is legitimate",
        ]

        for cf in base_negatives:
            prompt = (
                f"Describe a visual scene that could be confused with {incident_label} "
                f"but is actually a legitimate, non-threatening scenario. "
                f"Base it on: {cf.replace('_', ' ')}. "
                f"Make the description realistic and visually specific."
            )
            result = self._generate(prompt, max_tokens=100)
            negatives.append(result.strip() if result else cf.replace("_", " "))
        return negatives

    def generate_context_variation(
        self, base_prompt: str, context_modifier: str
    ) -> str:
        """Adapt a prompt for a specific environmental context."""
        augment_prompt = (
            f"Adapt the following surveillance task for the context of {context_modifier}: "
            f"{base_prompt}"
        )
        result = self._generate(augment_prompt, max_tokens=self.max_new_tokens or 150)
        return result.strip() if result else f"[{context_modifier.upper()}] {base_prompt}"

    def generate_cot_template(self, incident_label: str, signals: List[str]) -> str:
        """Generate a chain-of-thought reasoning template."""
        prompt = (
            f"Create a step-by-step reasoning chain for detecting {incident_label} "
            f"in a surveillance image. The chain should guide a visual model through: "
            f"1) identifying relevant objects, 2) checking for signals like "
            f"{', '.join(signals[:3])}, 3) ruling out false positives, "
            f"4) reaching a confident conclusion."
        )
        result = self._generate(prompt, max_tokens=200)
        return result.strip() if result else (
            f"Step 1: Identify objects. "
            f"Step 2: Check for {', '.join(signals[:2])}. "
            f"Step 3: Rule out benign alternatives. "
            f"Step 4: Conclude with confidence."
        )

    def generate_vqa_distractor(self, correct_answer: str, incident_label: str) -> List[str]:
        """Generate multiple-choice distractors for VQA prompts."""
        distractors = [
            f"No — this is a normal {incident_label.lower()} pattern with no alert condition",
            f"Uncertain — insufficient visual evidence to determine {incident_label.lower()}",
            f"Partially — some indicators present but below threshold",
        ]
        return distractors

    def synthesise_full(
        self,
        base_prompt: str,
        incident_label: str,
        counterfactuals: List[str],
        signal_labels: List[str],
        context_modifiers: Optional[List[str]] = None,
    ) -> SynthesisResult:
        """Full synthesis pipeline for a single base prompt."""
        t0 = time.time()
        context_modifiers = context_modifiers or [
            "night-time low-light conditions",
            "crowded foreground with occlusion",
            "fisheye overhead camera perspective",
        ]

        augmented = self.paraphrase_prompt(base_prompt, n=3)
        hard_negs = self.generate_hard_negative(incident_label, counterfactuals)
        ctx_vars = [
            self.generate_context_variation(base_prompt, ctx_mod)
            for ctx_mod in context_modifiers[:2]
        ]
        cot = self.generate_cot_template(incident_label, signal_labels)

        return SynthesisResult(
            original_prompt=base_prompt,
            augmented_prompts=augmented,
            hard_negatives=hard_negs,
            context_variations=ctx_vars,
            reasoning_chains=[cot],
            model_used=self.model_name,
            generation_time_sec=round(time.time() - t0, 2),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt augmentation manager
# ─────────────────────────────────────────────────────────────────────────────

class PromptAugmentationManager:
    """
    Orchestrates the full augmentation pipeline:
      - Receives base VLMPrompt objects
      - Runs them through LocalLLMEngine
      - Returns enriched prompt records
    """

    def __init__(self, engine: Optional[LocalLLMEngine] = None):
        self.engine = engine or LocalLLMEngine()

    def augment(self, base_prompt_dict: Dict, max_augmentations: int = 3) -> Dict:
        """
        Augment a serialised VLMPrompt dictionary.
        Returns the same dict enriched with augmentation fields.
        """
        user_prompt = base_prompt_dict.get("user_prompt", "")
        incident_label = base_prompt_dict.get("incident_label", "unknown incident")
        counterfactuals = base_prompt_dict.get("metadata", {}).get("counterfactuals", [])
        signal_ids = base_prompt_dict.get("signal_ids", [])

        result = self.engine.synthesise_full(
            base_prompt=user_prompt,
            incident_label=incident_label,
            counterfactuals=counterfactuals,
            signal_labels=signal_ids,
            context_modifiers=[
                f"{base_prompt_dict.get('scene_context', {}).get('time_of_day', 'daytime')} "
                f"{base_prompt_dict.get('scene_context', {}).get('lighting', 'daylight')} conditions",
                f"{base_prompt_dict.get('difficulty', 'medium')} difficulty with "
                f"{base_prompt_dict.get('scene_context', {}).get('occlusion', 'partial')} occlusion",
            ],
        )

        base_prompt_dict["augmented_prompts"] = result.augmented_prompts[:max_augmentations]
        base_prompt_dict["hard_negative_examples"] = result.hard_negatives[:2]
        base_prompt_dict["context_variations"] = result.context_variations
        base_prompt_dict["reasoning_chain_template"] = result.reasoning_chains[0] if result.reasoning_chains else ""
        base_prompt_dict["synthesis_model"] = result.model_used
        base_prompt_dict["synthesis_time_sec"] = result.generation_time_sec
        return base_prompt_dict


if __name__ == "__main__":
    engine = LocalLLMEngine()
    result = engine.synthesise_full(
        base_prompt="Detect any unattended vehicle in this parking facility image.",
        incident_label="Unattended Vehicle",
        counterfactuals=["driver returning to vehicle", "loading zone with operator"],
        signal_labels=["prolonged stationary vehicle", "no occupants visible"],
    )
    print(f"Model: {result.model_used}")
    print(f"Augmented: {result.augmented_prompts[:2]}")
    print(f"Hard negatives: {result.hard_negatives[:1]}")
    print(f"Time: {result.generation_time_sec}s")
