"""
VLM Prompt Generator
=====================
Main orchestration script.  Runs the full pipeline:

  1. Load & validate taxonomy
  2. Build base prompts via PromptBuilder (all incident × task × style × difficulty)
  3. Augment via LocalLLMEngine (paraphrase, hard-negatives, CoT)
  4. Score, rank and deduplicate
  5. Export to structured JSON with full metadata

Usage:
  python generate_prompts.py [--output PATH] [--max-per-incident N]
                              [--min-quality FLOAT] [--augment] [--seed INT]
                              [--model MODEL_NAME]
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from taxonomy import (
    SURVEILLANCE_DOMAIN, INCIDENT_CLASSES, INCIDENT_TYPES,
    AnnotationTask, Severity, get_all_incident_types,
)
from framework import (
    PromptBuilder, SceneContext, DifficultyLevel, PromptStyle,
    LocationType, CameraAngle, LightingCondition, OcclusionLevel,
    TemplateLibrary,
)
from llm_engine import LocalLLMEngine, PromptAugmentationManager

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "max_per_incident": 20,
    "min_quality": 0.25,
    "augment_with_llm": True,
    "max_augmentations": 3,
    "seed": 42,
    "output_path": "/home/claude/vlm_framework/output/vlm_prompts_generated.json",
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics tracker
# ─────────────────────────────────────────────────────────────────────────────

class GenerationStats:
    def __init__(self):
        self.total_generated = 0
        self.total_filtered = 0
        self.total_augmented = 0
        self.by_incident: Dict[str, int] = {}
        self.by_task: Dict[str, int] = {}
        self.by_severity: Dict[str, int] = {}
        self.quality_distribution: List[float] = []
        self.start_time = time.time()

    def record(self, prompt_dict: Dict):
        self.total_generated += 1
        inc_id = prompt_dict["incident_type_id"]
        self.by_incident[inc_id] = self.by_incident.get(inc_id, 0) + 1
        task = prompt_dict["annotation_task"]
        self.by_task[task] = self.by_task.get(task, 0) + 1
        sev = prompt_dict["severity"]
        self.by_severity[sev] = self.by_severity.get(sev, 0) + 1
        self.quality_distribution.append(prompt_dict["quality_scores"]["composite"])

    def summary(self) -> Dict:
        elapsed = round(time.time() - self.start_time, 1)
        qdist = self.quality_distribution
        return {
            "total_prompts": self.total_generated,
            "total_filtered_out": self.total_filtered,
            "total_augmented": self.total_augmented,
            "elapsed_sec": elapsed,
            "by_incident_type": dict(sorted(self.by_incident.items())),
            "by_annotation_task": dict(sorted(self.by_task.items())),
            "by_severity": dict(sorted(self.by_severity.items())),
            "quality_stats": {
                "mean": round(sum(qdist) / len(qdist), 4) if qdist else 0,
                "min": round(min(qdist), 4) if qdist else 0,
                "max": round(max(qdist), 4) if qdist else 0,
                "above_0_5": sum(1 for q in qdist if q >= 0.5),
                "above_0_7": sum(1 for q in qdist if q >= 0.7),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

class VLMPromptGenerator:
    def __init__(self, config: Dict):
        self.config = {**DEFAULT_CONFIG, **config}
        random.seed(self.config["seed"])
        self.builder = PromptBuilder(seed=self.config["seed"])
        self.stats = GenerationStats()

        if self.config["augment_with_llm"]:
            logger.info("Initialising local LLM engine...")
            engine = LocalLLMEngine(
                preferred_model=self.config.get("model"),
                device="auto",
            )
            self.augmentor = PromptAugmentationManager(engine)
        else:
            self.augmentor = None

    # ── Base prompt generation ────────────────────────────────────────────────

    def _generate_base_prompts(self) -> List[Dict]:
        all_prompts = []
        incident_types = get_all_incident_types()
        logger.info(f"Generating prompts for {len(incident_types)} incident types...")

        # Build class_id lookup
        class_lookup: Dict[str, str] = {}
        for cls in INCIDENT_CLASSES.values():
            for it in cls.incident_types:
                class_lookup[it.type_id] = cls.class_id

        for it in incident_types:
            class_id = class_lookup.get(it.type_id, "ic_unknown")
            prompts = self.builder.build_all_combinations(
                it, class_id=class_id,
                max_per_incident=self.config["max_per_incident"],
            )

            filtered = [
                p for p in prompts
                if p.composite_score() >= self.config["min_quality"]
            ]
            self.stats.total_filtered += (len(prompts) - len(filtered))

            for p in filtered:
                d = p.to_dict()
                self.stats.record(d)
                all_prompts.append(d)

            logger.info(f"  [{it.type_id}] {it.label}: {len(filtered)} prompts (quality >= {self.config['min_quality']})")

        return all_prompts

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment_prompts(self, prompts: List[Dict]) -> List[Dict]:
        if not self.augmentor:
            return prompts

        logger.info(f"Augmenting {len(prompts)} prompts with local LLM...")
        augmented = []
        for i, p in enumerate(prompts):
            if i % 20 == 0:
                logger.info(f"  Augmenting {i}/{len(prompts)}...")
            try:
                enriched = self.augmentor.augment(
                    p, max_augmentations=self.config["max_augmentations"]
                )
                augmented.append(enriched)
                self.stats.total_augmented += 1
            except Exception as e:
                logger.warning(f"Augmentation failed for {p.get('prompt_id', '?')}: {e}")
                augmented.append(p)
        return augmented

    # ── Diversity enforcement ─────────────────────────────────────────────────

    def _enforce_diversity(self, prompts: List[Dict]) -> List[Dict]:
        """
        Ensure balanced representation across:
          - Severity levels (at least 10% each for LOW-CRITICAL)
          - Annotation tasks (at least 5 per task)
          - Difficulty levels (roughly equal thirds)
        """
        result = prompts[:]
        severity_counts = {}
        task_counts = {}
        diff_counts = {}

        for p in result:
            severity_counts[p["severity"]] = severity_counts.get(p["severity"], 0) + 1
            task_counts[p["annotation_task"]] = task_counts.get(p["annotation_task"], 0) + 1
            diff_counts[p["difficulty"]] = diff_counts.get(p["difficulty"], 0) + 1

        # Log balance info
        logger.info("Prompt distribution by severity: " + str(severity_counts))
        logger.info("Prompt distribution by task: " + str(task_counts))
        logger.info("Prompt distribution by difficulty: " + str(diff_counts))
        return result

    # ── Export ────────────────────────────────────────────────────────────────

    def _build_output_schema(self, prompts: List[Dict]) -> Dict:
        """Build the final output JSON structure."""
        taxonomy_summary = {
            "domain": {
                "id": SURVEILLANCE_DOMAIN.domain_id,
                "label": SURVEILLANCE_DOMAIN.label,
                "description": SURVEILLANCE_DOMAIN.description,
            },
            "incident_classes": [
                {
                    "class_id": cls.class_id,
                    "label": cls.label,
                    "incident_type_count": len(cls.incident_types),
                    "domain_tags": cls.domain_tags,
                }
                for cls in INCIDENT_CLASSES.values()
            ],
            "incident_types": [
                {
                    "type_id": it.type_id,
                    "label": it.label,
                    "severity": it.severity.name,
                    "temporal_sensitivity": it.temporal_sensitivity.value,
                    "visual_complexity": it.visual_complexity.value,
                    "signal_count": len(it.signals),
                    "annotation_tasks": [t.value for t in it.annotation_tasks],
                    "frame_requirement": it.frame_requirement.value,
                }
                for it in get_all_incident_types()
            ],
        }

        framework_config = {
            "annotation_tasks_supported": [t.value for t in AnnotationTask],
            "prompt_styles_supported": [s.value for s in PromptStyle],
            "difficulty_levels": [d.value for d in DifficultyLevel],
            "location_types": [l.value for l in LocationType],
            "camera_angles": [c.value for c in CameraAngle],
            "lighting_conditions": [l.value for l in LightingCondition],
            "quality_dimensions": list(self.builder.scorer.WEIGHTS.keys()),
            "quality_weights": self.builder.scorer.WEIGHTS,
        }

        return {
            "schema_version": "2.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "VLM Prompt Framework — Public Safety & Security Surveillance",
            "total_prompts": len(prompts),
            "taxonomy": taxonomy_summary,
            "framework_config": framework_config,
            "generation_stats": self.stats.summary(),
            "prompts": prompts,
        }

    def run(self) -> str:
        """Execute the full generation pipeline. Returns output file path."""
        logger.info("=" * 60)
        logger.info("VLM Prompt Generation Pipeline Starting")
        logger.info("=" * 60)

        # Step 1: Generate base prompts
        t0 = time.time()
        base_prompts = self._generate_base_prompts()
        logger.info(f"Base generation: {len(base_prompts)} prompts in {time.time()-t0:.1f}s")

        # Step 2: Augment with LLM
        if self.augmentor:
            augmented = self._augment_prompts(base_prompts)
        else:
            augmented = base_prompts
            for p in augmented:
                p["augmented_prompts"] = []
                p["hard_negative_examples"] = []
                p["context_variations"] = []
                p["reasoning_chain_template"] = ""
                p["synthesis_model"] = "none"
                p["synthesis_time_sec"] = 0.0

        # Step 3: Enforce diversity
        final = self._enforce_diversity(augmented)

        # Step 4: Sort by composite quality score
        final.sort(key=lambda p: p["quality_scores"]["composite"], reverse=True)

        # Step 5: Build output
        output = self._build_output_schema(final)

        # Step 6: Write JSON
        out_path = Path(self.config["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        total_time = time.time() - self.stats.start_time
        logger.info("=" * 60)
        logger.info(f"✓ Generation complete!")
        logger.info(f"  Total prompts:    {len(final)}")
        logger.info(f"  Output file:      {out_path}")
        logger.info(f"  Total time:       {total_time:.1f}s")
        logger.info(f"  File size:        {out_path.stat().st_size / 1024:.1f} KB")
        logger.info("=" * 60)

        return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate VLM fine-tuning prompts from the public-safety taxonomy"
    )
    p.add_argument("--output", default=DEFAULT_CONFIG["output_path"],
                   help="Output JSON file path")
    p.add_argument("--max-per-incident", type=int, default=DEFAULT_CONFIG["max_per_incident"],
                   help="Maximum prompts per incident type")
    p.add_argument("--min-quality", type=float, default=DEFAULT_CONFIG["min_quality"],
                   help="Minimum composite quality score (0-1)")
    p.add_argument("--augment", action="store_true", default=True,
                   help="Enable LLM augmentation (default: True)")
    p.add_argument("--no-augment", action="store_false", dest="augment",
                   help="Disable LLM augmentation (faster)")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--model", type=str, default=None,
                   help="Preferred local model name (e.g. google/flan-t5-large)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "output_path": args.output,
        "max_per_incident": args.max_per_incident,
        "min_quality": args.min_quality,
        "augment_with_llm": args.augment,
        "seed": args.seed,
        "model": args.model,
    }
    generator = VLMPromptGenerator(config)
    output_file = generator.run()
    print(f"\nOutput saved to: {output_file}")
