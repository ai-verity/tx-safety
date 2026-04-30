"""
VLM Prompt Generator (Production)
===================================
Full pipeline: taxonomy → framework → synthesis → JSON output.
Uses the deterministic synthesis engine (no API, no downloads).
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import itertools
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from taxonomy import (
    SURVEILLANCE_DOMAIN, INCIDENT_CLASSES, INCIDENT_TYPES, SIGNALS,
    AnnotationTask, Severity, TemporalSensitivity, VisualComplexity,
    FrameRequirement, get_all_incident_types,
)
from framework import (
    PromptBuilder, SceneContext, DifficultyLevel, PromptStyle,
    LocationType, CameraAngle, LightingCondition, OcclusionLevel,
    TemplateLibrary,
)
from synthesis_engine import DeterministicSynthesisEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Extended context matrix — ensures maximal scene diversity
# ─────────────────────────────────────────────────────────────────────────────

FULL_CONTEXT_MATRIX = [
    # (LocationType, CameraAngle, LightingCondition, OcclusionLevel, crowd, weather, time)
    (LocationType.PARKING_FACILITY,    CameraAngle.HIGH_ANGLE,  LightingCondition.DAYLIGHT,           OcclusionLevel.NONE,    "sparse",   "clear",    "afternoon"),
    (LocationType.PARKING_FACILITY,    CameraAngle.HIGH_ANGLE,  LightingCondition.NIGHT_ILLUMINATED,  OcclusionLevel.PARTIAL, "sparse",   "clear",    "night"),
    (LocationType.TRANSIT_HUB,         CameraAngle.OVERHEAD,    LightingCondition.NIGHT_ILLUMINATED,  OcclusionLevel.HEAVY,   "dense",    "clear",    "night"),
    (LocationType.TRANSIT_HUB,         CameraAngle.HIGH_ANGLE,  LightingCondition.DAYLIGHT,           OcclusionLevel.PARTIAL, "moderate", "overcast", "morning"),
    (LocationType.URBAN_STREET,        CameraAngle.EYE_LEVEL,   LightingCondition.DAYLIGHT,           OcclusionLevel.PARTIAL, "moderate", "clear",    "morning"),
    (LocationType.URBAN_STREET,        CameraAngle.HIGH_ANGLE,  LightingCondition.TWILIGHT,           OcclusionLevel.PARTIAL, "sparse",   "overcast", "evening"),
    (LocationType.URBAN_STREET,        CameraAngle.EYE_LEVEL,   LightingCondition.NIGHT_LOWLIGHT,     OcclusionLevel.HEAVY,   "sparse",   "rain",     "night"),
    (LocationType.PUBLIC_PLAZA,        CameraAngle.HIGH_ANGLE,  LightingCondition.DAYLIGHT,           OcclusionLevel.NONE,    "dense",    "clear",    "afternoon"),
    (LocationType.PUBLIC_PLAZA,        CameraAngle.FISHEYE,     LightingCondition.DAYLIGHT,           OcclusionLevel.PARTIAL, "dense",    "clear",    "afternoon"),
    (LocationType.AIRPORT,             CameraAngle.HIGH_ANGLE,  LightingCondition.DAYLIGHT,           OcclusionLevel.NONE,    "sparse",   "clear",    "morning"),
    (LocationType.AIRPORT,             CameraAngle.OVERHEAD,    LightingCondition.NIGHT_ILLUMINATED,  OcclusionLevel.NONE,    "sparse",   "clear",    "night"),
    (LocationType.CRITICAL_INFRA,      CameraAngle.HIGH_ANGLE,  LightingCondition.DAYLIGHT,           OcclusionLevel.NONE,    "sparse",   "clear",    "afternoon"),
    (LocationType.CRITICAL_INFRA,      CameraAngle.HIGH_ANGLE,  LightingCondition.INFRARED,           OcclusionLevel.NONE,    "sparse",   "fog",      "night"),
    (LocationType.INDUSTRIAL_FACILITY, CameraAngle.HIGH_ANGLE,  LightingCondition.OVERCAST,           OcclusionLevel.PARTIAL, "sparse",   "overcast", "morning"),
    (LocationType.INDUSTRIAL_FACILITY, CameraAngle.EYE_LEVEL,   LightingCondition.NIGHT_ILLUMINATED,  OcclusionLevel.PARTIAL, "sparse",   "clear",    "night"),
    (LocationType.BORDER_CHECKPOINT,   CameraAngle.HIGH_ANGLE,  LightingCondition.DAYLIGHT,           OcclusionLevel.NONE,    "moderate", "clear",    "afternoon"),
    (LocationType.EVENT_VENUE,         CameraAngle.HIGH_ANGLE,  LightingCondition.NIGHT_ILLUMINATED,  OcclusionLevel.HEAVY,   "dense",    "clear",    "evening"),
    (LocationType.WATERFRONT,          CameraAngle.HIGH_ANGLE,  LightingCondition.OVERCAST,           OcclusionLevel.NONE,    "sparse",   "overcast", "morning"),
    (LocationType.RESIDENTIAL_AREA,    CameraAngle.HIGH_ANGLE,  LightingCondition.DAYLIGHT,           OcclusionLevel.PARTIAL, "sparse",   "clear",    "afternoon"),
    (LocationType.RESIDENTIAL_AREA,    CameraAngle.EYE_LEVEL,   LightingCondition.NIGHT_LOWLIGHT,     OcclusionLevel.HEAVY,   "sparse",   "rain",     "night"),
]


def context_from_tuple(t) -> SceneContext:
    return SceneContext(*t)


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.n = 0
        self.by_incident: Dict[str, int] = {}
        self.by_class: Dict[str, int] = {}
        self.by_task: Dict[str, int] = {}
        self.by_sev: Dict[str, int] = {}
        self.by_diff: Dict[str, int] = {}
        self.by_style: Dict[str, int] = {}
        self.quality: List[float] = []
        self.t0 = time.time()

    def record(self, p: Dict):
        self.n += 1
        self.by_incident[p["incident_type_id"]] = self.by_incident.get(p["incident_type_id"], 0) + 1
        self.by_class[p["incident_class_id"]] = self.by_class.get(p["incident_class_id"], 0) + 1
        self.by_task[p["annotation_task"]] = self.by_task.get(p["annotation_task"], 0) + 1
        self.by_sev[p["severity"]] = self.by_sev.get(p["severity"], 0) + 1
        self.by_diff[p["difficulty"]] = self.by_diff.get(p["difficulty"], 0) + 1
        self.by_style[p["prompt_style"]] = self.by_style.get(p["prompt_style"], 0) + 1
        self.quality.append(p["quality_scores"]["composite"])

    def summary(self) -> Dict:
        q = self.quality or [0]
        buckets = {
            ">=0.8": sum(1 for v in q if v >= 0.8),
            "0.6-0.8": sum(1 for v in q if 0.6 <= v < 0.8),
            "0.4-0.6": sum(1 for v in q if 0.4 <= v < 0.6),
            "<0.4": sum(1 for v in q if v < 0.4),
        }
        return {
            "total_prompts": self.n,
            "elapsed_sec": round(time.time() - self.t0, 1),
            "by_incident_type": dict(sorted(self.by_incident.items())),
            "by_incident_class": dict(sorted(self.by_class.items())),
            "by_annotation_task": dict(sorted(self.by_task.items())),
            "by_severity": dict(sorted(self.by_sev.items())),
            "by_difficulty": dict(sorted(self.by_diff.items())),
            "by_prompt_style": dict(sorted(self.by_style.items())),
            "quality_stats": {
                "mean": round(sum(q) / len(q), 4),
                "min": round(min(q), 4),
                "max": round(max(q), 4),
                "distribution": buckets,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

class ProductionGenerator:
    def __init__(
        self,
        max_per_incident: int = 24,
        min_quality: float = 0.20,
        seed: int = 42,
        output_path: str = "/home/claude/vlm_framework/output/vlm_prompts_generated.json",
        include_augmentation: bool = True,
    ):
        self.max_per_incident = max_per_incident
        self.min_quality = min_quality
        self.seed = seed
        self.output_path = Path(output_path)
        self.include_augmentation = include_augmentation
        self.builder = PromptBuilder(seed=seed)
        self.synth = DeterministicSynthesisEngine(seed=seed)
        self.stats = Stats()
        self._rng = random.Random(seed)

        # Build class-id lookup
        self.class_lookup: Dict[str, str] = {}
        for cls in INCIDENT_CLASSES.values():
            for it in cls.incident_types:
                self.class_lookup[it.type_id] = cls.class_id

    def _generate_for_incident(self, incident) -> List[Dict]:
        """Generate all prompt variants for one incident type."""
        class_id = self.class_lookup.get(incident.type_id, "ic_unknown")
        results = []

        # Select contexts — use those tuned for incident + a few generic ones
        tuned_contexts = SceneContext.from_incident(incident)
        # Add up to 3 from the full matrix (shuffled per incident for diversity)
        rng_ctx = random.Random(self._rng.randint(0, 99999))
        extra = rng_ctx.sample(FULL_CONTEXT_MATRIX, k=min(3, len(FULL_CONTEXT_MATRIX)))
        extra_contexts = [context_from_tuple(t) for t in extra]
        all_contexts = tuned_contexts + extra_contexts

        tasks = incident.annotation_tasks[:4]
        styles = list(PromptStyle)
        difficulties = list(DifficultyLevel)

        candidates = []
        for task, style, diff, ctx in itertools.product(tasks, styles, difficulties, all_contexts):
            try:
                p = self.builder.build(incident, task, style, diff, ctx, class_id)
                candidates.append(p)
            except Exception:
                continue

        # Rank by quality, enforce diversity (no duplicate task+style+diff combos)
        candidates.sort(key=lambda p: p.composite_score(), reverse=True)
        seen = set()
        selected = []
        for p in candidates:
            if p.composite_score() < self.min_quality:
                continue
            key = (p.annotation_task, p.prompt_style, p.difficulty)
            if key not in seen:
                seen.add(key)
                selected.append(p)
            if len(selected) >= self.max_per_incident:
                break

        # Convert to dicts and augment
        for p in selected:
            d = p.to_dict()
            if self.include_augmentation:
                # Gather signal data for synthesis
                all_vis_attrs = []
                for sig in incident.signals:
                    all_vis_attrs.extend(sig.visual_attributes[:2])

                synth_result = self.synth.synthesise(
                    base_prompt=p.user_prompt,
                    incident_label=incident.label,
                    annotation_task=p.annotation_task,
                    counterfactuals=incident.counterfactual_cues,
                    signal_labels=[s.label for s in incident.signals],
                    visual_attributes=all_vis_attrs[:4],
                    scene_ctx=d["scene_context"],
                    n_paraphrases=3,
                    n_hard_neg=2,
                )
                d.update(synth_result)
            else:
                d.update({
                    "augmented_prompts": [],
                    "hard_negative_examples": [],
                    "context_variations": [],
                    "reasoning_chain_template": "",
                    "vqa_answer_options": {},
                    "synthesis_model": "none",
                    "synthesis_time_sec": 0.0,
                })
            self.stats.record(d)
            results.append(d)

        return results

    def run(self) -> str:
        """Execute full pipeline and write output JSON."""
        logger.info("=" * 65)
        logger.info("VLM Prompt Generation Pipeline  |  Public Safety Surveillance")
        logger.info("=" * 65)

        all_prompts: List[Dict] = []
        incidents = get_all_incident_types()
        logger.info(f"Processing {len(incidents)} incident types across {len(INCIDENT_CLASSES)} classes...")

        for incident in incidents:
            prompts = self._generate_for_incident(incident)
            all_prompts.extend(prompts)
            cls_label = self.class_lookup.get(incident.type_id, "?")
            logger.info(
                f"  [{cls_label}] {incident.label[:40]:40s} → {len(prompts):3d} prompts"
            )

        # Sort final output by quality
        all_prompts.sort(key=lambda p: p["quality_scores"]["composite"], reverse=True)

        summary = self.stats.summary()
        logger.info("-" * 65)
        logger.info(f"Total prompts generated: {summary['total_prompts']}")
        logger.info(f"Quality distribution: {summary['quality_stats']['distribution']}")
        logger.info(f"Mean quality: {summary['quality_stats']['mean']}")

        # Build taxonomy index
        taxonomy_index = {
            "domain": {
                "id": SURVEILLANCE_DOMAIN.domain_id,
                "label": SURVEILLANCE_DOMAIN.label,
                "description": SURVEILLANCE_DOMAIN.description,
            },
            "incident_classes": [
                {
                    "class_id": cls.class_id,
                    "label": cls.label,
                    "description": cls.description,
                    "domain_tags": cls.domain_tags,
                    "incident_types": [it.type_id for it in cls.incident_types],
                }
                for cls in INCIDENT_CLASSES.values()
            ],
            "incident_type_index": {
                it.type_id: {
                    "label": it.label,
                    "description": it.description,
                    "severity": it.severity.name,
                    "severity_int": it.severity.value,
                    "temporal_sensitivity": it.temporal_sensitivity.value,
                    "visual_complexity": it.visual_complexity.value,
                    "frame_requirement": it.frame_requirement.value,
                    "annotation_tasks": [t.value for t in it.annotation_tasks],
                    "signal_count": len(it.signals),
                    "signals": [
                        {
                            "signal_id": s.signal_id,
                            "label": s.label,
                            "description": s.description,
                            "visual_attributes": s.visual_attributes,
                            "negative_attributes": s.negative_attributes,
                            "annotation_tasks": [t.value for t in s.annotation_tasks],
                            "frame_requirement": s.frame_requirement.value,
                            "temporal_window_sec": s.temporal_window_sec,
                        }
                        for s in it.signals
                    ],
                    "counterfactual_cues": it.counterfactual_cues,
                    "prompt_focus_objects": it.prompt_focus_objects,
                    "prompt_spatial_relations": it.prompt_spatial_relations,
                    "prompt_temporal_cues": it.prompt_temporal_cues,
                    "related_types": it.related_types,
                }
                for it in get_all_incident_types()
            },
            "signal_library": {
                sig_id: {
                    "signal_id": sig.signal_id,
                    "label": sig.label,
                    "description": sig.description,
                    "visual_attributes": sig.visual_attributes,
                    "negative_attributes": sig.negative_attributes,
                    "annotation_tasks": [t.value for t in sig.annotation_tasks],
                }
                for sig_id, sig in SIGNALS.items()
            },
        }

        framework_spec = {
            "annotation_tasks": {t.value: t.name for t in AnnotationTask},
            "prompt_styles": [s.value for s in PromptStyle],
            "difficulty_levels": [d.value for d in DifficultyLevel],
            "location_types": [l.value for l in LocationType],
            "camera_angles": [c.value for c in CameraAngle],
            "lighting_conditions": [l.value for l in LightingCondition],
            "occlusion_levels": [o.value for o in OcclusionLevel],
            "quality_scorer": {
                "dimensions": list(self.builder.scorer.WEIGHTS.keys()),
                "weights": self.builder.scorer.WEIGHTS,
                "formula": "composite = sum(score_i * weight_i for each dimension)",
            },
            "synthesis_engine": "DeterministicSynthesisEngine v2 (rule-based, no API)",
            "total_context_configurations": len(FULL_CONTEXT_MATRIX),
            "context_matrix_description": (
                "20 canonical scene configurations spanning location type × camera angle × "
                "lighting × occlusion × crowd density × weather × time of day"
            ),
        }

        output = {
            "schema_version": "2.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": {
                "name": "VLM Prompt Framework — Public Safety & Security Surveillance",
                "version": "2.1.0",
                "synthesis_backend": "DeterministicSynthesisEngine (no external API)",
                "seed": self.seed,
                "max_per_incident": self.max_per_incident,
                "min_quality_threshold": self.min_quality,
            },
            "taxonomy": taxonomy_index,
            "framework_spec": framework_spec,
            "generation_stats": summary,
            "prompts": all_prompts,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        size_kb = self.output_path.stat().st_size / 1024
        logger.info(f"Output: {self.output_path}  ({size_kb:.0f} KB)")
        logger.info("=" * 65)
        return str(self.output_path)


if __name__ == "__main__":
    gen = ProductionGenerator(
        max_per_incident=24,
        min_quality=0.20,
        seed=42,
        output_path=str(Path(__file__).parent / "output" / "vlm_prompts_generated.json"),
        include_augmentation=True,
    )
    gen.run()
