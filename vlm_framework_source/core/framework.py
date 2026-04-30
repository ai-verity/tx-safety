"""
VLM Prompt Framework
====================
Transforms taxonomy nodes into structured, high-quality prompts for VLM
fine-tuning.  Supports five prompt formats:

  1. DETECTION   – "Is there a <incident> visible in this image?"
  2. GROUNDING   – "Identify and localise all <incident> elements."
  3. CAPTIONING  – "Describe this scene focusing on <incident>."
  4. VQA         – Multi-choice / binary Q&A
  5. INSTRUCTION – Instruction-following style (for InstructBLIP / LLaVA)

Each prompt is parameterised by:
  - IncidentType (taxonomy)
  - SceneContext  (camera angle, lighting, location type)
  - DifficultyLevel (easy / medium / hard)
  - PromptStyle   (terse / descriptive / Socratic / chain-of-thought)

Quality dimensions measured per prompt:
  - specificity      (0–1) how precisely it identifies target behaviour
  - discriminability (0–1) how well it excludes false positives
  - completeness     (0–1) coverage of all required visual elements
  - temporal_clarity (0–1) whether time-based cues are included when needed
  - composite        weighted mean of above
"""

from __future__ import annotations
import random
import hashlib
import itertools
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any

from taxonomy import (
    IncidentType, ObservationSignal, AnnotationTask,
    TemporalSensitivity, VisualComplexity, Severity,
    FrameRequirement, CameraAngle, LightingCondition, OcclusionLevel,
    get_all_incident_types, INCIDENT_TYPES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Framework enumerations
# ─────────────────────────────────────────────────────────────────────────────

class DifficultyLevel(str, Enum):
    EASY   = "easy"    # ideal conditions, clear target
    MEDIUM = "medium"  # partial occlusion, mixed scene
    HARD   = "hard"    # low light, heavy occlusion, ambiguous context


class PromptStyle(str, Enum):
    TERSE         = "terse"          # short, imperative
    DESCRIPTIVE   = "descriptive"    # full context, rich language
    SOCRATIC      = "socratic"       # question-driven reasoning
    COT           = "chain_of_thought"  # step-by-step explicit reasoning


class LocationType(str, Enum):
    TRANSIT_HUB         = "transit_hub"
    AIRPORT             = "airport"
    URBAN_STREET        = "urban_street"
    PARKING_FACILITY    = "parking_facility"
    PUBLIC_PLAZA        = "public_plaza"
    INDUSTRIAL_FACILITY = "industrial_facility"
    CRITICAL_INFRA      = "critical_infrastructure"
    RESIDENTIAL_AREA    = "residential_area"
    EVENT_VENUE         = "event_venue"
    BORDER_CHECKPOINT   = "border_checkpoint"
    WATERFRONT          = "waterfront"


# ─────────────────────────────────────────────────────────────────────────────
# Scene context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneContext:
    location_type: LocationType
    camera_angle: CameraAngle
    lighting: LightingCondition
    occlusion: OcclusionLevel
    crowd_level: str           # "sparse" | "moderate" | "dense"
    weather: str               # e.g. "clear", "rain", "fog"
    time_of_day: str           # "morning", "afternoon", "evening", "night"

    def to_description(self) -> str:
        parts = [
            f"captured from a {self.camera_angle.value.replace('_', ' ')} perspective",
            f"in {self.lighting.value.replace('_', ' ')} conditions",
            f"at a {self.location_type.value.replace('_', ' ')}",
            f"during {self.time_of_day}",
        ]
        if self.weather not in ("clear",):
            parts.append(f"with {self.weather}")
        if self.occlusion != OcclusionLevel.NONE:
            parts.append(f"with {self.occlusion.value} occlusion")
        if self.crowd_level != "sparse":
            parts.append(f"crowd density: {self.crowd_level}")
        return ", ".join(parts)

    @staticmethod
    def random() -> "SceneContext":
        return SceneContext(
            location_type=random.choice(list(LocationType)),
            camera_angle=random.choice(list(CameraAngle)),
            lighting=random.choice(list(LightingCondition)),
            occlusion=random.choice(list(OcclusionLevel)),
            crowd_level=random.choice(["sparse", "moderate", "dense"]),
            weather=random.choice(["clear", "rain", "fog", "overcast"]),
            time_of_day=random.choice(["morning", "afternoon", "evening", "night"]),
        )

    @staticmethod
    def from_incident(it: IncidentType) -> List["SceneContext"]:
        """Return a curated set of contexts appropriate for the incident type."""
        base = []
        if it.visual_complexity == VisualComplexity.HIGH:
            base += [
                SceneContext(LocationType.PUBLIC_PLAZA, CameraAngle.HIGH_ANGLE,
                             LightingCondition.DAYLIGHT, OcclusionLevel.PARTIAL,
                             "dense", "clear", "afternoon"),
                SceneContext(LocationType.TRANSIT_HUB, CameraAngle.OVERHEAD,
                             LightingCondition.NIGHT_ILLUMINATED, OcclusionLevel.HEAVY,
                             "dense", "clear", "night"),
            ]
        if it.temporal_sensitivity == TemporalSensitivity.IMMEDIATE:
            base += [
                SceneContext(LocationType.URBAN_STREET, CameraAngle.EYE_LEVEL,
                             LightingCondition.DAYLIGHT, OcclusionLevel.PARTIAL,
                             "moderate", "clear", "morning"),
            ]
        if not base:
            base += [
                SceneContext(LocationType.PARKING_FACILITY, CameraAngle.HIGH_ANGLE,
                             LightingCondition.DAYLIGHT, OcclusionLevel.NONE,
                             "sparse", "clear", "afternoon"),
                SceneContext(LocationType.URBAN_STREET, CameraAngle.HIGH_ANGLE,
                             LightingCondition.NIGHT_ILLUMINATED, OcclusionLevel.PARTIAL,
                             "sparse", "clear", "night"),
            ]
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Prompt quality scorer
# ─────────────────────────────────────────────────────────────────────────────

class PromptQualityScorer:
    WEIGHTS = {
        "specificity":      0.30,
        "discriminability": 0.30,
        "completeness":     0.25,
        "temporal_clarity": 0.15,
    }

    @staticmethod
    def score(prompt_text: str, incident: IncidentType, context: SceneContext) -> Dict[str, float]:
        lower = prompt_text.lower()

        # Specificity: key focus objects mentioned
        focus_hits = sum(
            1 for obj in incident.prompt_focus_objects
            if any(tok in lower for tok in obj.replace("_", " ").split())
        )
        specificity = min(1.0, focus_hits / max(len(incident.prompt_focus_objects), 1))

        # Discriminability: counterfactuals or negation mentioned
        neg_keywords = ["not", "unlike", "distinguish", "false positive",
                        "contrast", "versus", "compared to", "rule out"]
        counter_hits = sum(1 for k in neg_keywords if k in lower)
        counter_hits += sum(
            1 for c in incident.counterfactual_cues
            if any(tok in lower for tok in c.replace("_", " ").split()[:2])
        )
        discriminability = min(1.0, counter_hits / 4.0)

        # Completeness: signal attributes covered
        all_attrs = list(itertools.chain.from_iterable(
            s.visual_attributes for s in incident.signals
        ))
        attr_hits = sum(
            1 for attr in all_attrs
            if any(tok in lower for tok in attr.replace("_", " ").split()[:2])
        )
        completeness = min(1.0, attr_hits / max(len(all_attrs), 1))

        # Temporal clarity
        temporal_hit = 0.0
        if incident.frame_requirement == FrameRequirement.MULTI:
            temporal_keywords = ["sequence", "over time", "frames", "duration",
                                  "sustained", "consecutive", "temporal", "period"]
            temporal_hit = min(1.0, sum(1 for k in temporal_keywords if k in lower) / 2.0)
        else:
            temporal_hit = 1.0 if "single" in lower or "frame" not in lower else 0.8

        scores = {
            "specificity":      round(specificity, 3),
            "discriminability": round(discriminability, 3),
            "completeness":     round(completeness, 3),
            "temporal_clarity": round(temporal_hit, 3),
        }
        composite = sum(
            scores[k] * PromptQualityScorer.WEIGHTS[k]
            for k in PromptQualityScorer.WEIGHTS
        )
        scores["composite"] = round(composite, 4)
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Individual prompt record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VLMPrompt:
    prompt_id: str
    incident_type_id: str
    incident_label: str
    incident_class_id: str
    severity: str
    annotation_task: str
    prompt_style: str
    difficulty: str
    scene_context: Dict[str, Any]
    system_instruction: str
    user_prompt: str
    assistant_hint: str          # ideal short answer / grounding description
    negative_example_cue: str    # what the model should NOT respond to
    quality_scores: Dict[str, float]
    frame_requirement: str
    temporal_window_sec: Optional[int]
    signal_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def composite_score(self) -> float:
        return self.quality_scores.get("composite", 0.0)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic template library
# ─────────────────────────────────────────────────────────────────────────────

class TemplateLibrary:
    """
    A structured set of prompt templates indexed by (AnnotationTask, PromptStyle).
    Each template is a callable that accepts keyword arguments and returns a
    (system_instruction, user_prompt, assistant_hint) tuple.
    """

    # ── System instruction variants ──────────────────────────────────────────
    SYSTEM_INSTRUCTIONS = {
        "detection": (
            "You are a precision surveillance analysis system. Your task is to detect and "
            "report the presence, location, and key attributes of security-relevant events "
            "within the provided image or image sequence. Be objective, precise, and concise."
        ),
        "classification": (
            "You are a scene classification expert for public-safety applications. Analyse "
            "the provided image and classify the primary event or condition present, selecting "
            "from known security incident categories. Justify your classification briefly."
        ),
        "grounding": (
            "You are a visual grounding model for security applications. When given an image "
            "and a description of a target object or event, identify its precise location using "
            "bounding coordinates or natural language spatial references."
        ),
        "captioning": (
            "You are a forensic-quality image captioning system. Generate a detailed, factual "
            "description of security-relevant visual content. Include object identities, "
            "spatial relationships, behavioural cues, and any anomalous features."
        ),
        "vqa": (
            "You are a visual question-answering system specialised in public-safety scenarios. "
            "Answer questions about the provided image accurately. Where a yes/no answer is "
            "appropriate, provide it followed by a brief justification."
        ),
        "counting": (
            "You are a visual counting and density estimation system. Count the specified "
            "objects or persons in the image as accurately as possible, noting any occlusion "
            "that affects count confidence."
        ),
        "temporal_change": (
            "You are a temporal scene analysis system. Compare the provided sequence of images "
            "to identify changes, emerging events, or developing situations relevant to "
            "public safety."
        ),
        "attribute_recog": (
            "You are an attribute recognition system for surveillance applications. Identify "
            "and describe the visual attributes of specified objects — including state, "
            "condition, colour, posture, and contextual anomalies."
        ),
        "scene_graph": (
            "You are a scene graph generator for security intelligence. Produce structured "
            "entity–relation–entity triples that capture the spatial and behavioural "
            "relationships relevant to security assessment."
        ),
    }

    # ── Detection templates ───────────────────────────────────────────────────
    DETECTION = {
        PromptStyle.TERSE: lambda it, ctx, diff, signals: (
            f"Detect any {it.label.lower()} in this image.",
            f"Scan the scene — {ctx.to_description()} — for visual indicators of {it.label.lower()}. "
            f"Report: present/absent, confidence, and location if present.",
            f"{'Present' if diff != DifficultyLevel.HARD else 'Possible'}: describe bounding region and key visual evidence.",
        ),
        PromptStyle.DESCRIPTIVE: lambda it, ctx, diff, signals: (
            f"Analyse this surveillance image for evidence of {it.label.lower()}.",
            (
                f"The following image was captured {ctx.to_description()}. "
                f"Your task is to determine whether a {it.label.lower()} is present. "
                f"Specifically, look for: {', '.join(s.label for s in signals)}. "
                f"Note the presence or absence of these visual cues: "
                f"{', '.join(signals[0].visual_attributes[:4] if signals else [])}. "
                f"If the incident is present, describe its location, the supporting evidence, "
                f"and the confidence of your assessment."
            ),
            f"Incident confirmed/not confirmed. Location: [describe]. Evidence: [list key visual cues].",
        ),
        PromptStyle.SOCRATIC: lambda it, ctx, diff, signals: (
            f"Is a {it.label.lower()} present in this image?",
            (
                f"Examine this image — {ctx.to_description()}. "
                f"Ask yourself: Are any of the following conditions satisfied? "
                f"{' | '.join(s.label for s in signals[:3])}. "
                f"What visual evidence supports or refutes the presence of {it.label.lower()}? "
                f"Consider possible false positives such as: "
                f"{', '.join(it.counterfactual_cues[:2]) if it.counterfactual_cues else 'none listed'}. "
                f"State your conclusion and the key observations that led to it."
            ),
            f"Yes/No, because [key observation]. Confidence: [0-100]%.",
        ),
        PromptStyle.COT: lambda it, ctx, diff, signals: (
            f"Think step by step: is there a {it.label.lower()} in this image?",
            (
                f"You are analysing a surveillance feed {ctx.to_description()}. "
                f"Work through the following steps:\n"
                f"Step 1 — Identify primary objects: List all security-relevant objects visible.\n"
                f"Step 2 — Match signals: Check for {', '.join(s.label for s in signals[:3])}.\n"
                f"Step 3 — Assess attributes: Do the objects show "
                f"{', '.join(signals[0].visual_attributes[:3] if signals else [])}?\n"
                f"Step 4 — Rule out false positives: Could this be "
                f"{', '.join(it.counterfactual_cues[:2]) if it.counterfactual_cues else 'a benign scenario'}?\n"
                f"Step 5 — Conclude: State whether {it.label.lower()} is confirmed, "
                f"probable, uncertain, or absent."
            ),
            f"Step 1: [objects]. Step 2: [signal match]. Step 3: [attributes]. Step 4: [FP check]. Step 5: [conclusion].",
        ),
    }

    # ── VQA templates ─────────────────────────────────────────────────────────
    VQA = {
        PromptStyle.TERSE: lambda it, ctx, diff, signals: (
            "Answer the following security assessment question.",
            f"Does the image ({ctx.to_description()}) show evidence of {it.label.lower()}? Answer Yes or No.",
            "Yes" if diff == DifficultyLevel.EASY else "No — [reason for negative].",
        ),
        PromptStyle.DESCRIPTIVE: lambda it, ctx, diff, signals: (
            "You are answering a structured security question.",
            (
                f"Image context: {ctx.to_description()}.\n"
                f"Question: Is a {it.label.lower()} occurring?\n"
                f"Options: (A) Yes — confirmed (B) Probable — insufficient evidence "
                f"(C) Not present (D) Cannot determine\n"
                f"Select the best option and provide a one-sentence justification."
            ),
            "(A) Yes — confirmed. Justification: [key visual evidence].",
        ),
        PromptStyle.SOCRATIC: lambda it, ctx, diff, signals: (
            "Answer security questions by reasoning from visual evidence.",
            (
                f"Looking at this image ({ctx.to_description()}):\n"
                f"Q1: What is the primary activity or event depicted?\n"
                f"Q2: Are any of the following present: {', '.join(s.label for s in signals[:2])}?\n"
                f"Q3: How does the scene differ from a normal, non-incident scenario?\n"
                f"Q4: Should this scene trigger a security alert? Justify."
            ),
            "A1: [activity]. A2: [present/absent]. A3: [difference]. A4: [yes/no + reason].",
        ),
        PromptStyle.COT: lambda it, ctx, diff, signals: (
            "Perform chain-of-thought reasoning to answer this visual question.",
            (
                f"Scene: {ctx.to_description()}.\n"
                f"Question: Does this scene represent {it.label.lower()}?\n"
                f"Reasoning chain:\n"
                f"  1. What objects/persons are in the foreground?\n"
                f"  2. What are their spatial relationships?\n"
                f"  3. Does the temporal pattern (if observable) match {it.temporal_sensitivity.value} indicators?\n"
                f"  4. Final answer with confidence percentage."
            ),
            "1.[objects]. 2.[relations]. 3.[temporal match]. 4. Answer: [Y/N], confidence: [%].",
        ),
    }

    # ── Grounding templates ───────────────────────────────────────────────────
    GROUNDING = {
        PromptStyle.TERSE: lambda it, ctx, diff, signals: (
            "Localise the security incident in this image.",
            f"Point to the {it.label.lower()} in the image. Describe its location using spatial language (top-left, centre, etc.).",
            "Located at [spatial description]. Key indicator at [sub-region].",
        ),
        PromptStyle.DESCRIPTIVE: lambda it, ctx, diff, signals: (
            "Perform visual grounding for a security incident.",
            (
                f"In the image ({ctx.to_description()}), identify and localise the region "
                f"that constitutes a {it.label.lower()}. "
                f"Describe the bounding region (e.g., 'upper-left quadrant'), the primary "
                f"object(s) involved, and the key visual attributes that confirm the identification: "
                f"{', '.join(signals[0].visual_attributes[:3] if signals else [])}."
            ),
            "Region: [location]. Objects: [list]. Confirming attributes: [list].",
        ),
        PromptStyle.SOCRATIC: lambda it, ctx, diff, signals: (
            "Locate the incident and explain your spatial reasoning.",
            (
                f"This scene was captured {ctx.to_description()}. "
                f"Where in the image is the {it.label.lower()} located? "
                f"What visual boundary separates the incident zone from the normal background? "
                f"Describe the spatial relationship between the incident and nearby fixed infrastructure."
            ),
            "Incident zone: [region]. Boundary: [description]. Relation to infrastructure: [description].",
        ),
        PromptStyle.COT: lambda it, ctx, diff, signals: (
            "Use step-by-step spatial reasoning to localise the incident.",
            (
                f"Scene: {ctx.to_description()}.\n"
                f"Task: Localise {it.label.lower()}.\n"
                f"Step 1 — Partition image into quadrants. In which quadrant(s) is the primary subject?\n"
                f"Step 2 — Identify anchor objects (fixed infrastructure, markings).\n"
                f"Step 3 — Describe the incident region relative to anchors.\n"
                f"Step 4 — Confirm by matching visual attributes: "
                f"{', '.join(signals[0].visual_attributes[:3] if signals else ['not available'])}."
            ),
            "Q1:[quadrant]. Anchors:[list]. Region:[relative description]. Confirmed by:[attributes].",
        ),
    }

    # ── Captioning templates ──────────────────────────────────────────────────
    CAPTIONING = {
        PromptStyle.TERSE: lambda it, ctx, diff, signals: (
            "Generate a security-focused scene caption.",
            f"Describe this image with focus on {it.label.lower()} indicators.",
            f"[Scene description emphasising {', '.join(it.prompt_focus_objects[:2])}].",
        ),
        PromptStyle.DESCRIPTIVE: lambda it, ctx, diff, signals: (
            "Generate a detailed forensic-quality security scene caption.",
            (
                f"Produce a comprehensive caption for this surveillance image captured "
                f"{ctx.to_description()}. Your caption should:\n"
                f"  1. Identify all security-relevant objects and persons.\n"
                f"  2. Describe any indicators of {it.label.lower()}.\n"
                f"  3. Note spatial relationships and proximity to infrastructure.\n"
                f"  4. Include any temporal cues visible (e.g., motion blur, lighting changes).\n"
                f"  5. Note anything that distinguishes this from a normal scene."
            ),
            "In this scene [full description] suggesting [incident]. Notable features: [list].",
        ),
        PromptStyle.SOCRATIC: lambda it, ctx, diff, signals: (
            "Produce a contextually-aware security scene narrative.",
            (
                f"What is happening in this image ({ctx.to_description()})? "
                f"Describe the scene as a security analyst would in an incident report, "
                f"focusing on evidence relevant to {it.label.lower()}. "
                f"What would a trained observer note first? What secondary cues support the assessment?"
            ),
            "At [location], [primary observation]. Secondary indicators include [list].",
        ),
        PromptStyle.COT: lambda it, ctx, diff, signals: (
            "Generate a chain-of-thought security caption.",
            (
                f"Scene context: {ctx.to_description()}.\n"
                f"Generate a caption following this reasoning structure:\n"
                f"  Background: [what is the normal expected state of this scene?]\n"
                f"  Deviation: [what specifically deviates from normal?]\n"
                f"  Evidence: [which visual attributes confirm deviation?]\n"
                f"  Assessment: [is this {it.label.lower()}? confidence?]\n"
                f"  Action: [what response does this scene warrant?]"
            ),
            "Background: [normal]. Deviation: [anomaly]. Evidence: [attributes]. Assessment: [verdict]. Action: [response].",
        ),
    }

    # ── Temporal / multi-frame templates ─────────────────────────────────────
    TEMPORAL = {
        PromptStyle.TERSE: lambda it, ctx, diff, signals: (
            "Analyse this image sequence for emerging security events.",
            f"Compare these frames and identify any developing {it.label.lower()}. Describe what changes between frames.",
            "Change detected: [description of development from frame N to frame M].",
        ),
        PromptStyle.DESCRIPTIVE: lambda it, ctx, diff, signals: (
            "Perform temporal change detection for security monitoring.",
            (
                f"You are presented with a sequence of frames captured "
                f"{ctx.to_description()}. "
                f"Analyse the temporal progression to determine if {it.label.lower()} is "
                f"developing. Look for: "
                f"{', '.join(it.prompt_temporal_cues[:3]) if it.prompt_temporal_cues else 'changes in object state or position'}. "
                f"Describe what changes between frames, the rate of change, and whether "
                f"the pattern is consistent with {it.label.lower()}."
            ),
            "Frame progression: [description]. Rate of change: [fast/slow]. Consistent with incident: [yes/no + reasoning].",
        ),
        PromptStyle.SOCRATIC: lambda it, ctx, diff, signals: (
            "Reason about temporal development of a potential security incident.",
            (
                f"Observe this sequence of images ({ctx.to_description()}). "
                f"At what point in the sequence does the scene transition from normal to "
                f"potentially concerning? What specific visual change triggers this transition? "
                f"How long has the condition been developing? "
                f"Is the temporal pattern consistent with {it.label.lower()}?"
            ),
            "Transition at frame [N]. Trigger: [change description]. Duration: [estimate]. Verdict: [Y/N + reason].",
        ),
        PromptStyle.COT: lambda it, ctx, diff, signals: (
            "Step-by-step temporal analysis of a security scene sequence.",
            (
                f"Scene: {ctx.to_description()}. Incident type under investigation: {it.label}.\n"
                f"Step 1 — Baseline frame: Describe the initial normal state.\n"
                f"Step 2 — Change detection: Identify what changes across subsequent frames.\n"
                f"Step 3 — Signal matching: Do changes match {', '.join(s.label for s in signals[:2])}?\n"
                f"Step 4 — Temporal window: Has the condition persisted for "
                f"{it.signals[0].temporal_window_sec or 'an extended'} seconds?\n"
                f"Step 5 — Conclusion: Confirm, probable, or dismiss {it.label.lower()}."
            ),
            "S1:[baseline]. S2:[changes]. S3:[signal match]. S4:[duration OK?]. S5:[conclusion].",
        ),
    }

    # ── Counting templates ────────────────────────────────────────────────────
    COUNTING = {
        PromptStyle.TERSE: lambda it, ctx, diff, signals: (
            "Count security-relevant objects in this image.",
            f"How many instances of {it.label.lower()} are visible in this scene?",
            "Count: [N]. Confidence: [high/medium/low due to occlusion].",
        ),
        PromptStyle.DESCRIPTIVE: lambda it, ctx, diff, signals: (
            "Perform security-relevant object counting.",
            (
                f"In this image ({ctx.to_description()}), count all instances relevant to "
                f"{it.label.lower()}. Provide: total count, a breakdown by sub-region if "
                f"multiple clusters exist, confidence level, and any occlusion factors that "
                f"may affect accuracy."
            ),
            "Total: [N]. Sub-regions: [breakdown]. Confidence: [level]. Occlusion note: [if any].",
        ),
        PromptStyle.SOCRATIC: lambda it, ctx, diff, signals: (
            "Count and reason about security-relevant population or objects.",
            (
                f"Scene: {ctx.to_description()}. How many {it.label.lower()} indicators are "
                f"present? Does the count exceed the threshold that would trigger an alert? "
                f"What makes precise counting difficult in this scene?"
            ),
            "Count: [N]. Threshold exceeded: [yes/no]. Counting challenges: [description].",
        ),
        PromptStyle.COT: lambda it, ctx, diff, signals: (
            "Step-by-step counting of security-relevant elements.",
            (
                f"Scene: {ctx.to_description()}. Target: instances of {it.label.lower()}.\n"
                f"Step 1 — Divide scene into grid zones.\n"
                f"Step 2 — Count in each zone.\n"
                f"Step 3 — Sum total.\n"
                f"Step 4 — Adjust for estimated occlusion.\n"
                f"Step 5 — Assess whether count warrants alert."
            ),
            "Zones: [counts]. Total: [N]. Occlusion adj.: [+/- N]. Alert warranted: [Y/N].",
        ),
    }

    @classmethod
    def get_template(cls, task: AnnotationTask, style: PromptStyle):
        mapping = {
            AnnotationTask.DETECTION:       cls.DETECTION,
            AnnotationTask.VQA:             cls.VQA,
            AnnotationTask.GROUNDING:       cls.GROUNDING,
            AnnotationTask.CAPTIONING:      cls.CAPTIONING,
            AnnotationTask.TEMPORAL_CHANGE: cls.TEMPORAL,
            AnnotationTask.COUNTING:        cls.COUNTING,
            # fallback to detection for unlisted tasks
            AnnotationTask.CLASSIFICATION:  cls.DETECTION,
            AnnotationTask.ATTRIBUTE_RECOG: cls.DETECTION,
            AnnotationTask.SCENE_GRAPH:     cls.CAPTIONING,
        }
        template_group = mapping.get(task, cls.DETECTION)
        template_fn = template_group.get(style, template_group[PromptStyle.DESCRIPTIVE])
        return template_fn


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs VLMPrompt objects from taxonomy nodes and scene contexts.
    """

    def __init__(self, seed: int = 42):
        self.scorer = PromptQualityScorer()
        self._rng = random.Random(seed)
        self._id_counter = 0

    def _make_id(self, incident_id: str, task: str, style: str, diff: str) -> str:
        self._id_counter += 1
        raw = f"{incident_id}:{task}:{style}:{diff}:{self._id_counter}"
        return "vlm_" + hashlib.md5(raw.encode()).hexdigest()[:12]

    def build(
        self,
        incident: IncidentType,
        task: AnnotationTask,
        style: PromptStyle,
        difficulty: DifficultyLevel,
        context: SceneContext,
        class_id: str = "ic_unknown",
    ) -> VLMPrompt:
        template_fn = TemplateLibrary.get_template(task, style)
        system_inst, user_prompt, assistant_hint = template_fn(
            incident, context, difficulty, incident.signals
        )

        # Enhance user prompt with difficulty modifiers
        if difficulty == DifficultyLevel.HARD:
            user_prompt += (
                " Note: visibility may be reduced by occlusion, distance or lighting. "
                "Report any partial evidence and estimate confidence accordingly."
            )
        elif difficulty == DifficultyLevel.MEDIUM:
            user_prompt += " Some elements may be partially occluded or at an angle."

        # Add negative example cue
        neg_cues = incident.counterfactual_cues
        neg_example = self._rng.choice(neg_cues).replace("_", " ") if neg_cues else (
            f"a normal, non-incident scene at a {context.location_type.value.replace('_', ' ')}"
        )

        quality = self.scorer.score(user_prompt, incident, context)

        return VLMPrompt(
            prompt_id=self._make_id(incident.type_id, task.value, style.value, difficulty.value),
            incident_type_id=incident.type_id,
            incident_label=incident.label,
            incident_class_id=class_id,
            severity=incident.severity.name,
            annotation_task=task.value,
            prompt_style=style.value,
            difficulty=difficulty.value,
            scene_context={
                "location_type": context.location_type.value,
                "camera_angle": context.camera_angle.value,
                "lighting": context.lighting.value,
                "occlusion": context.occlusion.value,
                "crowd_level": context.crowd_level,
                "weather": context.weather,
                "time_of_day": context.time_of_day,
            },
            system_instruction=system_inst,
            user_prompt=user_prompt,
            assistant_hint=assistant_hint,
            negative_example_cue=neg_example,
            quality_scores=quality,
            frame_requirement=incident.frame_requirement.value,
            temporal_window_sec=incident.signals[0].temporal_window_sec if incident.signals else None,
            signal_ids=[s.signal_id for s in incident.signals],
            metadata={
                "visual_complexity": incident.visual_complexity.value,
                "temporal_sensitivity": incident.temporal_sensitivity.value,
                "related_types": incident.related_types,
            },
        )

    def build_all_combinations(
        self,
        incident: IncidentType,
        class_id: str = "ic_unknown",
        max_per_incident: int = 24,
    ) -> List[VLMPrompt]:
        """
        Build prompts across all task × style × difficulty × context combinations,
        ranked by composite quality score.
        """
        contexts = SceneContext.from_incident(incident)
        tasks = incident.annotation_tasks[:4]  # top 4 tasks
        styles = list(PromptStyle)
        difficulties = list(DifficultyLevel)

        candidates: List[VLMPrompt] = []
        for task, style, diff, ctx in itertools.product(tasks, styles, difficulties, contexts):
            try:
                p = self.build(incident, task, style, diff, ctx, class_id)
                candidates.append(p)
            except Exception:
                continue

        # Rank by quality and deduplicate by (task, style, difficulty)
        candidates.sort(key=lambda p: p.composite_score(), reverse=True)
        seen = set()
        result = []
        for p in candidates:
            key = (p.annotation_task, p.prompt_style, p.difficulty)
            if key not in seen:
                seen.add(key)
                result.append(p)
            if len(result) >= max_per_incident:
                break
        return result


if __name__ == "__main__":
    from taxonomy import INCIDENT_TYPES, INCIDENT_CLASSES
    builder = PromptBuilder(seed=0)
    it = INCIDENT_TYPES["it_unattended_vehicle"]
    ctx = SceneContext(
        LocationType.PARKING_FACILITY, CameraAngle.HIGH_ANGLE,
        LightingCondition.DAYLIGHT, OcclusionLevel.NONE,
        "sparse", "clear", "afternoon"
    )
    p = builder.build(it, AnnotationTask.DETECTION, PromptStyle.DESCRIPTIVE,
                      DifficultyLevel.MEDIUM, ctx, "ic_vehicle")
    print(f"ID: {p.prompt_id}")
    print(f"Quality: {p.quality_scores}")
    print(f"\nSystem: {p.system_instruction[:80]}...")
    print(f"\nPrompt: {p.user_prompt[:200]}...")
    print(f"\nHint: {p.assistant_hint}")
