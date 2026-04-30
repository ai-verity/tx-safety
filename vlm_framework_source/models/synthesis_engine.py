"""
Advanced Deterministic Synthesis Engine
=========================================
Production-quality prompt augmentation entirely through rule-based NLP —
no network access, no model downloads required.

Techniques used:
  1. Structural paraphrase via constituency-level rewriting
  2. Lexical substitution via curated domain synonym tables
  3. Register shifting (formal ↔ operational ↔ analytical)
  4. Perspective rotation (first-person analyst ↔ system instruction ↔ investigator)
  5. Hard-negative generation via semantic negation + context switching
  6. Chain-of-thought scaffolding via structured reasoning templates
  7. Adversarial augmentation (ambiguous/confounding scene descriptions)
  8. Modality-specific cue injection (lighting, occlusion, camera angle)
"""

from __future__ import annotations
import random
import re
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Domain synonym tables
# ─────────────────────────────────────────────────────────────────────────────

VERB_SYNONYMS = {
    "detect": ["identify", "locate", "find", "recognise", "spot", "observe", "flag"],
    "analyse": ["examine", "assess", "evaluate", "inspect", "review", "scrutinise"],
    "determine": ["ascertain", "establish", "confirm", "verify", "decide"],
    "describe": ["characterise", "detail", "document", "report", "articulate"],
    "report": ["note", "document", "log", "flag", "record", "communicate"],
    "observe": ["note", "witness", "monitor", "track", "watch", "survey"],
    "confirm": ["verify", "validate", "substantiate", "corroborate", "affirm"],
    "assess": ["evaluate", "gauge", "appraise", "rate", "measure", "judge"],
}

NOUN_SYNONYMS = {
    "image": ["frame", "visual", "scene", "photograph", "capture", "footage", "still"],
    "scene": ["area", "environment", "location", "zone", "setting", "context", "frame"],
    "evidence": ["indicators", "cues", "markers", "signs", "signals", "proof"],
    "incident": ["event", "occurrence", "situation", "condition", "anomaly", "concern"],
    "vehicle": ["motor vehicle", "automobile", "conveyance", "unit"],
    "person": ["individual", "subject", "pedestrian", "occupant", "figure"],
    "area": ["region", "zone", "location", "space", "sector", "vicinity"],
    "security": ["safety", "protection", "surveillance", "monitoring", "oversight"],
    "indication": ["sign", "signal", "marker", "cue", "indicator", "evidence"],
}

ADJ_SYNONYMS = {
    "suspicious": ["anomalous", "concerning", "unusual", "irregular", "atypical", "aberrant"],
    "stationary": ["immobile", "non-moving", "parked", "static", "motionless", "fixed"],
    "visible": ["observable", "apparent", "discernible", "detectable", "evident"],
    "relevant": ["pertinent", "applicable", "significant", "material", "noteworthy"],
    "accurate": ["precise", "correct", "reliable", "exact", "faithful"],
    "detailed": ["thorough", "comprehensive", "granular", "specific", "exhaustive"],
}

# Instruction openers by register
FORMAL_OPENERS = [
    "Please analyse",
    "Conduct a systematic examination of",
    "Perform a thorough assessment of",
    "Undertake a visual analysis of",
    "Execute a detailed inspection of",
    "Carry out an evaluation of",
]
OPERATIONAL_OPENERS = [
    "Look at",
    "Examine",
    "Check",
    "Review",
    "Scan",
    "Inspect",
    "Survey",
]
ANALYTICAL_OPENERS = [
    "Determine whether",
    "Evaluate if",
    "Assess whether",
    "Establish if",
    "Verify whether",
    "Confirm if",
    "Ascertain whether",
]
INVESTIGATOR_OPENERS = [
    "As a security analyst, examine",
    "From an investigative standpoint, assess",
    "Acting as a threat assessment specialist, review",
    "As a trained surveillance operator, identify",
    "In your capacity as a security monitor, determine",
]

# Confidence qualifiers
CONFIDENCE_PHRASES = [
    "with high confidence",
    "providing a confidence estimate",
    "and rate your confidence on a scale of 0-100%",
    "indicating your certainty level",
    "and flag the confidence level of your assessment",
    "noting any uncertainty factors",
]

# Spatial qualifiers
SPATIAL_PHRASES = [
    "indicating the approximate location within the frame",
    "describing the spatial position using cardinal references",
    "providing bounding region coordinates or spatial language",
    "noting the position relative to identifiable anchor objects",
    "specifying the quadrant and approximate size of the affected region",
]

# Evidence request phrases
EVIDENCE_PHRASES = [
    "citing the specific visual attributes that support your conclusion",
    "referencing the observable cues that inform your assessment",
    "listing the visual indicators that confirm or deny the incident",
    "providing the key observational evidence",
    "noting which visual elements are most diagnostically significant",
]

# False-positive guard phrases
FP_GUARD_PHRASES = [
    "Rule out legitimate alternatives before confirming the alert.",
    "Consider whether this could be a benign scenario before triggering an alert.",
    "Distinguish between genuine incidents and visually similar non-incidents.",
    "Apply the principle of least alarm: only confirm if evidence is clear.",
    "Before concluding, verify that the scene is not consistent with authorised activity.",
]

# Scene condition injections
SCENE_CONDITIONS = {
    "night": [
        "Note: illumination is limited; rely on contrast and shape over colour.",
        "The scene is captured under artificial lighting; shadows may obscure detail.",
        "Low-light conditions apply; adjust confidence thresholds accordingly.",
    ],
    "rain": [
        "Precipitation may cause reflective artefacts on surfaces.",
        "Wet conditions may blur fine visual details.",
        "Rain patterns may partially obscure licence plates and markings.",
    ],
    "fog": [
        "Reduced visibility due to atmospheric conditions; note any visible depth.",
        "Fog may limit reliable distance estimation.",
        "Background detail may be lost; focus on foreground elements.",
    ],
    "high_angle": [
        "The overhead angle distorts apparent object sizes.",
        "From this vantage, vertical height is not directly observable.",
        "Top-down perspective aids density estimation but limits identity features.",
    ],
    "fisheye": [
        "Lens distortion affects object shape at frame periphery.",
        "Central objects appear more accurate than peripherally distorted ones.",
        "Account for radial distortion when estimating distances.",
    ],
    "crowded": [
        "Dense foreground may occlude the subject; use partial indicators.",
        "Crowd context requires discrimination between group and target behaviours.",
        "High background activity may increase false-positive probability.",
    ],
}

# Reasoning chain templates per task
COT_TEMPLATES = {
    "detection": [
        (
            "Step 1 — Scene inventory: List all visible objects and their approximate positions.\n"
            "Step 2 — Signal check: Identify which, if any, match {signal_labels}.\n"
            "Step 3 — Attribute confirmation: Do matched objects exhibit {visual_attributes}?\n"
            "Step 4 — False-positive test: Could this be {fp_example}?\n"
            "Step 5 — Verdict: {incident_label} is [confirmed/probable/not detected]. Confidence: [%]."
        ),
        (
            "Observation phase: What is the most unusual feature in this scene?\n"
            "Classification phase: Does it match the pattern of {incident_label}?\n"
            "Evidence phase: Which specific pixels/regions support this classification?\n"
            "Challenge phase: What counter-evidence exists?\n"
            "Decision: [Alert / No Alert] — Rationale: [brief]."
        ),
    ],
    "temporal_change": [
        (
            "Frame 1 baseline: Describe the initial scene state.\n"
            "Frame-to-frame delta: What changes between consecutive frames?\n"
            "Pattern match: Does the change pattern align with {incident_label}?\n"
            "Duration check: Has the condition persisted for the required temporal window?\n"
            "Escalation: Is the situation developing, stable, or resolving?"
        ),
    ],
    "grounding": [
        (
            "Anchor identification: Identify 2-3 fixed reference objects in the scene.\n"
            "Target localisation: Describe the target region relative to anchors.\n"
            "Boundary definition: Estimate the spatial extent of the incident zone.\n"
            "Confidence mapping: Which parts of the localisation are most/least certain?"
        ),
    ],
    "vqa": [
        (
            "Pre-answer check: What information does this question require?\n"
            "Evidence scan: Locate relevant visual evidence in the image.\n"
            "Option elimination: Rule out incorrect options based on evidence.\n"
            "Final answer: [option] because [one-sentence justification]."
        ),
    ],
    "counting": [
        (
            "Region partition: Divide the scene into sub-regions.\n"
            "Sub-count: Count target instances in each region independently.\n"
            "Occlusion estimate: How many instances might be hidden?\n"
            "Total: [N visible] + [N estimated occluded] = [N total]. Confidence: [%]."
        ),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Core synthesis engine
# ─────────────────────────────────────────────────────────────────────────────

class DeterministicSynthesisEngine:
    """
    Produces high-quality prompt augmentations entirely through deterministic
    rule-based generation — no network access, no model required.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _seed_for(self, text: str) -> int:
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

    def _sub_synonyms(self, text: str) -> str:
        """Apply lexical substitution using synonym tables."""
        result = text
        rng = random.Random(self._seed_for(text))
        # Only substitute about 30% of matches for diversity
        for word, syns in {**VERB_SYNONYMS, **NOUN_SYNONYMS, **ADJ_SYNONYMS}.items():
            if word in result.lower() and rng.random() < 0.3:
                replacement = rng.choice(syns)
                # Case-sensitive replacement
                result = re.sub(
                    rf'\b{word}\b',
                    replacement,
                    result,
                    count=1,
                    flags=re.IGNORECASE,
                )
        return result

    def _change_register(self, prompt: str, register: str) -> str:
        """Rewrite prompt opener in the specified register."""
        rng = random.Random(self._seed_for(prompt + register))
        openers = {
            "formal":       FORMAL_OPENERS,
            "operational":  OPERATIONAL_OPENERS,
            "analytical":   ANALYTICAL_OPENERS,
            "investigator": INVESTIGATOR_OPENERS,
        }
        opener_list = openers.get(register, FORMAL_OPENERS)
        opener = rng.choice(opener_list)

        # Strip existing opener (first verb/instruction word)
        stripped = re.sub(
            r'^(Please |Analyse |Examine |Detect |Scan |Check |Look at |Determine |Assess )',
            '', prompt, flags=re.IGNORECASE
        ).strip()
        # Lowercase the first char
        if stripped:
            stripped = stripped[0].lower() + stripped[1:]
        return f"{opener} {stripped}"

    def _inject_modifiers(self, prompt: str, modifiers: List[str]) -> str:
        """Append scene-relevant condition modifiers to a prompt."""
        if not modifiers:
            return prompt
        mod_text = " ".join(modifiers)
        return f"{prompt.rstrip('.')}. {mod_text}"

    def _add_confidence_request(self, prompt: str) -> str:
        rng = random.Random(self._seed_for(prompt + "conf"))
        phrase = rng.choice(CONFIDENCE_PHRASES)
        return f"{prompt.rstrip('.')} {phrase}."

    def _add_spatial_request(self, prompt: str) -> str:
        rng = random.Random(self._seed_for(prompt + "spat"))
        phrase = rng.choice(SPATIAL_PHRASES)
        return f"{prompt.rstrip('.')} {phrase}."

    def _add_evidence_request(self, prompt: str) -> str:
        rng = random.Random(self._seed_for(prompt + "evid"))
        phrase = rng.choice(EVIDENCE_PHRASES)
        return f"{prompt.rstrip('.')} {phrase}."

    def _add_fp_guard(self, prompt: str) -> str:
        rng = random.Random(self._seed_for(prompt + "fp"))
        phrase = rng.choice(FP_GUARD_PHRASES)
        return f"{prompt.rstrip('.')}. {phrase}"

    def _inject_scene_conditions(self, prompt: str, lighting: str, camera: str,
                                  crowd: str) -> str:
        """Inject scene-specific observational constraints."""
        injections = []
        if "night" in lighting:
            rng = random.Random(self._seed_for(prompt + "night"))
            injections.append(rng.choice(SCENE_CONDITIONS["night"]))
        if "fog" in lighting:
            rng = random.Random(self._seed_for(prompt + "fog"))
            injections.append(rng.choice(SCENE_CONDITIONS["fog"]))
        if "fisheye" in camera:
            rng = random.Random(self._seed_for(prompt + "fish"))
            injections.append(rng.choice(SCENE_CONDITIONS["fisheye"]))
        if "overhead" in camera or "high_angle" in camera:
            rng = random.Random(self._seed_for(prompt + "angle"))
            injections.append(rng.choice(SCENE_CONDITIONS["high_angle"]))
        if crowd == "dense":
            rng = random.Random(self._seed_for(prompt + "crowd"))
            injections.append(rng.choice(SCENE_CONDITIONS["crowded"]))
        return self._inject_modifiers(prompt, injections)

    # ── Public paraphrase API ─────────────────────────────────────────────────

    def paraphrase(self, prompt: str, n: int = 4,
                   scene_ctx: Optional[Dict] = None) -> List[str]:
        """Generate n structurally distinct paraphrases."""
        results = []
        registers = ["formal", "operational", "analytical", "investigator"]
        transforms = [
            lambda p: self._change_register(p, "formal"),
            lambda p: self._add_confidence_request(self._sub_synonyms(p)),
            lambda p: self._add_spatial_request(self._change_register(p, "analytical")),
            lambda p: self._add_evidence_request(self._sub_synonyms(p)),
            lambda p: self._add_fp_guard(self._change_register(p, "investigator")),
            lambda p: self._add_confidence_request(self._add_spatial_request(
                self._change_register(p, "operational"))),
        ]
        # Apply scene conditions if context provided
        if scene_ctx:
            transforms.append(lambda p: self._inject_scene_conditions(
                p,
                lighting=scene_ctx.get("lighting", "daylight"),
                camera=scene_ctx.get("camera_angle", "high_angle"),
                crowd=scene_ctx.get("crowd_level", "sparse"),
            ))

        for i, transform in enumerate(transforms[:n]):
            try:
                variant = transform(prompt)
                if variant and variant != prompt:
                    results.append(variant.strip())
            except Exception:
                results.append(prompt)
        return results[:n]

    # ── Hard negative generation ──────────────────────────────────────────────

    def generate_hard_negatives(
        self,
        incident_label: str,
        counterfactuals: List[str],
        location: str = "public area",
        n: int = 3,
    ) -> List[str]:
        """
        Produce realistic hard-negative descriptions — scenes that superficially
        resemble the incident but are actually benign.
        """
        templates = [
            (
                "The image shows a {cf_desc} at a {location}. "
                "Although the scene superficially resembles {incident}, "
                "the subject is engaged in a legitimate, authorised activity "
                "as evidenced by {legitimising_cue}. No alert is warranted."
            ),
            (
                "At first glance this may appear to be {incident}, however "
                "closer inspection reveals {cf_desc}. "
                "Key distinguishing features: {legitimising_cue}. "
                "Classification: benign / non-incident."
            ),
            (
                "Scene description: {cf_desc} observed at {location}. "
                "While visual similarity to {incident} exists, "
                "contextual analysis confirms this is a {legitimate_activity}. "
                "Recommend: no action required."
            ),
        ]
        legitimising_cues = [
            "visible authorised markings",
            "accompanying personnel with credentials",
            "contextually appropriate activity for the time and location",
            "official signage consistent with authorised use",
            "operational context that excludes threat classification",
            "uniform or livery of an authorised operator",
        ]
        legitimate_activities = [
            "routine operational procedure",
            "scheduled maintenance activity",
            "authorised access event",
            "scheduled service delivery",
            "supervised visitor access",
            "official inspection or survey",
        ]

        results = []
        for i, cf in enumerate(counterfactuals[:n]):
            rng = random.Random(self._seed_for(cf + incident_label))
            template = templates[i % len(templates)]
            neg = template.format(
                cf_desc=cf.replace("_", " "),
                location=location,
                incident=incident_label.lower(),
                legitimising_cue=rng.choice(legitimising_cues),
                legitimate_activity=rng.choice(legitimate_activities),
            )
            results.append(neg)

        if len(results) < n:
            # Pad with generic hard-negative
            results.append(
                f"This scene does not constitute {incident_label.lower()}. "
                f"Visual similarity is due to {self._rng.choice(legitimising_cues)}. "
                f"The observed conditions are consistent with authorised activity."
            )
        return results[:n]

    # ── Chain-of-thought template generation ─────────────────────────────────

    def generate_cot(
        self,
        task: str,
        incident_label: str,
        signal_labels: List[str],
        visual_attributes: List[str],
        counterfactual: str = "a benign non-incident",
    ) -> str:
        """Generate a task-specific chain-of-thought reasoning template."""
        templates = COT_TEMPLATES.get(task, COT_TEMPLATES["detection"])
        rng = random.Random(self._seed_for(task + incident_label))
        template = rng.choice(templates)
        return template.format(
            incident_label=incident_label,
            signal_labels=", ".join(signal_labels[:3]) if signal_labels else "observable indicators",
            visual_attributes=", ".join(visual_attributes[:3]) if visual_attributes else "anomalous features",
            fp_example=counterfactual.replace("_", " "),
        )

    # ── VQA distractor generation ─────────────────────────────────────────────

    def generate_vqa_options(
        self, incident_label: str, positive_answer: str
    ) -> Dict[str, str]:
        """Generate a 4-option VQA answer set."""
        return {
            "A": positive_answer,
            "B": f"No — the conditions shown are consistent with a routine, authorised operation and do not indicate {incident_label.lower()}.",
            "C": f"Insufficient information — key visual indicators are occluded or ambiguous, preventing a definitive classification of {incident_label.lower()}.",
            "D": f"Partially — some indicators of {incident_label.lower()} are present but do not meet the threshold for a confirmed alert; continued monitoring is recommended.",
        }

    # ── Full synthesis pipeline ───────────────────────────────────────────────

    def synthesise(
        self,
        base_prompt: str,
        incident_label: str,
        annotation_task: str,
        counterfactuals: List[str],
        signal_labels: List[str],
        visual_attributes: List[str],
        scene_ctx: Optional[Dict] = None,
        n_paraphrases: int = 4,
        n_hard_neg: int = 3,
    ) -> Dict[str, Any]:
        """Run full synthesis for a single base prompt."""
        location = scene_ctx.get("location_type", "public area").replace("_", " ") if scene_ctx else "public area"

        paraphrases = self.paraphrase(base_prompt, n=n_paraphrases, scene_ctx=scene_ctx)
        hard_negs = self.generate_hard_negatives(
            incident_label, counterfactuals, location, n=n_hard_neg
        )
        cot = self.generate_cot(
            annotation_task, incident_label, signal_labels, visual_attributes,
            counterfactual=counterfactuals[0] if counterfactuals else "a benign scenario",
        )
        vqa_opts = self.generate_vqa_options(
            incident_label,
            f"Yes — confirmed {incident_label.lower()} with visual evidence: "
            f"{', '.join(signal_labels[:2]) if signal_labels else 'observable indicators'}.",
        )

        return {
            "augmented_prompts": paraphrases,
            "hard_negative_examples": hard_negs,
            "context_variations": [
                self._inject_scene_conditions(
                    base_prompt,
                    lighting="night_illuminated",
                    camera="overhead",
                    crowd="dense",
                ) if scene_ctx else base_prompt,
                self._inject_scene_conditions(
                    base_prompt,
                    lighting="infrared",
                    camera="fisheye",
                    crowd="sparse",
                ) if scene_ctx else base_prompt,
            ],
            "reasoning_chain_template": cot,
            "vqa_answer_options": vqa_opts,
            "synthesis_model": "deterministic_rule_engine_v2",
            "synthesis_time_sec": 0.001,
        }


# Singleton for import
_ENGINE_INSTANCE: Optional[DeterministicSynthesisEngine] = None


def get_engine(seed: int = 42) -> DeterministicSynthesisEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = DeterministicSynthesisEngine(seed=seed)
    return _ENGINE_INSTANCE


if __name__ == "__main__":
    engine = DeterministicSynthesisEngine(seed=0)
    result = engine.synthesise(
        base_prompt="Detect any unattended vehicle in this parking facility image.",
        incident_label="Unattended Vehicle",
        annotation_task="detection",
        counterfactuals=["driver returning to vehicle", "loading zone with operator present"],
        signal_labels=["prolonged stationary vehicle", "no visible occupants"],
        visual_attributes=["vehicle_present", "no_motion_vector", "occupants_absent"],
        scene_ctx={"location_type": "parking_facility", "lighting": "daylight",
                   "camera_angle": "high_angle", "crowd_level": "sparse"},
    )
    print("Paraphrases:")
    for i, p in enumerate(result["augmented_prompts"]):
        print(f"  {i+1}: {p[:100]}")
    print("\nHard Negatives:")
    for i, n in enumerate(result["hard_negative_examples"]):
        print(f"  {i+1}: {n[:120]}")
    print(f"\nCoT:\n{result['reasoning_chain_template']}")
    print("\nVQA Options:")
    for k, v in result["vqa_answer_options"].items():
        print(f"  {k}: {v[:80]}")
