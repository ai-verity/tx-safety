# VLM Prompt Generation Framework

A data factory that automatically generates structured fine-tuning prompts for Vision Language Models (VLMs) used in **public safety and security surveillance** applications (CCTV, airports, transit hubs, public spaces, etc.).

---

## What It Does

Training a VLM to detect security incidents requires thousands of diverse, high-quality prompt-response pairs. Writing them manually is impractical. This framework generates them automatically by combining:

- A **domain taxonomy** of 18 security incident types with 30+ visual signals
- A **prompt construction engine** that generates all combinations across tasks, styles, difficulty levels, and scene contexts
- A **deterministic augmentation engine** that enriches each prompt with paraphrases, hard negatives, and reasoning chains

**Output:** A single structured JSON file containing 1,500–3,000 ready-to-use fine-tuning prompts — no internet, no API calls, no GPU required.

---

## Project Structure

```
vlm_framework_source/
├── core/
│   ├── taxonomy.py          # Domain knowledge: incident types, signals, metadata
│   └── framework.py         # Prompt builder, quality scorer, scene context
├── models/
│   ├── synthesis_engine.py  # Deterministic text augmentation (default, offline)
│   └── llm_engine.py        # Optional local LLM augmentation (requires GPU)
└── generators/
    ├── run_pipeline.py      # Production entry point (recommended)
    └── generate_prompts.py  # CLI entry point with configurable flags
```

---

## Quick Start

No installation required. Python 3.8+ standard library only.

```bash
cd generators
mkdir -p output
python run_pipeline.py
```

Output file: `generators/output/vlm_prompts_generated.json`

---

## Inputs

The application takes **no external data files**. All inputs are sourced from the hardcoded domain knowledge in the source files.

### Domain Knowledge (the real inputs)

| File | What it defines |
|---|---|
| `core/taxonomy.py` | 18 incident types, 30+ visual signals, severity levels, counterfactual cues |
| `core/framework.py` | Prompt templates, 4 styles, 3 difficulty levels, 6 annotation tasks |
| `generators/run_pipeline.py` | 20 scene context configurations (location × lighting × camera × weather) |

### Taxonomy — Incident Classes

| Class | Incident Types |
|---|---|
| Vehicle | Unattended vehicle, wrong-way vehicle, vehicle convoy, abandoned object from vehicle |
| Crowd | Overcrowding, crowd surge, public disorder, queue jumping |
| Perimeter | Perimeter breach, fence climbing, unauthorised access |
| Dumping | Illegal dumping, fly-tipping, hazardous material disposal |
| Behavioural/Threat | Suspicious package, physical altercation, loitering, aggressive behaviour |

### Prompt Dimensions (combinatorial)

| Dimension | Options |
|---|---|
| Annotation task | detection, vqa, grounding, captioning, temporal_change, counting |
| Prompt style | terse, descriptive, socratic, chain_of_thought |
| Difficulty | easy, medium, hard |
| Location type | 11 types (airport, transit hub, parking facility, urban street, etc.) |
| Camera angle | overhead, high_angle, eye_level, low_angle, fisheye, PTZ |
| Lighting | daylight, overcast, twilight, night_illuminated, night_lowlight, infrared, thermal |
| Occlusion | none, partial, heavy |

---

## Output

A single JSON file with four top-level sections:

### 1. `taxonomy`
Full domain index: incident classes, incident types, signal library with visual attributes and negative attributes.

### 2. `framework_spec`
Records the exact configuration used: all tasks, styles, difficulties, scene parameters, and quality scorer weights.

### 3. `generation_stats`
Run summary: total prompts, elapsed time, distribution by incident type / task / severity / difficulty, quality score statistics.

### 4. `prompts`
The main payload — each record contains:

```json
{
  "prompt_id": "it_unattended_vehicle_detection_chain_of_thought_hard_001",
  "incident_type_id": "it_unattended_vehicle",
  "incident_class_id": "ic_vehicle",
  "severity": "MODERATE",
  "annotation_task": "detection",
  "prompt_style": "chain_of_thought",
  "difficulty": "hard",
  "scene_context": {
    "location_type": "parking_facility",
    "camera_angle": "high_angle",
    "lighting_condition": "night_illuminated",
    "occlusion_level": "partial",
    "crowd_density": "sparse",
    "weather": "clear",
    "time_of_day": "night"
  },
  "system_instruction": "You are a surveillance analyst...",
  "user_prompt": "Step 1: Identify all vehicles...",
  "assistant_hint": "Look for vehicles parked beyond normal dwell time...",
  "quality_scores": {
    "composite": 0.74,
    "specificity": 0.80,
    "discriminability": 0.70,
    "completeness": 0.75,
    "temporal_clarity": 0.65
  },
  "augmented_prompts": ["Determine whether...", "Assess the image for..."],
  "hard_negative_examples": ["A service vehicle with an operator visible..."],
  "reasoning_chain_template": "Step 1: ... Step 2: ... Step 3: ...",
  "vqa_answer_options": { "yes": "...", "no": "...", "uncertain": "..." },
  "signal_ids": ["sig_prolonged_stationary", "sig_restricted_zone"],
  "frame_requirement": "multi",
  "temporal_window_sec": 300
}
```

---

## Running the Application

### Option 1 — Production Pipeline (recommended)

No flags, no dependencies, fully deterministic.

```bash
cd generators
python run_pipeline.py
```

Configuration is set directly in `run_pipeline.py`:

```python
ProductionGenerator(
    max_per_incident=24,      # max prompts kept per incident type
    min_quality=0.20,         # quality filter threshold (0–1)
    seed=42,                  # RNG seed for reproducibility
    output_path="./output/vlm_prompts_generated.json",
    include_augmentation=True
)
```

### Option 2 — CLI Pipeline (configurable flags)

```bash
cd generators

# Fast run — no augmentation
python generate_prompts.py \
  --output ./output/prompts.json \
  --no-augment

# Full run with all options
python generate_prompts.py \
  --output ./output/prompts.json \
  --max-per-incident 24 \
  --min-quality 0.20 \
  --no-augment \
  --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--output` | `./output/vlm_prompts_generated.json` | Output JSON path |
| `--max-per-incident` | `20` | Max prompts per incident type |
| `--min-quality` | `0.25` | Quality filter threshold (0–1) |
| `--augment` / `--no-augment` | augment on | Enable/disable text augmentation |
| `--seed` | `42` | RNG seed for reproducibility |
| `--model` | auto-detect | Local model name for LLM augmentation |

---

## Generation Pipeline

```
taxonomy.py  ──────────────────────────────────────┐
(18 incident types × 30+ signals)                  │
                                                    ▼
framework.py ──────────────────────────►  PromptBuilder
(4 styles × 3 difficulties × 6 tasks)              │
                                                    │  ~2,000–5,000 candidates
FULL_CONTEXT_MATRIX ───────────────────►            │  per incident type
(20 scene configurations)                           │
                                                    ▼
                                          Quality Scoring
                                          Filter (min 0.20)
                                          Deduplication
                                          Top 24 selected
                                                    │
                                                    ▼
                                    DeterministicSynthesisEngine
                                    - 3 paraphrases (formal / operational / analytical)
                                    - 2 hard negatives
                                    - Chain-of-thought template
                                    - VQA answer options
                                                    │
                                                    ▼
                                          JSON export
                                          ~1,500–3,000 prompts
```

---

## Quality Scoring

Each prompt is scored across four dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Specificity | 30% | How precisely the prompt identifies the target incident |
| Discriminability | 30% | How well it excludes false positives / benign scenes |
| Completeness | 25% | Coverage of all required visual elements and signals |
| Temporal Clarity | 15% | Whether time-based cues are included when needed |

Prompts below the `min_quality` threshold are discarded before export.

---

## What To Do With The Output

This framework generates the **text side** of VLM training pairs. The full workflow is:

```
[This app]              [You provide]           [Training]
Generate prompts   +    Real surveillance   →   Fine-tune a VLM
(JSON)                  images paired           (LLaVA, InternVL,
                        with prompts            Qwen-VL, PaliGemma)
```

1. **Run this app** to get the structured prompt JSON
2. **Collect images** matching each scene context (parking lots, transit hubs, etc.)
3. **Pair images with prompts** by incident type and scene context
4. **Fine-tune a VLM** using a framework like Hugging Face `transformers` or `LLaMA-Factory`
5. **Deploy** the fine-tuned model to analyse real camera feeds

---

## Extending the Framework

### Add a new incident type
Edit `core/taxonomy.py` — add a new `IncidentType` entry with signals, severity, annotation tasks, and counterfactual cues, then register it in the appropriate `IncidentClass`.

### Add a new prompt style
Edit `core/framework.py` — add a value to the `PromptStyle` enum and add the corresponding template in `TemplateLibrary`.

### Add a new scene context
Edit `generators/run_pipeline.py` — add a tuple to `FULL_CONTEXT_MATRIX` using existing `LocationType`, `CameraAngle`, `LightingCondition`, and `OcclusionLevel` values.

### Change quality weights
Edit `core/framework.py` in the `PromptQualityScorer` class — adjust the `WEIGHTS` dictionary (values must sum to 1.0).

---

## Design Principles

| Property | Detail |
|---|---|
| Fully offline | No API calls, no model downloads needed |
| Deterministic | Seeded RNG — same seed always produces same output |
| Quality-gated | Low-scoring prompts filtered before export |
| Diverse by design | Deduplicates on (task, style, difficulty) to prevent repetition |
| No dependencies | Runs on Python 3.8+ standard library only |
