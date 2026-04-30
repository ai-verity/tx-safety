"""
VLM Prompt Taxonomy
===================
Defines the canonical ontology for visual language model prompt generation
across public-safety, infrastructure-security, and behavioral-surveillance
domains.  All categories are domain-agnostic; no venue-specific names appear.

Hierarchy
---------
Domain → IncidentClass → IncidentType → ObservationSignal
Each node carries metadata needed for prompt construction:
  - severity (1-5)
  - temporal_sensitivity (immediate / short-term / long-term)
  - visual_complexity (low / medium / high)
  - required_context_frames (1 / multi)
  - annotation_task (detection / classification / grounding / caption / VQA)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enumerated primitives
# ─────────────────────────────────────────────────────────────────────────────

class Severity(int, Enum):
    INFORMATIONAL  = 1
    LOW            = 2
    MODERATE       = 3
    HIGH           = 4
    CRITICAL       = 5


class TemporalSensitivity(str, Enum):
    IMMEDIATE   = "immediate"     # sub-second response needed
    SHORT_TERM  = "short_term"    # 1–60 seconds
    LONG_TERM   = "long_term"     # minutes to hours


class VisualComplexity(str, Enum):
    LOW    = "low"     # single salient object, clean background
    MEDIUM = "medium"  # multiple objects, partial occlusion
    HIGH   = "high"    # crowds, dense scenes, overlapping elements


class AnnotationTask(str, Enum):
    DETECTION        = "detection"         # localise object / event
    CLASSIFICATION   = "classification"   # label scene or object
    GROUNDING        = "grounding"         # phrase → bounding box
    CAPTIONING       = "captioning"        # free-text description
    VQA              = "vqa"               # binary or multiple-choice Q&A
    COUNTING         = "counting"          # enumerate instances
    TEMPORAL_CHANGE  = "temporal_change"   # compare frames over time
    ATTRIBUTE_RECOG  = "attribute_recog"   # colour, posture, object state
    SCENE_GRAPH      = "scene_graph"       # entity–relation–entity triplets


class FrameRequirement(str, Enum):
    SINGLE = "single"
    MULTI  = "multi"


class CameraAngle(str, Enum):
    OVERHEAD    = "overhead"
    HIGH_ANGLE  = "high_angle"
    EYE_LEVEL   = "eye_level"
    LOW_ANGLE   = "low_angle"
    FISHEYE     = "fisheye"
    PTZ         = "ptz"


class LightingCondition(str, Enum):
    DAYLIGHT       = "daylight"
    OVERCAST       = "overcast"
    TWILIGHT       = "twilight"
    NIGHT_ILLUMINATED = "night_illuminated"
    NIGHT_LOWLIGHT    = "night_lowlight"
    INFRARED          = "infrared"
    THERMAL           = "thermal"


class OcclusionLevel(str, Enum):
    NONE     = "none"
    PARTIAL  = "partial"
    HEAVY    = "heavy"


# ─────────────────────────────────────────────────────────────────────────────
# Core data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObservationSignal:
    """
    An atomic visual cue that a VLM should learn to detect.
    Maps directly to a grounding-level prompt element.
    """
    signal_id: str
    label: str
    description: str
    visual_attributes: List[str]       # e.g. ["stationary", "no_occupants"]
    negative_attributes: List[str]     # what it is NOT (for contrastive prompts)
    annotation_tasks: List[AnnotationTask]
    frame_requirement: FrameRequirement = FrameRequirement.SINGLE
    temporal_window_sec: Optional[int] = None   # how long event must persist


@dataclass
class IncidentType:
    """
    A concrete real-world event the system must recognise.
    Groups one or more ObservationSignals.
    """
    type_id: str
    label: str
    description: str
    severity: Severity
    temporal_sensitivity: TemporalSensitivity
    visual_complexity: VisualComplexity
    signals: List[ObservationSignal]
    related_types: List[str] = field(default_factory=list)   # type_ids
    annotation_tasks: List[AnnotationTask] = field(default_factory=list)
    frame_requirement: FrameRequirement = FrameRequirement.SINGLE
    # Prompt construction hints
    prompt_focus_objects: List[str] = field(default_factory=list)
    prompt_spatial_relations: List[str] = field(default_factory=list)
    prompt_temporal_cues: List[str] = field(default_factory=list)
    counterfactual_cues: List[str] = field(default_factory=list)   # hard negatives


@dataclass
class IncidentClass:
    """
    A thematic grouping of IncidentTypes (e.g. Vehicle, Crowd, Perimeter).
    """
    class_id: str
    label: str
    description: str
    incident_types: List[IncidentType]
    domain_tags: List[str] = field(default_factory=list)


@dataclass
class Domain:
    """
    Top-level taxonomy node.
    """
    domain_id: str
    label: str
    description: str
    incident_classes: List[IncidentClass]


# ─────────────────────────────────────────────────────────────────────────────
# Observation Signal Library
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS: Dict[str, ObservationSignal] = {

    # ── Vehicle signals ──────────────────────────────────────────────────────
    "veh_stationary_prolonged": ObservationSignal(
        signal_id="veh_stationary_prolonged",
        label="Prolonged Stationary Vehicle",
        description="Vehicle has not moved for longer than the contextually normal dwell time",
        visual_attributes=["vehicle_present", "no_motion_vector", "occupants_absent_or_unverifiable",
                           "engine_state_unknown", "consistent_position_across_frames"],
        negative_attributes=["actively_loading_unloading", "driver_visible_and_present",
                              "designated_parking_zone_marker"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA, AnnotationTask.ATTRIBUTE_RECOG],
        frame_requirement=FrameRequirement.MULTI,
        temporal_window_sec=300,
    ),
    "veh_no_plates": ObservationSignal(
        signal_id="veh_no_plates",
        label="Missing or Obscured License Plates",
        description="Vehicle has no visible registration plates or plates are deliberately obscured",
        visual_attributes=["vehicle_present", "plate_region_empty_or_obscured",
                           "no_alphanumeric_sequence_visible"],
        negative_attributes=["plate_partially_dirty", "plate_at_angle_but_readable"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING, AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "veh_hazard_indicators": ObservationSignal(
        signal_id="veh_hazard_indicators",
        label="Vehicle Hazard Indicators Active",
        description="Hazard lights flashing, possibly indicating distress or improper parking",
        visual_attributes=["flashing_amber_lights", "four_way_indicator_pattern"],
        negative_attributes=["turn_signal_unilateral", "brake_lights_only"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.ATTRIBUTE_RECOG],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "veh_unattended_restricted": ObservationSignal(
        signal_id="veh_unattended_restricted",
        label="Unattended Vehicle in Restricted Zone",
        description="Vehicle parked or stopped in a zone where unattended vehicles are prohibited",
        visual_attributes=["vehicle_present", "restricted_zone_markings_visible",
                           "no_authorised_personnel_nearby"],
        negative_attributes=["authorised_service_vehicle", "emergency_vehicle_markings"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING,
                          AnnotationTask.CLASSIFICATION],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "veh_abandoned_degraded": ObservationSignal(
        signal_id="veh_abandoned_degraded",
        label="Vehicle Showing Abandonment Indicators",
        description="Structural, surface or contextual signs of abandonment (flat tyres, debris accumulation, damage)",
        visual_attributes=["flat_or_deflated_tyres", "debris_accumulation_on_or_around_vehicle",
                           "broken_windows_or_body_damage", "vegetation_growth_adjacent",
                           "dust_or_dirt_accumulation"],
        negative_attributes=["freshly_parked", "clean_exterior"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.ATTRIBUTE_RECOG],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "veh_oversized_wrong_zone": ObservationSignal(
        signal_id="veh_oversized_wrong_zone",
        label="Oversized or Non-Conforming Vehicle in Prohibited Area",
        description="Commercial, recreational or non-standard vehicle occupying a disallowed zone",
        visual_attributes=["large_vehicle_form_factor", "non_passenger_vehicle_type",
                           "zone_signage_visible_prohibiting"],
        negative_attributes=["designated_commercial_loading_zone", "authorised_delivery_markings"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.GROUNDING],
        frame_requirement=FrameRequirement.SINGLE,
    ),

    # ── Crowd / Pedestrian signals ────────────────────────────────────────────
    "crowd_density_threshold": ObservationSignal(
        signal_id="crowd_density_threshold",
        label="Crowd Density Exceeding Safe Threshold",
        description="Number of persons per unit area exceeds operationally safe limits",
        visual_attributes=["high_person_density", "limited_movement_space",
                           "overlapping_bounding_boxes", "queue_compression"],
        negative_attributes=["orderly_spaced_queue", "sparse_pedestrian_flow"],
        annotation_tasks=[AnnotationTask.COUNTING, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "crowd_rapid_aggregation": ObservationSignal(
        signal_id="crowd_rapid_aggregation",
        label="Rapid Crowd Aggregation",
        description="A group forms quickly around a focal point — may indicate altercation, collapse or attraction",
        visual_attributes=["increasing_person_count_over_frames", "centripetal_motion_vectors",
                           "focal_point_visible", "bystander_orientation_convergent"],
        negative_attributes=["dispersing_crowd", "stationary_queue"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.COUNTING],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "crowd_dispersal_stampede": ObservationSignal(
        signal_id="crowd_dispersal_stampede",
        label="Panicked or Forced Crowd Dispersal",
        description="Sudden high-velocity outward motion of crowd — potential stampede or evacuation",
        visual_attributes=["high_velocity_person_motion_vectors", "centrifugal_motion_pattern",
                           "falling_persons", "dropped_items"],
        negative_attributes=["normal_egress_flow", "end_of_event_dispersal"],
        annotation_tasks=[AnnotationTask.TEMPORAL_CHANGE, AnnotationTask.DETECTION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "crowd_loitering": ObservationSignal(
        signal_id="crowd_loitering",
        label="Loitering Individual or Group",
        description="Person(s) remaining stationary in a location without apparent legitimate purpose",
        visual_attributes=["stationary_person", "extended_dwell_time",
                           "non_purposeful_body_orientation", "repeated_area_revisit"],
        negative_attributes=["waiting_in_designated_queue", "seated_in_public_seating_area"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
        temporal_window_sec=120,
    ),
    "crowd_altercation": ObservationSignal(
        signal_id="crowd_altercation",
        label="Physical Altercation or Fight",
        description="Two or more persons engaged in physical conflict",
        visual_attributes=["rapid_limb_motion", "contact_between_persons",
                           "non_upright_posture", "bystander_retreat_motion"],
        negative_attributes=["greeting_embrace", "sports_activity"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "crowd_social_distancing": ObservationSignal(
        signal_id="crowd_social_distancing",
        label="Proximity Violation / Social Distancing Breach",
        description="Inter-person spacing below mandated minimum in regulated environments",
        visual_attributes=["inter_person_distance_below_threshold",
                           "high_contact_rate_between_persons"],
        negative_attributes=["family_unit", "socially_accepted_proximity"],
        annotation_tasks=[AnnotationTask.COUNTING, AnnotationTask.VQA, AnnotationTask.GROUNDING],
        frame_requirement=FrameRequirement.SINGLE,
    ),

    # ── Perimeter / Access Control signals ───────────────────────────────────
    "perim_fence_breach": ObservationSignal(
        signal_id="perim_fence_breach",
        label="Physical Perimeter Breach",
        description="Person or object passing through, over or under a security barrier",
        visual_attributes=["barrier_deformation_or_opening", "person_in_motion_over_barrier",
                           "cut_or_bent_material", "unauthorised_zone_entry"],
        negative_attributes=["authorised_gate_passage", "maintenance_personnel_with_escort"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "perim_gate_left_open": ObservationSignal(
        signal_id="perim_gate_left_open",
        label="Access Gate Left Unsecured",
        description="A controlled-access gate is open without an authorised person attending",
        visual_attributes=["gate_open_position", "no_personnel_in_gate_zone",
                           "access_control_mechanism_disengaged"],
        negative_attributes=["gate_in_controlled_opening_sequence", "attended_by_security_personnel"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
        temporal_window_sec=30,
    ),
    "perim_tailgating": ObservationSignal(
        signal_id="perim_tailgating",
        label="Tailgating / Piggybacking at Access Point",
        description="Unauthorised person follows authorised person through controlled access without independent authentication",
        visual_attributes=["two_or_more_persons_in_close_succession",
                           "single_authentication_event",
                           "second_person_enters_without_credential_check"],
        negative_attributes=["authorised_escort_procedure", "supervised_visitor_access"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.COUNTING,
                          AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "perim_surveillance_defeat": ObservationSignal(
        signal_id="perim_surveillance_defeat",
        label="Surveillance Equipment Tampering or Obstruction",
        description="Camera, sensor or lighting being deliberately tampered with, covered or repositioned",
        visual_attributes=["hand_near_camera_housing", "spray_paint_or_covering_material",
                           "sudden_image_occlusion", "camera_angle_displacement"],
        negative_attributes=["maintenance_technician_working", "accidental_occlusion_by_passing_object"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.VQA,
                          AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "perim_unauthorised_object": ObservationSignal(
        signal_id="perim_unauthorised_object",
        label="Unauthorised Object in Restricted Zone",
        description="Object left or placed in a sterile or access-controlled area without authorisation",
        visual_attributes=["foreign_object_present", "no_associated_authorised_person",
                           "object_inconsistent_with_zone_use"],
        negative_attributes=["authorised_equipment_cache", "maintenance_materials_in_use"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.GROUNDING],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "perim_laser_weapon": ObservationSignal(
        signal_id="perim_laser_weapon",
        label="Directed Energy / Laser Illumination",
        description="Visible laser beam directed at facility, equipment or personnel",
        visual_attributes=["coherent_beam_visible", "point_source_illumination",
                           "target_illumination_spot"],
        negative_attributes=["legitimate_survey_equipment_in_use", "laser_pointer_in_presentation"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING, AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
    ),

    # ── Illegal Dumping / Environmental signals ───────────────────────────────
    "dump_active_offload": ObservationSignal(
        signal_id="dump_active_offload",
        label="Active Illegal Dumping Behaviour",
        description="Person or vehicle actively depositing waste material in an unauthorised location",
        visual_attributes=["material_transfer_action", "vehicle_bed_or_boot_open",
                           "waste_material_airborne_or_falling", "non_designated_disposal_area"],
        negative_attributes=["authorised_waste_collection_vehicle", "designated_skip_area"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "dump_material_accumulation": ObservationSignal(
        signal_id="dump_material_accumulation",
        label="Illegal Waste Accumulation",
        description="Pile of waste, debris or hazardous material present in an unauthorised location",
        visual_attributes=["debris_pile_present", "non_containerised_waste",
                           "no_waste_collection_infrastructure_nearby",
                           "heterogeneous_material_composition"],
        negative_attributes=["authorised_construction_material_staging",
                              "contained_waste_receptacle"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.GROUNDING],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "dump_hazmat_indicator": ObservationSignal(
        signal_id="dump_hazmat_indicator",
        label="Potential Hazardous Material Dump",
        description="Discolouration, leakage, labelling or container type suggesting hazardous content",
        visual_attributes=["liquid_seepage_or_pooling", "chemical_staining",
                           "hazmat_container_shape_present", "unusual_colour_substance"],
        negative_attributes=["rain_runoff", "authorised_containment_system"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
    ),

    # ── Behavioural / Threat Indicator signals ────────────────────────────────
    "beh_concealment": ObservationSignal(
        signal_id="beh_concealment",
        label="Deliberate Identity Concealment",
        description="Person actively obscuring face or identity in a context where identification is expected",
        visual_attributes=["face_covering_in_non_cultural_context",
                           "hood_or_hat_pulled_low", "turned_away_from_camera_repeatedly"],
        negative_attributes=["religious_face_covering", "medical_mask_in_medical_context",
                              "cold_weather_appropriate_clothing"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "beh_suspicious_package": ObservationSignal(
        signal_id="beh_suspicious_package",
        label="Unattended or Suspicious Package",
        description="Bag, parcel or container left unattended in a public or restricted area",
        visual_attributes=["isolated_object_without_owner", "package_or_bag_shape",
                           "no_associated_person_within_proximity",
                           "object_inconsistent_with_surroundings"],
        negative_attributes=["person_momentarily_set_bag_down_while_present",
                              "authorised_mail_or_delivery_container"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING,
                          AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
        temporal_window_sec=60,
    ),
    "beh_surveillance_reconnaissance": ObservationSignal(
        signal_id="beh_surveillance_reconnaissance",
        label="Reconnaissance or Pre-Attack Surveillance Behaviour",
        description="Systematic observation, photographing or note-taking of facility infrastructure",
        visual_attributes=["camera_pointed_at_infrastructure", "repeated_passes_same_location",
                           "note_taking_near_access_points", "prolonged_stationary_observation"],
        negative_attributes=["tourist_photography_of_landmark", "press_accredited_journalist",
                              "authorised_survey_crew"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
    ),
    "beh_weapon_indicator": ObservationSignal(
        signal_id="beh_weapon_indicator",
        label="Visible or Suspected Weapon",
        description="Object consistent with a weapon visible or being carried",
        visual_attributes=["elongated_rigid_object_under_clothing",
                           "hand_grip_on_concealed_object",
                           "object_shape_consistent_with_firearm_or_edged_weapon"],
        negative_attributes=["tool_belt_worker", "uniformed_law_enforcement_holster"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "beh_person_down": ObservationSignal(
        signal_id="beh_person_down",
        label="Person in Distress or Incapacitated",
        description="Individual on ground or in abnormal posture suggesting injury, illness or incapacitation",
        visual_attributes=["prone_or_collapsed_posture", "no_self_directed_motion",
                           "bystander_attention_focused"],
        negative_attributes=["seated_person_on_ground_voluntarily", "yoga_or_exercise_pose"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "beh_fire_smoke": ObservationSignal(
        signal_id="beh_fire_smoke",
        label="Fire or Smoke Detected",
        description="Visible flames, smoke plume or thermal signature in or near facility",
        visual_attributes=["smoke_plume_visual", "flame_visible", "thermal_bloom",
                           "discolouration_or_haze"],
        negative_attributes=["cooking_smoke_in_designated_area",
                              "steam_from_hvac_exhaust"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.GROUNDING],
        frame_requirement=FrameRequirement.SINGLE,
    ),
    "beh_graffiti_vandalism": ObservationSignal(
        signal_id="beh_graffiti_vandalism",
        label="Vandalism or Graffiti in Progress",
        description="Person applying markings or causing damage to infrastructure surface",
        visual_attributes=["spray_can_or_marker_in_hand", "arm_motion_against_surface",
                           "fresh_marking_appearing_on_surface"],
        negative_attributes=["authorised_mural_artist", "maintenance_worker_marking"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Incident Type Definitions
# ─────────────────────────────────────────────────────────────────────────────

INCIDENT_TYPES: Dict[str, IncidentType] = {

    # ── Vehicle ──────────────────────────────────────────────────────────────
    "it_unattended_vehicle": IncidentType(
        type_id="it_unattended_vehicle",
        label="Unattended Vehicle",
        description="A vehicle present in a location where unattended vehicles pose an operational or security risk, with no driver or authorised attendant visible",
        severity=Severity.MODERATE,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["veh_stationary_prolonged"], SIGNALS["veh_unattended_restricted"]],
        related_types=["it_abandoned_vehicle", "it_suspicious_package"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.VQA,
                          AnnotationTask.TEMPORAL_CHANGE, AnnotationTask.ATTRIBUTE_RECOG],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["vehicle", "driver_seat", "zone_markings", "surrounding_personnel"],
        prompt_spatial_relations=["parked_at", "adjacent_to_restricted_area", "blocking_access_route"],
        prompt_temporal_cues=["duration_stationary", "time_since_occupant_last_visible"],
        counterfactual_cues=["driver_returning_to_vehicle", "loading_zone_with_operator_present",
                              "authorised_service_vehicle_with_markings"],
    ),
    "it_abandoned_vehicle": IncidentType(
        type_id="it_abandoned_vehicle",
        label="Abandoned Vehicle",
        description="A vehicle exhibiting physical or contextual evidence of permanent or long-term abandonment, representing an environmental, security or operational hazard",
        severity=Severity.MODERATE,
        temporal_sensitivity=TemporalSensitivity.LONG_TERM,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["veh_abandoned_degraded"], SIGNALS["veh_stationary_prolonged"],
                 SIGNALS["veh_no_plates"]],
        related_types=["it_unattended_vehicle", "it_illegal_dumping"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.ATTRIBUTE_RECOG, AnnotationTask.CAPTIONING],
        frame_requirement=FrameRequirement.SINGLE,
        prompt_focus_objects=["vehicle_body", "tyres", "windows", "registration_plate_area",
                               "surrounding_debris"],
        prompt_spatial_relations=["positioned_in_no_stopping_zone",
                                   "blocking_emergency_access_route"],
        prompt_temporal_cues=["debris_accumulation_pattern", "vegetation_growth_rate"],
        counterfactual_cues=["recently_parked_clean_vehicle", "vehicle_with_driver_visible"],
    ),
    "it_vehicle_wrong_zone": IncidentType(
        type_id="it_vehicle_wrong_zone",
        label="Vehicle in Prohibited Zone",
        description="Vehicle present in an area where its class, size or operational status is prohibited by regulation",
        severity=Severity.MODERATE,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["veh_oversized_wrong_zone"], SIGNALS["veh_unattended_restricted"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.GROUNDING],
        frame_requirement=FrameRequirement.SINGLE,
        prompt_focus_objects=["vehicle_type", "zone_signage", "road_markings"],
        prompt_spatial_relations=["within_exclusion_zone", "obstructing_designated_route"],
        counterfactual_cues=["authorised_vehicle_with_permit_displayed",
                              "emergency_vehicle_on_active_call"],
    ),

    # ── Crowd ────────────────────────────────────────────────────────────────
    "it_crowd_overcrowding": IncidentType(
        type_id="it_crowd_overcrowding",
        label="Overcrowding Event",
        description="Crowd density at a location exceeds safe occupancy thresholds",
        severity=Severity.HIGH,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.HIGH,
        signals=[SIGNALS["crowd_density_threshold"], SIGNALS["crowd_rapid_aggregation"]],
        related_types=["it_crowd_stampede", "it_crowd_altercation"],
        annotation_tasks=[AnnotationTask.COUNTING, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
        prompt_focus_objects=["persons_per_unit_area", "egress_routes", "barrier_integrity"],
        prompt_spatial_relations=["confined_within_barrier", "blocking_emergency_exit"],
        prompt_temporal_cues=["rate_of_crowd_growth", "queue_compression_velocity"],
        counterfactual_cues=["normal_pedestrian_flow", "orderly_queue_with_spacing"],
    ),
    "it_crowd_stampede": IncidentType(
        type_id="it_crowd_stampede",
        label="Crowd Stampede or Emergency Dispersal",
        description="Rapid uncontrolled crowd movement indicating panic, emergency or forced dispersal",
        severity=Severity.CRITICAL,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.HIGH,
        signals=[SIGNALS["crowd_dispersal_stampede"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["motion_vectors", "fallen_persons", "crowd_direction"],
        prompt_temporal_cues=["velocity_change_between_frames", "directionality_reversal"],
        counterfactual_cues=["normal_event_end_dispersal", "organised_evacuation_drill"],
    ),
    "it_loitering": IncidentType(
        type_id="it_loitering",
        label="Loitering in Sensitive Area",
        description="Person(s) remaining in a sensitive or restricted area without apparent legitimate purpose beyond the normal dwell threshold",
        severity=Severity.LOW,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["crowd_loitering"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA, AnnotationTask.ATTRIBUTE_RECOG],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["person_posture", "gaze_direction", "zone_boundary_markers"],
        prompt_temporal_cues=["cumulative_dwell_time", "repeated_return_to_location"],
        counterfactual_cues=["person_waiting_for_known_transport",
                              "authorised_break_area_usage"],
    ),
    "it_crowd_altercation": IncidentType(
        type_id="it_crowd_altercation",
        label="Altercation or Fighting",
        description="Physical conflict between individuals in a public or supervised space",
        severity=Severity.HIGH,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.HIGH,
        signals=[SIGNALS["crowd_altercation"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["persons_in_contact", "raised_limbs", "ground_contact"],
        counterfactual_cues=["sporting_activity", "theatrical_performance"],
    ),
    "it_crowd_management_flow": IncidentType(
        type_id="it_crowd_management_flow",
        label="Crowd Flow Management Violation",
        description="Crowd movement pattern that violates directional control or capacity management protocols",
        severity=Severity.MODERATE,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.HIGH,
        signals=[SIGNALS["crowd_density_threshold"], SIGNALS["crowd_social_distancing"]],
        annotation_tasks=[AnnotationTask.COUNTING, AnnotationTask.VQA,
                          AnnotationTask.SCENE_GRAPH],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["directional_signage", "queue_barrier", "crowd_motion_vectors"],
        prompt_temporal_cues=["flow_rate_per_minute", "counter_flow_detection"],
    ),

    # ── Perimeter ────────────────────────────────────────────────────────────
    "it_perimeter_breach": IncidentType(
        type_id="it_perimeter_breach",
        label="Physical Perimeter Security Breach",
        description="Unauthorised crossing of a physical security boundary",
        severity=Severity.CRITICAL,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["perim_fence_breach"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING,
                          AnnotationTask.VQA, AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["barrier_material", "breach_location", "intruder_body_position"],
        prompt_spatial_relations=["crossing_from_public_to_restricted_side"],
        counterfactual_cues=["authorised_personnel_using_designated_gate",
                              "maintenance_crew_with_escort"],
    ),
    "it_gate_unsecured": IncidentType(
        type_id="it_gate_unsecured",
        label="Unsecured Access Point",
        description="An access control gate, door or barrier left open or unlatched without authorised attendant",
        severity=Severity.HIGH,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["perim_gate_left_open"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["gate_latch_state", "surrounding_zone_activity"],
        counterfactual_cues=["gate_opening_for_authorised_vehicle",
                              "security_officer_present_at_gate"],
    ),
    "it_tailgating": IncidentType(
        type_id="it_tailgating",
        label="Tailgating at Controlled Access Point",
        description="One or more unauthorised persons gaining facility access by following an authenticated user through a controlled entry without separate authentication",
        severity=Severity.HIGH,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["perim_tailgating"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.COUNTING,
                          AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["authentication_device", "person_count_through_gate",
                               "inter_person_spacing"],
        counterfactual_cues=["supervised_visitor_escorted_by_badge_holder"],
    ),
    "it_camera_tampering": IncidentType(
        type_id="it_camera_tampering",
        label="Surveillance Infrastructure Tampering",
        description="Deliberate interference with surveillance cameras, lighting or intrusion sensors",
        severity=Severity.HIGH,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["perim_surveillance_defeat"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.VQA,
                          AnnotationTask.TEMPORAL_CHANGE],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["camera_housing", "hand_tool_near_device", "occlusion_material"],
        counterfactual_cues=["authorised_maintenance_technician_working_on_camera"],
    ),
    "it_laser_illumination": IncidentType(
        type_id="it_laser_illumination",
        label="Hostile Laser Illumination",
        description="Directed laser energy aimed at personnel, aircraft, vehicles or surveillance equipment",
        severity=Severity.CRITICAL,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["perim_laser_weapon"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING, AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["beam_origin", "illuminated_target", "operator_position"],
        counterfactual_cues=["surveying_equipment_in_authorised_use"],
    ),

    # ── Illegal Dumping ───────────────────────────────────────────────────────
    "it_illegal_dumping_active": IncidentType(
        type_id="it_illegal_dumping_active",
        label="Active Illegal Dumping",
        description="Real-time observation of waste being illegally deposited at an unauthorised location",
        severity=Severity.MODERATE,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["dump_active_offload"]],
        related_types=["it_illegal_dumping_evidence"],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["vehicle_or_person_depositing", "waste_material_type",
                               "disposal_site_context"],
        prompt_temporal_cues=["offloading_action_start_to_end"],
        counterfactual_cues=["authorised_collection_crew_with_vehicle_markings"],
    ),
    "it_illegal_dumping_evidence": IncidentType(
        type_id="it_illegal_dumping_evidence",
        label="Illegal Dumping — Evidence of Prior Event",
        description="Post-event identification of illegally deposited waste requiring investigation",
        severity=Severity.LOW,
        temporal_sensitivity=TemporalSensitivity.LONG_TERM,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["dump_material_accumulation"], SIGNALS["dump_hazmat_indicator"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.CAPTIONING, AnnotationTask.ATTRIBUTE_RECOG],
        frame_requirement=FrameRequirement.SINGLE,
        prompt_focus_objects=["waste_pile_composition", "containment_status", "site_markings"],
        counterfactual_cues=["authorised_material_staging_area", "contained_waste_bins"],
    ),

    # ── Behavioural / Pre-Incident Indicators ─────────────────────────────────
    "it_suspicious_behaviour": IncidentType(
        type_id="it_suspicious_behaviour",
        label="Suspicious Pre-Incident Behaviour",
        description="Behavioural pattern consistent with pre-attack reconnaissance, target selection or attack preparation",
        severity=Severity.HIGH,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["beh_surveillance_reconnaissance"], SIGNALS["beh_concealment"],
                 SIGNALS["beh_suspicious_package"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA, AnnotationTask.CAPTIONING],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["person_behaviour_pattern", "camera_or_note_taking_device",
                               "infrastructure_being_observed"],
        prompt_temporal_cues=["number_of_reconnaissance_passes", "dwell_time_at_observation_point"],
        counterfactual_cues=["tourist_photographing_landmark", "accredited_press_crew"],
    ),
    "it_unattended_package": IncidentType(
        type_id="it_unattended_package",
        label="Unattended or Suspicious Package",
        description="Bag, case or parcel left without an associated person in a public or restricted area",
        severity=Severity.HIGH,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["beh_suspicious_package"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.GROUNDING,
                          AnnotationTask.TEMPORAL_CHANGE, AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["package_shape_size_material", "surrounding_persons",
                               "placement_context"],
        prompt_temporal_cues=["time_since_owner_left", "nobody_approaching_object"],
        counterfactual_cues=["owner_momentarily_stepped_away_but_visible",
                              "delivery_package_in_marked_collection_area"],
    ),
    "it_person_down": IncidentType(
        type_id="it_person_down",
        label="Person Down or Medical Emergency",
        description="Individual incapacitated on the ground requiring immediate response",
        severity=Severity.CRITICAL,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["beh_person_down"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.SINGLE,
        prompt_focus_objects=["body_posture", "movement_or_lack_thereof", "bystander_response"],
        counterfactual_cues=["person_resting_or_sleeping_in_designated_area",
                              "person_performing_exercise"],
    ),
    "it_fire_smoke": IncidentType(
        type_id="it_fire_smoke",
        label="Fire or Smoke Incident",
        description="Active combustion or smoke event within or adjacent to a monitored facility",
        severity=Severity.CRITICAL,
        temporal_sensitivity=TemporalSensitivity.IMMEDIATE,
        visual_complexity=VisualComplexity.MEDIUM,
        signals=[SIGNALS["beh_fire_smoke"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.CLASSIFICATION,
                          AnnotationTask.GROUNDING],
        frame_requirement=FrameRequirement.SINGLE,
        prompt_focus_objects=["smoke_plume", "flame_location", "affected_structures"],
        counterfactual_cues=["steam_exhaust_from_hvac", "barbecue_smoke_in_designated_area"],
    ),
    "it_vandalism": IncidentType(
        type_id="it_vandalism",
        label="Vandalism or Property Damage",
        description="Deliberate damage to, or unauthorised marking of, infrastructure or property",
        severity=Severity.LOW,
        temporal_sensitivity=TemporalSensitivity.SHORT_TERM,
        visual_complexity=VisualComplexity.LOW,
        signals=[SIGNALS["beh_graffiti_vandalism"]],
        annotation_tasks=[AnnotationTask.DETECTION, AnnotationTask.TEMPORAL_CHANGE,
                          AnnotationTask.VQA],
        frame_requirement=FrameRequirement.MULTI,
        prompt_focus_objects=["tool_in_hand", "surface_being_marked", "markings_appearing"],
        counterfactual_cues=["authorised_muralist", "maintenance_worker_marking"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Incident Classes
# ─────────────────────────────────────────────────────────────────────────────

INCIDENT_CLASSES: Dict[str, IncidentClass] = {
    "ic_vehicle": IncidentClass(
        class_id="ic_vehicle",
        label="Vehicle Incidents",
        description="All incidents primarily involving motor vehicles — unattended, abandoned, misplaced",
        incident_types=[
            INCIDENT_TYPES["it_unattended_vehicle"],
            INCIDENT_TYPES["it_abandoned_vehicle"],
            INCIDENT_TYPES["it_vehicle_wrong_zone"],
        ],
        domain_tags=["transportation", "public_safety", "infrastructure"],
    ),
    "ic_crowd": IncidentClass(
        class_id="ic_crowd",
        label="Crowd and Pedestrian Incidents",
        description="All incidents involving groups of people — density, flow, conflict",
        incident_types=[
            INCIDENT_TYPES["it_crowd_overcrowding"],
            INCIDENT_TYPES["it_crowd_stampede"],
            INCIDENT_TYPES["it_loitering"],
            INCIDENT_TYPES["it_crowd_altercation"],
            INCIDENT_TYPES["it_crowd_management_flow"],
        ],
        domain_tags=["public_order", "event_management", "transit"],
    ),
    "ic_perimeter": IncidentClass(
        class_id="ic_perimeter",
        label="Physical Security Perimeter Incidents",
        description="All incidents involving the integrity of physical security boundaries",
        incident_types=[
            INCIDENT_TYPES["it_perimeter_breach"],
            INCIDENT_TYPES["it_gate_unsecured"],
            INCIDENT_TYPES["it_tailgating"],
            INCIDENT_TYPES["it_camera_tampering"],
            INCIDENT_TYPES["it_laser_illumination"],
        ],
        domain_tags=["access_control", "physical_security", "critical_infrastructure"],
    ),
    "ic_dumping": IncidentClass(
        class_id="ic_dumping",
        label="Illegal Dumping and Environmental Incidents",
        description="All incidents involving unauthorised disposal of waste or hazardous materials",
        incident_types=[
            INCIDENT_TYPES["it_illegal_dumping_active"],
            INCIDENT_TYPES["it_illegal_dumping_evidence"],
        ],
        domain_tags=["environmental", "code_enforcement", "public_health"],
    ),
    "ic_behavioural": IncidentClass(
        class_id="ic_behavioural",
        label="Behavioural and Pre-Incident Indicators",
        description="Behavioural signals that may precede a serious incident",
        incident_types=[
            INCIDENT_TYPES["it_suspicious_behaviour"],
            INCIDENT_TYPES["it_unattended_package"],
            INCIDENT_TYPES["it_person_down"],
            INCIDENT_TYPES["it_fire_smoke"],
            INCIDENT_TYPES["it_vandalism"],
        ],
        domain_tags=["threat_assessment", "public_safety", "emergency_response"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Top-level Domain
# ─────────────────────────────────────────────────────────────────────────────

SURVEILLANCE_DOMAIN = Domain(
    domain_id="dom_public_safety_surveillance",
    label="Public Safety & Security Surveillance",
    description="Comprehensive domain for real-time VLM-powered surveillance covering vehicles, crowds, perimeter integrity, environmental compliance and behavioural threat indicators",
    incident_classes=list(INCIDENT_CLASSES.values()),
)


def get_all_incident_types() -> List[IncidentType]:
    return list(INCIDENT_TYPES.values())


def get_incident_type(type_id: str) -> Optional[IncidentType]:
    return INCIDENT_TYPES.get(type_id)


def get_incidents_by_severity(min_severity: Severity) -> List[IncidentType]:
    return [it for it in INCIDENT_TYPES.values() if it.severity >= min_severity]


def get_incidents_by_task(task: AnnotationTask) -> List[IncidentType]:
    return [it for it in INCIDENT_TYPES.values() if task in it.annotation_tasks]


if __name__ == "__main__":
    print(f"Domain: {SURVEILLANCE_DOMAIN.label}")
    print(f"Incident Classes: {len(INCIDENT_CLASSES)}")
    print(f"Incident Types:   {len(INCIDENT_TYPES)}")
    print(f"Observation Signals: {len(SIGNALS)}")
    for cls in INCIDENT_CLASSES.values():
        print(f"\n  [{cls.class_id}] {cls.label}")
        for it in cls.incident_types:
            print(f"    ├─ {it.type_id}: {it.label} | sev={it.severity.name} | tasks={[t.value for t in it.annotation_tasks[:3]]}")
