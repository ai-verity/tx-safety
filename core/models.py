"""Shared data models for the TX Public Safety system."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class Severity(str, Enum):
    P1 = "P1"   # Critical / life-threatening
    P2 = "P2"   # High / serious
    P3 = "P3"   # Medium
    P4 = "P4"   # Low / informational


class IncidentType(str, Enum):
    SHOOTING      = "Shooting"
    VEHICLE       = "Vehicle Accident"
    FIRE          = "Fire"
    MEDICAL       = "Medical Emergency"
    PURSUIT       = "Pursuit"
    HAZMAT        = "Hazmat"
    BURGLARY      = "Burglary"
    ASSAULT       = "Assault"
    DISTURBANCE   = "Disturbance"
    SUSPICIOUS    = "Suspicious Activity"
    NATURAL       = "Natural Disaster"
    MISSING       = "Missing Person"
    TRAFFIC       = "Major Traffic"
    OTHER         = "Other"


class RawItem(BaseModel):
    """Raw unprocessed item from any ingestion agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    raw_text: str
    url: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class Incident(BaseModel):
    """Normalized, classified incident record."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    incident_type: IncidentType = IncidentType.OTHER
    severity: Severity = Severity.P4
    city: str = ""
    county: str = ""
    state: str = "TX"
    lat: Optional[float] = None
    lon: Optional[float] = None
    description: str = ""
    source: str = ""
    source_url: Optional[str] = None
    active: bool = True
    reported_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    embedding_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.model_dump()
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, Enum):
                d[k] = v.value
        return d


class AgentStatus(BaseModel):
    name: str
    status: str       # "running" | "idle" | "error" | "polling"
    last_run: Optional[datetime] = None
    items_processed: int = 0
    error: Optional[str] = None
