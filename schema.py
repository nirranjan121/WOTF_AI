from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class Severity(str, Enum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class ChecklistItem(BaseModel):
    id: str = Field(..., description="Unique identifier for the checklist item")
    title: Optional[str] = Field(None, description="Short elegant title of the checklist item (e.g. 'Form Loop')")
    description: str = Field(..., description="Human-readable condition to verify (e.g. 'Rope end passed through loop')")
    verified: bool = Field(default=False, description="Whether this item has been verified as correct")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence required to auto-pass")

class FailureCondition(BaseModel):
    id: str = Field(..., description="Unique identifier for the failure condition")
    trigger_description: str = Field(..., description="Detailed description of what triggers this failure (e.g. 'Rope is pulled too short')")
    mitigation_instruction: str = Field(..., description="TTS/Voice instruction given to the user to fix their mistake")
    severity: Severity = Field(default=Severity.CRITICAL, description="Severity of the failure")

class SkillStep(BaseModel):
    index: int = Field(..., description="1-indexed sequence number of the step")
    name: str = Field(..., description="Short name of the step")
    description: str = Field(..., description="Detailed explanation of what the user should do during this step")
    voice_guideline: Optional[str] = Field(None, description="Warm audio spoken tutoring instruction given at the beginning of the step")
    illustrative_image_prompt: Optional[str] = Field(None, description="Detailed prompt to render a step-by-step cartoon illustration of Nano Banana doing this action")
    checklist: List[ChecklistItem] = Field(default_factory=list, description="List of visual checkpoints that must be checked off")
    failure_conditions: List[FailureCondition] = Field(default_factory=list, description="List of critical errors to watch out for during this step")

class SkillProtocol(BaseModel):
    name: str = Field(..., description="Name of the physical skill (e.g., 'Tying a Bowline Knot')")
    description: str = Field(..., description="Overall description of the skill and its purpose")
    category: str = Field(..., description="Industry domain or trade (e.g., 'Maritime', 'Electrical', 'Crafts')")
    version: str = Field(default="1.0.0", description="Version of the synthesized protocol")
    steps: List[SkillStep] = Field(default_factory=list, description="Ordered list of sequential steps to complete the skill")
