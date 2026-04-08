"""
Data models for the Clinical Trial Patient Matcher Environment.

Defines the Pydantic models for Observation, Action, and Reward used by
the environment, grader, and inference script. All fields are documented
so that LLM agents can read the JSON schema and understand the interface.
"""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------

class TrialInfo(BaseModel):
    """A clinical trial available for patient enrollment."""

    trial_id: str = Field(..., description="Unique identifier for the trial (e.g. 'CARDIO_001')")
    title: str = Field(..., description="Human-readable trial title")
    inclusion_criteria: str = Field(
        ...,
        description="Comma-separated inclusion criteria the patient must satisfy",
    )
    exclusion_criteria: str = Field(
        ...,
        description="Comma-separated exclusion criteria that disqualify the patient",
    )


class Reward(BaseModel):
    """Reward signal returned after each environment step."""

    value: float = Field(..., ge=-1.0, le=1.0, description="Reward value in [-1.0, +1.0]")
    reason: str = Field(..., description="Human-readable explanation for this reward")


# ---------------------------------------------------------------------------
# Observation (extends OpenEnv Observation)
# ---------------------------------------------------------------------------

class ClinicalTrialObservation(Observation):
    """
    Observation returned by the Clinical Trial environment.

    Contains the current patient's EHR text, the list of available trials,
    and how many patients remain in the queue.
    """

    active_patient_ehr: str = Field(
        default="",
        description="Unstructured Electronic Health Record summary for the current patient",
    )
    available_trials: list[TrialInfo] = Field(
        default_factory=list,
        description="All trials available for enrollment in the current task",
    )
    patients_remaining: int = Field(
        default=0,
        description="Number of patients left in the queue (including current)",
    )
    reward_reason: str = Field(
        default="",
        description="Human-readable explanation for the last reward",
    )


# ---------------------------------------------------------------------------
# Action (extends OpenEnv Action — flat discriminated class)
# ---------------------------------------------------------------------------

class ClinicalTrialAction(Action):
    """
    Action taken by the agent for the current patient.

    Exactly ONE of the following action types must be chosen:

    * **enroll** — Enroll the patient in a specific trial.
      Requires `trial_id`.
    * **reject** — Reject the patient from all trials.
      Requires `reason`.
    * **request_lab** — Request a missing lab result before deciding.
      Requires `test_name`. Does NOT advance to the next patient.
    """

    action_type: Literal["enroll", "reject", "request_lab"] = Field(
        ...,
        description="The type of action: 'enroll', 'reject', or 'request_lab'",
    )
    trial_id: str | None = Field(
        default=None,
        description="Trial ID to enroll the patient in (required when action_type='enroll')",
    )
    reason: str | None = Field(
        default=None,
        description="Free-text rationale for rejection (required when action_type='reject')",
    )
    test_name: str | None = Field(
        default=None,
        description="Lab test to request, e.g. 'HbA1c', 'eGFR' (required when action_type='request_lab')",
    )

    @model_validator(mode="after")
    def _validate_fields_for_action_type(self) -> ClinicalTrialAction:
        match self.action_type:
            case "enroll":
                if not self.trial_id:
                    raise ValueError("trial_id is required when action_type='enroll'")
            case "reject":
                if not self.reason:
                    raise ValueError("reason is required when action_type='reject'")
            case "request_lab":
                if not self.test_name:
                    raise ValueError("test_name is required when action_type='request_lab'")
        return self


# ---------------------------------------------------------------------------
# State (extends OpenEnv State)
# ---------------------------------------------------------------------------

class ClinicalTrialState(State):
    """Extended environment state with clinical-trial-specific fields."""

    model_config = {"extra": "allow"}

    task_id: str = Field(default="task1", description="Current task identifier")
    current_patient_index: int = Field(default=0, description="Index of the current patient in the queue")
    total_patients: int = Field(default=0, description="Total patients in this episode")
    cumulative_reward: float = Field(default=0.0, description="Sum of all rewards so far")
    patients_remaining: int = Field(default=0, description="Patients still to be processed")
