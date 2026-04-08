"""Clinical Trial Patient Matcher — EnvClient wrapper."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    ClinicalTrialAction,
    ClinicalTrialObservation,
    ClinicalTrialState,
    TrialInfo,
)


class ClinicalTrialEnv(
    EnvClient[ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState]
):
    """
    WebSocket client for the Clinical Trial Patient Matcher environment.

    Example:
        >>> with ClinicalTrialEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="task1")
        ...     print(result.observation.active_patient_ehr)
        ...
        ...     result = client.step(
        ...         ClinicalTrialAction(action_type="enroll", trial_id="CARDIO_001")
        ...     )
    """

    def _step_payload(self, action: ClinicalTrialAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ClinicalTrialObservation]:
        obs_data = payload.get("observation", {})

        trials = [TrialInfo(**t) for t in obs_data.get("available_trials", [])]

        observation = ClinicalTrialObservation(
            active_patient_ehr=obs_data.get("active_patient_ehr", ""),
            available_trials=trials,
            patients_remaining=obs_data.get("patients_remaining", 0),
            reward_reason=obs_data.get("reward_reason", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ClinicalTrialState:
        return ClinicalTrialState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "task1"),
            current_patient_index=payload.get("current_patient_index", 0),
            total_patients=payload.get("total_patients", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            patients_remaining=payload.get("patients_remaining", 0),
        )
