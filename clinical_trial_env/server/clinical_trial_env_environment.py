"""
Clinical Trial Patient Matcher — Core Environment.

Implements the OpenEnv Environment interface. An episode consists of
processing a queue of patients against a set of clinical trials.
The agent can enroll, reject, or request missing lab data for each patient.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..grader import Grader
    from ..models import (
        ClinicalTrialAction,
        ClinicalTrialObservation,
        ClinicalTrialState,
        TrialInfo,
    )
except ImportError:
    from grader import Grader  # type: ignore[assignment]
    from models import (  # type: ignore[assignment]
        ClinicalTrialAction,
        ClinicalTrialObservation,
        ClinicalTrialState,
        TrialInfo,
    )


class ClinicalTrialEnvironment(Environment):
    """
    RL environment where an agent matches patients to clinical trials.

    State machine
    -------------
    reset(task_id)       → loads task data, builds patient queue, returns first Observation
    step(Enroll|Reject)  → grades, advances queue, returns next Observation (or done=True)
    step(RequestLabResult) → grades, appends lab to EHR, same patient (no advance)
    state                → current progress snapshot
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        self._grader = Grader(data_dir / "ground_truth.json")

        # Pre-load all task data
        self._task_data: dict[str, dict[str, Any]] = {}
        for fname in ("task1_data.json", "task2_data.json", "task3_data.json"):
            raw = json.loads((data_dir / fname).read_text(encoding="utf-8"))
            self._task_data[raw["task_id"]] = raw

        # Episode state — populated by reset()
        self._state = ClinicalTrialState(episode_id=str(uuid4()), step_count=0)
        self._patient_queue: list[dict[str, Any]] = []
        self._current_ehr: str = ""
        self._trials: list[TrialInfo] = []
        self._cumulative_reward: float = 0.0
        self._reward_history: list[float] = []
        self._max_possible_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "task1",
        **_kwargs: Any,
    ) -> ClinicalTrialObservation:
        """Load a task and return the first patient observation."""
        task = self._task_data[task_id]

        self._trials = [TrialInfo(**t) for t in task["trials"]]
        patients: list[dict[str, Any]] = list(task["patients"])  # shallow copy

        rng = random.Random(seed if seed is not None else 42)
        rng.shuffle(patients)

        self._patient_queue = patients
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._max_possible_reward = float(len(patients))

        self._state = ClinicalTrialState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            current_patient_index=0,
            total_patients=len(patients),
            cumulative_reward=0.0,
            patients_remaining=len(patients),
        )

        self._current_ehr = patients[0]["ehr"]

        return ClinicalTrialObservation(
            active_patient_ehr=self._current_ehr,
            available_trials=self._trials,
            patients_remaining=len(patients),
            done=False,
            reward=None,
            reward_reason="",
        )

    def step(self, action: ClinicalTrialAction, **_kwargs: Any) -> ClinicalTrialObservation:  # type: ignore[override]
        """Execute one agent action and return the resulting observation."""
        self._state.step_count += 1

        # If the episode is already finished, return a terminal observation
        # instead of crashing on an out-of-range patient index.
        if self._state.current_patient_index >= len(self._patient_queue):
            final_score = (
                0.0
                if self._max_possible_reward <= 0.0
                else max(
                    0.0,
                    min(1.0, self._cumulative_reward / self._max_possible_reward),
                )
            )
            return ClinicalTrialObservation(
                active_patient_ehr="",
                available_trials=self._trials,
                patients_remaining=0,
                done=True,
                reward=0.0,
                reward_reason="Episode already completed",
                metadata={
                    "final_score": final_score,
                    "reward_history": self._reward_history,
                },
            )

        task_id = self._state.task_id
        current_patient = self._patient_queue[self._state.current_patient_index]
        patient_id: str = current_patient["patient_id"]

        # --- RequestLabResult: stay on same patient ---
        if action.action_type == "request_lab":
            reward = self._grader.grade(task_id, patient_id, action, self._current_ehr)
            self._cumulative_reward += reward.value
            self._reward_history.append(reward.value)
            self._state.cumulative_reward = self._cumulative_reward

            # Append hidden lab data if available and not yet in EHR
            hidden_labs: dict[str, str] = current_patient.get("hidden_labs", {})
            test_name = action.test_name or ""
            lab_result = hidden_labs.get(test_name)
            if lab_result and test_name.lower() not in self._current_ehr.lower():
                self._current_ehr += f"\n\n[Lab Result - {test_name}]: {lab_result}"

            return ClinicalTrialObservation(
                active_patient_ehr=self._current_ehr,
                available_trials=self._trials,
                patients_remaining=self._state.patients_remaining,
                done=False,
                reward=reward.value,
                reward_reason=reward.reason,
            )

        # --- Enroll or Reject: grade and advance queue ---
        reward = self._grader.grade(task_id, patient_id, action, self._current_ehr)
        self._cumulative_reward += reward.value
        self._reward_history.append(reward.value)
        self._state.cumulative_reward = self._cumulative_reward

        self._state.current_patient_index += 1
        self._state.patients_remaining = (
            len(self._patient_queue) - self._state.current_patient_index
        )

        # Check if episode is done
        if self._state.current_patient_index >= len(self._patient_queue):
            final_score = max(
                0.0,
                min(1.0, self._cumulative_reward / self._max_possible_reward),
            )
            return ClinicalTrialObservation(
                active_patient_ehr="",
                available_trials=self._trials,
                patients_remaining=0,
                done=True,
                reward=reward.value,
                reward_reason=reward.reason,
                metadata={
                    "final_score": final_score,
                    "reward_history": self._reward_history,
                },
            )

        # Load next patient
        next_patient = self._patient_queue[self._state.current_patient_index]
        self._current_ehr = next_patient["ehr"]

        return ClinicalTrialObservation(
            active_patient_ehr=self._current_ehr,
            available_trials=self._trials,
            patients_remaining=self._state.patients_remaining,
            done=False,
            reward=reward.value,
            reward_reason=reward.reason,
        )

    @property
    def state(self) -> ClinicalTrialState:
        """Return the current environment state snapshot."""
        return self._state
