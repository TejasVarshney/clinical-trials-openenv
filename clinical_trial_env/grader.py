"""
Deterministic grader for the Clinical Trial Patient Matcher.

The grader compares agent actions against a hidden ground-truth mapping
and returns a Reward. It never uses randomness or LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .models import ClinicalTrialAction, Reward
except ImportError:
    from models import ClinicalTrialAction, Reward


class Grader:
    """Loads ground truth once and grades individual agent actions."""

    def __init__(self, ground_truth_path: Path | str) -> None:
        self._truth: dict[str, dict[str, str]] = json.loads(
            Path(ground_truth_path).read_text(encoding="utf-8")
        )

    def grade(
        self,
        task_id: str,
        patient_id: str,
        action: ClinicalTrialAction,
        ehr_text: str,
    ) -> Reward:
        """Return a deterministic Reward for the given action.

        Args:
            task_id: Which task is active (task1 / task2 / task3).
            patient_id: The patient being acted upon.
            action: The agent's chosen action.
            ehr_text: The current EHR text visible to the agent (used to
                      detect redundant lab requests).

        Returns:
            Reward with value in [-1.0, +1.0] and a human-readable reason.
        """
        expected = self._truth[task_id][patient_id]

        match action.action_type:
            case "enroll":
                if action.trial_id == expected:
                    return Reward(
                        value=1.0,
                        reason=f"Correct enrollment in {action.trial_id}",
                    )
                if expected == "reject":
                    return Reward(
                        value=-1.0,
                        reason=(
                            f"Incorrectly enrolled in {action.trial_id}; "
                            "patient should have been rejected"
                        ),
                    )
                return Reward(
                    value=-1.0,
                    reason=(
                        f"Enrolled in {action.trial_id} but the correct "
                        f"trial is {expected}"
                    ),
                )

            case "reject":
                if expected == "reject":
                    return Reward(
                        value=1.0,
                        reason=f"Correct rejection: {action.reason}",
                    )
                return Reward(
                    value=-1.0,
                    reason=(
                        f"Incorrectly rejected; patient should enroll "
                        f"in {expected}"
                    ),
                )

            case "request_lab":
                test_name = action.test_name or ""
                if test_name.lower() in ehr_text.lower():
                    return Reward(
                        value=-0.1,
                        reason=(
                            f"Unnecessary lab request: '{test_name}' "
                            "data is already present in the EHR"
                        ),
                    )
                return Reward(
                    value=0.0,
                    reason=f"Lab requested: {test_name}",
                )

        # Should never reach here due to Pydantic validation
        return Reward(value=0.0, reason="Unknown action type")  # pragma: no cover
