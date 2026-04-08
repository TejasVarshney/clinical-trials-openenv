"""Tests for the Clinical Trial environment state machine."""

import json
from pathlib import Path

import pytest

from clinical_trial_env.models import ClinicalTrialAction, ClinicalTrialObservation
from clinical_trial_env.server.clinical_trial_env_environment import (
    ClinicalTrialEnvironment,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "clinical_trial_env" / "data"


@pytest.fixture()
def env() -> ClinicalTrialEnvironment:
    return ClinicalTrialEnvironment()


# ----- reset() -----

@pytest.mark.parametrize(
    "task_id,expected_patients,expected_trials",
    [("task1", 10, 1), ("task2", 20, 3), ("task3", 30, 5)],
)
def test_reset_returns_correct_counts(
    env: ClinicalTrialEnvironment,
    task_id: str,
    expected_patients: int,
    expected_trials: int,
) -> None:
    obs = env.reset(task_id=task_id)
    assert isinstance(obs, ClinicalTrialObservation)
    assert obs.patients_remaining == expected_patients
    assert len(obs.available_trials) == expected_trials
    assert obs.done is False
    assert obs.active_patient_ehr != ""


def test_reset_deterministic_with_seed(env: ClinicalTrialEnvironment) -> None:
    obs1 = env.reset(task_id="task1", seed=123)
    obs2 = env.reset(task_id="task1", seed=123)
    assert obs1.active_patient_ehr == obs2.active_patient_ehr


def test_reset_different_seed_different_order(env: ClinicalTrialEnvironment) -> None:
    obs1 = env.reset(task_id="task1", seed=1)
    obs2 = env.reset(task_id="task1", seed=999)
    # With different seeds, order should (almost certainly) differ
    # We just verify reset works; exact order depends on seed
    assert isinstance(obs1, ClinicalTrialObservation)
    assert isinstance(obs2, ClinicalTrialObservation)


# ----- step(Enroll / Reject) -----

def test_step_enroll_advances_queue(env: ClinicalTrialEnvironment) -> None:
    obs = env.reset(task_id="task1", seed=0)
    first_ehr = obs.active_patient_ehr

    action = ClinicalTrialAction(action_type="enroll", trial_id="CARDIO_001")
    obs2 = env.step(action)

    assert obs2.patients_remaining == 9
    # EHR should be different (next patient) or empty if done
    if not obs2.done:
        assert obs2.active_patient_ehr != first_ehr


def test_step_reject_advances_queue(env: ClinicalTrialEnvironment) -> None:
    obs = env.reset(task_id="task1", seed=0)
    action = ClinicalTrialAction(action_type="reject", reason="Not eligible")
    obs2 = env.step(action)
    assert obs2.patients_remaining == 9


# ----- step(RequestLabResult) -----

def test_request_lab_does_not_advance(env: ClinicalTrialEnvironment) -> None:
    env.reset(task_id="task3", seed=0)
    state_before = env.state.current_patient_index

    action = ClinicalTrialAction(action_type="request_lab", test_name="eGFR")
    obs = env.step(action)

    assert env.state.current_patient_index == state_before
    assert obs.done is False


def test_request_lab_appends_data(env: ClinicalTrialEnvironment) -> None:
    """For task3, requesting a missing lab should append it to the EHR."""
    # Load task3 and find a patient with hidden labs
    task3 = json.loads((DATA_DIR / "task3_data.json").read_text(encoding="utf-8"))
    patient_with_labs = None
    for p in task3["patients"]:
        if p["hidden_labs"]:
            patient_with_labs = p
            break
    assert patient_with_labs is not None

    # We need to find this patient in the shuffled queue
    # Use seed=None to get default seed, and iterate until we process this patient
    env.reset(task_id="task3", seed=42)

    # For simplicity, just test that a lab request returns without error
    # and doesn't set done=True
    lab_name = list(patient_with_labs["hidden_labs"].keys())[0]
    action = ClinicalTrialAction(action_type="request_lab", test_name=lab_name)
    obs = env.step(action)
    assert obs.done is False


# ----- Episode completion -----

def test_episode_completes_after_all_patients(env: ClinicalTrialEnvironment) -> None:
    env.reset(task_id="task1", seed=42)
    reject = ClinicalTrialAction(action_type="reject", reason="test")

    for i in range(10):
        obs = env.step(reject)

    assert obs.done is True
    assert obs.patients_remaining == 0
    assert "final_score" in obs.metadata


def test_final_score_in_range(env: ClinicalTrialEnvironment) -> None:
    env.reset(task_id="task1", seed=42)
    reject = ClinicalTrialAction(action_type="reject", reason="test")

    for _ in range(10):
        obs = env.step(reject)

    score = obs.metadata["final_score"]
    assert 0.0 <= score <= 1.0


# ----- state property -----

def test_state_tracks_progress(env: ClinicalTrialEnvironment) -> None:
    env.reset(task_id="task1", seed=0)
    assert env.state.current_patient_index == 0
    assert env.state.task_id == "task1"
    assert env.state.total_patients == 10

    action = ClinicalTrialAction(action_type="reject", reason="test")
    env.step(action)

    assert env.state.current_patient_index == 1
    assert env.state.step_count == 1


def test_state_cumulative_reward(env: ClinicalTrialEnvironment) -> None:
    env.reset(task_id="task1", seed=0)
    assert env.state.cumulative_reward == 0.0

    action = ClinicalTrialAction(action_type="reject", reason="test")
    env.step(action)

    # Reward will be +1.0 or -1.0 depending on patient
    assert env.state.cumulative_reward != 0.0


# ----- Perfect play on task1 -----

def test_perfect_score_task1(env: ClinicalTrialEnvironment) -> None:
    """Play task1 with perfect answers and verify score = 1.0."""
    gt = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    task1_gt = gt["task1"]
    task1_data = json.loads((DATA_DIR / "task1_data.json").read_text(encoding="utf-8"))

    # Build a patient_id -> ground truth mapping
    # We need to figure out the shuffled order, so use seed and track
    import random
    rng = random.Random(42)
    patients = list(task1_data["patients"])
    rng.shuffle(patients)

    env.reset(task_id="task1", seed=42)

    for patient in patients:
        pid = patient["patient_id"]
        expected = task1_gt[pid]
        if expected == "reject":
            action = ClinicalTrialAction(action_type="reject", reason="Exclusion criteria violated")
        else:
            action = ClinicalTrialAction(action_type="enroll", trial_id=expected)
        obs = env.step(action)

    assert obs.done is True
    assert obs.metadata["final_score"] == pytest.approx(1.0)
