"""Tests for the deterministic grader."""

from pathlib import Path

import pytest

from clinical_trial_env.grader import Grader
from clinical_trial_env.models import ClinicalTrialAction

DATA_DIR = Path(__file__).resolve().parent.parent / "clinical_trial_env" / "data"


@pytest.fixture()
def grader() -> Grader:
    return Grader(DATA_DIR / "ground_truth.json")


# ----- Correct actions -----

def test_correct_enrollment(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="CARDIO_001")
    reward = grader.grade("task1", "P1_001", action, "some ehr text")
    assert reward.value == 1.0


def test_correct_rejection(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="reject", reason="Has Active Asthma")
    reward = grader.grade("task1", "P1_002", action, "some ehr text")
    assert reward.value == 1.0


# ----- Incorrect actions -----

def test_enroll_when_should_reject(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="CARDIO_001")
    reward = grader.grade("task1", "P1_002", action, "some ehr text")
    assert reward.value == -1.0


def test_reject_when_should_enroll(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="reject", reason="Not eligible")
    reward = grader.grade("task1", "P1_001", action, "some ehr text")
    assert reward.value == -1.0


def test_enroll_wrong_trial(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="HTN_202")
    reward = grader.grade("task2", "P2_001", action, "some ehr text")
    assert reward.value == -1.0
    assert "DIAB_201" in reward.reason


# ----- Lab requests -----

def test_unnecessary_lab_request(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="request_lab", test_name="HbA1c")
    ehr = "Recent lab: HbA1c 8.2%, fasting glucose 186."
    reward = grader.grade("task2", "P2_001", action, ehr)
    assert reward.value == pytest.approx(-0.1)


def test_necessary_lab_request(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="request_lab", test_name="eGFR")
    ehr = "Patient with CKD on Losartan."  # no eGFR mentioned
    reward = grader.grade("task3", "P3_001", action, ehr)
    assert reward.value == 0.0


# ----- Determinism -----

def test_grader_is_deterministic(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="CARDIO_001")
    results = [
        grader.grade("task1", "P1_001", action, "ehr text").value for _ in range(10)
    ]
    assert all(r == 1.0 for r in results)


# ----- Task 2 inference checks -----

def test_task2_diabetes_enrollment(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="DIAB_201")
    reward = grader.grade("task2", "P2_005", action, "ehr")
    assert reward.value == 1.0


def test_task2_htn_enrollment(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="HTN_202")
    reward = grader.grade("task2", "P2_003", action, "ehr")
    assert reward.value == 1.0


def test_task2_onco_enrollment(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="ONCO_203")
    reward = grader.grade("task2", "P2_004", action, "ehr")
    assert reward.value == 1.0


# ----- Task 3 checks -----

def test_task3_correct_enrollment(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="enroll", trial_id="CKD_301")
    reward = grader.grade("task3", "P3_001", action, "ehr")
    assert reward.value == 1.0


def test_task3_correct_rejection(grader: Grader) -> None:
    action = ClinicalTrialAction(action_type="reject", reason="Kidney transplant recipient")
    reward = grader.grade("task3", "P3_006", action, "ehr")
    assert reward.value == 1.0
