"""Submission-surface checks for the OpenEnv package."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "clinical_trial_env" / "data"


def test_root_submission_files_exist() -> None:
    assert (ROOT / "README.md").exists()
    assert (ROOT / "openenv.yaml").exists()
    assert (ROOT / "Dockerfile").exists()


def test_root_openenv_manifest_targets_package_app() -> None:
    text = (ROOT / "openenv.yaml").read_text(encoding="utf-8")
    assert "spec_version: 1" in text
    assert "clinical_trial_env.server.app:app" in text
    assert text.count("- id: task") == 3

def test_task_ground_truth_alignment() -> None:
    ground_truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    for task_file in ("task1_data.json", "task2_data.json", "task3_data.json"):
        task_data = json.loads((DATA_DIR / task_file).read_text(encoding="utf-8"))
        task_id = task_data["task_id"]
        patient_ids = {patient["patient_id"] for patient in task_data["patients"]}

        assert task_id in ground_truth
        assert set(ground_truth[task_id]) == patient_ids


def test_task3_has_diverse_hidden_labs() -> None:
    task3 = json.loads((DATA_DIR / "task3_data.json").read_text(encoding="utf-8"))
    hidden_lab_names: set[str] = set()
    hidden_lab_patients = 0

    for patient in task3["patients"]:
        hidden_labs = patient["hidden_labs"]
        if hidden_labs:
            hidden_lab_patients += 1
            hidden_lab_names.update(hidden_labs)

    assert hidden_lab_patients >= 10
    assert {"eGFR", "HbA1c", "ALT", "Platelet count", "CrCl", "FEV1", "PHQ-9"}.issubset(
        hidden_lab_names
    )


def test_inference_step_logs_are_structured(monkeypatch, capsys) -> None:
    import inference

    class FakeObservation:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, object]:
            return dict(self._payload)

    class FakeResult:
        def __init__(self, observation: FakeObservation, reward: float | None, done: bool) -> None:
            self.observation = observation
            self.reward = reward
            self.done = done

    class FakeEnv:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def sync(self) -> "FakeEnv":
            return self

        def __enter__(self) -> "FakeEnv":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def reset(self, task_id: str):
            observation = FakeObservation(
                {
                    "active_patient_ehr": "Example EHR",
                    "available_trials": [
                        {
                            "trial_id": "CARDIO_001",
                            "title": "Example Trial",
                            "inclusion_criteria": "Example inclusion",
                            "exclusion_criteria": "Example exclusion",
                        }
                    ],
                    "patients_remaining": 1,
                }
            )
            return FakeResult(observation, None, False)

        def step(self, action):
            observation = FakeObservation(
                {
                    "active_patient_ehr": "",
                    "available_trials": [
                        {
                            "trial_id": "CARDIO_001",
                            "title": "Example Trial",
                            "inclusion_criteria": "Example inclusion",
                            "exclusion_criteria": "Example exclusion",
                        }
                    ],
                    "patients_remaining": 0,
                    "metadata": {"final_score": 1.0},
                }
            )
            return FakeResult(observation, 1.0, True)

    class FakeChoiceMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = FakeChoiceMessage(content)

    class FakeCompletion:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeCompletion('{"action_type": "reject", "reason": "test"}')

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    monkeypatch.setattr(inference, "ClinicalTrialEnv", FakeEnv)
    monkeypatch.setattr(inference, "MODEL_NAME", "demo-model")

    summary = inference.run_task(FakeClient(), "task1")

    captured = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert captured[0].startswith("[START] ")
    assert captured[1].startswith("[STEP] ")
    assert captured[-1].startswith("[END] ")

    step_payload = json.loads(captured[1].split(" ", 1)[1])
    assert set(step_payload) >= {"step", "action", "reward", "done", "error"}
    assert isinstance(step_payload["action"], str)
    assert json.loads(step_payload["action"])["action_type"] == "reject"
    assert summary["success"] is True
