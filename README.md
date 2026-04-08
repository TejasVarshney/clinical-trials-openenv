# Clinical Trial Patient Matcher

OpenEnv environment for matching patients from unstructured EHR summaries to active clinical trials.

The task models a clinical research coordinator workflow: read the note, compare it against trial inclusion and exclusion criteria, and decide whether to enroll the patient, reject them, or request a missing lab before deciding.

## Why This Is Useful

This is a realistic evaluation task for agents that need structured reasoning over clinical text. It combines eligibility screening, proxy inference from medications and vitals, and incomplete-data handling where the model must ask for a missing lab before making a final decision.

## Task Progression

Task 1 is explicit matching. The EHR text mirrors the trial language closely, so direct keyword and rule matching is enough.

Task 2 requires inference from proxies such as medications, blood pressure readings, imaging, and prior treatments.

Task 3 introduces incomplete data. Some required values are hidden and only become visible after a correct RequestLabResult action.

## Action Space

The environment accepts three actions:

- enroll: enroll the current patient in one of the available trials
- reject: reject the patient with a free-text reason
- request_lab: request a missing lab such as HbA1c, eGFR, ALT, Platelet count, CrCl, FEV1, or PHQ-9

## Reward

Correct enrollment or correct rejection yields +1.0.

Incorrect enrollment or incorrect rejection yields -1.0.

Requesting a lab that is already present in the EHR yields -0.1.

The final episode score is normalized into the [0.0, 1.0] range using cumulative reward.

## Repository Layout

- [openenv.yaml](openenv.yaml) - canonical OpenEnv manifest
- [Dockerfile](Dockerfile) - root deployment image
- [inference.py](inference.py) - baseline LLM runner
- [clinical_trial_env/](clinical_trial_env/) - package implementation

## Local Run

Install the package in editable mode, then start the API server from the repository root:

```bash
python -m pip install -e ./clinical_trial_env
uvicorn clinical_trial_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

Run the baseline against the server by setting the LLM endpoint variables and then executing:

```bash
python inference.py
```

## Docker

Build and run the submission container from the repository root:

```bash
docker build -t clinical_trial_env .
docker run -p 8000:8000 clinical_trial_env
```

## Notes For Reviewers

The canonical OpenEnv entrypoint lives at the repository root so validation and deployment can start from the top-level workspace.

The implementation package under [clinical_trial_env/](clinical_trial_env/) contains the environment, grader, models, and client wrapper.
