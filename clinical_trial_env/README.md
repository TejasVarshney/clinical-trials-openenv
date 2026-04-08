# Clinical Trial Env Package

This package contains the implementation for the Clinical Trial Patient Matcher OpenEnv environment.

See the repository root [README.md](../README.md) for the full submission overview, deployment path, and task summary.

## Modules

- [models.py](models.py): typed Pydantic models for observations, actions, rewards, and state.
- [grader.py](grader.py): deterministic reward logic backed by hidden ground truth.
- [client.py](client.py): EnvClient wrapper used by the baseline script and external agents.
- [server/clinical_trial_env_environment.py](server/clinical_trial_env_environment.py): environment state machine.
- [server/app.py](server/app.py): FastAPI/OpenEnv server entrypoint.

## Data

- [data/task1_data.json](data/task1_data.json)
- [data/task2_data.json](data/task2_data.json)
- [data/task3_data.json](data/task3_data.json)
- [data/ground_truth.json](data/ground_truth.json)

The task data is intentionally tiered:

1. Explicit matching.
2. Proxy-based inference.
3. Missing-data handling with hidden labs.
